//! All CUDA calls live here: kernel launches, cuBLAS `GemmEx`, pinned
//! allocations, and peer copies. Nothing in this crate calls `unsafe`
//! anywhere else.
#![expect(unsafe_code, reason = "CUDA driver and cuBLAS calls are FFI-unsafe")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::result::CublasError;
use cudarc::cublas::safe::CudaBlas;
use cudarc::cublas::sys;
use cudarc::driver::result::DriverError;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::driver::sys as dsys;
use cudarc::nvrtc::compile_ptx;
use half::{bf16, f16};
use napkin_profile::{Dtype, Measured};

/// Anything the harness can fail with.
#[derive(Debug)]
pub(crate) struct BenchError(String);

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BenchError {}

impl From<DriverError> for BenchError {
    fn from(e: DriverError) -> Self {
        Self(e.to_string())
    }
}

impl From<CublasError> for BenchError {
    fn from(e: CublasError) -> Self {
        Self(e.to_string())
    }
}

impl From<cudarc::nvrtc::CompileError> for BenchError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        Self(e.to_string())
    }
}

const MIB: usize = 1 << 20;

/// Tunables for a full or `--quick` run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    rounds: u32,
    copy_mib: usize,
    elementwise_mib: usize,
    host_mib: usize,
    gemm_n: u32,
    chase_steps: u64,
    iterations_bw: u32,
    iterations_launch: u32,
    quick: bool,
}

impl Config {
    #[must_use]
    pub(crate) fn full() -> Self {
        Self {
            rounds: 7,
            copy_mib: 512,
            elementwise_mib: 256,
            host_mib: 512,
            gemm_n: 8192,
            chase_steps: 1_000_000,
            iterations_bw: 20,
            iterations_launch: 1_000,
            quick: false,
        }
    }

    #[must_use]
    pub(crate) fn quick() -> Self {
        Self {
            rounds: 3,
            copy_mib: 64,
            elementwise_mib: 64,
            host_mib: 64,
            gemm_n: 4096,
            chase_steps: 100_000,
            iterations_bw: 5,
            iterations_launch: 100,
            quick: true,
        }
    }
}

/// Median of event-timed rounds; each round records `iterations` calls of `op`
/// and stores per-iteration milliseconds.
fn cuda_samples(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    warmups: u32,
    iterations: u32,
    rounds: u32,
    mut op: impl FnMut() -> Result<(), BenchError>,
) -> Result<Vec<f64>, BenchError> {
    for _ in 0..warmups {
        op()?;
    }
    stream.synchronize()?;
    let mut samples = Vec::with_capacity(rounds as usize);
    for _ in 0..rounds {
        let start = ctx.new_event(None)?;
        let end = ctx.new_event(None)?;
        start.record(stream)?;
        for _ in 0..iterations {
            op()?;
        }
        end.record(stream)?;
        end.synchronize()?;
        samples.push(f64::from(start.elapsed_ms(&end)?) / f64::from(iterations));
    }
    Ok(samples)
}

/// Wall-clock median for latency-style metrics.
fn wall_samples(
    warmups: u32,
    iterations: u32,
    rounds: u32,
    mut op: impl FnMut() -> Result<(), BenchError>,
) -> Result<Vec<f64>, BenchError> {
    for _ in 0..warmups {
        op()?;
    }
    let mut samples = Vec::with_capacity(rounds as usize);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iterations {
            op()?;
        }
        samples.push(start.elapsed().as_secs_f64() * 1e3 / f64::from(iterations));
    }
    Ok(samples)
}

fn progress(name: &str, value: &str) {
    eprintln!("{name} {value}");
}

struct Kernels {
    noop: CudaFunction,
    addf: CudaFunction,
    chase: CudaFunction,
}

fn load_kernels(ctx: &Arc<CudaContext>) -> Result<Kernels, BenchError> {
    let ptx = compile_ptx(include_str!("kernels.cu"))?;
    let module = ctx.load_module(ptx)?;
    Ok(Kernels {
        noop: module.load_function("noop")?,
        addf: module.load_function("addf")?,
        chase: module.load_function("chase")?,
    })
}

fn launch_noop(stream: &Arc<CudaStream>, f: &CudaFunction) -> Result<(), BenchError> {
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream.launch_builder(f).launch(cfg)?;
    }
    Ok(())
}

fn sattolo_cycle(n: usize, seed: u64) -> Vec<u32> {
    let n32 = u32::try_from(n).unwrap_or(u32::MAX);
    let mut perm: Vec<u32> = (0..n32).collect();
    let mut rng = crate::stats::XorShift::new(seed);
    for i in (1..n).rev() {
        let j = usize::try_from(rng.next_u64() % i as u64).unwrap_or(0);
        perm.swap(i, j);
    }
    perm
}

fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = crate::stats::XorShift::new(seed);
    (0..n).map(|_| rng.next_f32()).collect()
}

struct GemmSpec {
    dtype: Dtype,
    atype: sys::cudaDataType_t,
    ctype: sys::cudaDataType_t,
    compute: sys::cublasComputeType_t,
    rtol: f64,
    atol: f64,
}

fn gemm_specs() -> Vec<GemmSpec> {
    vec![
        GemmSpec {
            dtype: Dtype::Fp16,
            atype: sys::cudaDataType_t::CUDA_R_16F,
            ctype: sys::cudaDataType_t::CUDA_R_16F,
            compute: sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            rtol: 0.03,
            atol: 0.25,
        },
        GemmSpec {
            dtype: Dtype::Bf16,
            atype: sys::cudaDataType_t::CUDA_R_16BF,
            ctype: sys::cudaDataType_t::CUDA_R_16BF,
            compute: sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            rtol: 0.08,
            atol: 1.0,
        },
        GemmSpec {
            dtype: Dtype::Tf32,
            atype: sys::cudaDataType_t::CUDA_R_32F,
            ctype: sys::cudaDataType_t::CUDA_R_32F,
            compute: sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            rtol: 0.01,
            atol: 0.08,
        },
        GemmSpec {
            dtype: Dtype::Fp32,
            atype: sys::cudaDataType_t::CUDA_R_32F,
            ctype: sys::cudaDataType_t::CUDA_R_32F,
            compute: sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
            rtol: 1e-4,
            atol: 1e-3,
        },
    ]
}

fn upload_as_dtype(
    stream: &Arc<CudaStream>,
    host: &[f32],
    spec: &GemmSpec,
) -> Result<CudaSlice<u8>, BenchError> {
    let bytes: Vec<u8> = match spec.atype {
        sys::cudaDataType_t::CUDA_R_16F => host
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        sys::cudaDataType_t::CUDA_R_16BF => host
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect(),
        _ => host.iter().flat_map(|&v| v.to_le_bytes()).collect(),
    };
    let mut dev = unsafe { stream.alloc::<u8>(bytes.len()) }?;
    stream.memcpy_htod(&bytes, &mut dev)?;
    Ok(dev)
}

fn read_as_f32(
    stream: &Arc<CudaStream>,
    dev: &CudaSlice<u8>,
    spec: &GemmSpec,
) -> Result<Vec<f32>, BenchError> {
    let bytes = stream.clone_dtoh(dev)?;
    let out = match spec.ctype {
        sys::cudaDataType_t::CUDA_R_16F => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_le_bytes(*b).to_f32())
            .collect(),
        sys::cudaDataType_t::CUDA_R_16BF => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b).to_f32())
            .collect(),
        _ => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    };
    Ok(out)
}

fn gemm_ex(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u8>,
    n: u32,
    spec: &GemmSpec,
) -> Result<(), BenchError> {
    let (a_ptr, _ga) = a.device_ptr(stream);
    let (b_ptr, _gb) = b.device_ptr(stream);
    let (c_ptr, _gc) = c.device_ptr_mut(stream);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let n_i = n.cast_signed();
    unsafe {
        sys::cublasGemmEx(
            *blas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            n_i,
            n_i,
            n_i,
            (&raw const alpha).cast::<core::ffi::c_void>(),
            a_ptr as *const core::ffi::c_void,
            spec.atype,
            n_i,
            b_ptr as *const core::ffi::c_void,
            spec.atype,
            n_i,
            (&raw const beta).cast::<core::ffi::c_void>(),
            c_ptr as *mut core::ffi::c_void,
            spec.ctype,
            n_i,
            spec.compute,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .result()
        .map_err(|e| BenchError(format!("cublasGemmEx: {e}")))?;
    }
    Ok(())
}

fn gemm_check(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    spec: &GemmSpec,
) -> Result<bool, BenchError> {
    let n: usize = 256;
    let a_host = random_f32(n * n, 0x000A_11CE);
    let b_host = random_f32(n * n, 0x0B0B_5EED);
    let a_dev = upload_as_dtype(stream, &a_host, spec)?;
    let b_dev = upload_as_dtype(stream, &b_host, spec)?;
    let elem = match spec.ctype {
        sys::cudaDataType_t::CUDA_R_32F => 4,
        _ => 2,
    };
    let mut c_dev = unsafe { stream.alloc::<u8>(n * n * elem) }?;
    gemm_ex(
        blas,
        stream,
        &a_dev,
        &b_dev,
        &mut c_dev,
        u32::try_from(n).unwrap_or(0),
        spec,
    )?;
    stream.synchronize()?;
    let c_host = read_as_f32(stream, &c_dev, spec)?;
    let mut ok = true;
    for col in 0..n {
        for row in 0..n {
            let idx = col * n + row;
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += f64::from(a_host[k * n + row]) * f64::from(b_host[col * n + k]);
            }
            let actual = f64::from(c_host[idx]);
            let tol = spec.atol + spec.rtol * acc.abs();
            if (actual - acc).abs() > tol {
                ok = false;
            }
        }
    }
    Ok(ok)
}

fn peer_supported(a: &Arc<CudaContext>, b: &Arc<CudaContext>) -> Result<bool, BenchError> {
    let mut can: i32 = 0;
    unsafe {
        dsys::cuDeviceCanAccessPeer(&raw mut can, a.cu_device(), b.cu_device())
            .result()
            .map_err(|e| BenchError(format!("cuDeviceCanAccessPeer: {e}")))?;
    }
    Ok(can != 0)
}

fn enable_peer(from: &Arc<CudaContext>, to: &Arc<CudaContext>) -> Result<(), BenchError> {
    from.bind_to_thread()?;
    unsafe {
        dsys::cuCtxEnablePeerAccess(to.cu_ctx(), 0)
            .result()
            .map_err(|e| BenchError(format!("cuCtxEnablePeerAccess: {e}")))?;
    }
    Ok(())
}

fn driver_version_string() -> String {
    let smi = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.lines().next().unwrap_or_default().trim().to_owned())
        .filter(|s| !s.is_empty());
    if let Some(v) = smi {
        return v;
    }
    let mut v: i32 = 0;
    unsafe {
        if dsys::cuDriverGetVersion(&raw mut v).result().is_err() {
            return "unknown".to_owned();
        }
    }
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

/// Runs the whole suite and returns the `Measured` payload.
///
/// # Errors
/// Returns `BenchError` on any CUDA, NVRTC, or cuBLAS failure.
#[expect(clippy::too_many_lines, reason = "sequential benchmark orchestration")]
#[expect(
    clippy::cast_precision_loss,
    reason = "byte and element counts fit in 2^53"
)]
pub(crate) fn run(cfg: &Config, gpu_request: &str) -> Result<Measured, BenchError> {
    let device_count = CudaContext::device_count()?.cast_unsigned();
    let ctx0 = CudaContext::new(0)?;
    let stream0 = ctx0.default_stream();
    let kernels = load_kernels(&ctx0)?;

    let mut samples_ms: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut checks: BTreeMap<String, bool> = BTreeMap::new();
    let mut record = |name: &str, samples: &[f64]| {
        samples_ms.insert(name.to_owned(), samples.to_vec());
        crate::stats::median(&mut samples.to_vec())
    };

    let launch_samples = cuda_samples(
        &ctx0,
        &stream0,
        20,
        cfg.iterations_launch,
        cfg.rounds,
        || launch_noop(&stream0, &kernels.noop),
    )?;
    let launch_us = record("launch_us", &launch_samples) * 1e3;
    progress("launch", &format!("{launch_us:.2} μs"));

    let sync_samples = wall_samples(10, 50, cfg.rounds, || {
        launch_noop(&stream0, &kernels.noop)?;
        stream0.synchronize()?;
        Ok(())
    })?;
    let sync_us = record("sync_us", &sync_samples) * 1e3;
    progress("launch+sync", &format!("{sync_us:.2} μs"));

    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(8);
    let host_elems = cfg.host_mib * MIB;
    let mut host_src = unsafe { ctx0.alloc_pinned_with_flags::<u8>(host_elems, 0) }?;
    let mut host_dst = unsafe { ctx0.alloc_pinned_with_flags::<u8>(host_elems, 0) }?;
    {
        let src = unsafe { std::slice::from_raw_parts_mut(host_src.as_mut_ptr()?, host_elems) };
        for (i, b) in src.iter_mut().enumerate() {
            *b = u8::try_from(i % 251).unwrap_or(0);
        }
    }
    let host_samples = wall_samples(5, 10, cfg.rounds, || {
        let src = unsafe { std::slice::from_raw_parts(host_src.as_ptr()?, host_elems) };
        let dst = unsafe { std::slice::from_raw_parts_mut(host_dst.as_mut_ptr()?, host_elems) };
        std::thread::scope(|s| {
            let chunk = host_elems / threads;
            let mut rest_src = src;
            let mut rest_dst = dst;
            let mut handles = Vec::new();
            for _ in 0..threads - 1 {
                let (a_src, b_src) = rest_src.split_at(chunk);
                let (a_dst, b_dst) = rest_dst.split_at_mut(chunk);
                rest_src = b_src;
                rest_dst = b_dst;
                handles.push(s.spawn(move || {
                    a_dst.copy_from_slice(a_src);
                }));
            }
            rest_dst.copy_from_slice(rest_src);
            for h in handles {
                let _ = h.join();
            }
        });
        Ok(())
    })?;
    let host_ms = record("host_copy_gbps", &host_samples);
    let host_copy_gbps = (2 * host_elems) as f64 / (host_ms / 1e3) / 1e9;
    progress(
        "host_copy",
        &format!("{host_copy_gbps:.0} GB/s ({threads} threads)"),
    );
    {
        let src = unsafe { std::slice::from_raw_parts(host_src.as_ptr()?, 4) };
        let dst = unsafe { std::slice::from_raw_parts(host_dst.as_ptr()?, 4) };
        checks.insert("host_copy".to_owned(), src == dst);
    }

    let copy_elems = cfg.copy_mib * MIB / 4;
    let mut pinned_a = unsafe { ctx0.alloc_pinned_with_flags::<f32>(copy_elems, 0) }?;
    let mut pinned_b = unsafe { ctx0.alloc_pinned_with_flags::<f32>(copy_elems, 0) }?;
    {
        let a = unsafe { std::slice::from_raw_parts_mut(pinned_a.as_mut_ptr()?, copy_elems) };
        for (i, v) in a.iter_mut().enumerate() {
            *v = i as f32;
        }
    }
    let mut dev_a = stream0.alloc_zeros::<f32>(copy_elems)?;

    let h2d_samples = cuda_samples(&ctx0, &stream0, 5, cfg.iterations_bw, cfg.rounds, || {
        let pinned = unsafe { std::slice::from_raw_parts(pinned_a.as_ptr()?, copy_elems) };
        stream0.memcpy_htod(pinned, &mut dev_a)?;
        Ok(())
    })?;
    let h2d_ms = record("h2d_gbps", &h2d_samples);
    let copy_bytes = (copy_elems * 4) as f64;
    let h2d_gbps = copy_bytes / (h2d_ms / 1e3) / 1e9;
    progress("h2d", &format!("{h2d_gbps:.0} GB/s"));

    let d2h_samples = cuda_samples(&ctx0, &stream0, 5, cfg.iterations_bw, cfg.rounds, || {
        let pinned = unsafe { std::slice::from_raw_parts_mut(pinned_b.as_mut_ptr()?, copy_elems) };
        stream0.memcpy_dtoh(&dev_a, pinned)?;
        Ok(())
    })?;
    let d2h_ms = record("d2h_gbps", &d2h_samples);
    let d2h_gbps = copy_bytes / (d2h_ms / 1e3) / 1e9;
    progress("d2h", &format!("{d2h_gbps:.0} GB/s"));

    let h2d_lat = wall_samples(5, 20, cfg.rounds, || {
        let one = [1.0f32];
        let mut dst = unsafe { stream0.alloc::<f32>(1) }?;
        stream0.memcpy_htod(&one, &mut dst)?;
        stream0.synchronize()?;
        Ok(())
    })?;
    let h2d_latency_us = record("h2d_latency_us", &h2d_lat) * 1e3;
    let d2h_lat = wall_samples(5, 20, cfg.rounds, || {
        let mut one = [0.0f32];
        stream0.memcpy_dtoh(&dev_a.slice(0..1), &mut one)?;
        stream0.synchronize()?;
        Ok(())
    })?;
    let d2h_latency_us = record("d2h_latency_us", &d2h_lat) * 1e3;
    progress(
        "h2d/d2h latency",
        &format!("{h2d_latency_us:.1}/{d2h_latency_us:.1} μs"),
    );

    stream0.synchronize()?;
    let probes_ok = {
        let a = unsafe { std::slice::from_raw_parts(pinned_a.as_ptr()?, 3) };
        let mid = copy_elems / 2;
        let b0 = unsafe { std::slice::from_raw_parts(pinned_b.as_ptr()?, 1) }[0];
        let bmid = unsafe { std::slice::from_raw_parts(pinned_b.as_ptr()?.add(mid), 1) }[0];
        let blast =
            unsafe { std::slice::from_raw_parts(pinned_b.as_ptr()?.add(copy_elems - 1), 1) }[0];
        let am = unsafe { std::slice::from_raw_parts(pinned_a.as_ptr()?.add(mid), 1) }[0];
        let al =
            unsafe { std::slice::from_raw_parts(pinned_a.as_ptr()?.add(copy_elems - 1), 1) }[0];
        a[0].abs() < f32::EPSILON
            && (b0 - a[0]).abs() < f32::EPSILON
            && (bmid - am).abs() < f32::EPSILON
            && (blast - al).abs() < f32::EPSILON
    };
    checks.insert("h2d_d2h_round_trip".to_owned(), probes_ok);

    let mut dev_src = stream0.alloc_zeros::<f32>(copy_elems)?;
    stream0.memcpy_htod(&vec![0.0f32; copy_elems], &mut dev_src)?;
    {
        let pattern = random_f32(copy_elems, 0x00C0_FFEE);
        stream0.memcpy_htod(&pattern, &mut dev_src)?;
    }
    let mut dev_dst = stream0.alloc_zeros::<f32>(copy_elems)?;
    let hbm_samples = cuda_samples(&ctx0, &stream0, 5, cfg.iterations_bw, cfg.rounds, || {
        stream0.memcpy_dtod(&dev_src, &mut dev_dst)?;
        Ok(())
    })?;
    let hbm_ms = record("hbm_gbps", &hbm_samples);
    let hbm_gbps = (2.0 * copy_bytes) / (hbm_ms / 1e3) / 1e9;
    progress("hbm_copy", &format!("{hbm_gbps:.0} GB/s"));
    {
        stream0.synchronize()?;
        let back = stream0.clone_dtoh(&dev_dst)?;
        let expected = random_f32(copy_elems, 0x00C0_FFEE);
        let mid = copy_elems / 2;
        checks.insert(
            "hbm_copy".to_owned(),
            (back[0] - expected[0]).abs() < f32::EPSILON
                && (back[mid] - expected[mid]).abs() < f32::EPSILON
                && (*back.last().map_or(&-1.0, |v| v) - expected[copy_elems - 1]).abs()
                    < f32::EPSILON,
        );
    }

    let ew_elems = cfg.elementwise_mib * MIB / 4;
    let a = random_f32(ew_elems, 0xA);
    let b = random_f32(ew_elems, 0xB);
    let dev_lhs = stream0.clone_htod(&a)?;
    let dev_rhs = stream0.clone_htod(&b)?;
    let mut dev_sum = stream0.alloc_zeros::<f32>(ew_elems)?;
    let n64 = ew_elems as u64;
    let ew_cfg = LaunchConfig {
        grid_dim: (u32::try_from(ew_elems.div_ceil(256)).unwrap_or(0), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let ew_samples = cuda_samples(&ctx0, &stream0, 5, cfg.iterations_bw, cfg.rounds, || {
        unsafe {
            stream0
                .launch_builder(&kernels.addf)
                .arg(&dev_lhs)
                .arg(&dev_rhs)
                .arg(&mut dev_sum)
                .arg(&n64)
                .launch(ew_cfg)?;
        }
        Ok(())
    })?;
    let ew_ms = record("elementwise_gbps", &ew_samples);
    let elementwise_gbps = (3 * ew_elems * 4) as f64 / (ew_ms / 1e3) / 1e9;
    progress("elementwise_add", &format!("{elementwise_gbps:.0} GB/s"));
    {
        stream0.synchronize()?;
        let c = stream0.clone_dtoh(&dev_sum)?;
        let i = ew_elems / 2;
        checks.insert(
            "elementwise_add".to_owned(),
            (c[i] - (a[i] + b[i])).abs() < 1e-6,
        );
    }

    let chase_elems = (1usize << 30) / 4;
    let perm = sattolo_cycle(chase_elems, 0xDEAD_BEEF);
    let dev_perm = stream0.clone_htod(&perm)?;
    let mut dev_out = stream0.alloc_zeros::<u32>(1)?;
    let chase_samples = cuda_samples(&ctx0, &stream0, 3, 3, cfg.rounds, || {
        let start: u64 = 0;
        let steps = cfg.chase_steps;
        unsafe {
            stream0
                .launch_builder(&kernels.chase)
                .arg(&dev_perm)
                .arg(&start)
                .arg(&steps)
                .arg(&mut dev_out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    })?;
    let chase_ms = record("hbm_latency_ns", &chase_samples);
    let hbm_latency_ns = chase_ms * 1e6 / cfg.chase_steps as f64;
    progress("hbm_latency", &format!("{hbm_latency_ns:.0} ns/load"));
    {
        let steps = chase_elems as u64;
        unsafe {
            stream0
                .launch_builder(&kernels.chase)
                .arg(&dev_perm)
                .arg(&0u64)
                .arg(&steps)
                .arg(&mut dev_out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream0.synchronize()?;
        let out = stream0.clone_dtoh(&dev_out)?;
        checks.insert("pointer_chase_cycle".to_owned(), out[0] == 0);
    }

    let mut g2g_gbps = None;
    let mut g2g_forward = None;
    let mut g2g_reverse = None;
    let mut g2g_latency = None;
    if device_count >= 2 {
        let ctx1 = CudaContext::new(1)?;
        let stream1 = ctx1.default_stream();
        if peer_supported(&ctx0, &ctx1)? && peer_supported(&ctx1, &ctx0)? {
            enable_peer(&ctx0, &ctx1)?;
            enable_peer(&ctx1, &ctx0)?;
            let peer_elems = cfg.copy_mib * MIB / 4;
            let pattern = random_f32(peer_elems, 0xFEED);
            let p_src0 = stream0.clone_htod(&pattern)?;
            let mut p_dst1 = unsafe { stream1.alloc::<f32>(peer_elems) }?;
            let p_src1 = stream1.clone_htod(&pattern)?;
            let mut p_dst0 = unsafe { stream0.alloc::<f32>(peer_elems) }?;
            let fwd = cuda_samples(&ctx1, &stream1, 5, cfg.iterations_bw, cfg.rounds, || {
                stream1.memcpy_dtod(&p_src0, &mut p_dst1)?;
                Ok(())
            })?;
            let rev = cuda_samples(&ctx0, &stream0, 5, cfg.iterations_bw, cfg.rounds, || {
                stream0.memcpy_dtod(&p_src1, &mut p_dst0)?;
                Ok(())
            })?;
            let fwd_gbps = copy_bytes / (record("g2g_forward_gbps", &fwd) / 1e3) / 1e9;
            let rev_gbps = copy_bytes / (record("g2g_reverse_gbps", &rev) / 1e3) / 1e9;
            g2g_forward = Some(fwd_gbps);
            g2g_reverse = Some(rev_gbps);
            g2g_gbps = Some(fwd_gbps.min(rev_gbps));
            progress("g2g", &format!("{fwd_gbps:.0}/{rev_gbps:.0} GB/s"));
            let lat = wall_samples(5, 20, cfg.rounds, || {
                let mut one = unsafe { stream1.alloc::<f32>(1) }?;
                stream1.memcpy_dtod(&p_src0.slice(0..1), &mut one)?;
                stream1.synchronize()?;
                Ok(())
            })?;
            g2g_latency = Some(record("g2g_latency_us", &lat) * 1e3);
            let check_fwd = {
                stream1.synchronize()?;
                let out = stream1.clone_dtoh(&p_dst1)?;
                (out[0] - pattern[0]).abs() < f32::EPSILON
                    && (out[peer_elems / 2] - pattern[peer_elems / 2]).abs() < f32::EPSILON
            };
            let check_rev = {
                stream0.synchronize()?;
                let out = stream0.clone_dtoh(&p_dst0)?;
                (out[0] - pattern[0]).abs() < f32::EPSILON
                    && (out[peer_elems / 2] - pattern[peer_elems / 2]).abs() < f32::EPSILON
            };
            checks.insert("g2g_0_to_1".to_owned(), check_fwd);
            checks.insert("g2g_1_to_0".to_owned(), check_rev);
        } else {
            eprintln!("peer access unavailable; skipping g2g");
        }
    } else {
        eprintln!("single GPU; skipping g2g");
    }

    let blas = CudaBlas::new(stream0.clone())?;
    let n = cfg.gemm_n as usize;
    let a_host = random_f32(n * n, 0x000F_ACEA);
    let b_host = random_f32(n * n, 0x000F_ACEB);
    let mut compute_tflops = BTreeMap::new();
    for spec in gemm_specs() {
        let a_dev = upload_as_dtype(&stream0, &a_host, &spec)?;
        let b_dev = upload_as_dtype(&stream0, &b_host, &spec)?;
        let elem = match spec.ctype {
            sys::cudaDataType_t::CUDA_R_32F => 4,
            _ => 2,
        };
        let mut c_dev = unsafe { stream0.alloc::<u8>(n * n * elem) }?;
        let gemm_samples = cuda_samples(
            &ctx0,
            &stream0,
            if cfg.quick { 3 } else { 10 },
            if cfg.quick { 3 } else { 10 },
            cfg.rounds,
            || {
                gemm_ex(
                    &blas,
                    &stream0,
                    &a_dev,
                    &b_dev,
                    &mut c_dev,
                    u32::try_from(n).unwrap_or(0),
                    &spec,
                )
            },
        )?;
        let key = format!("gemm_{}", spec.dtype);
        let ms = record(&key, &gemm_samples);
        let flops = 2.0 * (n as f64).powi(3);
        let tflops = flops / (ms / 1e3) / 1e12;
        compute_tflops.insert(spec.dtype, tflops);
        progress(
            &format!("gemm_{}", spec.dtype),
            &format!("{tflops:.0} TFLOP/s"),
        );
        checks.insert(
            format!("gemm_{}_check", spec.dtype),
            gemm_check(&blas, &stream0, &spec)?,
        );
    }

    let (cc_major, cc_minor) = ctx0.compute_capability()?;
    let sm_count =
        ctx0.attribute(dsys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let all_ok = checks.values().all(|v| *v);

    Ok(Measured {
        captured_at: crate::iso_now(),
        gpu_request: if cfg.quick {
            format!("{gpu_request} --quick")
        } else {
            gpu_request.to_owned()
        },
        device_name: ctx0.name()?,
        device_count,
        compute_capability: format!("{cc_major}.{cc_minor}"),
        sm_count: sm_count.cast_unsigned(),
        memory_bytes: ctx0.total_mem()? as u64,
        driver_version: driver_version_string(),
        cuda_version: "12.9".to_owned(),
        rounds: cfg.rounds,
        launch_us,
        sync_us: Some(sync_us),
        host_copy_gbps,
        h2d_gbps,
        d2h_gbps,
        h2d_latency_us: Some(h2d_latency_us),
        d2h_latency_us: Some(d2h_latency_us),
        hbm_gbps,
        elementwise_gbps,
        hbm_latency_ns: Some(hbm_latency_ns),
        g2g_gbps,
        g2g_forward_gbps: g2g_forward,
        g2g_reverse_gbps: g2g_reverse,
        g2g_latency_us: g2g_latency,
        compute_tflops,
        gemm_shape: {
            let n32 = u32::try_from(n).unwrap_or(0);
            [n32, n32, n32]
        },
        checks,
        samples_ms,
    })
    .and_then(|m| {
        if all_ok {
            Ok(m)
        } else {
            eprintln!(
                "FAILED CHECKS: {:?}",
                m.checks
                    .iter()
                    .filter(|(_, v)| !**v)
                    .map(|(k, _)| k)
                    .collect::<Vec<_>>()
            );
            println!(
                "{}",
                serde_json::to_string_pretty(&m).map_err(|e| BenchError(e.to_string()))?
            );
            Err(BenchError(
                "one or more correctness checks failed".to_owned(),
            ))
        }
    })
}
