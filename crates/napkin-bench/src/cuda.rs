//! All CUDA calls live here: kernel launches, cuBLAS `GemmEx`, pinned
//! allocations, and peer copies. Nothing in this crate calls `unsafe`
//! anywhere else.
#![expect(unsafe_code, reason = "CUDA driver and cuBLAS calls are FFI-unsafe")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::cublas::result::CublasError;
use cudarc::cublas::safe::CudaBlas;
use cudarc::cublas::sys;
use cudarc::cublaslt::sys::cublasLtMatmulAlgoConfigAttributes_t as LtAlgoAttr;
use cudarc::cublaslt::{result as lt_result, sys as ltsys};
use cudarc::driver::result::DriverError;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::driver::sys as dsys;
use cudarc::nvrtc::compile_ptx;
use half::{bf16, f16};
use napkin_profile::{Dtype, GemmAlgo, GemmCheck, GemmShape, GemmShapeSuite};
use napkin_profile::{Measured, RoundState, tflops};

use crate::gpu_state::{Monitor, Window, round_states};

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
    gemm_warmup: Duration,
    gemm_round: Duration,
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
            gemm_warmup: Duration::from_secs(1),
            gemm_round: Duration::from_millis(200),
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
            gemm_warmup: Duration::from_millis(100),
            gemm_round: Duration::from_millis(20),
            chase_steps: 100_000,
            iterations_bw: 5,
            iterations_launch: 100,
            quick: true,
        }
    }
}

/// Event-timed rounds; each round records `iterations` calls of `op` and
/// stores per-iteration milliseconds.
fn cuda_samples(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    warmups: u32,
    iterations: u32,
    rounds: u32,
    op: impl FnMut() -> Result<(), BenchError>,
) -> Result<Vec<f64>, BenchError> {
    Ok(cuda_rounds(ctx, stream, warmups, iterations, rounds, op)?
        .into_iter()
        .map(|(ms, _)| ms)
        .collect())
}

/// `cuda_samples` plus the host-clock window of each round, so NVML polls can
/// be matched to it. The stream is idle when a round starts, so its first
/// call begins on the GPU right after `start` is taken, and `end` is taken
/// once the last call has finished.
fn cuda_rounds(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    warmups: u32,
    iterations: u32,
    rounds: u32,
    mut op: impl FnMut() -> Result<(), BenchError>,
) -> Result<Vec<(f64, Window)>, BenchError> {
    for _ in 0..warmups {
        op()?;
    }
    stream.synchronize()?;
    let mut samples = Vec::with_capacity(rounds as usize);
    for _ in 0..rounds {
        let start = ctx.new_event(Some(dsys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = ctx.new_event(Some(dsys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let window_start = Instant::now();
        start.record(stream)?;
        for _ in 0..iterations {
            op()?;
        }
        end.record(stream)?;
        stream.synchronize()?;
        let window = Window {
            start: window_start,
            end: Instant::now(),
        };
        let ms =
            unsafe { cudarc::driver::result::event::elapsed(start.cu_event(), end.cu_event())? };
        samples.push((f64::from(ms) / f64::from(iterations), window));
    }
    Ok(samples)
}

/// `cuda_rounds` for work that heats the GPU enough for its power controller
/// to lower clocks, like an 8192³ fp16 GEMM on a 700 W H100. Such a GPU runs
/// the first ~100 ms at its maximum SM clock and then settles lower once the
/// software power cap engages, so rounds timed right after a short warm-up
/// catch the boost and later rounds catch the cap. Running `op` for `warmup`
/// first and then sizing every round to last about `round` reports the
/// sustained rate and averages each round over the controller's swings.
fn sustained_rounds(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    warmup: Duration,
    round: Duration,
    rounds: u32,
    mut op: impl FnMut() -> Result<(), BenchError>,
) -> Result<Vec<(f64, Window)>, BenchError> {
    const BATCH: u32 = 10;
    // The first batch pays one-time setup (cuBLAS loads its kernels lazily,
    // which took over a second on B300), so it neither counts towards the
    // warm-up nor sizes the rounds; otherwise a 0.7 ms GEMM was sized as a
    // 100 ms one and every "200 ms" round lasted 7 ms.
    let mut warm_since = None;
    let per_call = loop {
        let batch_start = Instant::now();
        for _ in 0..BATCH {
            op()?;
        }
        stream.synchronize()?;
        let per_call = batch_start.elapsed() / BATCH;
        if let Some(since) = warm_since
            && Instant::elapsed(&since) >= warmup
        {
            break per_call;
        }
        warm_since.get_or_insert_with(Instant::now);
    };
    let iterations = round
        .as_nanos()
        .div_ceil(per_call.as_nanos().max(1))
        .max(u128::from(BATCH));
    cuda_rounds(
        ctx,
        stream,
        0,
        u32::try_from(iterations).unwrap_or(u32::MAX),
        rounds,
        op,
    )
}

/// NVML-style PCI bus ID of `ctx`'s device, e.g. `"00000000:18:00.0"`.
fn pci_bus_id(ctx: &Arc<CudaContext>) -> Result<String, BenchError> {
    let attr = |a| ctx.attribute(a).map(i32::cast_unsigned);
    Ok(format!(
        "{:08x}:{:02x}:{:02x}.0",
        attr(dsys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
        attr(dsys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
        attr(dsys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?,
    ))
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

const GEMM_SHAPE_WORKSPACE_BYTES: usize = 32 * MIB;

struct LtHandle(ltsys::cublasLtHandle_t);

impl LtHandle {
    fn new() -> Result<Self, BenchError> {
        lt_result::create_handle()
            .map(Self)
            .map_err(|e| BenchError(format!("cublasLtCreate: {e}")))
    }
}

impl Drop for LtHandle {
    fn drop(&mut self) {
        let _ = unsafe { ltsys::cublasLtDestroy(self.0).result() };
    }
}

struct GemmShapeCase {
    name: &'static str,
    m: u32,
    n: u32,
    k: u32,
}

fn gemm_shape_cases() -> [GemmShapeCase; 5] {
    [
        GemmShapeCase {
            name: "square",
            m: 8192,
            n: 8192,
            k: 8192,
        },
        GemmShapeCase {
            name: "prefill",
            m: 4096,
            n: 28672,
            k: 8192,
        },
        GemmShapeCase {
            name: "decode-m8",
            m: 8,
            n: 8192,
            k: 8192,
        },
        GemmShapeCase {
            name: "decode-m64",
            m: 64,
            n: 8192,
            k: 8192,
        },
        GemmShapeCase {
            name: "square-k8191",
            m: 8192,
            n: 8192,
            k: 8191,
        },
    ]
}

struct LtMatmulDesc(ltsys::cublasLtMatmulDesc_t);

impl LtMatmulDesc {
    fn new() -> Result<Self, BenchError> {
        let desc = lt_result::create_matmul_desc(
            ltsys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            ltsys::cudaDataType_t::CUDA_R_32F,
        )
        .map_err(|e| BenchError(format!("cublasLtMatmulDescCreate: {e}")))?;
        let desc = Self(desc);
        let trans_a = sys::cublasOperation_t::CUBLAS_OP_T;
        let trans_b = sys::cublasOperation_t::CUBLAS_OP_N;
        for (attr, value) in [
            (
                ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &trans_a,
            ),
            (
                ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &trans_b,
            ),
        ] {
            let result = unsafe {
                ltsys::cublasLtMatmulDescSetAttribute(
                    desc.0,
                    attr,
                    std::ptr::from_ref(value).cast(),
                    std::mem::size_of_val(value),
                )
            };
            lt_status(result, "cublasLtMatmulDescSetAttribute")?;
        }
        Ok(desc)
    }
}

impl Drop for LtMatmulDesc {
    fn drop(&mut self) {
        let _ = unsafe { ltsys::cublasLtMatmulDescDestroy(self.0).result() };
    }
}

struct LtMatrixLayout(ltsys::cublasLtMatrixLayout_t);

impl LtMatrixLayout {
    fn new(
        data_type: sys::cudaDataType_t,
        rows: u32,
        cols: u32,
        leading_dimension: u32,
    ) -> Result<Self, BenchError> {
        let layout = lt_result::create_matrix_layout(
            lt_data_type(data_type)?,
            u64::from(rows),
            u64::from(cols),
            i64::from(leading_dimension),
        )
        .map_err(|e| BenchError(format!("cublasLtMatrixLayoutCreate: {e}")))?;
        Ok(Self(layout))
    }
}

impl Drop for LtMatrixLayout {
    fn drop(&mut self) {
        let _ = unsafe { ltsys::cublasLtMatrixLayoutDestroy(self.0).result() };
    }
}

fn lt_data_type(data_type: sys::cudaDataType_t) -> Result<ltsys::cudaDataType_t, BenchError> {
    match data_type {
        sys::cudaDataType_t::CUDA_R_16F => Ok(ltsys::cudaDataType_t::CUDA_R_16F),
        sys::cudaDataType_t::CUDA_R_16BF => Ok(ltsys::cudaDataType_t::CUDA_R_16BF),
        sys::cudaDataType_t::CUDA_R_32F => Ok(ltsys::cudaDataType_t::CUDA_R_32F),
        _ => Err(BenchError(format!(
            "unsupported cuBLASLt shape dtype: {data_type:?}"
        ))),
    }
}

struct LtPreference(ltsys::cublasLtMatmulPreference_t);

impl LtPreference {
    fn new(workspace_bytes: usize) -> Result<Self, BenchError> {
        let preference = lt_result::create_matmul_pref()
            .map_err(|e| BenchError(format!("cublasLtMatmulPreferenceCreate: {e}")))?;
        let preference = Self(preference);
        let result = unsafe {
            ltsys::cublasLtMatmulPreferenceSetAttribute(
                preference.0,
                ltsys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                std::ptr::from_ref(&workspace_bytes).cast(),
                std::mem::size_of_val(&workspace_bytes),
            )
        };
        lt_status(result, "cublasLtMatmulPreferenceSetAttribute")?;
        Ok(preference)
    }
}

impl Drop for LtPreference {
    fn drop(&mut self) {
        let _ = unsafe { ltsys::cublasLtMatmulPreferenceDestroy(self.0).result() };
    }
}

fn lt_status(status: ltsys::cublasStatus_t, operation: &str) -> Result<(), BenchError> {
    status
        .result()
        .map_err(|e| BenchError(format!("{operation}: {e}")))
}

fn lt_algo_attribute<T: Copy>(
    algo: &ltsys::cublasLtMatmulAlgo_t,
    attribute: ltsys::cublasLtMatmulAlgoConfigAttributes_t,
) -> Result<T, BenchError> {
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    let mut written = 0;
    let status = unsafe {
        ltsys::cublasLtMatmulAlgoConfigGetAttribute(
            algo,
            attribute,
            value.as_mut_ptr().cast(),
            std::mem::size_of::<T>(),
            &raw mut written,
        )
    };
    lt_status(status, "cublasLtMatmulAlgoConfigGetAttribute")?;
    if written != std::mem::size_of::<T>() {
        return Err(BenchError(format!(
            "cuBLASLt algorithm attribute wrote {written} bytes, expected {}",
            std::mem::size_of::<T>()
        )));
    }
    Ok(unsafe { value.assume_init() })
}

fn read_lt_algo(algo: &ltsys::cublasLtMatmulAlgo_t) -> Result<GemmAlgo, BenchError> {
    Ok(GemmAlgo {
        id: lt_algo_attribute(algo, LtAlgoAttr::CUBLASLT_ALGO_CONFIG_ID)?,
        tile: lt_algo_attribute(algo, LtAlgoAttr::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
        stages: lt_algo_attribute(algo, LtAlgoAttr::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
        split_k: lt_algo_attribute(algo, LtAlgoAttr::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
        reduction_scheme: lt_algo_attribute(
            algo,
            LtAlgoAttr::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        )?,
        cta_swizzling: lt_algo_attribute(algo, LtAlgoAttr::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
        custom_option: lt_algo_attribute(algo, LtAlgoAttr::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
        workspace_bytes: 0,
        waves: 0.0,
    })
}

fn rounded_inputs(spec: &GemmSpec, values: &mut [f32]) {
    match spec.atype {
        sys::cudaDataType_t::CUDA_R_16F => {
            for value in values.iter_mut() {
                *value = f16::from_f32(*value).to_f32();
            }
        }
        sys::cudaDataType_t::CUDA_R_16BF => {
            for value in values.iter_mut() {
                *value = bf16::from_f32(*value).to_f32();
            }
        }
        _ => {}
    }
}

fn sample_gemm_output(
    shape: &GemmShapeCase,
    spec: &GemmSpec,
    a: &[f32],
    b: &[f32],
    output: &[f32],
) -> GemmCheck {
    let output_elements =
        usize::try_from(shape.m).unwrap_or(0) * usize::try_from(shape.n).unwrap_or(0);
    let sampled = 256_usize.min(output_elements);
    let mut max_abs_error: f64 = 0.0;
    let mut passed = true;
    for sample in 0..sampled {
        let index = sample * output_elements / sampled;
        let row = index % usize::try_from(shape.m).unwrap_or(1);
        let col = index / usize::try_from(shape.m).unwrap_or(1);
        let mut reference = 0.0_f64;
        for k in 0..usize::try_from(shape.k).unwrap_or(0) {
            reference += f64::from(a[row * usize::try_from(shape.k).unwrap_or(0) + k])
                * f64::from(b[col * usize::try_from(shape.k).unwrap_or(0) + k]);
        }
        let error = (f64::from(output[index]) - reference).abs();
        max_abs_error = max_abs_error.max(error);
        passed &= error <= spec.atol + spec.rtol * reference.abs();
    }
    GemmCheck {
        sampled: u32::try_from(sampled).unwrap_or(u32::MAX),
        max_abs_error,
        rtol: spec.rtol,
        atol: spec.atol,
        passed,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "cuBLASLt matmul needs its full descriptor set"
)]
fn lt_matmul(
    handle: ltsys::cublasLtHandle_t,
    desc: &LtMatmulDesc,
    a: &CudaSlice<u8>,
    a_layout: &LtMatrixLayout,
    b: &CudaSlice<u8>,
    b_layout: &LtMatrixLayout,
    c: &CudaSlice<u8>,
    c_layout: &LtMatrixLayout,
    d: &mut CudaSlice<u8>,
    workspace: &mut CudaSlice<u8>,
    algo: &ltsys::cublasLtMatmulAlgo_t,
    stream: &Arc<CudaStream>,
) -> Result<(), BenchError> {
    let (a_ptr, _a_guard) = a.device_ptr(stream);
    let (b_ptr, _b_guard) = b.device_ptr(stream);
    let (c_ptr, _c_guard) = c.device_ptr(stream);
    let (d_ptr, _d_guard) = d.device_ptr_mut(stream);
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(stream);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let result = unsafe {
        ltsys::cublasLtMatmul(
            handle,
            desc.0,
            std::ptr::from_ref(&alpha).cast(),
            a_ptr as *const std::ffi::c_void,
            a_layout.0,
            b_ptr as *const std::ffi::c_void,
            b_layout.0,
            std::ptr::from_ref(&beta).cast(),
            c_ptr as *const std::ffi::c_void,
            c_layout.0,
            d_ptr as *mut std::ffi::c_void,
            c_layout.0,
            algo,
            workspace_ptr as *mut std::ffi::c_void,
            GEMM_SHAPE_WORKSPACE_BYTES,
            stream.cu_stream() as ltsys::cudaStream_t,
        )
    };
    lt_status(result, "cublasLtMatmul")
}

struct GemmShapeBuffers {
    a_host: Vec<f32>,
    b_host: Vec<f32>,
    a: CudaSlice<u8>,
    b: CudaSlice<u8>,
    c: CudaSlice<u8>,
    d: CudaSlice<u8>,
    a_layout: LtMatrixLayout,
    b_layout: LtMatrixLayout,
    c_layout: LtMatrixLayout,
}

impl GemmShapeBuffers {
    fn new(
        stream: &Arc<CudaStream>,
        shape: &GemmShapeCase,
        spec: &GemmSpec,
    ) -> Result<Self, BenchError> {
        let a_len = usize::try_from(shape.k)
            .ok()
            .and_then(|k| usize::try_from(shape.m).ok().and_then(|m| k.checked_mul(m)))
            .ok_or_else(|| {
                BenchError("GEMM A dimensions overflow host address space".to_owned())
            })?;
        let b_len = usize::try_from(shape.k)
            .ok()
            .and_then(|k| usize::try_from(shape.n).ok().and_then(|n| k.checked_mul(n)))
            .ok_or_else(|| {
                BenchError("GEMM B dimensions overflow host address space".to_owned())
            })?;
        let c_len = usize::try_from(shape.m)
            .ok()
            .and_then(|m| usize::try_from(shape.n).ok().and_then(|n| m.checked_mul(n)))
            .ok_or_else(|| {
                BenchError("GEMM C dimensions overflow host address space".to_owned())
            })?;
        let seed = u64::from(shape.m) << 32 ^ u64::from(shape.n) << 16 ^ u64::from(shape.k);
        let mut a_host = random_f32(a_len, seed ^ 0xA51);
        let mut b_host = random_f32(b_len, seed ^ 0xB52);
        rounded_inputs(spec, &mut a_host);
        rounded_inputs(spec, &mut b_host);
        let a = upload_as_dtype(stream, &a_host, spec)?;
        let b = upload_as_dtype(stream, &b_host, spec)?;
        let element_bytes = if spec.ctype == sys::cudaDataType_t::CUDA_R_32F {
            4
        } else {
            2
        };
        let result_bytes = c_len.checked_mul(element_bytes).ok_or_else(|| {
            BenchError("GEMM output size overflows host address space".to_owned())
        })?;
        let c = unsafe { stream.alloc::<u8>(result_bytes) }?;
        let d = unsafe { stream.alloc::<u8>(result_bytes) }?;
        let a_layout = LtMatrixLayout::new(spec.atype, shape.k, shape.m, shape.k)?;
        let b_layout = LtMatrixLayout::new(spec.atype, shape.k, shape.n, shape.k)?;
        let c_layout = LtMatrixLayout::new(spec.ctype, shape.m, shape.n, shape.m)?;
        Ok(Self {
            a_host,
            b_host,
            a,
            b,
            c,
            d,
            a_layout,
            b_layout,
            c_layout,
        })
    }
}

struct SelectedLtAlgorithm {
    heuristic: ltsys::cublasLtMatmulHeuristicResult_t,
    metadata: GemmAlgo,
}

fn select_lt_algorithm(
    handle: ltsys::cublasLtHandle_t,
    desc: &LtMatmulDesc,
    layouts: (&LtMatrixLayout, &LtMatrixLayout, &LtMatrixLayout),
    preference: &LtPreference,
    shape: &GemmShapeCase,
    spec: &GemmSpec,
) -> Result<SelectedLtAlgorithm, BenchError> {
    let (a_layout, b_layout, c_layout) = layouts;
    let mut heuristic = std::mem::MaybeUninit::<ltsys::cublasLtMatmulHeuristicResult_t>::uninit();
    let mut returned_algorithms = 0;
    let status = unsafe {
        ltsys::cublasLtMatmulAlgoGetHeuristic(
            handle,
            desc.0,
            a_layout.0,
            b_layout.0,
            c_layout.0,
            c_layout.0,
            preference.0,
            1,
            heuristic.as_mut_ptr(),
            &raw mut returned_algorithms,
        )
    };
    lt_status(status, "cublasLtMatmulAlgoGetHeuristic")?;
    if returned_algorithms != 1 {
        return Err(BenchError(format!(
            "cuBLASLt returned {returned_algorithms} heuristic algorithms for {} {}",
            shape.name, spec.dtype
        )));
    }
    let heuristic = unsafe { heuristic.assume_init() };
    if heuristic.state != ltsys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(BenchError(format!(
            "cuBLASLt heuristic selected an unusable algorithm for {} {}: {:?}",
            shape.name, spec.dtype, heuristic.state
        )));
    }
    let mut metadata = read_lt_algo(&heuristic.algo)?;
    metadata.workspace_bytes = u64::try_from(heuristic.workspaceSize).unwrap_or(u64::MAX);
    metadata.waves = f64::from(heuristic.wavesCount);
    Ok(SelectedLtAlgorithm {
        heuristic,
        metadata,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "measurement reuses the shared CUDA timing machinery"
)]
fn measure_gemm_shape(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    monitor: &Monitor,
    handle: ltsys::cublasLtHandle_t,
    workspace: &mut CudaSlice<u8>,
    shape: &GemmShapeCase,
    spec: &GemmSpec,
    cfg: &Config,
) -> Result<(GemmShape, Vec<f64>, Vec<RoundState>), BenchError> {
    let mut buffers = GemmShapeBuffers::new(stream, shape, spec)?;
    let desc = LtMatmulDesc::new()?;
    let preference = LtPreference::new(GEMM_SHAPE_WORKSPACE_BYTES)?;
    let selected = select_lt_algorithm(
        handle,
        &desc,
        (&buffers.a_layout, &buffers.b_layout, &buffers.c_layout),
        &preference,
        shape,
        spec,
    )?;
    let (rounds, polls) = monitor
        .watch(|| {
            sustained_rounds(
                ctx,
                stream,
                cfg.gemm_warmup,
                cfg.gemm_round,
                cfg.rounds,
                || {
                    lt_matmul(
                        handle,
                        &desc,
                        &buffers.a,
                        &buffers.a_layout,
                        &buffers.b,
                        &buffers.b_layout,
                        &buffers.c,
                        &buffers.c_layout,
                        &mut buffers.d,
                        workspace,
                        &selected.heuristic.algo,
                        stream,
                    )
                },
            )
        })
        .map_err(BenchError)?;
    let (rounds, windows): (Vec<f64>, Vec<Window>) = rounds?.into_iter().unzip();
    let measured_check = sample_gemm_output(
        shape,
        spec,
        &buffers.a_host,
        &buffers.b_host,
        &read_as_f32(stream, &buffers.d, spec)?,
    );
    let median_ms = crate::stats::median(&mut rounds.clone());
    let result = GemmShape {
        name: shape.name.to_owned(),
        m: shape.m,
        n: shape.n,
        k: shape.k,
        dtype: spec.dtype,
        compute_type: "CUBLAS_COMPUTE_32F".to_owned(),
        scale_type: "CUDA_R_32F".to_owned(),
        trans_a: "T".to_owned(),
        trans_b: "N".to_owned(),
        lda: shape.k,
        ldb: shape.k,
        ldc: shape.m,
        epilogue: "CUBLASLT_EPILOGUE_DEFAULT".to_owned(),
        algorithm: selected.metadata,
        workspace_limit_bytes: u64::try_from(GEMM_SHAPE_WORKSPACE_BYTES).unwrap_or(u64::MAX),
        tflops: tflops(
            2.0 * f64::from(shape.m) * f64::from(shape.n) * f64::from(shape.k),
            median_ms,
        ),
        check: measured_check,
    };
    Ok((result, rounds, round_states(&polls, &windows)))
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

// Device memory from cuMemAlloc, freed on drop. Peer copies need this
// rather than stream-ordered pool allocations: cuCtxEnablePeerAccess does
// not cover pool memory, so cuMemcpyPeerAsync on pool buffers is silently
// staged through the host at roughly PCIe speed (~37 GB/s instead of ~400).
struct PeerBuffer {
    ctx: Arc<CudaContext>,
    ptr: dsys::CUdeviceptr,
}

impl PeerBuffer {
    fn new(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Self, BenchError> {
        ctx.bind_to_thread()?;
        let mut ptr: dsys::CUdeviceptr = 0;
        unsafe {
            dsys::cuMemAlloc_v2(&raw mut ptr, bytes)
                .result()
                .map_err(|e| BenchError(format!("cuMemAlloc {bytes}B: {e}")))?;
        }
        Ok(Self {
            ctx: ctx.clone(),
            ptr,
        })
    }
}

impl Drop for PeerBuffer {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = dsys::cuMemFree_v2(self.ptr).result();
        }
    }
}

/// `PCIe` link details reported by nvidia-smi for GPU 0.
#[derive(Clone, Copy, Debug, Default)]
struct PcieLink {
    r#gen: Option<u8>,
    gen_max: Option<u8>,
    gen_host_max: Option<u8>,
    width: Option<u8>,
    width_max: Option<u8>,
}

fn pcie_link() -> PcieLink {
    let parse = |s: &str| s.trim().parse::<u8>().ok();
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "-i",
            "0",
            "--query-gpu=pcie.link.gen.gpucurrent,pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.current,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            let mut parts = s.trim().split(',');
            PcieLink {
                r#gen: parts.next().and_then(parse),
                gen_max: parts.next().and_then(parse),
                gen_host_max: parts.next().and_then(parse),
                width: parts.next().and_then(parse),
                width_max: parts.next().and_then(parse),
            }
        });
    out.unwrap_or_default()
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
    progress("devices", &format!("{device_count}"));
    let ctx0 = CudaContext::new(0)?;
    let stream0 = ctx0.default_stream();
    let kernels = load_kernels(&ctx0)?;
    progress("kernels", "loaded");

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
    let pcie = pcie_link();
    progress(
        "pcie",
        &format!(
            "gen{}x{} (gpu max gen{}x{}, host max gen{})",
            pcie.r#gen.map_or("?".to_owned(), |v| v.to_string()),
            pcie.width.map_or("?".to_owned(), |v| v.to_string()),
            pcie.gen_max.map_or("?".to_owned(), |v| v.to_string()),
            pcie.width_max.map_or("?".to_owned(), |v| v.to_string()),
            pcie.gen_host_max.map_or("?".to_owned(), |v| v.to_string()),
        ),
    );
    if let (Some(g), Some(g_max), Some(w), Some(w_max)) =
        (pcie.r#gen, pcie.gen_max, pcie.width, pcie.width_max)
    {
        checks.insert("pcie_full_link".to_owned(), g == g_max && w == w_max);
    }

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
            // Peer copies must use cuMemAlloc memory, not stream-ordered pool
            // allocations: cuCtxEnablePeerAccess does not cover pool memory, so
            // cuMemcpyPeerAsync on pool buffers is silently staged through the
            // host at roughly PCIe speed (~37 GB/s measured instead of ~400).
            let peer_bytes = cfg.copy_mib * MIB;
            let peer_elems = peer_bytes / 4;
            let pattern = random_f32(peer_elems, 0xFEED);
            let src0 = PeerBuffer::new(&ctx0, peer_bytes)?;
            let dst1 = PeerBuffer::new(&ctx1, peer_bytes)?;
            let src1 = PeerBuffer::new(&ctx1, peer_bytes)?;
            let dst0 = PeerBuffer::new(&ctx0, peer_bytes)?;
            unsafe {
                dsys::cuMemcpyHtoDAsync_v2(
                    src0.ptr,
                    pattern.as_ptr().cast::<core::ffi::c_void>(),
                    peer_bytes,
                    stream0.cu_stream(),
                )
                .result()
                .map_err(|e| BenchError(format!("upload src0: {e}")))?;
                dsys::cuMemcpyHtoDAsync_v2(
                    src1.ptr,
                    pattern.as_ptr().cast::<core::ffi::c_void>(),
                    peer_bytes,
                    stream1.cu_stream(),
                )
                .result()
                .map_err(|e| BenchError(format!("upload src1: {e}")))?;
            }
            stream0.synchronize()?;
            stream1.synchronize()?;
            let fwd = cuda_samples(&ctx1, &stream1, 5, cfg.iterations_bw, cfg.rounds, || {
                unsafe {
                    dsys::cuMemcpyPeerAsync(
                        dst1.ptr,
                        ctx1.cu_ctx(),
                        src0.ptr,
                        ctx0.cu_ctx(),
                        peer_bytes,
                        stream1.cu_stream(),
                    )
                    .result()
                    .map_err(|e| BenchError(format!("cuMemcpyPeerAsync: {e}")))?;
                }
                Ok(())
            })?;
            let rev = cuda_samples(&ctx0, &stream0, 5, cfg.iterations_bw, cfg.rounds, || {
                unsafe {
                    dsys::cuMemcpyPeerAsync(
                        dst0.ptr,
                        ctx0.cu_ctx(),
                        src1.ptr,
                        ctx1.cu_ctx(),
                        peer_bytes,
                        stream0.cu_stream(),
                    )
                    .result()
                    .map_err(|e| BenchError(format!("cuMemcpyPeerAsync: {e}")))?;
                }
                Ok(())
            })?;
            let fwd_gbps = copy_bytes / (record("g2g_forward_gbps", &fwd) / 1e3) / 1e9;
            let rev_gbps = copy_bytes / (record("g2g_reverse_gbps", &rev) / 1e3) / 1e9;
            g2g_forward = Some(fwd_gbps);
            g2g_reverse = Some(rev_gbps);
            g2g_gbps = Some(fwd_gbps.min(rev_gbps));
            progress("g2g", &format!("{fwd_gbps:.0}/{rev_gbps:.0} GB/s"));
            let lat = wall_samples(5, 20, cfg.rounds, || {
                unsafe {
                    dsys::cuMemcpyPeerAsync(
                        dst1.ptr,
                        ctx1.cu_ctx(),
                        src0.ptr,
                        ctx0.cu_ctx(),
                        4,
                        stream1.cu_stream(),
                    )
                    .result()
                    .map_err(|e| BenchError(format!("cuMemcpyPeerAsync: {e}")))?;
                }
                stream1.synchronize()?;
                Ok(())
            })?;
            g2g_latency = Some(record("g2g_latency_us", &lat) * 1e3);
            let check_fwd = {
                let mut out = vec![0.0f32; peer_elems];
                unsafe {
                    dsys::cuMemcpyDtoHAsync_v2(
                        out.as_mut_ptr().cast::<core::ffi::c_void>(),
                        dst1.ptr,
                        peer_bytes,
                        stream1.cu_stream(),
                    )
                    .result()
                    .map_err(|e| BenchError(format!("readback dst1: {e}")))?;
                }
                stream1.synchronize()?;
                (out[0] - pattern[0]).abs() < f32::EPSILON
                    && (out[peer_elems / 2] - pattern[peer_elems / 2]).abs() < f32::EPSILON
            };
            let check_rev = {
                let mut out = vec![0.0f32; peer_elems];
                unsafe {
                    dsys::cuMemcpyDtoHAsync_v2(
                        out.as_mut_ptr().cast::<core::ffi::c_void>(),
                        dst0.ptr,
                        peer_bytes,
                        stream0.cu_stream(),
                    )
                    .result()
                    .map_err(|e| BenchError(format!("readback dst0: {e}")))?;
                }
                stream0.synchronize()?;
                (out[0] - pattern[0]).abs() < f32::EPSILON
                    && (out[peer_elems / 2] - pattern[peer_elems / 2]).abs() < f32::EPSILON
            };
            checks.insert("g2g_0_to_1".to_owned(), check_fwd);
            checks.insert("g2g_1_to_0".to_owned(), check_rev);
            let staged_via_host = fwd_gbps.min(rev_gbps) <= h2d_gbps;
            checks.insert("g2g_faster_than_host".to_owned(), !staged_via_host);
        } else {
            eprintln!("peer access unavailable; skipping g2g");
        }
    } else {
        eprintln!("single GPU; skipping g2g");
    }

    let monitor = Monitor::open(pci_bus_id(&ctx0)?).map_err(BenchError)?;
    let mut gemm_states = BTreeMap::new();
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
        let (gemm_rounds, polls) = monitor
            .watch(|| {
                sustained_rounds(
                    &ctx0,
                    &stream0,
                    cfg.gemm_warmup,
                    cfg.gemm_round,
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
                )
            })
            .map_err(BenchError)?;
        let (gemm_samples, windows): (Vec<f64>, Vec<Window>) = gemm_rounds?.into_iter().unzip();
        let key = format!("gemm_{}", spec.dtype);
        gemm_states.insert(key.clone(), round_states(&polls, &windows));
        let ms = record(&key, &gemm_samples);
        let flops = 2.0 * (n as f64).powi(3);
        let peak_tflops = tflops(flops, ms);
        compute_tflops.insert(spec.dtype, peak_tflops);
        progress(
            &format!("gemm_{}", spec.dtype),
            &format!("{peak_tflops:.0} TFLOP/s"),
        );
        checks.insert(
            format!("gemm_{}_check", spec.dtype),
            gemm_check(&blas, &stream0, &spec)?,
        );
    }
    drop(a_host);
    drop(b_host);

    let lt_handle = LtHandle::new()?;
    let mut lt_workspace = unsafe { stream0.alloc::<u8>(GEMM_SHAPE_WORKSPACE_BYTES) }?;
    let library_version = unsafe { ltsys::cublasLtGetVersion() };
    let library = format!("cuBLASLt {library_version} (cublasLtGetVersion)");
    let mut gemm_shapes = Vec::new();
    let mut gemm_shape_samples = BTreeMap::new();
    let mut gemm_shape_checks = BTreeMap::new();
    let mut gemm_shape_states = BTreeMap::new();
    for shape in gemm_shape_cases() {
        for spec in gemm_specs()
            .into_iter()
            .filter(|spec| matches!(spec.dtype, Dtype::Fp16 | Dtype::Bf16))
        {
            let (measurement, samples, states) = measure_gemm_shape(
                &ctx0,
                &stream0,
                &monitor,
                lt_handle.0,
                &mut lt_workspace,
                &shape,
                &spec,
                cfg,
            )?;
            let key = measurement.samples_key();
            let median_ms = crate::stats::median(&mut samples.clone());
            gemm_shape_samples.insert(key.clone(), samples);
            gemm_shape_states.insert(key.clone(), states);
            gemm_shape_checks.insert(format!("{key}_check"), measurement.check.passed);
            progress(
                &key,
                &format!(
                    "{:.0} TFLOP/s ({median_ms:.3} ms median)",
                    measurement.tflops
                ),
            );
            gemm_shapes.push(measurement);
        }
    }

    let (cc_major, cc_minor) = ctx0.compute_capability()?;
    let sm_count =
        ctx0.attribute(dsys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let captured_at = crate::iso_now();
    let gpu_request = if cfg.quick {
        format!("{gpu_request} --quick")
    } else {
        gpu_request.to_owned()
    };
    let device_name = ctx0.name()?;
    let driver_version = driver_version_string();
    let all_ok = checks.values().all(|v| *v) && gemm_shape_checks.values().all(|v| *v);

    Ok(Measured {
        captured_at: captured_at.clone(),
        gpu_request: gpu_request.clone(),
        device_name: device_name.clone(),
        device_count,
        compute_capability: format!("{cc_major}.{cc_minor}"),
        sm_count: sm_count.cast_unsigned(),
        memory_bytes: ctx0.total_mem()? as u64,
        driver_version: driver_version.clone(),
        cuda_version: "12.9".to_owned(),
        rounds: cfg.rounds,
        launch_us,
        sync_us: Some(sync_us),
        host_copy_gbps,
        h2d_gbps,
        d2h_gbps,
        h2d_latency_us: Some(h2d_latency_us),
        pcie_gen: pcie.r#gen,
        pcie_gen_max: pcie.gen_max,
        pcie_gen_host_max: pcie.gen_host_max,
        pcie_width: pcie.width,
        pcie_width_max: pcie.width_max,
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
        gemm_shapes: Some(GemmShapeSuite {
            captured_at,
            gpu_request,
            device_name,
            driver_version,
            cuda_version: "12.9".to_owned(),
            library,
            pcie_gen: pcie.r#gen,
            pcie_width: pcie.width,
            shapes: gemm_shapes,
            checks: gemm_shape_checks,
            samples_ms: gemm_shape_samples,
            gpu_state: Some(
                monitor
                    .state(cfg.gemm_warmup, cfg.gemm_round, gemm_shape_states)
                    .map_err(BenchError)?,
            ),
        }),
        checks,
        samples_ms,
        gpu_state: Some(
            monitor
                .state(cfg.gemm_warmup, cfg.gemm_round, gemm_states)
                .map_err(BenchError)?,
        ),
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
                    .chain(
                        m.gemm_shapes
                            .iter()
                            .flat_map(|suite| suite.checks.iter())
                            .filter(|(_, v)| !**v)
                            .map(|(k, _)| k)
                    )
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
