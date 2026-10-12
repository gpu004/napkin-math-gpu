//! Roofline estimation: compute vs HBM on device, plus host, transfer,
//! interconnect, and launch stages, summed serially or overlapped.

use std::fmt;

use crate::profile::{Basis, Dtype, Profile};

/// How the stages compose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    Serial,
    Overlapped,
}

/// Which stage bounds the estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
    Balanced,
    Host,
    H2d,
    D2h,
    Interconnect,
    Launch,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Compute => "compute-bound",
            Self::Memory => "memory-bound",
            Self::Balanced => "balanced",
            Self::Host => "host-copy-bound",
            Self::H2d => "h2d-bound",
            Self::D2h => "d2h-bound",
            Self::Interconnect => "interconnect-bound",
            Self::Launch => "launch-bound",
        };
        f.write_str(name)
    }
}

/// Work to estimate. All quantities are non-negative finite numbers.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    pub dtype: Dtype,
    /// Total FLOPs.
    pub flops: f64,
    /// Bytes of HBM traffic (algorithmic bytes).
    pub hbm_bytes: f64,
    /// Bytes copied within host memory before the transfer.
    pub host_bytes: f64,
    /// Bytes moved host to device.
    pub h2d_bytes: f64,
    /// Bytes moved device to host.
    pub d2h_bytes: f64,
    /// Bytes moved between GPUs.
    pub g2g_bytes: f64,
    /// Kernel launches.
    pub launches: u64,
}

/// Stage times in seconds plus roofline diagnostics.
#[derive(Clone, Copy, Debug)]
pub struct Estimate {
    pub compute_s: f64,
    pub memory_s: f64,
    /// `max(compute_s, memory_s)`.
    pub device_s: f64,
    pub host_s: f64,
    pub h2d_s: f64,
    pub d2h_s: f64,
    pub g2g_s: f64,
    pub launch_s: f64,
    pub total_s: f64,
    /// FLOP per HBM byte; infinite when `flops > 0 && hbm_bytes == 0`.
    pub intensity: f64,
    /// `compute_tflops * 1000 / hbm_gbps` in FLOP/byte.
    pub ridge: f64,
    pub bottleneck: Bottleneck,
    /// Whether the dtype throughput came from measurement or the datasheet.
    pub compute_basis: Basis,
}

/// An estimate that could not be produced.
#[derive(Debug)]
pub struct EstimateError(pub String);

impl fmt::Display for EstimateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EstimateError {}

fn check_quantity(name: &str, value: f64) -> Result<(), EstimateError> {
    if !(value >= 0.0 && value.is_finite()) {
        return Err(EstimateError(format!(
            "{name} must be a non-negative finite quantity"
        )));
    }
    Ok(())
}

/// Estimates `workload` against `profile`'s resolved rates.
///
/// # Errors
/// Returns `EstimateError` for negative/NaN inputs, an unknown dtype, or a
/// workload stage that has no rate on a datasheet-only profile.
pub fn estimate(
    profile: &Profile,
    workload: &Workload,
    mode: Mode,
    force_ceiling: bool,
) -> Result<Estimate, EstimateError> {
    check_quantity("flops", workload.flops)?;
    check_quantity("hbm bytes", workload.hbm_bytes)?;
    check_quantity("host bytes", workload.host_bytes)?;
    check_quantity("h2d bytes", workload.h2d_bytes)?;
    check_quantity("d2h bytes", workload.d2h_bytes)?;
    check_quantity("g2g bytes", workload.g2g_bytes)?;

    let rates = profile.rates(force_ceiling);
    let compute = rates.compute_tflops.get(&workload.dtype).ok_or_else(|| {
        EstimateError(format!(
            "{} has no {} throughput, measured or datasheet",
            profile.id, workload.dtype
        ))
    })?;
    let compute_s = workload.flops / (compute.tflops * 1e12);
    let memory_s = workload.hbm_bytes / (rates.hbm_gbps * 1e9);

    let need = |rate: Option<f64>, flag: &str, what: &str| -> Result<f64, EstimateError> {
        rate.ok_or_else(|| {
            EstimateError(format!(
                "{} has no measured {what}; run napkin-bench on it or drop {flag}",
                profile.id
            ))
        })
    };

    let host_s = if workload.host_bytes > 0.0 {
        workload.host_bytes / (need(rates.host_copy_gbps, "--host", "host copy bandwidth")? * 1e9)
    } else {
        0.0
    };
    let h2d_s = if workload.h2d_bytes > 0.0 {
        workload.h2d_bytes / (need(rates.h2d_gbps, "--h2d", "h2d bandwidth")? * 1e9)
    } else {
        0.0
    };
    let d2h_s = if workload.d2h_bytes > 0.0 {
        workload.d2h_bytes / (need(rates.d2h_gbps, "--d2h", "d2h bandwidth")? * 1e9)
    } else {
        0.0
    };
    let g2g_s = workload.g2g_bytes / (rates.g2g_gbps * 1e9);
    #[expect(
        clippy::cast_precision_loss,
        reason = "launch counts never exceed 2^53"
    )]
    let launch_s = if workload.launches > 0 {
        workload.launches as f64 * need(rates.launch_us, "--launches", "launch latency")? / 1e6
    } else {
        0.0
    };

    let device_s = compute_s.max(memory_s);
    let stages = [host_s, h2d_s, device_s, g2g_s, d2h_s, launch_s];
    let total_s = match mode {
        Mode::Serial => stages.iter().sum(),
        Mode::Overlapped => stages.iter().copied().fold(0.0, f64::max),
    };

    let intensity = if workload.hbm_bytes == 0.0 {
        if workload.flops > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        workload.flops / workload.hbm_bytes
    };
    let ridge = compute.tflops * 1000.0 / rates.hbm_gbps;

    let ordered = [
        (device_s, None),
        (host_s, Some(Bottleneck::Host)),
        (h2d_s, Some(Bottleneck::H2d)),
        (d2h_s, Some(Bottleneck::D2h)),
        (g2g_s, Some(Bottleneck::Interconnect)),
        (launch_s, Some(Bottleneck::Launch)),
    ];
    let mut bottleneck = if compute_s > memory_s {
        Bottleneck::Compute
    } else if memory_s > compute_s {
        Bottleneck::Memory
    } else {
        Bottleneck::Balanced
    };
    let mut biggest = device_s;
    for (seconds, kind) in ordered.iter().skip(1) {
        if *seconds > biggest {
            biggest = *seconds;
            if let Some(kind) = kind {
                bottleneck = *kind;
            }
        }
    }

    Ok(Estimate {
        compute_s,
        memory_s,
        device_s,
        host_s,
        h2d_s,
        d2h_s,
        g2g_s,
        launch_s,
        total_s,
        intensity,
        ridge,
        bottleneck,
        compute_basis: compute.basis,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::profile::{Ceiling, Measured};

    fn fixture() -> Profile {
        let compute = BTreeMap::from([
            (Dtype::Fp16, 800.0),
            (Dtype::Bf16, 750.0),
            (Dtype::Tf32, 400.0),
            (Dtype::Fp32, 50.0),
        ]);
        Profile {
            id: "test".to_owned(),
            name: "Test GPU".to_owned(),
            architecture: "Test".to_owned(),
            modal_gpu: "TEST".to_owned(),
            memory_gb: 80.0,
            ceiling: Ceiling {
                hbm_gbps: 2000.0,
                nvlink_gbps: 300.0,
                compute_tflops: compute.clone(),
                source: "fixture".to_owned(),
            },
            measured: Some(Measured {
                captured_at: "2026-08-08T00:00:00+00:00".to_owned(),
                gpu_request: "TEST!:2".to_owned(),
                device_name: "Test GPU".to_owned(),
                device_count: 2,
                compute_capability: "9.0".to_owned(),
                sm_count: 132,
                memory_bytes: 80 * (1 << 30),
                driver_version: "580.95.05".to_owned(),
                cuda_version: "12.9".to_owned(),
                rounds: 7,
                launch_us: 5.0,
                sync_us: 8.0,
                host_copy_gbps: 100.0,
                h2d_gbps: 50.0,
                d2h_gbps: 40.0,
                h2d_latency_us: 5.0,
                d2h_latency_us: 6.0,
                hbm_gbps: 2000.0,
                elementwise_gbps: 1500.0,
                hbm_latency_ns: 400.0,
                g2g_gbps: Some(300.0),
                g2g_forward_gbps: Some(300.0),
                g2g_reverse_gbps: Some(300.0),
                g2g_latency_us: Some(3.0),
                compute_tflops: compute,
                gemm_shape: [8192, 8192, 8192],
                checks: BTreeMap::from([("hbm_copy".to_owned(), true)]),
                samples_ms: BTreeMap::new(),
            }),
        }
    }

    #[test]
    fn memory_bound_serial_estimate_includes_transfer_and_launch() {
        let workload = Workload {
            dtype: Dtype::Fp16,
            flops: 8e12,
            hbm_bytes: 200e9,
            host_bytes: 2e9,
            h2d_bytes: 5e9,
            d2h_bytes: 4e9,
            g2g_bytes: 30e9,
            launches: 10,
        };
        let e = estimate(&fixture(), &workload, Mode::Serial, false).unwrap();
        assert!((e.host_s - 0.020).abs() < 1e-9);
        assert!((e.compute_s - 0.010).abs() < 1e-9);
        assert!((e.memory_s - 0.100).abs() < 1e-9);
        assert!((e.h2d_s - 0.100).abs() < 1e-9);
        assert!((e.d2h_s - 0.100).abs() < 1e-9);
        assert!((e.g2g_s - 0.100).abs() < 1e-9);
        assert!((e.launch_s - 0.00005).abs() < 1e-9);
        assert!((e.total_s - 0.42005).abs() < 1e-9);
        assert_eq!(e.bottleneck, Bottleneck::Memory);
        assert!((e.intensity - 40.0).abs() < 1e-9);
        assert!((e.ridge - 400.0).abs() < 1e-9);
    }

    #[test]
    fn overlapped_total_is_max_of_stages() {
        let workload = Workload {
            dtype: Dtype::Fp16,
            flops: 8e12,
            hbm_bytes: 200e9,
            host_bytes: 2e9,
            h2d_bytes: 5e9,
            d2h_bytes: 4e9,
            g2g_bytes: 30e9,
            launches: 10,
        };
        let e = estimate(&fixture(), &workload, Mode::Overlapped, false).unwrap();
        assert!((e.total_s - 0.100).abs() < 1e-9);
    }

    #[test]
    fn compute_only_intensity_is_infinite() {
        let workload = Workload {
            dtype: Dtype::Fp16,
            flops: 8e12,
            hbm_bytes: 0.0,
            host_bytes: 0.0,
            h2d_bytes: 0.0,
            d2h_bytes: 0.0,
            g2g_bytes: 0.0,
            launches: 1,
        };
        let e = estimate(&fixture(), &workload, Mode::Serial, false).unwrap();
        assert!(e.intensity.is_infinite());
        assert_eq!(e.bottleneck, Bottleneck::Compute);
    }

    #[test]
    fn idle_workload_has_zero_intensity() {
        let workload = Workload {
            dtype: Dtype::Fp16,
            flops: 0.0,
            hbm_bytes: 0.0,
            host_bytes: 0.0,
            h2d_bytes: 0.0,
            d2h_bytes: 0.0,
            g2g_bytes: 0.0,
            launches: 0,
        };
        let e = estimate(&fixture(), &workload, Mode::Serial, false).unwrap();
        assert!((e.intensity - 0.0).abs() < f64::EPSILON);
        assert!((e.total_s - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn negative_and_nan_work_is_rejected() {
        let mut workload = Workload {
            dtype: Dtype::Fp16,
            flops: -1.0,
            hbm_bytes: 0.0,
            host_bytes: 0.0,
            h2d_bytes: 0.0,
            d2h_bytes: 0.0,
            g2g_bytes: 0.0,
            launches: 0,
        };
        assert!(estimate(&fixture(), &workload, Mode::Serial, false).is_err());
        workload.flops = f64::NAN;
        assert!(estimate(&fixture(), &workload, Mode::Serial, false).is_err());
    }

    #[test]
    fn datasheet_only_gpu_errors_on_h2d() {
        let mut profile = fixture();
        profile.measured = None;
        let workload = Workload {
            dtype: Dtype::Fp16,
            flops: 1e12,
            hbm_bytes: 1e9,
            host_bytes: 0.0,
            h2d_bytes: 1e9,
            d2h_bytes: 0.0,
            g2g_bytes: 0.0,
            launches: 0,
        };
        let err = estimate(&profile, &workload, Mode::Serial, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no measured h2d bandwidth"), "{err}");
    }

    #[test]
    fn missing_dtype_in_both_maps_errors() {
        let mut profile = fixture();
        profile.measured = None;
        let workload = Workload {
            dtype: Dtype::Fp8,
            flops: 1e12,
            hbm_bytes: 1e9,
            host_bytes: 0.0,
            h2d_bytes: 0.0,
            d2h_bytes: 0.0,
            g2g_bytes: 0.0,
            launches: 0,
        };
        let err = estimate(&profile, &workload, Mode::Serial, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no fp8 throughput"), "{err}");
    }
}
