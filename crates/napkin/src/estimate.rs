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

/// How the host waits for copies.
///
/// The measured copy latencies (`h2d_latency_us` and friends) time one
/// 4-byte copy plus the stream synchronize that waits for it, on the wall
/// clock. That fixed cost is charged once per host wait, so the mode decides
/// how many times it is paid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CopyMode {
    /// Each copy is issued and waited for before the next one, e.g. a loop of
    /// `cudaMemcpy`, or `cudaMemcpyAsync` followed by
    /// `cudaStreamSynchronize`. Every copy pays the fixed latency.
    Sync,
    /// The copies of one kind are queued back-to-back on a stream and the
    /// host waits once at the end, e.g. 1,000 `cudaMemcpyAsync` calls and
    /// one `cudaStreamSynchronize`. The fixed latency is paid once.
    Async,
}

impl fmt::Display for CopyMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sync => "sync",
            Self::Async => "async",
        };
        f.write_str(name)
    }
}

/// One kind of copy: `bytes` in total, split over `copies` operations.
///
/// Both must agree: bytes without copies, copies without bytes, or less than
/// one byte per copy are rejected rather than guessed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Transfer {
    pub bytes: f64,
    pub copies: u64,
}

/// Why a transfer time is only a lower bound on the real time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowerBound {
    /// The profile has no measured fixed latency for this copy kind, so only
    /// `bytes / bandwidth` is counted. That is never zero cost in reality.
    NoLatency,
    /// Async copies after the first: the host does not wait for them, and the
    /// cost of queueing each one has not been measured.
    QueuedCopies,
}

impl fmt::Display for LowerBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match self {
            Self::NoLatency => "no measured copy latency, bandwidth only",
            Self::QueuedCopies => "per-copy cost of queued async copies is unmeasured",
        };
        f.write_str(reason)
    }
}

/// One transfer kind's time and every input that produced it:
/// `seconds = waits × latency + bytes / bandwidth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferTime {
    pub seconds: f64,
    pub bytes: f64,
    pub copies: u64,
    pub mode: CopyMode,
    /// Host waits charged the fixed latency: `copies` for sync, one for async.
    pub waits: u64,
    /// Fixed cost per wait; `None` when unmeasured.
    pub latency_us: Option<f64>,
    pub gbps: f64,
    pub lower_bound: Option<LowerBound>,
}

impl TransferTime {
    /// Times `transfer` with the alpha-beta (latency plus bandwidth) model.
    ///
    /// The bandwidth is the bulk 512 MiB rate, which small copies never
    /// reach, and the latency is the 4-byte synchronous round trip. A 4-byte
    /// sync H2D copy at 5 μs and 50 GB/s is `5 μs + 4 B / 50 GB/s =
    /// 5.00008 μs`; at 512 MiB the same latency is under 0.1% of the time.
    fn new(transfer: Transfer, mode: CopyMode, gbps: f64, latency_us: Option<f64>) -> Self {
        let waits = match mode {
            CopyMode::Sync => transfer.copies,
            CopyMode::Async => 1,
        };
        let bandwidth_s = transfer.bytes / (gbps * 1e9);
        #[expect(clippy::cast_precision_loss, reason = "copy counts never exceed 2^53")]
        let (seconds, lower_bound) = match latency_us {
            Some(latency) => (
                waits as f64 * latency / 1e6 + bandwidth_s,
                (waits < transfer.copies).then_some(LowerBound::QueuedCopies),
            ),
            None => (bandwidth_s, Some(LowerBound::NoLatency)),
        };
        Self {
            seconds,
            bytes: transfer.bytes,
            copies: transfer.copies,
            mode,
            waits,
            latency_us,
            gbps,
            lower_bound,
        }
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
    /// Host-to-device copies.
    pub h2d: Transfer,
    /// Device-to-host copies.
    pub d2h: Transfer,
    /// GPU-to-GPU copies.
    pub g2g: Transfer,
    /// How the host waits for the copies above.
    pub copy_mode: CopyMode,
    /// Back-to-back kernel launches. Copies are not launches, and a copy's
    /// measured latency already includes its stream synchronize, so neither
    /// `launch_us` nor `sync_us` is added to copies.
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
    /// `None` when the workload has no copies of that kind.
    pub h2d: Option<TransferTime>,
    pub d2h: Option<TransferTime>,
    pub g2g: Option<TransferTime>,
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

impl Estimate {
    /// True when any transfer stage is a lower bound, which makes the total
    /// one too, serial or overlapped.
    #[must_use]
    pub fn is_lower_bound(&self) -> bool {
        [self.h2d, self.d2h, self.g2g]
            .iter()
            .flatten()
            .any(|t| t.lower_bound.is_some())
    }
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

fn missing_rate(profile: &Profile, flag: &str, what: &str) -> EstimateError {
    EstimateError(format!(
        "{} has no measured {what}; run napkin-bench on it or drop {flag}",
        profile.id
    ))
}

/// Validates one transfer kind and times it. `kind` is `"h2d"`, `"d2h"`, or
/// `"g2g"`, which is also the CLI flag name.
fn transfer_time(
    profile: &Profile,
    kind: &str,
    transfer: Transfer,
    mode: CopyMode,
    gbps: Option<f64>,
    latency_us: Option<f64>,
) -> Result<Option<TransferTime>, EstimateError> {
    check_quantity(&format!("{kind} bytes"), transfer.bytes)?;
    #[expect(clippy::cast_precision_loss, reason = "copy counts never exceed 2^53")]
    let copies = transfer.copies as f64;
    match (transfer.bytes > 0.0, transfer.copies) {
        (false, 0) => return Ok(None),
        (true, 0) => {
            return Err(EstimateError(format!(
                "{kind}: {} B needs at least one copy; pass --{kind}-copies 1 or more",
                transfer.bytes
            )));
        }
        (false, n) => {
            return Err(EstimateError(format!(
                "{kind}: {n} copies of zero bytes; pass --{kind} with the bytes they move"
            )));
        }
        (true, n) if transfer.bytes < copies => {
            return Err(EstimateError(format!(
                "{kind}: {} B over {n} copies is under one byte per copy",
                transfer.bytes
            )));
        }
        (true, _) => {}
    }
    let gbps = gbps
        .ok_or_else(|| missing_rate(profile, &format!("--{kind}"), &format!("{kind} bandwidth")))?;
    Ok(Some(TransferTime::new(transfer, mode, gbps, latency_us)))
}

/// Estimates `workload` against `profile`'s resolved rates.
///
/// # Errors
/// Returns `EstimateError` for negative/NaN inputs, an unknown dtype,
/// transfer bytes and copy counts that disagree, or a workload stage that has
/// no rate on a datasheet-only profile.
pub fn estimate(
    profile: &Profile,
    workload: &Workload,
    mode: Mode,
    force_ceiling: bool,
) -> Result<Estimate, EstimateError> {
    check_quantity("flops", workload.flops)?;
    check_quantity("hbm bytes", workload.hbm_bytes)?;
    check_quantity("host bytes", workload.host_bytes)?;

    let rates = profile.rates(force_ceiling);
    let compute = rates.compute_tflops.get(&workload.dtype).ok_or_else(|| {
        EstimateError(format!(
            "{} has no {} throughput, measured or datasheet",
            profile.id, workload.dtype
        ))
    })?;
    let compute_s = workload.flops / (compute.tflops * 1e12);
    let memory_s = workload.hbm_bytes / (rates.hbm_gbps * 1e9);

    let host_s = if workload.host_bytes > 0.0 {
        let gbps = rates
            .host_copy_gbps
            .ok_or_else(|| missing_rate(profile, "--host", "host copy bandwidth"))?;
        workload.host_bytes / (gbps * 1e9)
    } else {
        0.0
    };
    let copy_mode = workload.copy_mode;
    let h2d = transfer_time(
        profile,
        "h2d",
        workload.h2d,
        copy_mode,
        rates.h2d_gbps,
        rates.h2d_latency_us,
    )?;
    let d2h = transfer_time(
        profile,
        "d2h",
        workload.d2h,
        copy_mode,
        rates.d2h_gbps,
        rates.d2h_latency_us,
    )?;
    let g2g = transfer_time(
        profile,
        "g2g",
        workload.g2g,
        copy_mode,
        Some(rates.g2g_gbps),
        rates.g2g_latency_us,
    )?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "launch counts never exceed 2^53"
    )]
    let launch_s = if workload.launches > 0 {
        let launch_us = rates
            .launch_us
            .ok_or_else(|| missing_rate(profile, "--launches", "launch latency"))?;
        workload.launches as f64 * launch_us / 1e6
    } else {
        0.0
    };

    let seconds = |t: Option<TransferTime>| t.map_or(0.0, |t| t.seconds);
    let device_s = compute_s.max(memory_s);
    let stages = [
        host_s,
        seconds(h2d),
        device_s,
        seconds(g2g),
        seconds(d2h),
        launch_s,
    ];
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

    let bottleneck = bottleneck(stages, compute_s, memory_s);

    Ok(Estimate {
        compute_s,
        memory_s,
        device_s,
        host_s,
        h2d,
        d2h,
        g2g,
        launch_s,
        total_s,
        intensity,
        ridge,
        bottleneck,
        compute_basis: compute.basis,
    })
}

fn bottleneck(stages: [f64; 6], compute_s: f64, memory_s: f64) -> Bottleneck {
    let ordered = [
        (stages[2], None),
        (stages[0], Some(Bottleneck::Host)),
        (stages[1], Some(Bottleneck::H2d)),
        (stages[4], Some(Bottleneck::D2h)),
        (stages[3], Some(Bottleneck::Interconnect)),
        (stages[5], Some(Bottleneck::Launch)),
    ];
    let mut winner = if compute_s > memory_s {
        Bottleneck::Compute
    } else if memory_s > compute_s {
        Bottleneck::Memory
    } else {
        Bottleneck::Balanced
    };
    let mut biggest = stages[2];
    for (seconds, kind) in ordered.iter().skip(1) {
        if *seconds > biggest {
            biggest = *seconds;
            if let Some(kind) = kind {
                winner = *kind;
            }
        }
    }
    winner
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
                sync_us: Some(8.0),
                host_copy_gbps: 100.0,
                h2d_gbps: 50.0,
                d2h_gbps: 40.0,
                pcie_gen: None,
                pcie_gen_max: None,
                pcie_gen_host_max: None,
                pcie_width: None,
                pcie_width_max: None,
                h2d_latency_us: Some(5.0),
                d2h_latency_us: Some(6.0),
                hbm_gbps: 2000.0,
                elementwise_gbps: 1500.0,
                hbm_latency_ns: Some(400.0),
                g2g_gbps: Some(300.0),
                g2g_forward_gbps: Some(300.0),
                g2g_reverse_gbps: Some(300.0),
                g2g_latency_us: Some(3.0),
                compute_tflops: compute,
                gemm_shape: [8192, 8192, 8192],
                checks: BTreeMap::from([("hbm_copy".to_owned(), true)]),
                samples_ms: BTreeMap::new(),
                gpu_state: None,
            }),
        }
    }

    #[test]
    fn serial_estimate_includes_copy_latency_and_launch() {
        let workload = Workload {
            dtype: Dtype::Fp16,
            flops: 8e12,
            hbm_bytes: 200e9,
            host_bytes: 2e9,
            h2d: Transfer {
                bytes: 5e9,
                copies: 1,
            },
            d2h: Transfer {
                bytes: 4e9,
                copies: 1,
            },
            g2g: Transfer {
                bytes: 30e9,
                copies: 1,
            },
            copy_mode: CopyMode::Sync,
            launches: 10,
        };
        let e = estimate(&fixture(), &workload, Mode::Serial, false).unwrap();
        assert!((e.host_s - 0.020).abs() < 1e-9);
        assert!((e.compute_s - 0.010).abs() < 1e-9);
        assert!((e.memory_s - 0.100).abs() < 1e-9);
        assert!((e.h2d.unwrap().seconds - 0.100_005).abs() < 1e-9);
        assert!((e.d2h.unwrap().seconds - 0.100_006).abs() < 1e-9);
        assert!((e.g2g.unwrap().seconds - 0.100_003).abs() < 1e-9);
        assert!((e.launch_s - 0.00005).abs() < 1e-9);
        assert!((e.total_s - 0.420_064).abs() < 1e-9);
        assert_eq!(e.bottleneck, Bottleneck::D2h);
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
            h2d: Transfer {
                bytes: 5e9,
                copies: 1,
            },
            d2h: Transfer {
                bytes: 4e9,
                copies: 1,
            },
            g2g: Transfer {
                bytes: 30e9,
                copies: 1,
            },
            copy_mode: CopyMode::Sync,
            launches: 10,
        };
        let e = estimate(&fixture(), &workload, Mode::Overlapped, false).unwrap();
        assert!((e.total_s - 0.100_006).abs() < 1e-9);
    }

    #[test]
    fn compute_only_intensity_is_infinite() {
        let workload = Workload {
            dtype: Dtype::Fp16,
            flops: 8e12,
            hbm_bytes: 0.0,
            host_bytes: 0.0,
            h2d: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            d2h: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            g2g: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            copy_mode: CopyMode::Sync,
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
            h2d: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            d2h: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            g2g: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            copy_mode: CopyMode::Sync,
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
            h2d: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            d2h: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            g2g: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            copy_mode: CopyMode::Sync,
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
            h2d: Transfer {
                bytes: 1e9,
                copies: 1,
            },
            d2h: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            g2g: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            copy_mode: CopyMode::Sync,
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
            h2d: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            d2h: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            g2g: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            copy_mode: CopyMode::Sync,
            launches: 0,
        };
        let err = estimate(&profile, &workload, Mode::Serial, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no fp8 throughput"), "{err}");
    }

    fn copies_only(h2d: Transfer, copy_mode: CopyMode) -> Workload {
        Workload {
            dtype: Dtype::Fp16,
            flops: 0.0,
            hbm_bytes: 0.0,
            host_bytes: 0.0,
            h2d,
            d2h: Transfer::default(),
            g2g: Transfer::default(),
            copy_mode,
            launches: 0,
        }
    }

    fn h2d_time(profile: &Profile, workload: &Workload) -> TransferTime {
        estimate(profile, workload, Mode::Serial, false)
            .unwrap()
            .h2d
            .unwrap()
    }

    #[test]
    fn four_byte_sync_copy_pays_fixed_latency_plus_bandwidth() {
        let one_word = Transfer {
            bytes: 4.0,
            copies: 1,
        };
        let e = estimate(
            &fixture(),
            &copies_only(one_word, CopyMode::Sync),
            Mode::Serial,
            false,
        )
        .unwrap();
        let h2d = e.h2d.unwrap();
        assert!((h2d.seconds - 5.000_08e-6).abs() < 1e-15, "{}", h2d.seconds);
        assert_eq!(h2d.waits, 1);
        assert_eq!(h2d.lower_bound, None);
        assert!(!e.is_lower_bound());
        assert!((e.launch_s - 0.0).abs() < f64::EPSILON);
        assert!((e.total_s - 5.000_08e-6).abs() < 1e-15);
    }

    #[test]
    fn many_small_sync_copies_cost_more_than_one_large_copy() {
        let profile = fixture();
        let one = h2d_time(
            &profile,
            &copies_only(
                Transfer {
                    bytes: 1e9,
                    copies: 1,
                },
                CopyMode::Sync,
            ),
        );
        let many = h2d_time(
            &profile,
            &copies_only(
                Transfer {
                    bytes: 1e9,
                    copies: 1000,
                },
                CopyMode::Sync,
            ),
        );
        assert!((one.seconds - 0.020_005).abs() < 1e-12);
        assert!((many.seconds - 0.025).abs() < 1e-12);
        assert_eq!(many.waits, 1000);
        assert_eq!(many.lower_bound, None);
    }

    #[test]
    fn async_copies_pay_latency_once_and_are_lower_bounds() {
        let profile = fixture();
        let batch = Transfer {
            bytes: 1e9,
            copies: 1000,
        };
        let queued = h2d_time(&profile, &copies_only(batch, CopyMode::Async));
        assert!((queued.seconds - 0.020_005).abs() < 1e-12);
        assert_eq!(queued.waits, 1);
        assert_eq!(queued.lower_bound, Some(LowerBound::QueuedCopies));

        let single = Transfer {
            bytes: 4.0,
            copies: 1,
        };
        let sync = h2d_time(&profile, &copies_only(single, CopyMode::Sync));
        let async_one = h2d_time(&profile, &copies_only(single, CopyMode::Async));
        assert_eq!(
            sync,
            TransferTime {
                mode: CopyMode::Sync,
                ..async_one
            }
        );
        assert_eq!(async_one.lower_bound, None);
    }

    #[test]
    fn missing_latency_is_a_bandwidth_only_lower_bound() {
        let mut profile = fixture();
        if let Some(m) = profile.measured.as_mut() {
            m.h2d_latency_us = None;
        }
        let one_word = Transfer {
            bytes: 4.0,
            copies: 1,
        };
        let e = estimate(
            &profile,
            &copies_only(one_word, CopyMode::Sync),
            Mode::Serial,
            false,
        )
        .unwrap();
        let h2d = e.h2d.unwrap();
        assert!((h2d.seconds - 8e-11).abs() < 1e-20);
        assert_eq!(h2d.latency_us, None);
        assert_eq!(h2d.lower_bound, Some(LowerBound::NoLatency));
        assert!(e.is_lower_bound());
    }

    #[test]
    fn ceiling_g2g_has_no_latency_and_is_a_lower_bound() {
        let mut workload = copies_only(Transfer::default(), CopyMode::Sync);
        workload.g2g = Transfer {
            bytes: 300e9,
            copies: 1,
        };
        let e = estimate(&fixture(), &workload, Mode::Serial, true).unwrap();
        let g2g = e.g2g.unwrap();
        assert!((g2g.seconds - 1.0).abs() < 1e-12);
        assert_eq!(g2g.lower_bound, Some(LowerBound::NoLatency));
        assert!(e.is_lower_bound());
    }

    #[test]
    fn zero_copies_of_zero_bytes_is_no_transfer() {
        let e = estimate(
            &fixture(),
            &copies_only(Transfer::default(), CopyMode::Sync),
            Mode::Serial,
            false,
        )
        .unwrap();
        assert_eq!(e.h2d, None);
        assert!((e.total_s - 0.0).abs() < f64::EPSILON);
        assert!(!e.is_lower_bound());
    }

    #[test]
    fn inconsistent_copy_counts_are_rejected() {
        let cases = [
            (4.0, 0, "needs at least one copy"),
            (0.0, 3, "3 copies of zero bytes"),
            (4.0, 8, "under one byte per copy"),
        ];
        for (bytes, copies, message) in cases {
            let workload = copies_only(Transfer { bytes, copies }, CopyMode::Sync);
            let err = estimate(&fixture(), &workload, Mode::Serial, false)
                .unwrap_err()
                .to_string();
            assert!(err.contains(message), "{err}");
        }
    }
}
