//! GPU profile types shared by the estimator and benchmark harness.

pub mod collective;

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// GPU numeric precisions, ordered low to high bit-width.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Fp4,
    Fp8,
    Fp16,
    Bf16,
    Tf32,
    Fp32,
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Fp4 => "fp4",
            Self::Fp8 => "fp8",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Tf32 => "tf32",
            Self::Fp32 => "fp32",
        };
        f.write_str(name)
    }
}

/// Datasheet numbers, always dense.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ceiling {
    /// Theoretical HBM bandwidth, GB/s (10^9 bytes/s).
    pub hbm_gbps: f64,
    /// Per-GPU `NVLink` aggregate as NVIDIA publishes it, both directions
    /// summed, GB/s: H100 is 18 links × 50 GB/s = 900 GB/s, 450 GB/s each way.
    pub nvlink_gbps: f64,
    /// Dense peak throughput per dtype, TFLOP/s (10^12 FLOP/s).
    pub compute_tflops: BTreeMap<Dtype, f64>,
    /// Source URLs and how the figures were derived.
    pub source: String,
}

/// One GPU SKU's datasheet profile plus any measured calibration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Profile {
    /// Short identifier, e.g. `"h100"`.
    pub id: String,
    /// Full SKU name, e.g. `"H100 SXM 80GB"`.
    pub name: String,
    /// Architecture family, e.g. `"Hopper"`.
    pub architecture: String,
    /// Modal `gpu=` request that lands on this GPU, e.g. `"H100!"`.
    pub modal_gpu: String,
    /// HBM capacity in GB (10^9 bytes).
    pub memory_gb: f64,
    pub ceiling: Ceiling,
    /// Attached by the registry from `<id>.measured.json` when it exists.
    #[serde(skip)]
    pub measured: Option<Measured>,
}

impl Profile {
    /// Resolves the rates used for estimation: measured values when a measured
    /// profile is attached, else datasheet ceilings. `force_ceiling` implements
    /// `--ceiling`. GPU-to-GPU falls back to the `NVLink` ceiling when unmeasured;
    /// host-link, launch, and copy-latency figures have no datasheet fallback.
    #[must_use]
    pub fn rates(&self, force_ceiling: bool) -> Rates {
        let mut compute_tflops = BTreeMap::new();
        if !force_ceiling && let Some(m) = &self.measured {
            for (dtype, _) in self
                .ceiling
                .compute_tflops
                .iter()
                .chain(m.compute_tflops.iter())
            {
                let (tflops, basis) = match m.compute_tflops.get(dtype) {
                    Some(&v) => (v, Basis::Measured),
                    None => match self.ceiling.compute_tflops.get(dtype) {
                        Some(&v) => (v, Basis::Ceiling),
                        None => continue,
                    },
                };
                compute_tflops.insert(*dtype, ComputeRate { tflops, basis });
            }
            return Rates {
                compute_tflops,
                hbm_gbps: m.hbm_gbps,
                h2d_gbps: Some(m.h2d_gbps),
                d2h_gbps: Some(m.d2h_gbps),
                host_copy_gbps: Some(m.host_copy_gbps),
                g2g_gbps: m.g2g_gbps.unwrap_or(self.ceiling.nvlink_gbps),
                h2d_latency_us: m.h2d_latency_us,
                d2h_latency_us: m.d2h_latency_us,
                g2g_latency_us: m.g2g_latency_us,
                launch_us: Some(m.launch_us),
                basis: Basis::Measured,
            };
        }
        for (dtype, &tflops) in &self.ceiling.compute_tflops {
            compute_tflops.insert(
                *dtype,
                ComputeRate {
                    tflops,
                    basis: Basis::Ceiling,
                },
            );
        }
        Rates {
            compute_tflops,
            hbm_gbps: self.ceiling.hbm_gbps,
            h2d_gbps: None,
            d2h_gbps: None,
            host_copy_gbps: None,
            g2g_gbps: self.ceiling.nvlink_gbps,
            h2d_latency_us: None,
            d2h_latency_us: None,
            g2g_latency_us: None,
            launch_us: None,
            basis: Basis::Ceiling,
        }
    }
}

/// Output of `napkin-bench`; all throughputs are decimal
/// (GB/s = 1e9 B/s, TFLOP/s = 1e12 FLOP/s).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Measured {
    pub captured_at: String,
    pub gpu_request: String,
    pub device_name: String,
    pub device_count: u32,
    pub compute_capability: String,
    pub sm_count: u32,
    pub memory_bytes: u64,
    pub driver_version: String,
    pub cuda_version: String,
    pub rounds: u32,
    /// Back-to-back empty-kernel launches, event-timed / N.
    pub launch_us: f64,
    /// Launch + stream synchronize round trip, wall clock.
    pub sync_us: Option<f64>,
    /// Pinned host memcpy, read+write bytes counted, min(8, cpus) threads.
    pub host_copy_gbps: f64,
    /// 512 MiB pinned async copies, event-timed.
    pub h2d_gbps: f64,
    pub d2h_gbps: f64,
    /// `PCIe` link on GPU 0 at the time of the host copies; None if the
    /// platform does not report it.
    #[serde(default)]
    pub pcie_gen: Option<u8>,
    /// `min(gpumax, hostmax)`; None if neither max is reported.
    #[serde(default)]
    pub pcie_gen_max: Option<u8>,
    #[serde(default)]
    pub pcie_width: Option<u8>,
    #[serde(default)]
    pub pcie_width_max: Option<u8>,
    /// 4-byte synchronous copies, wall clock median: issuing one copy, the
    /// copy itself, and the stream synchronize that waits for it. It is a
    /// different quantity from `h2d_gbps`: the bandwidth rows are event-timed
    /// on the GPU and exclude the host round trip.
    pub h2d_latency_us: Option<f64>,
    pub d2h_latency_us: Option<f64>,
    /// 512 MiB device-to-device memcpy, read+write counted (2x).
    pub hbm_gbps: f64,
    /// NVRTC `c[i] = a[i] + b[i]` fp32, 256 MiB per buffer, 3x bytes counted.
    pub elementwise_gbps: f64,
    /// Single-thread pointer chase over a random cycle in a 1 GiB buffer.
    pub hbm_latency_ns: Option<f64>,
    /// 512 MiB peer copy, slower of the two directions.
    pub g2g_gbps: Option<f64>,
    pub g2g_forward_gbps: Option<f64>,
    pub g2g_reverse_gbps: Option<f64>,
    /// 4-byte peer copy plus stream synchronize, wall clock median; same
    /// convention as `h2d_latency_us`.
    pub g2g_latency_us: Option<f64>,
    /// cuBLAS `GemmEx` 8192^3 (fp16, bf16, tf32, fp32).
    pub compute_tflops: BTreeMap<Dtype, f64>,
    pub gemm_shape: [u32; 3],
    /// Every check must be true or `napkin-bench` exits non-zero.
    pub checks: BTreeMap<String, bool>,
    /// Raw per-round samples keyed by metric field name.
    pub samples_ms: BTreeMap<String, Vec<f64>>,
    /// GPU 0 clocks, power, temperature and clock-limit reasons for each
    /// timed GEMM round; None in profiles captured before it was recorded.
    #[serde(default)]
    pub gpu_state: Option<GpuState>,
}

impl Measured {
    /// Slowest round over fastest round minus one, e.g. `0.15` when the
    /// rounds of `samples_ms["gemm_fp16"]` range from 1.42 to 1.63 ms. This
    /// is the observed range of a handful of rounds, not a confidence
    /// interval. None when the metric has fewer than two positive rounds.
    #[must_use]
    pub fn round_spread(&self, metric: &str) -> Option<f64> {
        let rounds = self.samples_ms.get(metric)?;
        if rounds.len() < 2 || rounds.iter().any(|ms| *ms <= 0.0) {
            return None;
        }
        let fastest = rounds.iter().copied().fold(f64::INFINITY, f64::min);
        let slowest = rounds.iter().copied().fold(0.0, f64::max);
        Some(slowest / fastest - 1.0)
    }
}

/// GPU state behind the GEMM numbers, read through NVML on the GPU that ran
/// them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuState {
    /// NVML and wrapper versions, e.g. `"NVML 12.580.95.05 via nvml-wrapper 0.13"`.
    pub source: String,
    /// Target time between NVML polls while a round runs.
    pub poll_interval_ms: f64,
    pub max_sm_clock_mhz: u32,
    pub max_mem_clock_mhz: u32,
    /// Enforced board power limit.
    pub power_limit_w: f64,
    /// Untimed GEMM work, at least this long, before the first timed round of
    /// each dtype.
    pub warmup_ms: f64,
    /// Every timed round runs enough GEMMs to last at least this long.
    pub round_ms: f64,
    /// One entry per timed round, keyed and ordered like `samples_ms`.
    pub rounds: BTreeMap<String, Vec<RoundState>>,
}

/// What NVML reported while one timed round ran.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundState {
    /// NVML polls that landed inside the round; the spans are None and
    /// `power_w` is None when a round was too short to catch any.
    pub polls: u32,
    /// Host wall-clock length of the timed round, so a reader can check that
    /// the round really lasted at least `GpuState::round_ms`.
    pub wall_ms: f64,
    pub sm_clock_mhz: Option<Span>,
    pub mem_clock_mhz: Option<Span>,
    /// Energy-counter delta over the round divided by the time it covers.
    /// NVML's power reading is a one-second average on recent GPUs, so it
    /// would smear neighbouring rounds together. None when the counter
    /// ticked fewer than two times inside the round.
    pub power_w: Option<f64>,
    pub temperature_c: Option<Span>,
    /// How many polls saw each clock-limit reason, in NVML's names
    /// lowercased, e.g. `{"sw_power_cap": 40}`. Empty when clocks were never
    /// limited.
    pub clock_limit_reasons: BTreeMap<String, u32>,
}

/// Smallest, median and largest of the polled values.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Span {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

/// Where a rate came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    Measured,
    Ceiling,
}

impl fmt::Display for Basis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Measured => "measured",
            Self::Ceiling => "datasheet",
        };
        f.write_str(name)
    }
}

/// One dtype's throughput with its provenance.
#[derive(Clone, Copy, Debug)]
pub struct ComputeRate {
    pub tflops: f64,
    pub basis: Basis,
}

/// Resolved rates for estimation; measured values win when available.
#[derive(Clone, Debug)]
pub struct Rates {
    pub compute_tflops: BTreeMap<Dtype, ComputeRate>,
    pub hbm_gbps: f64,
    pub h2d_gbps: Option<f64>,
    pub d2h_gbps: Option<f64>,
    pub host_copy_gbps: Option<f64>,
    pub g2g_gbps: f64,
    /// Fixed cost of one synchronous copy (see `Measured::h2d_latency_us`).
    /// `None` means unmeasured, never zero.
    pub h2d_latency_us: Option<f64>,
    pub d2h_latency_us: Option<f64>,
    pub g2g_latency_us: Option<f64>,
    pub launch_us: Option<f64>,
    pub basis: Basis,
}
