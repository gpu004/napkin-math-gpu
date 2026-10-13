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
    /// GPU's maximum `PCIe` generation (`pcie.link.gen.gpumax`).
    #[serde(default)]
    pub pcie_gen_max: Option<u8>,
    /// Host-side maximum `PCIe` generation (`pcie.link.gen.hostmax`), when
    /// reported. When lower than `pcie_gen`, this bounds host↔device bandwidth;
    /// for example, HGX B300 links GPUs at Gen6 x16 to a switch, then Gen5 to
    /// the host.
    #[serde(default)]
    pub pcie_gen_host_max: Option<u8>,
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
    /// Raw per-round durations in milliseconds, keyed by metric field name.
    pub samples_ms: BTreeMap<String, Vec<f64>>,
    /// GPU 0 clocks, power, temperature and clock-limit reasons for each
    /// timed GEMM round; None in profiles captured before it was recorded.
    #[serde(default)]
    pub gpu_state: Option<GpuState>,
    /// cuBLASLt shape results and their independent run provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemm_shapes: Option<GemmShapeSuite>,
}

/// cuBLASLt shape measurements from a run that may differ from the scalar
/// measurements stored in [`Measured`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmShapeSuite {
    /// Time this shape suite was captured.
    pub captured_at: String,
    /// Requested Modal GPU and instance count for this run.
    pub gpu_request: String,
    /// CUDA device name reported by the run.
    pub device_name: String,
    /// NVIDIA driver version reported by the run.
    pub driver_version: String,
    /// CUDA runtime version used by the run.
    pub cuda_version: String,
    /// cuBLASLt library version used for all recorded shapes.
    pub library: String,
    /// `PCIe` generation observed for this run, when reported.
    #[serde(default)]
    pub pcie_gen: Option<u8>,
    /// `PCIe` link width observed for this run, when reported.
    #[serde(default)]
    pub pcie_width: Option<u8>,
    /// Ordered per-shape measurements.
    pub shapes: Vec<GemmShape>,
    /// Correctness results from the shape-suite run.
    pub checks: BTreeMap<String, bool>,
    /// Raw per-round shape timings in milliseconds.
    pub samples_ms: BTreeMap<String, Vec<f64>>,
    /// NVML GPU state observed during the shape-suite rounds.
    pub gpu_state: Option<GpuState>,
}

impl GemmShapeSuite {
    /// Slowest shape round over fastest round minus one.
    #[must_use]
    pub fn round_spread(&self, metric: &str) -> Option<f64> {
        round_spread(&self.samples_ms, metric)
    }
}

/// One cuBLASLt GEMM from the shape suite, D (M × N) = op(A) (M × K) ·
/// op(B) (K × N), column-major, alpha 1 beta 0.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmShape {
    /// Suite label, e.g. `"decode-m8"`.
    pub name: String,
    /// Output rows M, output columns N, and reduction dimension K.
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Element type of A, B, C, and D.
    pub dtype: Dtype,
    /// cuBLASLt compute type, e.g. `"CUBLAS_COMPUTE_32F"`.
    pub compute_type: String,
    /// cuBLASLt scale type, e.g. `"CUDA_R_32F"`.
    pub scale_type: String,
    /// Operation applied to A, e.g. `"T"` or `"N"`.
    pub trans_a: String,
    /// Operation applied to B, e.g. `"T"` or `"N"`.
    pub trans_b: String,
    /// Leading dimensions of A, B, and C/D in elements.
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    /// cuBLASLt epilogue, e.g. `"CUBLASLT_EPILOGUE_DEFAULT"`.
    pub epilogue: String,
    /// Algorithm attributes selected by the heuristic.
    pub algorithm: GemmAlgo,
    /// Workspace limit passed to the heuristic, in bytes.
    pub workspace_limit_bytes: u64,
    /// Decimal TFLOP/s: `flops()` divided by median round time in milliseconds.
    pub tflops: f64,
    /// Sampled-output comparison against a float64 CPU reference computed
    /// from the dtype-rounded inputs.
    pub check: GemmCheck,
}

impl GemmShape {
    /// Count of multiply-add FLOPs, with each multiply and add counted once.
    #[must_use]
    pub fn flops(&self) -> f64 {
        2.0 * f64::from(self.m) * f64::from(self.n) * f64::from(self.k)
    }

    /// Per-round timing key used in `Measured::samples_ms`.
    #[must_use]
    pub fn samples_key(&self) -> String {
        format!("gemm_shape_{}_{}", self.name, self.dtype)
    }
}

/// Raw enum values and workspace/waves returned by the cuBLASLt heuristic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmAlgo {
    /// cuBLASLt algorithm ID.
    pub id: i32,
    /// cuBLASLt tile configuration ID.
    pub tile: i32,
    /// Kernel pipeline stage configuration.
    pub stages: i32,
    /// Split-K factor.
    pub split_k: i32,
    /// Reduction scheme enum value.
    pub reduction_scheme: i32,
    /// CTA swizzling enum value.
    pub cta_swizzling: i32,
    /// Custom algorithm option enum value.
    pub custom_option: i32,
    /// Heuristic-selected workspace size, in bytes.
    pub workspace_bytes: u64,
    /// Heuristic's estimated number of waves.
    pub waves: f64,
}

/// Output comparison against a float64 reference, using dtype-rounded inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemmCheck {
    /// Number of deterministically sampled output elements.
    pub sampled: u32,
    /// Largest absolute error among sampled outputs.
    pub max_abs_error: f64,
    /// Relative tolerance used by the pass criterion.
    pub rtol: f64,
    /// Absolute tolerance used by the pass criterion.
    pub atol: f64,
    /// Whether every sampled output met `atol + rtol × |reference|`.
    pub passed: bool,
}

/// Decimal TFLOP/s for `flops` work completed in `ms` milliseconds.
#[must_use]
pub fn tflops(flops: f64, ms: f64) -> f64 {
    flops / (ms / 1e3) / 1e12
}

impl Measured {
    /// Slowest round over fastest round minus one, e.g. `0.15` when the
    /// rounds of `samples_ms["gemm_fp16"]` range from 1.42 to 1.63 ms. This
    /// is the observed range of a handful of rounds, not a confidence
    /// interval. None when the metric has fewer than two positive rounds.
    #[must_use]
    pub fn round_spread(&self, metric: &str) -> Option<f64> {
        round_spread(&self.samples_ms, metric)
    }
}

fn round_spread(samples_ms: &BTreeMap<String, Vec<f64>>, metric: &str) -> Option<f64> {
    let rounds = samples_ms.get(metric)?;
    if rounds.len() < 2 || rounds.iter().any(|ms| *ms <= 0.0) {
        return None;
    }
    let fastest = rounds.iter().copied().fold(f64::INFINITY, f64::min);
    let slowest = rounds.iter().copied().fold(0.0, f64::max);
    Some(slowest / fastest - 1.0)
}

impl Measured {
    /// Copies a successful run's shape suite without changing this profile's
    /// scalar measurements, samples, checks, or GPU state.
    ///
    /// # Errors
    /// Returns an error if the run has no shape suite, its hardware identity
    /// differs, or any run or suite check failed.
    pub fn merge_gemm_shapes(&mut self, run: Measured) -> Result<(), String> {
        let suite = run
            .gemm_shapes
            .as_ref()
            .ok_or_else(|| "run has no GEMM shape suite".to_owned())?;
        if self.device_name != run.device_name {
            return Err(format!(
                "device name differs: profile {:?}, run {:?}",
                self.device_name, run.device_name
            ));
        }
        if self.compute_capability != run.compute_capability {
            return Err(format!(
                "compute capability differs: profile {:?}, run {:?}",
                self.compute_capability, run.compute_capability
            ));
        }
        if self.sm_count != run.sm_count {
            return Err(format!(
                "SM count differs: profile {}, run {}",
                self.sm_count, run.sm_count
            ));
        }
        if suite.device_name != run.device_name {
            return Err(format!(
                "device name differs: run {:?}, shape suite {:?}",
                run.device_name, suite.device_name
            ));
        }
        if let Some((name, _)) = run.checks.iter().find(|(_, passed)| !**passed) {
            return Err(format!("run has failing check {name:?}"));
        }
        if let Some((name, _)) = suite.checks.iter().find(|(_, passed)| !**passed) {
            return Err(format!("shape suite has failing check {name:?}"));
        }
        self.gemm_shapes = run.gemm_shapes;
        Ok(())
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
    /// Target length of a timed round: the GEMM count per round is sized from
    /// the last warm-up batch, so real rounds land near it, not above it.
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
    /// the round came close to `GpuState::round_ms`.
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

impl Measured {
    /// Returns the `PCIe` generation available for host↔device transfers.
    #[must_use]
    pub fn pcie_effective_gen(&self) -> Option<u8> {
        self.pcie_gen.map(|generation| {
            self.pcie_gen_host_max
                .map_or(generation, |host_max| generation.min(host_max))
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_tflops_uses_milliseconds() {
        assert_eq!(tflops(1e12, 2.0), 500.0);
    }

    fn gpu_state(source: &str) -> GpuState {
        GpuState {
            source: source.to_owned(),
            poll_interval_ms: 100.0,
            max_sm_clock_mhz: 1000,
            max_mem_clock_mhz: 1000,
            power_limit_w: 300.0,
            warmup_ms: 100.0,
            round_ms: 100.0,
            rounds: BTreeMap::new(),
        }
    }

    fn measured(device_name: &str, memory_bytes: u64) -> Measured {
        Measured {
            captured_at: "2026-10-14T00:00:00Z".to_owned(),
            gpu_request: "GPU".to_owned(),
            device_name: device_name.to_owned(),
            device_count: 1,
            compute_capability: "9.0".to_owned(),
            sm_count: 1,
            memory_bytes,
            driver_version: "driver".to_owned(),
            cuda_version: "cuda".to_owned(),
            rounds: 3,
            launch_us: 1.0,
            sync_us: Some(1.0),
            host_copy_gbps: 2.0,
            h2d_gbps: 3.0,
            d2h_gbps: 4.0,
            pcie_gen: Some(5),
            pcie_gen_max: Some(5),
            pcie_gen_host_max: None,
            pcie_width: Some(16),
            pcie_width_max: Some(16),
            h2d_latency_us: Some(1.0),
            d2h_latency_us: Some(1.0),
            hbm_gbps: 5.0,
            elementwise_gbps: 6.0,
            hbm_latency_ns: Some(1.0),
            g2g_gbps: Some(7.0),
            g2g_forward_gbps: Some(7.0),
            g2g_reverse_gbps: Some(7.0),
            g2g_latency_us: Some(1.0),
            compute_tflops: BTreeMap::from([(Dtype::Fp16, 100.0)]),
            gemm_shape: [1, 1, 1],
            gemm_shapes: None,
            checks: BTreeMap::from([("scalar_check".to_owned(), true)]),
            samples_ms: BTreeMap::from([("scalar".to_owned(), vec![1.0, 1.1])]),
            gpu_state: Some(gpu_state("scalar")),
        }
    }

    fn shape_suite() -> GemmShapeSuite {
        GemmShapeSuite {
            captured_at: "2026-10-15T00:00:00Z".to_owned(),
            gpu_request: "GPU".to_owned(),
            device_name: "GPU".to_owned(),
            driver_version: "driver".to_owned(),
            cuda_version: "cuda".to_owned(),
            library: "cuBLASLt".to_owned(),
            pcie_gen: Some(5),
            pcie_width: Some(16),
            shapes: Vec::new(),
            checks: BTreeMap::from([("shape_check".to_owned(), true)]),
            samples_ms: BTreeMap::from([("shape".to_owned(), vec![1.0, 1.2])]),
            gpu_state: Some(gpu_state("shape")),
        }
    }

    #[test]
    fn shape_suite_spread_uses_its_own_round_samples() {
        assert!(
            shape_suite()
                .round_spread("shape")
                .is_some_and(|spread| (spread - 0.2).abs() < 1e-12)
        );
    }

    #[test]
    fn merge_copies_only_a_successful_shape_suite() {
        let mut profile = measured("GPU", 1024);
        let original = profile.clone();
        let mut run = measured("GPU", 1024);
        run.captured_at = "run".to_owned();
        run.samples_ms = BTreeMap::from([("run_metric".to_owned(), vec![2.0])]);
        run.gemm_shapes = Some(shape_suite());
        let expected_suite = format!("{:?}", run.gemm_shapes);

        assert!(profile.merge_gemm_shapes(run).is_ok());

        assert_eq!(format!("{:?}", profile.gemm_shapes), expected_suite);
        let mut profile_without_suite = profile;
        profile_without_suite.gemm_shapes = None;
        let mut original_without_suite = original;
        original_without_suite.gemm_shapes = None;
        assert_eq!(
            format!("{profile_without_suite:?}"),
            format!("{original_without_suite:?}")
        );
    }

    #[test]
    fn merge_rejects_mismatched_or_failed_runs() {
        let mut profile = measured("GPU", 1024);

        let mut wrong_device = measured("Other GPU", 1024);
        wrong_device.gemm_shapes = Some(shape_suite());
        assert!(
            profile
                .clone()
                .merge_gemm_shapes(wrong_device)
                .err()
                .is_some_and(|error| error.contains("device name differs"))
        );

        let mut wrong_compute_capability = measured("GPU", 1024);
        wrong_compute_capability.compute_capability = "8.0".to_owned();
        wrong_compute_capability.gemm_shapes = Some(shape_suite());
        assert!(
            profile
                .clone()
                .merge_gemm_shapes(wrong_compute_capability)
                .err()
                .is_some_and(|error| error.contains("compute capability differs"))
        );

        let mut wrong_sm_count = measured("GPU", 1024);
        wrong_sm_count.sm_count = 2;
        wrong_sm_count.gemm_shapes = Some(shape_suite());
        assert!(
            profile
                .clone()
                .merge_gemm_shapes(wrong_sm_count)
                .err()
                .is_some_and(|error| error.contains("SM count differs"))
        );

        let mut failed_run = measured("GPU", 1024);
        failed_run.gemm_shapes = Some(shape_suite());
        failed_run.checks.insert("failed".to_owned(), false);
        assert!(
            profile
                .clone()
                .merge_gemm_shapes(failed_run)
                .err()
                .is_some_and(|error| error.contains("failing check"))
        );

        let mut failed_suite = measured("GPU", 1024);
        let mut suite = shape_suite();
        suite.checks.insert("failed_shape".to_owned(), false);
        failed_suite.gemm_shapes = Some(suite);
        assert!(
            profile
                .clone()
                .merge_gemm_shapes(failed_suite)
                .err()
                .is_some_and(|error| error.contains("failing check"))
        );

        assert!(
            profile
                .merge_gemm_shapes(measured("GPU", 1024))
                .err()
                .is_some_and(|error| error.contains("no GEMM shape suite"))
        );
    }
}
