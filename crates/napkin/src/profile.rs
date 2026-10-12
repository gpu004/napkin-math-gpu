//! GPU profiles: datasheet ceilings plus optional `napkin-bench` measurements.

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
    /// Per-GPU one-way `NVLink` aggregate as NVIDIA publishes it, GB/s.
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
    /// 4-byte synchronous copies, wall clock median.
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
    /// 4-byte peer copy, synchronous, wall clock median.
    pub g2g_latency_us: Option<f64>,
    /// cuBLAS `GemmEx` 8192^3 (fp16, bf16, tf32, fp32).
    pub compute_tflops: BTreeMap<Dtype, f64>,
    pub gemm_shape: [u32; 3],
    /// Every check must be true or `napkin-bench` exits non-zero.
    pub checks: BTreeMap<String, bool>,
    /// Raw per-round samples keyed by metric field name.
    pub samples_ms: BTreeMap<String, Vec<f64>>,
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
    pub launch_us: Option<f64>,
    pub basis: Basis,
}

/// Registry parse failure.
#[derive(Debug)]
pub struct ProfileError(pub String);

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProfileError {}

const DATASHEET_SRC: &[(&str, &str)] = &[
    ("a100", include_str!("../../../profiles/a100.json")),
    ("h100", include_str!("../../../profiles/h100.json")),
    ("b200", include_str!("../../../profiles/b200.json")),
    ("b300", include_str!("../../../profiles/b300.json")),
];

include!(concat!(env!("OUT_DIR"), "/measured.rs"));

/// Parses every embedded datasheet profile and attaches any embedded
/// `<id>.measured.json`. Ordered a100, h100, b200, b300.
///
/// # Errors
/// Returns `ProfileError` if an embedded JSON file fails to parse.
pub fn builtin() -> Result<Vec<Profile>, ProfileError> {
    let mut profiles = Vec::with_capacity(DATASHEET_SRC.len());
    for (id, src) in DATASHEET_SRC {
        let mut profile: Profile =
            serde_json::from_str(src).map_err(|e| ProfileError(format!("profile {id}: {e}")))?;
        if let Some((_, measured_src)) = MEASURED_SRC.iter().find(|(mid, _)| mid == id) {
            profile.measured = Some(
                serde_json::from_str(measured_src)
                    .map_err(|e| ProfileError(format!("measured {id}: {e}")))?,
            );
        }
        profiles.push(profile);
    }
    Ok(profiles)
}

/// Looks up one profile by id.
///
/// # Errors
/// Returns `ProfileError` if an embedded JSON file fails to parse.
pub fn find(id: &str) -> Result<Option<Profile>, ProfileError> {
    for (pid, src) in DATASHEET_SRC {
        if *pid == id {
            let mut profile: Profile = serde_json::from_str(src)
                .map_err(|e| ProfileError(format!("profile {id}: {e}")))?;
            if let Some((_, measured_src)) = MEASURED_SRC.iter().find(|(mid, _)| mid == &id) {
                profile.measured = Some(
                    serde_json::from_str(measured_src)
                        .map_err(|e| ProfileError(format!("measured {id}: {e}")))?,
                );
            }
            return Ok(Some(profile));
        }
    }
    Ok(None)
}

impl Profile {
    /// Resolves the rates used for estimation: measured values when a measured
    /// profile is attached, else datasheet ceilings. `force_ceiling` implements
    /// `--ceiling`. GPU-to-GPU falls back to the `NVLink` ceiling when unmeasured;
    /// host-link and launch rates have no datasheet fallback.
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
            launch_us: None,
            basis: Basis::Ceiling,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_parses_every_embedded_profile() {
        let profiles = builtin().unwrap();
        assert_eq!(
            profiles.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
            vec!["a100", "h100", "b200", "b300"]
        );
        for p in &profiles {
            assert!(p.ceiling.hbm_gbps > 0.0);
            assert!(p.ceiling.nvlink_gbps > 0.0);
            assert!(!p.ceiling.compute_tflops.is_empty());
            assert_ne!(p.ceiling.source, "");
        }
    }

    #[test]
    fn find_returns_none_for_unknown_id() {
        assert!(find("t4").unwrap().is_none());
        assert!(find("b200").unwrap().is_some());
    }

    #[test]
    fn ceiling_basis_has_no_host_or_launch_rates() {
        let b200 = find("b200").unwrap().unwrap();
        let rates = b200.rates(false);
        assert_eq!(rates.basis, Basis::Ceiling);
        assert!(rates.h2d_gbps.is_none());
        assert!(rates.launch_us.is_none());
        assert!((rates.g2g_gbps - b200.ceiling.nvlink_gbps).abs() < f64::EPSILON);
    }
}
