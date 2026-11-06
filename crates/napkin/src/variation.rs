//! How far headline numbers move between machines: the canonical measured
//! profile against its repeated runs in `profiles/repeats/<id>/`.
//!
//! sirupsen/napkin-math keeps a latency only if it holds within 15% on every
//! machine measured; `MACHINE_FLAG` applies the same rule here, as the
//! slowest run over the fastest, minus one.

use crate::profile::{Dtype, Measured, Profile};
use crate::units::{fmt_duration, fmt_gbps, fmt_sig, fmt_tflops};

/// Spread between runs above which a number is said to vary by machine.
pub const MACHINE_FLAG: f64 = 0.15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Nanoseconds,
    Microseconds,
    Gbps,
    Tflops,
}

impl Unit {
    /// The unit a raw metric value is in, for column headers.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Nanoseconds => "ns",
            Self::Microseconds => "μs",
            Self::Gbps => "GB/s",
            Self::Tflops => "TFLOP/s",
        }
    }

    /// `value` at `sig` significant digits with a readable unit, e.g.
    /// H100's 2980 GB/s HBM copy at one digit is `"3 TB/s"`.
    #[must_use]
    pub fn format(self, value: f64, sig: u32) -> String {
        match self {
            Self::Nanoseconds => format!("{} ns", fmt_sig(value, sig)),
            Self::Microseconds => fmt_duration(value / 1e6, sig),
            Self::Gbps => fmt_gbps(value, sig),
            Self::Tflops => fmt_tflops(value, sig),
        }
    }
}

/// One napkin-bench number every full run records.
#[derive(Clone, Copy, Debug)]
pub struct Metric {
    pub name: &'static str,
    pub unit: Unit,
    pub value: fn(&Measured) -> Option<f64>,
}

pub const KERNEL_LAUNCH: Metric = Metric {
    name: "Kernel launch",
    unit: Unit::Microseconds,
    value: |m| Some(m.launch_us),
};
pub const LAUNCH_SYNC: Metric = Metric {
    name: "Launch + sync round trip",
    unit: Unit::Microseconds,
    value: |m| m.sync_us,
};
pub const HBM_LATENCY: Metric = Metric {
    name: "HBM random read",
    unit: Unit::Nanoseconds,
    value: |m| m.hbm_latency_ns,
};
pub const HBM_COPY: Metric = Metric {
    name: "HBM copy",
    unit: Unit::Gbps,
    value: |m| Some(m.hbm_gbps),
};
pub const HOST_COPY: Metric = Metric {
    name: "Pinned host RAM copy",
    unit: Unit::Gbps,
    value: |m| Some(m.host_copy_gbps),
};
pub const H2D_LATENCY: Metric = Metric {
    name: "Host → device latency",
    unit: Unit::Microseconds,
    value: |m| m.h2d_latency_us,
};
pub const H2D: Metric = Metric {
    name: "Host → device",
    unit: Unit::Gbps,
    value: |m| Some(m.h2d_gbps),
};
pub const D2H_LATENCY: Metric = Metric {
    name: "Device → host latency",
    unit: Unit::Microseconds,
    value: |m| m.d2h_latency_us,
};
pub const D2H: Metric = Metric {
    name: "Device → host",
    unit: Unit::Gbps,
    value: |m| Some(m.d2h_gbps),
};
pub const G2G_LATENCY: Metric = Metric {
    name: "GPU ↔ GPU latency",
    unit: Unit::Microseconds,
    value: |m| m.g2g_latency_us,
};
pub const G2G: Metric = Metric {
    name: "GPU ↔ GPU, slower direction",
    unit: Unit::Gbps,
    value: |m| m.g2g_gbps,
};
pub const BF16_GEMM: Metric = Metric {
    name: "Dense BF16 GEMM, 8192³",
    unit: Unit::Tflops,
    value: |m| m.compute_tflops.get(&Dtype::Bf16).copied(),
};

/// Table order for `napkin variation`.
pub const METRICS: [Metric; 12] = [
    KERNEL_LAUNCH,
    LAUNCH_SYNC,
    HBM_LATENCY,
    HBM_COPY,
    HOST_COPY,
    H2D_LATENCY,
    H2D,
    D2H_LATENCY,
    D2H,
    G2G_LATENCY,
    G2G,
    BF16_GEMM,
];

/// The lowest and highest value of one metric over a profile's runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub runs: usize,
    pub low: f64,
    pub high: f64,
}

impl Spread {
    /// Highest over lowest, minus one: for a latency that is the slowest run
    /// over the fastest, for a throughput the fastest over the slowest.
    #[must_use]
    pub fn fraction(&self) -> f64 {
        self.high / self.low - 1.0
    }

    #[must_use]
    pub fn varies_by_machine(&self) -> bool {
        self.fraction() > MACHINE_FLAG
    }
}

/// The canonical run followed by every repeat.
fn runs(profile: &Profile) -> impl Iterator<Item = &Measured> {
    profile
        .measured
        .iter()
        .chain(profile.repeats.iter().map(|repeat| &repeat.measured))
}

/// The metric's spread over the canonical run and its repeats; `None` until
/// at least two runs recorded a positive value.
#[must_use]
pub fn spread(profile: &Profile, metric: &Metric) -> Option<Spread> {
    let values: Vec<f64> = runs(profile)
        .filter_map(metric.value)
        .filter(|value| *value > 0.0)
        .collect();
    if values.len() < 2 {
        return None;
    }
    Some(Spread {
        runs: values.len(),
        low: values.iter().copied().reduce(f64::min)?,
        high: values.iter().copied().reduce(f64::max)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{Host, Repeat, find};

    fn with_repeats(hbm: &[f64]) -> Profile {
        let mut profile = find("h100").cloned().unwrap();
        let canonical = profile.measured.clone().unwrap();
        profile.repeats = hbm
            .iter()
            .map(|gbps| {
                let mut measured = canonical.clone();
                measured.hbm_gbps = *gbps;
                Repeat {
                    host: Host {
                        cloud_provider: "CLOUD_PROVIDER_OCI".to_owned(),
                        region: "us-central".to_owned(),
                        task_id: "ta-test".to_owned(),
                        gpu_uuids: vec!["GPU-a".to_owned(), "GPU-b".to_owned()],
                    },
                    measured,
                }
            })
            .collect();
        profile
    }

    #[test]
    fn no_spread_without_a_repeat() {
        let mut profile = find("h100").cloned().unwrap();
        profile.repeats.clear();
        assert_eq!(spread(&profile, &HBM_COPY), None);
    }

    #[test]
    fn spread_spans_canonical_and_repeats() {
        let canonical = find("h100").unwrap().measured.as_ref().unwrap().hbm_gbps;
        let profile = with_repeats(&[canonical * 0.9, canonical * 1.08]);
        let wide = spread(&profile, &HBM_COPY).unwrap();
        assert_eq!(wide.runs, 3);
        assert!((wide.fraction() - (1.08 / 0.9 - 1.0)).abs() < 1e-12);
        assert!(wide.varies_by_machine());
        let profile = with_repeats(&[canonical * 0.95]);
        assert!(!spread(&profile, &HBM_COPY).unwrap().varies_by_machine());
    }
}
