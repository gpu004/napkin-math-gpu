//! Estimation from measured NCCL Tests collective curves.

use std::fmt;

use napkin_profile::Dtype;
use napkin_profile::collective::{Collective, CollectiveEvidence, Curve, CurvePoint, Topology};

use crate::profile;

#[derive(Clone, Copy, Debug)]
pub struct Query<'a> {
    pub gpu: &'a str,
    pub gpu_count: u32,
    pub topology: Topology,
    pub collective: Collective,
    pub dtype: Dtype,
    pub bytes: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    Latency,
    Bandwidth,
}

impl fmt::Display for Regime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Latency => "latency",
            Self::Bandwidth => "bandwidth",
        })
    }
}

#[derive(Clone, Debug)]
pub struct CollectiveEstimate<'a> {
    pub evidence: &'a CollectiveEvidence,
    pub curve: &'a Curve,
    pub lower: &'a CurvePoint,
    pub upper: &'a CurvePoint,
    pub time_s: f64,
    pub algbw_gbps: f64,
    pub busbw_gbps: f64,
    pub floor_us: f64,
    pub peak: &'a CurvePoint,
    pub regime: Regime,
}

fn matches_query(evidence: &CollectiveEvidence, query: &Query<'_>) -> bool {
    evidence.gpu == query.gpu
        && evidence.gpu_count == query.gpu_count
        && evidence.topology == query.topology
        && evidence.dtype == query.dtype
}

fn measured_combinations(evidence: &[CollectiveEvidence]) -> String {
    if evidence.is_empty() {
        return "none".to_owned();
    }
    evidence
        .iter()
        .map(|item| {
            format!(
                "{} × {} {} {}",
                item.gpu, item.gpu_count, item.topology, item.dtype
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn bytes_as_f64(bytes: u64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "NCCL curve sizes are exactly representable through 2^53 bytes"
    )]
    {
        bytes as f64
    }
}

/// Estimates a collective using the supplied evidence set.
///
/// # Errors
/// Returns a useful rejection when query quantities or exact measured evidence
/// do not support the requested estimate; curves are never extrapolated.
pub fn estimate_with<'a>(
    evidence: &'a [CollectiveEvidence],
    query: &Query<'_>,
) -> Result<CollectiveEstimate<'a>, String> {
    if !query.bytes.is_finite() || query.bytes <= 0.0 {
        return Err("collective bytes must be finite and greater than zero".to_owned());
    }
    let item = evidence
        .iter()
        .find(|candidate| matches_query(candidate, query))
        .ok_or_else(|| {
            format!(
                "no NCCL Tests evidence for {} × {} over {} in {} (measured: {}); napkin does not scale collective curves across GPU counts or topologies",
                query.gpu,
                query.gpu_count,
                query.topology,
                query.dtype,
                measured_combinations(evidence)
            )
        })?;
    let curve = item.curve(query.collective).ok_or_else(|| {
        format!(
            "NCCL evidence for {} × {} over {} in {} has no {} curve",
            query.gpu, query.gpu_count, query.topology, query.dtype, query.collective
        )
    })?;
    let first = curve
        .points
        .first()
        .ok_or_else(|| format!("{} curve is empty", query.collective))?;
    let last = curve
        .points
        .last()
        .ok_or_else(|| format!("{} curve is empty", query.collective))?;
    if query.bytes > bytes_as_f64(last.bytes) {
        return Err(format!(
            "requested {} bytes exceeds the largest measured size of {} bytes; napkin does not extrapolate collective curves",
            query.bytes, last.bytes
        ));
    }

    let (lower, upper, time_us) = if query.bytes <= bytes_as_f64(first.bytes) {
        (first, first, first.time_us)
    } else if let Some(point) = curve
        .points
        .iter()
        .find(|point| query.bytes.to_bits() == bytes_as_f64(point.bytes).to_bits())
    {
        (point, point, point.time_us)
    } else {
        let (lower, upper) = curve
            .points
            .windows(2)
            .find(|pair| {
                query.bytes > bytes_as_f64(pair[0].bytes)
                    && query.bytes < bytes_as_f64(pair[1].bytes)
            })
            .map(|pair| (&pair[0], &pair[1]))
            .ok_or_else(|| "could not bracket collective size in measured curve".to_owned())?;
        let fraction = (query.bytes - bytes_as_f64(lower.bytes))
            / (bytes_as_f64(upper.bytes) - bytes_as_f64(lower.bytes));
        (
            lower,
            upper,
            lower.time_us + fraction * (upper.time_us - lower.time_us),
        )
    };

    let seconds = time_us / 1e6;
    let algbw_gbps = query.bytes / seconds / 1e9;
    let busbw_gbps = algbw_gbps * query.collective.bus_factor(query.gpu_count);
    let floor_us = curve
        .points
        .iter()
        .map(|point| point.time_us)
        .fold(f64::INFINITY, f64::min);
    let peak = curve
        .points
        .iter()
        .max_by(|left, right| left.busbw_gbps.total_cmp(&right.busbw_gbps))
        .ok_or_else(|| format!("{} curve is empty", query.collective))?;
    // Fixed per-call cost is at least half of the time, so bandwidth barely matters.
    let regime = if time_us < 2.0 * floor_us {
        Regime::Latency
    } else {
        Regime::Bandwidth
    };
    Ok(CollectiveEstimate {
        evidence: item,
        curve,
        lower,
        upper,
        time_s: seconds,
        algbw_gbps,
        busbw_gbps,
        floor_us,
        peak,
        regime,
    })
}

/// Estimates a collective from the embedded profiles' NCCL Tests evidence.
///
/// # Errors
/// Returns a useful rejection when query quantities or exact measured evidence
/// do not support the requested estimate; curves are never extrapolated.
pub fn estimate(query: &Query<'_>) -> Result<CollectiveEstimate<'static>, String> {
    estimate_with(profile::collectives(), query)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUN: &str = include_str!("../../napkin-profile/fixtures/nccl-tests/h100x2/run.json");
    const REDUCE: &str =
        include_str!("../../napkin-profile/fixtures/nccl-tests/h100x2/all_reduce_perf.json");
    const GATHER: &str =
        include_str!("../../napkin-profile/fixtures/nccl-tests/h100x2/all_gather_perf.json");
    const SCATTER: &str =
        include_str!("../../napkin-profile/fixtures/nccl-tests/h100x2/reduce_scatter_perf.json");

    fn evidence() -> Result<CollectiveEvidence, String> {
        napkin_profile::collective::parse_evidence(
            "profiles/nccl-tests/h100x2",
            RUN,
            [REDUCE, GATHER, SCATTER],
        )
    }

    fn query(collective: Collective, bytes: f64) -> Query<'static> {
        Query {
            gpu: "h100",
            gpu_count: 2,
            topology: Topology::Nvlink,
            collective,
            dtype: Dtype::Bf16,
            bytes,
        }
    }

    #[test]
    fn estimates_exact_all_reduce_point() -> Result<(), String> {
        let data = [evidence()?];
        let estimate = estimate_with(&data, &query(Collective::AllReduce, 32_768.0))?;
        // The measured 32 KiB row is 10.0802 μs, or 32,768 / (10.0802 × 1,000) GB/s.
        assert!((estimate.time_s - 10.0802e-6).abs() < 1e-12);
        assert!((estimate.algbw_gbps - 3.250_729).abs() < 1e-6);
        assert_eq!(estimate.lower.bytes, 32_768);
        assert_eq!(estimate.upper.bytes, 32_768);
        Ok(())
    }

    #[test]
    fn interpolates_all_gather_between_measured_points() -> Result<(), String> {
        let data = [evidence()?];
        let estimate = estimate_with(&data, &query(Collective::AllGather, 49_152.0))?;
        // 32 KiB and 64 KiB midpoint: (10.3996 + 10.8848) / 2 = 10.6422 μs.
        assert!((estimate.time_s - 10.6422e-6).abs() < 1e-12);
        assert_eq!(estimate.lower.bytes, 32_768);
        assert_eq!(estimate.upper.bytes, 65_536);
        Ok(())
    }

    #[test]
    fn uses_reduce_scatter_latency_floor_and_four_gib_peak() -> Result<(), String> {
        let data = [evidence()?];
        // 16 B is below the 32 B first row, so it inherits that row's 12.73735 μs.
        let small = estimate_with(&data, &query(Collective::ReduceScatter, 16.0))?;
        assert!((small.time_s - 12.73735e-6).abs() < 1e-12);
        assert_eq!(small.regime, Regime::Latency);

        // At 4 GiB the captured bus bandwidth is 527.994545 × (2-1)/2.
        let large = estimate_with(&data, &query(Collective::ReduceScatter, 4_294_967_296.0))?;
        assert!((large.time_s - 8_134.491_8e-6).abs() < 1e-9);
        assert!((large.busbw_gbps - 263.997_272).abs() < 1e-6);
        assert_eq!(large.peak.bytes, 4_294_967_296);
        assert_eq!(large.regime, Regime::Bandwidth);
        Ok(())
    }

    #[test]
    fn rejects_unmeasured_inputs_and_extrapolation() -> Result<(), String> {
        let data = [evidence()?];
        for q in [
            Query {
                gpu_count: 4,
                ..query(Collective::AllReduce, 16.0)
            },
            Query {
                topology: Topology::Pcie,
                ..query(Collective::AllReduce, 16.0)
            },
            Query {
                dtype: Dtype::Fp16,
                ..query(Collective::AllReduce, 16.0)
            },
            query(Collective::AllReduce, 4_294_967_297.0),
            query(Collective::AllReduce, 0.0),
            query(Collective::AllReduce, f64::NAN),
        ] {
            assert!(estimate_with(&data, &q).is_err());
        }
        Ok(())
    }

    #[test]
    fn embedded_h100_two_gpu_curves_are_available() {
        assert!(crate::profile::collectives().iter().any(|item| {
            item.gpu == "h100"
                && item.gpu_count == 2
                && item.topology == Topology::Nvlink
                && item.dtype == Dtype::Bf16
                && Collective::ALL
                    .iter()
                    .all(|collective| item.curve(*collective).is_some())
        }));
    }
}
