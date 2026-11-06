//! The memorization table that opens the README: one latency and one
//! throughput per row at one significant digit, like the table in
//! sirupsen/napkin-math. The detailed `numbers` tables carry the per-row
//! basis, round spread, and per-size times.

use std::fmt::Write as _;

use napkin_profile::collective::{Collective, CollectiveEvidence, Topology};

use super::{SIG, emit, transfer_basis};
use crate::profile::{Dtype, Measured, Profile, Rates};
use crate::units::{fmt_duration, fmt_gbps, fmt_tflops};
use crate::variation::{self, Metric, Unit};

/// Renders the summary for `profile`, or `None` without a measurement.
#[must_use]
pub fn summary(profile: &Profile, collectives: &[CollectiveEvidence]) -> Option<String> {
    let m = profile.measured.as_ref()?;
    let rates = profile.rates(false);
    let all_reduce = largest_nvlink_all_reduce(profile, collectives);
    let rows = rows(profile, m, &rates, all_reduce);
    let mut out = String::new();
    emit(&mut out, &["Operation", "Latency", "Throughput"]);
    emit(&mut out, &["---", "---:", "---:"]);
    for row in &rows {
        emit(
            &mut out,
            &[
                &row.name,
                &render(row.latency.as_ref()),
                &render(row.throughput.as_ref()),
            ],
        );
    }
    let collective_basis = all_reduce.map_or_else(String::new, |all_reduce| {
        format!(
            " The AllReduce row is the fastest call and the highest bus bandwidth on the NCCL \
             Tests curve ({}, {}).",
            all_reduce.evidence.modal_request,
            all_reduce
                .evidence
                .captured_at
                .get(..10)
                .unwrap_or(&all_reduce.evidence.captured_at),
        )
    });
    let _ = writeln!(
        out,
        "\n{}, one significant digit. Host ↔ device is the slower direction. Host and NVLink \
         throughput come from {}, the rest from napkin-bench's median round.{collective_basis}",
        profile.name,
        transfer_basis(&[rates.h2d, rates.d2h, Some(rates.g2g)], profile),
    );
    let varies_by_machine = rows
        .iter()
        .flat_map(|row| [&row.latency, &row.throughput])
        .flatten()
        .any(|cell| cell.varies_by_machine);
    if varies_by_machine {
        let _ = writeln!(
            out,
            "\n‡ napkin-bench's number for this cell differs by more than {:.0}% between runs \
             on different machines; `napkin variation` has the range.",
            variation::MACHINE_FLAG * 100.0
        );
    }
    Some(out)
}

/// One rounded summary value, and whether napkin-bench's number behind it
/// moves more than `variation::MACHINE_FLAG` between the canonical run and
/// its repeats on other machines.
struct Cell {
    text: String,
    varies_by_machine: bool,
}

impl Cell {
    /// A value with no repeated run behind it, such as an NCCL Tests curve.
    fn unrepeated(text: String) -> Self {
        Self {
            text,
            varies_by_machine: false,
        }
    }

    /// `text`, judged by the spread of every metric it summarizes: the host
    /// row's latency is the slower of H2D and D2H, so it varies if either
    /// direction does.
    fn judged(profile: &Profile, metrics: &[Metric], text: String) -> Self {
        let varies_by_machine = metrics.iter().any(|metric| {
            variation::spread(profile, metric).is_some_and(|spread| spread.varies_by_machine())
        });
        Self {
            text,
            varies_by_machine,
        }
    }
}

/// A ‡ after a value that varies by machine, as † follows one whose rounds
/// disagree; an empty cell for a row with no value in this column.
fn render(cell: Option<&Cell>) -> String {
    match cell {
        Some(Cell {
            text,
            varies_by_machine: true,
        }) => format!("{text} ‡"),
        Some(Cell { text, .. }) => text.clone(),
        None => String::new(),
    }
}

struct Row {
    name: String,
    latency: Option<Cell>,
    throughput: Option<Cell>,
}

/// The summary's rows, skipping any with no evidence in either column.
fn rows(
    profile: &Profile,
    m: &Measured,
    rates: &Rates,
    all_reduce: Option<AllReduceSummary<'_>>,
) -> Vec<Row> {
    let measured = |metric: Metric| {
        (metric.value)(m)
            .map(|value| Cell::judged(profile, &[metric], metric.unit.format(value, SIG)))
    };
    let row = |name: &str, latency, throughput| Row {
        name: name.to_owned(),
        latency,
        throughput,
    };
    let host_latencies = [variation::H2D_LATENCY, variation::D2H_LATENCY];
    let host_latency = host_latencies
        .iter()
        .filter_map(|metric| (metric.value)(m))
        .reduce(f64::max)
        .map(|us| Cell::judged(profile, &host_latencies, Unit::Microseconds.format(us, SIG)));
    let host_throughput = rates
        .h2d
        .into_iter()
        .chain(rates.d2h)
        .map(|rate| rate.gbps)
        .reduce(f64::min)
        .map(|gbps| {
            Cell::judged(
                profile,
                &[variation::H2D, variation::D2H],
                fmt_gbps(gbps, SIG),
            )
        });
    let nvlink_throughput = Cell::judged(profile, &[variation::G2G], fmt_gbps(rates.g2g.gbps, SIG));
    let mut rows = vec![
        row("Kernel launch", measured(variation::KERNEL_LAUNCH), None),
        row(
            "Launch + sync round trip",
            measured(variation::LAUNCH_SYNC),
            None,
        ),
        row("HBM random read", measured(variation::HBM_LATENCY), None),
        row(
            "HBM copy, read + write counted",
            None,
            measured(variation::HBM_COPY),
        ),
        row("Host ↔ device, pinned", host_latency, host_throughput),
        row(
            "GPU ↔ GPU, NVLink one-way",
            measured(variation::G2G_LATENCY),
            Some(nvlink_throughput),
        ),
    ];
    if let Some(all_reduce) = all_reduce {
        rows.push(row(
            &format!(
                "AllReduce, {} GPUs on NVLink, bus bandwidth",
                all_reduce.evidence.gpu_count
            ),
            Some(Cell::unrepeated(fmt_duration(
                all_reduce.floor_us / 1e6,
                SIG,
            ))),
            Some(Cell::unrepeated(fmt_gbps(all_reduce.peak_busbw_gbps, SIG))),
        ));
    }
    rows.push(row(
        "Dense BF16 GEMM, 8192³",
        None,
        measured(variation::BF16_GEMM),
    ));
    let decode = m.gemm_shapes.as_ref().and_then(|suite| {
        suite
            .shapes
            .iter()
            .find(|shape| shape.name == "decode-m8" && shape.dtype == Dtype::Bf16)
    });
    if let Some(shape) = decode {
        rows.push(row(
            &format!(
                "BF16 GEMM, {}×{}×{} (decode, {} tokens)",
                shape.m, shape.n, shape.k, shape.m
            ),
            None,
            Some(Cell::unrepeated(fmt_tflops(shape.tflops, SIG))),
        ));
    }
    rows.retain(|row| row.latency.is_some() || row.throughput.is_some());
    rows
}

/// The summary's `AllReduce` row: the fastest call is the per-call floor that
/// small messages, like a decode step's activations, pay; the highest bus
/// bandwidth is what large messages approach.
#[derive(Clone, Copy)]
struct AllReduceSummary<'a> {
    evidence: &'a CollectiveEvidence,
    floor_us: f64,
    peak_busbw_gbps: f64,
}

/// The `NVLink` bf16 `AllReduce` curve with the most GPUs for `profile`.
fn largest_nvlink_all_reduce<'a>(
    profile: &Profile,
    collectives: &'a [CollectiveEvidence],
) -> Option<AllReduceSummary<'a>> {
    collectives
        .iter()
        .filter(|evidence| {
            evidence.gpu == profile.id
                && evidence.topology == Topology::Nvlink
                && evidence.dtype == Dtype::Bf16
        })
        .filter_map(|evidence| {
            let curve = evidence.curve(Collective::AllReduce)?;
            let floor = curve
                .points
                .iter()
                .map(|point| point.time_us)
                .reduce(f64::min)?;
            let peak = curve
                .points
                .iter()
                .map(|point| point.busbw_gbps)
                .reduce(f64::max)?;
            Some(AllReduceSummary {
                evidence,
                floor_us: floor,
                peak_busbw_gbps: peak,
            })
        })
        .max_by_key(|all_reduce| all_reduce.evidence.gpu_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::find;
    use crate::table::tests::{h100_with, repeat_with_hbm, table_rows};

    #[test]
    fn summary_has_one_latency_and_one_throughput_per_row() {
        let mut h100 = find("h100").cloned().unwrap();
        h100.repeats.clear();
        let table = summary(&h100, crate::profile::collectives()).unwrap();
        let rows = table_rows(&table);
        assert_eq!(rows[0], ["Operation", "Latency", "Throughput"]);
        assert!(rows.iter().all(|row| row.len() == 3), "{table}");
        assert_eq!(rows.len(), 10, "{table}");
        for expected in [
            "| Kernel launch | 2 μs |  |",
            "| HBM random read | 300 ns |  |",
            "| HBM copy, read + write counted |  | 3 TB/s |",
            "| Host ↔ device, pinned | 9 μs | 60 GB/s |",
            "| GPU ↔ GPU, NVLink one-way | 10 μs | 400 GB/s |",
            "| AllReduce, 8 GPUs on NVLink, bus bandwidth | 40 μs | 500 GB/s |",
            "| Dense BF16 GEMM, 8192³ |  | 700 TFLOP/s |",
            "| BF16 GEMM, 8×8192×8192 (decode, 8 tokens) |  | 20 TFLOP/s |",
        ] {
            assert!(table.contains(expected), "{expected}\n{table}");
        }
        assert!(!table.contains('†'), "{table}");
        assert!(table.contains("(H100!:8, 2026-10-12)"), "{table}");
    }

    #[test]
    fn summary_drops_rows_it_has_no_evidence_for() {
        let mut profile = h100_with(&[], None);
        let table = summary(&profile, &[]).unwrap();
        assert!(!table.contains("AllReduce"), "{table}");
        assert!(!table.contains("decode"), "{table}");

        let measured = profile.measured.as_mut().unwrap();
        measured.sync_us = None;
        measured.hbm_latency_ns = None;
        let table = summary(&profile, &[]).unwrap();
        assert!(!table.contains("Launch + sync"), "{table}");
        assert!(!table.contains("HBM random read"), "{table}");
    }

    #[test]
    fn summary_takes_the_slower_host_direction() {
        let mut profile = h100_with(&[], None);
        profile.nvbandwidth = None;
        let measured = profile.measured.as_mut().unwrap();
        measured.h2d_latency_us = Some(5.0);
        measured.d2h_latency_us = Some(20.0);
        measured.h2d_gbps = 30.0;
        measured.d2h_gbps = 60.0;
        let table = summary(&profile, &[]).unwrap();
        assert!(
            table.contains("| Host ↔ device, pinned | 20 μs | 30 GB/s |"),
            "{table}"
        );
    }

    #[test]
    fn summary_marks_cells_that_vary_by_machine() {
        let mut profile = h100_with(&[], None);
        let table = summary(&profile, &[]).unwrap();
        assert!(!table.contains('‡'), "{table}");

        let hbm = profile.measured.as_ref().unwrap().hbm_gbps;
        profile.repeats = vec![repeat_with_hbm(
            &profile,
            hbm * 0.8,
            ["GPU-aaaaaaaa-1", "GPU-bbbbbbbb-2"],
        )];
        let table = summary(&profile, &[]).unwrap();
        assert!(
            table.contains("| HBM copy, read + write counted |  | 3 TB/s ‡ |"),
            "{table}"
        );
        assert!(table.contains("| Kernel launch | 2 μs |  |"), "{table}");
        assert!(table.contains("more than 15% between runs"), "{table}");
    }
}
