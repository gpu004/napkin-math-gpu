//! Markdown table rendering for the README "Numbers" section: one
//! significant digit, sized for memorization rather than faux precision.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use napkin_profile::collective::{Collective, CollectiveEvidence, Topology};

use crate::price::PriceRecord;
use crate::profile::{
    Basis, CopyPath, CrossCheck, Dtype, GpuState, Measured, Profile, Rates, RoundState, Span,
    TransferMetric, TransferRate,
};
use crate::units::{
    fmt_duration, fmt_gbps, fmt_intensity, fmt_money, fmt_money_sig, fmt_sig, fmt_tflops, round_sig,
};
use crate::variation::{self, Metric, Unit};

const SIG: u32 = 1;

/// A cell gets a † when the rounds behind it disagree by more than this
/// fraction (`Measured::round_spread`). Back-to-back rounds of one kernel on
/// an otherwise idle GPU usually agree to a few tenths of a percent (fp32
/// GEMM and HBM copy do on every GPU here), so a larger range means the GPU
/// changed state during the run and the median alone would hide it.
pub const SPREAD_FLAG: f64 = 0.05;

/// `samples_ms` keys in table order, with the name the footnote uses.
const SAMPLE_NAMES: [(&str, &str); 17] = [
    ("launch_us", "kernel launch"),
    ("sync_us", "launch + sync"),
    ("host_copy_gbps", "host RAM copy"),
    ("h2d_latency_us", "host → device latency"),
    ("h2d_gbps", "host → device"),
    ("d2h_latency_us", "device → host latency"),
    ("d2h_gbps", "device → host"),
    ("hbm_latency_ns", "HBM random read"),
    ("hbm_gbps", "HBM copy"),
    ("elementwise_gbps", "elementwise add"),
    ("g2g_latency_us", "GPU ↔ GPU latency"),
    ("g2g_forward_gbps", "GPU 0 → 1"),
    ("g2g_reverse_gbps", "GPU 1 → 0"),
    ("gemm_fp16", "fp16 GEMM"),
    ("gemm_bf16", "bf16 GEMM"),
    ("gemm_tf32", "tf32 GEMM"),
    ("gemm_fp32", "fp32 GEMM"),
];

fn flagged(m: &Measured, keys: &[&str]) -> bool {
    keys.iter()
        .filter_map(|key| m.round_spread(key))
        .any(|spread| spread > SPREAD_FLAG)
}

/// Appends a † to a non-empty `cell` whose rounds are flagged.
fn mark(m: &Measured, keys: &[&str], cell: String) -> String {
    if !cell.is_empty() && flagged(m, keys) {
        format!("{cell} †")
    } else {
        cell
    }
}

fn cell_time(bytes: f64, gbps: f64, exact: bool) -> String {
    let rate = if exact { gbps } else { round_sig(gbps, SIG) };
    if rate <= 0.0 {
        return String::new();
    }
    fmt_duration(bytes / (rate * 1e9), if exact { 3 } else { SIG })
}

fn row(cells: &[&str]) -> String {
    format!("| {} |", cells.join(" | "))
}

fn emit(out: &mut String, cells: &[&str]) {
    out.push_str(&row(cells));
    out.push('\n');
}

fn host_link_label(
    bandwidth: Option<String>,
    gpu_gen: Option<u8>,
    effective_gen: Option<u8>,
    width: Option<u8>,
) -> String {
    let Some(bandwidth) = bandwidth else {
        return "—".to_owned();
    };
    match (gpu_gen, effective_gen, width) {
        (Some(gpu_gen), Some(effective_gen), Some(width)) if effective_gen < gpu_gen => {
            format!("{bandwidth} Gen{effective_gen} x{width} (GPU Gen{gpu_gen})")
        }
        (Some(_), Some(effective_gen), Some(width)) => {
            format!("{bandwidth} Gen{effective_gen} x{width}")
        }
        _ => bandwidth,
    }
}

fn host_link_cell(rates: &Rates, measured: Option<&Measured>, sig: u32) -> String {
    let bandwidth = rates
        .h2d
        .map(|rate| rate.gbps)
        .into_iter()
        .chain(rates.d2h.map(|rate| rate.gbps))
        .reduce(f64::min)
        .map(|value| fmt_gbps(value, sig));
    let (gpu_gen, effective_gen, width) = measured.map_or((None, None, None), |m| {
        (m.pcie_gen, m.pcie_effective_gen(), m.pcie_width)
    });
    host_link_label(bandwidth, gpu_gen, effective_gen, width)
}

/// Renders dated price records and groups records with the same note.
#[must_use]
pub fn prices(records: &[PriceRecord]) -> String {
    let mut out = String::new();
    emit(
        &mut out,
        &[
            "Record",
            "Provider",
            "SKU",
            "GPU",
            "Price",
            "Per GPU-hour",
            "Terms",
            "Dated",
            "Source",
        ],
    );
    emit(
        &mut out,
        &[
            "---", "---", "---", "---", "---:", "---:", "---", "---", "---",
        ],
    );
    let mut notes: Vec<(&str, Vec<&str>)> = Vec::new();
    for record in records {
        let price = record.listed_price();
        let terms = record.terms_label();
        let source_text = record
            .source
            .strip_prefix("https://")
            .unwrap_or(record.source);
        let source = format!("[{source_text}]({})", record.source);
        let id = format!("`{}`", record.id);
        emit(
            &mut out,
            &[
                &id,
                record.provider,
                record.sku,
                record.gpu,
                &price,
                &fmt_money(record.per_gpu_hour(), record.currency),
                &terms,
                &record.date.to_string(),
                &source,
            ],
        );
        if let Some((_, ids)) = notes.iter_mut().find(|(note, _)| *note == record.note) {
            ids.push(record.id);
        } else {
            notes.push((record.note, vec![record.id]));
        }
    }
    out.push('\n');
    for (note, ids) in notes {
        let ids = ids
            .iter()
            .map(|id| format!("`{id}`"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "- {ids}: {note}");
    }
    out
}

/// Renders the two "Numbers" tables (data movement + dense GEMM) for a
/// measured profile. Returns `None` when the profile has no measurement.
#[must_use]
pub fn numbers(profile: &Profile, exact: bool) -> Option<String> {
    let m = profile.measured.as_ref()?;
    let rates = profile.rates(false);
    let sig = if exact { 3 } else { SIG };
    let mut out = String::new();
    append_operation_table(&mut out, m, &rates, exact);
    out.push('\n');
    append_gemm_table(&mut out, m, exact);
    let measured_shapes = if m
        .gemm_shapes
        .as_ref()
        .is_some_and(|suite| !suite.shapes.is_empty())
    {
        "; measured shapes follow below"
    } else {
        ""
    };
    let _ = writeln!(
        out,
        "\nDense GEMM rows are the cuBLAS `GemmEx` 8192³ rate, a large-GEMM library ceiling, not the rate of every GEMM{measured_shapes}."
    );
    spread_footnote(&mut out, m);
    emit_gemm_shape_rows(&mut out, m, sig, exact);
    if let Some(state) = &m.gpu_state {
        gpu_state_table(&mut out, state);
    }
    Some(out)
}

/// Renders the memorization table that opens the README: one latency and one
/// throughput per row at one significant digit, like the table in
/// sirupsen/napkin-math. The detailed `numbers` tables carry the per-row basis,
/// round spread, and per-size times. Returns `None` without a measurement.
#[must_use]
pub fn summary(profile: &Profile, collectives: &[CollectiveEvidence]) -> Option<String> {
    let m = profile.measured.as_ref()?;
    let rates = profile.rates(false);
    let all_reduce = largest_nvlink_all_reduce(profile, collectives);
    let mut out = String::new();
    emit(&mut out, &["Operation", "Latency", "Throughput"]);
    emit(&mut out, &["---", "---:", "---:"]);
    let mut varies_by_machine = false;
    for (name, latency, throughput) in summary_rows(profile, m, &rates, all_reduce) {
        if latency.is_none() && throughput.is_none() {
            continue;
        }
        varies_by_machine |= [&latency, &throughput]
            .iter()
            .any(|cell| cell.as_deref().is_some_and(|cell| cell.ends_with('‡')));
        emit(
            &mut out,
            &[
                &name,
                latency.as_deref().unwrap_or_default(),
                throughput.as_deref().unwrap_or_default(),
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

/// Appends a ‡ to a non-empty `cell` when any metric behind it varies by
/// machine, as `mark` appends a † when its rounds disagree.
fn machine_mark(profile: &Profile, metrics: &[Metric], cell: Option<String>) -> Option<String> {
    let varies = metrics.iter().any(|metric| {
        variation::spread(profile, metric).is_some_and(|spread| spread.varies_by_machine())
    });
    cell.map(|cell| if varies { format!("{cell} ‡") } else { cell })
}

/// The summary's rows as (operation, latency, throughput); rows whose
/// evidence is missing come back with both cells `None`.
fn summary_rows(
    profile: &Profile,
    m: &Measured,
    rates: &Rates,
    all_reduce: Option<AllReduceSummary<'_>>,
) -> Vec<(String, Option<String>, Option<String>)> {
    let latency = |us: Option<f64>| us.map(|value| fmt_duration(value / 1e6, SIG));
    let throughput = |gbps: Option<f64>| gbps.map(|value| fmt_gbps(value, SIG));
    let slower = |a: Option<f64>, b: Option<f64>, pick: fn(f64, f64) -> f64| match (a, b) {
        (Some(a), Some(b)) => Some(pick(a, b)),
        (a, b) => a.or(b),
    };
    let decode = m.gemm_shapes.as_ref().and_then(|suite| {
        suite
            .shapes
            .iter()
            .find(|shape| shape.name == "decode-m8" && shape.dtype == Dtype::Bf16)
    });
    let machine = |metrics: &[Metric], cell: Option<String>| machine_mark(profile, metrics, cell);
    vec![
        (
            "Kernel launch".to_owned(),
            machine(&[variation::KERNEL_LAUNCH], latency(Some(m.launch_us))),
            None,
        ),
        (
            "Launch + sync round trip".to_owned(),
            machine(&[variation::LAUNCH_SYNC], latency(m.sync_us)),
            None,
        ),
        (
            "HBM random read".to_owned(),
            machine(
                &[variation::HBM_LATENCY],
                m.hbm_latency_ns
                    .map(|ns| format!("{} ns", fmt_sig(ns, SIG))),
            ),
            None,
        ),
        (
            "HBM copy, read + write counted".to_owned(),
            None,
            machine(&[variation::HBM_COPY], throughput(Some(m.hbm_gbps))),
        ),
        (
            "Host ↔ device, pinned".to_owned(),
            machine(
                &[variation::H2D_LATENCY, variation::D2H_LATENCY],
                latency(slower(m.h2d_latency_us, m.d2h_latency_us, f64::max)),
            ),
            machine(
                &[variation::H2D, variation::D2H],
                throughput(slower(
                    rates.h2d.map(|rate| rate.gbps),
                    rates.d2h.map(|rate| rate.gbps),
                    f64::min,
                )),
            ),
        ),
        (
            "GPU ↔ GPU, NVLink one-way".to_owned(),
            machine(&[variation::G2G_LATENCY], latency(m.g2g_latency_us)),
            machine(&[variation::G2G], throughput(Some(rates.g2g.gbps))),
        ),
        (
            all_reduce.map_or_else(String::new, |all_reduce| {
                format!(
                    "AllReduce, {} GPUs on NVLink, bus bandwidth",
                    all_reduce.evidence.gpu_count
                )
            }),
            latency(all_reduce.map(|all_reduce| all_reduce.floor_us)),
            throughput(all_reduce.map(|all_reduce| all_reduce.peak_busbw_gbps)),
        ),
        (
            "Dense BF16 GEMM, 8192³".to_owned(),
            None,
            machine(
                &[variation::BF16_GEMM],
                m.compute_tflops
                    .get(&Dtype::Bf16)
                    .map(|tflops| fmt_tflops(*tflops, SIG)),
            ),
        ),
        (
            decode.map_or_else(String::new, |shape| {
                format!(
                    "BF16 GEMM, {}×{}×{} (decode, {} tokens)",
                    shape.m, shape.n, shape.k, shape.m
                )
            }),
            None,
            decode.map(|shape| fmt_tflops(shape.tflops, SIG)),
        ),
    ]
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

fn unit_label(unit: Unit) -> &'static str {
    match unit {
        Unit::Nanoseconds => "ns",
        Unit::Microseconds => "μs",
        Unit::Gbps => "GB/s",
        Unit::Tflops => "TFLOP/s",
    }
}

/// Renders how far each headline napkin-bench number moves between the
/// canonical run and its repeats on other machines, then which machine each
/// run used. Only GPUs with at least one repeat get a column.
#[must_use]
pub fn variation(profiles: &[Profile]) -> String {
    let repeated: Vec<&Profile> = profiles
        .iter()
        .filter(|profile| profile.measured.is_some() && !profile.repeats.is_empty())
        .collect();
    if repeated.is_empty() {
        return "No repeated runs yet: `modal run modal/bench.py --gpu h100 --repeat 3` records \
                some.\n"
            .to_owned();
    }
    let mut out = String::new();
    let mut header = vec!["napkin-bench number".to_owned()];
    header.extend(repeated.iter().map(|profile| {
        format!(
            "{} ({} runs)",
            profile.id.to_uppercase(),
            variation::runs(profile).count()
        )
    }));
    emit_owned(&mut out, &header);
    let mut align = vec!["---".to_owned()];
    align.extend(repeated.iter().map(|_| "---:".to_owned()));
    emit_owned(&mut out, &align);
    for metric in &variation::METRICS {
        let mut cells = vec![format!("{}, {}", metric.name, unit_label(metric.unit))];
        cells.extend(repeated.iter().map(|profile| {
            variation::spread(profile, metric).map_or_else(
                || "—".to_owned(),
                |spread| {
                    format!(
                        "{}–{} ({}%){}",
                        fmt_sig(spread.low, 3),
                        fmt_sig(spread.high, 3),
                        fmt_sig(spread.fraction() * 100.0, 2),
                        if spread.varies_by_machine() {
                            " ‡"
                        } else {
                            ""
                        }
                    )
                },
            )
        }));
        emit_owned(&mut out, &cells);
    }
    let _ = writeln!(
        out,
        "\nEach cell is the lowest–highest median round over the runs, then the spread: highest \
         over lowest, minus one. ‡ marks more than {:.0}%, the most sirupsen/napkin-math lets a \
         number move between machines. Transfers here are napkin-bench's own copies, not \
         NVBandwidth.\n",
        variation::MACHINE_FLAG * 100.0
    );
    append_variation_hosts(&mut out, &repeated);
    out.push_str(
        "\nGPU UUIDs are unique per physical GPU, shown by their first 8 hex digits, so runs with \
         different UUIDs used different GPUs. Modal shows no machine ID inside a container: two \
         2-GPU runs in the same provider and region may still share one 8-GPU node.\n",
    );
    out
}

/// One row per run behind `variation`: when it ran, on which driver, and on
/// which machine, as far as Modal lets a container tell.
fn append_variation_hosts(out: &mut String, repeated: &[&Profile]) {
    emit(
        out,
        &[
            "GPU",
            "Run captured (UTC)",
            "Driver",
            "Provider",
            "Region",
            "GPU UUIDs",
        ],
    );
    emit(out, &["---", "---", "---", "---", "---", "---"]);
    for profile in repeated {
        let gpu = profile.id.to_uppercase();
        if let Some(canonical) = &profile.measured {
            emit(
                out,
                &[
                    &gpu,
                    &format!("{} (canonical)", canonical.captured_at),
                    &canonical.driver_version,
                    "not recorded",
                    "not recorded",
                    "not recorded",
                ],
            );
        }
        for repeat in &profile.repeats {
            let uuids = repeat
                .host
                .gpu_uuids
                .iter()
                .map(|uuid| {
                    uuid.strip_prefix("GPU-")
                        .unwrap_or(uuid)
                        .chars()
                        .take(8)
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join(", ");
            emit(
                out,
                &[
                    &gpu,
                    &repeat.measured.captured_at,
                    &repeat.measured.driver_version,
                    repeat
                        .host
                        .cloud_provider
                        .strip_prefix("CLOUD_PROVIDER_")
                        .unwrap_or(&repeat.host.cloud_provider),
                    &repeat.host.region,
                    &uuids,
                ],
            );
        }
    }
}

/// One row of the unit-cost table: what is moved or computed, how many GPUs
/// the rental bills while it happens, and how long one GPU takes.
struct UnitCost {
    name: &'static str,
    gpus_billed: f64,
    seconds: fn(&Rates) -> Option<f64>,
}

const TB: f64 = 1e12;
const EFLOP: f64 = 1e18;

const UNIT_COSTS: [UnitCost; 4] = [
    UnitCost {
        name: "1 TB of HBM traffic, read + write counted",
        gpus_billed: 1.0,
        seconds: |rates| Some(TB / (rates.hbm_gbps * 1e9)),
    },
    UnitCost {
        name: "1 TB host ↔ device, pinned, slower direction",
        gpus_billed: 1.0,
        seconds: |rates| {
            rates
                .h2d
                .into_iter()
                .chain(rates.d2h)
                .map(|rate| rate.gbps)
                .reduce(f64::min)
                .map(|gbps| TB / (gbps * 1e9))
        },
    },
    // A peer copy needs both GPUs of the pair, as in `modal run ... --gpu
    // H100!:2`, and the rental bills both for the whole copy.
    UnitCost {
        name: "1 TB GPU → GPU over NVLink, both GPUs billed",
        gpus_billed: 2.0,
        seconds: |rates| Some(TB / (rates.g2g.gbps * 1e9)),
    },
    UnitCost {
        name: "1 EFLOP of dense BF16 GEMM at the 8192³ rate",
        gpus_billed: 1.0,
        seconds: |rates| {
            rates
                .compute_tflops
                .get(&Dtype::Bf16)
                .map(|rate| EFLOP / (rate.tflops * 1e12))
        },
    },
];

/// Renders what moving 1 TB or doing 1 EFLOP of dense BF16 costs on each
/// priced GPU, like the "$ / TiB read" table in sirupsen/napkin-math: one
/// GPU's time at the profile's rate, times the GPUs billed, times the price
/// record's per-second price. One column per price record, so two providers
/// for one GPU get two columns.
#[must_use]
pub fn unit_costs(profiles: &[Profile], records: &[PriceRecord]) -> String {
    let columns: Vec<(&PriceRecord, &Profile, Rates)> = records
        .iter()
        .filter_map(|record| {
            let profile = profiles.iter().find(|p| p.id == record.gpu)?;
            Some((record, profile, profile.rates(false)))
        })
        .collect();
    let mut out = String::new();
    let mut header = vec!["Work".to_owned()];
    header.extend(
        columns
            .iter()
            .map(|(record, profile, _)| format!("{} `{}`", profile.id.to_uppercase(), record.id)),
    );
    emit_owned(&mut out, &header);
    let mut align = vec!["---".to_owned()];
    align.extend(columns.iter().map(|_| "---:".to_owned()));
    emit_owned(&mut out, &align);

    let mut price = vec!["Price per GPU-hour".to_owned()];
    price.extend(
        columns
            .iter()
            .map(|(record, _, _)| fmt_money(record.per_gpu_hour(), record.currency)),
    );
    emit_owned(&mut out, &price);
    for row in &UNIT_COSTS {
        let mut cells = vec![row.name.to_owned()];
        cells.extend(columns.iter().map(|(record, _, rates)| {
            (row.seconds)(rates).map_or_else(
                || "—".to_owned(),
                |seconds| {
                    fmt_money_sig(
                        row.gpus_billed * seconds * record.per_gpu_hour() / 3600.0,
                        record.currency,
                        SIG,
                    )
                },
            )
        }));
        emit_owned(&mut out, &cells);
    }
    let _ = writeln!(
        out,
        "\nEach cell is the time one GPU takes at the rates in `napkin table` and `napkin \
         compare`, times the GPUs billed, times the price record, at one significant digit. It \
         assumes billing covers only that work at full speed: no startup, idle time, or rounding \
         up to the billing increment, which `napkin cost` adds."
    );
    out
}

fn emit_owned(out: &mut String, cells: &[String]) {
    emit(out, &cells.iter().map(String::as_str).collect::<Vec<_>>());
}

fn append_operation_table(out: &mut String, m: &Measured, rates: &Rates, exact: bool) {
    emit(
        out,
        &[
            "Operation",
            "Latency",
            "Throughput",
            "1 MB",
            "1 GB",
            "Basis",
        ],
    );
    emit(out, &["---", "---:", "---:", "---:", "---:", "---"]);
    for cells in operation_rows(m, rates, exact) {
        if cells[1..5].iter().all(String::is_empty) {
            continue;
        }
        emit(
            out,
            &[
                &cells[0], &cells[1], &cells[2], &cells[3], &cells[4], &cells[5],
            ],
        );
    }
}

fn operation_rows(m: &Measured, rates: &Rates, exact: bool) -> Vec<[String; 6]> {
    let sig = if exact { 3 } else { SIG };
    let lat_us = |value: f64| fmt_duration(value / 1e6, sig);
    let gbps = |value: f64| fmt_gbps(value, sig);
    let mb = |value: f64| cell_time(1e6, value, exact);
    let gb = |value: f64| cell_time(1e9, value, exact);
    let mut rows = non_transfer_rows(m, sig, &gbps, &mb, &gb, &lat_us);
    rows.insert(
        3,
        transfer_row(
            "Host → device, pinned",
            m.h2d_latency_us,
            rates.h2d,
            sig,
            exact,
        ),
    );
    rows.insert(
        4,
        transfer_row(
            "Device → host, pinned",
            m.d2h_latency_us,
            rates.d2h,
            sig,
            exact,
        ),
    );
    rows.push(transfer_row(
        "GPU ↔ GPU, NVLink one-way, slower direction",
        m.g2g_latency_us,
        Some(rates.g2g),
        sig,
        exact,
    ));
    rows
}

fn non_transfer_rows(
    m: &Measured,
    sig: u32,
    gbps: &impl Fn(f64) -> String,
    mb: &impl Fn(f64) -> String,
    gb: &impl Fn(f64) -> String,
    lat_us: &impl Fn(f64) -> String,
) -> Vec<[String; 6]> {
    vec![
        [
            "Kernel launch (back-to-back)".to_owned(),
            mark(m, &["launch_us"], lat_us(m.launch_us)),
            String::new(),
            String::new(),
            String::new(),
            "napkin-bench".to_owned(),
        ],
        [
            "Launch + sync round trip".to_owned(),
            mark(m, &["sync_us"], m.sync_us.map_or_else(String::new, lat_us)),
            String::new(),
            String::new(),
            String::new(),
            "napkin-bench".to_owned(),
        ],
        [
            "Pinned host RAM copy".to_owned(),
            String::new(),
            mark(m, &["host_copy_gbps"], gbps(m.host_copy_gbps)),
            mb(m.host_copy_gbps),
            gb(m.host_copy_gbps),
            "napkin-bench".to_owned(),
        ],
        [
            "HBM random read (pointer chase)".to_owned(),
            mark(
                m,
                &["hbm_latency_ns"],
                m.hbm_latency_ns
                    .map_or_else(String::new, |ns| format!("{} ns", fmt_sig(ns, sig))),
            ),
            String::new(),
            String::new(),
            String::new(),
            "napkin-bench".to_owned(),
        ],
        [
            "HBM copy (read + write)".to_owned(),
            String::new(),
            mark(m, &["hbm_gbps"], gbps(m.hbm_gbps)),
            mb(m.hbm_gbps),
            gb(m.hbm_gbps),
            "napkin-bench".to_owned(),
        ],
        [
            "Elementwise add".to_owned(),
            String::new(),
            mark(m, &["elementwise_gbps"], gbps(m.elementwise_gbps)),
            mb(m.elementwise_gbps),
            gb(m.elementwise_gbps),
            "napkin-bench".to_owned(),
        ],
    ]
}

fn transfer_row(
    name: &str,
    latency_us: Option<f64>,
    rate: Option<TransferRate>,
    sig: u32,
    exact: bool,
) -> [String; 6] {
    [
        name.to_owned(),
        latency_us.map_or_else(String::new, |value| fmt_duration(value / 1e6, sig)),
        rate.map_or_else(String::new, |value| fmt_gbps(value.gbps, sig)),
        rate.map_or_else(String::new, |value| cell_time(1e6, value.gbps, exact)),
        rate.map_or_else(String::new, |value| cell_time(1e9, value.gbps, exact)),
        rate.map_or_else(String::new, |value| {
            transfer_row_basis(value, latency_us.is_some())
        }),
    ]
}

fn transfer_row_basis(rate: TransferRate, has_latency: bool) -> String {
    let mut label = format!(
        "{}{}",
        rate.basis,
        if rate.basis == Basis::Nvbandwidth {
            " CE"
        } else {
            ""
        }
    );
    if has_latency && rate.basis == Basis::Nvbandwidth {
        label.push_str("; latency napkin-bench");
    }
    label
}

fn append_gemm_table(out: &mut String, m: &Measured, exact: bool) {
    let sig = if exact { 3 } else { SIG };
    emit(
        out,
        &[
            "Dense GEMM, 8192³",
            "Throughput",
            "1 GFLOP",
            "1 TFLOP",
            "Ridge point",
            "Basis",
        ],
    );
    emit(out, &["---", "---:", "---:", "---:", "---:", "---"]);
    for dtype in [Dtype::Fp16, Dtype::Bf16, Dtype::Tf32, Dtype::Fp32] {
        emit_gemm_row(out, m, dtype, sig, exact);
    }
}

fn spread_footnote(out: &mut String, m: &Measured) {
    let mut spreads: Vec<String> = SAMPLE_NAMES
        .iter()
        .filter_map(|(key, name)| {
            m.round_spread(key)
                .filter(|spread| *spread > SPREAD_FLAG)
                .map(|spread| format!("{name} {}%", fmt_sig(spread * 100.0, 2)))
        })
        .collect();
    if let Some(suite) = &m.gemm_shapes {
        spreads.extend(suite.shapes.iter().filter_map(|shape| {
            suite
                .round_spread(&shape.samples_key())
                .filter(|spread| *spread > SPREAD_FLAG)
                .map(|spread| {
                    format!(
                        "{} {} GEMM {}%",
                        shape.name,
                        shape.dtype,
                        fmt_sig(spread * 100.0, 2)
                    )
                })
        }));
    }
    if spreads.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "\n† Rounds disagree by more than {:.0}%: {}. Spread is the slowest measured round over \
         the fastest, minus one: the observed range, not a confidence interval. The cell shows the \
         median round.",
        SPREAD_FLAG * 100.0,
        spreads.join(", "),
    );
}

fn emit_gemm_shape_rows(out: &mut String, m: &Measured, sig: u32, exact: bool) {
    let Some(suite) = &m.gemm_shapes else {
        return;
    };
    if suite.shapes.is_empty() {
        return;
    }
    out.push('\n');
    emit(
        out,
        &[
            "cuBLASLt GEMM shape",
            "Dimensions (M×N×K)",
            "dtype",
            "Throughput",
            "of 8192³ peak",
            "1 TFLOP",
        ],
    );
    emit(out, &["---", "---:", "---", "---:", "---:", "---:"]);
    for shape in &suite.shapes {
        let rate = if exact {
            shape.tflops
        } else {
            round_sig(shape.tflops, SIG)
        };
        let dimensions = format!("{}×{}×{}", shape.m, shape.n, shape.k);
        let key = shape.samples_key();
        let peak_percent = m
            .compute_tflops
            .get(&shape.dtype)
            .map(|peak| {
                format!(
                    "{}%",
                    fmt_sig(shape.tflops / peak * 100.0, if exact { 3 } else { 2 })
                )
            })
            .unwrap_or_default();
        let throughput = fmt_tflops(shape.tflops, sig);
        let throughput = if suite
            .round_spread(&key)
            .is_some_and(|spread| spread > SPREAD_FLAG)
        {
            format!("{throughput} †")
        } else {
            throughput
        };
        emit(
            out,
            &[
                &shape.name,
                &dimensions,
                &shape.dtype.to_string(),
                &throughput,
                &peak_percent,
                &fmt_duration(1.0 / rate, sig),
            ],
        );
    }
    let pcie = match (suite.pcie_gen, suite.pcie_width) {
        (Some(generation), Some(width)) => format!(", PCIe Gen{generation} x{width}"),
        (Some(generation), None) => format!(", PCIe Gen{generation}"),
        (None, Some(width)) => format!(", PCIe x{width}"),
        (None, None) => String::new(),
    };
    let different_run = if suite.captured_at == m.captured_at {
        String::new()
    } else {
        format!(
            "; a different run from the generic peak captured {}",
            m.captured_at
        )
    };
    let _ = writeln!(
        out,
        "\nShape suite: {} captured {} from {} on {} (driver {}, CUDA {}{}){different_run}; % of peak uses the preserved profile's generic 8192³ `GemmEx` rate. TN column-major, fp32 accumulation and CUDA_R_32F scaling, 32 MiB heuristic workspace; query once before warm-up and reuse the selected algorithm.",
        suite.library,
        suite.captured_at,
        suite.gpu_request,
        suite.device_name,
        suite.driver_version,
        suite.cuda_version,
        pcie,
    );
}

fn gpu_state_table(out: &mut String, state: &GpuState) {
    out.push('\n');
    emit(
        out,
        &[
            "GEMM rounds",
            "SM clock",
            "Memory clock",
            "Power",
            "Temperature",
            "Clock limits",
        ],
    );
    emit(out, &["---", "---:", "---:", "---:", "---:", "---"]);
    for dtype in [Dtype::Fp16, Dtype::Bf16, Dtype::Tf32, Dtype::Fp32] {
        let Some(rounds) = state.rounds.get(&format!("gemm_{dtype}")) else {
            continue;
        };
        let lows_highs = |pick: fn(&RoundState) -> Option<Span>| {
            let spans: Vec<_> = rounds.iter().filter_map(pick).collect();
            fmt_range(
                spans.iter().map(|span| span.min).reduce(f64::min),
                spans.iter().map(|span| span.max).reduce(f64::max),
            )
        };
        let mut power: Vec<f64> = rounds.iter().filter_map(|round| round.power_w).collect();
        power.sort_by(f64::total_cmp);
        let polls: u32 = rounds.iter().map(|round| round.polls).sum();
        let mut reasons = BTreeMap::new();
        for round in rounds {
            for (name, count) in &round.clock_limit_reasons {
                *reasons.entry(name.as_str()).or_insert(0_u32) += count;
            }
        }
        let limits = reasons
            .iter()
            .map(|(name, count)| {
                format!(
                    "{name} {:.0}%",
                    f64::from(*count) * 100.0 / f64::from(polls.max(1))
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        emit(
            out,
            &[
                &dtype.to_string(),
                &with_unit(&lows_highs(|round| round.sm_clock_mhz), "MHz"),
                &with_unit(&lows_highs(|round| round.mem_clock_mhz), "MHz"),
                &with_unit(
                    &median(&power).map_or_else(String::new, |w| format!("{w:.0}")),
                    "W",
                ),
                &with_unit(&lows_highs(|round| round.temperature_c), "°C"),
                if limits.is_empty() { "none" } else { &limits },
            ],
        );
    }
    let _ = write!(
        out,
        "\nRead through {} every {} ms on the GPU that ran the GEMMs, during the timed rounds \
         only. Board limits: {:.0} W, SM clock {} MHz, memory clock {} MHz. Power is the median \
         round's energy-counter change between its first and last counter update; the counter \
         updates only a few times per round, so a single round in `gpu_state` can read above the \
         board limit. Clock limits give the share of polls that reported each \
         NVML clock-limit reason.\n",
        state.source,
        fmt_sig(state.poll_interval_ms, 2),
        state.power_limit_w,
        state.max_sm_clock_mhz,
        state.max_mem_clock_mhz,
    );
}

fn fmt_range(low: Option<f64>, high: Option<f64>) -> String {
    match (low, high) {
        (Some(low), Some(high)) if low.round() < high.round() => {
            format!("{low:.0}–{high:.0}")
        }
        (Some(value), _) | (None, Some(value)) => format!("{value:.0}"),
        (None, None) => String::new(),
    }
}

fn median(sorted: &[f64]) -> Option<f64> {
    let mid = sorted.len() / 2;
    match sorted.len() {
        0 => None,
        n if n % 2 == 1 => sorted.get(mid).copied(),
        _ => Some((sorted.get(mid - 1)? + sorted.get(mid)?) / 2.0),
    }
}

fn with_unit(value: &str, unit: &str) -> String {
    if value.is_empty() {
        "—".to_owned()
    } else {
        format!("{value} {unit}")
    }
}

fn emit_gemm_row(out: &mut String, m: &Measured, dtype: Dtype, sig: u32, exact: bool) {
    if let Some(&tflops) = m.compute_tflops.get(&dtype) {
        let rate = if exact {
            tflops
        } else {
            round_sig(tflops, SIG)
        };
        let ridge = fmt_intensity(tflops * 1000.0 / m.hbm_gbps, if exact { 4 } else { 2 });
        let ts = |flops: f64| fmt_duration(flops / (rate * 1e12), sig);
        emit(
            out,
            &[
                &dtype.to_string(),
                &mark(m, &[&format!("gemm_{dtype}")], fmt_tflops(tflops, sig)),
                &ts(1e9),
                &ts(1e12),
                &ridge,
                "napkin-bench",
            ],
        );
    }
}

fn basis_label(profile: &Profile) -> String {
    match (profile.rates(false).basis, &profile.measured) {
        (Basis::NapkinBench, Some(m)) => {
            format!(
                "napkin-bench {}",
                &m.captured_at[..10.min(m.captured_at.len())]
            )
        }
        _ => "datasheet".to_owned(),
    }
}

fn transfer_basis_label(profile: &Profile, rate: TransferRate) -> String {
    match rate.basis {
        Basis::NapkinBench => profile.measured.as_ref().map_or_else(
            || "napkin-bench".to_owned(),
            |m| {
                format!(
                    "napkin-bench {}",
                    &m.captured_at[..10.min(m.captured_at.len())]
                )
            },
        ),
        Basis::Nvbandwidth => profile.nvbandwidth.as_ref().map_or_else(
            || "NVBandwidth".to_owned(),
            |evidence| {
                format!(
                    "NVBandwidth {}",
                    &evidence.captured_at[..10.min(evidence.captured_at.len())]
                )
            },
        ),
        Basis::Datasheet => "datasheet".to_owned(),
    }
}

fn transfer_basis(rates: &[Option<TransferRate>], profile: &Profile) -> String {
    let mut labels = Vec::new();
    for rate in rates.iter().flatten() {
        let label = transfer_basis_label(profile, *rate);
        if !labels.contains(&label) {
            labels.push(label);
        }
    }
    if labels.is_empty() {
        "datasheet".to_owned()
    } else {
        labels.join("; ")
    }
}

/// Renders the cross-GPU compare tables: one row per profile, then measured
/// versus datasheet for the GPUs that have a calibration.
#[must_use]
pub fn compare(profiles: &[Profile], exact: bool) -> String {
    let sig = if exact { 3 } else { SIG };
    let mut out = String::new();
    emit(
        &mut out,
        &[
            "GPU",
            "Memory",
            "HBM",
            "Dense BF16",
            "BF16 ridge",
            "NVLink",
            "Host link",
            "Launch",
            "Basis",
            "Transfer basis",
        ],
    );
    emit(
        &mut out,
        &[
            "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---",
        ],
    );
    for p in profiles {
        let rates = p.rates(false);
        let host_link = host_link_cell(&rates, p.measured.as_ref(), sig);
        let bf16 = rates
            .compute_tflops
            .get(&Dtype::Bf16)
            .map_or_else(String::new, |r| fmt_tflops(r.tflops, sig));
        let ridge = rates
            .compute_tflops
            .get(&Dtype::Bf16)
            .map_or_else(String::new, |r| {
                fmt_intensity(r.tflops * 1000.0 / rates.hbm_gbps, 2)
            });
        emit(
            &mut out,
            &[
                &p.name,
                &format!("{} GB", fmt_sig(p.memory_gb, 3)),
                &fmt_gbps(rates.hbm_gbps, sig),
                &bf16,
                &ridge,
                &fmt_gbps(p.ceiling.nvlink_gbps, sig),
                &host_link,
                &rates
                    .launch_us
                    .map_or_else(|| "—".to_owned(), |v| fmt_duration(v / 1e6, sig)),
                &basis_label(p),
                &transfer_basis(&[rates.h2d, rates.d2h, Some(rates.g2g)], p),
            ],
        );
    }

    append_measured_table(&mut out, profiles);
    append_transfer_cross_check_table(&mut out, profiles);
    out
}

fn append_measured_table(out: &mut String, profiles: &[Profile]) {
    let measured: Vec<&Profile> = profiles.iter().filter(|p| p.measured.is_some()).collect();
    if !measured.is_empty() {
        out.push('\n');
        emit(
            out,
            &["GPU", "HBM measured / ceiling", "BF16 measured / ceiling"],
        );
        emit(out, &["---", "---:", "---:"]);
        for p in measured {
            let Some(m) = &p.measured else { continue };
            let hbm_pct = m.hbm_gbps / p.ceiling.hbm_gbps * 100.0;
            let bf16_cell = m
                .compute_tflops
                .get(&Dtype::Bf16)
                .zip(p.ceiling.compute_tflops.get(&Dtype::Bf16))
                .map_or_else(
                    || "—".to_owned(),
                    |(mv, cv)| format!("{mv:.0} / {cv:.0} TFLOP/s ({:.0}%)", mv / cv * 100.0),
                );
            emit(
                out,
                &[
                    &p.name,
                    &format!(
                        "{:.0} / {:.0} GB/s ({hbm_pct:.0}%)",
                        m.hbm_gbps, p.ceiling.hbm_gbps
                    ),
                    &bf16_cell,
                ],
            );
        }
    }
}

fn append_transfer_cross_check_table(out: &mut String, profiles: &[Profile]) {
    let evidenced: Vec<&Profile> = profiles
        .iter()
        .filter(|profile| profile.nvbandwidth.is_some())
        .collect();
    if !evidenced.is_empty() {
        out.push('\n');
        emit(
            out,
            &[
                "GPU",
                "Transfer",
                "NVBandwidth CE",
                "NVBandwidth SM",
                "napkin-bench",
                "CE / napkin-bench",
                "PCIe (NVBandwidth / napkin-bench)",
            ],
        );
        emit(out, &["---", "---", "---:", "---:", "---:", "---:", "---"]);
        for profile in evidenced {
            let (Some(evidence), Some(measured)) = (&profile.nvbandwidth, &profile.measured) else {
                continue;
            };
            for check in profile.transfer_cross_check() {
                let (ce, sm) = cross_check_rates(evidence, &check);
                let pcie = match check.metric {
                    TransferMetric::Peer { .. } => "n/a (NVLink)".to_owned(),
                    TransferMetric::H2d | TransferMetric::D2h => format!(
                        "{} / {}",
                        pcie_label(
                            evidence
                                .pcie_links
                                .first()
                                .map(|link| (link.r#gen, link.width)),
                        ),
                        pcie_label(measured.pcie_gen.zip(measured.pcie_width))
                    ),
                };
                let metric = check.metric.to_string();
                let measured = fmt_gbps(check.napkin_bench_gbps, 3);
                emit(
                    out,
                    &[
                        &profile.name,
                        &metric,
                        &ce.map_or_else(|| "—".to_owned(), |rate| fmt_gbps(rate, 3)),
                        &sm.map_or_else(|| "—".to_owned(), |rate| fmt_gbps(rate, 3)),
                        &measured,
                        &format!("{:.3}", check.ratio),
                        &pcie,
                    ],
                );
            }
        }
    }
}

fn cross_check_rates(
    profile: &crate::profile::Nvbandwidth,
    check: &CrossCheck,
) -> (Option<f64>, Option<f64>) {
    match check.metric {
        TransferMetric::H2d => (profile.h2d(CopyPath::Ce), profile.h2d(CopyPath::Sm)),
        TransferMetric::D2h => (profile.d2h(CopyPath::Ce), profile.d2h(CopyPath::Sm)),
        TransferMetric::Peer { src, dst } => (
            profile.peer(CopyPath::Ce, src, dst),
            profile.peer(CopyPath::Sm, src, dst),
        ),
    }
}

fn pcie_label(link: Option<(u8, u8)>) -> String {
    link.map_or_else(
        || "—".to_owned(),
        |(generation, width)| format!("Gen{generation} x{width}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{Host, Repeat, find};

    fn h100_with(samples: &[(&str, Vec<f64>)], gpu_state: Option<GpuState>) -> Profile {
        let mut profile = find("h100").cloned().unwrap();
        let measured = profile.measured.as_mut().unwrap();
        measured.samples_ms = samples
            .iter()
            .map(|(key, rounds)| ((*key).to_owned(), rounds.clone()))
            .collect();
        measured.gpu_state = gpu_state;
        measured.gemm_shapes = None;
        profile.repeats.clear();
        profile
    }

    fn repeat_with_hbm(profile: &Profile, hbm_gbps: f64, uuids: [&str; 2]) -> Repeat {
        let mut measured = profile.measured.clone().unwrap();
        measured.hbm_gbps = hbm_gbps;
        measured.captured_at = "2026-11-05T23:50:00Z".to_owned();
        Repeat {
            host: Host {
                cloud_provider: "CLOUD_PROVIDER_OCI".to_owned(),
                region: "us-central".to_owned(),
                task_id: "ta-test".to_owned(),
                gpu_uuids: uuids.iter().map(|uuid| (*uuid).to_owned()).collect(),
            },
            measured,
        }
    }

    #[test]
    fn flags_gemm_rounds_that_disagree_beyond_threshold() {
        let profile = h100_with(
            &[
                ("gemm_fp16", vec![1.40, 1.45, 1.61]),
                ("gemm_fp32", vec![21.0, 21.0, 21.01]),
            ],
            None,
        );
        let table = numbers(&profile, false).unwrap();
        assert!(table.contains("| fp16 | 700 TFLOP/s † |"), "{table}");
        assert!(table.contains("| fp32 | 50 TFLOP/s |"), "{table}");
        assert!(table.contains("fp16 GEMM 15%."), "{table}");
        assert!(!table.contains("fp32 GEMM"), "{table}");
        assert!(table.contains("not a confidence interval"), "{table}");
    }

    #[test]
    fn omits_footnote_when_rounds_agree() {
        let profile = h100_with(&[("gemm_fp16", vec![1.40, 1.41, 1.42])], None);
        let table = numbers(&profile, false).unwrap();
        assert!(!table.contains('†'), "{table}");
    }

    #[test]
    fn renders_shape_rates_and_names_their_spread() {
        let mut profile = h100_with(&[], None);
        let mut shapes = Vec::new();
        for (name, m, n, k) in [
            ("square", 8192, 8192, 8192),
            ("prefill", 4096, 28672, 8192),
            ("decode-m8", 8, 8192, 8192),
            ("decode-m64", 64, 8192, 8192),
            ("square-k8191", 8192, 8192, 8191),
        ] {
            for dtype in [Dtype::Fp16, Dtype::Bf16] {
                let mut shape = shape_fixture("prefill", 4096, 28672, 8192, Dtype::Bf16);
                shape.name = name.to_owned();
                shape.m = m;
                shape.n = n;
                shape.k = k;
                shape.dtype = dtype;
                shapes.push(shape);
            }
        }
        let mut suite = shape_suite_fixture(shapes);
        suite
            .samples_ms
            .insert("gemm_shape_prefill_bf16".to_owned(), vec![2.0, 2.2]);
        profile.measured.as_mut().unwrap().gemm_shapes = Some(suite);
        let table = numbers(&profile, true).unwrap();
        assert!(table.contains("| Dense GEMM, 8192³ |"), "{table}");
        assert!(
            table.contains("Dense GEMM rows are the cuBLAS `GemmEx` 8192³ rate, a large-GEMM library ceiling, not the rate of every GEMM; measured shapes follow below."),
            "{table}"
        );
        assert!(
            table.contains("| prefill | 4096×28672×8192 | bf16 | 123 TFLOP/s † |"),
            "{table}"
        );
        assert!(table.contains("prefill bf16 GEMM 10%."), "{table}");
        assert!(
            table.contains("Shape suite: cuBLASLt 120901 captured"),
            "{table}"
        );
        assert!(
            table.contains("different run from the generic peak"),
            "{table}"
        );
        assert!(table.contains("TN column-major"), "{table}");
        let rows: Vec<_> = table
            .lines()
            .filter(|line| {
                [
                    "square",
                    "prefill",
                    "decode-m8",
                    "decode-m64",
                    "square-k8191",
                ]
                .iter()
                .any(|name| line.starts_with(&format!("| {name} |")))
            })
            .collect();
        let expected = [
            "| square | 8192×8192×8192 | fp16 |",
            "| square | 8192×8192×8192 | bf16 |",
            "| prefill | 4096×28672×8192 | fp16 |",
            "| prefill | 4096×28672×8192 | bf16 |",
            "| decode-m8 | 8×8192×8192 | fp16 |",
            "| decode-m8 | 8×8192×8192 | bf16 |",
            "| decode-m64 | 64×8192×8192 | fp16 |",
            "| decode-m64 | 64×8192×8192 | bf16 |",
            "| square-k8191 | 8192×8192×8191 | fp16 |",
            "| square-k8191 | 8192×8192×8191 | bf16 |",
        ];
        for (row, expected) in rows.iter().zip(expected) {
            assert!(row.starts_with(expected), "{row}");
        }
        assert_eq!(rows.len(), expected.len());
        let without_shapes = numbers(&h100_with(&[], None), false).unwrap();
        assert!(
            without_shapes.contains("Dense GEMM rows are the cuBLAS `GemmEx` 8192³ rate, a large-GEMM library ceiling, not the rate of every GEMM."),
            "{without_shapes}"
        );
        assert!(!without_shapes.contains("measured shapes follow below"));
    }

    #[test]
    fn renders_shape_peak_shares_as_percentages() {
        let mut profile = h100_with(&[], None);
        let peak = profile.measured.as_ref().unwrap().compute_tflops[&Dtype::Fp16];
        let shapes = [
            ("full", 100.0),
            ("quarter", 25.74),
            ("small", 3.276),
            ("fifth", 19.8),
        ]
        .into_iter()
        .map(|(name, percentage)| {
            let mut shape = shape_fixture(name, 8, 8, 8, Dtype::Fp16);
            shape.tflops = peak * percentage / 100.0;
            shape
        })
        .collect();
        profile.measured.as_mut().unwrap().gemm_shapes = Some(shape_suite_fixture(shapes));

        let assert_shares = |table: &str, expected: &[(&str, &str)]| {
            for (name, percentage) in expected {
                let row = table
                    .lines()
                    .find(|line| line.starts_with(&format!("| {name} |")))
                    .unwrap_or_default();
                assert!(row.contains(&format!("| {percentage} |")), "{row}");
            }
        };

        let rounded = numbers(&profile, false).unwrap();
        assert!(rounded.contains("| of 8192³ peak |"), "{rounded}");
        assert_shares(
            &rounded,
            &[
                ("full", "100%"),
                ("quarter", "26%"),
                ("small", "3.3%"),
                ("fifth", "20%"),
            ],
        );

        let exact = numbers(&profile, true).unwrap();
        assert_shares(
            &exact,
            &[
                ("full", "100%"),
                ("quarter", "25.7%"),
                ("small", "3.28%"),
                ("fifth", "19.8%"),
            ],
        );
    }

    fn shape_fixture(
        name: &str,
        m: u32,
        n: u32,
        k: u32,
        dtype: Dtype,
    ) -> crate::profile::GemmShape {
        crate::profile::GemmShape {
            name: name.to_owned(),
            m,
            n,
            k,
            dtype,
            compute_type: "CUBLAS_COMPUTE_32F".to_owned(),
            scale_type: "CUDA_R_32F".to_owned(),
            trans_a: "T".to_owned(),
            trans_b: "N".to_owned(),
            lda: k,
            ldb: k,
            ldc: m,
            epilogue: "CUBLASLT_EPILOGUE_DEFAULT".to_owned(),
            algorithm: crate::profile::GemmAlgo {
                id: 1,
                tile: 1,
                stages: 1,
                split_k: 0,
                reduction_scheme: 0,
                cta_swizzling: 0,
                custom_option: 0,
                workspace_bytes: 0,
                waves: 1.0,
            },
            workspace_limit_bytes: 32 * 1024 * 1024,
            tflops: 123.0,
            check: crate::profile::GemmCheck {
                sampled: 256,
                max_abs_error: 0.1,
                rtol: 0.08,
                atol: 1.0,
                passed: true,
            },
        }
    }

    fn shape_suite_fixture(
        shapes: Vec<crate::profile::GemmShape>,
    ) -> crate::profile::GemmShapeSuite {
        crate::profile::GemmShapeSuite {
            captured_at: "2026-10-14T00:00:00Z".to_owned(),
            gpu_request: "H100!:2".to_owned(),
            device_name: "NVIDIA H100 SXM 80GB".to_owned(),
            driver_version: "580.95.05".to_owned(),
            cuda_version: "12.9".to_owned(),
            library: "cuBLASLt 120901".to_owned(),
            pcie_gen: Some(5),
            pcie_width: Some(16),
            shapes,
            checks: std::collections::BTreeMap::new(),
            samples_ms: std::collections::BTreeMap::new(),
            gpu_state: None,
        }
    }

    #[test]
    fn renders_gpu_state_per_gemm_dtype() {
        let round = |sm: f64, power: f64, capped: u32| RoundState {
            polls: 4,
            wall_ms: 200.0,
            sm_clock_mhz: Some(Span {
                min: sm,
                median: sm,
                max: 1980.0,
            }),
            mem_clock_mhz: Some(Span {
                min: 2619.0,
                median: 2619.0,
                max: 2619.0,
            }),
            power_w: Some(power),
            temperature_c: None,
            clock_limit_reasons: BTreeMap::from([("sw_power_cap".to_owned(), capped)]),
        };
        let state = GpuState {
            source: "NVML 12.580 via nvml-wrapper 0.13".to_owned(),
            poll_interval_ms: 5.0,
            max_sm_clock_mhz: 1980,
            max_mem_clock_mhz: 2619,
            power_limit_w: 700.0,
            warmup_ms: 1000.0,
            round_ms: 200.0,
            rounds: BTreeMap::from([(
                "gemm_fp16".to_owned(),
                vec![round(1980.0, 650.0, 0), round(1590.0, 699.6, 4)],
            )]),
        };
        let table = numbers(&h100_with(&[], Some(state)), false).unwrap();
        assert!(
            table.contains("| fp16 | 1590–1980 MHz | 2619 MHz | 675 W | — | sw_power_cap 50% |"),
            "{table}"
        );
        assert!(table.contains("Board limits: 700 W"), "{table}");
    }

    fn table_rows(table: &str) -> Vec<Vec<&str>> {
        table
            .lines()
            .filter(|line| line.starts_with("| ") && !line.starts_with("| ---"))
            .map(|line| {
                line.trim_matches('|')
                    .split(" | ")
                    .map(str::trim)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

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

    #[test]
    fn variation_without_repeats_says_how_to_record_them() {
        let table = variation(&[h100_with(&[], None)]);
        assert!(table.contains("--repeat 3"), "{table}");
    }

    #[test]
    fn variation_renders_spread_and_the_machine_of_each_run() {
        let mut profile = h100_with(&[], None);
        let hbm = profile.measured.as_ref().unwrap().hbm_gbps;
        profile.repeats = vec![
            repeat_with_hbm(&profile, 2000.0, ["GPU-aaaaaaaa-1", "GPU-bbbbbbbb-2"]),
            repeat_with_hbm(&profile, hbm, ["GPU-cccccccc-3", "GPU-dddddddd-4"]),
        ];
        let table = variation(&[profile]);
        let rows = table_rows(&table);
        assert_eq!(rows[0], ["napkin-bench number", "H100 (3 runs)"]);
        assert!(
            table.contains(&format!(
                "| HBM copy, GB/s | 2000–{} ({}%) ‡ |",
                fmt_sig(hbm, 3),
                fmt_sig((hbm / 2000.0 - 1.0) * 100.0, 2)
            )),
            "{table}"
        );
        assert!(table.contains("| Kernel launch, μs | "), "{table}");
        assert!(
            table.contains("| H100 | 2026-11-05T23:50:00Z | 580.95.05 | OCI | us-central | aaaaaaaa, bbbbbbbb |"),
            "{table}"
        );
        assert!(
            table
                .contains("(canonical) | 580.95.05 | not recorded | not recorded | not recorded |"),
            "{table}"
        );
    }

    #[test]
    fn unit_costs_multiply_one_gpus_time_by_gpus_billed_and_price() {
        let mut profile = h100_with(&[], None);
        profile.nvbandwidth = None;
        let measured = profile.measured.as_mut().unwrap();
        measured.hbm_gbps = 1000.0;
        measured.h2d_gbps = 100.0;
        measured.d2h_gbps = 50.0;
        measured.g2g_gbps = Some(500.0);
        measured.compute_tflops.insert(Dtype::Bf16, 1000.0);
        let mut record = *crate::price::find("modal-h100").unwrap();
        record.amount = 1.0;
        record.per = crate::price::Per::Second;
        let table = unit_costs(&[profile], &[record]);
        let rows = table_rows(&table);
        assert_eq!(rows[0], ["Work", "H100 `modal-h100`"]);
        assert_eq!(
            rows[1..],
            [
                vec!["Price per GPU-hour", "$3600"],
                // 1 TB / 1000 GB/s = 1 s at $1/s.
                vec!["1 TB of HBM traffic, read + write counted", "$1"],
                // The slower direction, 50 GB/s: 20 s.
                vec!["1 TB host ↔ device, pinned, slower direction", "$20"],
                // 1 TB / 500 GB/s = 2 s on each of two billed GPUs.
                vec!["1 TB GPU → GPU over NVLink, both GPUs billed", "$4"],
                // 10¹⁸ FLOP / 10¹⁵ FLOP/s = 1000 s.
                vec!["1 EFLOP of dense BF16 GEMM at the 8192³ rate", "$1000"],
            ]
        );
    }

    #[test]
    fn unit_costs_skip_records_for_unknown_gpus() {
        let mut record = *crate::price::find("modal-h100").unwrap();
        record.gpu = "t4";
        let table = unit_costs(crate::profile::builtin(), &[record]);
        assert_eq!(table_rows(&table)[0], ["Work"]);
    }

    #[test]
    fn host_limited_link_shows_effective_and_gpu_generations() {
        assert_eq!(
            host_link_label(Some("60 GB/s".to_owned()), Some(6), Some(5), Some(16)),
            "60 GB/s Gen5 x16 (GPU Gen6)"
        );
    }

    #[test]
    fn unbounded_host_link_shows_gpu_generation() {
        assert_eq!(
            host_link_label(Some("60 GB/s".to_owned()), Some(5), Some(5), Some(16)),
            "60 GB/s Gen5 x16"
        );
    }

    #[test]
    fn missing_link_info_keeps_only_bandwidth() {
        assert_eq!(
            host_link_label(Some("60 GB/s".to_owned()), None, None, None),
            "60 GB/s"
        );
    }
}
