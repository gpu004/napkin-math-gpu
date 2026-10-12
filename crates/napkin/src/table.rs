//! Markdown table rendering for the README "Numbers" section: one
//! significant digit, sized for memorization rather than faux precision.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::price::PriceRecord;
use crate::profile::{
    Basis, CopyPath, CrossCheck, Dtype, GpuState, Measured, Profile, Rates, RoundState, Span,
    TransferRate,
};
use crate::units::{
    fmt_duration, fmt_gbps, fmt_intensity, fmt_money, fmt_sig, fmt_tflops, round_sig,
};

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
        .h2d_gbps
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
            let pcie = format!(
                "{} / {}",
                pcie_label(
                    evidence
                        .pcie_links
                        .first()
                        .map(|link| (link.r#gen, link.width)),
                ),
                pcie_label(measured.pcie_gen.zip(measured.pcie_width))
            );
            for check in profile.transfer_cross_check() {
                let (ce, sm) = cross_check_rates(evidence, &check);
                let measured = fmt_gbps(check.napkin_bench_gbps, 3);
                emit(
                    out,
                    &[
                        &profile.name,
                        check.metric,
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
        "H2D" => (profile.h2d(CopyPath::Ce), profile.h2d(CopyPath::Sm)),
        "D2H" => (profile.d2h(CopyPath::Ce), profile.d2h(CopyPath::Sm)),
        "GPU0→GPU1" => (
            profile.peer(CopyPath::Ce, 0, 1),
            profile.peer(CopyPath::Sm, 0, 1),
        ),
        "GPU1→GPU0" => (
            profile.peer(CopyPath::Ce, 1, 0),
            profile.peer(CopyPath::Sm, 1, 0),
        ),
        _ => (None, None),
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
    use crate::profile::find;

    fn h100_with(samples: &[(&str, Vec<f64>)], gpu_state: Option<GpuState>) -> Profile {
        let mut profile = find("h100").cloned().unwrap();
        let measured = profile.measured.as_mut().unwrap();
        measured.samples_ms = samples
            .iter()
            .map(|(key, rounds)| ((*key).to_owned(), rounds.clone()))
            .collect();
        measured.gpu_state = gpu_state;
        measured.gemm_shapes = None;
        profile
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
