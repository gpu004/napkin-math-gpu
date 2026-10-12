//! Markdown table rendering for the README "Numbers" section: one
//! significant digit, sized for memorization rather than faux precision.

use crate::profile::{Dtype, Profile};
use crate::units::{fmt_duration, fmt_gbps, fmt_intensity, fmt_sig, fmt_tflops, round_sig};

const SIG: u32 = 1;

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

/// Renders the two "Numbers" tables (data movement + dense GEMM) for a
/// measured profile. Returns `None` when the profile has no measurement.
#[must_use]
pub fn numbers(profile: &Profile, exact: bool) -> Option<String> {
    let m = profile.measured.as_ref()?;
    let sig = if exact { 3 } else { SIG };
    let lat_us = |v: f64| fmt_duration(v / 1e6, sig);
    let gbps = |v: f64| fmt_gbps(v, sig);
    let mb = |v: f64| cell_time(1e6, v, exact);
    let gb = |v: f64| cell_time(1e9, v, exact);

    let mut out = String::new();
    emit(
        &mut out,
        &["Operation", "Latency", "Throughput", "1 MB", "1 GB"],
    );
    emit(&mut out, &["---", "---:", "---:", "---:", "---:"]);
    let rows: [[String; 5]; 8] = [
        [
            "Kernel launch (back-to-back)".to_owned(),
            lat_us(m.launch_us),
            String::new(),
            String::new(),
            String::new(),
        ],
        [
            "Launch + sync round trip".to_owned(),
            lat_us(m.sync_us),
            String::new(),
            String::new(),
            String::new(),
        ],
        [
            "Pinned host RAM copy".to_owned(),
            String::new(),
            gbps(m.host_copy_gbps),
            mb(m.host_copy_gbps),
            gb(m.host_copy_gbps),
        ],
        [
            "Host → device, pinned".to_owned(),
            lat_us(m.h2d_latency_us),
            gbps(m.h2d_gbps),
            mb(m.h2d_gbps),
            gb(m.h2d_gbps),
        ],
        [
            "Device → host, pinned".to_owned(),
            lat_us(m.d2h_latency_us),
            gbps(m.d2h_gbps),
            mb(m.d2h_gbps),
            gb(m.d2h_gbps),
        ],
        [
            "HBM random read (pointer chase)".to_owned(),
            format!("{} ns", fmt_sig(m.hbm_latency_ns, sig)),
            String::new(),
            String::new(),
            String::new(),
        ],
        [
            "HBM copy (read + write)".to_owned(),
            String::new(),
            gbps(m.hbm_gbps),
            mb(m.hbm_gbps),
            gb(m.hbm_gbps),
        ],
        [
            "Elementwise add".to_owned(),
            String::new(),
            gbps(m.elementwise_gbps),
            mb(m.elementwise_gbps),
            gb(m.elementwise_gbps),
        ],
    ];
    for r in &rows {
        emit(&mut out, &[&r[0], &r[1], &r[2], &r[3], &r[4]]);
    }
    emit(
        &mut out,
        &[
            "GPU ↔ GPU, NVLink one-way",
            &m.g2g_latency_us.map_or_else(String::new, &lat_us),
            &m.g2g_gbps.map_or_else(String::new, &gbps),
            &m.g2g_gbps.map_or_else(String::new, &mb),
            &m.g2g_gbps.map_or_else(String::new, &gb),
        ],
    );

    out.push('\n');
    emit(
        &mut out,
        &[
            "Dense GEMM",
            "Throughput",
            "1 GFLOP",
            "1 TFLOP",
            "Ridge point",
        ],
    );
    emit(&mut out, &["---", "---:", "---:", "---:", "---:"]);
    for dtype in [Dtype::Fp16, Dtype::Bf16, Dtype::Tf32, Dtype::Fp32] {
        if let Some(&tflops) = m.compute_tflops.get(&dtype) {
            let rate = if exact {
                tflops
            } else {
                round_sig(tflops, SIG)
            };
            let ridge = fmt_intensity(tflops * 1000.0 / m.hbm_gbps, if exact { 4 } else { 2 });
            let ts = |flops: f64| fmt_duration(flops / (rate * 1e12), sig);
            emit(
                &mut out,
                &[
                    &dtype.to_string(),
                    &fmt_tflops(tflops, sig),
                    &ts(1e9),
                    &ts(1e12),
                    &ridge,
                ],
            );
        }
    }
    Some(out)
}

fn basis_label(profile: &Profile) -> String {
    match &profile.measured {
        Some(m) => format!("measured {}", &m.captured_at[..10.min(m.captured_at.len())]),
        None => "datasheet".to_owned(),
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
        ],
    );
    emit(
        &mut out,
        &[
            "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---",
        ],
    );
    for p in profiles {
        let rates = p.rates(false);
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
                &rates
                    .host_copy_gbps
                    .map_or_else(|| "—".to_owned(), |v| fmt_gbps(v, sig)),
                &rates
                    .launch_us
                    .map_or_else(|| "—".to_owned(), |v| fmt_duration(v / 1e6, sig)),
                &basis_label(p),
            ],
        );
    }

    let measured: Vec<&Profile> = profiles.iter().filter(|p| p.measured.is_some()).collect();
    if !measured.is_empty() {
        out.push('\n');
        emit(
            &mut out,
            &["GPU", "HBM measured / ceiling", "BF16 measured / ceiling"],
        );
        emit(&mut out, &["---", "---:", "---:"]);
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
                &mut out,
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
    out
}
