//! The README's cost of moving 1 TB or doing 1 EFLOP on each priced GPU,
//! like the "$ / TiB read" table in sirupsen/napkin-math.

use std::fmt::Write as _;

use super::{SIG, emit_owned};
use crate::price::PriceRecord;
use crate::profile::{Dtype, Profile, Rates};
use crate::units::{fmt_money, fmt_money_sig};

/// One row of the unit-cost table: what is moved or computed, how many GPUs
/// the rental bills while it happens, and how long one GPU takes.
struct UnitCost {
    name: &'static str,
    gpus_billed: u32,
    seconds: fn(&Rates) -> Option<f64>,
}

const TB: f64 = 1e12;
const EFLOP: f64 = 1e18;

const UNIT_COSTS: [UnitCost; 4] = [
    UnitCost {
        name: "1 TB of HBM traffic, read + write counted",
        gpus_billed: 1,
        seconds: |rates| Some(TB / (rates.hbm_gbps * 1e9)),
    },
    UnitCost {
        name: "1 TB host ↔ device, pinned, slower direction",
        gpus_billed: 1,
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
        gpus_billed: 2,
        seconds: |rates| Some(TB / (rates.g2g.gbps * 1e9)),
    },
    UnitCost {
        name: "1 EFLOP of dense BF16 GEMM at the 8192³ rate",
        gpus_billed: 1,
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
                        f64::from(row.gpus_billed) * seconds * record.per_gpu_hour() / 3600.0,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::tests::{h100_with, table_rows};

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
}
