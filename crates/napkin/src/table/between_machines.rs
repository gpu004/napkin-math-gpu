//! The README's "Between machines" table: how far each headline napkin-bench
//! number moves between the canonical run and its repeats, then which machine
//! each run used.

use std::fmt::Write as _;

use super::{emit, emit_owned};
use crate::profile::{Host, Measured, Profile};
use crate::units::fmt_sig;
use crate::variation::{self, Spread};

/// A GPU with repeats, and the canonical run they are compared against.
/// `build.rs` rejects repeats without a canonical run, so every profile with
/// repeats has one.
struct Repeated<'a> {
    profile: &'a Profile,
    canonical: &'a Measured,
}

/// Renders the spread table and host table. Only GPUs with at least one
/// repeat get a column.
#[must_use]
pub fn variation(profiles: &[Profile]) -> String {
    let repeated: Vec<Repeated<'_>> = profiles
        .iter()
        .filter(|profile| !profile.repeats.is_empty())
        .filter_map(|profile| {
            Some(Repeated {
                profile,
                canonical: profile.measured.as_ref()?,
            })
        })
        .collect();
    if repeated.is_empty() {
        return "No repeated runs yet: `modal run modal/bench.py --gpu h100 --repeat 3` records \
                some.\n"
            .to_owned();
    }
    let mut out = String::new();
    let mut header = vec!["napkin-bench number".to_owned()];
    header.extend(repeated.iter().map(|gpu| {
        format!(
            "{} ({} runs)",
            gpu.profile.id.to_uppercase(),
            gpu.profile.repeats.len() + 1
        )
    }));
    emit_owned(&mut out, &header);
    let mut align = vec!["---".to_owned()];
    align.extend(repeated.iter().map(|_| "---:".to_owned()));
    emit_owned(&mut out, &align);
    for metric in &variation::METRICS {
        let mut cells = vec![format!("{}, {}", metric.name, metric.unit.label())];
        cells.extend(
            repeated
                .iter()
                .map(|gpu| spread_cell(variation::spread(gpu.profile, metric))),
        );
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
    append_hosts(&mut out, &repeated);
    out.push_str(
        "\nGPU UUIDs are unique per physical GPU, shown by their first 8 hex digits, so runs with \
         different UUIDs used different GPUs. Modal shows no machine ID inside a container: two \
         2-GPU runs in the same provider and region may still share one 8-GPU node.\n",
    );
    out
}

fn spread_cell(spread: Option<Spread>) -> String {
    spread.map_or_else(
        || "—".to_owned(),
        |spread| {
            let mark = if spread.varies_by_machine() {
                " ‡"
            } else {
                ""
            };
            format!(
                "{}–{} ({}%){mark}",
                fmt_sig(spread.low, 3),
                fmt_sig(spread.high, 3),
                fmt_sig(spread.fraction() * 100.0, 2),
            )
        },
    )
}

/// One row per run: when it ran, on which driver, and on which machine, as
/// far as Modal lets a container tell. The canonical runs predate host
/// recording.
fn append_hosts(out: &mut String, repeated: &[Repeated<'_>]) {
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
    for gpu in repeated {
        let name = gpu.profile.id.to_uppercase();
        let not_recorded = "not recorded";
        emit(
            out,
            &[
                &name,
                &format!("{} (canonical)", gpu.canonical.captured_at),
                &gpu.canonical.driver_version,
                not_recorded,
                not_recorded,
                not_recorded,
            ],
        );
        for repeat in &gpu.profile.repeats {
            emit(
                out,
                &[
                    &name,
                    &repeat.measured.captured_at,
                    &repeat.measured.driver_version,
                    provider(&repeat.host),
                    &repeat.host.region,
                    &short_uuids(&repeat.host),
                ],
            );
        }
    }
}

/// `"CLOUD_PROVIDER_OCI"` as `"OCI"`.
fn provider(host: &Host) -> &str {
    host.cloud_provider
        .strip_prefix("CLOUD_PROVIDER_")
        .unwrap_or(&host.cloud_provider)
}

/// `"GPU-77ac2056-60c3-…"` as `"77ac2056"`, joined: enough to tell GPUs apart
/// in a table.
fn short_uuids(host: &Host) -> String {
    host.gpu_uuids
        .iter()
        .map(|uuid| {
            let hex = uuid.strip_prefix("GPU-").unwrap_or(uuid);
            hex.get(..8).unwrap_or(hex)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::test_support::{repeat_with_hbm, table_rows};
    use crate::table::tests::h100_with;

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
}
