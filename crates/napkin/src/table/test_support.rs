//! Fixtures shared by the renderer modules' tests.

use crate::profile::{Host, Profile, Repeat};

/// A repeat of `profile`'s canonical run on another machine, identical but
/// for its HBM copy rate, so a test controls exactly one spread.
pub(super) fn repeat_with_hbm(profile: &Profile, hbm_gbps: f64, uuids: [&str; 2]) -> Repeat {
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

/// A markdown table's header and body rows as trimmed cells.
pub(super) fn table_rows(table: &str) -> Vec<Vec<&str>> {
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
