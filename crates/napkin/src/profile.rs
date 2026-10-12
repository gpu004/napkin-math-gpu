//! Built-in GPU profiles, generated from `profiles/*.json` by `build.rs` and validated at build time;
//! types come from `napkin-profile`.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use napkin_profile::collective::{Collective, CollectiveEvidence, Curve, CurvePoint, Topology};

pub use napkin_profile::*;

static BUILTIN: LazyLock<Vec<Profile>> = LazyLock::new(generated_profiles);
static COLLECTIVES: LazyLock<Vec<CollectiveEvidence>> = LazyLock::new(generated_collectives);

/// Returns the built-in profiles in a100, h100, b200, b300 order.
pub fn builtin() -> &'static [Profile] {
    BUILTIN.as_slice()
}

/// Returns the built-in NCCL Tests collective evidence.
pub fn collectives() -> &'static [CollectiveEvidence] {
    COLLECTIVES.as_slice()
}

/// Finds a built-in profile by its identifier.
pub fn find(id: &str) -> Option<&'static Profile> {
    BUILTIN.iter().find(|profile| profile.id == id)
}

include!(concat!(env!("OUT_DIR"), "/profiles.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_profiles_keep_datasheet_order() {
        assert_eq!(
            builtin()
                .iter()
                .map(|profile| profile.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a100", "h100", "b200", "b300"]
        );
    }

    #[test]
    fn find_returns_none_for_unknown_id() {
        assert!(find("t4").is_none());
        assert!(find("b200").is_some());
    }

    #[test]
    fn measured_json_without_pcie_fields_parses() {
        let json = include_str!("../../../profiles/h100.measured.json")
            .replace(r#""pcie_gen": 5,"#, "")
            .replace(r#""pcie_gen_max": 5,"#, "")
            .replace(r#""pcie_width": 16,"#, "")
            .replace(r#""pcie_width_max": 16,"#, "");
        let measured: napkin_profile::Measured = serde_json::from_str(&json).unwrap();
        assert_eq!(measured.pcie_gen, None);
        assert_eq!(measured.pcie_width_max, None);
    }

    #[test]
    fn all_builtin_gpus_have_measured_profiles() {
        for id in ["a100", "h100", "b200", "b300"] {
            assert!(
                find(id).is_some_and(|profile| profile.measured.is_some()),
                "{id}"
            );
        }
    }

    #[test]
    fn ceiling_basis_has_no_host_or_launch_rates() {
        let b200 = find("b200").unwrap();
        let rates = b200.rates(true);
        assert_eq!(rates.basis, Basis::Ceiling);
        assert!(rates.h2d_gbps.is_none());
        assert!(rates.launch_us.is_none());
        assert!(rates.h2d_latency_us.is_none());
        assert!(rates.g2g_latency_us.is_none());
        assert!((rates.g2g_gbps - b200.ceiling.nvlink_gbps).abs() < f64::EPSILON);
    }
}
