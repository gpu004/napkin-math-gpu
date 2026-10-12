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
        let parsed = serde_json::from_str::<napkin_profile::Measured>(&json);
        assert!(
            parsed.is_ok(),
            "measured profile with missing PCIe fields parses"
        );
        if let Ok(measured) = parsed {
            assert_eq!(measured.pcie_gen, None);
            assert_eq!(measured.pcie_gen_host_max, None);
            assert_eq!(measured.pcie_width_max, None);
        }
    }

    #[test]
    fn pcie_effective_gen_is_capped_by_host_max() {
        let parsed = serde_json::from_str::<napkin_profile::Measured>(include_str!(
            "../../../profiles/h100.measured.json"
        ));
        assert!(parsed.is_ok(), "H100 measured profile fixture parses");
        if let Ok(mut measured) = parsed {
            measured.pcie_gen = Some(6);
            measured.pcie_gen_host_max = Some(5);
            assert_eq!(measured.pcie_effective_gen(), Some(5));

            measured.pcie_gen_host_max = None;
            assert_eq!(measured.pcie_effective_gen(), Some(6));

            measured.pcie_gen = None;
            assert_eq!(measured.pcie_effective_gen(), None);
        }
    }

    #[test]
    fn measured_json_without_gpu_state_parses() {
        let mut json: serde_json::Value =
            serde_json::from_str(include_str!("../../../profiles/h100.measured.json")).unwrap();
        json.as_object_mut().unwrap().remove("gpu_state");
        let measured: napkin_profile::Measured = serde_json::from_value(json).unwrap();
        assert!(measured.gpu_state.is_none());
    }

    #[test]
    fn round_spread_is_slowest_over_fastest() {
        let mut measured = find("h100").unwrap().measured.clone().unwrap();
        measured.samples_ms = BTreeMap::from([("gemm_fp16".to_owned(), vec![1.5, 1.2, 1.8, 1.2])]);
        let spread = measured.round_spread("gemm_fp16").unwrap();
        assert!((spread - 0.5).abs() < 1e-12);
        assert!(measured.round_spread("gemm_bf16").is_none());
        measured.samples_ms = BTreeMap::from([("gemm_fp16".to_owned(), vec![1.5])]);
        assert!(measured.round_spread("gemm_fp16").is_none());
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
