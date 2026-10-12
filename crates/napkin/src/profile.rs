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
    fn ceiling_basis_has_no_host_or_launch_rates() -> Result<(), String> {
        let Some(b200) = find("b200") else {
            return Err("built-in b200 profile is missing".to_owned());
        };
        let rates = b200.rates(true);
        assert_eq!(rates.basis, Basis::Datasheet);
        assert!(rates.h2d.is_none());
        assert!(rates.launch_us.is_none());
        assert!(rates.h2d_latency_us.is_none());
        assert!(rates.g2g_latency_us.is_none());
        assert_eq!(rates.g2g.basis, Basis::Datasheet);
        assert!((rates.g2g.gbps - b200.ceiling.nvlink_gbps).abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn transfer_metrics_display_and_match_link_semantics() -> Result<(), String> {
        let Some(mut profile) = find("h100").cloned() else {
            return Err("built-in h100 profile is missing".to_owned());
        };
        let Some(measured) = profile.measured.as_mut() else {
            return Err("built-in h100 measured profile is missing".to_owned());
        };
        measured.pcie_gen = Some(4);

        let checks = profile.transfer_cross_check();
        assert_eq!(
            checks.iter().map(|check| check.metric).collect::<Vec<_>>(),
            vec![
                TransferMetric::H2d,
                TransferMetric::D2h,
                TransferMetric::Peer { src: 0, dst: 1 },
                TransferMetric::Peer { src: 1, dst: 0 },
            ]
        );
        assert_eq!(
            checks
                .iter()
                .map(|check| check.same_link)
                .collect::<Vec<_>>(),
            vec![false, false, true, true]
        );
        assert_eq!(TransferMetric::H2d.to_string(), "H2D");
        assert_eq!(TransferMetric::D2h.to_string(), "D2H");
        assert_eq!(
            TransferMetric::Peer { src: 0, dst: 1 }.to_string(),
            "GPU0→GPU1"
        );
        Ok(())
    }

    fn parsed_fixture() -> Result<Nvbandwidth, String> {
        Nvbandwidth::parse(
            include_str!("../tests/fixtures/h100.nvbandwidth.json"),
            include_str!("../tests/fixtures/h100.nvbandwidth.run.json"),
        )
    }

    fn parse_error(raw: &str, run: &str) -> String {
        match Nvbandwidth::parse(raw, run) {
            Ok(_) => "expected NVBandwidth parsing to fail".to_owned(),
            Err(error) => error,
        }
    }

    #[test]
    fn nvbandwidth_fixture_normalizes_directional_metrics_and_provenance() -> Result<(), String> {
        let evidence = parsed_fixture()?;
        assert_eq!(evidence.h2d(CopyPath::Ce), Some(55.5174));
        assert_eq!(evidence.d2h(CopyPath::Ce), Some(55.492));
        assert_eq!(evidence.d2h(CopyPath::Sm), Some(40.92));
        assert_eq!(evidence.peer(CopyPath::Ce, 0, 1), Some(395.085));
        assert_eq!(evidence.peer(CopyPath::Ce, 1, 0), Some(395.067));
        assert_eq!(evidence.g2g(CopyPath::Ce), Some(395.067));
        assert_eq!(evidence.buffer_bytes, 536_870_912);
        assert_eq!(evidence.samples, 7);
        assert_eq!(evidence.statistic, Statistic::Median);
        assert_eq!(evidence.cuda_runtime_version, "12.9");
        assert_eq!(evidence.cuda_driver_version, "13.0");
        assert_eq!(evidence.driver_version, "580.95.05");
        assert_eq!(evidence.gpu_request, "H100!:2");
        assert_eq!(evidence.version, "v0.10.0");
        assert!(evidence.verified);
        assert_eq!(evidence.bandwidths.len(), 16);
        assert!(evidence.bandwidths.iter().any(|rate| {
            rate.testcase == "device_to_device_memcpy_read_ce"
                && rate.src == Endpoint::Gpu(0)
                && rate.dst == Endpoint::Gpu(1)
                && rate.initiator == 1
                && (rate.gbps - 395.085).abs() < 1e-6
        }));
        Ok(())
    }

    #[test]
    fn nvbandwidth_rejects_non_decimal_units() {
        let raw = include_str!("../tests/fixtures/h100.nvbandwidth.json")
            .replace("bandwidth (GB/s)", "bandwidth (GiB/s)");
        let error = parse_error(
            &raw,
            include_str!("../tests/fixtures/h100.nvbandwidth.run.json"),
        );
        assert!(error.contains("host_to_device_memcpy_ce"), "{error}");
    }

    #[test]
    fn nvbandwidth_rejects_failed_testcase() {
        let raw = include_str!("../tests/fixtures/h100.nvbandwidth.json")
            .replace(r#""status" : "Passed""#, r#""status" : "Error""#);
        let error = parse_error(
            &raw,
            include_str!("../tests/fixtures/h100.nvbandwidth.run.json"),
        );
        assert!(error.contains("status"), "{error}");
    }

    #[test]
    fn nvbandwidth_rejects_top_level_error() {
        let raw = include_str!("../tests/fixtures/h100.nvbandwidth.json").replace(
            r#""version" :"#,
            "\"error\" : \"synthetic failure\",\n\t\t\"version\" :",
        );
        let error = parse_error(
            &raw,
            include_str!("../tests/fixtures/h100.nvbandwidth.run.json"),
        );
        assert!(error.contains("contains error"), "{error}");
    }

    #[test]
    fn nvbandwidth_rejects_disabled_verification_and_failed_run() {
        let raw = include_str!("../tests/fixtures/h100.nvbandwidth.json");
        let run = include_str!("../tests/fixtures/h100.nvbandwidth.run.json");
        let skipped = run.replace(
            r#""--disableAffinity","#,
            r#""--skipVerification", "--disableAffinity","#,
        );
        let error = parse_error(raw, &skipped);
        assert!(error.contains("disables verification"), "{error}");
        let failed = run.replace(r#""exit_code": 0,"#, r#""exit_code": 1,"#);
        let error = parse_error(raw, &failed);
        assert!(error.contains("exit_code"), "{error}");
    }

    #[test]
    fn nvbandwidth_identity_matches_profile_name_count_and_request() -> Result<(), String> {
        let evidence = parsed_fixture()?;
        let a100 = find("a100")
            .and_then(|profile| profile.measured.as_ref())
            .ok_or_else(|| "built-in a100 measured profile is missing".to_owned())?;
        assert!(matches!(
            evidence.check_identity(a100),
            Err(ref error) if error.contains("device_name")
        ));
        let h100 = find("h100")
            .and_then(|profile| profile.measured.as_ref())
            .ok_or_else(|| "built-in h100 measured profile is missing".to_owned())?;
        let mut changed = h100.clone();
        changed.device_count = 3;
        assert!(matches!(
            evidence.check_identity(&changed),
            Err(ref error) if error.contains("device_count")
        ));
        Ok(())
    }

    #[test]
    fn h100_transfer_rates_use_nvbandwidth_and_ceiling_override() -> Result<(), String> {
        let h100 = find("h100").ok_or_else(|| "built-in h100 profile is missing".to_owned())?;
        let rates = h100.rates(false);
        assert_eq!(rates.h2d.map(|rate| rate.basis), Some(Basis::Nvbandwidth));
        assert_eq!(rates.d2h.map(|rate| rate.basis), Some(Basis::Nvbandwidth));
        assert_eq!(rates.g2g.basis, Basis::Nvbandwidth);
        let ceiling = h100.rates(true);
        assert_eq!(ceiling.h2d, None);
        assert_eq!(ceiling.d2h, None);
        assert_eq!(ceiling.g2g.basis, Basis::Datasheet);
        Ok(())
    }
}
