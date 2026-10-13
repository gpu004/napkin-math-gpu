//! GPU napkin math: estimate workload performance from datasheet ceilings and
//! measured GPU profiles, in the spirit of sirupsen/napkin-math.

#![forbid(unsafe_code)]

pub mod collective;
pub mod cost;
pub mod estimate;
pub mod fit;
pub mod llm;
pub mod price;
pub mod profile;
pub mod table;
pub mod units;
pub mod validation;

/// Merge a successful benchmark run's GEMM shape suite into an existing
/// measured profile without replacing its scalar measurements.
///
/// # Errors
/// Returns an error if either file cannot be read, the typed merge fails, or
/// the merged profile cannot be written.
pub fn merge_shapes(
    run_path: &std::path::Path,
    profile_path: &std::path::Path,
) -> Result<(), String> {
    let run_json = std::fs::read_to_string(run_path)
        .map_err(|error| format!("cannot read {}: {error}", run_path.display()))?;
    let run: profile::Measured = serde_json::from_str(&run_json)
        .map_err(|error| format!("cannot parse {}: {error}", run_path.display()))?;
    let profile_json = std::fs::read_to_string(profile_path)
        .map_err(|error| format!("cannot read {}: {error}", profile_path.display()))?;
    let merged = merge_gemm_shapes_json(&profile_json, run)
        .map_err(|error| format!("cannot merge {}: {error}", profile_path.display()))?;
    std::fs::write(profile_path, merged)
        .map_err(|error| format!("cannot write {}: {error}", profile_path.display()))
}

/// Merges a run's shape suite while preserving every existing profile byte.
///
/// # Errors
/// Returns an error if the profile or run is invalid, the typed merge fails,
/// or the byte-preserving splice does not produce the expected profile.
pub fn merge_gemm_shapes_json(
    profile_text: &str,
    run: profile::Measured,
) -> Result<String, String> {
    let measured: profile::Measured = serde_json::from_str(profile_text)
        .map_err(|error| format!("cannot parse profile JSON: {error}"))?;
    if measured.gemm_shapes.is_some() {
        return Err(
            "profile already has a GEMM shape suite; restore the profile from main first"
                .to_owned(),
        );
    }
    let suite = run
        .gemm_shapes
        .clone()
        .ok_or_else(|| "run has no GEMM shape suite".to_owned())?;
    let mut expected = measured.clone();
    expected
        .merge_gemm_shapes(run)
        .map_err(|error| format!("cannot merge GEMM shape suite: {error}"))?;

    let trailing_newline = profile_text.ends_with('\n');
    let profile_body = if trailing_newline {
        profile_text.strip_suffix('\n').unwrap_or(profile_text)
    } else {
        profile_text
    };
    let prefix = profile_body.strip_suffix("\n}").ok_or_else(|| {
        "profile JSON must end with `\\n}` and an optional trailing newline".to_owned()
    })?;
    let suite_json = serde_json::to_string_pretty(&suite)
        .map_err(|error| format!("cannot serialize GEMM shape suite: {error}"))?
        .lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let trailing_newline = if trailing_newline { "\n" } else { "" };
    let merged = format!("{prefix},\n  \"gemm_shapes\": {suite_json}\n}}{trailing_newline}");
    let parsed: profile::Measured = serde_json::from_str(&merged)
        .map_err(|error| format!("cannot parse merged profile JSON: {error}"))?;
    let expected_value = serde_json::to_value(expected)
        .map_err(|error| format!("cannot compare merged profile: {error}"))?;
    let parsed_value = serde_json::to_value(parsed)
        .map_err(|error| format!("cannot compare merged profile: {error}"))?;
    if parsed_value != expected_value {
        return Err("merged profile JSON differs from the typed merge".to_owned());
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDirectory(PathBuf);

    impl TempDirectory {
        fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let path = std::env::temp_dir().join(format!(
                "napkin-merge-shapes-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn file_merge_replaces_only_the_gemm_shape_suite() -> Result<(), Box<dyn std::error::Error>> {
        let Some(mut original) = profile::find("h100").and_then(|gpu| gpu.measured.clone()) else {
            return Err("H100 measured profile is missing".into());
        };
        original.gemm_shapes = None;
        let mut run = original.clone();
        run.captured_at = "2026-10-15T00:00:00Z".to_owned();
        run.gemm_shapes = Some(profile::GemmShapeSuite {
            captured_at: "2026-10-15T00:00:00Z".to_owned(),
            gpu_request: "H100!:2".to_owned(),
            device_name: original.device_name.clone(),
            driver_version: "580.95.05".to_owned(),
            cuda_version: "12.9".to_owned(),
            library: "cuBLASLt 120901".to_owned(),
            pcie_gen: Some(5),
            pcie_width: Some(16),
            shapes: Vec::new(),
            checks: std::collections::BTreeMap::from([("shape_check".to_owned(), true)]),
            samples_ms: std::collections::BTreeMap::new(),
            gpu_state: None,
        });

        let temp = TempDirectory::new()?;
        let run_path = temp.0.join("run.json");
        let profile_path = temp.0.join("profile.json");
        std::fs::write(
            &run_path,
            format!("{}\n", serde_json::to_string_pretty(&run)?),
        )?;
        std::fs::write(
            &profile_path,
            format!("{}\n", serde_json::to_string_pretty(&original)?),
        )?;

        merge_shapes(&run_path, &profile_path)?;

        let merged_json = std::fs::read_to_string(&profile_path)?;
        assert!(merged_json.ends_with('\n'));
        let merged: profile::Measured = serde_json::from_str(&merged_json)?;
        let mut merged_without_suite = merged.clone();
        merged_without_suite.gemm_shapes = None;
        let mut original_without_suite = original;
        original_without_suite.gemm_shapes = None;
        assert_eq!(
            format!("{merged_without_suite:?}"),
            format!("{original_without_suite:?}")
        );
        assert_eq!(
            format!("{:?}", merged.gemm_shapes),
            format!("{:?}", run.gemm_shapes)
        );
        Ok(())
    }

    fn fixture_run(profile: &profile::Measured) -> profile::Measured {
        let mut run = profile.clone();
        run.checks = std::collections::BTreeMap::from([("run_check".to_owned(), true)]);
        run.gemm_shapes = Some(profile::GemmShapeSuite {
            captured_at: "2026-10-15T00:00:00Z".to_owned(),
            gpu_request: profile.gpu_request.clone(),
            device_name: profile.device_name.clone(),
            driver_version: profile.driver_version.clone(),
            cuda_version: profile.cuda_version.clone(),
            library: "cuBLASLt fixture".to_owned(),
            pcie_gen: profile.pcie_gen,
            pcie_width: profile.pcie_width,
            shapes: Vec::new(),
            checks: std::collections::BTreeMap::from([("shape_check".to_owned(), true)]),
            samples_ms: std::collections::BTreeMap::new(),
            gpu_state: None,
        });
        run
    }

    #[test]
    fn json_merge_preserves_each_measured_profile() -> Result<(), Box<dyn std::error::Error>> {
        let profile_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../profiles");
        let mut paths = std::fs::read_dir(profile_dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().ends_with(".measured.json"))
            })
            .collect::<Vec<_>>();
        paths.sort();
        assert!(!paths.is_empty(), "no committed measured profiles found");

        let expected_profiles = paths.len();
        let mut merged_profiles = 0;
        for path in paths {
            let source = std::fs::read_to_string(&path)?;
            let mut original: profile::Measured = serde_json::from_str(&source)?;
            let trailing_newline = source.ends_with('\n');
            let body = if trailing_newline {
                source.strip_suffix('\n').unwrap_or(&source)
            } else {
                &source
            };
            let mut profile_text = source.clone();
            if original.gemm_shapes.take().is_some() {
                let prefix = body
                    .strip_suffix("\n}")
                    .ok_or("profile does not end in a newline and closing brace")?;
                let suite_start = prefix
                    .rfind(",\n  \"gemm_shapes\":")
                    .ok_or("profile suite field is missing from its JSON text")?;
                profile_text = format!(
                    "{}\n}}{}",
                    &prefix[..suite_start],
                    if trailing_newline { "\n" } else { "" }
                );
            }
            let prefix = profile_text
                .strip_suffix(if trailing_newline { "\n" } else { "" })
                .unwrap_or(&profile_text)
                .strip_suffix("\n}")
                .ok_or("profile does not end in a newline and closing brace")?;
            let mut original_without_suite = original;
            original_without_suite.gemm_shapes = None;
            let merged =
                merge_gemm_shapes_json(&profile_text, fixture_run(&original_without_suite))?;
            assert!(merged.starts_with(prefix), "{}", path.display());
            assert_eq!(
                merged.ends_with('\n'),
                trailing_newline,
                "{}",
                path.display()
            );

            let mut merged_profile: profile::Measured = serde_json::from_str(&merged)?;
            assert!(merged_profile.gemm_shapes.is_some(), "{}", path.display());
            merged_profile.gemm_shapes = None;
            assert_eq!(
                serde_json::to_value(merged_profile)?,
                serde_json::to_value(original_without_suite)?,
                "{}",
                path.display()
            );
            merged_profiles += 1;
        }
        assert_eq!(merged_profiles, expected_profiles);
        Ok(())
    }

    #[test]
    fn json_merge_rejects_a_profile_that_already_has_a_suite()
    -> Result<(), Box<dyn std::error::Error>> {
        let Some(mut original) = profile::find("h100").and_then(|gpu| gpu.measured.clone()) else {
            return Err("H100 measured profile is missing".into());
        };
        original.gemm_shapes = fixture_run(&original).gemm_shapes;
        let profile_text = format!("{}\n", serde_json::to_string_pretty(&original)?);
        let error = merge_gemm_shapes_json(&profile_text, fixture_run(&original))
            .err()
            .ok_or("merge unexpectedly accepted an existing shape suite")?;
        assert!(error.contains("restore the profile from main first"));
        Ok(())
    }
}
