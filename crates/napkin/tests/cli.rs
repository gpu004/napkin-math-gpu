use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn napkin(args: &[&str]) -> (String, String, i32) {
    let out = match Command::new(env!("CARGO_BIN_EXE_napkin"))
        .args(args)
        .output()
    {
        Ok(out) => out,
        Err(e) => return (String::new(), format!("napkin failed to run: {e}"), -1),
    };
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn gpus_lists_all_profiles() {
    let (stdout, _, code) = napkin(&["gpus"]);
    assert_eq!(code, 0);
    for id in ["a100", "h100", "b200", "b300"] {
        assert!(stdout.contains(id), "{stdout}");
    }
}

#[test]
fn merge_shapes_command_accepts_the_run_and_profile_paths() {
    let (stdout, stderr, code) = napkin(&["merge-shapes", "--help"]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("--from <RUN>"), "{stdout}");
    assert!(stdout.contains("<PROFILE>"), "{stdout}");
}

#[test]
fn estimate_runs_on_datasheet_gpu() {
    let (stdout, stderr, code) = napkin(&[
        "estimate", "--gpu", "b200", "--dtype", "bf16", "--flops", "200T", "--bytes", "1TB",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("B200"), "{stdout}");
    assert!(stdout.contains("total"), "{stdout}");
}

#[test]
fn gemm_estimate_reports_derived_work_and_unmatched_shape() {
    let (stdout, stderr, code) = napkin(&[
        "estimate",
        "--gpu",
        "h100",
        "--dtype",
        "bf16",
        "--gemm",
        "4096x4096x4096",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(
        stdout.contains("GEMM FLOPs  2 × 4096 × 4096 × 4096"),
        "{stdout}"
    );
    assert!(stdout.contains("minimum HBM (A+B+C)"), "{stdout}");
    assert!(
        stdout.contains("no measured bf16 GEMM shape 4096x4096x4096"),
        "{stdout}"
    );
    assert!(
        stdout.contains("used the measured 8192³ GemmEx peak, a large-GEMM ceiling"),
        "{stdout}"
    );
    assert!(stdout.contains("measured shapes:"), "{stdout}");
    assert!(stdout.contains("; no interpolation"), "{stdout}");
}

#[test]
fn tflops_override_is_reported_as_the_compute_source() {
    let (stdout, stderr, code) = napkin(&[
        "estimate", "--gpu", "h100", "--dtype", "bf16", "--tflops", "300", "--flops", "1e12",
        "--bytes", "1e9",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(
        stdout.contains("(compute: --tflops override at 300 TFLOP/s)"),
        "{stdout}"
    );
}

#[test]
fn gemm_estimate_rejects_bad_dimensions_and_ceiling() {
    let (_, stderr, code) = napkin(&["estimate", "--gpu", "h100", "--gemm", "0x2x3"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("three positive dimensions"), "{stderr}");
    let (_, stderr, code) = napkin(&["estimate", "--gpu", "h100", "--gemm", "8x8x8", "--ceiling"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("cannot be used with"), "{stderr}");
    let (_, stderr, code) = napkin(&[
        "estimate", "--gpu", "h100", "--gemm", "8x8x8", "--tflops", "300", "--flops", "1e12",
        "--bytes", "1e9",
    ]);
    assert_eq!(code, 2);
    assert!(stderr.contains("cannot be used with"), "{stderr}");
}

#[test]
fn estimate_rejects_h2d_on_datasheet_gpu() {
    let (_, stderr, code) = napkin(&[
        "estimate",
        "--gpu",
        "b200",
        "--dtype",
        "bf16",
        "--flops",
        "200T",
        "--bytes",
        "1TB",
        "--h2d",
        "2GB",
        "--ceiling",
    ]);
    assert_eq!(code, 2);
    assert!(stderr.contains("no measured h2d bandwidth"), "{stderr}");
}

#[test]
fn estimate_rejects_garbage_units() {
    let (_, _, code) = napkin(&[
        "estimate", "--gpu", "h100", "--dtype", "bf16", "--flops", "oops", "--bytes", "1TB",
    ]);
    assert_eq!(code, 2);
}

#[test]
fn fit_reports_gpu_count() {
    let (stdout, _, code) = napkin(&["fit", "--gpu", "h100", "--params", "70B", "--bits", "16"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("GPUs"), "{stdout}");
}

#[test]
fn fit_reports_replicated_per_gpu_memory() {
    let (stdout, _, code) = napkin(&[
        "fit",
        "--gpu",
        "h100",
        "--params",
        "70B",
        "--bits",
        "16",
        "--per-gpu",
        "2GB",
    ]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("per GPU     2 GB (replicated on every GPU)"),
        "{stdout}"
    );
    assert!(
        stdout.contains(
            "fits        no — needs at least 2 GPUs (weights + extra sharded, per-GPU replicated)"
        ),
        "{stdout}"
    );
}

#[test]
fn llm_explains_a_hypothetical_70b_recipe() {
    let (stdout, stderr, code) = napkin(&[
        "llm",
        "--gpu",
        "h100",
        "--layers",
        "80",
        "--hidden",
        "8192",
        "--heads",
        "64",
        "--kv-heads",
        "8",
        "--head-dim",
        "128",
        "--params",
        "70B",
        "--batch",
        "8",
        "--prompt",
        "2048",
        "--generated",
        "512",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("KV cache"), "{stdout}");
    assert!(stdout.contains("$ napkin estimate"), "{stdout}");
    assert!(stdout.contains("hypothetical"), "{stdout}");
    assert!(
        stdout.lines().all(|line| line.chars().count() <= 100),
        "{stdout}"
    );
}

#[test]
fn llm_reports_when_a_small_model_fits() {
    let (stdout, stderr, code) = napkin(&[
        "llm",
        "--gpu",
        "h100",
        "--layers",
        "32",
        "--hidden",
        "4096",
        "--heads",
        "32",
        "--kv-heads",
        "8",
        "--head-dim",
        "128",
        "--params",
        "8B",
        "--prompt",
        "1024",
        "--generated",
        "128",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("fits on one H100 SXM 80GB"), "{stdout}");
    assert!(!stdout.contains("hypothetical"), "{stdout}");
}

#[test]
fn compare_prints_table() {
    let (stdout, _, code) = napkin(&["compare"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("| GPU |"), "{stdout}");
    assert!(stdout.contains("CE / napkin-bench"), "{stdout}");
    assert!(stdout.contains("n/a (NVLink)"), "{stdout}");
}

#[test]
fn cost_requires_exactly_one_price_source() {
    let (_, _, code) = napkin(&["cost", "--gpus", "2", "--time", "10s"]);
    assert_eq!(code, 2);
    let (_, _, code) = napkin(&[
        "cost",
        "--gpus",
        "2",
        "--time",
        "10s",
        "--price",
        "3",
        "--offer",
        "modal-h100",
    ]);
    assert_eq!(code, 2);
}

#[test]
fn cost_formats_supplied_price_and_workload() {
    let (stdout, stderr, code) = napkin(&["cost", "--gpus", "2", "--price", "3", "--time", "10s"]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("cost        $0.0167 per job"), "{stdout}");
    assert!(!stdout.contains("/$"), "{stdout}");
    let (stdout, stderr, code) = napkin(&[
        "cost",
        "--gpus",
        "2",
        "--price",
        "3",
        "--time",
        "10s",
        "--throughput",
        "2.5K",
        "--unit",
        "tokens",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("tokens/$"), "{stdout}");
}

#[test]
fn cost_offer_and_invalid_inputs() {
    let (stdout, stderr, code) = napkin(&[
        "cost",
        "--gpus",
        "2",
        "--time",
        "10s",
        "--offer",
        "modal-h100",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("2026-10-12"), "{stdout}");
    assert!(stdout.contains("https://modal.com/pricing"), "{stdout}");
    let (_, stderr, code) = napkin(&["cost", "--gpus", "1", "--time", "10s", "--offer", "unknown"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("no price record"), "{stderr}");
    let (_, _, code) = napkin(&["cost", "--gpus", "0", "--time", "10s", "--price", "3"]);
    assert_eq!(code, 2);
}

#[test]
fn prices_lists_all_builtin_records() {
    let (stdout, stderr, code) = napkin(&["prices"]);
    assert_eq!(code, 0, "{stderr}");
    for id in ["modal-a100", "modal-h100", "modal-b200", "modal-b300"] {
        assert!(stdout.contains(id), "{stdout}");
    }
}

#[test]
fn utilization_reverses_peak_per_dollar_ranking() {
    for (args, expected) in [
        (
            &[
                "cost",
                "--gpus",
                "1",
                "--price",
                "3.95",
                "--time",
                "1400s",
                "--granularity",
                "1s",
            ][..],
            "cost        $1.54 per job",
        ),
        (
            &[
                "cost",
                "--gpus",
                "1",
                "--price",
                "6.25",
                "--time",
                "630s",
                "--granularity",
                "1s",
            ],
            "cost        $1.09 per job",
        ),
        (
            &[
                "cost",
                "--gpus",
                "1",
                "--price",
                "3.95",
                "--time",
                "1400s",
                "--utilization",
                "0.5",
                "--granularity",
                "1s",
            ],
            "cost        $3.07 per job",
        ),
        (
            &[
                "cost",
                "--gpus",
                "1",
                "--price",
                "6.25",
                "--time",
                "630s",
                "--utilization",
                "0.3",
                "--granularity",
                "1s",
            ],
            "cost        $3.65 per job",
        ),
    ] {
        let (stdout, stderr, code) = napkin(args);
        assert_eq!(code, 0, "{stderr}");
        assert!(stdout.contains(expected), "{stdout}");
    }
}

struct TempReadme(PathBuf);

impl Drop for TempReadme {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn stale_readme(name: &str, content: &str) -> std::io::Result<TempReadme> {
    let path = std::env::temp_dir().join(format!("napkin-readme-{name}-{}.md", std::process::id()));
    fs::write(&path, content).map(|()| TempReadme(path))
}

#[test]
fn readme_check_and_write_conflict() {
    let (_, _, code) = napkin(&["readme", "--check", "--write"]);
    assert_eq!(code, 2);
}

#[test]
fn readme_check_preserves_stale_file() -> Result<(), Box<dyn std::error::Error>> {
    let path = stale_readme(
        "check",
        "<!-- napkin compare -->\nstale\n<!-- /napkin -->\n",
    )?;
    let original = fs::read_to_string(&path.0)?;
    let path_arg = path.0.to_string_lossy();
    let (_, _, code) = napkin(&["readme", "--check", path_arg.as_ref()]);
    assert_ne!(code, 0);
    assert_eq!(fs::read_to_string(&path.0)?, original);
    Ok(())
}

#[test]
fn validate_renders_committed_case_numbers() {
    let (stdout, stderr, code) = napkin(&["validate"]);
    assert_eq!(code, 0, "{stderr}");
    for row in [
        "| [h100-gemm-bf16](#h100-gemm-bf16) | 11 ms | compute-bound | 11 ms | 1.01 | 0.80–1.25 | within |",
        "| [h100-pcie-roundtrip](#h100-pcie-roundtrip) | 620 ms | d2h-bound | 3.7 s | 6.02 | 0.90–1.10 | outside (slower) |",
        "| [h100-tiny-kernels](#h100-tiny-kernels) | 17 ms | launch-bound | 62 ms | 3.64 | 0.50–2.00 | outside (slower) |",
    ] {
        assert!(
            stdout.contains(row),
            "missing literal row {row:?}:\n{stdout}"
        );
    }
    for literal in [
        "89 samples × 10 runs per sample",
        "14.4814 GB/s",
        "6.76848 GB/s",
        "| profiled_cuda_kernels | 10000 |",
    ] {
        assert!(stdout.contains(literal), "missing {literal:?}:\n{stdout}");
    }
}

#[test]
fn validate_check_preserves_stale_file() -> Result<(), Box<dyn std::error::Error>> {
    let path = stale_readme("validation-check", "stale report\n")?;
    let original = fs::read_to_string(&path.0)?;
    let path_arg = path.0.to_string_lossy();
    let (_, stderr, code) = napkin(&["validate", "--check", path_arg.as_ref()]);
    assert_ne!(code, 0, "{stderr}");
    assert!(stderr.contains("validation report is stale"), "{stderr}");
    assert_eq!(fs::read_to_string(&path.0)?, original);
    Ok(())
}

#[test]
fn validate_write_generates_checkable_report() -> Result<(), Box<dyn std::error::Error>> {
    let path = stale_readme("validation-write", "")?;
    let path_arg = path.0.to_string_lossy();
    let (_, stderr, code) = napkin(&["validate", "--write", path_arg.as_ref()]);
    assert_eq!(code, 0, "{stderr}");
    let report = fs::read_to_string(&path.0)?;
    assert!(report.starts_with("# Validation: napkin estimates against real runs\n"));
    let (_, stderr, code) = napkin(&["validate", "--check", path_arg.as_ref()]);
    assert_eq!(code, 0, "{stderr}");
    Ok(())
}

#[test]
fn readme_rewrites_stale_file() -> Result<(), Box<dyn std::error::Error>> {
    let path = stale_readme(
        "rewrite",
        "<!-- napkin compare -->\nstale\n<!-- /napkin -->\n",
    )?;
    let path_arg = path.0.to_string_lossy();
    let (_, _, code) = napkin(&["readme", path_arg.as_ref()]);
    assert_eq!(code, 0);
    let (_, _, code) = napkin(&["readme", "--check", path_arg.as_ref()]);
    assert_eq!(code, 0);
    Ok(())
}

#[test]
fn readme_checks_every_path() -> Result<(), Box<dyn std::error::Error>> {
    let current = stale_readme("current", "no markers\n")?;
    let stale1 = stale_readme(
        "stale1",
        "<!-- napkin compare -->\nstale\n<!-- /napkin -->\n",
    )?;
    let stale2 = stale_readme(
        "stale2",
        "<!-- napkin compare -->\nstale\n<!-- /napkin -->\n",
    )?;
    let current_arg = current.0.to_string_lossy().into_owned();
    let stale1_arg = stale1.0.to_string_lossy().into_owned();
    let stale2_arg = stale2.0.to_string_lossy().into_owned();

    let (_, stderr, code) = napkin(&["readme", "--check", &current_arg, &stale1_arg, &stale2_arg]);
    assert_ne!(code, 0);
    assert!(stderr.contains(&stale1_arg), "{stderr}");
    assert!(stderr.contains(&stale2_arg), "{stderr}");
    assert!(!stderr.contains(&current_arg), "{stderr}");
    assert_eq!(fs::read_to_string(&current.0)?, "no markers\n");
    for stale in [&stale1, &stale2] {
        assert_eq!(
            fs::read_to_string(&stale.0)?,
            "<!-- napkin compare -->\nstale\n<!-- /napkin -->\n"
        );
    }

    let (_, stderr, code) = napkin(&["readme", &current_arg, &stale1_arg, &stale2_arg]);
    assert_eq!(code, 0, "{stderr}");
    let (_, stderr, code) = napkin(&["readme", "--check", &current_arg, &stale1_arg, &stale2_arg]);
    assert_eq!(code, 0, "{stderr}");
    Ok(())
}

#[test]
fn repo_markdown_blocks_are_current() {
    let readme = concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md");
    let problems = concat!(env!("CARGO_MANIFEST_DIR"), "/../../PROBLEMS.md");
    let (_, stderr, code) = napkin(&["readme", "--check", readme, problems]);
    assert_eq!(code, 0, "{stderr}");
}

#[test]
fn readme_example_marker_runs_cli_and_rewrites_body() -> Result<(), Box<dyn std::error::Error>> {
    let path = stale_readme(
        "example",
        "<!-- napkin example fit --gpu h100 --params 7B -->\nstale\n<!-- /napkin -->\n",
    )?;
    let path_arg = path.0.to_string_lossy();
    let (_, stderr, code) = napkin(&["readme", path_arg.as_ref()]);
    assert_eq!(code, 0, "{stderr}");
    let contents = fs::read_to_string(&path.0)?;
    assert!(
        contents.contains("$ napkin fit --gpu h100 --params 7B"),
        "{contents}"
    );
    assert!(contents.contains("fits"), "{contents}");
    Ok(())
}
