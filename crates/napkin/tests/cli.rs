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
fn estimate_runs_on_datasheet_gpu() {
    let (stdout, stderr, code) = napkin(&[
        "estimate", "--gpu", "b200", "--dtype", "bf16", "--flops", "200T", "--bytes", "1TB",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("B200"), "{stdout}");
    assert!(stdout.contains("total"), "{stdout}");
}

#[test]
fn estimate_rejects_h2d_on_datasheet_gpu() {
    let (_, stderr, code) = napkin(&[
        "estimate", "--gpu", "b200", "--dtype", "bf16", "--flops", "200T", "--bytes", "1TB",
        "--h2d", "2GB",
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
fn compare_prints_table() {
    let (stdout, _, code) = napkin(&["compare"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("| GPU |"), "{stdout}");
}

struct TempReadme(PathBuf);

impl Drop for TempReadme {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn stale_readme(name: &str) -> std::io::Result<TempReadme> {
    let path = std::env::temp_dir().join(format!("napkin-readme-{name}-{}.md", std::process::id()));
    fs::write(&path, "<!-- napkin compare -->\nstale\n<!-- /napkin -->\n")
        .map(|()| TempReadme(path))
}

#[test]
fn readme_check_and_write_conflict() {
    let (_, _, code) = napkin(&["readme", "--check", "--write"]);
    assert_eq!(code, 2);
}

#[test]
fn readme_check_preserves_stale_file() -> Result<(), Box<dyn std::error::Error>> {
    let path = stale_readme("check")?;
    let original = fs::read_to_string(&path.0)?;
    let path_arg = path.0.to_string_lossy();
    let (_, _, code) = napkin(&["readme", "--check", path_arg.as_ref()]);
    assert_ne!(code, 0);
    assert_eq!(fs::read_to_string(&path.0)?, original);
    Ok(())
}

#[test]
fn readme_rewrites_stale_file() -> Result<(), Box<dyn std::error::Error>> {
    let path = stale_readme("rewrite")?;
    let path_arg = path.0.to_string_lossy();
    let (_, _, code) = napkin(&["readme", path_arg.as_ref()]);
    assert_eq!(code, 0);
    let (_, _, code) = napkin(&["readme", "--check", path_arg.as_ref()]);
    assert_eq!(code, 0);
    Ok(())
}
