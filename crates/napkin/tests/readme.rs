use napkin::profile::Dtype;
use napkin::units::round_sig;
use napkin::{price, profile};
use std::process::Command;

#[test]
fn readme_napkin_blocks_match_renderers() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))
        .expect("README.md must exist at the repo root");
    let h100 = profile::find("h100").unwrap();
    let all = profile::builtin();
    let expected_blocks: Vec<(&str, String)> = vec![
        (
            "<!-- napkin table --gpu h100 -->",
            napkin::table::numbers(h100, false).expect("h100 must have a measured profile"),
        ),
        (
            "<!-- napkin compare -->",
            napkin::table::compare(all, false),
        ),
        (
            "<!-- napkin prices -->",
            napkin::table::prices(price::builtin()),
        ),
    ];
    for (marker, body) in expected_blocks {
        let start = readme
            .find(marker)
            .expect("README is missing the napkin table marker");
        let after = &readme[start + marker.len()..];
        let end = after
            .find("<!-- /napkin -->")
            .expect("README napkin marker is not closed");
        let found = after[..end].trim();
        assert_eq!(
            found,
            body.trim(),
            "README marker {marker} is stale; run `napkin readme --write README.md`"
        );
    }
}

#[test]
fn readme_cost_examples_match_cli() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))
        .expect("README.md must exist at the repo root");
    let mut in_text_block = false;
    let mut examples = 0;
    for line in readme.lines() {
        if line == "```text" {
            in_text_block = true;
            continue;
        }
        if line == "```" {
            in_text_block = false;
            continue;
        }
        if !in_text_block || !line.starts_with("$ napkin cost ") {
            continue;
        }
        let args = line.split_whitespace().skip(2).collect::<Vec<_>>();
        let output = Command::new(env!("CARGO_BIN_EXE_napkin"))
            .args(args)
            .output()
            .expect("napkin binary must run");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected = readme
            .lines()
            .skip_while(|candidate| *candidate != line)
            .skip(1)
            .take_while(|candidate| !candidate.starts_with("$ ") && *candidate != "```")
            .fold(String::new(), |mut output, line| {
                output.push_str(line);
                output.push('\n');
                output
            });
        assert_eq!(String::from_utf8_lossy(&output.stdout), expected);
        examples += 1;
    }
    assert!(examples >= 2, "expected at least two README cost examples");
}

#[test]
fn llm_recipe_follow_up_commands_have_example_markers() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))
        .expect("README.md must exist at the repo root");
    let mut in_llm_example = false;
    for line in readme.lines().map(str::trim) {
        if line.starts_with("<!-- napkin example llm ") {
            in_llm_example = true;
            continue;
        }
        if in_llm_example && line == "<!-- /napkin -->" {
            in_llm_example = false;
            continue;
        }
        if in_llm_example
            && let Some(arguments) = line.strip_prefix("$ napkin ")
            && (arguments.starts_with("fit ") || arguments.starts_with("estimate "))
        {
            let marker = format!("<!-- napkin example {arguments} -->");
            assert!(
                readme.lines().any(|candidate| candidate.trim() == marker),
                "missing example marker for recipe follow-up: {marker}"
            );
        }
    }
}

fn problems() -> std::io::Result<String> {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../PROBLEMS.md"))
}

fn same(value: f64, expected: f64) -> bool {
    (value - expected).abs() <= 1e-9 * expected.abs()
}

#[expect(
    clippy::too_many_lines,
    reason = "each problem's teaching constant stays visible as its own table row"
)]
#[test]
fn problem_teaching_constants_match_profiles() -> Result<(), Box<dyn std::error::Error>> {
    let problems = problems()?;
    let h100 = profile::find("h100").ok_or("missing h100 profile")?;
    let measured = h100
        .measured
        .as_ref()
        .ok_or("h100 has no measured profile")?;
    let h2d_latency_us = measured.h2d_latency_us.ok_or("h100 has no h2d latency")?;
    let bf16 = measured
        .compute_tflops
        .get(&Dtype::Bf16)
        .ok_or("h100 has no measured bf16")?;
    let bf16_ceiling = h100
        .ceiling
        .compute_tflops
        .get(&Dtype::Bf16)
        .ok_or("h100 has no bf16 ceiling")?;
    let price = price::find("modal-h100").ok_or("missing modal-h100 price")?;
    for (text, value, pinned) in [
        (
            "| HBM bandwidth, `M` | 3 TB/s |",
            round_sig(measured.hbm_gbps, 1),
            3000.0,
        ),
        (
            "measured device-to-device copy, 3003 GB/s",
            round_sig(measured.hbm_gbps, 4),
            3003.0,
        ),
        (
            "| Dense BF16, `F` | 730 TFLOP/s |",
            round_sig(*bf16, 2),
            730.0,
        ),
        (
            "| Dense BF16, datasheet | 990 TFLOP/s |",
            round_sig(*bf16_ceiling, 2),
            990.0,
        ),
        (
            "half of NVIDIA's 1979 TFLOP/s sparse headline",
            2.0 * bf16_ceiling,
            1979.0,
        ),
        (
            "| HBM, datasheet | 3.35 TB/s |",
            h100.ceiling.hbm_gbps,
            3350.0,
        ),
        (
            "| Host → device, `P` | 55 GB/s |",
            round_sig(measured.h2d_gbps, 2),
            55.0,
        ),
        (
            "| Fixed cost per copy, `L_copy` | 10 μs |",
            round_sig(h2d_latency_us, 1),
            10.0,
        ),
        (
            "measured 4-byte synchronous copy, 9.8 μs",
            round_sig(h2d_latency_us, 2),
            9.8,
        ),
        (
            "| Kernel launch, `L_launch` | 2 μs |",
            round_sig(measured.launch_us, 1),
            2.0,
        ),
        ("| Memory | 80 GB |", h100.memory_gb, 80.0),
        (
            "| Price | $3.95/GPU-hour |",
            round_sig(price.per_gpu_hour(), 3),
            3.95,
        ),
        (
            "runs at 2795 GB/s",
            round_sig(measured.elementwise_gbps, 4),
            2795.0,
        ),
        (
            "for the H100's 132 SMs",
            f64::from(measured.sm_count),
            132.0,
        ),
    ] {
        assert!(
            problems.contains(text),
            "PROBLEMS.md no longer says {text:?}"
        );
        assert!(
            same(value, pinned),
            "PROBLEMS.md pins {text:?} but the profile now gives {value}; update the problems"
        );
    }
    let total = measured.memory_bytes;
    assert_eq!(
        total, 85_017_624_576,
        "update the GB versus GiB note in problem 4"
    );
    assert!(problems.contains("`cuDeviceTotalMem` is 85,017,624,576 B"));
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "each worked answer stays visible as its own table row"
)]
#[test]
fn problem_answers_follow_from_teaching_constants() -> Result<(), Box<dyn std::error::Error>> {
    const GIB: f64 = 1_073_741_824.0;
    const M: f64 = 3e12;
    const F: f64 = 730e12;
    const P: f64 = 55e9;
    const L_COPY: f64 = 10e-6;
    const L_LAUNCH: f64 = 2e-6;
    const USABLE: f64 = 72e9;
    const WEIGHTS: f64 = 8e9 * 16.0 / 8.0;
    const KV_PER_TOKEN: f64 = 2.0 * 32.0 * 8.0 * 128.0 * 2.0;
    const PRICE: f64 = 3.95;
    let problems = problems()?;
    let per_request = 9216.0 * KV_PER_TOKEN;
    let decode_kv = 2048.0 * KV_PER_TOKEN;
    let batch_1_rate = 1.0 / ((WEIGHTS + decode_kv) / M);
    let batch_32_rate = 32.0 / ((WEIGHTS + 32.0 * decode_kv) / M);
    let largest_batch = ((USABLE - WEIGHTS) / per_request).floor();
    let ridge = F / M;
    let sparse_ridge = 1979e12 / 3.35e12;
    let dense_ceiling_ridge = 990e12 / 3.35e12;
    // (text in PROBLEMS.md, value from the teaching constants, significant
    // digits shown, value shown)
    for (text, value, sig, shown) in [
        (
            "1 GiB      = 2³⁰ B = 1.07 × 10⁹ B = 1.07 GB",
            GIB / 1e9,
            3,
            1.07,
        ),
        (
            "copy bytes = 2 × 1.07 GB = 2.15 GB",
            2.0 * GIB / 1e9,
            3,
            2.15,
        ),
        ("add bytes  = 3 × 1.07 GB = 3.2 GB", 3.0 * GIB / 1e9, 2, 3.2),
        ("about 0.7 ms for the copy", 2.0 * GIB / M * 1e3, 1, 0.7),
        ("and 1.1 ms for the add", 3.0 * GIB / M * 1e3, 2, 1.1),
        ("allocation gives 0.36 ms", GIB / M * 1e3, 2, 0.36),
        (
            "reading 1 GiB as 1 GB drops 7%",
            (GIB / 1e9 - 1.0) * 100.0,
            1,
            7.0,
        ),
        ("an intensity of 0.08 FLOP/B", 1.0 / 12.0, 1, 0.08),
        (
            "about 100 ms for 10,000 copies",
            (10_000.0 * L_COPY + 40e6 / P) * 1e3,
            1,
            100.0,
        ),
        (
            "against 0.7 ms for one packed copy",
            (L_COPY + 40e6 / P) * 1e3,
            1,
            0.7,
        ),
        ("10 μs × 55 GB/s ≈ 550 KB", L_COPY * P / 1e3, 2, 550.0),
        (
            "F / M = 730 × 10¹² / 3 × 10¹² ≈ 240  FLOP/B",
            ridge,
            2,
            240.0,
        ),
        (
            "N / 3     = 240  →  N ≈ 720",
            3.0 * round_sig(ridge, 2),
            2,
            720.0,
        ),
        (
            "above N ≈ 700 by the measured numbers",
            3.0 * ridge,
            1,
            700.0,
        ),
        ("`1979 / 3.35 ≈ 590 FLOP/B`", sparse_ridge, 2, 590.0),
        ("crossover at N ≈ 1800", 3.0 * sparse_ridge, 2, 1800.0),
        ("`990 / 3.35 ≈ 300 FLOP/B`", dense_ceiling_ridge, 1, 300.0),
        ("N ≈ 900.", 3.0 * dense_ceiling_ridge, 1, 900.0),
        (
            "`2 × 512³ = 268M` FLOPs",
            2.0 * 512_f64.powi(3) / 1e6,
            3,
            268.0,
        ),
        ("`6 × 512² = 1.57 MB`", 6.0 * 512_f64.powi(2) / 1e6, 3, 1.57),
        (
            "2.15G FLOPs over 6.29 MB",
            2.0 * 1024_f64.powi(3) / 1e9,
            3,
            2.15,
        ),
        (
            "2.15G FLOPs over 6.29 MB",
            6.0 * 1024_f64.powi(2) / 1e6,
            3,
            6.29,
        ),
        ("× 2 B = 131,072 B ≈ 131 KB", KV_PER_TOKEN, 6, 131_072.0),
        ("tokens × 131 KB ≈ 1.2 GB", per_request / 1e9, 2, 1.2),
        ("64 × 1.2 GB ≈ 77 GB", 64.0 * per_request / 1e9, 2, 77.0),
        (
            "+ 16 GB weights = 93 GB > 72 GB",
            (WEIGHTS + 64.0 * per_request) / 1e9,
            2,
            93.0,
        ),
        ("/ 1.2 GB per request ≈ 46 requests", largest_batch, 2, 46.0),
        (
            "`46 × 1.21 GB = 55.6 GB`",
            largest_batch * per_request / 1e9,
            3,
            55.6,
        ),
        (
            "mixing the two units silently costs\n2.4%",
            (KV_PER_TOKEN / 128e3 - 1.0) * 100.0,
            2,
            2.4,
        ),
        ("85.0 GB or 79.2 GiB", 85_017_624_576.0 / GIB, 3, 79.2),
        (
            "conservative by about 6%",
            (85_017_624_576.0 / 80e9 - 1.0) * 100.0,
            1,
            6.0,
        ),
        (
            "524 KB, and only 11 requests fit",
            4.0 * KV_PER_TOKEN / 1e3,
            3,
            524.0,
        ),
        (
            "524 KB, and only 11 requests fit",
            ((USABLE - WEIGHTS) / (4.0 * per_request)).floor(),
            2,
            11.0,
        ),
        (
            "about 2 ms unfused",
            (6e9 / M + 3.0 * L_LAUNCH) * 1e3,
            1,
            2.0,
        ),
        ("and 0.7 ms fused", (2e9 / M + L_LAUNCH) * 1e3, 1, 0.7),
        ("1.5 × 10⁹ / 6 × 10⁹ = 0.25 FLOP/B", 1.5e9 / 6e9, 2, 0.25),
        ("3 × 10¹² B/s ≈ 0.7 μs`", 2e6 / M * 1e6, 1, 0.7),
        ("2,048 tokens ≈ 0.27 GB", decode_kv / 1e9, 2, 0.27),
        (
            "16 GB + 0.27 GB = 16.3 GB per step",
            (WEIGHTS + decode_kv) / 1e9,
            3,
            16.3,
        ),
        (
            "≈ 5.4 ms → 1 / 5.4 ms",
            (WEIGHTS + decode_kv) / M * 1e3,
            2,
            5.4,
        ),
        ("about 180 tokens/s at batch 1", batch_1_rate, 2, 180.0),
        (
            "32 × 0.27 GB = 24.6 GB per step",
            (WEIGHTS + 32.0 * decode_kv) / 1e9,
            3,
            24.6,
        ),
        (
            "≈ 8.2 ms → 32 / 8.2 ms",
            (WEIGHTS + 32.0 * decode_kv) / M * 1e3,
            2,
            8.2,
        ),
        ("3,900 tokens/s at batch 32", batch_32_rate, 2, 3900.0),
        ("20 times more", batch_32_rate / batch_1_rate, 1, 20.0),
        (
            "A step costs 50% more time",
            (WEIGHTS + 32.0 * decode_kv) / (WEIGHTS + decode_kv),
            2,
            1.5,
        ),
        ("/ 730 × 10¹² ≈ 0.7 ms", 2.0 * 8e9 * 32.0 / F * 1e3, 1, 0.7),
        (
            "each request reads\n1.1 GB per step",
            8192.0 * KV_PER_TOKEN / 1e9,
            2,
            1.1,
        ),
        (
            "batch 32 reads 34 GB of KV",
            32.0 * 8192.0 * KV_PER_TOKEN / 1e9,
            2,
            34.0,
        ),
        ("≈ 650K tokens per GPU-hour", 180.0 * 3600.0 / 1e3, 2, 650.0),
        (
            "about $6 per million tokens at batch 1",
            PRICE / (180.0 * 3600.0) * 1e6,
            1,
            6.0,
        ),
        ("≈ 14M tokens per GPU-hour", 3900.0 * 3600.0 / 1e6, 2, 14.0),
        (
            "$0.28 at batch 32",
            PRICE / (3900.0 * 3600.0) * 1e6,
            2,
            0.28,
        ),
    ] {
        assert!(
            problems.contains(text),
            "PROBLEMS.md no longer says {text:?}"
        );
        assert!(
            same(round_sig(value, sig), shown),
            "{text:?}: the teaching constants give {value}, not {shown}"
        );
    }
    Ok(())
}
