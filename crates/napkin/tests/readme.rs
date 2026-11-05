use napkin::collective::{self, Query};
use napkin::profile::collective::{Collective, Topology};
use napkin::profile::{Basis, Dtype};
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
        (
            "<!-- napkin breakeven --gpu h100 --vs b200 --dtype bf16 -->",
            napkin::table::breakeven(
                price::builtin(),
                h100,
                profile::find("b200").unwrap(),
                Dtype::Bf16,
            )
            .expect("h100 and b200 must have bf16 rates"),
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

/// The prose under the `napkin breakeven` table quotes rounded prices, rate
/// ratios, and job costs. This fails when a price record or measured rate
/// moves one of them; update the README sentence and this test together.
#[test]
fn readme_price_comparison_follows_from_records() -> Result<(), Box<dyn std::error::Error>> {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))?;
    let record = |id: &str| price::find(id).ok_or(format!("missing price record {id}"));
    let bf16_tflops = |gpu: &str| -> Result<f64, String> {
        profile::find(gpu)
            .and_then(|profile| {
                profile
                    .rates(false)
                    .compute_tflops
                    .get(&Dtype::Bf16)
                    .copied()
            })
            .map(|rate| rate.tflops)
            .ok_or(format!("{gpu} has no bf16 rate"))
    };
    let h100_aws = record("aws-h100")?.per_gpu_hour();
    let b200_aws = record("aws-b200")?.per_gpu_hour();
    let h100_gcp_spot = record("gcp-h100-spot")?.per_gpu_hour();
    let b200_gcp_spot = record("gcp-b200-spot")?.per_gpu_hour();
    let (h100, b200) = (bf16_tflops("h100")?, bf16_tflops("b200")?);
    // The utilization example's 1400 s at 50% on the H100 and 750 s at 30%
    // on the B200, billed at AWS on-demand prices.
    let job_cost = |busy_s: f64, utilization: f64, per_gpu_hour: f64| {
        busy_s / utilization * per_gpu_hour / 3600.0
    };
    // (text in README.md, value from the records, significant digits shown,
    // value shown)
    for (text, value, sig, shown) in [
        ("AWS charges $14.2", b200_aws, 3, 14.2),
        ("against $6.88 for an H100", h100_aws, 3, 6.88),
        ("2.07 times as much", b200_aws / h100_aws, 3, 2.07),
        ("only 1.92 times the H100's", b200 / h100, 3, 1.92),
        ("costs $5.35", job_cost(1400.0, 0.5, h100_aws), 3, 5.35),
        (
            "against $9.89 on the B200",
            job_cost(750.0, 0.3, b200_aws),
            3,
            9.89,
        ),
        (
            "down to 0.39 of the H100's utilization",
            (h100 / h100_gcp_spot) / (b200 / b200_gcp_spot),
            2,
            0.39,
        ),
        ("give $10,400", b200_aws * 730.0 / 1e3, 3, 10.4),
        (
            "and $8,190",
            record("aws-b200-1y")?.per_gpu_hour() * 730.0 / 1e3,
            3,
            8.19,
        ),
    ] {
        assert!(readme.contains(text), "README.md no longer says {text:?}");
        assert!(
            (round_sig(value, sig) - shown).abs() < 1e-9,
            "README.md says {text:?} but the records give {value}"
        );
    }
    assert!(
        b200_aws / h100_aws > b200 / h100,
        "README.md says the B200 loses at AWS on-demand prices even at equal utilization"
    );
    assert!(
        b200_gcp_spot < h100_gcp_spot,
        "README.md says a GCP spot B200 GPU-hour costs less than an H100's"
    );
    Ok(())
}

#[test]
fn readme_copy_sizes_match_h100_profile() -> Result<(), Box<dyn std::error::Error>> {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))?;
    let rates = profile::find("h100")
        .ok_or("missing h100 profile")?
        .rates(false);
    let latency_us = rates.h2d_latency_us.ok_or("h100 has no h2d latency")?;
    let gbps = rates.h2d.ok_or("h100 has no h2d rate")?.gbps;
    // μs × GB/s = 10⁻⁶ s × 10⁹ B/s = 10³ B, so the product is in KB.
    let break_even_kb = latency_us * gbps;
    let text = format!(
        "about {} KB and {} KB",
        round_sig(napkin::table::SPREAD_FLAG * break_even_kb, 1),
        round_sig(break_even_kb, 1)
    );
    assert!(
        readme.contains(&text),
        "README's copy-latency prose should say {text:?}; update it with the profile"
    );
    Ok(())
}

#[test]
fn validation_report_matches_cli_output() {
    let expected =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../VALIDATION.md"))
            .expect("VALIDATION.md must exist at the repo root");
    let output = Command::new(env!("CARGO_BIN_EXE_napkin"))
        .arg("validate")
        .output()
        .expect("napkin validate must run");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), expected);
}

#[test]
fn validation_estimate_examples_match_cli_output() -> Result<(), Box<dyn std::error::Error>> {
    let report =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../VALIDATION.md"))?;
    let lines = report.lines().collect::<Vec<_>>();
    let mut examples = 0;
    for (index, line) in lines.iter().enumerate() {
        let Some(command) = line.strip_prefix("$ napkin estimate ") else {
            continue;
        };
        let output = Command::new(env!("CARGO_BIN_EXE_napkin"))
            .args(std::iter::once("estimate").chain(command.split_whitespace()))
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected = lines[index + 1..]
            .iter()
            .take_while(|candidate| **candidate != "```")
            .fold(String::new(), |mut expected, line| {
                expected.push_str(line);
                expected.push('\n');
                expected
            });
        assert_eq!(String::from_utf8_lossy(&output.stdout), expected);
        examples += 1;
    }
    assert_eq!(
        examples, 3,
        "expected one estimate command per validation case"
    );
    Ok(())
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

fn eight_h100s(op: Collective, bytes: f64) -> Result<f64, String> {
    collective::estimate(&Query {
        gpu: "h100",
        gpu_count: 8,
        topology: Topology::Nvlink,
        collective: op,
        dtype: Dtype::Bf16,
        bytes,
    })
    .map(|estimate| estimate.time_s)
}

/// The "Across machines" example composes measured eight-GPU curves with the
/// datasheet RDMA link; when either moves, the README arithmetic must too.
#[test]
fn cross_machine_example_follows_from_curves_and_network() -> Result<(), Box<dyn std::error::Error>>
{
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))?;
    let network = profile::find("h100")
        .and_then(|h100| h100.ceiling.network.as_ref())
        .ok_or("h100 has no datasheet network")?;
    assert_eq!(network.nic_gbit_per_gpu, 400.0);
    assert_eq!(network.gbps(), 50.0);
    assert_eq!(network.gpus_per_node, 8);
    assert_eq!(network.latency_us, 5.0);
    let latency = network.latency_us / 1e6;
    let rate = network.gbps() * 1e9;
    let ms = |seconds: f64| seconds * 1e3;
    let us = |seconds: f64| seconds * 1e6;

    let gradients = 4.0 * f64::from(1_u32 << 30);
    let reduce_scatter = eight_h100s(Collective::ReduceScatter, gradients)?;
    let all_gather = eight_h100s(Collective::AllGather, gradients)?;
    let all_reduce = eight_h100s(Collective::AllReduce, gradients)?;
    // A two-rank AllReduce sends 2(n-1)/n = 1× its shard, in two steps.
    let shard = gradients / f64::from(network.gpus_per_node);
    let wire = shard / rate + 2.0 * latency;
    let serial = reduce_scatter + wire + all_gather;
    let pipelined = reduce_scatter.max(wire).max(all_gather);
    assert!(same(round_sig(ms(reduce_scatter), 3), 10.6));
    assert!(same(round_sig(ms(all_gather), 3), 10.6));
    assert!(same(round_sig(ms(all_reduce), 3), 15.7));
    assert!(same(round_sig(shard / 1e6, 3), 537.0));
    assert!(same(round_sig(ms(wire), 3), 10.7));
    assert!(same(round_sig(ms(serial), 2), 32.0));
    assert!(same(round_sig(ms(pipelined), 2), 11.0));
    assert!(same(round_sig(ms(8.0 * shard / rate), 2), 86.0));
    assert!(same(round_sig(ms(shard / 43e9), 3), 12.5));

    let activations = 128.0 * 1024.0;
    let small_scatter = eight_h100s(Collective::ReduceScatter, activations)?;
    let small_gather = eight_h100s(Collective::AllGather, activations)?;
    let small_reduce = eight_h100s(Collective::AllReduce, activations)?;
    let small_wire = activations / f64::from(network.gpus_per_node) / rate + 2.0 * latency;
    let small = small_scatter + small_wire + small_gather;
    assert!(same(round_sig(us(small_scatter), 3), 39.9));
    assert!(same(round_sig(us(small_gather), 3), 42.4));
    assert!(same(round_sig(us(small_reduce), 3), 38.9));
    assert!(same(round_sig(us(small), 1), 90.0));
    assert!(same(round_sig(ms(160.0 * small), 2), 15.0));
    assert!(same(round_sig(ms(160.0 * small_reduce), 2), 6.2));

    for text in [
        "400 Gbit/s / 8 bits = 50 GB/s per GPU, each direction",
        "8 GPUs × one ConnectX-7 each = 8 × 50 GB/s = 400 GB/s out",
        "latency     5 μs one-way",
        "4 GiB / 8 = 512 MiB shard",
        "512 MiB = 537 MB, so 537 MB / 50 GB/s = 10.7 ms, plus two 5 μs steps",
        "40 GiB/s (43 GB/s), below the 50 GB/s (46.6 GiB/s) line rate",
        "reduce-scatter 10.6 ms",
        "network        10.7 ms    datasheet: 537 MB / 50 GB/s + 2 × 5 μs",
        "all-gather     10.6 ms",
        "serial         32 ms",
        "pipelined      ≥ 11 ms",
        "measured 15.7 ms",
        "between 11 and 32 ms",
        "8 × 10.7 = 86 ms",
        "43 GB/s would make the\nnetwork stage 12.5 ms",
        "39.9 μs ReduceScatter",
        "42.4 μs AllGather, for about 90 μs",
        "cost 15 ms instead of 6.2 ms",
    ] {
        assert!(
            readme.contains(text),
            "README \"Across machines\" no longer says {text:?}"
        );
    }
    Ok(())
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

/// The weight-loading example borrows storage rates from Simon Eskildsen's
/// Napkin Math, measured on 48 vCPU CPU VMs, and combines them with this
/// repo's measured H100 host → device rate. If either moves, the prose
/// arithmetic and the derived cold start in the cost section must move too.
#[test]
fn readme_weight_loading_follows_from_cited_rates() -> Result<(), Box<dyn std::error::Error>> {
    const GIB: f64 = 1_073_741_824.0;
    const MIB: f64 = 1_048_576.0;
    const WEIGHTS: f64 = 70e9 * 16.0 / 8.0;
    // The prose divides by the rounded rates in its table, as a reader would
    // on paper: 140 GB / 3.2 GB/s is 44 s, while the unrounded 3 GiB/s gives
    // 43.5 s.
    const BLOB: f64 = 3.2e9;
    const SSD: f64 = 5.4e9;
    const PCIE: f64 = 56e9;
    const ONE_CONNECTION: f64 = 100.0 * MIB;
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))?;
    let h100 = profile::find("h100").ok_or("missing h100 profile")?;
    let h2d = h100
        .rates(false)
        .h2d
        .ok_or("h100 has no published h2d rate")?;
    assert_eq!(h2d.basis, Basis::Nvbandwidth);
    let blob_s = WEIGHTS / BLOB;
    let ssd_s = WEIGHTS / SSD;
    let pcie_s = WEIGHTS / PCIE;
    // (text in README.md, value from the cited rates, significant digits
    // shown, value shown)
    for (text, value, sig, shown) in [
        ("| 3 GiB/s = 3.2 GB/s |", 3.0 * GIB / 1e9, 2, BLOB / 1e9),
        ("| 5 GiB/s = 5.4 GB/s |", 5.0 * GIB / 1e9, 2, SSD / 1e9),
        (
            "| Host → device, pinned | 56 GB/s |",
            h2d.gbps,
            2,
            PCIE / 1e9,
        ),
        ("140 GB / 3.2 GB/s ≈ 44 s", blob_s, 2, 44.0),
        ("140 GB / 5.4 GB/s ≈ 26 s", ssd_s, 2, 26.0),
        ("140 GB / 56 GB/s  ≈ 2.5 s", pcie_s, 2, 2.5),
        ("10 to 20 times slower", ssd_s / pcie_s, 1, 10.0),
        ("10 to 20 times slower", blob_s / pcie_s, 1, 20.0),
        ("`44 + 26 + 2.5 ≈ 72 s`", blob_s + ssd_s + pcie_s, 2, 72.0),
        (
            "17.5 GB over its own PCIe link",
            WEIGHTS / 8.0 / 1e9,
            3,
            17.5,
        ),
        ("about 0.3 s if all eight", WEIGHTS / 8.0 / PCIE, 1, 0.3),
        (
            "`140 GB / 0.105 GB/s ≈ 1300 s`",
            ONE_CONNECTION / 1e9,
            3,
            0.105,
        ),
        (
            "`140 GB / 0.105 GB/s ≈ 1300 s`",
            WEIGHTS / ONE_CONNECTION,
            2,
            1300.0,
        ),
        ("about 22 minutes", WEIGHTS / ONE_CONNECTION / 60.0, 2, 22.0),
        ("65 s and 27–30 s", WEIGHTS / (2.0 * GIB), 2, 65.0),
        ("65 s and 27–30 s", WEIGHTS / (4.9 * GIB), 2, 27.0),
        ("65 s and 27–30 s", WEIGHTS / (4.3 * GIB), 2, 30.0),
        ("wait at least 44 s", blob_s, 2, 44.0),
        ("--startup 44s", blob_s, 2, 44.0),
        ("`--startup 26s`", ssd_s, 2, 26.0),
        ("at most 2.5 s of either startup", pcie_s, 2, 2.5),
    ] {
        assert!(readme.contains(text), "README.md no longer says {text:?}");
        assert!(
            same(round_sig(value, sig), shown),
            "{text:?}: the cited rates give {value}, not {shown}"
        );
    }
    Ok(())
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
    let h2d = h100
        .rates(false)
        .h2d
        .ok_or("h100 has no published h2d rate")?;
    let measured = h100
        .measured
        .as_ref()
        .ok_or("h100 has no measured profile")?;
    let h2d_latency_us = measured.h2d_latency_us.ok_or("h100 has no h2d latency")?;
    let bf16 = measured
        .compute_tflops
        .get(&Dtype::Bf16)
        .ok_or("h100 has no measured bf16")?;
    let fp16 = measured
        .compute_tflops
        .get(&Dtype::Fp16)
        .ok_or("h100 has no measured fp16")?;
    let bf16_ceiling = h100
        .ceiling
        .compute_tflops
        .get(&Dtype::Bf16)
        .ok_or("h100 has no bf16 ceiling")?;
    let price = price::find("modal-h100").ok_or("missing modal-h100 price")?;
    let h2d_row =
        "| Host → device, `P` | 56 GB/s | NVBandwidth CE 512 MiB pinned copy, PCIe Gen5 x16 |";
    assert_eq!(h2d.basis, Basis::Nvbandwidth);
    assert!(h2d_row.contains("NVBandwidth CE"));
    for (text, value, pinned) in [
        (
            "| HBM bandwidth, `M` | 3 TB/s |",
            round_sig(measured.hbm_gbps, 1),
            3000.0,
        ),
        (
            "measured device-to-device copy, 3002 GB/s",
            round_sig(measured.hbm_gbps, 4),
            3002.0,
        ),
        (
            "| Dense BF16, `F` | 690 TFLOP/s |",
            round_sig(*bf16, 2),
            690.0,
        ),
        (
            "the 220 FLOP/B fp16 ridge",
            round_sig(fp16 / measured.hbm_gbps * 1e3, 2),
            220.0,
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
        (h2d_row, round_sig(h2d.gbps, 2), 56.0),
        (
            "| Fixed cost per copy, `L_copy` | 9 μs |",
            round_sig(h2d_latency_us, 1),
            9.0,
        ),
        (
            "measured 4-byte synchronous copy, 8.8 μs",
            round_sig(h2d_latency_us, 2),
            8.8,
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
            "runs at 2794 GB/s",
            round_sig(measured.elementwise_gbps, 4),
            2794.0,
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
    const F: f64 = 690e12;
    const P: f64 = 56e9;
    const L_COPY: f64 = 9e-6;
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
            "about 90 ms for 10,000 copies",
            (10_000.0 * L_COPY + 40e6 / P) * 1e3,
            1,
            90.0,
        ),
        (
            "against 0.7 ms for one packed copy",
            (L_COPY + 40e6 / P) * 1e3,
            1,
            0.7,
        ),
        ("9 μs × 56 GB/s ≈ 500 KB", L_COPY * P / 1e3, 2, 500.0),
        (
            "under the 25 KB latency size",
            napkin::table::SPREAD_FLAG * L_COPY * P / 1e3,
            2,
            25.0,
        ),
        (
            "F / M = 690 × 10¹² / 3 × 10¹² ≈ 230  FLOP/B",
            ridge,
            2,
            230.0,
        ),
        (
            "N / 3     = 230  →  N ≈ 690",
            3.0 * round_sig(ridge, 2),
            2,
            690.0,
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
        ("/ 690 × 10¹² ≈ 0.7 ms", 2.0 * 8e9 * 32.0 / F * 1e3, 1, 0.7),
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
