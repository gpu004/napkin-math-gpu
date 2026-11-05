use napkin::collective::{self, Query};
use napkin::profile::collective::{Collective, Topology};
use napkin::profile::{Basis, Dtype};
use napkin::units::round_sig;
use napkin::{price, profile};
use std::process::Command;

struct Quote {
    text: &'static str,
    value: f64,
    sig: u32,
    shown: f64,
}

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

#[expect(
    clippy::too_many_lines,
    reason = "each quoted number stays visible as its own table row"
)]
#[test]
fn readme_price_comparison_follows_from_records() -> Result<(), Box<dyn std::error::Error>> {
    const H100_BUSY_S: f64 = 1400.0;
    const H100_UTILIZATION: f64 = 0.5;
    const B200_BUSY_S: f64 = 750.0;
    const B200_UTILIZATION: f64 = 0.3;
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
    let job_cost = |busy_s: f64, utilization: f64, per_gpu_hour: f64| {
        busy_s / utilization * per_gpu_hour / 3600.0
    };
    for Quote {
        text,
        value,
        sig,
        shown,
    } in [
        Quote {
            text: "AWS charges $14.2",
            value: b200_aws,
            sig: 3,
            shown: 14.2,
        },
        Quote {
            text: "against $6.88 for an H100",
            value: h100_aws,
            sig: 3,
            shown: 6.88,
        },
        Quote {
            text: "2.07 times as much",
            value: b200_aws / h100_aws,
            sig: 3,
            shown: 2.07,
        },
        Quote {
            text: "only 1.92 times the H100's",
            value: b200 / h100,
            sig: 3,
            shown: 1.92,
        },
        Quote {
            text: "costs $5.35",
            value: job_cost(H100_BUSY_S, H100_UTILIZATION, h100_aws),
            sig: 3,
            shown: 5.35,
        },
        Quote {
            text: "against $9.89 on the B200",
            value: job_cost(B200_BUSY_S, B200_UTILIZATION, b200_aws),
            sig: 3,
            shown: 9.89,
        },
        Quote {
            text: "down to 0.39 of the H100's utilization",
            value: (h100 / h100_gcp_spot) / (b200 / b200_gcp_spot),
            sig: 2,
            shown: 0.39,
        },
        Quote {
            text: "give $10,400",
            value: b200_aws * 730.0 / 1e3,
            sig: 3,
            shown: 10.4,
        },
        Quote {
            text: "and $8,190",
            value: record("aws-b200-1y")?.per_gpu_hour() * 730.0 / 1e3,
            sig: 3,
            shown: 8.19,
        },
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
    let break_even_kb = latency_us * 1e-6 * gbps * 1e9 / 1e3;
    let text = format!(
        "about {} KB and {} KB",
        round_sig(napkin::table::LATENCY_SIZE_SLOWDOWN * break_even_kb, 1),
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
    let shard = gradients / f64::from(network.gpus_per_node);
    let wire = Collective::AllReduce.bus_factor(2) * shard / rate + 2.0 * latency;
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

#[expect(
    clippy::too_many_lines,
    reason = "each quoted number stays visible as its own table row"
)]
#[test]
fn readme_weight_loading_follows_from_cited_rates() -> Result<(), Box<dyn std::error::Error>> {
    const GIB: f64 = 1_073_741_824.0;
    const MIB: f64 = 1_048_576.0;
    const WEIGHTS: f64 = 70e9 * 16.0 / 8.0;
    const ROUNDED_BLOB: f64 = 3.2e9;
    const ROUNDED_SSD: f64 = 5.4e9;
    const ROUNDED_PCIE: f64 = 56e9;
    const ONE_CONNECTION: f64 = 100.0 * MIB;
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))?;
    let h100 = profile::find("h100").ok_or("missing h100 profile")?;
    let h2d = h100
        .rates(false)
        .h2d
        .ok_or("h100 has no published h2d rate")?;
    assert_eq!(h2d.basis, Basis::Nvbandwidth);
    let blob_s = WEIGHTS / ROUNDED_BLOB;
    let ssd_s = WEIGHTS / ROUNDED_SSD;
    let pcie_s = WEIGHTS / ROUNDED_PCIE;
    for Quote {
        text,
        value,
        sig,
        shown,
    } in [
        Quote {
            text: "| 3 GiB/s = 3.2 GB/s |",
            value: 3.0 * GIB / 1e9,
            sig: 2,
            shown: ROUNDED_BLOB / 1e9,
        },
        Quote {
            text: "| 5 GiB/s = 5.4 GB/s |",
            value: 5.0 * GIB / 1e9,
            sig: 2,
            shown: ROUNDED_SSD / 1e9,
        },
        Quote {
            text: "| Host → device, pinned | 56 GB/s |",
            value: h2d.gbps,
            sig: 2,
            shown: ROUNDED_PCIE / 1e9,
        },
        Quote {
            text: "140 GB / 3.2 GB/s ≈ 44 s",
            value: blob_s,
            sig: 2,
            shown: 44.0,
        },
        Quote {
            text: "140 GB / 5.4 GB/s ≈ 26 s",
            value: ssd_s,
            sig: 2,
            shown: 26.0,
        },
        Quote {
            text: "140 GB / 56 GB/s  ≈ 2.5 s",
            value: pcie_s,
            sig: 2,
            shown: 2.5,
        },
        Quote {
            text: "10 to 20 times slower",
            value: ssd_s / pcie_s,
            sig: 1,
            shown: 10.0,
        },
        Quote {
            text: "10 to 20 times slower",
            value: blob_s / pcie_s,
            sig: 1,
            shown: 20.0,
        },
        Quote {
            text: "`44 + 26 + 2.5 ≈ 72 s`",
            value: blob_s + ssd_s + pcie_s,
            sig: 2,
            shown: 72.0,
        },
        Quote {
            text: "17.5 GB over its own PCIe link",
            value: WEIGHTS / 8.0 / 1e9,
            sig: 3,
            shown: 17.5,
        },
        Quote {
            text: "about 0.3 s if all eight",
            value: WEIGHTS / 8.0 / ROUNDED_PCIE,
            sig: 1,
            shown: 0.3,
        },
        Quote {
            text: "`140 GB / 0.105 GB/s ≈ 1300 s`",
            value: ONE_CONNECTION / 1e9,
            sig: 3,
            shown: 0.105,
        },
        Quote {
            text: "`140 GB / 0.105 GB/s ≈ 1300 s`",
            value: WEIGHTS / ONE_CONNECTION,
            sig: 2,
            shown: 1300.0,
        },
        Quote {
            text: "about 22 minutes",
            value: WEIGHTS / ONE_CONNECTION / 60.0,
            sig: 2,
            shown: 22.0,
        },
        Quote {
            text: "65 s and 27–30 s",
            value: WEIGHTS / (2.0 * GIB),
            sig: 2,
            shown: 65.0,
        },
        Quote {
            text: "65 s and 27–30 s",
            value: WEIGHTS / (4.9 * GIB),
            sig: 2,
            shown: 27.0,
        },
        Quote {
            text: "65 s and 27–30 s",
            value: WEIGHTS / (4.3 * GIB),
            sig: 2,
            shown: 30.0,
        },
        Quote {
            text: "wait at least 44 s",
            value: blob_s,
            sig: 2,
            shown: 44.0,
        },
        Quote {
            text: "--startup 44s",
            value: blob_s,
            sig: 2,
            shown: 44.0,
        },
        Quote {
            text: "`--startup 26s`",
            value: ssd_s,
            sig: 2,
            shown: 26.0,
        },
        Quote {
            text: "at most 2.5 s of either startup",
            value: pcie_s,
            sig: 2,
            shown: 2.5,
        },
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
    for Quote {
        text,
        value,
        sig,
        shown,
    } in [
        Quote {
            text: "1 GiB      = 2³⁰ B = 1.07 × 10⁹ B = 1.07 GB",
            value: GIB / 1e9,
            sig: 3,
            shown: 1.07,
        },
        Quote {
            text: "copy bytes = 2 × 1.07 GB = 2.15 GB",
            value: 2.0 * GIB / 1e9,
            sig: 3,
            shown: 2.15,
        },
        Quote {
            text: "add bytes  = 3 × 1.07 GB = 3.2 GB",
            value: 3.0 * GIB / 1e9,
            sig: 2,
            shown: 3.2,
        },
        Quote {
            text: "about 0.7 ms for the copy",
            value: 2.0 * GIB / M * 1e3,
            sig: 1,
            shown: 0.7,
        },
        Quote {
            text: "and 1.1 ms for the add",
            value: 3.0 * GIB / M * 1e3,
            sig: 2,
            shown: 1.1,
        },
        Quote {
            text: "allocation gives 0.36 ms",
            value: GIB / M * 1e3,
            sig: 2,
            shown: 0.36,
        },
        Quote {
            text: "reading 1 GiB as 1 GB drops 7%",
            value: (GIB / 1e9 - 1.0) * 100.0,
            sig: 1,
            shown: 7.0,
        },
        Quote {
            text: "an intensity of 0.08 FLOP/B",
            value: 1.0 / 12.0,
            sig: 1,
            shown: 0.08,
        },
        Quote {
            text: "about 90 ms for 10,000 copies",
            value: (10_000.0 * L_COPY + 40e6 / P) * 1e3,
            sig: 1,
            shown: 90.0,
        },
        Quote {
            text: "against 0.7 ms for one packed copy",
            value: (L_COPY + 40e6 / P) * 1e3,
            sig: 1,
            shown: 0.7,
        },
        Quote {
            text: "9 μs × 56 GB/s ≈ 500 KB",
            value: L_COPY * P / 1e3,
            sig: 2,
            shown: 500.0,
        },
        Quote {
            text: "under the 25 KB latency size",
            value: napkin::table::LATENCY_SIZE_SLOWDOWN * L_COPY * P / 1e3,
            sig: 2,
            shown: 25.0,
        },
        Quote {
            text: "F / M = 690 × 10¹² / 3 × 10¹² ≈ 230  FLOP/B",
            value: ridge,
            sig: 2,
            shown: 230.0,
        },
        Quote {
            text: "N / 3     = 230  →  N ≈ 690",
            value: 3.0 * round_sig(ridge, 2),
            sig: 2,
            shown: 690.0,
        },
        Quote {
            text: "above N ≈ 700 by the measured numbers",
            value: 3.0 * ridge,
            sig: 1,
            shown: 700.0,
        },
        Quote {
            text: "`1979 / 3.35 ≈ 590 FLOP/B`",
            value: sparse_ridge,
            sig: 2,
            shown: 590.0,
        },
        Quote {
            text: "crossover at N ≈ 1800",
            value: 3.0 * sparse_ridge,
            sig: 2,
            shown: 1800.0,
        },
        Quote {
            text: "`990 / 3.35 ≈ 300 FLOP/B`",
            value: dense_ceiling_ridge,
            sig: 1,
            shown: 300.0,
        },
        Quote {
            text: "N ≈ 900.",
            value: 3.0 * dense_ceiling_ridge,
            sig: 1,
            shown: 900.0,
        },
        Quote {
            text: "`2 × 512³ = 268M` FLOPs",
            value: 2.0 * 512_f64.powi(3) / 1e6,
            sig: 3,
            shown: 268.0,
        },
        Quote {
            text: "`6 × 512² = 1.57 MB`",
            value: 6.0 * 512_f64.powi(2) / 1e6,
            sig: 3,
            shown: 1.57,
        },
        Quote {
            text: "2.15G FLOPs over 6.29 MB",
            value: 2.0 * 1024_f64.powi(3) / 1e9,
            sig: 3,
            shown: 2.15,
        },
        Quote {
            text: "2.15G FLOPs over 6.29 MB",
            value: 6.0 * 1024_f64.powi(2) / 1e6,
            sig: 3,
            shown: 6.29,
        },
        Quote {
            text: "× 2 B = 131,072 B ≈ 131 KB",
            value: KV_PER_TOKEN,
            sig: 6,
            shown: 131_072.0,
        },
        Quote {
            text: "tokens × 131 KB ≈ 1.2 GB",
            value: per_request / 1e9,
            sig: 2,
            shown: 1.2,
        },
        Quote {
            text: "64 × 1.2 GB ≈ 77 GB",
            value: 64.0 * per_request / 1e9,
            sig: 2,
            shown: 77.0,
        },
        Quote {
            text: "+ 16 GB weights = 93 GB > 72 GB",
            value: (WEIGHTS + 64.0 * per_request) / 1e9,
            sig: 2,
            shown: 93.0,
        },
        Quote {
            text: "/ 1.2 GB per request ≈ 46 requests",
            value: largest_batch,
            sig: 2,
            shown: 46.0,
        },
        Quote {
            text: "`46 × 1.21 GB = 55.6 GB`",
            value: largest_batch * per_request / 1e9,
            sig: 3,
            shown: 55.6,
        },
        Quote {
            text: "mixing the two units silently costs\n2.4%",
            value: (KV_PER_TOKEN / 128e3 - 1.0) * 100.0,
            sig: 2,
            shown: 2.4,
        },
        Quote {
            text: "85.0 GB or 79.2 GiB",
            value: 85_017_624_576.0 / GIB,
            sig: 3,
            shown: 79.2,
        },
        Quote {
            text: "conservative by about 6%",
            value: (85_017_624_576.0 / 80e9 - 1.0) * 100.0,
            sig: 1,
            shown: 6.0,
        },
        Quote {
            text: "524 KB, and only 11 requests fit",
            value: 4.0 * KV_PER_TOKEN / 1e3,
            sig: 3,
            shown: 524.0,
        },
        Quote {
            text: "524 KB, and only 11 requests fit",
            value: ((USABLE - WEIGHTS) / (4.0 * per_request)).floor(),
            sig: 2,
            shown: 11.0,
        },
        Quote {
            text: "about 2 ms unfused",
            value: (6e9 / M + 3.0 * L_LAUNCH) * 1e3,
            sig: 1,
            shown: 2.0,
        },
        Quote {
            text: "and 0.7 ms fused",
            value: (2e9 / M + L_LAUNCH) * 1e3,
            sig: 1,
            shown: 0.7,
        },
        Quote {
            text: "1.5 × 10⁹ / 6 × 10⁹ = 0.25 FLOP/B",
            value: 1.5e9 / 6e9,
            sig: 2,
            shown: 0.25,
        },
        Quote {
            text: "3 × 10¹² B/s ≈ 0.7 μs`",
            value: 2e6 / M * 1e6,
            sig: 1,
            shown: 0.7,
        },
        Quote {
            text: "2,048 tokens ≈ 0.27 GB",
            value: decode_kv / 1e9,
            sig: 2,
            shown: 0.27,
        },
        Quote {
            text: "16 GB + 0.27 GB = 16.3 GB per step",
            value: (WEIGHTS + decode_kv) / 1e9,
            sig: 3,
            shown: 16.3,
        },
        Quote {
            text: "≈ 5.4 ms → 1 / 5.4 ms",
            value: (WEIGHTS + decode_kv) / M * 1e3,
            sig: 2,
            shown: 5.4,
        },
        Quote {
            text: "about 180 tokens/s at batch 1",
            value: batch_1_rate,
            sig: 2,
            shown: 180.0,
        },
        Quote {
            text: "32 × 0.27 GB = 24.6 GB per step",
            value: (WEIGHTS + 32.0 * decode_kv) / 1e9,
            sig: 3,
            shown: 24.6,
        },
        Quote {
            text: "≈ 8.2 ms → 32 / 8.2 ms",
            value: (WEIGHTS + 32.0 * decode_kv) / M * 1e3,
            sig: 2,
            shown: 8.2,
        },
        Quote {
            text: "3,900 tokens/s at batch 32",
            value: batch_32_rate,
            sig: 2,
            shown: 3900.0,
        },
        Quote {
            text: "20 times more",
            value: batch_32_rate / batch_1_rate,
            sig: 1,
            shown: 20.0,
        },
        Quote {
            text: "A step costs 50% more time",
            value: (WEIGHTS + 32.0 * decode_kv) / (WEIGHTS + decode_kv),
            sig: 2,
            shown: 1.5,
        },
        Quote {
            text: "/ 690 × 10¹² ≈ 0.7 ms",
            value: 2.0 * 8e9 * 32.0 / F * 1e3,
            sig: 1,
            shown: 0.7,
        },
        Quote {
            text: "each request reads\n1.1 GB per step",
            value: 8192.0 * KV_PER_TOKEN / 1e9,
            sig: 2,
            shown: 1.1,
        },
        Quote {
            text: "batch 32 reads 34 GB of KV",
            value: 32.0 * 8192.0 * KV_PER_TOKEN / 1e9,
            sig: 2,
            shown: 34.0,
        },
        Quote {
            text: "≈ 650K tokens per GPU-hour",
            value: 180.0 * 3600.0 / 1e3,
            sig: 2,
            shown: 650.0,
        },
        Quote {
            text: "about $6 per million tokens at batch 1",
            value: PRICE / (180.0 * 3600.0) * 1e6,
            sig: 1,
            shown: 6.0,
        },
        Quote {
            text: "≈ 14M tokens per GPU-hour",
            value: 3900.0 * 3600.0 / 1e6,
            sig: 2,
            shown: 14.0,
        },
        Quote {
            text: "$0.28 at batch 32",
            value: PRICE / (3900.0 * 3600.0) * 1e6,
            sig: 2,
            shown: 0.28,
        },
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
