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
