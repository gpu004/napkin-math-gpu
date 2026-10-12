use napkin::profile;

#[test]
fn readme_napkin_blocks_match_renderers() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))
        .expect("README.md must exist at the repo root");
    let h100 = profile::find("h100").unwrap().unwrap();
    let all = profile::builtin().unwrap();
    let expected_blocks: Vec<(&str, String)> = vec![
        (
            "<!-- napkin table --gpu h100 -->",
            napkin::table::numbers(&h100, false).expect("h100 must have a measured profile"),
        ),
        (
            "<!-- napkin compare -->",
            napkin::table::compare(&all, false),
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
