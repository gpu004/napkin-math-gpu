use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use napkin_profile::collective::{Collective, CollectiveEvidence, Curve, CurvePoint, Topology};
use napkin_profile::{Ceiling, Dtype, Measured, Profile};
use serde::Deserialize;

const DATASHEET_IDS: [&str; 4] = ["a100", "h100", "b200", "b300"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PriceRecordInput {
    id: String,
    provider: String,
    sku: String,
    gpu: String,
    currency: String,
    amount: f64,
    per: PricePerInput,
    terms: PriceTermsInput,
    preemptible: bool,
    date: String,
    source: String,
    note: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum PricePerInput {
    Second,
    Minute,
    Hour,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum PriceTermsInput {
    OnDemand,
    Spot,
}

#[derive(Debug)]
struct ProfilePaths {
    datasheets: BTreeMap<String, PathBuf>,
    measured: BTreeMap<String, PathBuf>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("napkin build.rs: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_err(|e| e.to_string())?;
    let profiles_dir = Path::new(&manifest_dir).join("../../profiles");
    println!("cargo:rerun-if-changed={}", profiles_dir.display());
    let paths = profile_paths(&profiles_dir)?;
    for (id, path) in &paths.measured {
        if !paths.datasheets.contains_key(id) {
            return Err(format!(
                "measured profile {} has no matching datasheet file",
                path.display()
            ));
        }
    }
    let profiles = load_profiles(&paths.datasheets, &paths.measured)?;
    let nccl_dir = profiles_dir.join("nccl-tests");
    println!("cargo:rerun-if-changed={}", nccl_dir.display());
    let collectives = load_collectives(&nccl_dir, &profiles)?;
    let generated = emit_profiles(&profiles, &collectives);
    let out_dir = env::var("OUT_DIR").map_err(|e| e.to_string())?;
    let output = Path::new(&out_dir).join("profiles.rs");
    fs::write(&output, generated)
        .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
    let prices_dir = Path::new(&manifest_dir).join("../../prices");
    println!("cargo:rerun-if-changed={}", prices_dir.display());
    let prices = load_prices(&prices_dir)?;
    let output = Path::new(&out_dir).join("prices.rs");
    fs::write(&output, emit_prices(&prices)?)
        .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
    Ok(())
}

fn load_prices(prices_dir: &Path) -> Result<Vec<PriceRecordInput>, String> {
    let entries = fs::read_dir(prices_dir)
        .map_err(|error| format!("cannot read {}: {error}", prices_dir.display()))?;
    let mut paths = entries
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| format!("cannot read entry in {}: {error}", prices_dir.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    let mut records = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for path in paths {
        if !path.is_file()
            || path.extension().and_then(|extension| extension.to_str()) != Some("json")
        {
            continue;
        }
        let json = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let file_records: Vec<PriceRecordInput> = serde_json::from_str(&json)
            .map_err(|error| format!("invalid price JSON in {}: {error}", path.display()))?;
        for record in file_records {
            validate_price(&record, &path)?;
            if !ids.insert(record.id.clone()) {
                return Err(format!("duplicate price record id {:?}", record.id));
            }
            records.push(record);
        }
    }
    Ok(records)
}

fn validate_price(record: &PriceRecordInput, path: &Path) -> Result<(), String> {
    let source = path.display();
    if record.id.is_empty()
        || !record
            .id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(format!(
            "price record in {source} has invalid id {:?}; expected [a-z0-9-]+",
            record.id
        ));
    }
    for (name, value) in [
        ("provider", record.provider.as_str()),
        ("sku", record.sku.as_str()),
        ("note", record.note.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("price record {:?} has empty {name}", record.id));
        }
    }
    if !DATASHEET_IDS.contains(&record.gpu.as_str()) {
        return Err(format!(
            "price record {:?} has unknown gpu {:?}",
            record.id, record.gpu
        ));
    }
    if record.currency.len() != 3
        || !record
            .currency
            .bytes()
            .all(|byte| byte.is_ascii_uppercase())
    {
        return Err(format!(
            "price record {:?} has invalid currency {:?}",
            record.id, record.currency
        ));
    }
    if !(record.amount > 0.0 && record.amount.is_finite()) {
        return Err(format!(
            "price record {:?} has nonpositive or non-finite amount",
            record.id
        ));
    }
    if !record.source.starts_with("https://") {
        return Err(format!(
            "price record {:?} source must start with https://",
            record.id
        ));
    }
    if parse_date(&record.date).is_none() {
        return Err(format!(
            "price record {:?} has invalid calendar date {:?}",
            record.id, record.date
        ));
    }
    Ok(())
}

fn parse_date(value: &str) -> Option<(i32, u8, u8)> {
    if value.len() != 10
        || value.as_bytes().get(4) != Some(&b'-')
        || value.as_bytes().get(7) != Some(&b'-')
    {
        return None;
    }
    let year = value.get(..4)?.parse::<i32>().ok()?;
    let month = value.get(5..7)?.parse::<u8>().ok()?;
    let day = value.get(8..)?.parse::<u8>().ok()?;
    if !value
        .bytes()
        .enumerate()
        .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
        || !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
    {
        return None;
    }
    Some((year, month, day))
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn emit_prices(prices: &[PriceRecordInput]) -> Result<String, String> {
    let mut generated = String::from(
        "#[expect(clippy::unreadable_literal, reason = \"preserve source price precision\")]\nstatic GENERATED_PRICES: &[PriceRecord] = &[\n",
    );
    for record in prices {
        let (year, month, day) = parse_date(&record.date)
            .ok_or_else(|| format!("invalid date in price record {:?}", record.id))?;
        let per = match record.per {
            PricePerInput::Second => "Per::Second",
            PricePerInput::Minute => "Per::Minute",
            PricePerInput::Hour => "Per::Hour",
        };
        let terms = match record.terms {
            PriceTermsInput::OnDemand => "Terms::OnDemand",
            PriceTermsInput::Spot => "Terms::Spot",
        };
        let _ = writeln!(
            generated,
            "    PriceRecord {{ id: {:?}, provider: {:?}, sku: {:?}, gpu: {:?}, currency: {:?}, amount: {:?}_f64, per: {per}, terms: {terms}, preemptible: {}, date: Date {{ year: {year}, month: {month}, day: {day} }}, source: {:?}, note: {:?} }},",
            record.id,
            record.provider,
            record.sku,
            record.gpu,
            record.currency,
            record.amount,
            record.preemptible,
            record.source,
            record.note
        );
    }
    generated.push_str("];\n");
    Ok(generated)
}

fn profile_paths(profiles_dir: &Path) -> Result<ProfilePaths, String> {
    let mut datasheets = BTreeMap::new();
    let mut measured = BTreeMap::new();
    let entries = fs::read_dir(profiles_dir)
        .map_err(|error| format!("cannot read {}: {error}", profiles_dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("cannot read entry in {}: {error}", profiles_dir.display()))?
            .path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(id) = name.strip_suffix(".measured.json") {
            measured.insert(id.to_owned(), path);
        } else if let Some(id) = name.strip_suffix(".json") {
            if !DATASHEET_IDS.contains(&id) {
                return Err(format!(
                    "unexpected datasheet profile {} (id {id:?} is not in DATASHEET_IDS)",
                    path.display()
                ));
            }
            datasheets.insert(id.to_owned(), path);
        }
    }
    Ok(ProfilePaths {
        datasheets,
        measured,
    })
}

fn load_profiles(
    datasheet_paths: &BTreeMap<String, PathBuf>,
    measured_paths: &BTreeMap<String, PathBuf>,
) -> Result<Vec<Profile>, String> {
    let mut profiles = Vec::with_capacity(DATASHEET_IDS.len());
    for id in DATASHEET_IDS {
        let path = datasheet_paths.get(id).ok_or_else(|| {
            format!(
                "missing datasheet profile for {id:?}: expected {}",
                Path::new("profiles").join(format!("{id}.json")).display()
            )
        })?;
        let json = fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let mut profile: Profile = serde_json::from_str(&json)
            .map_err(|error| format!("invalid profile JSON in {}: {error}", path.display()))?;
        if profile.id != id {
            return Err(format!(
                "profile id {:?} in {} does not match file stem {id:?}",
                profile.id,
                path.display()
            ));
        }
        validate_ceiling(id, &profile.ceiling)?;
        if let Some(measured_path) = measured_paths.get(id) {
            let json = fs::read_to_string(measured_path)
                .map_err(|error| format!("cannot read {}: {error}", measured_path.display()))?;
            let measured: Measured = serde_json::from_str(&json).map_err(|error| {
                format!(
                    "invalid measured profile JSON in {}: {error}",
                    measured_path.display()
                )
            })?;
            validate_measured(id, &measured)?;
            profile.measured = Some(measured);
        }
        profiles.push(profile);
    }
    Ok(profiles)
}

fn validate_ceiling(id: &str, ceiling: &Ceiling) -> Result<(), String> {
    if ceiling.hbm_gbps <= 0.0 {
        return Err(format!("profile {id:?} has nonpositive ceiling.hbm_gbps"));
    }
    if ceiling.nvlink_gbps <= 0.0 {
        return Err(format!(
            "profile {id:?} has nonpositive ceiling.nvlink_gbps"
        ));
    }
    if ceiling.compute_tflops.is_empty() {
        return Err(format!(
            "profile {id:?} has an empty ceiling.compute_tflops map"
        ));
    }
    if ceiling.source.is_empty() {
        return Err(format!("profile {id:?} has an empty ceiling.source"));
    }
    Ok(())
}

fn validate_measured(id: &str, measured: &Measured) -> Result<(), String> {
    if let Some((name, _)) = measured.checks.iter().find(|(_, passed)| !**passed) {
        return Err(format!(
            "measured profile {id:?} has failing check {name:?}"
        ));
    }
    Ok(())
}

fn load_collectives(
    nccl_dir: &Path,
    profiles: &[Profile],
) -> Result<Vec<CollectiveEvidence>, String> {
    let entries = fs::read_dir(nccl_dir)
        .map_err(|error| format!("cannot read {}: {error}", nccl_dir.display()))?;
    let required = [
        "run.json",
        "topo.txt",
        "nccl_topology.txt",
        "all_reduce_perf.txt",
        "all_reduce_perf.json",
        "all_gather_perf.txt",
        "all_gather_perf.json",
        "reduce_scatter_perf.txt",
        "reduce_scatter_perf.json",
    ];
    let mut evidence = Vec::new();
    for entry in entries {
        let directory = entry
            .map_err(|error| format!("cannot read entry in {}: {error}", nccl_dir.display()))?
            .path();
        if !directory.is_dir() {
            continue;
        }
        let name = directory
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("invalid NCCL evidence directory {}", directory.display()))?;
        let gpu = name
            .split_once('x')
            .map(|(gpu, _)| gpu)
            .ok_or_else(|| format!("NCCL evidence directory {name:?} must be <gpu>x<count>"))?;
        if !DATASHEET_IDS.contains(&gpu) {
            return Err(format!(
                "NCCL evidence directory {name:?} has GPU id {gpu:?} not in DATASHEET_IDS"
            ));
        }
        for file in required {
            let path = directory.join(file);
            if !path.is_file() {
                return Err(format!("missing NCCL evidence file {}", path.display()));
            }
        }
        let read = |name: &str| -> Result<String, String> {
            let path = directory.join(name);
            fs::read_to_string(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))
        };
        let logical_dir = format!("profiles/nccl-tests/{name}");
        let run_json = read("run.json")?;
        let reduce = read("all_reduce_perf.json")?;
        let gather = read("all_gather_perf.json")?;
        let scatter = read("reduce_scatter_perf.json")?;
        let item = napkin_profile::collective::parse_evidence(
            &logical_dir,
            &run_json,
            [&reduce, &gather, &scatter],
        )
        .map_err(|error| format!("invalid NCCL evidence in {logical_dir}: {error}"))?;
        if let Some(profile) = profiles.iter().find(|profile| profile.id == item.gpu)
            && let Some(measured) = &profile.measured
            && measured.device_name != item.device_name
        {
            return Err(format!(
                "NCCL evidence in {logical_dir} names {:?}, measured profile {} names {:?}",
                item.device_name, item.gpu, measured.device_name
            ));
        }
        if evidence.iter().any(|existing: &CollectiveEvidence| {
            existing.gpu == item.gpu
                && existing.gpu_count == item.gpu_count
                && existing.topology == item.topology
                && existing.dtype == item.dtype
        }) {
            return Err(format!(
                "duplicate NCCL evidence for {} × {} {} {}",
                item.gpu, item.gpu_count, item.topology, item.dtype
            ));
        }
        evidence.push(item);
    }
    evidence.sort_by(|left, right| {
        (
            &left.gpu,
            left.gpu_count,
            left.topology.to_string(),
            left.dtype.to_string(),
        )
            .cmp(&(
                &right.gpu,
                right.gpu_count,
                right.topology.to_string(),
                right.dtype.to_string(),
            ))
    });
    Ok(evidence)
}

fn emit_profiles(profiles: &[Profile], collectives: &[CollectiveEvidence]) -> String {
    let mut generated = String::from("fn generated_profiles() -> Vec<Profile> {\n    vec![\n");
    for profile in profiles {
        generated.push_str("        generated_profile_");
        generated.push_str(&profile.id);
        generated.push_str("(),\n");
    }
    generated.push_str("    ]\n}\n");
    for profile in profiles {
        emit_profile(profile, &mut generated);
    }
    generated.push_str("fn generated_collectives() -> Vec<CollectiveEvidence> {\n    vec![\n");
    for (index, _) in collectives.iter().enumerate() {
        let _ = writeln!(generated, "        generated_collective_{index}(),");
    }
    generated.push_str("    ]\n}\n");
    for (index, evidence) in collectives.iter().enumerate() {
        emit_collective(index, evidence, &mut generated);
        for (curve_index, curve) in evidence.curves.iter().enumerate() {
            emit_curve(index, curve_index, curve, &mut generated);
        }
    }
    generated
}

fn emit_collective(index: usize, evidence: &CollectiveEvidence, generated: &mut String) {
    let fields = [
        ("gpu", string_literal(&evidence.gpu)),
        ("device_name", string_literal(&evidence.device_name)),
        ("gpu_count", format!("{}_u32", evidence.gpu_count)),
        ("topology", topology_literal(evidence.topology).to_owned()),
        ("dtype", dtype_literal(evidence.dtype).to_owned()),
        ("redop", string_literal(&evidence.redop)),
        ("rank_layout", string_literal(&evidence.rank_layout)),
        ("modal_request", string_literal(&evidence.modal_request)),
        ("captured_at", string_literal(&evidence.captured_at)),
        ("driver_version", string_literal(&evidence.driver_version)),
        ("nccl_version", string_literal(&evidence.nccl_version)),
        ("nccl_tests", string_literal(&evidence.nccl_tests)),
        ("raw_dir", string_literal(&evidence.raw_dir)),
    ];
    let _ = writeln!(
        generated,
        "fn generated_collective_{index}() -> CollectiveEvidence {{\n    CollectiveEvidence {{"
    );
    for (field, value) in fields {
        emit_field(generated, "        ", field, &value);
    }
    generated.push_str("        curves: vec![\n");
    for curve_index in 0..evidence.curves.len() {
        let _ = writeln!(
            generated,
            "            generated_curve_{index}_{curve_index}(),"
        );
    }
    generated.push_str("        ],\n    }\n}\n");
}

fn emit_curve(evidence_index: usize, curve_index: usize, curve: &Curve, generated: &mut String) {
    let collective = match curve.collective {
        Collective::AllReduce => "Collective::AllReduce",
        Collective::AllGather => "Collective::AllGather",
        Collective::ReduceScatter => "Collective::ReduceScatter",
    };
    let _ = writeln!(
        generated,
        "#[expect(clippy::unreadable_literal, reason = \"preserve NCCL evidence precision\")]\nfn generated_curve_{evidence_index}_{curve_index}() -> Curve {{\n    Curve {{\n        collective: {collective},\n        command: {},\n        points: vec![",
        string_literal(&curve.command),
    );
    for point in &curve.points {
        emit_curve_point(point, generated);
    }
    generated.push_str("        ],\n    }\n}\n");
}

fn emit_curve_point(point: &CurvePoint, generated: &mut String) {
    let _ = writeln!(
        generated,
        "            CurvePoint {{ bytes: {}_u64, time_us: {}, algbw_gbps: {}, busbw_gbps: {} }},",
        point.bytes,
        float_literal(point.time_us),
        float_literal(point.algbw_gbps),
        float_literal(point.busbw_gbps)
    );
}

fn topology_literal(topology: Topology) -> &'static str {
    match topology {
        Topology::Nvlink => "Topology::Nvlink",
        Topology::Pcie => "Topology::Pcie",
    }
}

fn emit_profile(profile: &Profile, generated: &mut String) {
    let Profile {
        id,
        name,
        architecture,
        modal_gpu,
        memory_gb,
        ceiling,
        measured,
    } = profile;
    let Ceiling {
        hbm_gbps,
        nvlink_gbps,
        compute_tflops,
        source,
    } = ceiling;
    if measured.is_some() {
        generated.push_str(
            "#[expect(clippy::unreadable_literal, reason = \"preserve source profile precision\")]\n",
        );
    }
    generated.push_str("fn generated_profile_");
    generated.push_str(id);
    generated.push_str("() -> Profile {\n    Profile {\n");
    emit_field(generated, "        ", "id", &string_literal(id));
    emit_field(generated, "        ", "name", &string_literal(name));
    emit_field(
        generated,
        "        ",
        "architecture",
        &string_literal(architecture),
    );
    emit_field(
        generated,
        "        ",
        "modal_gpu",
        &string_literal(modal_gpu),
    );
    emit_field(
        generated,
        "        ",
        "memory_gb",
        &float_literal(*memory_gb),
    );
    generated.push_str("        ceiling: Ceiling {\n");
    emit_field(
        generated,
        "            ",
        "hbm_gbps",
        &float_literal(*hbm_gbps),
    );
    emit_field(
        generated,
        "            ",
        "nvlink_gbps",
        &float_literal(*nvlink_gbps),
    );
    emit_field(
        generated,
        "            ",
        "compute_tflops",
        &dtype_map(compute_tflops),
    );
    emit_field(generated, "            ", "source", &string_literal(source));
    generated.push_str("        },\n");
    let measured = measured
        .as_ref()
        .map_or_else(|| "None".to_owned(), emit_measured);
    emit_field(generated, "        ", "measured", &measured);
    generated.push_str("    }\n}\n");
}

fn emit_measured(measured: &Measured) -> String {
    let Measured {
        captured_at,
        gpu_request,
        device_name,
        device_count,
        compute_capability,
        sm_count,
        memory_bytes,
        driver_version,
        cuda_version,
        rounds,
        launch_us,
        sync_us,
        host_copy_gbps,
        h2d_gbps,
        d2h_gbps,
        pcie_gen,
        pcie_gen_max,
        pcie_width,
        pcie_width_max,
        h2d_latency_us,
        d2h_latency_us,
        hbm_gbps,
        elementwise_gbps,
        hbm_latency_ns,
        g2g_gbps,
        g2g_forward_gbps,
        g2g_reverse_gbps,
        g2g_latency_us,
        compute_tflops,
        gemm_shape,
        checks,
        samples_ms,
    } = measured;
    let fields = [
        ("captured_at", string_literal(captured_at)),
        ("gpu_request", string_literal(gpu_request)),
        ("device_name", string_literal(device_name)),
        ("device_count", format!("{device_count}_u32")),
        ("compute_capability", string_literal(compute_capability)),
        ("sm_count", format!("{sm_count}_u32")),
        ("memory_bytes", format!("{memory_bytes}_u64")),
        ("driver_version", string_literal(driver_version)),
        ("cuda_version", string_literal(cuda_version)),
        ("rounds", format!("{rounds}_u32")),
        ("launch_us", float_literal(*launch_us)),
        ("sync_us", option_float_literal(*sync_us)),
        ("host_copy_gbps", float_literal(*host_copy_gbps)),
        ("h2d_gbps", float_literal(*h2d_gbps)),
        ("d2h_gbps", float_literal(*d2h_gbps)),
        ("pcie_gen", option_u8_literal(*pcie_gen)),
        ("pcie_gen_max", option_u8_literal(*pcie_gen_max)),
        ("pcie_width", option_u8_literal(*pcie_width)),
        ("pcie_width_max", option_u8_literal(*pcie_width_max)),
        ("h2d_latency_us", option_float_literal(*h2d_latency_us)),
        ("d2h_latency_us", option_float_literal(*d2h_latency_us)),
        ("hbm_gbps", float_literal(*hbm_gbps)),
        ("elementwise_gbps", float_literal(*elementwise_gbps)),
        ("hbm_latency_ns", option_float_literal(*hbm_latency_ns)),
        ("g2g_gbps", option_float_literal(*g2g_gbps)),
        ("g2g_forward_gbps", option_float_literal(*g2g_forward_gbps)),
        ("g2g_reverse_gbps", option_float_literal(*g2g_reverse_gbps)),
        ("g2g_latency_us", option_float_literal(*g2g_latency_us)),
        ("compute_tflops", dtype_map(compute_tflops)),
        (
            "gemm_shape",
            format!(
                "[{}_u32, {}_u32, {}_u32]",
                gemm_shape[0], gemm_shape[1], gemm_shape[2]
            ),
        ),
        ("checks", bool_map(checks)),
        ("samples_ms", samples_map(samples_ms)),
    ];
    let mut generated = String::from("Some(Measured {\n");
    for (name, value) in fields {
        emit_field(&mut generated, "            ", name, &value);
    }
    generated.push_str("        })");
    generated
}

fn emit_field(generated: &mut String, indent: &str, name: &str, value: &str) {
    let _ = writeln!(generated, "{indent}{name}: {value},");
}

fn string_literal(value: &str) -> String {
    format!("{value:?}.to_owned()")
}

fn float_literal(value: f64) -> String {
    format!("{value:?}_f64")
}

fn option_float_literal(value: Option<f64>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", float_literal(value)),
    )
}

fn option_u8_literal(value: Option<u8>) -> String {
    value.map_or_else(|| "None".to_owned(), |value| format!("Some({value}_u8)"))
}

fn dtype_literal(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Fp4 => "Dtype::Fp4",
        Dtype::Fp8 => "Dtype::Fp8",
        Dtype::Fp16 => "Dtype::Fp16",
        Dtype::Bf16 => "Dtype::Bf16",
        Dtype::Tf32 => "Dtype::Tf32",
        Dtype::Fp32 => "Dtype::Fp32",
    }
}

fn dtype_map(values: &BTreeMap<Dtype, f64>) -> String {
    let entries = values
        .iter()
        .map(|(dtype, value)| format!("({}, {})", dtype_literal(*dtype), float_literal(*value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("BTreeMap::from([{entries}])")
}

fn bool_map(values: &BTreeMap<String, bool>) -> String {
    let entries = values
        .iter()
        .map(|(name, value)| format!("({}, {value})", string_literal(name)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("BTreeMap::from([{entries}])")
}

fn samples_map(values: &BTreeMap<String, Vec<f64>>) -> String {
    let entries = values
        .iter()
        .map(|(name, samples)| {
            let samples = samples
                .iter()
                .map(|sample| float_literal(*sample))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({}, vec![{samples}])", string_literal(name))
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("BTreeMap::from([{entries}])")
}
