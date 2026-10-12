//! Shared types for topology-aware NCCL collective estimates.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Dtype;

/// NCCL collective algorithms supported by the measured profile data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Collective {
    AllReduce,
    AllGather,
    ReduceScatter,
}

impl Collective {
    pub const ALL: [Self; 3] = [Self::AllReduce, Self::AllGather, Self::ReduceScatter];

    #[must_use]
    pub const fn binary(self) -> &'static str {
        match self {
            Self::AllReduce => "all_reduce_perf",
            Self::AllGather => "all_gather_perf",
            Self::ReduceScatter => "reduce_scatter_perf",
        }
    }

    #[must_use]
    pub fn bus_factor(self, ranks: u32) -> f64 {
        let neighbors = f64::from(ranks.saturating_sub(1));
        let ranks = f64::from(ranks.max(1));
        match self {
            Self::AllReduce => 2.0 * neighbors / ranks,
            Self::AllGather | Self::ReduceScatter => neighbors / ranks,
        }
    }

    #[must_use]
    pub const fn size_convention(self) -> &'static str {
        match self {
            Self::AllReduce => "per-rank buffer",
            Self::AllGather => "total gathered output",
            Self::ReduceScatter => "total per-rank input",
        }
    }
}

impl fmt::Display for Collective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::AllReduce => "AllReduce",
            Self::AllGather => "AllGather",
            Self::ReduceScatter => "ReduceScatter",
        })
    }
}

/// The measured communication path used by NCCL.
///
/// `nvlink` means one node where NCCL built every ring over `NVLink` and moved
/// data GPU-to-GPU via P2P, without host memory or a network. `pcie` is
/// available as a name, but estimates require matching measurement evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Topology {
    Nvlink,
    Pcie,
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nvlink => "nvlink",
            Self::Pcie => "pcie",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurvePoint {
    pub bytes: u64,
    pub time_us: f64,
    pub algbw_gbps: f64,
    pub busbw_gbps: f64,
}

#[derive(Clone, Debug)]
pub struct Curve {
    pub collective: Collective,
    pub command: String,
    pub points: Vec<CurvePoint>,
}

#[derive(Clone, Debug)]
pub struct CollectiveEvidence {
    pub gpu: String,
    pub device_name: String,
    pub gpu_count: u32,
    pub topology: Topology,
    pub dtype: Dtype,
    pub redop: String,
    pub rank_layout: String,
    pub modal_request: String,
    pub captured_at: String,
    pub driver_version: String,
    pub nccl_version: String,
    pub nccl_tests: String,
    pub raw_dir: String,
    pub curves: Vec<Curve>,
}

impl CollectiveEvidence {
    #[must_use]
    pub fn curve(&self, collective: Collective) -> Option<&Curve> {
        self.curves
            .iter()
            .find(|curve| curve.collective == collective)
    }
}

#[derive(Deserialize)]
struct TestJson {
    args: Vec<String>,
    config: TestConfig,
    results: Vec<TestRow>,
    out_of_bounds: OutOfBounds,
}

#[derive(Deserialize)]
struct TestConfig {
    nthreads: u32,
    ngpus: u32,
    validation: u32,
}

#[derive(Deserialize)]
struct OutOfBounds {
    okay: String,
}

#[derive(Deserialize)]
struct TestRow {
    size: u64,
    r#type: String,
    redop: String,
    out_of_place: TestResult,
    in_place: TestResult,
}

#[derive(Deserialize)]
struct TestResult {
    time: f64,
    alg_bw: f64,
    bus_bw: f64,
    nwrong: f64,
}

fn dtype_nccl_name(dtype: Dtype) -> Result<&'static str, String> {
    match dtype {
        Dtype::Bf16 => Ok("bfloat16"),
        Dtype::Fp16 => Ok("half"),
        Dtype::Fp32 => Ok("float"),
        Dtype::Fp4 | Dtype::Fp8 | Dtype::Tf32 => Err(format!(
            "NCCL Tests does not support collective dtype {dtype}"
        )),
    }
}

fn bytes_as_f64(bytes: u64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "NCCL collective sizes in these profiles are exactly representable"
    )]
    {
        bytes as f64
    }
}

fn bandwidth_matches(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= (expected.abs() * 1e-3).max(1e-5)
}

fn validate_test_json(
    collective: Collective,
    ranks: u32,
    document: &TestJson,
) -> Result<(), String> {
    if !document
        .args
        .first()
        .is_some_and(|binary| binary.ends_with(collective.binary()))
    {
        return Err(format!(
            "{} JSON args[0] must end in {}",
            collective.binary(),
            collective.binary()
        ));
    }
    if document.config.validation != 1 {
        return Err(format!(
            "{} correctness checks were off",
            collective.binary()
        ));
    }
    if document.out_of_bounds.okay != "true" {
        return Err(format!(
            "{} out_of_bounds.okay was not \"true\"",
            collective.binary()
        ));
    }
    if u64::from(document.config.ngpus) * u64::from(document.config.nthreads) != u64::from(ranks) {
        return Err(format!(
            "{} rank mismatch: {} GPUs × {} threads does not match {ranks} ranks",
            collective.binary(),
            document.config.ngpus,
            document.config.nthreads
        ));
    }
    Ok(())
}

fn parse_point(
    collective: Collective,
    expected_type: &str,
    expected_redop: &str,
    bus_factor: f64,
    row: TestRow,
) -> Result<Option<CurvePoint>, String> {
    if row.out_of_place.nwrong != 0.0 || row.in_place.nwrong != 0.0 {
        return Err(format!(
            "{} size {} has incorrect results (nwrong != 0)",
            collective.binary(),
            row.size
        ));
    }
    if row.r#type != expected_type {
        return Err(format!(
            "{} size {} has dtype {:?}, expected {expected_type:?}",
            collective.binary(),
            row.size,
            row.r#type
        ));
    }
    if row.redop != expected_redop {
        return Err(format!(
            "{} size {} has redop {:?}, expected {expected_redop:?}",
            collective.binary(),
            row.size,
            row.redop
        ));
    }
    let result = row.out_of_place;
    if !result.time.is_finite() || result.time <= 0.0 {
        return Err(format!(
            "{} size {} has non-finite or nonpositive time",
            collective.binary(),
            row.size
        ));
    }
    if row.size == 0 {
        return Ok(None);
    }
    let recomputed_algbw = bytes_as_f64(row.size) / (result.time * 1e3);
    let recomputed_busbw = result.alg_bw * bus_factor;
    if !result.alg_bw.is_finite()
        || !result.bus_bw.is_finite()
        || !bandwidth_matches(result.alg_bw, recomputed_algbw)
        || !bandwidth_matches(result.bus_bw, recomputed_busbw)
    {
        return Err(format!(
            "{} size {} bandwidth does not match the NCCL Tests size convention or bus factor",
            collective.binary(),
            row.size
        ));
    }
    Ok(Some(CurvePoint {
        bytes: row.size,
        time_us: result.time,
        algbw_gbps: result.alg_bw,
        busbw_gbps: result.bus_bw,
    }))
}

/// Parses and validates one NCCL Tests JSON curve.
///
/// # Errors
/// Returns a description of invalid arguments, validation evidence, rows, or
/// size/bandwidth conventions.
pub fn parse_curve(
    collective: Collective,
    ranks: u32,
    dtype: Dtype,
    json: &str,
) -> Result<Vec<CurvePoint>, String> {
    let expected_type = dtype_nccl_name(dtype)?;
    let document: TestJson = serde_json::from_str(json)
        .map_err(|error| format!("invalid {} JSON: {error}", collective.binary()))?;
    validate_test_json(collective, ranks, &document)?;

    let expected_redop = match collective {
        Collective::AllGather => "none",
        Collective::AllReduce | Collective::ReduceScatter => "sum",
    };
    let bus_factor = collective.bus_factor(ranks);
    let mut points = Vec::new();
    for row in document.results {
        let Some(point) = parse_point(collective, expected_type, expected_redop, bus_factor, row)?
        else {
            continue;
        };
        if points
            .last()
            .is_some_and(|previous: &CurvePoint| point.bytes <= previous.bytes)
        {
            return Err(format!(
                "{} sizes are not strictly increasing after dropping size-0 rows",
                collective.binary()
            ));
        }
        points.push(point);
    }
    if points.is_empty() {
        return Err(format!("{} curve is empty", collective.binary()));
    }
    Ok(points)
}

#[derive(Deserialize)]
struct RunJson {
    captured_at: String,
    modal_request: String,
    gpu_count: u32,
    rank_layout: String,
    topology: Topology,
    dtype: String,
    redop: String,
    nccl_tests: NcclTests,
    nvidia_smi: Vec<String>,
    commands: std::collections::BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct NcclTests {
    tag: String,
    commit: String,
}

fn parse_directory(raw_dir: &str) -> Result<(String, u32), String> {
    let directory = Path::new(raw_dir)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid NCCL evidence directory {raw_dir:?}"))?;
    let (gpu, count) = directory
        .split_once('x')
        .ok_or_else(|| format!("NCCL evidence directory {directory:?} must be <gpu>x<count>"))?;
    let gpu_count = count
        .parse::<u32>()
        .map_err(|error| format!("invalid GPU count in directory {directory:?}: {error}"))?;
    if gpu.is_empty() || gpu_count == 0 {
        return Err(format!(
            "NCCL evidence directory {directory:?} must name a GPU and positive count"
        ));
    }
    Ok((gpu.to_owned(), gpu_count))
}

fn parse_dtype(name: &str) -> Result<Dtype, String> {
    match name {
        "bfloat16" => Ok(Dtype::Bf16),
        "half" | "fp16" => Ok(Dtype::Fp16),
        "float" | "fp32" => Ok(Dtype::Fp32),
        _ => Err(format!("unsupported NCCL evidence dtype {name:?}")),
    }
}

fn nccl_version(json: &str) -> Result<u32, String> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|error| format!("invalid NCCL Tests JSON: {error}"))?;
    value
        .get("nccl_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| "NCCL Tests JSON is missing a valid nccl_version".to_owned())
}

fn format_nccl_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        version / 10_000,
        (version % 10_000) / 100,
        version % 100
    )
}

/// Parses run metadata and its three validated curve documents.
///
/// `curves` must be in [`Collective::ALL`] order. The directory basename is
/// the profile id and GPU count, such as `h100x2`.
///
/// # Errors
/// Returns a description of invalid run metadata, device identity, or curves.
pub fn parse_evidence(
    raw_dir: &str,
    run_json: &str,
    curves: [&str; 3],
) -> Result<CollectiveEvidence, String> {
    let (gpu, directory_count) = parse_directory(raw_dir)?;
    let run: RunJson = serde_json::from_str(run_json)
        .map_err(|error| format!("invalid NCCL run.json in {raw_dir}: {error}"))?;
    if directory_count != run.gpu_count {
        return Err(format!(
            "NCCL directory {raw_dir} encodes {directory_count} GPUs, run.json reports {}",
            run.gpu_count
        ));
    }
    let dtype = parse_dtype(&run.dtype)?;
    let mut smi_rows = run.nvidia_smi.iter().skip(1);
    let first = smi_rows
        .next()
        .ok_or_else(|| format!("NCCL run.json in {raw_dir} has no nvidia-smi device rows"))?;
    let first_fields = first.split(',').map(str::trim).collect::<Vec<_>>();
    if first_fields.len() < 3 {
        return Err(format!(
            "NCCL nvidia-smi row in {raw_dir} has fewer than three columns"
        ));
    }
    let device_name = first_fields[1].to_owned();
    let driver_version = first_fields[2].to_owned();
    let mut row_count = 1_u32;
    for row in smi_rows {
        let fields = row.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() < 3 || fields[1] != device_name || fields[2] != driver_version {
            return Err(format!(
                "NCCL nvidia-smi rows in {raw_dir} have inconsistent device name or driver"
            ));
        }
        row_count += 1;
    }
    if row_count != run.gpu_count {
        return Err(format!(
            "NCCL nvidia-smi in {raw_dir} has {row_count} device rows, expected {}",
            run.gpu_count
        ));
    }

    let version = nccl_version(curves[0])?;
    let mut parsed_curves = Vec::with_capacity(Collective::ALL.len());
    for (collective, json) in Collective::ALL.into_iter().zip(curves) {
        let curve_version = nccl_version(json)?;
        if curve_version != version {
            return Err(format!(
                "NCCL version mismatch in {}: expected {}, found {}",
                collective.binary(),
                format_nccl_version(version),
                format_nccl_version(curve_version)
            ));
        }
        let points = parse_curve(collective, run.gpu_count, dtype, json)
            .map_err(|error| format!("invalid {} evidence: {error}", collective.binary()))?;
        let command = run
            .commands
            .get(collective.binary())
            .ok_or_else(|| {
                format!(
                    "NCCL run.json in {raw_dir} has no command for {}",
                    collective.binary()
                )
            })?
            .clone();
        parsed_curves.push(Curve {
            collective,
            command,
            points,
        });
    }
    let commit_short = run.nccl_tests.commit.chars().take(7).collect::<String>();
    Ok(CollectiveEvidence {
        gpu,
        device_name,
        gpu_count: run.gpu_count,
        topology: run.topology,
        dtype,
        redop: run.redop,
        rank_layout: run.rank_layout,
        modal_request: run.modal_request,
        captured_at: run.captured_at,
        driver_version,
        nccl_version: format_nccl_version(version),
        nccl_tests: format!("{} ({commit_short})", run.nccl_tests.tag),
        raw_dir: raw_dir.to_owned(),
        curves: parsed_curves,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUN: &str = include_str!("../fixtures/nccl-tests/h100x2/run.json");
    const REDUCE: &str = include_str!("../fixtures/nccl-tests/h100x2/all_reduce_perf.json");
    const GATHER: &str = include_str!("../fixtures/nccl-tests/h100x2/all_gather_perf.json");
    const SCATTER: &str = include_str!("../fixtures/nccl-tests/h100x2/reduce_scatter_perf.json");

    fn parse(collective: Collective, json: &str) -> Result<Vec<CurvePoint>, String> {
        parse_curve(collective, 2, Dtype::Bf16, json)
    }

    #[test]
    fn parses_literal_collective_points() -> Result<(), String> {
        let reduce = parse(Collective::AllReduce, REDUCE)?;
        assert_eq!(reduce[0].bytes, 8);
        assert!((reduce[0].time_us - 10.4621).abs() < 1e-9);
        assert!((reduce[0].busbw_gbps - 0.000_765).abs() < 1e-9);

        let gather = parse(Collective::AllGather, GATHER)?;
        assert_eq!(gather[0].bytes, 32);
        assert!((gather[0].time_us - 12.14735).abs() < 1e-9);
        assert!((gather[0].busbw_gbps - 0.001_317).abs() < 1e-9);

        let scatter = parse(Collective::ReduceScatter, SCATTER)?;
        assert_eq!(scatter[0].bytes, 32);
        assert!((scatter[0].time_us - 12.73735).abs() < 1e-9);
        assert!((scatter[0].busbw_gbps - 0.001_256).abs() < 1e-9);
        assert_eq!(reduce.last().map(|point| point.bytes), Some(4_294_967_296));
        assert_eq!(gather.last().map(|point| point.bytes), Some(4_294_967_296));
        assert_eq!(scatter.last().map(|point| point.bytes), Some(4_294_967_296));
        Ok(())
    }

    #[test]
    fn drops_zero_size_rows_from_gather_and_scatter() -> Result<(), String> {
        assert_eq!(parse(Collective::AllGather, GATHER)?.len(), 5);
        assert_eq!(parse(Collective::ReduceScatter, SCATTER)?.len(), 5);
        Ok(())
    }

    #[test]
    fn rejects_invalid_curve_evidence() {
        let wrongs = REDUCE.replace(r#""nwrong":0.0"#, r#""nwrong":1.0"#);
        let validation = REDUCE.replace(r#""validation": 1"#, r#""validation": 0"#);
        let bounds = REDUCE.replace(r#""okay": "true""#, r#""okay": "false""#);
        let binary = REDUCE.replace("all_reduce_perf", "all_gather_perf");
        let redop = REDUCE.replace(r#""redop":"sum""#, r#""redop":"none""#);
        let busbw = REDUCE.replace(r#""bus_bw":0.000765"#, r#""bus_bw":0.01"#);
        assert!(parse(Collective::AllReduce, &wrongs).is_err());
        assert!(parse(Collective::AllReduce, &validation).is_err());
        assert!(parse(Collective::AllReduce, &bounds).is_err());
        assert!(parse(Collective::AllReduce, &binary).is_err());
        assert!(parse_curve(Collective::AllReduce, 4, Dtype::Bf16, REDUCE).is_err());
        assert!(parse(Collective::AllReduce, &redop).is_err());
        assert!(parse(Collective::AllReduce, &busbw).is_err());
        assert!(parse_curve(Collective::AllReduce, 2, Dtype::Fp8, REDUCE).is_err());
    }

    #[test]
    fn parses_real_run_evidence() -> Result<(), String> {
        let evidence =
            parse_evidence("profiles/nccl-tests/h100x2", RUN, [REDUCE, GATHER, SCATTER])?;
        assert_eq!(evidence.gpu, "h100");
        assert_eq!(evidence.gpu_count, 2);
        assert_eq!(evidence.topology, Topology::Nvlink);
        assert_eq!(evidence.dtype, Dtype::Bf16);
        assert_eq!(evidence.nccl_version, "2.27.3");
        assert_eq!(evidence.device_name, "NVIDIA H100 80GB HBM3");
        assert_eq!(evidence.driver_version, "580.95.05");
        assert_eq!(evidence.nccl_tests, "v2.21.1 (afd59ab)");
        Ok(())
    }
}
