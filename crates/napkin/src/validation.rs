//! Prediction-versus-observation cases embedded from `validation/`.

use serde::Deserialize;

use crate::estimate::{self, CopyMode, Estimate, Mode, Transfer, Workload};
use crate::profile::{self, Dtype};

struct EmbeddedCase {
    id: &'static str,
    case_json: &'static str,
    run_json: &'static str,
    observation_json: &'static str,
    finding: &'static str,
    source_files: &'static [&'static str],
}

include!(concat!(env!("OUT_DIR"), "/validation.rs"));

/// A parsed case and its prediction, observation, and run metadata.
#[derive(Debug)]
pub struct Case {
    pub id: String,
    pub title: String,
    pub gpu: String,
    pub modal_request: String,
    pub job: String,
    pub derivation: Vec<String>,
    pub match_items: Vec<String>,
    pub excludes: Vec<String>,
    pub finding: String,
    pub observation_file: String,
    pub source_files: Vec<String>,
    pub observation_tool: String,
    pub observation_metric: String,
    pub workload: Workload,
    pub mode: Mode,
    pub estimate: Estimate,
    pub observed_s: f64,
    pub ratio: f64,
    pub tolerance_low: f64,
    pub tolerance_high: f64,
    pub tolerance_why: String,
    pub verdict: String,
    pub samples: Option<usize>,
    pub number_per_run: Option<u64>,
    pub directions: Vec<Direction>,
    pub run: RunInfo,
}

/// One observed `NVBandwidth` copy direction.
#[derive(Debug)]
pub struct Direction {
    pub name: String,
    pub bandwidth_gbps: f64,
    pub bandwidth_text: String,
    pub seconds: f64,
}

/// Runtime metadata recorded by the Modal runner.
#[derive(Debug, Default, Deserialize)]
pub struct RunInfo {
    pub captured_at: Option<String>,
    pub modal_request: Option<String>,
    #[serde(default)]
    pub gpus: Vec<GpuInfo>,
    pub driver: Option<String>,
    pub cuda_driver: Option<String>,
    pub torch: Option<String>,
    pub torch_cuda: Option<String>,
    pub cublas: Option<String>,
    pub tag: Option<String>,
    pub commit: Option<String>,
    pub profiled_cuda_kernels: Option<u64>,
    pub case_seconds: Option<f64>,
    pub monitor: Option<MonitorInfo>,
}

/// GPU identity captured by Modal.
#[derive(Debug, Default, Deserialize)]
pub struct GpuInfo {
    pub name: Option<String>,
}

/// Peak link and sampled operating range captured during a case.
#[derive(Debug, Default, Deserialize)]
pub struct MonitorInfo {
    pub pcie_gen_peak: Option<f64>,
    pub pcie_width_peak: Option<f64>,
    pub sm_clock_mhz: Option<RangeInfo>,
    pub power_w: Option<RangeInfo>,
}

/// Minimum and maximum for a sampled run metric.
#[derive(Debug, Default, Deserialize)]
pub struct RangeInfo {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseInput {
    id: String,
    title: String,
    gpu: String,
    modal_request: String,
    job: String,
    workload: WorkloadInput,
    derivation: Vec<String>,
    observation: ObservationInput,
    tolerance: ToleranceInput,
    #[serde(rename = "match")]
    match_items: Vec<String>,
    excludes: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkloadInput {
    dtype: Dtype,
    flops: f64,
    hbm_bytes: f64,
    #[serde(default)]
    h2d: Option<TransferInput>,
    #[serde(default)]
    d2h: Option<TransferInput>,
    #[serde(default)]
    copy_mode: CopyModeInput,
    #[serde(default)]
    launches: u64,
    mode: WorkloadModeInput,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum CopyModeInput {
    #[default]
    Sync,
    Async,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum WorkloadModeInput {
    Serial,
    Overlap,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransferInput {
    bytes: f64,
    copies: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationInput {
    kind: ObservationKind,
    file: String,
    tool: String,
    metric: String,
    #[serde(default)]
    testcases: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ObservationKind {
    Timer,
    Nvbandwidth,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToleranceInput {
    low: f64,
    high: f64,
    why: String,
}

#[derive(Debug, Deserialize)]
struct TimerObservation {
    times_s: Vec<f64>,
    number_per_run: u64,
}

struct ParsedObservation {
    seconds: f64,
    samples: Option<usize>,
    number_per_run: Option<u64>,
    directions: Vec<Direction>,
}

#[derive(Debug, Deserialize)]
struct NvBandwidthDocument {
    nvbandwidth: NvBandwidthPayload,
}

#[derive(Debug, Deserialize)]
struct NvBandwidthPayload {
    testcases: Vec<NvBandwidthTestcase>,
}

#[derive(Debug, Deserialize)]
struct NvBandwidthTestcase {
    name: String,
    status: String,
    bandwidth_matrix: Vec<Vec<String>>,
}

/// Parses and estimates every committed validation case.
///
/// # Errors
/// Returns an error for malformed case/evidence JSON, missing profiles,
/// invalid observations, or a workload the estimator cannot model.
pub fn cases() -> Result<Vec<Case>, String> {
    GENERATED_VALIDATION_CASES.iter().map(parse_case).collect()
}

fn parse_case(embedded: &EmbeddedCase) -> Result<Case, String> {
    let input: CaseInput = serde_json::from_str(embedded.case_json)
        .map_err(|error| format!("invalid case JSON for {}: {error}", embedded.id))?;
    if input.id != embedded.id {
        return Err(format!(
            "validation case directory {:?} does not match case.json id {:?}",
            embedded.id, input.id
        ));
    }
    let (workload, mode) = build_workload(&input.workload);
    let profile = profile::find(&input.gpu).ok_or_else(|| {
        format!(
            "validation case {} has unknown GPU {:?}",
            embedded.id, input.gpu
        )
    })?;
    let estimate = estimate::estimate(profile, &workload, mode, false)
        .map_err(|error| format!("cannot estimate validation case {}: {error}", embedded.id))?;
    let run: RunInfo = serde_json::from_str(embedded.run_json)
        .map_err(|error| format!("invalid run JSON for {}: {error}", embedded.id))?;
    let observation = parse_observation(embedded, &input.observation, &workload)?;
    if !(input.tolerance.low.is_finite()
        && input.tolerance.high.is_finite()
        && input.tolerance.low > 0.0
        && input.tolerance.low <= input.tolerance.high)
    {
        return Err(format!(
            "validation case {} has invalid tolerance band",
            embedded.id
        ));
    }
    if !(observation.seconds.is_finite()
        && observation.seconds > 0.0
        && estimate.total_s.is_finite()
        && estimate.total_s > 0.0)
    {
        return Err(format!(
            "validation case {} needs positive finite predicted and observed times",
            embedded.id
        ));
    }
    let ratio = observation.seconds / estimate.total_s;
    let verdict = if ratio < input.tolerance.low {
        "outside (faster)"
    } else if ratio > input.tolerance.high {
        "outside (slower)"
    } else {
        "within"
    };
    Ok(Case {
        id: embedded.id.to_owned(),
        title: input.title,
        gpu: input.gpu,
        modal_request: input.modal_request,
        job: input.job,
        derivation: input.derivation,
        match_items: input.match_items,
        excludes: input.excludes,
        finding: embedded.finding.to_owned(),
        observation_file: input.observation.file,
        source_files: embedded
            .source_files
            .iter()
            .map(|file| (*file).to_owned())
            .collect(),
        observation_tool: input.observation.tool,
        observation_metric: input.observation.metric,
        workload,
        mode,
        estimate,
        observed_s: observation.seconds,
        ratio,
        tolerance_low: input.tolerance.low,
        tolerance_high: input.tolerance.high,
        tolerance_why: input.tolerance.why,
        verdict: verdict.to_owned(),
        samples: observation.samples,
        number_per_run: observation.number_per_run,
        directions: observation.directions,
        run,
    })
}

fn build_workload(input: &WorkloadInput) -> (Workload, Mode) {
    let workload = Workload {
        dtype: input.dtype,
        flops: input.flops,
        hbm_bytes: input.hbm_bytes,
        host_bytes: 0.0,
        h2d: transfer(input.h2d.as_ref()),
        d2h: transfer(input.d2h.as_ref()),
        g2g: Transfer {
            bytes: 0.0,
            copies: 0,
        },
        copy_mode: match input.copy_mode {
            CopyModeInput::Sync => CopyMode::Sync,
            CopyModeInput::Async => CopyMode::Async,
        },
        launches: input.launches,
    };
    let mode = match input.mode {
        WorkloadModeInput::Serial => Mode::Serial,
        WorkloadModeInput::Overlap => Mode::Overlapped,
    };
    (workload, mode)
}

fn parse_observation(
    embedded: &EmbeddedCase,
    input: &ObservationInput,
    workload: &Workload,
) -> Result<ParsedObservation, String> {
    match input.kind {
        ObservationKind::Timer => {
            let timer: TimerObservation =
                serde_json::from_str(embedded.observation_json).map_err(|error| {
                    format!("invalid timer observation for {}: {error}", embedded.id)
                })?;
            let observed_s = median(&timer.times_s).map_err(|error| {
                format!("invalid timer observation for {}: {error}", embedded.id)
            })?;
            Ok(ParsedObservation {
                seconds: observed_s,
                samples: Some(timer.times_s.len()),
                number_per_run: Some(timer.number_per_run),
                directions: Vec::new(),
            })
        }
        ObservationKind::Nvbandwidth => {
            let parsed: NvBandwidthDocument = serde_json::from_str(embedded.observation_json)
                .map_err(|error| {
                    format!(
                        "invalid NVBandwidth observation for {}: {error}",
                        embedded.id
                    )
                })?;
            let directions =
                parse_nvbandwidth(&parsed, &input.testcases, workload).map_err(|error| {
                    format!(
                        "invalid NVBandwidth observation for {}: {error}",
                        embedded.id
                    )
                })?;
            let seconds = directions.iter().map(|direction| direction.seconds).sum();
            Ok(ParsedObservation {
                seconds,
                samples: None,
                number_per_run: None,
                directions,
            })
        }
    }
}

fn transfer(input: Option<&TransferInput>) -> Transfer {
    input.map_or(
        Transfer {
            bytes: 0.0,
            copies: 0,
        },
        |transfer| Transfer {
            bytes: transfer.bytes,
            copies: transfer.copies,
        },
    )
}

fn median(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("no timer samples".to_owned());
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("timer samples must be finite and non-negative".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Ok(sorted[middle])
    } else {
        let lower = sorted[middle - 1];
        let upper = sorted[middle];
        Ok(lower + (upper - lower) / 2.0)
    }
}

fn parse_nvbandwidth(
    document: &NvBandwidthDocument,
    names: &[String],
    workload: &Workload,
) -> Result<Vec<Direction>, String> {
    let mut directions = Vec::with_capacity(names.len());
    for name in names {
        let testcase = document
            .nvbandwidth
            .testcases
            .iter()
            .find(|testcase| testcase.name == *name)
            .ok_or_else(|| format!("testcase {name:?} is missing"))?;
        if testcase.status != "Passed" {
            return Err(format!(
                "testcase {name:?} has status {:?}, expected Passed",
                testcase.status
            ));
        }
        let bandwidth_text = testcase
            .bandwidth_matrix
            .first()
            .and_then(|row| row.first())
            .ok_or_else(|| format!("testcase {name:?} has no bandwidth_matrix[0][0]"))?
            .clone();
        let bandwidth_gbps = bandwidth_text
            .parse::<f64>()
            .map_err(|error| format!("testcase {name:?} has invalid bandwidth: {error}"))?;
        if !(bandwidth_gbps.is_finite() && bandwidth_gbps > 0.0) {
            return Err(format!(
                "testcase {name:?} bandwidth must be positive and finite"
            ));
        }
        let (direction_name, bytes) = if name.starts_with("host_to_device_") {
            ("host→device", workload.h2d.bytes)
        } else if name.starts_with("device_to_host_") {
            ("device→host", workload.d2h.bytes)
        } else {
            return Err(format!("testcase {name:?} has an unsupported direction"));
        };
        if bytes <= 0.0 {
            return Err(format!(
                "testcase {name:?} has no matching workload transfer bytes"
            ));
        }
        directions.push(Direction {
            name: direction_name.to_owned(),
            bandwidth_gbps,
            bandwidth_text,
            seconds: bytes / (bandwidth_gbps * 1e9),
        });
    }
    if directions.is_empty() {
        return Err("no NVBandwidth testcases were declared".to_owned());
    }
    Ok(directions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_odd_and_even_sample_counts() -> Result<(), String> {
        assert_eq!(median(&[9.0, 1.0, 5.0])?, 5.0);
        assert_eq!(median(&[9.0, 1.0, 5.0, 3.0])?, 4.0);
        Ok(())
    }

    #[test]
    fn nvbandwidth_converts_literal_directional_rates_to_seconds() -> Result<(), String> {
        let document: NvBandwidthDocument = serde_json::from_str(
            r#"{"nvbandwidth":{"testcases":[
                {"name":"host_to_device_memcpy_ce","status":"Passed","bandwidth_matrix":[["14.4814"]]},
                {"name":"device_to_host_memcpy_ce","status":"Passed","bandwidth_matrix":[["6.76848"]]}
            ]}}"#,
        )
        .map_err(|error| error.to_string())?;
        let workload = Workload {
            dtype: Dtype::Fp32,
            flops: 0.0,
            hbm_bytes: 0.0,
            host_bytes: 0.0,
            h2d: Transfer {
                bytes: 16.0 * 1024.0 * 1024.0 * 1024.0,
                copies: 16,
            },
            d2h: Transfer {
                bytes: 16.0 * 1024.0 * 1024.0 * 1024.0,
                copies: 16,
            },
            g2g: Transfer {
                bytes: 0.0,
                copies: 0,
            },
            copy_mode: CopyMode::Async,
            launches: 0,
        };
        let directions = parse_nvbandwidth(
            &document,
            &[
                "host_to_device_memcpy_ce".to_owned(),
                "device_to_host_memcpy_ce".to_owned(),
            ],
            &workload,
        )?;
        assert_eq!(directions[0].bandwidth_text, "14.4814");
        assert_eq!(directions[1].bandwidth_text, "6.76848");
        assert!((directions[0].seconds - 1.186_4).abs() < 0.001);
        assert!((directions[1].seconds - 2.538_4).abs() < 0.001);
        Ok(())
    }
}
