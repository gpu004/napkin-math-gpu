use std::fmt;

use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyPath {
    Ce,
    Sm,
}

impl fmt::Display for CopyPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ce => "CE",
            Self::Sm => "SM",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    Host,
    Gpu(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Statistic {
    Median,
    Mean,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bandwidth {
    pub testcase: String,
    pub path: CopyPath,
    pub access: Option<Access>,
    pub src: Endpoint,
    pub dst: Endpoint,
    pub initiator: u32,
    pub gbps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Latency {
    pub testcase: String,
    pub a: Endpoint,
    pub b: Endpoint,
    pub ns: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcieLink {
    pub r#gen: u8,
    pub gen_max: u8,
    pub width: u8,
    pub width_max: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Nvbandwidth {
    pub version: String,
    pub git_version: String,
    pub repository: String,
    pub commit: String,
    pub command: Vec<String>,
    pub captured_at: String,
    pub gpu_request: String,
    pub driver_version: String,
    pub cuda_runtime_version: String,
    pub cuda_driver_version: String,
    pub devices: Vec<String>,
    pub pcie_links: Vec<PcieLink>,
    pub topology: String,
    pub buffer_bytes: u64,
    pub samples: u32,
    pub statistic: Statistic,
    /// Parsing succeeds only when verification was enabled and every testcase passed.
    pub verified: bool,
    pub bandwidths: Vec<Bandwidth>,
    pub latencies: Vec<Latency>,
}

impl Nvbandwidth {
    /// Parses an `NVBandwidth` JSON result together with its launcher metadata.
    ///
    /// # Errors
    /// Returns an error when the run failed, skipped verification, has invalid
    /// matrices or metadata, or does not match the launcher record.
    pub fn parse(raw: &str, run: &str, topo: &str) -> Result<Self, String> {
        let raw: RawDocument = serde_json::from_str(raw)
            .map_err(|error| format!("invalid NVBandwidth JSON: {error}"))?;
        let run: RunRecord = serde_json::from_str(run)
            .map_err(|error| format!("invalid NVBandwidth run JSON: {error}"))?;
        let output = raw.nvbandwidth;
        validate_run(&output, &run)?;
        let (buffer_bytes, samples, statistic) = run_settings(&run.command)?;
        let devices = parse_devices(&output, &run)?;
        let pcie_links = parse_pcie_links(&run);
        let (bandwidths, latencies) = parse_testcases(&output, devices.len())?;

        Ok(Self {
            version: output.version,
            git_version: output.git_version,
            repository: run.repository,
            commit: run.commit,
            command: run.command,
            captured_at: run.captured_at,
            gpu_request: run.gpu_request,
            driver_version: output.driver_version,
            cuda_runtime_version: cuda_version(output.cuda_runtime_version),
            cuda_driver_version: cuda_version(output.cuda_driver_version),
            devices,
            pcie_links,
            topology: topo.to_owned(),
            buffer_bytes,
            samples,
            statistic,
            verified: true,
            bandwidths,
            latencies,
        })
    }

    /// Checks the evidence's GPU identity against a measured profile.
    ///
    /// # Errors
    /// Returns an error when the GPU name, device count, or request differs.
    pub fn check_identity(&self, measured: &super::Measured) -> Result<(), String> {
        if self
            .devices
            .iter()
            .any(|device| device != &measured.device_name)
        {
            return Err(format!(
                "NVBandwidth devices {:?} do not match measured device_name {:?}",
                self.devices, measured.device_name
            ));
        }
        if usize::try_from(measured.device_count).ok() != Some(self.devices.len()) {
            return Err(format!(
                "NVBandwidth device count {} does not match measured device_count {}",
                self.devices.len(),
                measured.device_count
            ));
        }
        if self.gpu_request != measured.gpu_request {
            return Err(format!(
                "NVBandwidth gpu_request {:?} does not match measured gpu_request {:?}",
                self.gpu_request, measured.gpu_request
            ));
        }
        Ok(())
    }

    pub fn h2d(&self, path: CopyPath) -> Option<f64> {
        self.bandwidths
            .iter()
            .filter(|rate| rate.path == path && rate.src == Endpoint::Host)
            .map(|rate| rate.gbps)
            .reduce(f64::min)
    }

    pub fn d2h(&self, path: CopyPath) -> Option<f64> {
        self.bandwidths
            .iter()
            .filter(|rate| rate.path == path && rate.dst == Endpoint::Host)
            .map(|rate| rate.gbps)
            .reduce(f64::min)
    }

    pub fn peer(&self, path: CopyPath, src: u32, dst: u32) -> Option<f64> {
        self.bandwidths
            .iter()
            .filter(|rate| {
                rate.path == path
                    && rate.src == Endpoint::Gpu(src)
                    && rate.dst == Endpoint::Gpu(dst)
            })
            .map(|rate| rate.gbps)
            .reduce(f64::min)
    }

    pub fn g2g(&self, path: CopyPath) -> Option<f64> {
        self.bandwidths
            .iter()
            .filter(|rate| {
                rate.path == path
                    && matches!(rate.src, Endpoint::Gpu(_))
                    && matches!(rate.dst, Endpoint::Gpu(_))
            })
            .map(|rate| rate.gbps)
            .reduce(f64::min)
    }
}

fn validate_run(output: &RawOutput, run: &RunRecord) -> Result<(), String> {
    if output.error.is_some() {
        return Err("NVBandwidth output contains error".to_owned());
    }
    if run.exit_code != 0 {
        return Err(format!(
            "NVBandwidth run exit_code is {}, expected 0",
            run.exit_code
        ));
    }
    if run
        .command
        .iter()
        .any(|arg| arg == "--skipVerification" || arg == "-s")
    {
        return Err("NVBandwidth command disables verification".to_owned());
    }
    Ok(())
}

fn run_settings(command: &[String]) -> Result<(u64, u32, Statistic), String> {
    let buffer_mib = command_value(command, "--bufferSize")?;
    let buffer_bytes = buffer_mib
        .parse::<u64>()
        .map_err(|_| format!("NVBandwidth --bufferSize is not an integer: {buffer_mib:?}"))?
        .checked_mul(1024 * 1024)
        .ok_or_else(|| "NVBandwidth --bufferSize overflows buffer_bytes".to_owned())?;
    let samples_value = command_value(command, "--testSamples")?;
    let samples = samples_value
        .parse::<u32>()
        .map_err(|_| format!("NVBandwidth --testSamples is not an integer: {samples_value:?}"))?;
    let statistic = if command.iter().any(|arg| arg == "--useMean") {
        Statistic::Mean
    } else {
        Statistic::Median
    };
    Ok((buffer_bytes, samples, statistic))
}

fn parse_devices(output: &RawOutput, run: &RunRecord) -> Result<Vec<String>, String> {
    let devices = output
        .devices
        .iter()
        .map(|device| device_name(device))
        .collect::<Result<Vec<_>, _>>()?;
    if devices.len() != run.gpus.len() {
        return Err(format!(
            "GPU Device list has {} devices but run gpus has {}",
            devices.len(),
            run.gpus.len()
        ));
    }
    for (index, (device, gpu)) in devices.iter().zip(&run.gpus).enumerate() {
        if device != &gpu.name {
            return Err(format!(
                "GPU Device list device {index} name {device:?} does not match run gpus name {:?}",
                gpu.name
            ));
        }
    }
    if output.driver_version
        != run
            .gpus
            .first()
            .map_or("", |gpu| gpu.driver_version.as_str())
        || run
            .gpus
            .iter()
            .any(|gpu| gpu.driver_version != output.driver_version)
    {
        return Err(format!(
            "Driver Version {:?} does not match run gpus driver_version",
            output.driver_version
        ));
    }
    Ok(devices)
}

fn parse_pcie_links(run: &RunRecord) -> Vec<PcieLink> {
    let mut pcie_links = run
        .pcie_links
        .iter()
        .map(|link| {
            (
                link.index,
                PcieLink {
                    r#gen: link.pcie_gen,
                    gen_max: link.pcie_gen_max,
                    width: link.pcie_width,
                    width_max: link.pcie_width_max,
                },
            )
        })
        .collect::<Vec<_>>();
    pcie_links.sort_by_key(|(index, _)| *index);
    pcie_links.into_iter().map(|(_, link)| link).collect()
}

fn parse_testcases(
    output: &RawOutput,
    gpu_count: usize,
) -> Result<(Vec<Bandwidth>, Vec<Latency>), String> {
    let mut seen = Vec::with_capacity(output.testcases.len());
    let mut bandwidths = Vec::new();
    let mut latencies = Vec::new();
    for testcase in &output.testcases {
        if seen.contains(&testcase.name.as_str()) {
            return Err(format!("testcase {} appears more than once", testcase.name));
        }
        seen.push(testcase.name.as_str());
        if testcase.status != "Passed" {
            return Err(format!(
                "testcase {} status is {:?}, expected Passed",
                testcase.name, testcase.status
            ));
        }
        if let Some((path, host_to_device)) = host_bandwidth_testcase(&testcase.name) {
            bandwidths.extend(parse_host_bandwidth(
                testcase,
                path,
                host_to_device,
                gpu_count,
            )?);
        } else if let Some((path, access)) = peer_bandwidth_testcase(&testcase.name) {
            bandwidths.extend(parse_peer_bandwidth(testcase, path, access, gpu_count)?);
        } else if testcase.name == "host_device_latency_sm" {
            latencies.extend(parse_host_latency(testcase, gpu_count)?);
        } else if testcase.name == "device_to_device_latency_sm" {
            latencies.extend(parse_peer_latency(testcase, gpu_count)?);
        } else {
            return Err(format!("unknown testcase {}", testcase.name));
        }
    }

    for required in ["host_to_device_memcpy_ce", "device_to_host_memcpy_ce"] {
        if !seen.contains(&required) {
            return Err(format!("missing required testcase {required}"));
        }
    }
    if gpu_count >= 2 {
        for required in [
            "device_to_device_memcpy_read_ce",
            "device_to_device_memcpy_write_ce",
        ] {
            if !seen.contains(&required) {
                return Err(format!("missing required testcase {required}"));
            }
        }
    }
    Ok((bandwidths, latencies))
}

fn parse_host_bandwidth(
    testcase: &RawTestcase,
    path: CopyPath,
    host_to_device: bool,
    gpu_count: usize,
) -> Result<Vec<Bandwidth>, String> {
    check_bandwidth_units(testcase)?;
    check_shape(&testcase.name, &testcase.bandwidth_matrix, 1, gpu_count)?;
    testcase.bandwidth_matrix[0]
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            let gbps = parse_positive(cell, &testcase.name, "bandwidth_matrix")?;
            let gpu = gpu_index(index, &testcase.name)?;
            let (src, dst) = if host_to_device {
                (Endpoint::Host, Endpoint::Gpu(gpu))
            } else {
                (Endpoint::Gpu(gpu), Endpoint::Host)
            };
            Ok(Bandwidth {
                testcase: testcase.name.clone(),
                path,
                access: None,
                src,
                dst,
                initiator: gpu,
                gbps,
            })
        })
        .collect()
}

fn parse_peer_bandwidth(
    testcase: &RawTestcase,
    path: CopyPath,
    access: Access,
    gpu_count: usize,
) -> Result<Vec<Bandwidth>, String> {
    check_bandwidth_units(testcase)?;
    check_shape(
        &testcase.name,
        &testcase.bandwidth_matrix,
        gpu_count,
        gpu_count,
    )?;
    let mut bandwidths = Vec::new();
    for (row, values) in testcase.bandwidth_matrix.iter().enumerate() {
        for (column, cell) in values.iter().enumerate() {
            if row == column {
                continue;
            }
            let gbps = parse_positive(cell, &testcase.name, "bandwidth_matrix")?;
            let row_gpu = gpu_index(row, &testcase.name)?;
            let column_gpu = gpu_index(column, &testcase.name)?;
            let (src, dst) = match access {
                Access::Read => (column_gpu, row_gpu),
                Access::Write => (row_gpu, column_gpu),
            };
            bandwidths.push(Bandwidth {
                testcase: testcase.name.clone(),
                path,
                access: Some(access),
                src: Endpoint::Gpu(src),
                dst: Endpoint::Gpu(dst),
                initiator: row_gpu,
                gbps,
            });
        }
    }
    Ok(bandwidths)
}

fn parse_host_latency(testcase: &RawTestcase, gpu_count: usize) -> Result<Vec<Latency>, String> {
    check_latency_units(testcase)?;
    check_shape(&testcase.name, &testcase.bandwidth_matrix, 1, gpu_count)?;
    testcase.bandwidth_matrix[0]
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            Ok(Latency {
                testcase: testcase.name.clone(),
                a: Endpoint::Host,
                b: Endpoint::Gpu(gpu_index(index, &testcase.name)?),
                ns: parse_positive(cell, &testcase.name, "bandwidth_matrix")?,
            })
        })
        .collect()
}

fn parse_peer_latency(testcase: &RawTestcase, gpu_count: usize) -> Result<Vec<Latency>, String> {
    check_latency_units(testcase)?;
    check_shape(
        &testcase.name,
        &testcase.bandwidth_matrix,
        gpu_count,
        gpu_count,
    )?;
    let mut latencies = Vec::new();
    for (row, values) in testcase.bandwidth_matrix.iter().enumerate() {
        for (column, cell) in values.iter().enumerate() {
            if row == column {
                continue;
            }
            latencies.push(Latency {
                testcase: testcase.name.clone(),
                a: Endpoint::Gpu(gpu_index(row, &testcase.name)?),
                b: Endpoint::Gpu(gpu_index(column, &testcase.name)?),
                ns: parse_positive(cell, &testcase.name, "bandwidth_matrix")?,
            });
        }
    }
    Ok(latencies)
}

fn check_bandwidth_units(testcase: &RawTestcase) -> Result<(), String> {
    if !testcase.bandwidth_description.ends_with("bandwidth (GB/s)") {
        return Err(format!(
            "testcase {} bandwidth_description must end with bandwidth (GB/s)",
            testcase.name
        ));
    }
    Ok(())
}

fn check_latency_units(testcase: &RawTestcase) -> Result<(), String> {
    if !testcase.bandwidth_description.ends_with("(ns)") {
        return Err(format!(
            "testcase {} bandwidth_description must end with (ns)",
            testcase.name
        ));
    }
    Ok(())
}

fn gpu_index(index: usize, testcase: &str) -> Result<u32, String> {
    u32::try_from(index).map_err(|_| format!("testcase {testcase} GPU index overflows u32"))
}

#[derive(Deserialize)]
struct RawDocument {
    nvbandwidth: RawOutput,
}

#[derive(Deserialize)]
struct RawOutput {
    version: String,
    git_version: String,
    #[serde(rename = "CUDA Runtime Version")]
    cuda_runtime_version: u32,
    #[serde(rename = "CUDA Driver API Version")]
    cuda_driver_version: u32,
    #[serde(rename = "Driver Version")]
    driver_version: String,
    #[serde(rename = "GPU Device list")]
    devices: Vec<String>,
    error: Option<serde_json::Value>,
    testcases: Vec<RawTestcase>,
}

#[derive(Deserialize)]
struct RawTestcase {
    name: String,
    status: String,
    bandwidth_description: String,
    bandwidth_matrix: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct RunRecord {
    repository: String,
    commit: String,
    command: Vec<String>,
    exit_code: i32,
    gpu_request: String,
    captured_at: String,
    gpus: Vec<RunGpu>,
    pcie_links: Vec<RunPcieLink>,
}

#[derive(Deserialize)]
struct RunGpu {
    name: String,
    driver_version: String,
}

#[derive(Deserialize)]
struct RunPcieLink {
    index: u32,
    pcie_gen: u8,
    pcie_gen_max: u8,
    pcie_width: u8,
    pcie_width_max: u8,
}

fn command_value<'a>(command: &'a [String], flag: &str) -> Result<&'a str, String> {
    let Some(index) = command.iter().position(|arg| arg == flag) else {
        return Err(format!("NVBandwidth command is missing {flag}"));
    };
    command
        .get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("NVBandwidth command is missing value for {flag}"))
}

fn device_name(device: &str) -> Result<String, String> {
    let (_, description) = device
        .split_once(": ")
        .ok_or_else(|| format!("GPU Device list entry {device:?} has no index separator"))?;
    let name = description
        .split_once(" (")
        .map_or(description, |(name, _)| name)
        .trim();
    if name.is_empty() {
        return Err(format!(
            "GPU Device list entry {device:?} has an empty name"
        ));
    }
    Ok(name.to_owned())
}

fn cuda_version(version: u32) -> String {
    format!("{}.{}", version / 1000, version % 1000 / 10)
}

fn host_bandwidth_testcase(name: &str) -> Option<(CopyPath, bool)> {
    match name {
        "host_to_device_memcpy_ce" => Some((CopyPath::Ce, true)),
        "host_to_device_memcpy_sm" => Some((CopyPath::Sm, true)),
        "device_to_host_memcpy_ce" => Some((CopyPath::Ce, false)),
        "device_to_host_memcpy_sm" => Some((CopyPath::Sm, false)),
        _ => None,
    }
}

fn peer_bandwidth_testcase(name: &str) -> Option<(CopyPath, Access)> {
    match name {
        "device_to_device_memcpy_read_ce" => Some((CopyPath::Ce, Access::Read)),
        "device_to_device_memcpy_write_ce" => Some((CopyPath::Ce, Access::Write)),
        "device_to_device_memcpy_read_sm" => Some((CopyPath::Sm, Access::Read)),
        "device_to_device_memcpy_write_sm" => Some((CopyPath::Sm, Access::Write)),
        _ => None,
    }
}

fn check_shape(
    testcase: &str,
    matrix: &[Vec<String>],
    rows: usize,
    columns: usize,
) -> Result<(), String> {
    if matrix.len() != rows || matrix.iter().any(|row| row.len() != columns) {
        return Err(format!(
            "testcase {testcase} bandwidth_matrix shape is not {rows}x{columns}"
        ));
    }
    Ok(())
}

fn parse_positive(value: &str, testcase: &str, field: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("testcase {testcase} {field} contains nonnumeric value {value:?}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!(
            "testcase {testcase} {field} contains non-finite or nonpositive value {value:?}"
        ));
    }
    Ok(parsed)
}
