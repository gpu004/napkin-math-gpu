//! Clocks, power, temperature and clock-limit reasons of the GPU running the
//! timed GEMM rounds, read through NVML. nvml-wrapper loads
//! `libnvidia-ml.so.1` at run time, the same way cudarc loads the CUDA
//! libraries, so the harness still builds on machines without a driver.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use napkin_profile::{GpuState, RoundState, Span};
use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::{Device, Nvml};

use crate::stats::median;

const POLL_INTERVAL: Duration = Duration::from_millis(5);

/// One NVML reading.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Poll {
    at: Instant,
    sm_mhz: u32,
    mem_mhz: u32,
    temperature_c: u32,
    energy_mj: u64,
    reasons: ThrottleReasons,
}

impl Poll {
    fn read(device: &Device<'_>) -> Result<Self, String> {
        let nvml = |what: &str, e: nvml_wrapper::error::NvmlError| format!("NVML {what}: {e}");
        Ok(Self {
            sm_mhz: device
                .clock_info(Clock::SM)
                .map_err(|e| nvml("SM clock", e))?,
            mem_mhz: device
                .clock_info(Clock::Memory)
                .map_err(|e| nvml("memory clock", e))?,
            temperature_c: device
                .temperature(TemperatureSensor::Gpu)
                .map_err(|e| nvml("temperature", e))?,
            energy_mj: device
                .total_energy_consumption()
                .map_err(|e| nvml("energy counter", e))?,
            reasons: device
                .current_throttle_reasons()
                .map_err(|e| nvml("clock-limit reasons", e))?,
            at: Instant::now(),
        })
    }
}

/// Host-clock interval in which one timed round ran on the GPU.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    pub(crate) start: Instant,
    pub(crate) end: Instant,
}

/// NVML handle for the GPU the GEMM rounds run on.
pub(crate) struct Monitor {
    nvml: Nvml,
    pci_bus_id: String,
}

impl Monitor {
    /// Opens NVML for the GPU at `pci_bus_id`, e.g. `"00000000:18:00.0"`.
    /// CUDA and NVML can number GPUs differently, so the bus ID is what ties
    /// the NVML readings to the CUDA device that ran the GEMMs.
    pub(crate) fn open(pci_bus_id: String) -> Result<Self, String> {
        let nvml = Nvml::init().map_err(|e| format!("NVML init: {e}"))?;
        let monitor = Self { nvml, pci_bus_id };
        Poll::read(&monitor.device()?)?;
        Ok(monitor)
    }

    fn device(&self) -> Result<Device<'_>, String> {
        self.nvml
            .device_by_pci_bus_id(self.pci_bus_id.as_str())
            .map_err(|e| format!("NVML device {}: {e}", self.pci_bus_id))
    }

    /// Runs `work` on the calling thread, which owns the CUDA context, while a
    /// second thread polls NVML every `POLL_INTERVAL`.
    pub(crate) fn watch<T>(&self, work: impl FnOnce() -> T) -> Result<(T, Vec<Poll>), String> {
        let device = self.device()?;
        let stop = AtomicBool::new(false);
        thread::scope(|scope| {
            let poller = scope.spawn(|| {
                let mut polls = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    polls.push(Poll::read(&device)?);
                    thread::sleep(POLL_INTERVAL);
                }
                polls.push(Poll::read(&device)?);
                Ok::<_, String>(polls)
            });
            let out = work();
            stop.store(true, Ordering::Relaxed);
            let polls = poller
                .join()
                .map_err(|_| "NVML poller thread panicked".to_owned())??;
            Ok((out, polls))
        })
    }

    /// Board-level facts plus one `RoundState` per window, keyed like
    /// `Measured::samples_ms`.
    pub(crate) fn state(
        &self,
        warmup: Duration,
        round: Duration,
        rounds: BTreeMap<String, Vec<RoundState>>,
    ) -> Result<GpuState, String> {
        let device = self.device()?;
        let nvml = |what: &str, e: nvml_wrapper::error::NvmlError| format!("NVML {what}: {e}");
        let version = self
            .nvml
            .sys_nvml_version()
            .map_err(|e| nvml("version", e))?;
        Ok(GpuState {
            source: format!("NVML {version} via nvml-wrapper 0.13"),
            poll_interval_ms: POLL_INTERVAL.as_secs_f64() * 1e3,
            max_sm_clock_mhz: device
                .max_clock_info(Clock::SM)
                .map_err(|e| nvml("max SM clock", e))?,
            max_mem_clock_mhz: device
                .max_clock_info(Clock::Memory)
                .map_err(|e| nvml("max memory clock", e))?,
            power_limit_w: f64::from(
                device
                    .enforced_power_limit()
                    .map_err(|e| nvml("power limit", e))?,
            ) / 1e3,
            warmup_ms: warmup.as_secs_f64() * 1e3,
            round_ms: round.as_secs_f64() * 1e3,
            rounds,
        })
    }
}

/// Summarizes the polls that fall inside each window.
pub(crate) fn round_states(polls: &[Poll], windows: &[Window]) -> Vec<RoundState> {
    windows
        .iter()
        .map(|w| {
            let inside: Vec<&Poll> = polls
                .iter()
                .filter(|p| p.at >= w.start && p.at <= w.end)
                .collect();
            let power_w = average_power(&inside);
            let mut clock_limit_reasons = BTreeMap::new();
            for p in &inside {
                for (name, _) in p.reasons.iter_names() {
                    *clock_limit_reasons
                        .entry(name.to_lowercase())
                        .or_insert(0_u32) += 1;
                }
            }
            RoundState {
                polls: u32::try_from(inside.len()).unwrap_or(u32::MAX),
                sm_clock_mhz: span(inside.iter().map(|p| f64::from(p.sm_mhz))),
                mem_clock_mhz: span(inside.iter().map(|p| f64::from(p.mem_mhz))),
                power_w,
                temperature_c: span(inside.iter().map(|p| f64::from(p.temperature_c))),
                clock_limit_reasons,
            }
        })
        .collect()
}

/// Average power over a round from NVML's energy counter. The counter only
/// ticks every few tens of milliseconds on an H100, so two polls 5 ms apart
/// often read the same value and a naive delta would claim 0 W, or a huge
/// wattage when one tick lands between them. Measuring from the first poll
/// that saw a new value to the last one spans whole ticks only. None when the
/// counter ticked fewer than two times inside the round.
#[expect(
    clippy::cast_precision_loss,
    reason = "one round's energy delta in millijoules is far below 2^52"
)]
fn average_power(polls: &[&Poll]) -> Option<f64> {
    let ticks: Vec<&Poll> = polls
        .windows(2)
        .filter(|pair| pair[1].energy_mj != pair[0].energy_mj)
        .map(|pair| pair[1])
        .collect();
    let (first, last) = (ticks.first()?, ticks.last()?);
    (last.at > first.at && last.energy_mj > first.energy_mj).then(|| {
        (last.energy_mj - first.energy_mj) as f64 / 1e3 / (last.at - first.at).as_secs_f64()
    })
}

fn span(values: impl Iterator<Item = f64>) -> Option<Span> {
    let mut values: Vec<f64> = values.collect();
    let min = values.iter().copied().reduce(f64::min)?;
    let max = values.iter().copied().reduce(f64::max)?;
    Some(Span {
        min,
        median: median(&mut values),
        max,
    })
}
