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

/// Summarizes the polls that fall inside each window. `polls` is everything
/// one `Monitor::watch` collected, warm-up included, so the energy counter's
/// update period can be measured over the whole watch.
pub(crate) fn round_states(polls: &[Poll], windows: &[Window]) -> Vec<RoundState> {
    let period = counter_period(polls);
    windows
        .iter()
        .map(|w| {
            let inside: Vec<&Poll> = polls
                .iter()
                .filter(|p| p.at >= w.start && p.at <= w.end)
                .collect();
            let power_w = period.and_then(|period| average_power(&inside, period));
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
                wall_ms: (w.end - w.start).as_secs_f64() * 1e3,
                sm_clock_mhz: span(inside.iter().map(|p| f64::from(p.sm_mhz))),
                mem_clock_mhz: span(inside.iter().map(|p| f64::from(p.mem_mhz))),
                power_w,
                temperature_c: span(inside.iter().map(|p| f64::from(p.temperature_c))),
                clock_limit_reasons,
            }
        })
        .collect()
}

/// Polls that saw the energy counter change since the poll before them.
fn ticks<'a>(polls: impl IntoIterator<Item = &'a Poll>) -> Vec<&'a Poll> {
    let polls: Vec<&Poll> = polls.into_iter().collect();
    polls
        .windows(2)
        .filter(|pair| pair[1].energy_mj != pair[0].energy_mj)
        .map(|pair| pair[1])
        .collect()
}

/// Mean time between energy-counter updates over one whole watch. Each
/// update is seen up to one poll gap late, but that error falls only on the
/// first and last update, so over a watch of a few seconds it is a percent or
/// two. None when the counter changed fewer than two times.
fn counter_period(polls: &[Poll]) -> Option<Duration> {
    let ticks = ticks(polls);
    let (first, last) = (ticks.first()?, ticks.last()?);
    let intervals = u32::try_from(ticks.len() - 1).ok()?;
    (intervals > 0).then(|| (last.at - first.at) / intervals)
}

/// Average power over a round from NVML's energy counter. On an H100 the
/// counter advances about every 100 ms, while a poll lands every 9–25 ms
/// because reading the counter alone takes 4–35 ms (#31). The energy between
/// the first and last update seen inside a ~200 ms round is exact, a whole
/// number of counter periods, but the time between the two polls that saw
/// those updates is off by up to a poll gap at each end. Dividing by that time
/// swung single rounds ±35% and above the board's power limit, so this
/// divides by the whole periods between the two updates instead, at the
/// `period` measured over the whole watch. When the counter updates on every
/// poll, the period is about one poll gap and the rounding changes little.
/// None when the counter changed fewer than two times inside the round.
#[expect(
    clippy::cast_precision_loss,
    reason = "one round's energy delta in millijoules is far below 2^52"
)]
fn average_power(polls: &[&Poll], period: Duration) -> Option<f64> {
    let ticks = ticks(polls.iter().copied());
    let (first, last) = (ticks.first()?, ticks.last()?);
    let periods = ((last.at - first.at).as_secs_f64() / period.as_secs_f64()).round();
    (periods >= 1.0 && last.energy_mj > first.energy_mj).then(|| {
        (last.energy_mj - first.energy_mj) as f64 / 1e3 / (periods * period.as_secs_f64())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Gaps between polls, cycled, in the 9–25 ms range #31 measured on an
    /// H100 where reading the energy counter took 4–35 ms.
    const POLL_GAPS_MS: [u64; 8] = [9, 25, 13, 11, 24, 10, 14, 19];
    const WATCH_MS: u64 = 2_000;
    const POWER_W: f64 = 660.0;

    /// Polls over one watch of a steady `POWER_W` load, reading a counter
    /// that advances every `update_ms`, plus seven back-to-back 200 ms
    /// rounds after a 350 ms warm-up, as `sustained_rounds` times them. With
    /// 100 ms updates each round then holds the updates at its 50 ms and
    /// 150 ms marks, the shortest span that yields a power.
    fn watch(update_ms: u64) -> (Vec<Poll>, Vec<Window>) {
        let start = Instant::now();
        let energy_mj = |t_ms: u64| {
            let updates = t_ms / update_ms;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "whole millijoules of a positive power over at most two seconds"
            )]
            let mj_per_update = (POWER_W * update_ms as f64) as u64;
            updates * mj_per_update
        };
        let mut polls = Vec::new();
        let mut t_ms = 0;
        for gap in POLL_GAPS_MS.iter().cycle() {
            if t_ms > WATCH_MS {
                break;
            }
            polls.push(Poll {
                at: start + Duration::from_millis(t_ms),
                sm_mhz: 1_980,
                mem_mhz: 2_619,
                temperature_c: 60,
                energy_mj: energy_mj(t_ms),
                reasons: ThrottleReasons::empty(),
            });
            t_ms += gap;
        }
        let windows = (0..7)
            .map(|round| Window {
                start: start + Duration::from_millis(350 + 200 * round),
                end: start + Duration::from_millis(550 + 200 * round),
            })
            .collect();
        (polls, windows)
    }

    fn powers(update_ms: u64) -> Vec<f64> {
        let (polls, windows) = watch(update_ms);
        round_states(&polls, &windows)
            .iter()
            .filter_map(|round| round.power_w)
            .collect()
    }

    #[test]
    fn slow_counter_reads_the_steady_power_in_every_round() {
        let powers = powers(100);
        assert_eq!(powers.len(), 7, "{powers:?}");
        for power in &powers {
            assert!((power / POWER_W - 1.0).abs() < 0.02, "{powers:?}");
        }
    }

    #[test]
    fn counter_that_updates_every_poll_still_reads_the_steady_power() {
        let powers = powers(1);
        assert_eq!(powers.len(), 7, "{powers:?}");
        for power in &powers {
            assert!((power / POWER_W - 1.0).abs() < 0.05, "{powers:?}");
        }
    }
}
