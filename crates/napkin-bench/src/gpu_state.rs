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
    let counter = Counter::of(polls);
    windows
        .iter()
        .map(|w| {
            let inside: Vec<&Poll> = polls
                .iter()
                .filter(|p| p.at >= w.start && p.at <= w.end)
                .collect();
            let power_w = counter.and_then(|counter| average_power(&inside, counter));
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

/// How NVML's energy counter advanced during one watch, relative to the
/// polls that read it.
#[derive(Clone, Copy, Debug)]
enum Counter {
    /// The counter advanced less often than every second poll gap, so no
    /// gap held two updates and each poll that saw a change saw exactly one.
    /// On an H100 it advances about every 100 ms while polls land 9–25 ms
    /// apart (#31). The period is the mean over the whole watch: each
    /// update is seen up to one poll gap late, but that error falls only on
    /// the first and last update of a watch a few seconds long.
    PerPeriod(Duration),
    /// The counter can advance more than once between two polls, so changes
    /// seen do not count updates and the time between polls is used instead.
    Elapsed,
}

impl Counter {
    /// None when the counter changed fewer than two times in the watch.
    fn of(polls: &[Poll]) -> Option<Self> {
        let ticks = ticks(polls);
        let (first, last) = (ticks.first()?, ticks.last()?);
        let intervals = u32::try_from(ticks.len() - 1).ok()?;
        let period = (intervals > 0).then(|| (last.at - first.at) / intervals)?;
        let longest_gap = polls.windows(2).map(|pair| pair[1].at - pair[0].at).max()?;
        // Two updates can only merge into one change inside a gap longer
        // than the period, and each merge stretches the mean spacing of the
        // changes seen. Stretching it past twice every gap would take more
        // than half the updates merged, which needs gaps longer than the
        // period, so passing this check rules merges out. A 101 ms gap on a
        // 100 ms counter fails it and falls back to elapsed time.
        Some(if period > 2 * longest_gap {
            Self::PerPeriod(period)
        } else {
            Self::Elapsed
        })
    }
}

/// Average power over a round from NVML's energy counter: the energy between
/// the first and last change seen inside the round, over the time it covers.
/// With a slow counter that time is the updates between those two changes
/// times the counter period. The time between the two polls that saw them is
/// off by up to a poll gap at each end, because reading the counter alone
/// takes 4–35 ms on an H100, and dividing by it swung single ~200 ms rounds
/// ±35% and above the board's power limit (#31). None when the counter changed
/// fewer than two times inside the round.
#[expect(
    clippy::cast_precision_loss,
    reason = "one round's energy delta in millijoules and its update count are far below 2^52"
)]
fn average_power(polls: &[&Poll], counter: Counter) -> Option<f64> {
    let ticks = ticks(polls.iter().copied());
    let (first, last) = (ticks.first()?, ticks.last()?);
    let seconds = match counter {
        Counter::PerPeriod(period) => (ticks.len() - 1) as f64 * period.as_secs_f64(),
        Counter::Elapsed => (last.at - first.at).as_secs_f64(),
    };
    (seconds > 0.0 && last.energy_mj > first.energy_mj)
        .then(|| (last.energy_mj - first.energy_mj) as f64 / 1e3 / seconds)
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
    const POWER_W: u32 = 660;

    /// Polls over one watch of a steady `POWER_W` load, reading a counter
    /// that advances every `update_ms`, plus seven back-to-back 200 ms
    /// rounds after a 350 ms warm-up, as `sustained_rounds` times them. With
    /// 100 ms updates each round then holds the updates at its 50 ms and
    /// 150 ms marks, the shortest span that yields a power.
    fn watch(update_ms: u64) -> (Vec<Poll>, Vec<Window>) {
        let start = Instant::now();
        let energy_mj = |t_ms: u64| t_ms / update_ms * update_ms * u64::from(POWER_W);
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
            assert!(
                (power / f64::from(POWER_W) - 1.0).abs() < 0.02,
                "{powers:?}"
            );
        }
    }

    #[test]
    fn counter_that_updates_every_poll_still_reads_the_steady_power() {
        let powers = powers(1);
        assert_eq!(powers.len(), 7, "{powers:?}");
        for power in &powers {
            assert!(
                (power / f64::from(POWER_W) - 1.0).abs() < 0.01,
                "{powers:?}"
            );
        }
    }

    /// One 101 ms gap swallows the updates at 1000 and 1100 ms, so the
    /// changes seen undercount the updates and their mean spacing grows to
    /// about 106 ms. Counting changes would read about 1,250 W.
    #[test]
    fn gap_that_holds_two_updates_falls_back_to_elapsed_time() {
        let start = Instant::now();
        let at = |t_ms: u64| start + Duration::from_millis(t_ms);
        let mut times: Vec<u64> = (0..=200)
            .map(|i| 1 + 10 * i)
            .filter(|t_ms| !(1_001..=1_091).contains(t_ms))
            .chain([999, 1_100])
            .collect();
        times.sort_unstable();
        let polls: Vec<Poll> = times
            .iter()
            .map(|&t_ms| Poll {
                at: at(t_ms),
                sm_mhz: 1_980,
                mem_mhz: 2_619,
                temperature_c: 60,
                energy_mj: t_ms / 100 * 100 * u64::from(POWER_W),
                reasons: ThrottleReasons::empty(),
            })
            .collect();
        let windows = [Window {
            start: at(890),
            end: at(1_100),
        }];
        let power = round_states(&polls, &windows)[0].power_w;
        assert!(
            power.is_some_and(|power| (power / f64::from(POWER_W) - 1.0).abs() < 0.02),
            "{power:?}"
        );
    }

    /// A round whose first update is seen 52 ms late and whose last is seen
    /// 1 ms late: the polls that saw its three updates are 149 ms apart, not
    /// 200 ms. #31 put the p95 delay in seeing an update at 51 ms. The delay
    /// comes from slow reads, as on an H100: the poll stamped 1019 ms read the
    /// counter at 999 ms, just before the update at 1000 ms, and the poll
    /// stamped 1052 ms read it at 1037 ms, so no gap is longer than 33 ms.
    #[test]
    fn late_poll_at_a_round_edge_does_not_change_its_power() {
        let start = Instant::now();
        let at = |t_ms: u64| start + Duration::from_millis(t_ms);
        let mut reads: Vec<(u64, u64)> = (0..=200)
            .map(|i| (1 + 10 * i, 1 + 10 * i))
            .filter(|(t_ms, _)| !(1_001..=1_051).contains(t_ms))
            .chain([(1_019, 999), (1_052, 1_037)])
            .collect();
        reads.sort_unstable();
        let polls: Vec<Poll> = reads
            .iter()
            .map(|&(t_ms, read_ms)| Poll {
                at: at(t_ms),
                sm_mhz: 1_980,
                mem_mhz: 2_619,
                temperature_c: 60,
                energy_mj: read_ms / 100 * 100 * u64::from(POWER_W),
                reasons: ThrottleReasons::empty(),
            })
            .collect();
        let windows = [Window {
            start: at(995),
            end: at(1_205),
        }];
        let power = round_states(&polls, &windows)[0].power_w;
        assert!(
            power.is_some_and(|power| (power / f64::from(POWER_W) - 1.0).abs() < 0.01),
            "{power:?}"
        );
    }
}
