//! Estimates billed GPU cost from useful busy time and overhead.

use std::fmt;

/// Work represented by a completed count or a rate during busy time.
#[derive(Clone, Copy, Debug)]
pub enum Work {
    Count(f64),
    Rate(f64),
}

/// Timing assumptions for a GPU job.
#[derive(Clone, Copy, Debug)]
pub struct Job {
    pub gpus: u32,
    pub busy_s: f64,
    pub utilization: f64,
    pub startup_s: f64,
    pub idle_s: f64,
    pub granularity_s: Option<f64>,
    pub work: Option<Work>,
}

/// The resulting running, billed, and useful-time cost estimates.
#[derive(Clone, Copy, Debug)]
pub struct Cost {
    pub running_s: f64,
    pub wall_s: f64,
    pub billed_s: f64,
    pub gpu_hours: f64,
    pub cost: f64,
    pub useful: f64,
    pub work: Option<f64>,
    pub work_per_dollar: Option<f64>,
}

/// An invalid job or price assumption.
#[derive(Debug)]
pub struct CostError(pub String);

impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CostError {}

fn positive_finite(name: &str, value: f64) -> Result<(), CostError> {
    if !(value > 0.0 && value.is_finite()) {
        return Err(CostError(format!(
            "{name} must be a positive finite quantity"
        )));
    }
    Ok(())
}

fn nonnegative_finite(name: &str, value: f64) -> Result<(), CostError> {
    if !(value >= 0.0 && value.is_finite()) {
        return Err(CostError(format!(
            "{name} must be a non-negative finite quantity"
        )));
    }
    Ok(())
}

/// Estimates cost for the same elapsed wall time on each rented GPU.
///
/// `busy_s` is the time the job keeps each GPU working at the modelled rate;
/// utilization is the fraction of on-GPU time it sustains that rate.
///
/// # Errors
/// Returns `CostError` for invalid inputs or non-finite derived costs.
pub fn cost(job: &Job, price_per_gpu_hour: f64) -> Result<Cost, CostError> {
    if job.gpus == 0 {
        return Err(CostError("gpus must be at least 1".to_owned()));
    }
    positive_finite("busy time", job.busy_s)?;
    if !(job.utilization > 0.0 && job.utilization <= 1.0 && job.utilization.is_finite()) {
        return Err(CostError(
            "utilization must be finite and in (0, 1]".to_owned(),
        ));
    }
    nonnegative_finite("startup time", job.startup_s)?;
    nonnegative_finite("idle time", job.idle_s)?;
    positive_finite("price per GPU-hour", price_per_gpu_hour)?;
    if let Some(granularity) = job.granularity_s {
        positive_finite("billing granularity", granularity)?;
    }
    if let Some(work) = job.work {
        let value = match work {
            Work::Count(value) | Work::Rate(value) => value,
        };
        nonnegative_finite("work", value)?;
    }

    let running_s = job.busy_s / job.utilization;
    let wall_s = job.startup_s + running_s + job.idle_s;
    let billed_s = job.granularity_s.map_or(wall_s, |granularity| {
        let units = wall_s / granularity;
        let units = if (units - units.round()).abs() < 1e-9 {
            units.round()
        } else {
            units.ceil()
        };
        units * granularity
    });
    let gpu_hours = f64::from(job.gpus) * billed_s / 3600.0;
    let cost = gpu_hours * price_per_gpu_hour;
    let useful = job.busy_s / billed_s;
    let work = job.work.map(|work| match work {
        Work::Count(value) => value,
        Work::Rate(value) => value * job.busy_s,
    });
    if ![running_s, wall_s, billed_s, gpu_hours, cost, useful]
        .into_iter()
        .all(f64::is_finite)
        || cost <= 0.0
    {
        return Err(CostError(
            "job assumptions produce a non-finite or zero cost".to_owned(),
        ));
    }
    let work_per_dollar = work.map(|value| value / cost);
    if work_per_dollar.is_some_and(|value| !value.is_finite()) {
        return Err(CostError(
            "job assumptions produce a non-finite work-per-dollar estimate".to_owned(),
        ));
    }
    Ok(Cost {
        running_s,
        wall_s,
        billed_s,
        gpu_hours,
        cost,
        useful,
        work,
        work_per_dollar,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::fmt_money;

    fn job() -> Job {
        Job {
            gpus: 2,
            busy_s: 10.0,
            utilization: 1.0,
            startup_s: 0.0,
            idle_s: 0.0,
            granularity_s: None,
            work: None,
        }
    }

    #[test]
    fn ten_seconds_on_two_gpus_at_three_dollars() {
        let result = cost(&job(), 3.0).unwrap();
        assert!((result.cost - 2.0 * 10.0 / 3600.0 * 3.0).abs() < 1e-9);
        assert_eq!(fmt_money(result.cost, "USD"), "$0.0167");
    }

    #[test]
    fn gpu_count_scales_cost_linearly() {
        let two = cost(&job(), 3.0).unwrap().cost;
        let mut four_gpu = job();
        four_gpu.gpus = 4;
        assert!((cost(&four_gpu, 3.0).unwrap().cost - 2.0 * two).abs() < 1e-12);
    }

    #[test]
    fn utilization_increases_running_cost_and_reduces_useful_fraction() {
        let full = cost(&job(), 3.0).unwrap();
        let mut half_job = job();
        half_job.utilization = 0.5;
        let half = cost(&half_job, 3.0).unwrap();
        assert_eq!(half.running_s, 2.0 * full.running_s);
        assert_eq!(half.cost, 2.0 * full.cost);
        assert_eq!(half.useful, 0.5);
    }

    #[test]
    fn billing_granularity_rounds_the_whole_wall_time() {
        let mut input = job();
        input.startup_s = 30.0;
        input.idle_s = 60.0;
        input.granularity_s = Some(60.0);
        let result = cost(&input, 3.0).unwrap();
        assert_eq!(result.wall_s, 100.0);
        assert_eq!(result.billed_s, 120.0);
        assert_eq!(result.cost, 0.2);
        assert_eq!(result.useful, 10.0 / 120.0);
    }

    #[test]
    fn exact_billing_multiple_is_not_rounded_up() {
        let mut input = job();
        input.busy_s = 59.9;
        input.startup_s = 0.1;
        input.granularity_s = Some(60.0);
        assert_eq!(cost(&input, 3.0).unwrap().billed_s, 60.0);
    }

    #[test]
    fn computes_work_and_work_per_dollar() {
        let mut input = job();
        input.work = Some(Work::Rate(2500.0));
        let rate = cost(&input, 3.0).unwrap();
        assert_eq!(rate.work, Some(25_000.0));
        assert_eq!(
            rate.work_per_dollar,
            Some(25_000.0 / (2.0 * 10.0 / 3600.0 * 3.0))
        );
        input.work = Some(Work::Count(1e6));
        assert_eq!(cost(&input, 3.0).unwrap().work, Some(1e6));
        input.work = None;
        assert_eq!(cost(&input, 3.0).unwrap().work, None);
    }

    #[test]
    fn rejects_invalid_assumptions() {
        let mut input = job();
        input.gpus = 0;
        assert!(cost(&input, 3.0).is_err());
        input = job();
        input.busy_s = 0.0;
        assert!(cost(&input, 3.0).is_err());
        input.busy_s = f64::NAN;
        assert!(cost(&input, 3.0).is_err());
        input = job();
        input.utilization = 0.0;
        assert!(cost(&input, 3.0).is_err());
        input.utilization = 1.1;
        assert!(cost(&input, 3.0).is_err());
        input.utilization = f64::NAN;
        assert!(cost(&input, 3.0).is_err());
        input = job();
        input.startup_s = -1.0;
        assert!(cost(&input, 3.0).is_err());
        input.startup_s = f64::INFINITY;
        assert!(cost(&input, 3.0).is_err());
        input = job();
        input.idle_s = -1.0;
        assert!(cost(&input, 3.0).is_err());
        input.idle_s = f64::NAN;
        assert!(cost(&input, 3.0).is_err());
        input = job();
        input.granularity_s = Some(0.0);
        assert!(cost(&input, 3.0).is_err());
        input.granularity_s = Some(f64::INFINITY);
        assert!(cost(&input, 3.0).is_err());
        input = job();
        assert!(cost(&input, 0.0).is_err());
        assert!(cost(&input, f64::NAN).is_err());
        input.work = Some(Work::Count(-1.0));
        assert!(cost(&input, 3.0).is_err());
        input.work = Some(Work::Rate(f64::INFINITY));
        assert!(cost(&input, 3.0).is_err());
    }
}
