//! Estimates billed GPU cost from useful busy time and overhead.

use std::fmt;

/// Work represented by a completed count or a rate during busy time.
#[derive(Clone, Copy, Debug)]
pub enum Work {
    Count(f64),
    Rate(f64),
}

/// Work completed and its cost efficiency.
#[derive(Clone, Copy, Debug)]
pub struct WorkTotal {
    pub source: Work,
    pub units: f64,
    pub per_dollar: f64,
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
    pub work: Option<WorkTotal>,
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
    let work = job.work.map(|source| {
        let units = match source {
            Work::Count(value) => value,
            Work::Rate(value) => value * job.busy_s,
        };
        WorkTotal {
            source,
            units,
            per_dollar: units / cost,
        }
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
    if work.is_some_and(|total| !total.units.is_finite() || !total.per_dollar.is_finite()) {
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
    })
}

#[cfg(test)]
mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;

    use super::*;
    use crate::units::fmt_money;

    fn near(actual: f64, expected: f64) {
        let difference = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs()).max(1e-12);
        assert!(
            difference <= 1e-9 * scale,
            "{actual} differs from {expected} by {difference}"
        );
    }

    // Times stay inside `[0, 1e5]` seconds, utilization inside `[0.05, 1]`,
    // and the price inside `[0.01, 100]` USD per GPU-hour, so every product
    // is finite and the `1e-9` relative comparison is not an ulp comparison.
    fn draw_job(tc: &TestCase) -> (Job, f64) {
        let job = Job {
            gpus: tc.draw(gs::integers::<u32>().min_value(1).max_value(64)),
            busy_s: tc.draw(gs::floats::<f64>().min_value(1e-3).max_value(1e5)),
            utilization: tc.draw(gs::floats::<f64>().min_value(0.05).max_value(1.0)),
            startup_s: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1e5)),
            idle_s: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1e5)),
            granularity_s: None,
            work: None,
        };
        let price = tc.draw(gs::floats::<f64>().min_value(0.01).max_value(100.0));
        (job, price)
    }

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
        let total = rate.work.unwrap();
        assert!(matches!(total.source, Work::Rate(2500.0)));
        assert_eq!(total.units, 25_000.0);
        assert_eq!(total.per_dollar, 25_000.0 / (2.0 * 10.0 / 3600.0 * 3.0));
        input.work = Some(Work::Count(1e6));
        assert_eq!(cost(&input, 3.0).unwrap().work.unwrap().units, 1e6);
        input.work = None;
        assert!(cost(&input, 3.0).unwrap().work.is_none());
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

    #[hegel::test]
    fn billed_cost_follows_from_busy_time_utilization_and_price(tc: TestCase) {
        let (job, price) = draw_job(&tc);
        let result = cost(&job, price).expect("bounded job");
        let running_s = job.busy_s / job.utilization;
        let wall_s = job.startup_s + running_s + job.idle_s;
        near(result.running_s, running_s);
        near(result.wall_s, wall_s);
        near(result.billed_s, wall_s);
        near(
            result.gpu_hours,
            f64::from(job.gpus) * result.billed_s / 3_600.0,
        );
        near(result.cost, result.gpu_hours * price);
        near(result.useful, job.busy_s / result.billed_s);
        assert!(result.useful > 0.0 && result.useful <= 1.0 + 1e-9);
        assert!(result.work.is_none());

        let mut more_gpus = job;
        more_gpus.gpus = job.gpus.saturating_mul(2);
        let scaled = cost(&more_gpus, price).expect("doubled GPU count");
        near(
            scaled.cost,
            result.cost * f64::from(more_gpus.gpus) / f64::from(job.gpus),
        );
        near(
            cost(&job, price * 2.0).expect("doubled price").cost,
            result.cost * 2.0,
        );
    }

    // Granularity is a whole number of seconds in `[1, 3600]`. The bill is
    // the wall clock rounded up to that quantum, and an exact multiple is
    // left alone.
    #[hegel::test]
    fn billing_quantum_rounds_the_wall_clock_up(tc: TestCase) {
        let (mut job, price) = draw_job(&tc);
        let granularity = tc.draw(gs::floats::<f64>().min_value(1.0).max_value(3_600.0));
        job.granularity_s = Some(granularity);
        let result = cost(&job, price).expect("job with a billing quantum");
        assert!(result.billed_s + 1e-9 >= result.wall_s);
        assert!(result.billed_s - result.wall_s < granularity + 1e-6);
        near(
            result.billed_s / granularity,
            (result.billed_s / granularity).round(),
        );
        near(result.useful, job.busy_s / result.billed_s);
    }

    #[hegel::test]
    fn work_per_dollar_is_completed_work_over_the_bill(tc: TestCase) {
        let (mut job, price) = draw_job(&tc);
        let rate = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1e9));
        job.work = Some(Work::Rate(rate));
        let rated = cost(&job, price).expect("rated work");
        let rated_total = rated.work.expect("rate produces a work total");
        near(rated_total.units, rate * job.busy_s);
        near(rated_total.per_dollar, rated_total.units / rated.cost);

        let count = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1e12));
        job.work = Some(Work::Count(count));
        let counted = cost(&job, price).expect("counted work");
        let counted_total = counted.work.expect("count produces a work total");
        near(counted_total.units, count);
        near(counted_total.per_dollar, count / counted.cost);
    }
}
