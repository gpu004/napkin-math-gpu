#[must_use]
pub(crate) fn median(samples: &mut [f64]) -> Option<f64> {
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    let upper = *samples.get(n / 2)?;
    if n % 2 == 1 {
        Some(upper)
    } else {
        Some(f64::midpoint(*samples.get(n / 2 - 1)?, upper))
    }
}

#[derive(Debug)]
pub(crate) struct XorShift(u64);

impl XorShift {
    #[must_use]
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        #[expect(clippy::cast_possible_truncation, reason = "the value is in [-1, 1]")]
        let v = (f64::from(bits) / f64::from(1u32 << 24) * 2.0 - 1.0) as f32;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_of_no_samples_is_none() {
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn median_picks_the_middle_or_averages_the_two_middles() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }
}
