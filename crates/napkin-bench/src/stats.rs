//! Small statistics helpers.

/// Median of a non-empty sample set.
#[must_use]
pub fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        f64::midpoint(samples[n / 2 - 1], samples[n / 2])
    }
}

/// Deterministic xorshift64* for benchmark inputs.
#[derive(Debug)]
pub struct XorShift(u64);

impl XorShift {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        #[expect(clippy::cast_possible_truncation, reason = "the value is in [-1, 1]")]
        let v = (f64::from(bits) / f64::from(1u32 << 24) * 2.0 - 1.0) as f32;
        v
    }
}
