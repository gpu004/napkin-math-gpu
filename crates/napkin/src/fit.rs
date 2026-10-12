//! Capacity arithmetic: does the working set fit, and on how many GPUs.

use std::fmt;

use crate::profile::Profile;

/// Result of a fit calculation.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    /// `params * bits / 8`, in bytes.
    pub weight_bytes: f64,
    /// Non-weight bytes supplied by `--extra`.
    pub extra_bytes: f64,
    /// `weight_bytes + extra_bytes`.
    pub working_bytes: f64,
    /// Per-GPU capacity after `headroom`.
    pub usable_bytes_per_gpu: f64,
    /// Whether the working set fits on one GPU.
    pub fits_one: bool,
    /// `ceil(working / usable)`; weight-sharding lower bound.
    pub min_gpus: u64,
}

/// A fit that could not be computed.
#[derive(Debug)]
pub struct FitError(pub String);

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FitError {}

/// Computes the capacity floor. `headroom` is the fraction of each GPU's memory
/// reserved for runtime state, buffers, and fragmentation.
///
/// # Errors
/// Returns `FitError` for non-positive or non-finite inputs, or
/// `headroom >= 1`.
pub fn fit(
    profile: &Profile,
    params: f64,
    bits: f64,
    extra_bytes: f64,
    headroom: f64,
) -> Result<Fit, FitError> {
    if !(params > 0.0 && params.is_finite()) {
        return Err(FitError("params must be positive and finite".to_owned()));
    }
    if !(bits > 0.0 && bits.is_finite()) {
        return Err(FitError("bits must be positive and finite".to_owned()));
    }
    if !(extra_bytes >= 0.0 && extra_bytes.is_finite()) {
        return Err(FitError("extra must be non-negative and finite".to_owned()));
    }
    if !(0.0..1.0).contains(&headroom) {
        return Err(FitError("headroom must be in [0, 1)".to_owned()));
    }
    let weight_bytes = params * bits / 8.0;
    let working_bytes = weight_bytes + extra_bytes;
    let usable = profile.memory_gb * 1e9 * (1.0 - headroom);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "working/usable is positive and finite"
    )]
    let min_gpus = (working_bytes / usable).ceil() as u64;
    Ok(Fit {
        weight_bytes,
        extra_bytes,
        working_bytes,
        usable_bytes_per_gpu: usable,
        fits_one: working_bytes <= usable,
        min_gpus,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::find;

    #[test]
    fn seventy_b_bf16_needs_two_h100s() {
        let h100 = find("h100").unwrap().unwrap();
        let f = fit(&h100, 70e9, 16.0, 0.0, 0.1).unwrap();
        assert!((f.weight_bytes - 140e9).abs() < 1.0);
        assert!(!f.fits_one);
        assert_eq!(f.min_gpus, 2);
    }

    #[test]
    fn small_model_fits_one_gpu() {
        let h100 = find("h100").unwrap().unwrap();
        let f = fit(&h100, 7e9, 16.0, 0.0, 0.1).unwrap();
        assert!(f.fits_one);
        assert_eq!(f.min_gpus, 1);
    }

    #[test]
    fn rejects_bad_inputs() {
        let h100 = find("h100").unwrap().unwrap();
        assert!(fit(&h100, 0.0, 16.0, 0.0, 0.1).is_err());
        assert!(fit(&h100, 7e9, 16.0, 0.0, 1.0).is_err());
        assert!(fit(&h100, 7e9, f64::NAN, 0.0, 0.1).is_err());
    }
}
