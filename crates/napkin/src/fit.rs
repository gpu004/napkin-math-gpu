//! Capacity arithmetic: does the working set fit, and on how many GPUs.

use std::fmt;

use crate::profile::Profile;

/// Result of a fit calculation.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    /// `params * bits / 8`, in bytes.
    pub weight_bytes: f64,
    /// Non-weight bytes supplied by `--extra` that shard with the weights.
    pub extra_bytes: f64,
    /// Bytes held by every GPU regardless of GPU count.
    pub per_gpu_bytes: f64,
    /// `weight_bytes + extra_bytes + per_gpu_bytes` (one-GPU working set).
    pub working_bytes: f64,
    /// Per-GPU capacity after `headroom`.
    pub usable_bytes_per_gpu: f64,
    /// Whether the working set fits on one GPU.
    pub fits_one: bool,
    /// Lower bound assuming weights and extra bytes shard evenly, ignoring
    /// communication.
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

/// Computes the capacity floor. `extra_bytes` shard with the weights, while
/// `per_gpu_bytes` are held by every GPU. `headroom` is the fraction of each
/// GPU's memory reserved for runtime state, buffers, and fragmentation.
///
/// # Errors
/// Returns `FitError` for non-positive or non-finite inputs, or
/// `headroom >= 1`.
pub fn fit(
    profile: &Profile,
    params: f64,
    bits: f64,
    extra_bytes: f64,
    per_gpu_bytes: f64,
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
    if !(per_gpu_bytes >= 0.0 && per_gpu_bytes.is_finite()) {
        return Err(FitError(
            "per-GPU bytes must be non-negative and finite".to_owned(),
        ));
    }
    if !(0.0..1.0).contains(&headroom) {
        return Err(FitError("headroom must be in [0, 1)".to_owned()));
    }
    let weight_bytes = params * bits / 8.0;
    let usable = profile.memory_gb * 1e9 * (1.0 - headroom);
    if per_gpu_bytes >= usable {
        return Err(FitError(
            "per-GPU bytes leave no usable memory for weights".to_owned(),
        ));
    }
    let working_bytes = weight_bytes + extra_bytes + per_gpu_bytes;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the quotient is positive and the result is bounded by u64"
    )]
    let min_gpus = ((weight_bytes + extra_bytes) / (usable - per_gpu_bytes)).ceil() as u64;
    Ok(Fit {
        weight_bytes,
        extra_bytes,
        per_gpu_bytes,
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
        let h100 = find("h100").unwrap();
        let f = fit(h100, 70e9, 16.0, 0.0, 0.0, 0.1).unwrap();
        assert!((f.weight_bytes - 140e9).abs() < 1.0);
        assert!(!f.fits_one);
        assert_eq!(f.min_gpus, 2);
    }

    #[test]
    fn small_model_fits_one_gpu() {
        let h100 = find("h100").unwrap();
        let f = fit(h100, 7e9, 16.0, 0.0, 0.0, 0.1).unwrap();
        assert!(f.fits_one);
        assert_eq!(f.min_gpus, 1);
    }

    #[test]
    fn rejects_bad_inputs() {
        let h100 = find("h100").unwrap();
        assert!(fit(h100, 0.0, 16.0, 0.0, 0.0, 0.1).is_err());
        assert!(fit(h100, 7e9, 16.0, 0.0, 0.0, 1.0).is_err());
        assert!(fit(h100, 7e9, f64::NAN, 0.0, 0.0, 0.1).is_err());
        assert!(fit(h100, 7e9, 16.0, 0.0, -1.0, 0.1).is_err());
        assert!(fit(h100, 7e9, 16.0, 0.0, f64::NAN, 0.1).is_err());
    }

    #[test]
    fn replicated_overhead_is_not_divided() {
        let h100 = find("h100").unwrap();
        let f = fit(h100, 70e9, 16.0, 0.0, 10e9, 0.1).unwrap();
        assert_eq!(f.min_gpus, 3);
        assert!((f.working_bytes - 150e9).abs() < 1.0);
    }

    #[test]
    fn per_gpu_overhead_must_leave_memory_for_weights() {
        let h100 = find("h100").unwrap();
        let error = fit(h100, 7e9, 16.0, 0.0, 72e9, 0.1).unwrap_err();
        assert!(error.to_string().contains("leave no usable memory"));
    }
}
