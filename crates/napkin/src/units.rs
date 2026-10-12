//! Quantity parsing and napkin formatting: a few significant digits, always
//! with units attached.

use std::fmt;

/// A quantity that failed to parse.
#[derive(Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

fn split_unit(s: &str) -> Result<(f64, &str), ParseError> {
    let s = s.trim();
    let boundary = s
        .find(|c: char| c.is_alphabetic() && c != 'e' && c != 'E')
        .unwrap_or(s.len());
    let (num, unit) = s.split_at(boundary);
    let (num, unit) =
        if unit.is_empty() && num.ends_with(['e', 'E']) && num.trim().parse::<f64>().is_err() {
            num.split_at(num.len() - 1)
        } else {
            (num, unit)
        };
    let value: f64 = num
        .trim()
        .parse()
        .map_err(|_| ParseError(format!("cannot parse {s:?} as a quantity")))?;
    Ok((value, unit.trim()))
}

fn si_suffix(unit: &str) -> Option<f64> {
    match unit {
        "" => Some(1.0),
        "K" => Some(1e3),
        "M" => Some(1e6),
        "G" => Some(1e9),
        "T" => Some(1e12),
        "P" => Some(1e15),
        "E" => Some(1e18),
        _ => None,
    }
}

fn byte_suffix(unit: &str) -> Option<f64> {
    match unit {
        "" | "B" => Some(1.0),
        "KB" => Some(1e3),
        "MB" => Some(1e6),
        "GB" => Some(1e9),
        "TB" => Some(1e12),
        "PB" => Some(1e15),
        "KiB" => Some(2.0_f64.powi(10)),
        "MiB" => Some(2.0_f64.powi(20)),
        "GiB" => Some(2.0_f64.powi(30)),
        "TiB" => Some(2.0_f64.powi(40)),
        "PiB" => Some(2.0_f64.powi(50)),
        _ => si_suffix(unit),
    }
}

fn count_suffix(unit: &str) -> Option<f64> {
    match unit {
        "B" => Some(1e9),
        _ => si_suffix(unit),
    }
}

fn parse_with(s: &str, suffix: fn(&str) -> Option<f64>) -> Result<f64, ParseError> {
    let (value, unit) = split_unit(s)?;
    let multiplier =
        suffix(unit).ok_or_else(|| ParseError(format!("unrecognized unit in {s:?}")))?;
    Ok(value * multiplier)
}

/// Parses a plain scaled number: `"2e14"`, `"200T"`, `"140G"`.
///
/// # Errors
/// Returns `ParseError` for malformed input or an unknown suffix.
pub fn parse_number(s: &str) -> Result<f64, ParseError> {
    parse_with(s, si_suffix)
}

/// Parses a byte quantity: `"1TB"`, `"512MiB"`, `"1.5 GB"`, `"700B"` (bytes).
///
/// # Errors
/// Returns `ParseError` for malformed input or an unknown suffix.
pub fn parse_bytes(s: &str) -> Result<f64, ParseError> {
    parse_with(s, byte_suffix)
}

/// Parses a parameter count: `"70B"` is 70 billion, `"7M"` is 7 million.
///
/// # Errors
/// Returns `ParseError` for malformed input or an unknown suffix.
pub fn parse_count(s: &str) -> Result<f64, ParseError> {
    parse_with(s, count_suffix)
}

/// Parses a duration with an explicit unit and returns seconds.
///
/// # Errors
/// Returns `ParseError` for a missing, ambiguous, or unrecognized time unit.
pub fn parse_duration(s: &str) -> Result<f64, ParseError> {
    let (value, unit) = split_unit(s)?;
    let multiplier = match unit {
        "ns" => 1e-9,
        "us" | "μs" => 1e-6,
        "ms" => 1e-3,
        "s" => 1.0,
        "min" => 60.0,
        "h" => 3600.0,
        "" => {
            return Err(ParseError(format!(
                "duration {s:?} needs a unit; add a unit, e.g. 10s"
            )));
        }
        _ => return Err(ParseError(format!("unrecognized duration unit in {s:?}"))),
    };
    Ok(value * multiplier)
}

/// Formats a price to three significant digits.
#[must_use]
pub fn fmt_money(value: f64, currency: &str) -> String {
    if currency == "USD" {
        format!("${}", fmt_sig(value, 3))
    } else {
        format!("{} {currency}", fmt_sig(value, 3))
    }
}

/// Formats a count with an SI suffix through tera.
#[must_use]
pub fn fmt_count(value: f64, sig: u32) -> String {
    let magnitude = value.abs();
    let (scale, suffix) = if magnitude >= 1e12 {
        (1e12, "T")
    } else if magnitude >= 1e9 {
        (1e9, "G")
    } else if magnitude >= 1e6 {
        (1e6, "M")
    } else if magnitude >= 1e3 {
        (1e3, "K")
    } else {
        return fmt_sig(value, sig);
    };
    format!("{}{suffix}", fmt_sig(value / scale, sig))
}

/// Rounds `v` to `sig` significant digits.
#[must_use]
pub fn round_sig(v: f64, sig: u32) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let digits = v.abs().log10().floor();
    let factor = 10_f64.powf(digits - f64::from(sig) + 1.0);
    (v / factor).round() * factor
}

/// Formats `v` at `sig` significant digits without trailing zeros.
#[must_use]
pub fn fmt_sig(v: f64, sig: u32) -> String {
    let rounded = round_sig(v, sig);
    if rounded == 0.0 {
        return "0".to_owned();
    }
    if rounded.fract() == 0.0 && rounded.abs() < 1e15 {
        format!("{rounded:.0}")
    } else {
        let decimal_places =
            (f64::from(sig.saturating_sub(1)) - rounded.abs().log10().floor()).max(0.0);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the decimal precision is bounded by the significant digit count and f64 range"
        )]
        let decimals = decimal_places as usize;
        format!("{rounded:.decimals$}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
}

/// Formats a duration in seconds as ns/μs/ms/s at `sig` significant digits.
#[must_use]
pub fn fmt_duration(seconds: f64, sig: u32) -> String {
    if seconds == 0.0 {
        return "0 s".to_owned();
    }
    let abs = seconds.abs();
    if abs < 1e-6 {
        format!("{} ns", fmt_sig(seconds * 1e9, sig))
    } else if abs < 1e-3 {
        format!("{} μs", fmt_sig(seconds * 1e6, sig))
    } else if abs < 1.0 {
        format!("{} ms", fmt_sig(seconds * 1e3, sig))
    } else {
        format!("{} s", fmt_sig(seconds, sig))
    }
}

/// Formats a GB/s rate, promoting to TB/s above 1,000.
#[must_use]
pub fn fmt_gbps(gbps: f64, sig: u32) -> String {
    if gbps.abs() >= 1000.0 {
        format!("{} TB/s", fmt_sig(gbps / 1000.0, sig))
    } else {
        format!("{} GB/s", fmt_sig(gbps, sig))
    }
}

/// Formats a TFLOP/s rate, promoting to PFLOP/s above 1,000.
#[must_use]
pub fn fmt_tflops(tflops: f64, sig: u32) -> String {
    if tflops.abs() >= 1000.0 {
        format!("{} PFLOP/s", fmt_sig(tflops / 1000.0, sig))
    } else {
        format!("{} TFLOP/s", fmt_sig(tflops, sig))
    }
}

/// Formats an arithmetic intensity in FLOP/byte.
#[must_use]
pub fn fmt_intensity(flops_per_byte: f64, sig: u32) -> String {
    if flops_per_byte.is_infinite() {
        return "infinite FLOP/B".to_owned();
    }
    format!("{} FLOP/B", fmt_sig(flops_per_byte, sig))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_scaled_numbers() {
        assert!((parse_number("2e14").unwrap() - 2e14).abs() < f64::EPSILON);
        assert!((parse_number("200T").unwrap() - 200e12).abs() < 1.0);
        assert!((parse_number("140G").unwrap() - 140e9).abs() < 1.0);
        assert!((parse_number("512").unwrap() - 512.0).abs() < f64::EPSILON);
        assert!(parse_number("x").is_err());
    }

    #[test]
    fn parses_byte_quantities() {
        assert!((parse_bytes("1TB").unwrap() - 1e12).abs() < 1.0);
        assert!((parse_bytes("512MiB").unwrap() - 512.0 * 1024.0 * 1024.0).abs() < 1.0);
        assert!((parse_bytes("1.5 GB").unwrap() - 1.5e9).abs() < 1.0);
        assert!((parse_bytes("700B").unwrap() - 700.0).abs() < f64::EPSILON);
        assert!(parse_bytes("5QB").is_err());
    }

    #[test]
    fn parses_parameter_counts() {
        assert!((parse_count("70B").unwrap() - 70e9).abs() < 1.0);
        assert!((parse_count("7M").unwrap() - 7e6).abs() < 1.0);
        assert!((parse_count("1.8T").unwrap() - 1.8e12).abs() < 1.0);
    }

    #[test]
    fn rounds_to_one_significant_digit() {
        assert!((round_sig(55.0, 1) - 60.0).abs() < f64::EPSILON);
        assert!((round_sig(717.0, 1) - 700.0).abs() < f64::EPSILON);
        assert!((round_sig(393.0, 1) - 400.0).abs() < f64::EPSILON);
        assert!((round_sig(4.66, 1) - 5.0).abs() < f64::EPSILON);
        assert!((round_sig(2999.0, 1) - 3000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rounds_ridge_to_two_significant_digits() {
        assert!((round_sig(239.2, 2) - 240.0).abs() < f64::EPSILON);
    }

    #[test]
    fn formats_durations_and_rates() {
        assert_eq!(fmt_duration(0.29, 2), "290 ms");
        assert_eq!(fmt_duration(4.66e-6, 1), "5 μs");
        assert_eq!(fmt_gbps(3000.0, 1), "3 TB/s");
        assert_eq!(fmt_gbps(55.0, 1), "60 GB/s");
        assert_eq!(fmt_tflops(717.0, 1), "700 TFLOP/s");
        assert_eq!(fmt_tflops(2250.0, 1), "2 PFLOP/s");
    }

    #[test]
    fn parses_explicit_durations() {
        assert_eq!(parse_duration("1ns").unwrap(), 1e-9);
        assert_eq!(parse_duration("2us").unwrap(), 2e-6);
        assert_eq!(parse_duration("3μs").unwrap(), 3e-6);
        assert_eq!(parse_duration("4ms").unwrap(), 4e-3);
        assert_eq!(parse_duration("5s").unwrap(), 5.0);
        assert_eq!(parse_duration("2min").unwrap(), 120.0);
        assert_eq!(parse_duration("1h").unwrap(), 3600.0);
        assert!(parse_duration("10").is_err());
        assert!(parse_duration("1m").is_err());
    }

    #[test]
    fn formats_money_and_counts() {
        assert_eq!(fmt_money(0.016_666, "USD"), "$0.0167");
        assert_eq!(fmt_money(0.016_666, "EUR"), "0.0167 EUR");
        assert_eq!(fmt_count(1.5e6, 3), "1.5M");
        assert_eq!(fmt_count(25e3, 3), "25K");
        assert_eq!(fmt_count(900.0, 3), "900");
        assert_eq!(fmt_sig(0.016_666, 3), "0.0167");
        assert_eq!(fmt_sig(0.005_555, 3), "0.00556");
    }
}
