//! Dated GPU price records generated from `prices/*.json` by `build.rs`.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Age after which a price record should be re-checked.
pub const STALE_AFTER_DAYS: i64 = 90;

/// Unit used by a provider's listed price.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Per {
    Second,
    Minute,
    Hour,
    /// 365 days (8,760 hours). Azure quotes a 1-year reservation as one
    /// total for the term, e.g. $551,221 for an 8-GPU H100 VM.
    Year,
}

impl Per {
    #[must_use]
    pub fn seconds(self) -> f64 {
        match self {
            Self::Second => 1.0,
            Self::Minute => 60.0,
            Self::Hour => 3600.0,
            Self::Year => 365.0 * 86_400.0,
        }
    }
}

impl fmt::Display for Per {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Second => "s",
            Self::Minute => "min",
            Self::Hour => "h",
            Self::Year => "year",
        })
    }
}

/// Provider's purchase terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terms {
    OnDemand,
    Spot,
    /// A reservation, savings plan, or committed-use discount that bills
    /// every hour of the term whether or not a job is running.
    Commitment {
        years: u8,
    },
}

impl fmt::Display for Terms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OnDemand => f.write_str("on-demand"),
            Self::Spot => f.write_str("spot"),
            Self::Commitment { years } => write!(f, "{years}-year commitment"),
        }
    }
}

/// Calendar date without a time zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

impl Date {
    /// Converts this civil date to days since 1970-01-01.
    #[must_use]
    pub fn days_since_epoch(self) -> i64 {
        let year = i64::from(self.year) - i64::from(self.month <= 2);
        let era = year.div_euclid(400);
        let year_of_era = year - era * 400;
        let month = i64::from(self.month);
        let day_of_year =
            (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(self.day) - 1;
        let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
        era * 146_097 + day_of_era - 719_468
    }

    /// Converts days since 1970-01-01 to a civil date.
    #[must_use]
    pub fn from_days(days: i64) -> Self {
        let days = days + 719_468;
        let era = days.div_euclid(146_097);
        let day_of_era = days - era * 146_097;
        let year_of_era =
            (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let mut year = year_of_era + era * 400;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let month_prime = (5 * day_of_year + 2) / 153;
        let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
        let month = month_prime + if month_prime < 10 { 3 } else { -9 };
        year += i64::from(month <= 2);
        Self {
            year: i32::try_from(year).unwrap_or(if year < 0 { i32::MIN } else { i32::MAX }),
            month: u8::try_from(month).unwrap_or(1),
            day: u8::try_from(day).unwrap_or(1),
        }
    }

    /// Returns the current UTC calendar date when the system clock is valid.
    #[must_use]
    pub fn today() -> Option<Self> {
        let seconds = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
        let days = i64::try_from(seconds / 86_400).ok()?;
        Some(Self::from_days(days))
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// One provider price and its dated source.
#[derive(Clone, Copy, Debug)]
pub struct PriceRecord {
    pub id: &'static str,
    pub provider: &'static str,
    pub sku: &'static str,
    pub gpu: &'static str,
    /// GPUs the listed amount pays for: 1 for Modal's per-GPU price, 8 for
    /// an AWS `p5.48xlarge` instance.
    pub gpus: u32,
    pub currency: &'static str,
    pub amount: f64,
    pub per: Per,
    pub terms: Terms,
    pub preemptible: bool,
    pub date: Date,
    pub source: &'static str,
    pub note: &'static str,
}

impl PriceRecord {
    /// Price formatted with its currency, billing unit, and the GPUs it buys.
    #[must_use]
    pub fn listed_price(&self) -> String {
        let price = if self.currency == "USD" {
            format!("${}/{per}", self.amount, per = self.per)
        } else {
            format!("{} {}/{per}", self.amount, self.currency, per = self.per)
        };
        if self.gpus == 1 {
            price
        } else {
            format!("{price} for {} GPUs", self.gpus)
        }
    }

    /// Terms formatted with any preemptibility qualifier.
    #[must_use]
    pub fn terms_label(&self) -> String {
        if self.preemptible {
            format!("{}, preemptible", self.terms)
        } else {
            self.terms.to_string()
        }
    }

    /// Price normalized to one GPU-hour.
    #[must_use]
    pub fn per_gpu_hour(&self) -> f64 {
        self.amount * 3600.0 / self.per.seconds() / f64::from(self.gpus)
    }

    /// Signed age in days relative to `today`.
    #[must_use]
    pub fn age_days(&self, today: Date) -> i64 {
        today.days_since_epoch() - self.date.days_since_epoch()
    }

    /// Whether this record is more than 90 days old.
    #[must_use]
    pub fn is_stale(&self, today: Date) -> bool {
        self.age_days(today) > STALE_AFTER_DAYS
    }
}

/// Returns all built-in price records in file-name order.
#[must_use]
pub fn builtin() -> &'static [PriceRecord] {
    GENERATED_PRICES
}

/// Finds a built-in price record by its identifier.
#[must_use]
pub fn find(id: &str) -> Option<&'static PriceRecord> {
    builtin().iter().find(|record| record.id == id)
}

/// Pairs each `gpu_a` record with the `gpu_b` record from the same provider,
/// terms, and currency, in record order. AWS on-demand `p5.48xlarge` (H100)
/// pairs with AWS on-demand `p6-b200.48xlarge` (B200); GCP's spot H100 has no
/// partner if GCP lists no spot B200. `build.rs` allows one record per
/// provider, GPU, and terms, so each record has at most one partner.
#[must_use]
pub fn pairs<'a>(
    records: &'a [PriceRecord],
    gpu_a: &str,
    gpu_b: &str,
) -> Vec<(&'a PriceRecord, &'a PriceRecord)> {
    records
        .iter()
        .filter(|a| a.gpu == gpu_a)
        .filter_map(|a| {
            records
                .iter()
                .find(|b| {
                    b.gpu == gpu_b
                        && b.provider == a.provider
                        && b.terms == a.terms
                        && b.currency == a.currency
                })
                .map(|b| (a, b))
        })
        .collect()
}

include!(concat!(env!("OUT_DIR"), "/prices.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile;
    use std::collections::BTreeSet;

    #[test]
    fn builtin_records_have_unique_ids_and_known_gpus() {
        let ids: BTreeSet<&str> = builtin().iter().map(|record| record.id).collect();
        assert_eq!(ids.len(), builtin().len());
        for record in builtin() {
            assert!(profile::find(record.gpu).is_some(), "{}", record.gpu);
        }
    }

    #[test]
    fn modal_h100_hourly_price_is_normalized() {
        let record = find("modal-h100").unwrap();
        assert!((record.per_gpu_hour() - 3.9492).abs() < 1e-12);
    }

    #[test]
    fn instance_price_is_divided_across_its_gpus() {
        let record = find("aws-h100").unwrap();
        assert_eq!(record.gpus, 8);
        assert!((record.per_gpu_hour() - 55.04 / 8.0).abs() < 1e-12);
    }

    #[test]
    fn yearly_reservation_is_spread_over_8760_hours() {
        let record = find("azure-h100-1y").unwrap();
        assert_eq!(record.per, Per::Year);
        assert!((record.per_gpu_hour() - 551_221.0 / 8760.0 / 8.0).abs() < 1e-9);
    }

    #[test]
    fn listed_price_includes_currency_billing_unit_and_gpus() {
        let mut record = *find("modal-h100").unwrap();
        assert_eq!(record.listed_price(), "$0.001097/s");
        record.currency = "EUR";
        assert_eq!(record.listed_price(), "0.001097 EUR/s");
        assert_eq!(
            find("azure-h100-1y").unwrap().listed_price(),
            "$551221/year for 8 GPUs"
        );
    }

    #[test]
    fn terms_label_includes_preemptibility_and_commitment_length() {
        let mut record = *find("modal-h100").unwrap();
        assert_eq!(record.terms_label(), "on-demand, preemptible");
        record.preemptible = false;
        assert_eq!(record.terms_label(), "on-demand");
        assert_eq!(
            find("gcp-b200-3y").unwrap().terms_label(),
            "3-year commitment"
        );
    }

    #[test]
    fn pairs_match_provider_and_terms() {
        let ids: Vec<(&str, &str)> = pairs(builtin(), "h100", "b200")
            .into_iter()
            .map(|(a, b)| (a.id, b.id))
            .collect();
        for pair in [
            ("aws-h100", "aws-b200"),
            ("gcp-h100-spot", "gcp-b200-spot"),
            ("modal-h100", "modal-b200"),
        ] {
            assert!(ids.contains(&pair), "{ids:?}");
        }
        assert!(!ids.iter().any(|(a, _)| *a == "gcp-h100"), "{ids:?}");
        assert!(!ids.iter().any(|(a, _)| a.starts_with("azure")), "{ids:?}");
    }

    #[test]
    fn records_become_stale_after_ninety_days() {
        let record = PriceRecord {
            id: "sample",
            provider: "Provider",
            sku: "SKU",
            gpu: "h100",
            gpus: 1,
            currency: "USD",
            amount: 1.0,
            per: Per::Hour,
            terms: Terms::OnDemand,
            preemptible: false,
            date: Date {
                year: 2026,
                month: 10,
                day: 12,
            },
            source: "https://example.com",
            note: "note",
        };
        let today = record.date;
        assert_eq!(record.age_days(today), 0);
        assert!(!record.is_stale(today));
        assert!(!record.is_stale(Date::from_days(today.days_since_epoch() + 90)));
        assert!(record.is_stale(Date::from_days(today.days_since_epoch() + 91)));
    }

    #[test]
    fn date_conversions_round_trip_and_epoch_is_zero() {
        for date in [
            Date {
                year: 2028,
                month: 2,
                day: 29,
            },
            Date {
                year: 1970,
                month: 1,
                day: 1,
            },
        ] {
            assert_eq!(Date::from_days(date.days_since_epoch()), date);
        }
        assert_eq!(
            Date {
                year: 1970,
                month: 1,
                day: 1
            }
            .days_since_epoch(),
            0
        );
    }

    #[test]
    fn today_is_current_or_later_than_price_cutoff() {
        assert!(Date::today().is_some_and(|today| {
            today
                >= Date {
                    year: 2026,
                    month: 11,
                    day: 5,
                }
        }));
    }
}
