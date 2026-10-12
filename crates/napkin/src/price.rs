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
}

impl Per {
    #[must_use]
    pub fn seconds(self) -> f64 {
        match self {
            Self::Second => 1.0,
            Self::Minute => 60.0,
            Self::Hour => 3600.0,
        }
    }
}

impl fmt::Display for Per {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Second => "s",
            Self::Minute => "min",
            Self::Hour => "h",
        })
    }
}

/// Provider's purchase terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terms {
    OnDemand,
    Spot,
}

impl fmt::Display for Terms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::OnDemand => "on-demand",
            Self::Spot => "spot",
        })
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
    /// Price formatted with its currency and billing unit.
    #[must_use]
    pub fn listed_price(&self) -> String {
        if self.currency == "USD" {
            format!("${}/{per}", self.amount, per = self.per)
        } else {
            format!("{} {}/{per}", self.amount, self.currency, per = self.per)
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
        self.amount * 3600.0 / self.per.seconds()
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
    fn listed_price_includes_currency_and_billing_unit() {
        let mut record = *find("modal-h100").unwrap();
        assert_eq!(record.listed_price(), "$0.001097/s");
        record.currency = "EUR";
        assert_eq!(record.listed_price(), "0.001097 EUR/s");
    }

    #[test]
    fn terms_label_includes_preemptibility() {
        let mut record = *find("modal-h100").unwrap();
        assert_eq!(record.terms_label(), "on-demand, preemptible");
        record.preemptible = false;
        assert_eq!(record.terms_label(), "on-demand");
    }

    #[test]
    fn records_become_stale_after_ninety_days() {
        let record = PriceRecord {
            id: "sample",
            provider: "Provider",
            sku: "SKU",
            gpu: "h100",
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
                    month: 10,
                    day: 12,
                }
        }));
    }
}
