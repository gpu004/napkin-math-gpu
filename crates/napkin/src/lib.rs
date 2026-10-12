//! GPU napkin math: estimate workload performance from datasheet ceilings and
//! measured GPU profiles, in the spirit of sirupsen/napkin-math.

#![forbid(unsafe_code)]

pub mod cost;
pub mod estimate;
pub mod fit;
pub mod price;
pub mod profile;
pub mod table;
pub mod units;
