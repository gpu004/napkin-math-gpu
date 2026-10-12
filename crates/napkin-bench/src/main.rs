//! `napkin-bench`: measures one or two GPUs and prints a `Measured` JSON
//! profile on stdout. Progress goes to stderr. Exits non-zero if any
//! correctness check fails.

mod cuda;
mod gpu_state;
mod stats;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut quick = false;
    let mut gpu_request = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--quick" => quick = true,
            "--gpu-request" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("napkin-bench: --gpu-request needs a value");
                    return ExitCode::from(2);
                }
                gpu_request.clone_from(&args[i]);
            }
            "--help" | "-h" => {
                eprintln!("usage: napkin-bench [--quick] [--gpu-request \"H100!:2\"]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("napkin-bench: unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    let cfg = if quick {
        cuda::Config::quick()
    } else {
        cuda::Config::full()
    };
    match cuda::run(&cfg, &gpu_request) {
        Ok(measured) => match serde_json::to_string_pretty(&measured) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("napkin-bench: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("napkin-bench: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Current UTC date/time as `YYYY-MM-DD`, without a chrono dependency.
#[must_use]
fn iso_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let days = (secs / 86_400).cast_signed();
    let secs_of_day = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!(
        "{year:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60
    )
}
