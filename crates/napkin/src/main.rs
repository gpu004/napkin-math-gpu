//! `napkin`: the GPU napkin-math CLI.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use napkin::estimate::{self, Mode, Workload};
use napkin::fit;
use napkin::profile::{self, Basis, Dtype, Profile};
use napkin::units::{self, fmt_duration, fmt_intensity};

/// GPU napkin math: estimate workload performance from first principles.
#[derive(Parser, Debug)]
#[command(name = "napkin", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List the built-in GPU profiles.
    Gpus,
    /// Print the markdown "Numbers" tables for one GPU.
    Table {
        #[arg(long, default_value = "h100")]
        gpu: String,
        /// Print unrounded values.
        #[arg(long)]
        exact: bool,
    },
    /// Print the markdown cross-GPU comparison table.
    Compare {
        /// Print unrounded values.
        #[arg(long)]
        exact: bool,
    },
    /// Estimate a workload against a GPU profile.
    Estimate {
        #[arg(long, default_value = "h100")]
        gpu: String,
        #[arg(long, value_enum, default_value_t = Dtype::Fp16)]
        dtype: Dtype,
        /// Total FLOPs, e.g. "200T", "2e14".
        #[arg(long)]
        flops: String,
        /// HBM bytes touched, e.g. "1TB".
        #[arg(long)]
        bytes: String,
        /// Host-to-device bytes, e.g. "2GB".
        #[arg(long)]
        h2d: Option<String>,
        /// Device-to-host bytes.
        #[arg(long)]
        d2h: Option<String>,
        /// Bytes copied within host memory.
        #[arg(long)]
        host: Option<String>,
        /// GPU-to-GPU bytes.
        #[arg(long)]
        g2g: Option<String>,
        /// Kernel launches.
        #[arg(long, default_value_t = 0)]
        launches: u64,
        /// Model stages as overlapped (max) instead of serial (sum).
        #[arg(long)]
        overlap: bool,
        /// Force datasheet ceilings even when a measurement exists.
        #[arg(long)]
        ceiling: bool,
    },
    /// Capacity: does the model fit, and on how many GPUs.
    Fit {
        #[arg(long, default_value = "h100")]
        gpu: String,
        /// Parameter count, e.g. "70B".
        #[arg(long)]
        params: String,
        /// Bits per parameter.
        #[arg(long, default_value_t = 16.0)]
        bits: f64,
        /// Extra non-weight bytes, e.g. "20GB".
        #[arg(long)]
        extra: Option<String>,
        /// Per-GPU memory fraction held in reserve.
        #[arg(long, default_value_t = 0.1)]
        headroom: f64,
    },
    /// Rewrite the `<!-- napkin ... -->` marker blocks in README.md.
    Readme {
        /// Exit non-zero if the marker blocks are stale.
        #[arg(long)]
        check: bool,
        /// Rewrite the file in place (default when neither flag is given).
        #[arg(long, conflicts_with = "check")]
        write: bool,
        /// Markdown file to update.
        #[arg(default_value = "README.md")]
        path: PathBuf,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("napkin: {err}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<String, String> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Gpus => cmd_gpus(),
        Command::Table { gpu, exact } => cmd_table(gpu, *exact),
        Command::Compare { exact } => cmd_compare(*exact),
        Command::Estimate {
            gpu,
            dtype,
            flops,
            bytes,
            h2d,
            d2h,
            host,
            g2g,
            launches,
            overlap,
            ceiling,
        } => cmd_estimate(
            gpu,
            *dtype,
            flops,
            bytes,
            h2d.as_deref(),
            d2h.as_deref(),
            host.as_deref(),
            g2g.as_deref(),
            *launches,
            *overlap,
            *ceiling,
        ),
        Command::Fit {
            gpu,
            params,
            bits,
            extra,
            headroom,
        } => cmd_fit(gpu, params, *bits, extra.as_deref(), *headroom),
        Command::Readme { check, path, .. } => cmd_readme(path, *check),
    }
}

fn get_profile(id: &str) -> Result<Profile, String> {
    profile::find(id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("unknown GPU {id:?}; try `napkin gpus`"))
}

fn cmd_gpus() -> Result<String, String> {
    let profiles = profile::builtin().map_err(|e| e.to_string())?;
    let mut out = String::new();
    for p in &profiles {
        let basis = match &p.measured {
            Some(m) => format!("measured {}", &m.captured_at[..10.min(m.captured_at.len())]),
            None => "datasheet".to_owned(),
        };
        let _ = writeln!(
            out,
            "{:<6} {:<20} {:<6} GB   {:<8} {}",
            p.id, p.name, p.memory_gb, p.modal_gpu, basis
        );
    }
    Ok(out)
}

fn cmd_table(gpu: &str, exact: bool) -> Result<String, String> {
    let p = get_profile(gpu)?;
    napkin::table::numbers(&p, exact)
        .ok_or_else(|| format!("{gpu} has no measured profile; run napkin-bench on it"))
}

fn cmd_compare(exact: bool) -> Result<String, String> {
    let profiles = profile::builtin().map_err(|e| e.to_string())?;
    Ok(napkin::table::compare(&profiles, exact))
}

fn opt_bytes(value: Option<&str>) -> Result<f64, String> {
    value.map_or(Ok(0.0), |s| {
        units::parse_bytes(s).map_err(|e| e.to_string())
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "clap flatten would hide the flags"
)]
fn cmd_estimate(
    gpu: &str,
    dtype: Dtype,
    flops: &str,
    bytes: &str,
    h2d: Option<&str>,
    d2h: Option<&str>,
    host: Option<&str>,
    g2g: Option<&str>,
    launches: u64,
    overlap: bool,
    ceiling: bool,
) -> Result<String, String> {
    let p = get_profile(gpu)?;
    let workload = Workload {
        dtype,
        flops: units::parse_number(flops).map_err(|e| e.to_string())?,
        hbm_bytes: units::parse_bytes(bytes).map_err(|e| e.to_string())?,
        host_bytes: opt_bytes(host)?,
        h2d_bytes: opt_bytes(h2d)?,
        d2h_bytes: opt_bytes(d2h)?,
        g2g_bytes: opt_bytes(g2g)?,
        launches,
    };
    let mode = if overlap {
        Mode::Overlapped
    } else {
        Mode::Serial
    };
    let e = estimate::estimate(&p, &workload, mode, ceiling).map_err(|e| e.to_string())?;

    let basis = match (&p.measured, ceiling) {
        (Some(m), false) => format!("measured {}", &m.captured_at[..10.min(m.captured_at.len())]),
        _ => "datasheet".to_owned(),
    };
    let mut out = format!("{} · {dtype} · {basis}\n", p.name);
    let _ = writeln!(
        out,
        "{:<12}{}   ridge {}   {}",
        "intensity",
        fmt_intensity(e.intensity, 2),
        fmt_intensity(e.ridge, 2),
        e.bottleneck
    );
    let mut line = |name: &str, seconds: f64, show: bool| {
        if show {
            let _ = writeln!(out, "{name:<12}{}", fmt_duration(seconds, 2));
        }
    };
    line("compute", e.compute_s, true);
    line("memory", e.memory_s, true);
    line("device", e.device_s, true);
    line("host", e.host_s, e.host_s > 0.0);
    line("h2d", e.h2d_s, e.h2d_s > 0.0);
    line("d2h", e.d2h_s, e.d2h_s > 0.0);
    line("g2g", e.g2g_s, e.g2g_s > 0.0);
    line("launch", e.launch_s, e.launch_s > 0.0);
    let _ = writeln!(
        out,
        "{:<12}{}   {}",
        "total",
        fmt_duration(e.total_s, 2),
        e.bottleneck
    );
    if e.compute_basis == Basis::Ceiling && p.measured.is_some() && !ceiling {
        let _ = writeln!(out, "(compute: no measured {dtype}; datasheet used)");
    }
    Ok(out)
}

fn cmd_fit(
    gpu: &str,
    params: &str,
    bits: f64,
    extra: Option<&str>,
    headroom: f64,
) -> Result<String, String> {
    let p = get_profile(gpu)?;
    let f = fit::fit(
        &p,
        units::parse_count(params).map_err(|e| e.to_string())?,
        bits,
        opt_bytes(extra)?,
        headroom,
    )
    .map_err(|e| e.to_string())?;

    let gb = |b: f64| format!("{} GB", units::fmt_sig(b / 1e9, 3));
    let mut out = format!("{} · {params} params · {bits}-bit\n", p.name);
    let _ = writeln!(out, "weights     {}", gb(f.weight_bytes));
    if f.extra_bytes > 0.0 {
        let _ = writeln!(out, "extra       {}", gb(f.extra_bytes));
    }
    let _ = writeln!(out, "working set {}", gb(f.working_bytes));
    let _ = writeln!(
        out,
        "usable      {} per GPU ({:.0}% headroom)",
        gb(f.usable_bytes_per_gpu),
        headroom * 100.0
    );
    if f.fits_one {
        out.push_str("fits        yes, on one GPU\n");
    } else {
        let _ = writeln!(
            out,
            "fits        no — needs at least {} GPUs (weights sharded)",
            f.min_gpus
        );
    }
    Ok(out)
}

const MARKER_PREFIX: &str = "<!-- napkin ";
const MARKER_END: &str = "<!-- /napkin -->";

fn render_marker(command: &str) -> Result<String, String> {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    match tokens.as_slice() {
        ["table", "--gpu", gpu] => cmd_table(gpu, false),
        ["compare"] => cmd_compare(false),
        _ => Err(format!("unsupported marker command {command:?}")),
    }
}

fn rewrite_markers(content: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut lines = content.lines().peekable();
    while let Some(line) = lines.next() {
        out.push_str(line);
        out.push('\n');
        if let Some(rest) = line.trim().strip_prefix(MARKER_PREFIX) {
            let command = rest.strip_suffix("-->").map_or(rest, |c| c).trim();
            let body = render_marker(command)?;
            out.push_str(&body);
            let mut closed = false;
            for skipped in lines.by_ref() {
                if skipped.trim() == MARKER_END {
                    out.push_str(skipped);
                    out.push('\n');
                    closed = true;
                    break;
                }
            }
            if !closed {
                return Err(format!("marker {line:?} has no {MARKER_END} line"));
            }
        }
    }
    Ok(out)
}

fn cmd_readme(path: &std::path::Path, check: bool) -> Result<String, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let updated = rewrite_markers(&content)?;
    if check {
        if updated == content {
            return Ok(format!("{}: napkin blocks are current\n", path.display()));
        }
        return Err(format!(
            "{}: napkin blocks are stale; run `napkin readme --write`",
            path.display()
        ));
    }
    if updated != content {
        std::fs::write(path, &updated)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        return Ok(format!("{}: napkin blocks updated\n", path.display()));
    }
    Ok(format!("{}: already current\n", path.display()))
}
