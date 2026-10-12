//! `napkin`: the GPU napkin-math CLI.

use std::fmt::Write as _;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgGroup, Parser, Subcommand};
use napkin::collective::{self, Query, Regime};
use napkin::cost::{self, Job, Work};
use napkin::estimate::{self, CopyMode, Mode, Transfer, TransferTime, Workload};
use napkin::fit;
use napkin::llm::{self, Model, Workload as LlmWorkload};
use napkin::price::{self, Date, STALE_AFTER_DAYS};
use napkin::profile::collective::{Collective, Topology};
use napkin::profile::{self, Basis, Dtype, Profile};
use napkin::units::{
    self, fmt_binary_bytes, fmt_bytes, fmt_count, fmt_duration, fmt_gbps, fmt_intensity, fmt_money,
    fmt_si, fmt_sig,
};
use napkin::validation::{self, Case, RangeInfo};

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
        /// Number of copies the --h2d bytes are split into [default: 1 with --h2d].
        #[arg(long)]
        h2d_copies: Option<u64>,
        /// Device-to-host bytes.
        #[arg(long)]
        d2h: Option<String>,
        /// Number of copies the --d2h bytes are split into [default: 1 with --d2h].
        #[arg(long)]
        d2h_copies: Option<u64>,
        /// Bytes copied within host memory.
        #[arg(long)]
        host: Option<String>,
        /// GPU-to-GPU bytes.
        #[arg(long)]
        g2g: Option<String>,
        /// Number of copies the --g2g bytes are split into [default: 1 with --g2g].
        #[arg(long)]
        g2g_copies: Option<u64>,
        /// sync: the host waits for every copy; async: copies of one kind are
        /// queued and waited for once.
        #[arg(long, value_enum, default_value_t = CopyMode::Sync)]
        copy_mode: CopyMode,
        /// Back-to-back kernel launches; copies are counted separately.
        #[arg(long, default_value_t = 0)]
        launches: u64,
        /// Model stages as overlapped (max) instead of serial (sum).
        #[arg(long)]
        overlap: bool,
        /// Force datasheet ceilings even when a measurement exists.
        #[arg(long)]
        ceiling: bool,
    },
    /// Estimate one NCCL collective from measured NCCL Tests curves.
    Collective {
        #[arg(long, default_value = "h100")]
        gpu: String,
        /// GPUs taking part.
        #[arg(long)]
        count: u32,
        #[arg(long, value_enum)]
        topology: Topology,
        #[arg(long, value_enum)]
        op: Collective,
        #[arg(long, value_enum, default_value_t = Dtype::Bf16)]
        dtype: Dtype,
        /// NCCL Tests size: per-rank buffer for `AllReduce`, total gathered
        /// output for `AllGather`, or total per-rank input for `ReduceScatter`.
        #[arg(long)]
        bytes: String,
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
        /// Non-weight bytes that shard with the weights across GPUs, e.g. KV cache.
        #[arg(long)]
        extra: Option<String>,
        /// Bytes every GPU holds regardless of GPU count, e.g. CUDA context and activation workspace.
        #[arg(long)]
        per_gpu: Option<String>,
        /// Per-GPU memory fraction held in reserve.
        #[arg(long, default_value_t = 0.1)]
        headroom: f64,
    },
    /// Estimate billed GPU cost from busy time and dated or supplied pricing.
    #[command(group(ArgGroup::new("price_source").required(true).args(["price", "offer"])))]
    Cost {
        /// Number of GPUs you rent; napkin fit's minimum is a capacity floor, not a runtime.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
        gpus: u32,
        /// Time each GPU spends working at the modelled rate, with units (e.g. 10s).
        #[arg(long)]
        time: String,
        /// Price per GPU-hour.
        #[arg(long)]
        price: Option<String>,
        /// Use a dated built-in price record.
        #[arg(long)]
        offer: Option<String>,
        /// Currency for --price.
        #[arg(long, default_value = "USD", requires = "price")]
        currency: String,
        /// Fraction of on-GPU time spent working at the modelled rate.
        #[arg(long)]
        utilization: Option<f64>,
        /// Cold-start time for each rented GPU, with units.
        #[arg(long)]
        startup: Option<String>,
        /// Paid idle time after running, with units.
        #[arg(long)]
        idle: Option<String>,
        /// Billing increment to round wall time up to, with units.
        #[arg(long)]
        granularity: Option<String>,
        /// Completed amount of work.
        #[arg(long, conflicts_with = "throughput")]
        work: Option<String>,
        /// Measured work per busy second.
        #[arg(long, conflicts_with = "work")]
        throughput: Option<String>,
        /// Label for work units.
        #[arg(long, default_value = "units")]
        unit: String,
    },
    /// List dated GPU price records and their age.
    Prices,
    /// Derive memory, prefill and decode inputs for a transformer from explicit architecture and workload flags.
    Llm(LlmArgs),
    /// Rewrite the `<!-- napkin ... -->` marker blocks in Markdown files (README.md by default).
    Readme {
        /// Exit non-zero if the marker blocks are stale.
        #[arg(long)]
        check: bool,
        /// Rewrite the files in place (default when neither flag is given).
        #[arg(long, conflicts_with = "check")]
        write: bool,
        /// Markdown files to update.
        #[arg(default_value = "README.md", num_args = 1..)]
        paths: Vec<PathBuf>,
    },
    /// Render the prediction-versus-observation validation report.
    Validate {
        /// Write the report to this markdown file.
        #[arg(long, value_name = "PATH", conflicts_with = "check")]
        write: Option<PathBuf>,
        /// Exit non-zero if the report is stale.
        #[arg(long, value_name = "PATH", conflicts_with = "write")]
        check: Option<PathBuf>,
    },
}

#[derive(clap::Args, Debug)]
struct LlmArgs {
    /// GPU profile to use.
    #[arg(long, default_value = "h100")]
    gpu: String,
    /// Compute and activation dtype.
    #[arg(long, value_enum, default_value_t = Dtype::Bf16)]
    dtype: Dtype,
    /// Number of transformer layers.
    #[arg(long)]
    layers: NonZeroU64,
    /// Hidden width.
    #[arg(long)]
    hidden: NonZeroU64,
    /// Query heads.
    #[arg(long)]
    heads: NonZeroU64,
    /// KV heads (equal to --heads for multi-head attention, fewer for GQA).
    #[arg(long)]
    kv_heads: NonZeroU64,
    /// Width of each attention head.
    #[arg(long)]
    head_dim: NonZeroU64,
    /// Parameter count, e.g. "70B".
    #[arg(long)]
    params: String,
    /// Bits per model weight.
    #[arg(long, default_value_t = 16.0)]
    weight_bits: f64,
    /// Bits per KV-cache element.
    #[arg(long, default_value_t = 16.0)]
    kv_bits: f64,
    /// Concurrent requests.
    #[arg(long, default_value = "1")]
    batch: NonZeroU64,
    /// Prompt tokens per request.
    #[arg(long)]
    prompt: NonZeroU64,
    /// Generated tokens per request.
    #[arg(long)]
    generated: NonZeroU64,
    /// Bytes every GPU holds regardless of GPU count, e.g. CUDA context and activation workspace.
    #[arg(long)]
    per_gpu: Option<String>,
    /// Per-GPU memory fraction held in reserve.
    #[arg(long)]
    headroom: Option<f64>,
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
    dispatch(&cli.command)
}

fn dispatch(command: &Command) -> Result<String, String> {
    match command {
        Command::Gpus => Ok(cmd_gpus()),
        Command::Table { gpu, exact } => cmd_table(gpu, *exact),
        Command::Compare { exact } => Ok(cmd_compare(*exact)),
        Command::Estimate {
            gpu,
            dtype,
            flops,
            bytes,
            h2d,
            h2d_copies,
            d2h,
            d2h_copies,
            host,
            g2g,
            g2g_copies,
            copy_mode,
            launches,
            overlap,
            ceiling,
        } => cmd_estimate(
            gpu,
            *dtype,
            flops,
            bytes,
            transfer(h2d.as_deref(), *h2d_copies)?,
            transfer(d2h.as_deref(), *d2h_copies)?,
            host.as_deref(),
            transfer(g2g.as_deref(), *g2g_copies)?,
            *copy_mode,
            *launches,
            *overlap,
            *ceiling,
        ),
        Command::Collective {
            gpu,
            count,
            topology,
            op,
            dtype,
            bytes,
        } => cmd_collective(gpu, *count, *topology, *op, *dtype, bytes),
        Command::Fit {
            gpu,
            params,
            bits,
            extra,
            per_gpu,
            headroom,
        } => cmd_fit(
            gpu,
            params,
            *bits,
            extra.as_deref(),
            per_gpu.as_deref(),
            *headroom,
        ),
        Command::Llm(args) => cmd_llm(args),
        Command::Cost {
            gpus,
            time,
            price,
            offer,
            currency,
            utilization,
            startup,
            idle,
            granularity,
            work,
            throughput,
            unit,
        } => cmd_cost(
            *gpus,
            time,
            price.as_deref(),
            offer.as_deref(),
            currency,
            *utilization,
            startup.as_deref(),
            idle.as_deref(),
            granularity.as_deref(),
            work.as_deref(),
            throughput.as_deref(),
            unit,
        ),
        Command::Prices => cmd_prices(),
        Command::Readme { check, paths, .. } => cmd_readme(paths, *check),
        Command::Validate { write, check } => cmd_validate(write.as_deref(), check.as_deref()),
    }
}

fn cmd_collective(
    gpu: &str,
    gpu_count: u32,
    topology: Topology,
    op: Collective,
    dtype: Dtype,
    bytes: &str,
) -> Result<String, String> {
    let gpu_profile = get_profile(gpu)?;
    let query = Query {
        gpu,
        gpu_count,
        topology,
        collective: op,
        dtype,
        bytes: units::parse_bytes(bytes).map_err(|error| error.to_string())?,
    };
    let estimate = collective::estimate(&query)?;
    let evidence = estimate.evidence;
    let redop = if op == Collective::AllGather {
        String::new()
    } else {
        format!(" {}", evidence.redop)
    };
    let mut out = format!(
        "{} × {} · {} · {} · {}{} · {} ({})\n",
        gpu_profile.name,
        gpu_count,
        topology,
        op,
        dtype,
        redop,
        fmt_binary_bytes(query.bytes, 3),
        op.size_convention()
    );
    let regime_detail = match estimate.regime {
        Regime::Latency => "fixed per-call cost is at least half the time",
        Regime::Bandwidth => "time grows with size",
    };
    let _ = writeln!(out, "regime      {} ({regime_detail})", estimate.regime);
    let _ = writeln!(out, "time        {}", fmt_duration(estimate.time_s, 3));
    let bus_factor = match op {
        Collective::AllReduce => "2(n-1)/n",
        Collective::AllGather | Collective::ReduceScatter => "(n-1)/n",
    };
    let _ = writeln!(
        out,
        "collective  algbw {} · busbw {} (busbw = algbw × {bus_factor})",
        fmt_gbps(estimate.algbw_gbps, 3),
        fmt_gbps(estimate.busbw_gbps, 3)
    );
    if estimate.lower == estimate.upper {
        let _ = write!(
            out,
            "curve       {} {}",
            fmt_curve_bytes(estimate.lower.bytes),
            fmt_duration(estimate.lower.time_us / 1e6, 3)
        );
    } else {
        let _ = write!(
            out,
            "curve       {} {} .. {} {}",
            fmt_curve_bytes(estimate.lower.bytes),
            fmt_duration(estimate.lower.time_us / 1e6, 3),
            fmt_curve_bytes(estimate.upper.bytes),
            fmt_duration(estimate.upper.time_us / 1e6, 3)
        );
    }
    let _ = writeln!(
        out,
        " · floor {} · peak busbw {} at {}",
        fmt_duration(estimate.floor_us / 1e6, 3),
        fmt_gbps(estimate.peak.busbw_gbps, 3),
        fmt_curve_bytes(estimate.peak.bytes)
    );
    let peer_copy = gpu_profile
        .measured
        .as_ref()
        .and_then(|measured| measured.g2g_gbps)
        .map_or_else(
            || "not measured".to_owned(),
            |rate| {
                format!(
                    "{} measured (napkin-bench cuMemcpyPeer, one GPU pair; not used here)",
                    fmt_gbps(rate, 3)
                )
            },
        );
    let _ = writeln!(out, "peer copy   {peer_copy}");
    let _ = writeln!(
        out,
        "nvlink      {} per-GPU datasheet aggregate, both directions (not used here)",
        fmt_gbps(gpu_profile.ceiling.nvlink_gbps, 3)
    );
    let captured_date = evidence
        .captured_at
        .get(..10)
        .map_or(evidence.captured_at.as_str(), |captured| captured);
    let _ = writeln!(
        out,
        "evidence    NCCL {} · nccl-tests {} · {} · {} · {}",
        evidence.nccl_version,
        evidence.nccl_tests,
        evidence.modal_request,
        captured_date,
        evidence.raw_dir
    );
    Ok(out)
}

fn fmt_curve_bytes(bytes: u64) -> String {
    #[expect(
        clippy::cast_precision_loss,
        reason = "NCCL curve sizes are exactly representable through 2^53 bytes"
    )]
    let bytes = bytes as f64;
    fmt_binary_bytes(bytes, 3)
}

fn get_profile(id: &str) -> Result<&'static Profile, String> {
    profile::find(id).ok_or_else(|| format!("unknown GPU {id:?}; try `napkin gpus`"))
}

fn cmd_gpus() -> String {
    let mut out = String::new();
    for p in profile::builtin() {
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
    out
}

fn cmd_table(gpu: &str, exact: bool) -> Result<String, String> {
    let p = get_profile(gpu)?;
    napkin::table::numbers(p, exact)
        .ok_or_else(|| format!("{gpu} has no measured profile; run napkin-bench on it"))
}

fn cmd_compare(exact: bool) -> String {
    napkin::table::compare(profile::builtin(), exact)
}

fn opt_bytes(value: Option<&str>) -> Result<f64, String> {
    value.map_or(Ok(0.0), |s| {
        units::parse_bytes(s).map_err(|e| e.to_string())
    })
}

/// Without an explicit count, bytes on the command line are one copy; an
/// explicit count is passed through so `estimate` can reject a mismatch.
fn transfer(bytes: Option<&str>, copies: Option<u64>) -> Result<Transfer, String> {
    let bytes = opt_bytes(bytes)?;
    Ok(Transfer {
        bytes,
        copies: copies.unwrap_or(u64::from(bytes > 0.0)),
    })
}

fn bound(lower_bound: bool) -> &'static str {
    if lower_bound { "≥ " } else { "" }
}

fn transfer_line(out: &mut String, name: &str, t: &TransferTime) {
    let noun = if t.copies == 1 { "copy" } else { "copies" };
    let latency = t.latency_us.map_or_else(String::new, |us| {
        format!("{} × {} + ", t.waits, fmt_duration(us / 1e6, 2))
    });
    let _ = write!(
        out,
        "{name:<12}{}{}   {} {} {noun}: {latency}{} / {}",
        bound(t.lower_bound.is_some()),
        fmt_duration(t.seconds, 2),
        t.copies,
        t.mode,
        fmt_bytes(t.bytes, 2),
        fmt_gbps(t.gbps, 2),
    );
    if let Some(reason) = t.lower_bound {
        let _ = write!(out, "; lower bound: {reason}");
    }
    out.push('\n');
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
    h2d: Transfer,
    d2h: Transfer,
    host: Option<&str>,
    g2g: Transfer,
    copy_mode: CopyMode,
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
        h2d,
        d2h,
        g2g,
        copy_mode,
        launches,
    };
    let mode = if overlap {
        Mode::Overlapped
    } else {
        Mode::Serial
    };
    let e = estimate::estimate(p, &workload, mode, ceiling).map_err(|e| e.to_string())?;

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
    for (name, stage) in [("h2d", e.h2d), ("d2h", e.d2h), ("g2g", e.g2g)] {
        if let Some(t) = stage {
            transfer_line(&mut out, name, &t);
        }
    }
    let mut line = |name: &str, seconds: f64, show: bool| {
        if show {
            let _ = writeln!(out, "{name:<12}{}", fmt_duration(seconds, 2));
        }
    };
    line("launch", e.launch_s, e.launch_s > 0.0);
    let _ = writeln!(
        out,
        "{:<12}{}{}   {}",
        "total",
        bound(e.is_lower_bound()),
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
    per_gpu: Option<&str>,
    headroom: f64,
) -> Result<String, String> {
    let p = get_profile(gpu)?;
    let f = fit::fit(
        p,
        units::parse_count(params).map_err(|e| e.to_string())?,
        bits,
        opt_bytes(extra)?,
        opt_bytes(per_gpu)?,
        headroom,
    )
    .map_err(|e| e.to_string())?;

    let gb = |b: f64| format!("{} GB", units::fmt_sig(b / 1e9, 3));
    let mut out = format!("{} · {params} params · {bits}-bit\n", p.name);
    let _ = writeln!(out, "weights     {}", gb(f.weight_bytes));
    if f.extra_bytes > 0.0 {
        let _ = writeln!(out, "extra       {}", gb(f.extra_bytes));
    }
    if f.per_gpu_bytes > 0.0 {
        let _ = writeln!(
            out,
            "per GPU     {} (replicated on every GPU)",
            gb(f.per_gpu_bytes)
        );
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
            "fits        no — needs at least {} GPUs (weights + extra sharded, per-GPU replicated)",
            f.min_gpus
        );
    }
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "keep each cost assumption visible as an independent CLI flag"
)]
fn cmd_cost(
    gpus: u32,
    time: &str,
    price: Option<&str>,
    offer: Option<&str>,
    currency: &str,
    utilization: Option<f64>,
    startup: Option<&str>,
    idle: Option<&str>,
    granularity: Option<&str>,
    work: Option<&str>,
    throughput: Option<&str>,
    unit: &str,
) -> Result<String, String> {
    if currency.len() != 3 || !currency.bytes().all(|byte| byte.is_ascii_uppercase()) {
        return Err("currency must be exactly three uppercase ASCII letters".to_owned());
    }
    let (price_per_gpu_hour, first_line, price_line) = if let Some(record_id) = offer {
        let record = price::find(record_id).ok_or_else(|| {
            format!("no price record {record_id:?}; run `napkin prices` or pass --price")
        })?;
        let today =
            Date::today().ok_or_else(|| "system clock is before the Unix epoch".to_owned())?;
        let age = record.age_days(today);
        let display_price = record.listed_price();
        let terms = record.terms_label();
        let first_line = format!(
            "{gpus} {} · {} {} ({}): {display_price} = {}/GPU-hour",
            if gpus == 1 { "GPU" } else { "GPUs" },
            record.provider,
            record.sku,
            record.id,
            fmt_money(record.per_gpu_hour(), record.currency)
        );
        let price_line = if record.is_stale(today) {
            format!(
                "price       STALE: dated {}, {age} days old (stale after {STALE_AFTER_DAYS} days); re-check {} or pass --price\n",
                record.date, record.source
            )
        } else {
            format!(
                "price       {terms} · dated {}, {age} days old · {}\n",
                record.date, record.source
            )
        };
        (record.per_gpu_hour(), first_line, Some(price_line))
    } else {
        let price = price.ok_or_else(|| "pass exactly one of --price or --offer".to_owned())?;
        let amount = units::parse_number(price).map_err(|error| error.to_string())?;
        let first_line = format!(
            "{gpus} {} · {}/GPU-hour (--price)",
            if gpus == 1 { "GPU" } else { "GPUs" },
            fmt_money(amount, currency)
        );
        (amount, first_line, None)
    };
    let busy_s = units::parse_duration(time).map_err(|error| error.to_string())?;
    let utilization_value = utilization.unwrap_or(1.0);
    let startup_s = startup
        .map(units::parse_duration)
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or(0.0);
    let idle_s = idle
        .map(units::parse_duration)
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or(0.0);
    let granularity_s = granularity
        .map(units::parse_duration)
        .transpose()
        .map_err(|error| error.to_string())?;
    let work = match (work, throughput) {
        (Some(value), _) => Some(Work::Count(
            units::parse_number(value).map_err(|error| error.to_string())?,
        )),
        (_, Some(value)) => Some(Work::Rate(
            units::parse_number(value).map_err(|error| error.to_string())?,
        )),
        (None, None) => None,
    };
    let job = Job {
        gpus,
        busy_s,
        utilization: utilization_value,
        startup_s,
        idle_s,
        granularity_s,
        work,
    };
    let result = cost::cost(&job, price_per_gpu_hour).map_err(|error| error.to_string())?;
    let mut out = format!("{first_line}\n");
    if let Some(line) = price_line {
        out.push_str(&line);
    }
    CostDisplay {
        utilization,
        startup,
        idle,
        currency,
        unit,
    }
    .append(&mut out, &job, &result);
    Ok(out)
}

struct CostDisplay<'a> {
    utilization: Option<f64>,
    startup: Option<&'a str>,
    idle: Option<&'a str>,
    currency: &'a str,
    unit: &'a str,
}

impl CostDisplay<'_> {
    fn append(&self, out: &mut String, job: &Job, result: &cost::Cost) {
        let _ = writeln!(
            out,
            "{:<12}{} per GPU (--time)",
            "busy",
            fmt_duration(job.busy_s, 3)
        );
        let utilization_source = if self.utilization.is_some() {
            "--utilization"
        } else {
            "default; pass --utilization"
        };
        let _ = writeln!(
            out,
            "{:<12}{}% ({utilization_source})",
            "utilization",
            fmt_sig(job.utilization * 100.0, 2)
        );
        let _ = writeln!(
            out,
            "{:<12}{} = busy / utilization",
            "running",
            fmt_duration(result.running_s, 3)
        );
        let startup_source = if self.startup.is_some() {
            "--startup"
        } else {
            "default; pass --startup"
        };
        let _ = writeln!(
            out,
            "{:<12}{} ({startup_source})",
            "startup",
            fmt_duration(job.startup_s, 3)
        );
        let idle_source = if self.idle.is_some() {
            "--idle"
        } else {
            "default; pass --idle"
        };
        let _ = writeln!(
            out,
            "{:<12}{} ({idle_source})",
            "idle",
            fmt_duration(job.idle_s, 3)
        );
        let _ = writeln!(
            out,
            "{:<12}{} = startup + running + idle",
            "wall",
            fmt_duration(result.wall_s, 3)
        );
        let billing_note = job.granularity_s.map_or_else(
            || "exact; pass --granularity".to_owned(),
            |seconds| format!("rounded up to {}", fmt_duration(seconds, 3)),
        );
        let _ = writeln!(
            out,
            "{:<12}{} per GPU ({billing_note})",
            "billed",
            fmt_duration(result.billed_s, 3)
        );
        let gpu_label = if job.gpus == 1 { "GPU" } else { "GPUs" };
        let _ = writeln!(
            out,
            "{:<12}{} = {} {gpu_label} × billed",
            "GPU-hours",
            fmt_sig(result.gpu_hours, 3),
            job.gpus
        );
        let _ = writeln!(
            out,
            "{:<12}{} per job",
            "cost",
            fmt_money(result.cost, self.currency)
        );
        let _ = writeln!(
            out,
            "{:<12}{}% of billed GPU time is busy",
            "useful",
            fmt_sig(result.useful * 100.0, 2)
        );
        self.append_work(out, result);
    }

    fn append_work(&self, out: &mut String, result: &cost::Cost) {
        if let Some(total) = result.work {
            match total.source {
                Work::Count(_) => {
                    let _ = writeln!(
                        out,
                        "work        {} {} (--work)",
                        fmt_count(total.units, 3),
                        self.unit
                    );
                }
                Work::Rate(rate) => {
                    let _ = writeln!(
                        out,
                        "work        {} {} = {} {}/s × busy",
                        fmt_count(total.units, 3),
                        self.unit,
                        fmt_count(rate, 3),
                        self.unit
                    );
                }
            }
            write_per_dollar(out, total.per_dollar, self.unit, self.currency);
        } else {
            out.push_str(
                "work        none (pass --work or --throughput: a workload rate, never peak FLOPs)\n",
            );
        }
    }
}

fn write_per_dollar(out: &mut String, value: f64, unit: &str, currency: &str) {
    let currency = if currency == "USD" { "$" } else { currency };
    let _ = writeln!(out, "per dollar  {} {unit}/{currency}", fmt_count(value, 3));
}

fn cmd_prices() -> Result<String, String> {
    let today = Date::today().ok_or_else(|| "system clock is before the Unix epoch".to_owned())?;
    let mut out = napkin::table::prices(price::builtin());
    out.push('\n');
    let _ = writeln!(
        out,
        "today {today}; records older than {STALE_AFTER_DAYS} days are stale"
    );
    for record in price::builtin() {
        let age = record.age_days(today);
        if record.is_stale(today) {
            let _ = writeln!(out, "{}  STALE, {age} days old", record.id);
        } else {
            let _ = writeln!(out, "{}  {age} days old", record.id);
        }
    }
    Ok(out)
}

fn byte_arg(bytes: f64) -> String {
    fmt_bytes(bytes, 3).replace(' ', "")
}

fn flops_arg(flops: f64) -> String {
    fmt_si(flops, 3)
}

fn fmt_tokens(value: f64) -> String {
    format!("{value:.0}")
}

fn cmd_llm(args: &LlmArgs) -> Result<String, String> {
    let p = get_profile(&args.gpu)?;
    let model = Model {
        layers: args.layers,
        hidden: args.hidden,
        heads: args.heads,
        kv_heads: args.kv_heads,
        head_dim: args.head_dim,
        params: units::parse_count(&args.params).map_err(|e| e.to_string())?,
        weight_bits: args.weight_bits,
        kv_bits: args.kv_bits,
    };
    let workload = LlmWorkload {
        batch: args.batch,
        prompt_tokens: args.prompt,
        generated_tokens: args.generated,
        activation: args.dtype,
    };
    let recipe = llm::recipe(&model, &workload).map_err(|e| e.to_string())?;
    let per_gpu_bytes = opt_bytes(args.per_gpu.as_deref())?;
    let capacity = fit::fit(
        p,
        model.params,
        model.weight_bits,
        recipe.kv_cache_bytes,
        per_gpu_bytes,
        args.headroom.unwrap_or(0.1),
    )
    .map_err(|e| e.to_string())?;
    let group_size = args.heads.get() / args.kv_heads.get();
    let batch = args.batch.get();
    let prompt = args.prompt.get();
    let generated = args.generated.get();
    let layers = args.layers.get();
    let hidden = args.hidden.get();
    let heads = args.heads.get();
    let kv_heads = args.kv_heads.get();
    let head_dim = args.head_dim.get();
    let weight_bits = units::fmt_sig(model.weight_bits, 3);
    let kv_bits = units::fmt_sig(model.kv_bits, 3);
    let mut out = format!(
        "{} · {} params · {layers} layers · {heads} heads ({kv_heads} KV heads, GQA {group_size}:1) × {head_dim} head dim · hidden {hidden}\n",
        p.name, args.params
    );
    let _ = writeln!(
        out,
        "weights {weight_bits}-bit · KV {kv_bits}-bit · activations {} · batch {batch} × ({prompt} prompt + {generated} generated)\n",
        args.dtype
    );
    render_llm_memory(&mut out, args, &model, &recipe, per_gpu_bytes);
    render_llm_prefill(&mut out, args, &recipe);
    render_llm_decode(&mut out, args, &recipe);

    if capacity.fits_one {
        let _ = writeln!(
            out,
            "\nfits on one {}: {} of {} usable",
            p.name,
            fmt_bytes(capacity.working_bytes, 3),
            fmt_bytes(capacity.usable_bytes_per_gpu, 3)
        );
    } else {
        let _ = writeln!(
            out,
            "\nhypothetical: the {} working set exceeds {} usable on one {},\nso both estimates assume one GPU's bandwidth and compute. Sharding needs at least {} GPUs;\nit divides weights and KV, not per-GPU memory, and a tensor-parallel speedup needs\na communication term (napkin collective) before you claim one.",
            fmt_bytes(capacity.working_bytes, 3),
            fmt_bytes(capacity.usable_bytes_per_gpu, 3),
            p.name,
            capacity.min_gpus
        );
    }
    Ok(out)
}

fn render_llm_memory(
    out: &mut String,
    args: &LlmArgs,
    model: &Model,
    recipe: &llm::Recipe,
    per_gpu_bytes: f64,
) {
    let layers = args.layers.get();
    let batch = args.batch.get();
    let prompt = args.prompt.get();
    let generated = args.generated.get();
    let kv_heads = args.kv_heads.get();
    let head_dim = args.head_dim.get();
    let weight_bits = units::fmt_sig(model.weight_bits, 3);
    out.push_str("memory: what must stay resident\n");
    let _ = writeln!(
        out,
        "{:<13}{} × {} / 8 = {}",
        "weights",
        args.params,
        weight_bits,
        fmt_bytes(recipe.weight_bytes, 3)
    );
    let _ = writeln!(
        out,
        "{:<13}2 × {layers} layers × {kv_heads} KV heads × {head_dim} × {} B = {}",
        "KV/token",
        units::fmt_sig(model.kv_bits / 8.0, 3),
        fmt_bytes(recipe.kv_bytes_per_token, 3)
    );
    let _ = writeln!(
        out,
        "{:<13}{batch} × ({prompt} + {generated}) = {}",
        "live tokens",
        fmt_tokens(recipe.live_tokens)
    );
    let _ = writeln!(
        out,
        "{:<13}{} × {} = {}",
        "KV cache",
        fmt_bytes(recipe.kv_bytes_per_token, 3),
        fmt_tokens(recipe.live_tokens),
        fmt_bytes(recipe.kv_cache_bytes, 3)
    );
    let _ = write!(
        out,
        "$ napkin fit --gpu {} --params {} --bits {} --extra {}",
        args.gpu,
        args.params,
        weight_bits,
        byte_arg(recipe.kv_cache_bytes)
    );
    if args.per_gpu.is_some() {
        let _ = write!(out, " --per-gpu {}", byte_arg(per_gpu_bytes));
    }
    if let Some(headroom) = args.headroom {
        let _ = write!(out, " --headroom {}", units::fmt_sig(headroom, 3));
    }
    out.push_str("\n\n");
}

fn render_llm_prefill(out: &mut String, args: &LlmArgs, recipe: &llm::Recipe) {
    let batch = args.batch.get();
    let prompt = args.prompt.get();
    let layers = args.layers.get();
    let hidden = args.hidden.get();
    let heads = args.heads.get();
    let head_dim = args.head_dim.get();
    let prefill_tokens = recipe.prefill.tokens;
    let _ = writeln!(
        out,
        "prefill: one forward over {batch} × {prompt} = {} prompt tokens",
        fmt_tokens(prefill_tokens)
    );
    let _ = writeln!(
        out,
        "{:<13}2 × {} × {} = {} FLOP",
        "linear",
        args.params,
        fmt_tokens(prefill_tokens),
        flops_arg(recipe.prefill.linear_flops)
    );
    let _ = writeln!(
        out,
        "{:<13}2 × {layers} × {heads} × {head_dim} × {batch} × {prompt}² (causal) = {} FLOP",
        "attention",
        flops_arg(recipe.prefill.attention_flops)
    );
    let _ = writeln!(
        out,
        "{:<13}read once = {}",
        "weights",
        fmt_bytes(recipe.prefill.weight_bytes, 3)
    );
    let _ = writeln!(
        out,
        "{:<13}{} × {} = {}",
        "KV write",
        fmt_bytes(recipe.kv_bytes_per_token, 3),
        fmt_tokens(prefill_tokens),
        fmt_bytes(recipe.prefill.kv_bytes, 3)
    );
    let _ = writeln!(
        out,
        "{:<13}2 × {layers} × {} × {hidden} × {} B (floor) = {}",
        "activations",
        fmt_tokens(prefill_tokens),
        units::fmt_sig(llm::element_bytes(args.dtype), 3),
        fmt_bytes(recipe.prefill.activation_bytes, 3)
    );
    let _ = writeln!(
        out,
        "$ napkin estimate --gpu {} --dtype {} --flops {} --bytes {}",
        args.gpu,
        args.dtype,
        flops_arg(recipe.prefill.flops()),
        byte_arg(recipe.prefill.hbm_bytes())
    );
    out.push('\n');
}

fn render_llm_decode(out: &mut String, args: &LlmArgs, recipe: &llm::Recipe) {
    let batch = args.batch.get();
    let layers = args.layers.get();
    let hidden = args.hidden.get();
    let heads = args.heads.get();
    let head_dim = args.head_dim.get();
    let decode_bytes = recipe.decode.hbm_bytes();
    let context = fmt_tokens(recipe.decode_context);
    let _ = writeln!(
        out,
        "decode step: {batch} new tokens at context {context} (the last, largest step)"
    );
    let _ = writeln!(
        out,
        "{:<13}2 × {} × {batch} = {} FLOP",
        "linear",
        args.params,
        flops_arg(recipe.decode.linear_flops)
    );
    let _ = writeln!(
        out,
        "{:<13}4 × {layers} × {heads} × {head_dim} × {batch} × {context} = {} FLOP",
        "attention",
        flops_arg(recipe.decode.attention_flops)
    );
    let _ = writeln!(
        out,
        "{:<13}read once, shared by {batch} requests = {}",
        "weights",
        fmt_bytes(recipe.decode.weight_bytes, 3)
    );
    let _ = writeln!(
        out,
        "{:<13}{} × {batch} × {context} = {}",
        "KV read",
        fmt_bytes(recipe.kv_bytes_per_token, 3),
        fmt_bytes(recipe.decode.kv_bytes, 3)
    );
    let _ = writeln!(
        out,
        "{:<13}2 × {layers} × {batch} × {hidden} × {} B (floor) = {}",
        "activations",
        units::fmt_sig(llm::element_bytes(args.dtype), 3),
        fmt_bytes(recipe.decode.activation_bytes, 3)
    );
    let _ = writeln!(
        out,
        "{:<13}{} / {batch} = {}",
        "per token",
        fmt_bytes(decode_bytes, 3),
        fmt_bytes(decode_bytes / recipe.decode.tokens, 3)
    );
    let _ = writeln!(
        out,
        "$ napkin estimate --gpu {} --dtype {} --flops {} --bytes {}",
        args.gpu,
        args.dtype,
        flops_arg(recipe.decode.flops()),
        byte_arg(decode_bytes)
    );
}

const MARKER_PREFIX: &str = "<!-- napkin ";
const MARKER_END: &str = "<!-- /napkin -->";

fn render_marker(command: &str) -> Result<String, String> {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    match tokens.as_slice() {
        ["table", "--gpu", gpu] => cmd_table(gpu, false),
        ["compare"] => Ok(cmd_compare(false)),
        ["prices"] => Ok(napkin::table::prices(price::builtin())),
        ["example", args @ ..] => {
            let cli = Cli::try_parse_from(std::iter::once("napkin").chain(args.iter().copied()))
                .map_err(|error| error.to_string())?;
            if matches!(&cli.command, Command::Readme { .. }) {
                return Err("example markers cannot run readme".to_owned());
            }
            let command_output = dispatch(&cli.command)?;
            let mut rendered = format!("```text\n$ napkin {}\n{command_output}", args.join(" "));
            if !rendered.ends_with('\n') {
                rendered.push('\n');
            }
            rendered.push_str("```\n");
            Ok(rendered)
        }
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

fn cmd_readme(paths: &[PathBuf], check: bool) -> Result<String, String> {
    let mut out = String::new();
    let mut stale = Vec::new();
    for path in paths {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let updated = rewrite_markers(&content)?;
        if check {
            if updated == content {
                let _ = writeln!(out, "{}: napkin blocks are current", path.display());
            } else {
                stale.push(format!(
                    "{}: napkin blocks are stale; run `napkin readme --write`",
                    path.display()
                ));
            }
        } else if updated != content {
            std::fs::write(path, &updated)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            let _ = writeln!(out, "{}: napkin blocks updated", path.display());
        } else {
            let _ = writeln!(out, "{}: already current", path.display());
        }
    }
    if stale.is_empty() {
        Ok(out)
    } else {
        Err(stale.join("\n"))
    }
}

fn cmd_validate(
    write: Option<&std::path::Path>,
    check: Option<&std::path::Path>,
) -> Result<String, String> {
    let report = render_validation_report()?;
    if let Some(path) = check {
        let content = std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if content == report {
            return Ok(format!(
                "{}: validation report is current\n",
                path.display()
            ));
        }
        return Err(format!(
            "{}: validation report is stale; run `napkin validate --write {}`",
            path.display(),
            path.display()
        ));
    }
    if let Some(path) = write {
        let content = std::fs::read_to_string(path).ok();
        if content.as_deref() != Some(report.as_str()) {
            std::fs::write(path, &report)
                .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
            return Ok(format!("{}: validation report updated\n", path.display()));
        }
        return Ok(format!("{}: already current\n", path.display()));
    }
    Ok(report)
}

fn render_validation_report() -> Result<String, String> {
    let cases = validation::cases()?;
    let mut report = String::from(
        "# Validation: napkin estimates against real runs\n\n\
<!-- Generated by `cargo run -p napkin -- validate --write VALIDATION.md` from `validation/<case>/`; edit those files instead. -->\n\n\
Each case is one real job on Modal. The error ratio is observed time / predicted time, so a ratio above 1 means the real run was slower than napkin. Each tolerance was committed in `case.json` before the observation run; profile constants were not tuned to fit these cases. The runner is `modal run modal/validate.py`, an opt-in tool that is not part of CI.\n\n\
| case | predicted | dominant term | observed | observed / predicted | tolerance | verdict |\n\
| --- | ---: | --- | ---: | ---: | ---: | --- |\n",
    );
    for case in &cases {
        let _ = writeln!(
            report,
            "| [{}](#{}) | {} | {} | {} | {:.2} | {:.2}–{:.2} | {} |",
            case.id,
            case.id,
            fmt_duration(case.estimate.total_s, 2),
            case.estimate.bottleneck,
            fmt_duration(case.observed_s, 2),
            case.ratio,
            case.tolerance_low,
            case.tolerance_high,
            case.verdict
        );
    }
    for case in &cases {
        render_validation_case(&mut report, case)?;
    }
    Ok(report)
}

fn render_validation_case(report: &mut String, case: &Case) -> Result<(), String> {
    let (command, estimate_output) = render_estimate_example(case)?;
    let _ = write!(
        report,
        "\n## {}: {}\n\n### Job\n\n{}\n\n### Match\n\n",
        case.id, case.title, case.job
    );
    for item in &case.match_items {
        let _ = writeln!(report, "- {item}");
    }
    report.push_str("\n### Estimate inputs\n\n");
    for item in &case.derivation {
        let _ = writeln!(report, "- {item}");
    }
    report.push_str("\n```text\n$ ");
    report.push_str(&command);
    report.push('\n');
    report.push_str(&estimate_output);
    report.push_str("```\n\n### Tolerance, declared before the run\n\n");
    let _ = writeln!(
        report,
        "{:.2}–{:.2}: {}\n\n### Observation\n\n- Tool: {}\n- Metric: {}\n- Observed: {}",
        case.tolerance_low,
        case.tolerance_high,
        case.tolerance_why,
        case.observation_tool,
        case.observation_metric,
        fmt_duration(case.observed_s, 2)
    );
    if let (Some(samples), Some(number_per_run)) = (case.samples, case.number_per_run) {
        let _ = writeln!(
            report,
            "- Timer samples: {samples} samples × {number_per_run} runs per sample."
        );
    }
    for direction in &case.directions {
        let _ = writeln!(
            report,
            "- {}: {} GB/s → {}.",
            direction.name,
            direction.bandwidth_text,
            fmt_duration(direction.seconds, 4)
        );
    }
    if let Some(kernels) = case.run.profiled_cuda_kernels {
        let _ = writeln!(report, "- Profiler counted {kernels} CUDA kernels.");
    }
    report.push_str("- Evidence: ");
    for (index, file) in case.source_files.iter().enumerate() {
        if index > 0 {
            report.push_str(" · ");
        }
        let _ = write!(report, "[{file}](validation/{}/{file})", case.id);
    }
    report.push_str("\n\n### Run\n\n| field | captured value |\n| --- | --- |\n");
    render_run_table(report, case);
    let _ = writeln!(
        report,
        "\n### Result\n\n{:.2} observed / predicted — **{}**.\n\n### Excluded from the estimate\n",
        case.ratio, case.verdict
    );
    for item in &case.excludes {
        let _ = writeln!(report, "- {item}");
    }
    report.push_str("\n### Finding\n\n");
    report.push_str(&case.finding);
    if !case.finding.ends_with('\n') {
        report.push('\n');
    }
    Ok(())
}

fn render_estimate_example(case: &Case) -> Result<(String, String), String> {
    let workload = &case.workload;
    let flops = format!("{:.0}", workload.flops);
    let bytes = format!("{:.0}", workload.hbm_bytes);
    let mut command = vec![
        "napkin".to_owned(),
        "estimate".to_owned(),
        "--gpu".to_owned(),
        case.gpu.clone(),
        "--dtype".to_owned(),
        workload.dtype.to_string(),
        "--flops".to_owned(),
        flops.clone(),
        "--bytes".to_owned(),
        bytes.clone(),
    ];
    let add_transfer = |name: &str, transfer: Transfer, command: &mut Vec<String>| {
        if transfer.bytes > 0.0 {
            command.push(format!("--{name}"));
            command.push(format!("{:.0}", transfer.bytes));
            if transfer.copies != 1 {
                command.push(format!("--{name}-copies"));
                command.push(transfer.copies.to_string());
            }
        }
    };
    add_transfer("h2d", workload.h2d, &mut command);
    add_transfer("d2h", workload.d2h, &mut command);
    if workload.copy_mode == CopyMode::Async {
        command.extend(["--copy-mode".to_owned(), "async".to_owned()]);
    }
    if workload.launches > 0 {
        command.extend(["--launches".to_owned(), workload.launches.to_string()]);
    }
    if case.mode == Mode::Overlapped {
        command.push("--overlap".to_owned());
    }
    let command_text = command.join(" ");
    let output = cmd_estimate(
        &case.gpu,
        workload.dtype,
        &flops,
        &bytes,
        workload.h2d,
        workload.d2h,
        None,
        workload.g2g,
        workload.copy_mode,
        workload.launches,
        case.mode == Mode::Overlapped,
        false,
    )?;
    Ok((command_text, output))
}

fn render_run_table(report: &mut String, case: &Case) {
    let run = &case.run;
    add_run_row(report, "captured_at", run.captured_at.as_deref());
    add_run_row(
        report,
        "modal_request",
        run.modal_request.as_deref().or(Some(&case.modal_request)),
    );
    add_run_row(
        report,
        "GPU",
        run.gpus.first().and_then(|gpu| gpu.name.as_deref()),
    );
    add_run_row(report, "driver", run.driver.as_deref());
    add_run_row(report, "cuda_driver", run.cuda_driver.as_deref());
    add_run_row(report, "torch", run.torch.as_deref());
    add_run_row(report, "torch_cuda", run.torch_cuda.as_deref());
    add_run_row(report, "cublas", run.cublas.as_deref());
    add_run_row(
        report,
        "NVBandwidth",
        match (run.tag.as_deref(), run.commit.as_deref()) {
            (Some(tag), Some(commit)) => Some(format!("{tag} · {commit}")),
            (Some(tag), None) => Some(tag.to_owned()),
            (None, Some(commit)) => Some(commit.to_owned()),
            (None, None) => None,
        }
        .as_deref(),
    );
    if let Some(monitor) = &run.monitor
        && let (Some(generation), Some(width)) = (monitor.pcie_gen_peak, monitor.pcie_width_peak)
    {
        let _ = writeln!(report, "| PCIe peak | Gen {generation:.0} × {width:.0} |");
    }
    if let Some(monitor) = &run.monitor {
        add_run_row(
            report,
            "SM clock",
            format_range(monitor.sm_clock_mhz.as_ref(), "MHz", 0).as_deref(),
        );
        add_run_row(
            report,
            "Power",
            format_range(monitor.power_w.as_ref(), "W", 2).as_deref(),
        );
    }
    if let Some(seconds) = run.case_seconds {
        let _ = writeln!(report, "| case_seconds (GPU time) | {seconds:.1} s |");
    }
    if let Some(kernels) = run.profiled_cuda_kernels {
        let _ = writeln!(report, "| profiled_cuda_kernels | {kernels} |");
    }
}

fn add_run_row(report: &mut String, label: &str, value: Option<&str>) {
    if let Some(value) = value {
        let _ = writeln!(report, "| {label} | {value} |");
    }
}

fn format_range(range: Option<&RangeInfo>, unit: &str, precision: usize) -> Option<String> {
    let range = range?;
    let (min, max) = (range.min?, range.max?);
    Some(format!("{min:.precision$}–{max:.precision$} {unit}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collective_example_marker_renders_user_facing_command_output() -> Result<(), String> {
        let command = "example collective --gpu h100 --count 2 --topology nvlink --op all-reduce --bytes 16KiB";
        let output = render_marker(command)?;
        assert!(output.starts_with(
            "```text\n$ napkin collective --gpu h100 --count 2 --topology nvlink --op all-reduce --bytes 16KiB\n"
        ));
        assert!(output.contains("H100 SXM 80GB × 2 · nvlink · AllReduce · bf16 sum · 16 KiB"));
        assert!(output.ends_with("\n```\n"));
        Ok(())
    }
}
