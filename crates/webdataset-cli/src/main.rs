//! `wds` — command line tools for WebDataset shards.
//!
//! Shards are ordinary tar files, so `tar` already works on them. What `tar`
//! does not know is that files sharing a basename are one sample, which is what
//! every subcommand here is built around.

use std::io::Write;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use webdataset::{Error, Result, Sample, Value, WebDataset};

mod commands;

/// Inspect and reshape WebDataset shards.
#[derive(Debug, Parser)]
#[command(name = "wds", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Print progress and diagnostics to stderr.
    #[arg(long, short, global = true)]
    verbose: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List the samples in one or more shards.
    Ls(commands::ls::Args),
    /// Summarise the samples in one or more shards.
    Info(commands::info::Args),
    /// Copy samples from several shards into one archive.
    Cat(commands::cat::Args),
    /// Rewrite samples into a new series of shards.
    Split(commands::split::Args),
    /// Write each sample out as files in a directory.
    Extract(commands::extract::Args),
    /// Build shards from a directory of files.
    Create(commands::create::Args),
}

/// Whether `--verbose` was given.
static VERBOSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Print a progress note to stderr, but only under `--verbose`.
macro_rules! note {
    ($($arg:tt)*) => {
        if $crate::VERBOSE.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = writeln!(std::io::stderr(), $($arg)*);
        }
    };
}

pub(crate) use note;

fn main() -> ExitCode {
    let cli = Cli::parse();
    VERBOSE.store(cli.verbose, std::sync::atomic::Ordering::Relaxed);
    let outcome = match cli.command {
        Command::Ls(args) => commands::ls::run(args),
        Command::Info(args) => commands::info::run(args),
        Command::Cat(args) => commands::cat::run(args),
        Command::Split(args) => commands::split::run(args),
        Command::Extract(args) => commands::extract::run(args),
        Command::Create(args) => commands::create::run(args),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "wds: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Open the given shard patterns as a dataset, without shuffling.
fn open(shards: &[String], limit: Option<usize>) -> Result<WebDataset> {
    if shards.is_empty() {
        return Err(Error::value("no shards given"));
    }
    let mut dataset = WebDataset::builder_from(shards).empty_check(false).build()?;
    if let Some(n) = limit {
        dataset = dataset.take(n);
    }
    Ok(dataset)
}

/// Describe a value briefly, for listings.
fn describe(value: &Value) -> String {
    match value {
        Value::Bytes(b) => format!("{} bytes", b.len()),
        Value::Text(s) if s.len() <= 40 => format!("{s:?}"),
        Value::Text(s) => format!("{:?}...", &s[..char_boundary_at_or_before(s, 37)]),
        Value::Tensor(t) => format!("{} {:?}", t.dtype().long_name(), t.shape()),
        other => format!("{other:?}"),
    }
}

/// The number of payload bytes a sample holds.
fn sample_size(sample: &Sample) -> usize {
    sample.iter().filter_map(|(_, v)| v.as_bytes().map(|b| b.len())).sum()
}

/// The largest char boundary of `text` at or before `index`.
///
/// `str::floor_char_boundary` would do, but it is newer than this crate's MSRV.
fn char_boundary_at_or_before(text: &str, index: usize) -> usize {
    let mut at = index.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}
