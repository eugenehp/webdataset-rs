//! `wds ls` — list the samples in a set of shards.

use std::io::Write;

use clap::Args as ClapArgs;
use webdataset::Result;

use crate::{describe, open, sample_size};

/// Arguments for `wds ls`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Shard URLs or brace patterns, e.g. `data-{000..009}.tar`.
    #[arg(required = true)]
    pub shards: Vec<String>,

    /// Stop after this many samples.
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,

    /// Show each field and its size, not just the key.
    #[arg(long, short)]
    pub long: bool,

    /// Print one JSON object per sample.
    #[arg(long)]
    pub json: bool,
}

/// List the samples in `args.shards`.
pub fn run(args: Args) -> Result<()> {
    let dataset = open(&args.shards, args.limit)?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    for sample in dataset.iter() {
        let sample = sample?;
        let key = sample.key().unwrap_or("<no key>");

        if args.json {
            let fields: Vec<serde_json::Value> = sample
                .iter()
                .filter(|(name, _)| !name.starts_with("__"))
                .map(|(name, value)| serde_json::json!({ "name": name, "value": describe(value) }))
                .collect();
            let line = serde_json::json!({
                "key": key,
                "url": sample.url(),
                "size": sample_size(&sample),
                "fields": fields,
            });
            writeln!(out, "{line}")?;
            continue;
        }

        if !args.long {
            writeln!(out, "{key}\t{}", sample.field_names().join(" "))?;
            continue;
        }

        writeln!(out, "{key}  ({} bytes)", sample_size(&sample))?;
        for (name, value) in sample.iter().filter(|(name, _)| !name.starts_with("__")) {
            writeln!(out, "    {name:<20} {}", describe(value))?;
        }
    }
    out.flush()?;
    Ok(())
}
