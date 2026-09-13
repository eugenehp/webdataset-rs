//! `wds cat` — copy samples from several shards into one archive.

use std::io::Write;

use clap::Args as ClapArgs;
use webdataset::{Result, TarWriter};

use crate::open;

/// Arguments for `wds cat`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Shard URLs or brace patterns.
    #[arg(required = true)]
    pub shards: Vec<String>,

    /// Where to write; `-` means standard output.
    #[arg(long, short, default_value = "-")]
    pub output: String,

    /// Stop after this many samples.
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,

    /// Keep only samples that have all of these fields.
    #[arg(long = "field", short)]
    pub fields: Vec<String>,
}

/// Concatenate the samples in `args.shards` into one archive.
pub fn run(args: Args) -> Result<()> {
    let dataset = open(&args.shards, args.limit)?;
    let mut writer = TarWriter::create(&args.output)?;
    let required = args.fields.clone();

    let mut written = 0usize;
    for sample in dataset.iter() {
        let sample = sample?;
        if !required.iter().all(|f| sample.contains_key(f)) {
            continue;
        }
        writer.write(&sample)?;
        written += 1;
    }
    writer.close()?;
    crate::note!("wrote {written} samples to {}", args.output);
    Ok(())
}
