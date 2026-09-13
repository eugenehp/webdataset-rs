//! `wds split` — rewrite samples into a new series of shards.

use std::io::Write;

use clap::Args as ClapArgs;
use webdataset::filters::SampleIteratorExt;
use webdataset::{Result, ShardWriter};

use crate::open;

/// Arguments for `wds split`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Shard URLs or brace patterns to read.
    #[arg(required = true)]
    pub shards: Vec<String>,

    /// Output pattern, e.g. `train-%06d.tar`.
    #[arg(long, short, required = true)]
    pub output: String,

    /// Start at most this many samples in each output shard.
    #[arg(long, default_value_t = 100_000)]
    pub max_count: usize,

    /// Start a new shard once this many payload bytes have been written.
    #[arg(long, default_value_t = 3_000_000_000)]
    pub max_size: u64,

    /// Shuffle samples through a buffer of this size.
    #[arg(long)]
    pub shuffle: Option<usize>,

    /// Seed the shuffle, so the split can be reproduced.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Stop after this many samples.
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,
}

/// Reshard `args.shards` into the pattern `args.output`.
pub fn run(args: Args) -> Result<()> {
    let dataset = open(&args.shards, args.limit)?;
    let mut writer = ShardWriter::new(&args.output)?.with_max_count(args.max_count).with_max_size(args.max_size);

    let source = dataset.iter();
    let stream: Box<dyn Iterator<Item = Result<webdataset::Sample>>> = match args.shuffle {
        Some(bufsize) => Box::new(source.shuffled(bufsize, args.seed)),
        None => Box::new(source),
    };

    for sample in stream {
        writer.write(&sample?)?;
    }
    let total = writer.total();
    let shards = writer.shard_count();
    writer.close()?;

    writeln!(std::io::stderr(), "wrote {total} samples into {shards} shards")?;
    Ok(())
}
