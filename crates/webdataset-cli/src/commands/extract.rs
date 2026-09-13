//! `wds extract` — write each sample out as files in a directory.

use std::io::Write;
use std::path::PathBuf;

use clap::Args as ClapArgs;
use webdataset::{Error, Result};

use crate::open;

/// Arguments for `wds extract`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Shard URLs or brace patterns.
    #[arg(required = true)]
    pub shards: Vec<String>,

    /// Directory to write into; created if it does not exist.
    #[arg(long, short, required = true)]
    pub output: PathBuf,

    /// Stop after this many samples.
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,

    /// Extract only these fields.
    #[arg(long = "field", short)]
    pub fields: Vec<String>,
}

/// Extract samples from `args.shards` into `args.output`.
pub fn run(args: Args) -> Result<()> {
    let dataset = open(&args.shards, args.limit)?;
    std::fs::create_dir_all(&args.output)?;

    let mut written = 0usize;
    for sample in dataset.iter() {
        let sample = sample?;
        let key = sample.key().ok_or_else(|| Error::value("sample has no __key__"))?;

        for (name, value) in sample.iter().filter(|(name, _)| !name.starts_with("__")) {
            if !args.fields.is_empty() && !args.fields.contains(name) {
                continue;
            }
            let bytes = value.expect_bytes(name)?;
            let path = args.output.join(format!("{key}.{name}"));
            // A key may contain slashes, which name a subdirectory.
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, bytes).map_err(|e| Error::Io(e).context(format!("writing {}", path.display())))?;
            written += 1;
        }
    }
    crate::note!("wrote {written} files to {}", args.output.display());
    Ok(())
}
