//! `wds info` — summarise the contents of a set of shards.

use std::collections::BTreeMap;
use std::io::Write;

use clap::Args as ClapArgs;
use webdataset::Result;

use crate::{open, sample_size};

/// Arguments for `wds info`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Shard URLs or brace patterns.
    #[arg(required = true)]
    pub shards: Vec<String>,

    /// Stop after this many samples.
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,

    /// Print the summary as JSON.
    #[arg(long)]
    pub json: bool,
}

/// What `wds info` counts.
#[derive(Debug, Default)]
struct Summary {
    samples: usize,
    bytes: usize,
    shards: BTreeMap<String, usize>,
    fields: BTreeMap<String, FieldStats>,
}

/// Per-field counts.
#[derive(Debug, Default)]
struct FieldStats {
    count: usize,
    bytes: usize,
}

/// Summarise the samples in `args.shards`.
pub fn run(args: Args) -> Result<()> {
    let dataset = open(&args.shards, args.limit)?;
    let mut summary = Summary::default();

    for sample in dataset.iter() {
        let sample = sample?;
        summary.samples += 1;
        summary.bytes += sample_size(&sample);
        if let Some(url) = sample.url() {
            *summary.shards.entry(url.to_string()).or_default() += 1;
        }
        for (name, value) in sample.iter().filter(|(name, _)| !name.starts_with("__")) {
            let stats = summary.fields.entry(name.clone()).or_default();
            stats.count += 1;
            stats.bytes += value.as_bytes().map(|b| b.len()).unwrap_or(0);
        }
    }

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    if args.json {
        let fields: BTreeMap<&String, serde_json::Value> = summary
            .fields
            .iter()
            .map(|(name, stats)| (name, serde_json::json!({ "count": stats.count, "bytes": stats.bytes })))
            .collect();
        let report = serde_json::json!({
            "samples": summary.samples,
            "bytes": summary.bytes,
            "shards": summary.shards,
            "fields": fields,
        });
        writeln!(out, "{}", serde_json::to_string_pretty(&report)?)?;
        out.flush()?;
        return Ok(());
    }

    writeln!(out, "samples   {}", summary.samples)?;
    writeln!(out, "bytes     {} ({})", summary.bytes, human_bytes(summary.bytes))?;
    writeln!(out, "shards    {}", summary.shards.len())?;
    if let Some(mean) = summary.bytes.checked_div(summary.samples) {
        writeln!(out, "mean size {}", human_bytes(mean))?;
    }

    if !summary.fields.is_empty() {
        writeln!(out, "\nfield                 count      bytes  present")?;
        for (name, stats) in &summary.fields {
            let present = 100.0 * stats.count as f64 / summary.samples.max(1) as f64;
            writeln!(out, "{name:<20} {:>6} {:>10}  {present:>5.1}%", stats.count, human_bytes(stats.bytes))?;
        }
    }

    if summary.shards.len() > 1 {
        writeln!(out, "\nshard                                            samples")?;
        for (url, count) in &summary.shards {
            let shown = shorten(url, 48);
            writeln!(out, "{shown:<48} {count:>7}")?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Render a byte count in the largest unit that keeps it readable.
fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 { format!("{bytes} B") } else { format!("{value:.1} {}", UNITS[unit]) }
}

/// Trim a long URL from the left, keeping the distinguishing tail.
fn shorten(text: &str, width: usize) -> String {
    if text.len() <= width {
        return text.to_string();
    }
    let mut at = text.len() - (width - 3);
    while at < text.len() && !text.is_char_boundary(at) {
        at += 1;
    }
    format!("...{}", &text[at..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_byte_counts() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2_500), "2.5 kB");
        assert_eq!(human_bytes(3_000_000_000), "3.0 GB");
    }

    #[test]
    fn shortens_from_the_left() {
        assert_eq!(shorten("short", 10), "short");
        let shortened = shorten("abcdefghijklmnop", 10);
        assert_eq!(shortened, "...jklmnop");
        assert_eq!(shortened.len(), 10, "the result should fit the requested width");
    }
}
