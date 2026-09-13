//! `wds create` — build shards from a directory of files.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Args as ClapArgs;
use webdataset::{Error, Result, Sample, ShardWriter, Value};

/// Arguments for `wds create`.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Directory to read files from.
    #[arg(required = true)]
    pub input: PathBuf,

    /// Output pattern, e.g. `train-%06d.tar`.
    #[arg(long, short, required = true)]
    pub output: String,

    /// At most this many samples per shard.
    #[arg(long, default_value_t = 100_000)]
    pub max_count: usize,

    /// Start a new shard once this many payload bytes have been written.
    #[arg(long, default_value_t = 3_000_000_000)]
    pub max_size: u64,

    /// Recurse into subdirectories.
    #[arg(long, short)]
    pub recursive: bool,
}

/// Build shards from the files under `args.input`.
///
/// Files are grouped into samples by basename, the same rule the format uses
/// when reading, so `cat.jpg` and `cat.cls` become one sample.
pub fn run(args: Args) -> Result<()> {
    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    collect(&args.input, &args.input, args.recursive, &mut groups)?;

    if groups.is_empty() {
        return Err(Error::value(format!("no files found under {}", args.input.display())));
    }

    let mut writer = ShardWriter::new(&args.output)?.with_max_count(args.max_count).with_max_size(args.max_size);

    for (key, paths) in groups {
        let mut sample = Sample::with_key(&key);
        for path in paths {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .ok_or_else(|| Error::value(format!("{} has no file name", path.display())))?;
            // The extension is everything after the first dot of the basename.
            let Some((_, extension)) = name.split_once('.') else { continue };
            let data = std::fs::read(&path).map_err(|e| Error::Io(e).context(format!("reading {}", path.display())))?;
            sample.insert(extension.to_ascii_lowercase(), Value::Bytes(data.into()));
        }
        if sample.field_names().is_empty() {
            continue;
        }
        writer.write(&sample)?;
    }

    let total = writer.total();
    let shards = writer.shard_count();
    writer.close()?;
    writeln!(std::io::stderr(), "wrote {total} samples into {shards} shards")?;
    Ok(())
}

/// Walk `directory`, grouping files by their extension-free basename.
fn collect(root: &Path, directory: &Path, recursive: bool, groups: &mut BTreeMap<String, Vec<PathBuf>>) -> Result<()> {
    let entries =
        std::fs::read_dir(directory).map_err(|e| Error::Io(e).context(format!("reading {}", directory.display())))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                collect(root, &path, recursive, groups)?;
            }
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some((base, _)) = name.split_once('.') else { continue };

        // Keep the directory prefix in the key so subdirectories stay distinct.
        let relative = path.parent().and_then(|p| p.strip_prefix(root).ok()).unwrap_or(Path::new(""));
        let key = match relative.as_os_str().is_empty() {
            true => base.to_string(),
            false => format!("{}/{base}", relative.to_string_lossy()),
        };
        groups.entry(key).or_default().push(path);
    }
    Ok(())
}
