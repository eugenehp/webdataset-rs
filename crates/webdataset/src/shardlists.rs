//! Producing the stream of shard URLs a pipeline reads from.
//!
//! Every WebDataset pipeline starts by deciding which shards to read and in
//! what order. The choices are:
//!
//! - [`SimpleShardList`] — a fixed list, optionally shuffled once per epoch.
//! - [`ResampledShards`] — shards drawn with replacement, so every worker has
//!   an endless supply and no epoch ever ends ragged.
//! - [`DirectoryShardList`] — shards claimed from a directory as they appear.
//! - [`MultiShardSample`] — several sources mixed by a YAML specification.
//!
//! Splitting across processes and workers is done by separate stages,
//! [`SplitByNode`] and [`SplitByWorker`], so that a pipeline can choose its own
//! sharding policy.
//!
//! ```
//! use webdataset::shardlists::SimpleShardList;
//! use webdataset::pipeline::{DataPipeline, Stage};
//!
//! let shards = SimpleShardList::new(["data-{000..002}.tar"])?;
//! assert_eq!(shards.urls(), ["data-000.tar", "data-001.tar", "data-002.tar"]);
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rand::prelude::*;
use rand::rngs::StdRng;
use webdataset_core::error::{Error, Result};
use webdataset_core::sample::Sample;
use webdataset_core::utils::{expand_urls, make_seed, seed_from_str, worker_info};

use crate::pipeline::{SampleStream, Stage};

/// Build the one-field sample that carries a shard URL down the pipeline.
pub fn shard_sample(url: impl Into<String>) -> Sample {
    let mut sample = Sample::new();
    sample.set_url(url);
    sample
}

/// A fixed list of shard URLs.
///
/// With a seed, the list is shuffled independently each epoch; without one, the
/// order is left alone so that a pipeline is reproducible by default.
#[derive(Debug)]
pub struct SimpleShardList {
    urls: Vec<String>,
    seed: Option<u64>,
    epoch: AtomicUsize,
}

impl SimpleShardList {
    /// Expand `patterns` — brace expressions and `::` lists — into shard URLs.
    pub fn new<S: AsRef<str>>(patterns: impl IntoIterator<Item = S>) -> Result<SimpleShardList> {
        let mut urls = Vec::new();
        for pattern in patterns {
            urls.extend(expand_urls(pattern.as_ref())?);
        }
        if urls.is_empty() {
            return Err(Error::value("shard list is empty"));
        }
        Ok(SimpleShardList { urls, seed: None, epoch: AtomicUsize::new(0) })
    }

    /// Use these URLs exactly as given, without expanding anything.
    pub fn verbatim<S: Into<String>>(urls: impl IntoIterator<Item = S>) -> SimpleShardList {
        SimpleShardList { urls: urls.into_iter().map(Into::into).collect(), seed: None, epoch: AtomicUsize::new(0) }
    }

    /// Shuffle the shard order, differently each epoch, starting from `seed`.
    ///
    /// Put the split stages *after* this one only if there is a single worker:
    /// [`SplitByWorker`] and [`SplitByNode`] divide by position, so they must
    /// see the same order in every worker. The usual arrangement — and the one
    /// [`WebDataset`](crate::WebDataset) builds — leaves this list unshuffled
    /// and puts a [`Shuffle`](crate::stages::Shuffle) stage after the split, so
    /// each worker reorders its own shards.
    pub fn shuffled(mut self, seed: u64) -> SimpleShardList {
        self.seed = Some(seed);
        self
    }

    /// The expanded shard URLs.
    pub fn urls(&self) -> &[String] {
        &self.urls
    }

    /// How many shards are in the list.
    pub fn len(&self) -> usize {
        self.urls.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.urls.is_empty()
    }
}

impl Stage for SimpleShardList {
    fn apply(&self, _input: SampleStream) -> SampleStream {
        let mut urls = self.urls.clone();
        if let Some(seed) = self.seed {
            let epoch = self.epoch.fetch_add(1, Ordering::Relaxed) as u64;
            urls.shuffle(&mut StdRng::seed_from_u64(make_seed(&[seed, epoch])));
        }
        Box::new(urls.into_iter().map(|url| Ok(shard_sample(url))))
    }
}

/// Shards drawn with replacement, forever by default.
///
/// Resampling is how multi-node training usually runs: every worker draws its
/// own shards, so no worker ever runs dry and the epoch length is set by the
/// consumer rather than by how the data happened to be split.
#[derive(Debug)]
pub struct ResampledShards {
    urls: Vec<String>,
    count: Option<usize>,
    seed: u64,
    deterministic: bool,
    epoch: AtomicUsize,
}

impl ResampledShards {
    /// Resample from the shards `patterns` expands to.
    pub fn new<S: AsRef<str>>(patterns: impl IntoIterator<Item = S>) -> Result<ResampledShards> {
        let mut urls = Vec::new();
        for pattern in patterns {
            urls.extend(expand_urls(pattern.as_ref())?);
        }
        if urls.is_empty() {
            return Err(Error::value("cannot resample from an empty shard list"));
        }
        Ok(ResampledShards { urls, count: None, seed: 0, deterministic: false, epoch: AtomicUsize::new(0) })
    }

    /// Draw exactly `count` shards per epoch instead of an endless stream.
    pub fn with_count(mut self, count: usize) -> ResampledShards {
        self.count = Some(count);
        self
    }

    /// Set the base seed.
    pub fn with_seed(mut self, seed: u64) -> ResampledShards {
        self.seed = seed;
        self
    }

    /// Draw the same shards on every run, given the same worker and epoch.
    ///
    /// Off by default, matching the Python library: two runs of the same job
    /// should normally see different data.
    pub fn deterministic(mut self, deterministic: bool) -> ResampledShards {
        self.deterministic = deterministic;
        self
    }

    /// The shards being drawn from.
    pub fn urls(&self) -> &[String] {
        &self.urls
    }
}

impl Stage for ResampledShards {
    fn apply(&self, _input: SampleStream) -> SampleStream {
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed) as u64;
        let worker = worker_info();
        let mut parts = vec![worker.seed(), epoch, self.seed];
        if !self.deterministic {
            // Mix in real entropy so two runs of the same job diverge.
            parts.push(rand::rng().random());
        }
        let mut rng = StdRng::seed_from_u64(make_seed(&parts));

        let urls = self.urls.clone();
        let draw = move || {
            let index = rng.random_range(0..urls.len());
            Ok(shard_sample(urls[index].clone()))
        };
        match self.count {
            Some(n) => Box::new(std::iter::repeat_with(draw).take(n)),
            None => Box::new(std::iter::repeat_with(draw)),
        }
    }
}

/// Keeps only the shards belonging to this distributed rank.
///
/// Shards are dealt out round-robin, so rank `r` of `w` takes shards
/// `r, r + w, r + 2w, ...`. With a single process this is a no-op.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitByNode;

impl Stage for SplitByNode {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let info = worker_info();
        if info.world_size <= 1 {
            return input;
        }
        Box::new(input.skip(info.rank).step_by(info.world_size))
    }
}

/// Keeps only the shards belonging to this loader worker.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitByWorker;

impl Stage for SplitByWorker {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let info = worker_info();
        if info.num_workers <= 1 {
            return input;
        }
        Box::new(input.skip(info.worker).step_by(info.num_workers))
    }
}

/// Fails loudly when a pipeline that has no node splitter is run distributed.
///
/// Silently training every rank on the same data is a subtle and expensive
/// mistake, so the default is to refuse rather than to guess.
#[derive(Debug, Clone, Copy, Default)]
pub struct SingleNodeOnly;

impl Stage for SingleNodeOnly {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let info = worker_info();
        if info.world_size > 1 {
            return Box::new(std::iter::once(Err(Error::value(
                "this pipeline has no node splitter but is running on multiple nodes; \
                 add SplitByNode, or resample shards instead",
            ))));
        }
        input
    }
}

/// What to do with a shard once a [`DirectoryShardList`] has finished with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Disposal {
    /// Put it back so another pass can pick it up again.
    #[default]
    Resample,
    /// Rename it to `<name>._done_` so it is not picked up again.
    Keep,
    /// Delete it.
    Unlink,
}

/// Shards claimed from a directory as they appear.
///
/// A shard is claimed by renaming it to `<name>._<pid>_`, which makes the claim
/// atomic across processes sharing the directory: exactly one process can win
/// the rename. Abandoned claims from processes that are no longer running are
/// recovered on each pass.
#[derive(Debug)]
pub struct DirectoryShardList {
    directory: PathBuf,
    pattern: String,
    disposal: Disposal,
    poll: Option<std::time::Duration>,
    timeout: std::time::Duration,
    newest_first: bool,
}

impl DirectoryShardList {
    /// Watch `directory` for shards matching `*.{tar,tgz,tar.gz}`.
    pub fn new(directory: impl Into<PathBuf>) -> DirectoryShardList {
        DirectoryShardList {
            directory: directory.into(),
            pattern: "*.tar".to_string(),
            disposal: Disposal::Resample,
            poll: Some(std::time::Duration::from_secs(1)),
            timeout: std::time::Duration::from_secs(3600),
            newest_first: false,
        }
    }

    /// Match shards with this glob pattern instead.
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> DirectoryShardList {
        self.pattern = pattern.into();
        self
    }

    /// What to do with each shard once it has been read.
    pub fn with_disposal(mut self, disposal: Disposal) -> DirectoryShardList {
        self.disposal = disposal;
        self
    }

    /// Wait this long between scans; `None` stops as soon as the directory is empty.
    pub fn with_poll(mut self, poll: Option<std::time::Duration>) -> DirectoryShardList {
        self.poll = poll;
        self
    }

    /// Give up after this long without finding a shard.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> DirectoryShardList {
        self.timeout = timeout;
        self
    }

    /// Take the newest shard rather than a random one.
    pub fn newest_first(mut self, newest_first: bool) -> DirectoryShardList {
        self.newest_first = newest_first;
        self
    }

    /// Return claims whose owning process has exited.
    fn recover_abandoned(&self) {
        let Ok(entries) = std::fs::read_dir(&self.directory) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(pid) = claim_pid(&name) else { continue };
            if !process_is_running(pid) {
                self.dispose(&entry.path());
            }
        }
    }

    /// Apply the configured disposal to a claimed shard.
    fn dispose(&self, claimed: &std::path::Path) {
        let original = strip_claim(claimed);
        let outcome = match self.disposal {
            Disposal::Unlink => std::fs::remove_file(claimed),
            Disposal::Keep => std::fs::rename(claimed, original.with_extension("_done_")),
            Disposal::Resample => std::fs::rename(claimed, &original),
        };
        if let Err(e) = outcome {
            log::warn!("could not dispose of {}: {e}", claimed.display());
        }
    }
}

/// The pid embedded in a claimed shard name such as `a.tar._1234_`.
fn claim_pid(name: &str) -> Option<u32> {
    let rest = name.rsplit_once("._")?.1;
    rest.strip_suffix('_')?.parse().ok()
}

/// Strip the `._<pid>_` claim suffix from a path.
fn strip_claim(path: &std::path::Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let original = match name.rsplit_once("._") {
        Some((base, rest)) if rest.ends_with('_') => base.to_string(),
        _ => name,
    };
    path.with_file_name(original)
}

#[cfg(unix)]
fn process_is_running(pid: u32) -> bool {
    // Signal 0 performs the permission and existence checks without signalling.
    std::path::Path::new(&format!("/proc/{pid}")).exists()
        || std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
}

#[cfg(not(unix))]
fn process_is_running(_pid: u32) -> bool {
    true
}

impl Stage for DirectoryShardList {
    fn apply(&self, _input: SampleStream) -> SampleStream {
        let directory = self.directory.clone();
        let pattern = self.pattern.clone();
        let poll = self.poll;
        let timeout = self.timeout;
        let newest_first = self.newest_first;
        let disposal = self.disposal;
        let mut rng = StdRng::seed_from_u64(rand::rng().random());
        let mut claimed: Option<PathBuf> = None;
        let start = std::time::Instant::now();

        Box::new(std::iter::from_fn(move || {
            let watcher = DirectoryShardList {
                directory: directory.clone(),
                pattern: pattern.clone(),
                disposal,
                ..DirectoryShardList::new(directory.clone())
            };
            if let Some(previous) = claimed.take() {
                watcher.dispose(&previous);
                watcher.recover_abandoned();
            }
            loop {
                if start.elapsed() > timeout {
                    return None;
                }
                let glob_pattern = directory.join(&pattern);
                let mut candidates: Vec<PathBuf> = glob::glob(&glob_pattern.to_string_lossy())
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter(|p| p.is_file())
                    .collect();

                if candidates.is_empty() {
                    match poll {
                        Some(interval) => {
                            std::thread::sleep(interval);
                            continue;
                        }
                        None => return None,
                    }
                }

                let chosen = if newest_first {
                    candidates.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
                    candidates.pop().expect("candidates is not empty")
                } else {
                    let index = rng.random_range(0..candidates.len());
                    candidates.swap_remove(index)
                };

                // Renaming is the claim: only one process can win it.
                let claim = chosen.with_file_name(format!(
                    "{}._{}_",
                    chosen.file_name().unwrap_or_default().to_string_lossy(),
                    std::process::id()
                ));
                if std::fs::rename(&chosen, &claim).is_err() {
                    continue;
                }
                claimed = Some(claim.clone());
                return Some(Ok(shard_sample(claim.to_string_lossy().into_owned())));
            }
        }))
    }
}

/// One source within a [`MultiShardSample`] specification.
#[derive(Debug, Clone)]
pub struct MultiSource {
    /// A label for logging.
    pub name: String,
    /// The shards this source can contribute.
    pub urls: Vec<String>,
    /// Draw this many shards with replacement each epoch.
    pub resample: Option<usize>,
    /// Or choose this many shards without replacement each epoch.
    pub choose: Option<usize>,
}

/// Several shard sources mixed together, epoch by epoch.
///
/// Each source can contribute all of its shards, a fixed number chosen without
/// replacement (`choose`), or a fixed number drawn with replacement
/// (`resample`), which is how datasets of very different sizes are balanced.
#[derive(Debug)]
pub struct MultiShardSample {
    sources: Vec<MultiSource>,
    rng: Mutex<StdRng>,
}

impl MultiShardSample {
    /// Mix these sources.
    pub fn new(sources: Vec<MultiSource>) -> Result<MultiShardSample> {
        for source in &sources {
            if source.resample.is_some() && source.choose.is_some() {
                return Err(Error::value(format!("{}: set only one of resample and choose", source.name)));
            }
            if source.choose.is_some_and(|n| n > source.urls.len()) {
                return Err(Error::value(format!(
                    "{}: cannot choose {} shards from {}",
                    source.name,
                    source.choose.unwrap_or(0),
                    source.urls.len()
                )));
            }
        }
        Ok(MultiShardSample { sources, rng: Mutex::new(StdRng::seed_from_u64(rand::rng().random())) })
    }

    /// Parse a YAML specification.
    ///
    /// ```yaml
    /// prefix: "pipe:curl -s -L https://host/"
    /// buckets: dataset-a
    /// datasets:
    ///   - name: big
    ///     shards: shard-{000000..000999}.tar
    ///     choose: 100
    ///   - name: small
    ///     buckets: dataset-b
    ///     shards: shard-{000..009}.tar
    ///     resample: 100
    /// ```
    #[cfg(feature = "yaml")]
    pub fn from_yaml(text: &str) -> Result<MultiShardSample> {
        let spec: serde_yaml::Value =
            serde_yaml::from_str(text).map_err(|e| Error::format(format!("bad dataset spec: {e}")))?;
        let get = |value: &serde_yaml::Value, key: &str| value.get(key).cloned();

        let prefix = get(&spec, "prefix").and_then(|v| v.as_str().map(expand_path)).unwrap_or_default();
        let default_buckets = buckets_of(&spec);

        let datasets = get(&spec, "datasets")
            .and_then(|v| v.as_sequence().cloned())
            .ok_or_else(|| Error::format("dataset spec has no `datasets` list"))?;

        let mut sources = Vec::new();
        for dataset in datasets {
            let buckets = {
                let own = buckets_of(&dataset);
                if own.is_empty() { default_buckets.clone() } else { own }
            };
            let bucket = buckets.first().cloned().unwrap_or_default();
            let name = get(&dataset, "name")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("@{bucket}"));

            let shards = get(&dataset, "shards").ok_or_else(|| Error::format(format!("{name}: no `shards`")))?;
            let patterns: Vec<String> = match &shards {
                serde_yaml::Value::String(s) => vec![s.clone()],
                serde_yaml::Value::Sequence(items) => {
                    items.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
                }
                _ => return Err(Error::format(format!("{name}: `shards` must be a string or a list"))),
            };

            let mut urls = Vec::new();
            for pattern in patterns {
                for url in expand_urls(&expand_path(&pattern))? {
                    urls.push(format!("{prefix}{}", join_bucket(&bucket, &url)));
                }
            }

            sources.push(MultiSource {
                name,
                urls,
                resample: get(&dataset, "resample").and_then(|v| v.as_u64()).map(|n| n as usize),
                choose: get(&dataset, "choose").and_then(|v| v.as_u64()).map(|n| n as usize),
            });
        }
        MultiShardSample::new(sources)
    }

    /// Read a YAML specification from a file.
    #[cfg(feature = "yaml")]
    pub fn from_yaml_file(path: impl AsRef<std::path::Path>) -> Result<MultiShardSample> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).map_err(|e| Error::Io(e).context(format!("reading {}", path.display())))?;
        MultiShardSample::from_yaml(&text)
    }

    /// Pin the shard selection to `seed`, so every node picks the same shards.
    pub fn set_seed(&self, seed: u64) {
        *self.rng.lock().expect("rng lock") = StdRng::seed_from_u64(seed);
    }

    /// The sources being mixed.
    pub fn sources(&self) -> &[MultiSource] {
        &self.sources
    }

    /// Draw this epoch's shard list.
    pub fn shards_for_epoch(&self) -> Vec<String> {
        let mut rng = self.rng.lock().expect("rng lock");
        let mut out = Vec::new();
        for source in &self.sources {
            match (source.resample, source.choose) {
                (Some(n), _) if !source.urls.is_empty() => {
                    for _ in 0..n {
                        out.push(source.urls[rng.random_range(0..source.urls.len())].clone());
                    }
                }
                (_, Some(n)) => {
                    let mut shuffled = source.urls.clone();
                    shuffled.shuffle(&mut *rng);
                    shuffled.truncate(n);
                    out.extend(shuffled);
                }
                _ => out.extend(source.urls.iter().cloned()),
            }
        }
        out.shuffle(&mut *rng);
        out
    }
}

impl Stage for MultiShardSample {
    fn apply(&self, _input: SampleStream) -> SampleStream {
        Box::new(self.shards_for_epoch().into_iter().map(|url| Ok(shard_sample(url))))
    }
}

#[cfg(feature = "yaml")]
fn buckets_of(value: &serde_yaml::Value) -> Vec<String> {
    match value.get("buckets") {
        Some(serde_yaml::Value::String(s)) => vec![expand_path(s)],
        Some(serde_yaml::Value::Sequence(items)) => items.iter().filter_map(|v| v.as_str().map(expand_path)).collect(),
        _ => Vec::new(),
    }
}

/// Expand `~` and `$VAR` in a path fragment, as the Python spec loader does.
#[cfg(feature = "yaml")]
fn expand_path(text: &str) -> String {
    let expanded = match text.strip_prefix("~/") {
        Some(rest) => match std::env::var("HOME") {
            Ok(home) => format!("{home}/{rest}"),
            Err(_) => text.to_string(),
        },
        None => text.to_string(),
    };
    // `${VAR}` and `$VAR`, left as-is when the variable is not set.
    let mut out = String::with_capacity(expanded.len());
    let mut rest = expanded.as_str();
    while let Some(at) = rest.find('$') {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 1..];
        let (name, remainder) = match tail.strip_prefix('{') {
            Some(braced) => match braced.find('}') {
                Some(end) => (&braced[..end], &braced[end + 1..]),
                None => ("", tail),
            },
            None => {
                let end = tail.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(tail.len());
                (&tail[..end], &tail[end..])
            }
        };
        match std::env::var(name) {
            Ok(value) => out.push_str(&value),
            Err(_) => {
                out.push('$');
                out.push_str(name);
            }
        }
        rest = remainder;
    }
    out.push_str(rest);
    out
}

#[cfg(feature = "yaml")]
fn join_bucket(bucket: &str, url: &str) -> String {
    if bucket.is_empty() {
        return url.to_string();
    }
    if bucket.ends_with('/') { format!("{bucket}{url}") } else { format!("{bucket}/{url}") }
}

/// A per-worker seed that differs between workers but is stable for one.
pub fn worker_seed(base: &str) -> u64 {
    let info = worker_info();
    make_seed(&[seed_from_str(base), info.seed()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use webdataset_core::utils::with_worker;

    fn urls_of(stage: &dyn Stage) -> Vec<String> {
        stage.apply(crate::pipeline::empty_stream()).map(|s| s.unwrap().url().unwrap().to_string()).collect()
    }

    #[test]
    fn expands_brace_patterns() {
        let shards = SimpleShardList::new(["s-{000..002}.tar"]).unwrap();
        assert_eq!(shards.urls(), ["s-000.tar", "s-001.tar", "s-002.tar"]);
        assert_eq!(urls_of(&shards).len(), 3);
    }

    #[test]
    fn keeps_order_without_a_seed_and_varies_with_one() {
        let ordered = SimpleShardList::new(["s-{00..19}.tar"]).unwrap();
        assert_eq!(urls_of(&ordered), urls_of(&ordered));

        let shuffled = SimpleShardList::new(["s-{00..19}.tar"]).unwrap().shuffled(1);
        let first = urls_of(&shuffled);
        let second = urls_of(&shuffled);
        assert_ne!(first, second, "each epoch should reshuffle");

        let mut sorted = first.clone();
        sorted.sort();
        assert_eq!(sorted, ordered.urls(), "shuffling must not lose shards");
    }

    #[test]
    fn rejects_an_empty_shard_list() {
        assert!(SimpleShardList::new(Vec::<String>::new()).is_err());
        assert!(ResampledShards::new(Vec::<String>::new()).is_err());
    }

    #[test]
    fn resamples_with_replacement() {
        let shards = ResampledShards::new(["s-{000..009}.tar"]).unwrap().with_count(200);
        let drawn = urls_of(&shards);
        assert_eq!(drawn.len(), 200);

        let mut unique = drawn.clone();
        unique.sort();
        unique.dedup();
        assert!(unique.len() <= 10);
        assert!(unique.len() > 1, "200 draws from 10 shards should hit more than one");
    }

    #[test]
    fn resamples_deterministically_when_asked() {
        let shards = ResampledShards::new(["s-{000..099}.tar"]).unwrap().with_count(20).deterministic(true);
        // A deterministic list still advances its epoch, so compare two fresh ones.
        let a = urls_of(&ResampledShards::new(["s-{000..099}.tar"]).unwrap().with_count(20).deterministic(true));
        let b = urls_of(&ResampledShards::new(["s-{000..099}.tar"]).unwrap().with_count(20).deterministic(true));
        assert_eq!(a, b);
        assert_eq!(urls_of(&shards).len(), 20);
    }

    #[test]
    fn resampling_never_ends_without_a_count() {
        let shards = ResampledShards::new(["a.tar", "b.tar"]).unwrap();
        assert_eq!(shards.apply(crate::pipeline::empty_stream()).take(1000).count(), 1000);
    }

    #[test]
    fn splits_shards_across_workers() {
        let shards = SimpleShardList::new(["s-{0..7}.tar"]).unwrap();
        let mut seen = Vec::new();
        for worker in 0..4 {
            let assigned = with_worker(worker, 4, || {
                SplitByWorker
                    .apply(shards.apply(crate::pipeline::empty_stream()))
                    .map(|s| s.unwrap().url().unwrap().to_string())
                    .collect::<Vec<_>>()
            });
            assert_eq!(assigned.len(), 2, "worker {worker}");
            seen.extend(assigned);
        }
        seen.sort();
        assert_eq!(seen, shards.urls(), "every shard goes to exactly one worker");
    }

    #[test]
    fn splitting_is_a_no_op_with_one_worker() {
        let shards = SimpleShardList::new(["s-{0..3}.tar"]).unwrap();
        let split = SplitByWorker.apply(shards.apply(crate::pipeline::empty_stream()));
        assert_eq!(split.count(), 4);
    }

    #[test]
    fn single_node_only_refuses_to_guess() {
        let shards = SimpleShardList::new(["a.tar"]).unwrap();
        let ok: Vec<_> = SingleNodeOnly.apply(shards.apply(crate::pipeline::empty_stream())).collect();
        assert!(ok.iter().all(Result::is_ok));
    }

    #[test]
    fn parses_claim_names() {
        assert_eq!(claim_pid("a.tar._1234_"), Some(1234));
        assert_eq!(claim_pid("a.tar"), None);
        assert_eq!(strip_claim(std::path::Path::new("/d/a.tar._12_")), PathBuf::from("/d/a.tar"));
    }

    #[test]
    fn claims_shards_from_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a.tar", "b.tar"] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }

        let shards = DirectoryShardList::new(dir.path()).with_poll(None).with_disposal(Disposal::Unlink);
        let claimed: Vec<String> =
            shards.apply(crate::pipeline::empty_stream()).map(|s| s.unwrap().url().unwrap().to_string()).collect();

        assert_eq!(claimed.len(), 2);
        assert!(claimed.iter().all(|c| c.contains("._")), "{claimed:?}");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0, "unlink disposal empties the directory");
    }

    #[cfg(feature = "yaml")]
    #[test]
    fn parses_a_yaml_specification() {
        let spec = r#"
prefix: "pipe:curl -s -L https://host/"
buckets: main
datasets:
  - name: big
    shards: s-{000..009}.tar
    choose: 3
  - name: small
    buckets: other
    shards:
      - t-{0..1}.tar
    resample: 5
"#;
        let sample = MultiShardSample::from_yaml(spec).unwrap();
        assert_eq!(sample.sources().len(), 2);
        assert_eq!(sample.sources()[0].urls.len(), 10);
        assert_eq!(sample.sources()[0].urls[0], "pipe:curl -s -L https://host/main/s-000.tar");
        assert_eq!(sample.sources()[1].urls[0], "pipe:curl -s -L https://host/other/t-0.tar");

        sample.set_seed(4);
        let shards = sample.shards_for_epoch();
        assert_eq!(shards.len(), 8, "3 chosen plus 5 resampled");
    }

    #[cfg(feature = "yaml")]
    #[test]
    fn rejects_contradictory_specifications() {
        let spec = "datasets:\n  - shards: a.tar\n    choose: 5\n    resample: 5\n";
        assert!(MultiShardSample::from_yaml(spec).is_err());

        let spec = "datasets:\n  - shards: a.tar\n    choose: 5\n";
        assert!(MultiShardSample::from_yaml(spec).is_err(), "cannot choose 5 of 1");
    }
}
