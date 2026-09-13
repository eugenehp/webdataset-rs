//! Shared helpers: filename splitting, worker identity, seeds, and secure mode.

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::braceexpand::braceexpand;
use crate::error::{Error, Result};
use crate::prelude::*;

/// Split a path into its basename and its full extension.
///
/// The basename is everything up to the first dot after the last slash; the
/// extension is everything after it. Files in a WebDataset share a basename
/// exactly when they belong to the same sample.
///
/// ```
/// use webdataset_core::utils::base_plus_ext;
///
/// assert_eq!(base_plus_ext("a/b/c.png"), Some(("a/b/c", "png")));
/// assert_eq!(base_plus_ext("a/b/c.seg.png"), Some(("a/b/c", "seg.png")));
/// assert_eq!(base_plus_ext("noextension"), None);
/// ```
pub fn base_plus_ext(path: &str) -> Option<(&str, &str)> {
    // Everything after the last slash is the file name; the basename runs up to
    // its first dot, and the extension is the rest. A name that starts with a
    // dot has no basename, so it belongs to no sample.
    let start = path.rfind('/').map(|at| at + 1).unwrap_or(0);
    let dot = start + path[start..].find('.')?;
    if dot == start {
        return None;
    }
    Some((&path[..dot], &path[dot + 1..]))
}

/// Whether a tar member name is shard-level metadata that should be skipped.
///
/// These are top-level entries wrapped in double underscores, such as
/// `__index__` or the `__dup__/` directory.
pub fn is_shard_metadata(name: &str) -> bool {
    if !name.contains('/') && name.starts_with("__") && name.ends_with("__") {
        return true;
    }
    // The Python default `skip_meta` regexp: `__[^/]*__($|/)`.
    let Some(rest) = name.strip_prefix("__") else {
        return false;
    };
    let Some(end) = rest.find("__") else {
        return false;
    };
    if rest[..end].contains('/') {
        return false;
    }
    let after = &rest[end + 2..];
    after.is_empty() || after.starts_with('/')
}

/// Substitute `${NAME}` with the environment variable `WDS_NAME`.
///
/// Fails when a referenced variable is not set, matching the reference
/// implementation's assertion.
#[cfg(feature = "std")]
pub fn envsubst(text: &str) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("${") {
        out.push_str(&rest[..at]);
        let body = &rest[at + 2..];
        let Some(end) = body.find('}') else {
            // An unterminated `${` is literal text, as in the shell.
            out.push_str(&rest[at..]);
            return Ok(out);
        };
        let name = &body[..end];
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            out.push_str(&rest[at..at + 2 + end + 1]);
            rest = &body[end + 1..];
            continue;
        }
        let variable = format!("WDS_{name}");
        match std::env::var(&variable) {
            Ok(value) => out.push_str(&value),
            Err(_) => return Err(Error::value(format!("missing environment variable {variable}"))),
        }
        rest = &body[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Without `std` there is no environment to substitute from, so the text is
/// returned unchanged.
#[cfg(not(feature = "std"))]
pub fn envsubst(text: &str) -> Result<String> {
    Ok(String::from(text))
}

/// Expand a URL specification into a concrete list of shard URLs.
///
/// The specification is split on `::`, each part has `${VAR}` substituted from
/// `WDS_*` environment variables, and brace expressions are expanded.
///
/// ```
/// use webdataset_core::utils::expand_urls;
///
/// let urls = expand_urls("a-{0..1}.tar::b.tar").unwrap();
/// assert_eq!(urls, ["a-0.tar", "a-1.tar", "b.tar"]);
/// ```
pub fn expand_urls(spec: &str) -> Result<Vec<String>> {
    let mut result = Vec::new();
    for part in spec.split("::") {
        // Substitution can itself introduce `${...}`, so iterate to a fixpoint
        // with the same bound the reference implementation uses.
        let mut url = part.to_string();
        for _ in 0..10 {
            let next = envsubst(&url)?;
            if next == url {
                break;
            }
            url = next;
        }
        result.extend(braceexpand(&url)?);
    }
    Ok(result)
}

/// Combine several values into a 31-bit seed, mirroring `utils.make_seed`.
pub fn make_seed(parts: &[u64]) -> u64 {
    let mut seed: u64 = 0;
    for part in parts {
        seed = seed.wrapping_mul(31).wrapping_add(mix(*part)) & 0x7FFF_FFFF;
    }
    seed
}

/// A cheap avalanche used by [`make_seed`] so that adjacent inputs diverge.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// Hash a string into a seed component.
pub fn seed_from_str(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Which process and worker this code is running in.
///
/// Mirrors `pytorch_worker_info`: distributed rank and world size identify the
/// node, worker and count identify the loader worker within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerInfo {
    /// This process's index among all distributed processes.
    pub rank: usize,
    /// The total number of distributed processes.
    pub world_size: usize,
    /// This thread's index among the loader workers of this process.
    pub worker: usize,
    /// The total number of loader workers in this process.
    pub num_workers: usize,
}

impl Default for WorkerInfo {
    fn default() -> Self {
        WorkerInfo { rank: 0, world_size: 1, worker: 0, num_workers: 1 }
    }
}

impl WorkerInfo {
    /// A deterministic per-worker seed, as `pytorch_worker_seed` computes it.
    pub fn seed(&self) -> u64 {
        (self.rank * 1000 + self.worker) as u64
    }
}

/// Report the current worker identity.
///
/// The per-thread binding set by [`with_worker`] wins; otherwise
/// `RANK`/`WORLD_SIZE` and `WORKER`/`NUM_WORKERS` are read from the
/// environment. See [`crate::workers`] for how the binding is stored on targets
/// without thread-local storage.
pub fn worker_info() -> WorkerInfo {
    let mut info = WorkerInfo::default();
    #[cfg(feature = "std")]
    if let (Ok(rank), Ok(world)) = (env_usize("RANK"), env_usize("WORLD_SIZE")) {
        info.rank = rank;
        info.world_size = world.max(1);
    }
    if let Some((worker, num_workers)) = crate::workers::current() {
        info.worker = worker;
        info.num_workers = num_workers.max(1);
        return info;
    }
    #[cfg(feature = "std")]
    if let (Ok(worker), Ok(num)) = (env_usize("WORKER"), env_usize("NUM_WORKERS")) {
        info.worker = worker;
        info.num_workers = num.max(1);
    }
    info
}

pub use crate::workers::{clear_worker, set_thread_id_hook, set_worker, with_worker};

#[cfg(feature = "std")]
fn env_usize(name: &str) -> core::result::Result<usize, ()> {
    std::env::var(name).map_err(|_| ())?.parse().map_err(|_| ())
}

static SECURE: AtomicBool = AtomicBool::new(false);
static SECURE_INIT: AtomicBool = AtomicBool::new(false);

/// Whether secure mode is on.
///
/// Secure mode disables the `pipe:` and `file:` URL schemes, URL rewriting from
/// the environment, and decoders that execute embedded code. It is enabled by
/// setting `WDS_SECURE=1` or by calling [`set_enforce_security`].
pub fn enforce_security() -> bool {
    if !SECURE_INIT.swap(true, Ordering::Relaxed) {
        #[cfg(feature = "std")]
        {
            let on = std::env::var("WDS_SECURE").map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
            SECURE.store(on, Ordering::Relaxed);
        }
    }
    SECURE.load(Ordering::Relaxed)
}

/// Turn secure mode on or off explicitly.
pub fn set_enforce_security(on: bool) {
    SECURE_INIT.store(true, Ordering::Relaxed);
    SECURE.store(on, Ordering::Relaxed);
}

/// Fail when secure mode forbids `what`.
pub fn check_security(what: &str) -> Result<()> {
    if enforce_security() {
        return Err(Error::Security(what.to_string()));
    }
    Ok(())
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A process-unique counter, used to keep temporary file names from colliding.
pub fn next_unique() -> usize {
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Format `value` into a `printf`-style integer pattern such as `%06d`.
///
/// Shard writers take their output patterns in this form for compatibility
/// with the Python API, where the pattern is used with the `%` operator.
///
/// ```
/// use webdataset_core::utils::format_shard_pattern;
///
/// assert_eq!(format_shard_pattern("out-%06d.tar", 12).unwrap(), "out-000012.tar");
/// assert_eq!(format_shard_pattern("out-%d.tar", 12).unwrap(), "out-12.tar");
/// ```
pub fn format_shard_pattern(pattern: &str, value: usize) -> Result<String> {
    let mut out = String::with_capacity(pattern.len() + 8);
    let mut chars = pattern.chars().peekable();
    let mut substituted = false;

    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            out.push('%');
            continue;
        }
        let mut width = String::new();
        while let Some(&d) = chars.peek() {
            if d.is_ascii_digit() {
                width.push(d);
                chars.next();
            } else {
                break;
            }
        }
        match chars.next() {
            Some('d') | Some('i') => {
                let zero_padded = width.starts_with('0');
                let w: usize = width.parse().unwrap_or(0);
                if zero_padded {
                    out.push_str(&format!("{value:0>w$}"));
                } else {
                    out.push_str(&format!("{value:>w$}"));
                }
                substituted = true;
            }
            other => {
                return Err(Error::value(format!(
                    "shard pattern {pattern:?} has an unsupported conversion %{}",
                    other.unwrap_or(' ')
                )));
            }
        }
    }

    if !substituted {
        return Err(Error::value(format!("shard pattern {pattern:?} has no %d conversion")));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_names_the_way_the_format_requires() {
        assert_eq!(base_plus_ext("a.png"), Some(("a", "png")));
        assert_eq!(base_plus_ext("dir/a.seg.png"), Some(("dir/a", "seg.png")));
        assert_eq!(base_plus_ext("a/b.c/d.png"), Some(("a/b.c/d", "png")));
        assert_eq!(base_plus_ext("plain"), None);
    }

    #[test]
    fn recognises_shard_metadata() {
        assert!(is_shard_metadata("__index__"));
        assert!(is_shard_metadata("__dup__/a.png"));
        assert!(!is_shard_metadata("dir/__index__"));
        assert!(!is_shard_metadata("normal.png"));
    }

    #[test]
    fn formats_shard_patterns() {
        assert_eq!(format_shard_pattern("s-%06d.tar", 7).unwrap(), "s-000007.tar");
        assert_eq!(format_shard_pattern("s-%d.tar", 7).unwrap(), "s-7.tar");
        assert_eq!(format_shard_pattern("100%%-%d", 1).unwrap(), "100%-1");
        assert!(format_shard_pattern("no-conversion.tar", 1).is_err());
        assert!(format_shard_pattern("%s.tar", 1).is_err());
    }

    #[test]
    fn overrides_worker_identity_per_thread() {
        let info = with_worker(2, 4, worker_info);
        assert_eq!(info.worker, 2);
        assert_eq!(info.num_workers, 4);
        assert_eq!(worker_info().worker, 0, "override is scoped to the closure");
    }

    #[test]
    fn derives_stable_distinct_seeds() {
        assert_eq!(make_seed(&[1, 2, 3]), make_seed(&[1, 2, 3]));
        assert_ne!(make_seed(&[1, 2, 3]), make_seed(&[1, 2, 4]));
        assert!(make_seed(&[u64::MAX, 7]) <= 0x7FFF_FFFF);
    }
}
