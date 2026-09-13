//! Open a URL, whatever its scheme.
//!
//! `gopen` is WebDataset's one-stop opener. It maps a URL scheme to a handler
//! and returns a byte stream, so the rest of the library never has to care
//! whether a shard lives on local disk, behind HTTPS, or in an object store.
//!
//! ```no_run
//! use std::io::Read;
//! use webdataset_io::gopen;
//!
//! let mut stream = gopen("testdata/sample.tgz")?;
//! let mut first = [0u8; 8];
//! stream.read_exact(&mut first)?;
//! # Ok::<(), webdataset_core::Error>(())
//! ```
//!
//! Handlers for additional schemes can be installed with [`register_scheme`].

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::{Arc, LazyLock, RwLock};

use webdataset_core::error::{Error, Result};
use webdataset_core::utils::check_security;

use crate::url;

#[cfg(feature = "subprocess")]
use crate::pipe::{Pipe, shell};
#[cfg(feature = "subprocess")]
use std::process::Command;

/// A readable byte stream whose completion can fail.
///
/// Reading from a subprocess only reveals an error once the child exits, so
/// callers should invoke [`Fetch::finish`] after the last read.
pub trait Fetch: Read + Send {
    /// Finish reading and report any error the source noticed at the end.
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A writable byte stream whose completion can fail.
pub trait Sink: Write + Send {
    /// Flush, close, and report any error the destination noticed.
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

impl Fetch for File {}
impl Fetch for std::io::Stdin {}
#[cfg(feature = "subprocess")]
impl Fetch for Pipe {
    fn finish(&mut self) -> Result<()> {
        Pipe::finish(self)
    }
}

impl Sink for File {
    fn finish(&mut self) -> Result<()> {
        self.flush()?;
        Ok(())
    }
}
impl Sink for std::io::Stdout {
    fn finish(&mut self) -> Result<()> {
        self.flush()?;
        Ok(())
    }
}
#[cfg(feature = "subprocess")]
impl Sink for Pipe {
    fn finish(&mut self) -> Result<()> {
        self.flush()?;
        Pipe::finish(self)
    }
}

/// Opens URLs of one scheme.
pub trait SchemeHandler: Send + Sync {
    /// Open the URL for reading.
    fn open_read(&self, url: &str) -> Result<Box<dyn Fetch>>;

    /// Open the URL for writing.
    fn open_write(&self, url: &str) -> Result<Box<dyn Sink>> {
        Err(Error::unsupported(format!("writing to {url}")))
    }
}

/// The scheme dispatch table, mirroring `gopen_schemes` in the Python library.
static SCHEMES: LazyLock<RwLock<HashMap<String, Arc<dyn SchemeHandler>>>> =
    LazyLock::new(|| RwLock::new(default_schemes()));

/// Install a handler for `scheme`, replacing any existing one.
pub fn register_scheme(scheme: &str, handler: Arc<dyn SchemeHandler>) {
    SCHEMES.write().expect("scheme table poisoned").insert(scheme.to_string(), handler);
}

/// Look up the handler for `scheme`.
pub fn scheme_handler(scheme: &str) -> Option<Arc<dyn SchemeHandler>> {
    SCHEMES.read().expect("scheme table poisoned").get(scheme).cloned()
}

/// The schemes that currently have handlers.
pub fn registered_schemes() -> Vec<String> {
    let mut names: Vec<String> = SCHEMES.read().expect("scheme table poisoned").keys().cloned().collect();
    names.sort();
    names
}

/// Open `url` for reading.
///
/// `-` reads standard input. URLs without a scheme, and `file:` URLs, are read
/// from the local filesystem — both are refused in secure mode.
pub fn gopen(url: &str) -> Result<Box<dyn Fetch>> {
    log::debug!("gopen {url}");
    if url == "-" {
        return Ok(Box::new(std::io::stdin()));
    }
    let url = rewrite_url(url)?;
    match url::scheme(&url) {
        None | Some("file") => {
            check_security("opening local files")?;
            let path = url::to_local_path(&url);
            let file = File::open(&path).map_err(|e| Error::Io(e).context(format!("opening {path}")))?;
            Ok(Box::new(file))
        }
        Some(scheme) => match scheme_handler(scheme) {
            Some(handler) => handler.open_read(&url),
            None => Err(Error::unsupported(format!("{url}: no gopen handler for scheme {scheme:?}"))),
        },
    }
}

/// Open `url` for writing.
///
/// `-` writes to standard output.
pub fn gopen_write(url: &str) -> Result<Box<dyn Sink>> {
    log::debug!("gopen (write) {url}");
    if url == "-" {
        return Ok(Box::new(std::io::stdout()));
    }
    let url = rewrite_url(url)?;
    match url::scheme(&url) {
        None | Some("file") => {
            check_security("opening local files")?;
            let path = url::to_local_path(&url);
            let file = File::create(&path).map_err(|e| Error::Io(e).context(format!("creating {path}")))?;
            Ok(Box::new(file))
        }
        Some(scheme) => match scheme_handler(scheme) {
            Some(handler) => handler.open_write(&url),
            None => Err(Error::unsupported(format!("{url}: no gopen handler for scheme {scheme:?}"))),
        },
    }
}

/// Apply the rewrite rules in `GOPEN_REWRITE`.
///
/// The variable holds `;`-separated `prefix=replacement` rules; the first rule
/// whose prefix matches the start of the URL is applied. Rewriting is refused
/// in secure mode, since it lets the environment redirect reads.
pub fn rewrite_url(url: &str) -> Result<String> {
    let Ok(rules) = std::env::var("GOPEN_REWRITE") else {
        return Ok(url.to_string());
    };
    check_security("rewriting URLs from GOPEN_REWRITE")?;
    for rule in rules.split(';') {
        let Some((prefix, replacement)) = rule.split_once('=') else {
            continue;
        };
        if let Some(rest) = url.strip_prefix(prefix) {
            let rewritten = format!("{replacement}{rest}");
            log::debug!("GOPEN_REWRITE {url} -> {rewritten}");
            return Ok(rewritten);
        }
    }
    Ok(url.to_string())
}

#[cfg(feature = "subprocess")]
/// Run a shell command and read its output.
struct PipeScheme;

#[cfg(feature = "subprocess")]
impl SchemeHandler for PipeScheme {
    fn open_read(&self, url: &str) -> Result<Box<dyn Fetch>> {
        Ok(Box::new(Pipe::read(command_for(url)?)?))
    }

    #[cfg(feature = "subprocess")]
    fn open_write(&self, url: &str) -> Result<Box<dyn Sink>> {
        Ok(Box::new(Pipe::write(command_for(url)?)?))
    }
}

#[cfg(feature = "subprocess")]
fn command_for(url: &str) -> Result<Command> {
    check_security("opening pipe: URLs")?;
    let script = url.strip_prefix("pipe:").ok_or_else(|| Error::value(format!("{url} is not a pipe: URL")))?;
    Ok(shell(script))
}

#[cfg(feature = "subprocess")]
/// Transfer with an external command, e.g. `curl` or `gsutil`.
struct CommandScheme {
    /// Builds the argv for reading `url`.
    read: fn(&str) -> Vec<String>,
    /// Builds the argv for writing `url`, when writing is supported.
    write: Option<fn(&str) -> Vec<String>>,
    /// Exit statuses to accept when reading.
    read_ok: &'static [i32],
    /// Exit statuses to accept when writing.
    write_ok: &'static [i32],
}

#[cfg(feature = "subprocess")]
impl SchemeHandler for CommandScheme {
    fn open_read(&self, url: &str) -> Result<Box<dyn Fetch>> {
        let argv = (self.read)(url);
        Ok(Box::new(Pipe::read(argv_command(&argv))?.ignore_status(self.read_ok)))
    }

    #[cfg(feature = "subprocess")]
    fn open_write(&self, url: &str) -> Result<Box<dyn Sink>> {
        let Some(build) = self.write else {
            return Err(Error::unsupported(format!("writing to {url}")));
        };
        let argv = build(url);
        Ok(Box::new(Pipe::write(argv_command(&argv))?.ignore_status(self.write_ok)))
    }
}

#[cfg(feature = "subprocess")]
fn argv_command(argv: &[String]) -> Command {
    let mut command = Command::new(&argv[0]);
    command.args(&argv[1..]);
    command
}

#[cfg(feature = "subprocess")]
fn curl_read(url: &str) -> Vec<String> {
    ["curl", "--connect-timeout", "30", "--retry", "30", "--retry-delay", "2", "-f", "-s", "-L", url]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[cfg(feature = "subprocess")]
fn curl_write(url: &str) -> Vec<String> {
    ["curl", "-f", "-s", "-X", "PUT", "-L", "-T", "-", url].iter().map(|s| s.to_string()).collect()
}

#[cfg(feature = "subprocess")]
fn htgs_read(url: &str) -> Vec<String> {
    // `htgs:` is Google Cloud Storage read over plain HTTPS.
    let rewritten = format!("https://storage.googleapis.com/{}", url.trim_start_matches("htgs://"));
    ["curl", "-s", "-L", &rewritten].iter().map(|s| s.to_string()).collect()
}

#[cfg(feature = "subprocess")]
fn gsutil_read(url: &str) -> Vec<String> {
    ["gsutil", "cat", url].iter().map(|s| s.to_string()).collect()
}

#[cfg(feature = "subprocess")]
fn gsutil_write(url: &str) -> Vec<String> {
    ["gsutil", "cp", "-", url].iter().map(|s| s.to_string()).collect()
}

#[cfg(feature = "subprocess")]
fn ais_read(url: &str) -> Vec<String> {
    ["ais", "get", url, "-"].iter().map(|s| s.to_string()).collect()
}

#[cfg(feature = "subprocess")]
fn ais_write(url: &str) -> Vec<String> {
    ["ais", "put", "-", url].iter().map(|s| s.to_string()).collect()
}

#[cfg(feature = "subprocess")]
/// Read from the Hugging Face Hub over HTTPS.
struct HuggingFaceScheme;

#[cfg(feature = "subprocess")]
impl SchemeHandler for HuggingFaceScheme {
    fn open_read(&self, url: &str) -> Result<Box<dyn Fetch>> {
        let resolved = resolve_hf_url(url)?;
        let mut argv = curl_read(&resolved);
        if let Some(token) = hf_token() {
            argv.insert(argv.len() - 1, "-H".to_string());
            argv.insert(argv.len() - 1, format!("Authorization: Bearer {token}"));
        }
        Ok(Box::new(Pipe::read(argv_command(&argv))?.ignore_status(&[141, 23])))
    }
}

#[cfg(feature = "subprocess")]
fn hf_token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"].iter().find_map(|name| std::env::var(name).ok())
}

#[cfg(feature = "subprocess")]
/// Turn `hf://datasets/owner/name/path` into a `resolve` URL on huggingface.co.
///
/// A `@revision` suffix on the repository selects a branch or commit; without
/// one, `main` is used.
pub fn resolve_hf_url(url: &str) -> Result<String> {
    let rest = url.strip_prefix("hf://").ok_or_else(|| Error::value(format!("{url} is not an hf:// URL")))?;
    let (kind, rest) = match rest.split_once('/') {
        Some(("datasets", tail)) => ("datasets/", tail),
        Some(("spaces", tail)) => ("spaces/", tail),
        _ => ("", rest),
    };
    let mut parts = rest.splitn(3, '/');
    let owner = parts.next().filter(|s| !s.is_empty());
    let name = parts.next().filter(|s| !s.is_empty());
    let path = parts.next().filter(|s| !s.is_empty());
    let (Some(owner), Some(name), Some(path)) = (owner, name, path) else {
        return Err(Error::value(format!("{url}: expected hf://[datasets/]owner/name/path")));
    };
    let (name, revision) = match name.split_once('@') {
        Some((name, revision)) => (name, revision),
        None => (name, "main"),
    };
    Ok(format!("https://huggingface.co/{kind}{owner}/{name}/resolve/{revision}/{path}"))
}

/// The scheme table the library starts with.
///
/// Without the `subprocess` feature the table is empty: only the local
/// filesystem, which [`gopen`] handles directly, and whatever
/// [`register_scheme`] adds.
fn default_schemes() -> HashMap<String, Arc<dyn SchemeHandler>> {
    #[allow(unused_mut)]
    let mut table: HashMap<String, Arc<dyn SchemeHandler>> = HashMap::new();

    #[cfg(feature = "subprocess")]
    {
        let curl: Arc<dyn SchemeHandler> = Arc::new(CommandScheme {
            read: curl_read,
            write: Some(curl_write),
            read_ok: &[141, 23],
            write_ok: &[141, 26],
        });
        let gsutil: Arc<dyn SchemeHandler> = Arc::new(CommandScheme {
            read: gsutil_read,
            write: Some(gsutil_write),
            read_ok: &[141, 23],
            write_ok: &[141, 26],
        });
        let ais: Arc<dyn SchemeHandler> = Arc::new(CommandScheme {
            read: ais_read,
            write: Some(ais_write),
            read_ok: &[141, 23],
            write_ok: &[141, 26],
        });
        let htgs: Arc<dyn SchemeHandler> =
            Arc::new(CommandScheme { read: htgs_read, write: None, read_ok: &[141, 23], write_ok: &[] });

        table.insert("pipe".into(), Arc::new(PipeScheme));
        for scheme in ["http", "https", "sftp", "ftps", "ftp", "scp"] {
            table.insert(scheme.into(), curl.clone());
        }
        table.insert("gs".into(), gsutil);
        table.insert("htgs".into(), htgs);
        table.insert("ais".into(), ais.clone());
        table.insert("hf".into(), Arc::new(HuggingFaceScheme));

        // `USE_AIS_FOR=gs:s3` routes those schemes through the AIStore cache.
        if let Ok(schemes) = std::env::var("USE_AIS_FOR") {
            for scheme in schemes.split(':').filter(|s| !s.is_empty()) {
                table.insert(scheme.into(), ais.clone());
            }
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_local_files_without_a_scheme() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, b"contents").unwrap();

        let mut out = String::new();
        gopen(path.to_str().unwrap()).unwrap().read_to_string(&mut out).unwrap();
        assert_eq!(out, "contents");

        let mut out = String::new();
        gopen(&format!("file://{}", path.display())).unwrap().read_to_string(&mut out).unwrap();
        assert_eq!(out, "contents");
    }

    #[test]
    fn writes_local_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let mut sink = gopen_write(path.to_str().unwrap()).unwrap();
        sink.write_all(b"written").unwrap();
        sink.finish().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "written");
    }

    #[cfg(feature = "subprocess")]
    #[test]
    fn runs_pipe_urls() {
        let mut out = String::new();
        gopen("pipe:printf piped").unwrap().read_to_string(&mut out).unwrap();
        assert_eq!(out, "piped");
    }

    #[test]
    fn rejects_unknown_schemes() {
        let Err(err) = gopen("nosuchscheme://host/a.tar") else {
            panic!("an unknown scheme should not open");
        };
        assert!(matches!(err, Error::Unsupported(_)), "{err}");
    }

    #[test]
    fn reports_missing_files_with_context() {
        let Err(err) = gopen("/no/such/file/anywhere.tar") else {
            panic!("a missing file should not open");
        };
        assert!(err.to_string().contains("anywhere.tar"), "{err}");
    }

    #[cfg(feature = "subprocess")]
    #[test]
    fn registers_the_expected_schemes() {
        let schemes = registered_schemes();
        for expected in ["pipe", "http", "https", "gs", "ais", "hf", "htgs"] {
            assert!(schemes.contains(&expected.to_string()), "missing {expected} in {schemes:?}");
        }
    }

    #[cfg(feature = "subprocess")]
    #[test]
    fn resolves_hugging_face_urls() {
        assert_eq!(
            resolve_hf_url("hf://datasets/owner/set/shard-000.tar").unwrap(),
            "https://huggingface.co/datasets/owner/set/resolve/main/shard-000.tar"
        );
        assert_eq!(
            resolve_hf_url("hf://datasets/owner/set@v2/dir/shard.tar").unwrap(),
            "https://huggingface.co/datasets/owner/set/resolve/v2/dir/shard.tar"
        );
        assert!(resolve_hf_url("hf://owner").is_err());
    }

    #[cfg(feature = "subprocess")]
    #[test]
    fn builds_the_documented_curl_invocation() {
        let argv = curl_read("https://host/a.tar");
        assert_eq!(argv[0], "curl");
        assert!(argv.contains(&"-L".to_string()));
        assert_eq!(argv.last().unwrap(), "https://host/a.tar");
    }
}
