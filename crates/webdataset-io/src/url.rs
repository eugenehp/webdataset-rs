//! Minimal URL handling: enough to pick a scheme handler and recover a path.
//!
//! WebDataset URLs are frequently not URLs at all — `pipe:curl -s http://...`
//! and bare relative paths are both common — so a full URL parser would reject
//! inputs the format expects to work. This module does only what the dispatcher
//! needs.

/// The scheme of a URL, or `None` for a bare path.
///
/// ```
/// use webdataset_io::url::scheme;
///
/// assert_eq!(scheme("https://host/a.tar"), Some("https"));
/// assert_eq!(scheme("pipe:curl -s http://host/a.tar"), Some("pipe"));
/// assert_eq!(scheme("data/a.tar"), None);
/// assert_eq!(scheme("/abs/a.tar"), None);
/// ```
pub fn scheme(url: &str) -> Option<&str> {
    let colon = url.find(':')?;
    let candidate = &url[..colon];
    let mut chars = candidate.chars();
    let first = chars.next()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.') {
        return None;
    }
    Some(candidate)
}

/// The path component of a URL: everything after the authority, minus query
/// and fragment. For bare paths the input is returned unchanged.
///
/// ```
/// use webdataset_io::url::path;
///
/// assert_eq!(path("https://host/dir/a.tar?x=1"), "/dir/a.tar");
/// assert_eq!(path("file:///dir/a.tar"), "/dir/a.tar");
/// assert_eq!(path("data/a.tar"), "data/a.tar");
/// ```
pub fn path(url: &str) -> &str {
    let rest = match scheme(url) {
        Some(s) => &url[s.len() + 1..],
        None => url,
    };
    let rest = match rest.strip_prefix("//") {
        Some(after_slashes) => match after_slashes.find('/') {
            Some(at) => &after_slashes[at..],
            None => "",
        },
        None => rest,
    };
    let rest = rest.split('#').next().unwrap_or(rest);
    rest.split('?').next().unwrap_or(rest)
}

/// The local filesystem path a `file:` URL or bare path refers to.
///
/// Percent escapes are decoded, so `file:///tmp/a%20b.tar` becomes
/// `/tmp/a b.tar`.
pub fn to_local_path(url: &str) -> String {
    let raw = match scheme(url) {
        Some("file") => path(url),
        Some(_) | None => url,
    };
    percent_decode(raw)
}

/// Whether this URL names something on the local filesystem.
pub fn is_local(url: &str) -> bool {
    matches!(scheme(url), None | Some("file"))
}

/// Decode `%XX` escapes; anything malformed is passed through untouched.
pub fn percent_decode(text: &str) -> String {
    if !text.contains('%') {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok());
            if let Some(byte) = hex {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Percent-encode everything except unreserved characters and `safe`.
pub fn percent_encode(text: &str, safe: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        let c = byte as char;
        if c.is_ascii_alphanumeric() || safe.contains(c) {
            out.push(c);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// The last component of a URL's path.
pub fn basename(url: &str) -> &str {
    let p = path(url);
    match p.rfind('/') {
        Some(at) => &p[at + 1..],
        None => p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_schemes() {
        assert_eq!(scheme("gs://bucket/a.tar"), Some("gs"));
        assert_eq!(scheme("pipe:cat a.tar"), Some("pipe"));
        assert_eq!(scheme("a.tar"), None);
        assert_eq!(scheme("./a:b.tar"), None, "a leading dot is not a scheme character");
        assert_eq!(scheme("-"), None);
    }

    #[test]
    fn extracts_paths() {
        assert_eq!(path("gs://bucket/dir/a.tar"), "/dir/a.tar");
        assert_eq!(path("https://host/a.tar#frag"), "/a.tar");
        assert_eq!(path("https://host"), "");
        assert_eq!(path("/abs/a.tar"), "/abs/a.tar");
    }

    #[test]
    fn decodes_file_urls() {
        assert_eq!(to_local_path("file:///tmp/a%20b.tar"), "/tmp/a b.tar");
        assert_eq!(to_local_path("testdata/a.tar"), "testdata/a.tar");
        assert!(is_local("file:///x"));
        assert!(is_local("x/y.tar"));
        assert!(!is_local("https://host/x.tar"));
    }

    #[test]
    fn round_trips_percent_escapes() {
        let original = "a b/c:d";
        let encoded = percent_encode(original, "_-");
        assert_eq!(percent_decode(&encoded), original);
    }

    #[test]
    fn takes_basenames() {
        assert_eq!(basename("https://host/dir/a.tar"), "a.tar");
        assert_eq!(basename("a.tar"), "a.tar");
    }
}
