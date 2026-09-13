//! Turning raw file bytes into usable values.
//!
//! A sample arrives from the archive as a map of extension to bytes. The
//! [`Decoder`] walks a chain of handlers for each field, stopping at the first
//! one that claims it. Handlers may also rewrite the field and hand it on —
//! that is how `.json.gz` is decompressed and then parsed as JSON.
//!
//! The default chain is:
//!
//! 1. [`GzFilter`] — decompress `.gz` and continue with the remaining extension.
//! 2. any handlers the caller supplied, in order.
//! 3. [`BasicHandlers`] — `.txt`, `.cls`, `.json`, `.npy`, `.ten`, and friends.
//!
//! ```
//! use webdataset::decode::Decoder;
//! use webdataset_core::{Sample, Value};
//!
//! let mut sample = Sample::with_key("k");
//! sample.insert("cls", Value::Bytes("7".into()));
//! sample.insert("json", Value::Bytes(br#"{"n": 1}"#.as_slice().into()));
//!
//! let decoded = Decoder::default().decode(sample)?;
//! assert_eq!(decoded.get("cls").unwrap().as_i64(), Some(7));
//! assert!(decoded.get("json").unwrap().as_map().is_some());
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;

use webdataset_core::error::{Error, Result};
use webdataset_core::npy;
use webdataset_core::sample::{Sample, is_meta};
use webdataset_core::value::Value;

/// What a handler decided about a field.
#[derive(Debug)]
pub enum Decoded {
    /// Not mine; try the next handler.
    Skipped,
    /// Decoded to this value.
    Value(Value),
    /// Rewritten; keep decoding with this extension and these bytes.
    Continue {
        /// The remaining extension, e.g. `json` after stripping `.gz`.
        key: String,
        /// The rewritten bytes.
        data: Vec<u8>,
    },
}

/// Decodes one field of a sample.
pub trait DecodeHandler: Send + Sync + std::fmt::Debug {
    /// Try to decode `data`, which was stored under the extension `key`.
    fn decode(&self, key: &str, data: &[u8]) -> Result<Decoded>;
}

/// The last extension component, lowercased: `a.seg.PNG` gives `png`.
fn last_extension(key: &str) -> String {
    key.rsplit('.').next().unwrap_or(key).to_ascii_lowercase()
}

/// Decompresses `.gz` fields and hands the rest of the name on.
#[derive(Debug, Clone, Copy, Default)]
pub struct GzFilter;

impl DecodeHandler for GzFilter {
    fn decode(&self, key: &str, data: &[u8]) -> Result<Decoded> {
        let Some(stripped) = key.strip_suffix(".gz").or_else(|| (key == "gz").then_some("")) else {
            return Ok(Decoded::Skipped);
        };
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(data)
            .read_to_end(&mut out)
            .map_err(|e| Error::decode(key, format!("gzip: {e}")))?;
        Ok(Decoded::Continue { key: stripped.to_string(), data: out })
    }
}

/// The standard extension-to-value mapping.
///
/// | extension | becomes |
/// |---|---|
/// | `txt`, `text`, `transcript` | [`Value::Text`] |
/// | `cls`, `cls2`, `class`, `count`, `index`, `inx`, `id` | [`Value::Int`] |
/// | `json`, `jsn` | parsed JSON |
/// | `npy` | [`Value::Tensor`] |
/// | `npz` | a map of tensors (feature `npz`) |
/// | `ten`, `tb` | a list of tensors |
/// | `mp`, `msg` | MessagePack (feature `msgpack`) |
/// | `cbor` | CBOR (feature `cbor`) |
///
/// Anything else is left as bytes. Python's pickle-based extensions (`pkl`,
/// `pyd`, `pth`) are reported as unsupported rather than silently ignored,
/// since executing pickles is not something this library will do.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicHandlers;

impl DecodeHandler for BasicHandlers {
    fn decode(&self, key: &str, data: &[u8]) -> Result<Decoded> {
        let value = match last_extension(key).as_str() {
            "txt" | "text" | "transcript" => {
                Value::Text(std::str::from_utf8(data).map_err(|e| Error::decode(key, e))?.to_string())
            }
            "cls" | "cls2" | "class" | "count" | "index" | "inx" | "id" => {
                let text = std::str::from_utf8(data).map_err(|e| Error::decode(key, e))?;
                Value::Int(text.trim().parse::<i64>().map_err(|e| Error::decode(key, e))?)
            }
            "json" | "jsn" => {
                Value::from(serde_json::from_slice::<serde_json::Value>(data).map_err(|e| Error::decode(key, e))?)
            }
            "npy" => Value::Tensor(npy::from_npy(data).map_err(|e| Error::decode(key, e))?),
            "npz" => decode_npz(key, data)?,
            "ten" | "tb" | "tenbin" => Value::List(
                webdataset_tenbin::decode_buffer(data)
                    .map_err(|e| Error::decode(key, e))?
                    .into_iter()
                    .map(Value::Tensor)
                    .collect(),
            ),
            "mp" | "msg" | "msgpack" => decode_msgpack(key, data)?,
            "cbor" => decode_cbor(key, data)?,
            "pkl" | "pickle" | "pyd" | "pth" => {
                return Err(Error::unsupported(format!(
                    "{key}: Python pickles are not decoded; re-export the field as .json, .npy or .ten"
                )));
            }
            _ => return Ok(Decoded::Skipped),
        };
        Ok(Decoded::Value(value))
    }
}

#[cfg(feature = "npz")]
fn decode_npz(key: &str, data: &[u8]) -> Result<Value> {
    use webdataset_core::fields::Fields;
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(data)).map_err(|e| Error::decode(key, format!("npz: {e}")))?;
    let mut fields = Fields::default();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| Error::decode(key, format!("npz: {e}")))?;
        let name = entry.name().trim_end_matches(".npy").to_string();
        let mut body = Vec::new();
        entry.read_to_end(&mut body).map_err(|e| Error::decode(key, format!("npz: {e}")))?;
        fields.insert(name, Value::Tensor(npy::from_npy(&body).map_err(|e| Error::decode(key, e))?));
    }
    Ok(Value::Map(fields))
}

#[cfg(not(feature = "npz"))]
fn decode_npz(key: &str, _data: &[u8]) -> Result<Value> {
    Err(Error::unsupported(format!("{key}: rebuild webdataset with the `npz` feature to decode .npz")))
}

#[cfg(feature = "msgpack")]
fn decode_msgpack(key: &str, data: &[u8]) -> Result<Value> {
    let json: serde_json::Value = rmp_serde::from_slice(data).map_err(|e| Error::decode(key, e))?;
    Ok(Value::from(json))
}

#[cfg(not(feature = "msgpack"))]
fn decode_msgpack(key: &str, _data: &[u8]) -> Result<Value> {
    Err(Error::unsupported(format!("{key}: rebuild webdataset with the `msgpack` feature to decode .mp")))
}

#[cfg(feature = "cbor")]
fn decode_cbor(key: &str, data: &[u8]) -> Result<Value> {
    let json: serde_json::Value = ciborium::from_reader(data).map_err(|e| Error::decode(key, format!("cbor: {e}")))?;
    Ok(Value::from(json))
}

#[cfg(not(feature = "cbor"))]
fn decode_cbor(key: &str, _data: &[u8]) -> Result<Value> {
    Err(Error::unsupported(format!("{key}: rebuild webdataset with the `cbor` feature to decode .cbor")))
}

/// A handler that fires for a fixed set of extensions.
///
/// Extensions may contain dots, in which case that many components of the field
/// name must match: `seg.png` fires for `mask.seg.png` but not for `mask.png`.
pub struct ExtensionHandler<F> {
    extensions: Vec<Vec<String>>,
    decode: F,
}

impl<F> std::fmt::Debug for ExtensionHandler<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionHandler").field("extensions", &self.extensions).finish_non_exhaustive()
    }
}

impl<F> ExtensionHandler<F>
where
    F: Fn(&[u8]) -> Result<Value> + Send + Sync,
{
    /// Fire `decode` for any of the space-separated `extensions`.
    pub fn new(extensions: &str, decode: F) -> ExtensionHandler<F> {
        let extensions = extensions
            .to_ascii_lowercase()
            .split_whitespace()
            .map(|e| e.split('.').map(str::to_string).collect())
            .collect();
        ExtensionHandler { extensions, decode }
    }
}

impl<F> DecodeHandler for ExtensionHandler<F>
where
    F: Fn(&[u8]) -> Result<Value> + Send + Sync,
{
    fn decode(&self, key: &str, data: &[u8]) -> Result<Decoded> {
        let parts: Vec<String> = key.to_ascii_lowercase().split('.').map(str::to_string).collect();
        for target in &self.extensions {
            if target.len() <= parts.len() && parts[parts.len() - target.len()..] == target[..] {
                return Ok(Decoded::Value((self.decode)(data)?));
            }
        }
        Ok(Decoded::Skipped)
    }
}

/// Decodes every field of a sample.
#[derive(Debug, Clone)]
pub struct Decoder {
    handlers: Vec<Arc<dyn DecodeHandler>>,
    only: Option<HashSet<String>>,
    partial: bool,
}

impl Default for Decoder {
    fn default() -> Decoder {
        Decoder::new(Vec::new())
    }
}

impl Decoder {
    /// A decoder running `handlers` between the default pre- and post-handlers.
    pub fn new(handlers: Vec<Arc<dyn DecodeHandler>>) -> Decoder {
        let mut chain: Vec<Arc<dyn DecodeHandler>> = vec![Arc::new(GzFilter)];
        chain.extend(handlers);
        chain.push(Arc::new(BasicHandlers));
        Decoder { handlers: chain, only: None, partial: false }
    }

    /// A decoder with an exact handler chain and no defaults.
    pub fn raw(handlers: Vec<Arc<dyn DecodeHandler>>) -> Decoder {
        Decoder { handlers, only: None, partial: false }
    }

    /// Append a handler, which runs before the default post-handlers.
    pub fn with(mut self, handler: impl DecodeHandler + 'static) -> Decoder {
        let at = self.handlers.len().saturating_sub(1);
        self.handlers.insert(at, Arc::new(handler));
        self
    }

    /// Decode only these fields, leaving the others as they are.
    pub fn only(mut self, fields: impl IntoIterator<Item = impl Into<String>>) -> Decoder {
        self.only = Some(fields.into_iter().map(Into::into).collect());
        self
    }

    /// Leave already-decoded fields alone instead of failing on them.
    ///
    /// Use this when decoding runs twice over the same stream.
    pub fn partial(mut self, partial: bool) -> Decoder {
        self.partial = partial;
        self
    }

    /// Decode a single field.
    pub fn decode_field(&self, key: &str, data: &[u8]) -> Result<Value> {
        let mut key = key.to_string();
        let mut data = data.to_vec();
        // A handler may rewrite the field and ask to continue; bound the number
        // of rewrites so a misbehaving handler cannot loop forever.
        for _ in 0..16 {
            let mut rewritten = None;
            for handler in &self.handlers {
                match handler.decode(&key, &data)? {
                    Decoded::Skipped => continue,
                    Decoded::Value(value) => return Ok(value),
                    Decoded::Continue { key: k, data: d } => {
                        rewritten = Some((k, d));
                        break;
                    }
                }
            }
            match rewritten {
                Some((k, d)) => {
                    key = k;
                    data = d;
                }
                None => return Ok(Value::Bytes(data.into())),
            }
        }
        Err(Error::decode(key, "handlers kept rewriting the field"))
    }

    /// Decode every field of `sample`.
    ///
    /// Metadata fields are decoded as UTF-8 text when possible and otherwise
    /// left untouched.
    pub fn decode(&self, sample: Sample) -> Result<Sample> {
        let mut out = Sample::new();
        for (name, value) in sample {
            if is_meta(&name) {
                out.insert(name, decode_metadata(value));
                continue;
            }
            if self.only.as_ref().is_some_and(|only| !only.contains(&name)) {
                out.insert(name, value);
                continue;
            }
            match value {
                Value::Bytes(data) => {
                    let decoded = self.decode_field(&name, &data)?;
                    out.insert(name, decoded);
                }
                other if self.partial => {
                    out.insert(name, other);
                }
                other => {
                    return Err(Error::decode(&name, format!("expected raw bytes, found {}", other.type_name())));
                }
            }
        }
        Ok(out)
    }
}

/// Metadata is usually a URL or a key, so present it as text when it is one.
fn decode_metadata(value: Value) -> Value {
    match value {
        Value::Bytes(data) => match std::str::from_utf8(&data) {
            Ok(text) => Value::Text(text.to_string()),
            Err(_) => Value::Bytes(data),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn field(key: &str, data: &[u8]) -> Result<Value> {
        Decoder::default().decode_field(key, data)
    }

    #[test]
    fn decodes_the_basic_extensions() {
        assert_eq!(field("txt", b"hello").unwrap(), Value::Text("hello".into()));
        assert_eq!(field("cls", b"42").unwrap(), Value::Int(42));
        assert_eq!(field("cls", b"42\n").unwrap(), Value::Int(42), "trailing newlines are common");
        assert_eq!(field("json", br#"[1, 2]"#).unwrap(), Value::List(vec![Value::Int(1), Value::Int(2)]));
    }

    #[test]
    fn leaves_unknown_extensions_as_bytes() {
        let value = field("bin", b"\x00\x01").unwrap();
        assert_eq!(value.as_bytes().unwrap().as_ref(), b"\x00\x01");
    }

    #[test]
    fn decompresses_and_then_decodes() {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(b"compressed text").unwrap();
        let gz = encoder.finish().unwrap();

        assert_eq!(field("txt.gz", &gz).unwrap(), Value::Text("compressed text".into()));
        // Without a recognised inner extension the bytes come back raw.
        assert_eq!(field("bin.gz", &gz).unwrap().as_bytes().unwrap().as_ref(), b"compressed text");
    }

    #[test]
    fn round_trips_npy_and_tenbin() {
        let tensor = webdataset_core::Tensor::from_f32_shaped(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let decoded = field("npy", &webdataset_core::npy::to_npy(&tensor)).unwrap();
        assert_eq!(decoded.as_tensor().unwrap(), &tensor);

        let decoded = field("ten", &webdataset_tenbin::encode_buffer(std::slice::from_ref(&tensor))).unwrap();
        assert_eq!(decoded.as_list().unwrap()[0].as_tensor().unwrap(), &tensor);
    }

    #[test]
    fn reports_pickles_as_unsupported() {
        let err = field("pth", b"anything").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err}");
    }

    #[test]
    fn honours_only_and_partial() {
        let mut sample = Sample::with_key("k");
        sample.insert("cls", Value::Bytes("7".into()));
        sample.insert("txt", Value::Bytes("raw".into()));

        let decoded = Decoder::default().only(["cls"]).decode(sample.clone()).unwrap();
        assert_eq!(decoded.get("cls").unwrap().as_i64(), Some(7));
        assert!(decoded.get("txt").unwrap().as_bytes().is_some(), "txt should be left alone");

        let mut already = Sample::with_key("k");
        already.insert("cls", Value::Int(7));
        assert!(Decoder::default().decode(already.clone()).is_err());
        assert!(Decoder::default().partial(true).decode(already).is_ok());
    }

    #[test]
    fn decodes_metadata_as_text() {
        let mut sample = Sample::with_key("k");
        sample.insert("__url__", Value::Bytes("http://host/s.tar".into()));
        let decoded = Decoder::default().decode(sample).unwrap();
        assert_eq!(decoded.url(), Some("http://host/s.tar"));
    }

    #[test]
    fn runs_custom_handlers_before_the_defaults() {
        let handler = ExtensionHandler::new("txt", |data| Ok(Value::Int(data.len() as i64)));
        let decoder = Decoder::new(vec![Arc::new(handler)]);
        assert_eq!(decoder.decode_field("txt", b"12345").unwrap(), Value::Int(5));
        assert_eq!(decoder.decode_field("cls", b"5").unwrap(), Value::Int(5), "defaults still apply");
    }

    #[test]
    fn matches_multi_component_extensions() {
        let handler = ExtensionHandler::new("seg.png", |_| Ok(Value::Text("segmentation".into())));
        let decoder = Decoder::new(vec![Arc::new(handler)]);
        assert_eq!(decoder.decode_field("seg.png", b"x").unwrap(), Value::Text("segmentation".into()));
        assert!(decoder.decode_field("png", b"x").unwrap().as_bytes().is_some());
    }

    #[test]
    fn reports_bad_input_with_the_field_name() {
        let err = field("cls", b"not a number").unwrap_err();
        assert!(err.to_string().contains("cls"), "{err}");
    }
}
