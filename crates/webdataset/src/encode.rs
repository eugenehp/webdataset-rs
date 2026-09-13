//! Turning values back into the bytes stored in a shard.
//!
//! [`DefaultEncoder`] is the inverse of [`Decoder`](crate::decode::Decoder):
//! the extension a value is stored under decides how it is serialised, so
//! writing `{"cls": Value::Int(7), "json": …}` produces a `.cls` file holding
//! `7` and a `.json` file holding JSON.
//!
//! ```
//! use std::sync::Arc;
//! use webdataset::encode::DefaultEncoder;
//! use webdataset_core::{Sample, Value};
//! use webdataset_shard::{Encoder, TarWriter};
//!
//! let encoder = DefaultEncoder::new();
//! assert_eq!(encoder.encode("cls", &Value::Int(7))?.as_ref(), b"7");
//! assert_eq!(encoder.encode("txt", &Value::Text("hi".into()))?.as_ref(), b"hi");
//!
//! let dir = tempfile::tempdir()?;
//! let path = dir.path().join("out.tar");
//! let mut writer = TarWriter::create(path.to_str().unwrap())?.with_encoder(Arc::new(encoder));
//! let mut sample = Sample::with_key("k");
//! sample.insert("cls", Value::Int(7));
//! writer.write(&sample)?;
//! writer.close()?;
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::io::Write;

use bytes::Bytes;
use webdataset_core::error::{Error, Result};
use webdataset_core::npy;
use webdataset_core::value::Value;
use webdataset_shard::writer::Encoder;

/// Encodes values according to the extension they are stored under.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultEncoder {
    /// Whether a `.gz` suffix compresses the encoded bytes.
    compress_gz: bool,
}

impl DefaultEncoder {
    /// An encoder that gzips fields whose extension ends in `.gz`.
    pub fn new() -> DefaultEncoder {
        DefaultEncoder { compress_gz: true }
    }

    /// Store `.gz` fields verbatim instead of compressing them.
    pub fn without_gz_compression(mut self) -> DefaultEncoder {
        self.compress_gz = false;
        self
    }
}

impl Encoder for DefaultEncoder {
    fn encode(&self, extension: &str, value: &Value) -> Result<Bytes> {
        // Metadata fields are written as-is, and must already be text.
        if extension.starts_with('_') {
            return match value {
                Value::Text(text) => Ok(Bytes::from(text.clone().into_bytes())),
                Value::Bytes(bytes) => Ok(bytes.clone()),
                other => Err(Error::encode(extension, format!("metadata must be text, not {}", other.type_name()))),
            };
        }

        let (inner, compress) = match extension.strip_suffix(".gz") {
            Some(inner) if self.compress_gz => (inner, true),
            _ => (extension, false),
        };

        let encoded = encode_value(inner, value)?;
        if !compress {
            return Ok(encoded);
        }
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&encoded).map_err(|e| Error::encode(extension, e))?;
        Ok(Bytes::from(encoder.finish().map_err(|e| Error::encode(extension, e))?))
    }
}

/// Serialise `value` for the given extension.
fn encode_value(extension: &str, value: &Value) -> Result<Bytes> {
    // Raw bytes are always stored verbatim, whatever the extension says.
    if let Value::Bytes(bytes) = value {
        return Ok(bytes.clone());
    }

    let last = extension.rsplit('.').next().unwrap_or(extension).to_ascii_lowercase();
    let bytes = match last.as_str() {
        "txt" | "text" | "transcript" | "html" | "htm" => match value {
            Value::Text(text) => text.clone().into_bytes(),
            other => return Err(Error::encode(extension, format!("expected text, found {}", other.type_name()))),
        },
        "cls" | "cls2" | "class" | "count" | "index" | "inx" | "id" => {
            let n = value
                .as_i64()
                .ok_or_else(|| Error::encode(extension, format!("expected an integer, found {}", value.type_name())))?;
            n.to_string().into_bytes()
        }
        "json" | "jsn" => serde_json::to_vec(&value.to_json()?).map_err(|e| Error::encode(extension, e))?,
        "npy" => match value.as_tensor() {
            Some(tensor) => npy::to_npy(tensor),
            None => {
                return Err(Error::encode(extension, format!("expected a tensor, found {}", value.type_name())));
            }
        },
        "npz" => encode_npz(extension, value)?,
        "ten" | "tenbin" | "tb" => {
            let tensors = match value {
                Value::Tensor(t) => vec![t.clone()],
                Value::List(items) => items
                    .iter()
                    .map(|v| {
                        v.as_tensor().cloned().ok_or_else(|| {
                            Error::encode(extension, format!("expected tensors, found {}", v.type_name()))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                other => {
                    return Err(Error::encode(
                        extension,
                        format!("expected a tensor or a list of tensors, found {}", other.type_name()),
                    ));
                }
            };
            webdataset_tenbin::encode_buffer(&tensors)
        }
        "mp" | "msg" | "msgpack" => encode_msgpack(extension, value)?,
        "cbor" => encode_cbor(extension, value)?,
        "pyd" | "pickle" | "pkl" | "pth" => {
            return Err(Error::unsupported(format!(
                "{extension}: Python pickles are not written; store the field as .json, .npy or .ten"
            )));
        }
        _ => return encode_fallback(extension, value),
    };
    Ok(Bytes::from(bytes))
}

/// Handle the extensions that depend on optional features, or give up.
fn encode_fallback(extension: &str, value: &Value) -> Result<Bytes> {
    #[cfg(feature = "image")]
    {
        if let Ok(bytes) = crate::images::encode(extension, value) {
            return Ok(Bytes::from(bytes));
        }
    }
    match value {
        Value::Text(text) => Ok(Bytes::from(text.clone().into_bytes())),
        other => Err(Error::encode(
            extension,
            format!("no encoder for this extension, and {} is not raw bytes", other.type_name()),
        )),
    }
}

#[cfg(feature = "npz")]
fn encode_npz(extension: &str, value: &Value) -> Result<Vec<u8>> {
    use zip::write::SimpleFileOptions;

    let map = value
        .as_map()
        .ok_or_else(|| Error::encode(extension, format!("expected a map of tensors, found {}", value.type_name())))?;

    let mut out = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, entry) in map {
        let tensor = entry
            .as_tensor()
            .ok_or_else(|| Error::encode(extension, format!("{name} is {}, not a tensor", entry.type_name())))?;
        out.start_file(format!("{name}.npy"), options).map_err(|e| Error::encode(extension, e))?;
        out.write_all(&npy::to_npy(tensor)).map_err(|e| Error::encode(extension, e))?;
    }
    Ok(out.finish().map_err(|e| Error::encode(extension, e))?.into_inner())
}

#[cfg(not(feature = "npz"))]
fn encode_npz(extension: &str, _value: &Value) -> Result<Vec<u8>> {
    Err(Error::unsupported(format!("{extension}: rebuild webdataset with the `npz` feature to write .npz")))
}

#[cfg(feature = "msgpack")]
fn encode_msgpack(extension: &str, value: &Value) -> Result<Vec<u8>> {
    rmp_serde::to_vec_named(&value.to_json()?).map_err(|e| Error::encode(extension, e))
}

#[cfg(not(feature = "msgpack"))]
fn encode_msgpack(extension: &str, _value: &Value) -> Result<Vec<u8>> {
    Err(Error::unsupported(format!("{extension}: rebuild webdataset with the `msgpack` feature to write .mp")))
}

#[cfg(feature = "cbor")]
fn encode_cbor(extension: &str, value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    ciborium::into_writer(&value.to_json()?, &mut out).map_err(|e| Error::encode(extension, format!("cbor: {e}")))?;
    Ok(out)
}

#[cfg(not(feature = "cbor"))]
fn encode_cbor(extension: &str, _value: &Value) -> Result<Vec<u8>> {
    Err(Error::unsupported(format!("{extension}: rebuild webdataset with the `cbor` feature to write .cbor")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::Decoder;
    use webdataset_core::tensor::Tensor;

    fn encode(extension: &str, value: &Value) -> Result<Bytes> {
        DefaultEncoder::new().encode(extension, value)
    }

    /// Encode then decode, which should return the value unchanged.
    fn round_trip(extension: &str, value: Value) {
        let encoded = encode(extension, &value).unwrap_or_else(|e| panic!("encoding {extension}: {e}"));
        let decoded = Decoder::default()
            .decode_field(extension, &encoded)
            .unwrap_or_else(|e| panic!("decoding {extension}: {e}"));
        assert_eq!(decoded, value, "{extension} did not survive a round trip");
    }

    #[test]
    fn round_trips_the_basic_extensions() {
        round_trip("txt", Value::Text("some text".into()));
        round_trip("cls", Value::Int(42));
        round_trip("json", Value::List(vec![Value::Int(1), Value::Text("two".into())]));
        round_trip("npy", Value::Tensor(Tensor::from_f32_shaped(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])));
        round_trip("ten", Value::List(vec![Value::Tensor(Tensor::from_f32(&[1.0, 2.0]))]));
    }

    #[test]
    fn round_trips_through_gzip() {
        round_trip("txt.gz", Value::Text("compressible".into()));
        let raw = encode("txt.gz", &Value::Text("compressible".into())).unwrap();
        assert_eq!(&raw[..2], &[0x1f, 0x8b]);
    }

    #[test]
    fn stores_raw_bytes_verbatim() {
        let bytes = Value::Bytes(Bytes::from_static(b"\x00\xff raw"));
        assert_eq!(encode("anything", &bytes).unwrap().as_ref(), b"\x00\xff raw");
        assert_eq!(encode("json", &bytes).unwrap().as_ref(), b"\x00\xff raw");
    }

    #[test]
    fn writes_metadata_as_text() {
        assert_eq!(encode("_meta", &Value::Text("v".into())).unwrap().as_ref(), b"v");
        assert!(encode("_meta", &Value::Int(1)).is_err());
    }

    #[test]
    fn reports_type_mismatches() {
        let err = encode("cls", &Value::Text("not a number".into())).unwrap_err();
        assert!(err.to_string().contains("cls"), "{err}");
        assert!(encode("npy", &Value::Int(1)).is_err());
        assert!(encode("pth", &Value::Int(1)).is_err());
    }

    #[test]
    fn encodes_a_single_tensor_as_a_one_element_tenbin() {
        let tensor = Tensor::from_f32(&[1.0, 2.0]);
        let encoded = encode("ten", &Value::Tensor(tensor.clone())).unwrap();
        assert_eq!(webdataset_tenbin::decode_buffer(&encoded).unwrap(), vec![tensor]);
    }

    #[cfg(feature = "msgpack")]
    #[test]
    fn round_trips_msgpack() {
        round_trip("mp", Value::List(vec![Value::Int(1), Value::Int(2)]));
    }

    #[cfg(feature = "cbor")]
    #[test]
    fn round_trips_cbor() {
        round_trip("cbor", Value::Text("cbor text".into()));
    }

    #[cfg(feature = "npz")]
    #[test]
    fn round_trips_npz() {
        let mut fields = webdataset_core::fields::Fields::default();
        fields.insert("a".to_string(), Value::Tensor(Tensor::from_f32(&[1.0, 2.0])));
        fields.insert("b".to_string(), Value::Tensor(Tensor::from_f32_shaped(&[3.0, 4.0], vec![1, 2])));
        round_trip("npz", Value::Map(fields));
    }
}
