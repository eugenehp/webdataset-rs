//! The dynamically typed values that make up a [`Sample`](crate::Sample).
//!
//! A WebDataset sample is a bag of files, and what a file decodes to depends on
//! its extension: `.txt` becomes a string, `.cls` an integer, `.json` a tree,
//! `.npy` a tensor, `.jpg` an image. [`Value`] is the union of those
//! possibilities, plus [`Value::Custom`] as an escape hatch for decoders the
//! library does not know about.

use alloc::sync::Arc;
use core::any::Any;
use core::fmt;

use crate::fields::Fields;
use bytes::Bytes;

use crate::error::{Error, Result};
use crate::prelude::*;
use crate::tensor::Tensor;

/// A value carried by a sample field.
#[derive(Clone)]
#[non_exhaustive]
pub enum Value {
    /// Absent or JSON `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// An integer, e.g. the contents of a `.cls` file.
    Int(i64),
    /// A floating point number.
    Float(f64),
    /// Text, e.g. the contents of a `.txt` file.
    Text(String),
    /// Undecoded (or deliberately raw) file contents.
    Bytes(Bytes),
    /// An ordered sequence, e.g. a JSON array or a collated batch column.
    List(Vec<Value>),
    /// A string-keyed mapping, e.g. a JSON object or an `.npz` archive.
    Map(Fields),
    /// A dense numeric array, e.g. from `.npy` or `.ten`.
    Tensor(Tensor),
    /// A decoded image.
    #[cfg(feature = "image")]
    Image(Arc<image::DynamicImage>),
    /// Anything else a user-supplied decoder produced.
    Custom(Arc<dyn CustomValue>),
}

/// The trait objects accepted by [`Value::Custom`].
pub trait CustomValue: Any + fmt::Debug + Send + Sync {
    /// Upcast so callers can downcast to the concrete type.
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any + fmt::Debug + Send + Sync> CustomValue for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Value {
    /// Wrap an arbitrary value so it can live inside a sample.
    pub fn custom<T: CustomValue>(value: T) -> Value {
        Value::Custom(Arc::new(value))
    }

    /// A short type name, useful in error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Text(_) => "text",
            Value::Bytes(_) => "bytes",
            Value::List(_) => "list",
            Value::Map(_) => "map",
            Value::Tensor(_) => "tensor",
            #[cfg(feature = "image")]
            Value::Image(_) => "image",
            Value::Custom(_) => "custom",
        }
    }

    /// Borrow the raw bytes, if this value is undecoded.
    pub fn as_bytes(&self) -> Option<&Bytes> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// Borrow the text, if this value is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Read this value as an integer, accepting `Int`, `Bool` and whole `Float`s.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(*b as i64),
            // Whole floats convert; anything fractional or out of range does
            // not. Written without `f64::fract` so this works without `std`.
            Value::Float(f) if *f == (*f as i64) as f64 => Some(*f as i64),
            _ => None,
        }
    }

    /// Read this value as a float, accepting `Float` and `Int`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Borrow the list elements, if this value is a list.
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow the map, if this value is a map.
    pub fn as_map(&self) -> Option<&Fields> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow the tensor, if this value is one.
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Value::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow the decoded image, if this value is one.
    #[cfg(feature = "image")]
    pub fn as_image(&self) -> Option<&image::DynamicImage> {
        match self {
            Value::Image(i) => Some(i),
            _ => None,
        }
    }

    /// Downcast a [`Value::Custom`] to a concrete type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        match self {
            // Dereference through the `Arc` explicitly: the blanket impl below
            // also covers `Arc<dyn CustomValue>`, so `c.as_any()` would resolve
            // to the `Arc` itself rather than the value inside it.
            Value::Custom(c) => (**c).as_any().downcast_ref::<T>(),
            _ => None,
        }
    }

    /// Like [`Value::as_bytes`] but produces a descriptive error.
    pub fn expect_bytes(&self, key: &str) -> Result<&Bytes> {
        self.as_bytes().ok_or_else(|| Error::value(format!("{key}: expected bytes, found {}", self.type_name())))
    }

    /// Convert to a `serde_json::Value` where a faithful mapping exists.
    #[cfg(feature = "json")]
    ///
    /// Bytes are rejected rather than silently mangled; tensors become nested
    /// arrays of numbers; images and custom values are unsupported.
    pub fn to_json(&self) -> Result<serde_json::Value> {
        use serde_json::Value as J;
        Ok(match self {
            Value::Null => J::Null,
            Value::Bool(b) => J::Bool(*b),
            Value::Int(i) => J::Number((*i).into()),
            Value::Float(f) => serde_json::Number::from_f64(*f).map(J::Number).unwrap_or(J::Null),
            Value::Text(s) => J::String(s.clone()),
            Value::List(v) => J::Array(v.iter().map(|x| x.to_json()).collect::<Result<_>>()?),
            Value::Map(m) => {
                let mut o = serde_json::Map::new();
                for (k, v) in m {
                    o.insert(k.clone(), v.to_json()?);
                }
                J::Object(o)
            }
            Value::Tensor(t) => J::Array(
                t.to_f64_vec()
                    .into_iter()
                    .map(|f| serde_json::Number::from_f64(f).map(J::Number).unwrap_or(J::Null))
                    .collect(),
            ),
            other => {
                return Err(Error::unsupported(format!("cannot represent {} as json", other.type_name())));
            }
        })
    }
}

#[cfg(feature = "json")]
impl From<serde_json::Value> for Value {
    fn from(v: serde_json::Value) -> Value {
        use serde_json::Value as J;
        match v {
            J::Null => Value::Null,
            J::Bool(b) => Value::Bool(b),
            J::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else {
                    Value::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            J::String(s) => Value::Text(s),
            J::Array(a) => Value::List(a.into_iter().map(Value::from).collect()),
            J::Object(o) => Value::Map(o.into_iter().map(|(k, v)| (k, Value::from(v))).collect()),
        }
    }
}

macro_rules! from_impl {
    ($($t:ty => $variant:expr),* $(,)?) => {
        $(impl From<$t> for Value {
            fn from(v: $t) -> Value {
                #[allow(clippy::redundant_closure_call)]
                ($variant)(v)
            }
        })*
    };
}

from_impl! {
    bool => Value::Bool,
    i64 => Value::Int,
    f64 => Value::Float,
    String => Value::Text,
    Bytes => Value::Bytes,
    Tensor => Value::Tensor,
    Vec<Value> => Value::List,
    Fields => Value::Map,
}

impl From<&str> for Value {
    fn from(v: &str) -> Value {
        Value::Text(v.to_string())
    }
}

impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Value {
        Value::Bytes(Bytes::from(v))
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Value {
        Value::Int(v as i64)
    }
}

impl From<usize> for Value {
    fn from(v: usize) -> Value {
        Value::Int(v as i64)
    }
}

impl From<f32> for Value {
    fn from(v: f32) -> Value {
        Value::Float(v as f64)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => f.write_str("Null"),
            Value::Bool(b) => write!(f, "Bool({b})"),
            Value::Int(i) => write!(f, "Int({i})"),
            Value::Float(x) => write!(f, "Float({x})"),
            Value::Text(s) => write!(f, "Text({:?})", Truncated(s)),
            Value::Bytes(b) => write!(f, "Bytes({} bytes)", b.len()),
            Value::List(v) => f.debug_tuple("List").field(v).finish(),
            Value::Map(m) => f.debug_tuple("Map").field(m).finish(),
            Value::Tensor(t) => write!(f, "Tensor({} {:?})", t.dtype().long_name(), t.shape()),
            #[cfg(feature = "image")]
            Value::Image(i) => {
                use image::GenericImageView;
                let (w, h) = i.dimensions();
                write!(f, "Image({w}x{h})")
            }
            Value::Custom(c) => write!(f, "Custom({c:?})"),
        }
    }
}

struct Truncated<'a>(&'a str);

impl fmt::Debug for Truncated<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() <= 60 {
            write!(f, "{}", self.0)
        } else {
            let cut = self.0.char_indices().nth(57).map(|(i, _)| i).unwrap_or(self.0.len());
            write!(f, "{}...", &self.0[..cut])
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Text(a), Value::Text(b)) => a == b,
            (Value::Bytes(a), Value::Bytes(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Map(a), Value::Map(b)) => a == b,
            (Value::Tensor(a), Value::Tensor(b)) => a == b,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "json")]
    #[test]
    fn converts_json_both_ways() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": 1, "b": [true, null, "x"]}"#).unwrap();
        let value = Value::from(json.clone());
        assert_eq!(value.to_json().unwrap(), json);
    }

    #[test]
    fn coerces_numbers() {
        assert_eq!(Value::Int(3).as_f64(), Some(3.0));
        assert_eq!(Value::Float(3.0).as_i64(), Some(3));
        assert_eq!(Value::Float(3.5).as_i64(), None);
    }

    #[test]
    fn round_trips_custom_values() {
        #[derive(Debug, PartialEq)]
        struct Mine(u32);
        let v = Value::custom(Mine(7));
        assert_eq!(v.downcast_ref::<Mine>(), Some(&Mine(7)));
        assert_eq!(v.downcast_ref::<u32>(), None);
    }

    #[test]
    fn truncates_long_text_in_debug() {
        let long = "x".repeat(200);
        let shown = format!("{:?}", Value::Text(long));
        assert!(shown.len() < 80, "{shown}");
    }
}
