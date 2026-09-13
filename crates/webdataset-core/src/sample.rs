//! The [`Sample`] type: one training example, assembled from the files in a
//! tar archive that share a basename.

use indexmap::map::Iter as MapIter;

use crate::fields::Fields;

use crate::error::{Error, Result};
use crate::prelude::*;
use crate::value::Value;

/// The field holding the sample's basename within its shard.
pub const KEY: &str = "__key__";
/// The field holding the URL of the shard the sample came from.
pub const URL: &str = "__url__";
/// The field holding the local path of the shard, when it was read from disk.
pub const LOCAL_PATH: &str = "__local_path__";
/// A truthy value in this field marks the sample as unusable.
pub const BAD: &str = "__bad__";

/// The prefix and suffix that mark a field as metadata rather than data.
pub const META_PREFIX: &str = "__";
/// See [`META_PREFIX`].
pub const META_SUFFIX: &str = "__";

/// One training example: an insertion-ordered map from file extension to value.
///
/// Fields whose names start with `__` are metadata and are skipped by decoding,
/// encoding, and key extraction.
///
/// ```
/// use webdataset_core::{Sample, Value};
///
/// let mut sample = Sample::with_key("image0001");
/// sample.insert("cls", Value::Int(7));
/// sample.insert("txt", Value::Text("a caption".into()));
///
/// assert_eq!(sample.key(), Some("image0001"));
/// assert_eq!(sample.get("cls").and_then(Value::as_i64), Some(7));
/// assert_eq!(sample.field_names(), vec!["cls", "txt"]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sample {
    fields: Fields,
}

impl Sample {
    /// An empty sample with no fields at all.
    pub fn new() -> Sample {
        Sample::default()
    }

    /// An empty sample carrying only its `__key__`.
    pub fn with_key(key: impl Into<String>) -> Sample {
        let mut s = Sample::new();
        s.insert(KEY, Value::Text(key.into()));
        s
    }

    /// The sample's `__key__`, if it has one.
    pub fn key(&self) -> Option<&str> {
        self.get(KEY).and_then(Value::as_str)
    }

    /// The URL of the shard this sample came from.
    pub fn url(&self) -> Option<&str> {
        self.get(URL).and_then(Value::as_str)
    }

    /// The on-disk path of the shard, when it was read or cached locally.
    pub fn local_path(&self) -> Option<&str> {
        self.get(LOCAL_PATH).and_then(Value::as_str)
    }

    /// Set the `__key__` field.
    pub fn set_key(&mut self, key: impl Into<String>) {
        self.insert(KEY, Value::Text(key.into()));
    }

    /// Set the `__url__` field.
    pub fn set_url(&mut self, url: impl Into<String>) {
        self.insert(URL, Value::Text(url.into()));
    }

    /// Look up a field.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    /// Look up a field for modification.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Value> {
        self.fields.get_mut(name)
    }

    /// Look up the first field that exists among several alternatives.
    ///
    /// Alternatives are given either as a slice or, following the Python API,
    /// as a single `;`-separated string (see [`Sample::get_first_spec`]).
    pub fn get_first(&self, names: &[&str]) -> Option<&Value> {
        names.iter().find_map(|n| self.get(n))
    }

    /// Look up the first field named by a `;`-separated alternation such as
    /// `"png;jpg;jpeg"`.
    pub fn get_first_spec(&self, spec: &str) -> Option<&Value> {
        spec.split(';').find_map(|n| self.get(n))
    }

    /// Like [`Sample::get_first_spec`] but reports which alternatives were tried.
    pub fn require_first_spec(&self, spec: &str) -> Result<&Value> {
        self.get_first_spec(spec).ok_or_else(|| Error::MissingKey {
            wanted: spec.split(';').map(str::to_string).collect(),
            available: self.fields.keys().cloned().collect(),
        })
    }

    /// Insert a field, returning the value it replaced.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<Value>) -> Option<Value> {
        self.fields.insert(name.into(), value.into())
    }

    /// Remove a field, returning its value.
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.fields.shift_remove(name)
    }

    /// Whether a field is present.
    pub fn contains_key(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// The number of fields, metadata included.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the sample has no fields at all.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Iterate over all fields in insertion order.
    pub fn iter(&self) -> MapIter<'_, String, Value> {
        self.fields.iter()
    }

    /// All field names, metadata included.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// The names of the data (non-`__`) fields.
    pub fn field_names(&self) -> Vec<&str> {
        self.keys().filter(|k| !is_meta(k)).collect()
    }

    /// Borrow the underlying map.
    pub fn as_map(&self) -> &Fields {
        &self.fields
    }

    /// Consume the sample and return the underlying map.
    pub fn into_map(self) -> Fields {
        self.fields
    }

    /// Whether this sample should be passed downstream.
    ///
    /// Mirrors `valid_sample` in the Python implementation: a sample is valid
    /// when it has at least one field and is not marked `__bad__`.
    pub fn is_valid(&self) -> bool {
        !self.fields.is_empty() && !matches!(self.get(BAD), Some(Value::Bool(true)))
    }

    /// Rename a field, preserving its position in the field order.
    pub fn rename(&mut self, from: &str, to: impl Into<String>) -> bool {
        let Some(index) = self.fields.get_index_of(from) else {
            return false;
        };
        let (_, value) = self.fields.shift_remove_index(index).expect("index was just looked up");
        self.fields.shift_insert(index, to.into(), value);
        true
    }
}

impl From<Fields> for Sample {
    fn from(fields: Fields) -> Sample {
        Sample { fields }
    }
}

impl FromIterator<(String, Value)> for Sample {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Sample {
        Sample { fields: iter.into_iter().collect() }
    }
}

impl<'a> IntoIterator for &'a Sample {
    type Item = (&'a String, &'a Value);
    type IntoIter = MapIter<'a, String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.iter()
    }
}

impl IntoIterator for Sample {
    type Item = (String, Value);
    type IntoIter = indexmap::map::IntoIter<String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.into_iter()
    }
}

impl core::ops::Index<&str> for Sample {
    type Output = Value;

    fn index(&self, name: &str) -> &Value {
        self.get(name).unwrap_or_else(|| panic!("no field {name:?} in sample; have {:?}", self.field_names()))
    }
}

/// Whether a field name denotes metadata (`__key__`, `__url__`, ...).
pub fn is_meta(name: &str) -> bool {
    name.starts_with(META_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_metadata_from_data_fields() {
        let mut s = Sample::with_key("k");
        s.set_url("pipe:cat shard.tar");
        s.insert("png", Value::Bytes(vec![1, 2, 3].into()));
        assert_eq!(s.field_names(), vec!["png"]);
        assert_eq!(s.key(), Some("k"));
        assert_eq!(s.url(), Some("pipe:cat shard.tar"));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn resolves_alternatives_in_order() {
        let mut s = Sample::with_key("k");
        s.insert("jpg", Value::Int(1));
        s.insert("png", Value::Int(2));
        assert_eq!(s.get_first_spec("png;jpg").and_then(Value::as_i64), Some(2));
        assert_eq!(s.get_first_spec("cls;jpg").and_then(Value::as_i64), Some(1));
        assert!(s.get_first_spec("cls;wnid").is_none());
        assert!(s.require_first_spec("cls").is_err());
    }

    #[test]
    fn treats_empty_and_bad_samples_as_invalid() {
        assert!(!Sample::new().is_valid());
        let mut s = Sample::with_key("k");
        assert!(s.is_valid());
        s.insert(BAD, Value::Bool(true));
        assert!(!s.is_valid());
    }

    #[test]
    fn renames_in_place() {
        let mut s = Sample::with_key("k");
        s.insert("a", Value::Int(1));
        s.insert("b", Value::Int(2));
        assert!(s.rename("a", "z"));
        assert_eq!(s.field_names(), vec!["z", "b"]);
        assert!(!s.rename("nope", "x"));
    }
}
