//! The map type used for sample fields.
//!
//! Fields are keyed by short file extensions such as `png` or `cls`, and their
//! order matters — it is the order the files appeared in the archive. That
//! calls for an insertion-ordered map, and for a hasher that is cheap on short
//! keys rather than one hardened against adversarial input.
//!
//! [`FieldHasher`] is FNV-1a: deterministic, fast on a handful of bytes, and
//! available without the standard library, which is what lets the data model
//! build for `no_std` targets.

use core::hash::{BuildHasherDefault, Hasher};

use indexmap::IndexMap;

use crate::prelude::*;
use crate::value::Value;

/// The hasher used for field maps.
pub type FieldHasher = BuildHasherDefault<Fnv1a>;

/// An insertion-ordered map from field name to value.
pub type Fields = IndexMap<String, Value, FieldHasher>;

/// An insertion-ordered map with the same hasher, over any value type.
pub type Map<V> = IndexMap<String, V, FieldHasher>;

/// An empty field map.
pub fn fields() -> Fields {
    Fields::default()
}

/// The FNV-1a hash, 64-bit variant.
#[derive(Debug, Clone, Copy)]
pub struct Fnv1a(u64);

impl Default for Fnv1a {
    fn default() -> Fnv1a {
        Fnv1a(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for Fnv1a {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= *byte as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(text: &str) -> u64 {
        let mut hasher = Fnv1a::default();
        hasher.write(text.as_bytes());
        hasher.finish()
    }

    #[test]
    fn hashes_distinctly_and_stably() {
        assert_eq!(hash("png"), hash("png"));
        assert_ne!(hash("png"), hash("cls"));
        assert_ne!(hash("png"), hash("jpg"));
    }

    #[test]
    fn matches_the_reference_fnv1a_vector() {
        // The published FNV-1a 64 test vector for "a".
        assert_eq!(hash("a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn keeps_insertion_order() {
        let mut map = fields();
        for name in ["zzz", "aaa", "mmm"] {
            map.insert(name.to_string(), Value::Null);
        }
        assert_eq!(map.keys().map(String::as_str).collect::<Vec<_>>(), ["zzz", "aaa", "mmm"]);
    }
}
