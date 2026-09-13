//! Tooling for this repository, kept out of the published crates.
//!
//! - `parity-dump` writes the digest stream that `parity/` compares against the
//!   reference implementation.
//! - `readme-snippets` exists only to fail the build if the README's examples
//!   stop compiling.
//!
//! Neither belongs in `webdataset`'s own `examples/`, where they would ship to
//! anyone depending on the crate.
