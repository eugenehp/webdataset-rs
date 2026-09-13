//! Error type shared across the `webdataset` crates.

use crate::prelude::*;
use core::fmt;

/// Convenience alias for results produced by this workspace.
pub type Result<T> = core::result::Result<T, Error>;

/// The error type used throughout the webdataset crates.
///
/// Errors carry enough context to be reported usefully by the pluggable
/// [`Handler`](crate::handlers::Handler) implementations: the URL of the shard
/// and the key of the sample are attached as they travel up the pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An underlying I/O failure.
    #[cfg(feature = "std")]
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON value could not be parsed or serialized.
    #[cfg(feature = "json")]
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Bytes that were expected to be UTF-8 were not.
    #[error("utf-8 error: {0}")]
    Utf8(#[from] core::str::Utf8Error),

    /// A sample field could not be decoded.
    #[error("cannot decode {key}: {message}")]
    Decode {
        /// The field (file extension) that failed to decode.
        key: String,
        /// A human readable description of the failure.
        message: String,
    },

    /// A sample field could not be encoded for writing.
    #[error("cannot encode {key}: {message}")]
    Encode {
        /// The field that failed to encode.
        key: String,
        /// A human readable description of the failure.
        message: String,
    },

    /// A required key was not present in a sample.
    #[error("missing key: did not find {wanted:?} in {available:?}")]
    MissingKey {
        /// The key(s) that were looked up.
        wanted: Vec<String>,
        /// The keys the sample actually had.
        available: Vec<String>,
    },

    /// The same key occurred twice while grouping files into a sample.
    #[error("duplicate key {key} in sample {sample_key}")]
    DuplicateKey {
        /// The duplicated field name.
        key: String,
        /// The key of the sample being assembled.
        sample_key: String,
    },

    /// An operation was refused because secure mode is enabled.
    #[error("refused in secure mode: {0}")]
    Security(String),

    /// A subprocess used to open a URL failed.
    #[error("subprocess failed: {0}")]
    Subprocess(String),

    /// Malformed input data (bad archive, bad header, bad brace expression).
    #[error("malformed input: {0}")]
    Format(String),

    /// The requested functionality is not available in this build.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A value had the right shape but the wrong contents.
    #[error("invalid value: {0}")]
    Value(String),

    /// A pipeline stage that was declared non-empty produced no data.
    #[error("empty: {0}")]
    Empty(String),

    /// An error with additional context attached.
    #[error("{context}: {source}")]
    Context {
        /// What was being attempted.
        context: String,
        /// The underlying error.
        #[source]
        source: Box<Error>,
    },
}

impl Error {
    /// Build an [`Error::Format`] from anything printable.
    pub fn format(message: impl fmt::Display) -> Self {
        Error::Format(message.to_string())
    }

    /// Build an [`Error::Value`] from anything printable.
    pub fn value(message: impl fmt::Display) -> Self {
        Error::Value(message.to_string())
    }

    /// Build an [`Error::Unsupported`] from anything printable.
    pub fn unsupported(message: impl fmt::Display) -> Self {
        Error::Unsupported(message.to_string())
    }

    /// Build an [`Error::Decode`] for the given field.
    pub fn decode(key: impl Into<String>, message: impl fmt::Display) -> Self {
        Error::Decode { key: key.into(), message: message.to_string() }
    }

    /// Build an [`Error::Encode`] for the given field.
    pub fn encode(key: impl Into<String>, message: impl fmt::Display) -> Self {
        Error::Encode { key: key.into(), message: message.to_string() }
    }

    /// Attach context to an error, mirroring the way the Python implementation
    /// appends the offending URL to `exn.args`.
    pub fn context(self, context: impl fmt::Display) -> Self {
        Error::Context { context: context.to_string(), source: Box::new(self) }
    }
}

/// Extension trait for adding context to a [`Result`].
pub trait ResultExt<T> {
    /// Attach context lazily to the error branch of a result.
    fn with_context<C: fmt::Display>(self, f: impl FnOnce() -> C) -> Result<T>;
}

impl<T, E: Into<Error>> ResultExt<T> for core::result::Result<T, E> {
    fn with_context<C: fmt::Display>(self, f: impl FnOnce() -> C) -> Result<T> {
        self.map_err(|e| e.into().context(f()))
    }
}
