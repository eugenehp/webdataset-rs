//! Pluggable exception handlers.
//!
//! These mirror the handlers in the Python implementation. A handler inspects
//! an [`Error`] and decides what the pipeline stage that produced it should do
//! next. Because Rust iterators cannot unwind and resume, "re-raise" means
//! "forward the error to the consumer" rather than "panic".
//!
//! ```
//! use webdataset_core::handlers::{warn_and_continue, Action};
//! use webdataset_core::Error;
//!
//! let handler = warn_and_continue();
//! assert_eq!(handler.handle(&Error::value("boom")), Action::Continue);
//! ```

use alloc::sync::Arc;
#[cfg(feature = "std")]
use core::time::Duration;

use crate::error::Error;

/// What a pipeline stage should do after an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Drop the offending item and keep going.
    Continue,
    /// Stop producing items; the stream ends cleanly.
    Stop,
    /// Forward the error downstream so the consumer sees it.
    Reraise,
}

/// Decides how a pipeline stage reacts to an error.
pub trait Handler: Send + Sync + core::fmt::Debug {
    /// Inspect `error` and decide what happens next.
    fn handle(&self, error: &Error) -> Action;
}

/// A shared, cheaply clonable handler.
pub type HandlerRef = Arc<dyn Handler>;

macro_rules! simple_handler {
    ($name:ident, $ctor:ident, $action:expr, $warn:expr, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl Handler for $name {
            fn handle(&self, error: &Error) -> Action {
                if $warn {
                    log::warn!("{error}");
                    // Matches the Python implementation: slow the stream down a
                    // little so warnings do not scroll past unnoticed.
                    #[cfg(feature = "std")]
                    std::thread::sleep(Duration::from_millis(500));
                }
                $action
            }
        }

        #[doc = $doc]
        pub fn $ctor() -> HandlerRef {
            Arc::new($name)
        }
    };
}

simple_handler!(
    ReraiseException,
    reraise_exception,
    Action::Reraise,
    false,
    "Forward the error to the consumer of the pipeline."
);
simple_handler!(
    IgnoreAndContinue,
    ignore_and_continue,
    Action::Continue,
    false,
    "Silently drop the offending item and continue."
);
simple_handler!(
    WarnAndContinue,
    warn_and_continue,
    Action::Continue,
    true,
    "Log the error, drop the offending item, and continue."
);
simple_handler!(IgnoreAndStop, ignore_and_stop, Action::Stop, false, "Silently end the stream.");
simple_handler!(WarnAndStop, warn_and_stop, Action::Stop, true, "Log the error and end the stream.");

/// Wraps a closure so it can be used as a [`Handler`].
pub struct FnHandler<F>(pub F);

impl<F> core::fmt::Debug for FnHandler<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("FnHandler")
    }
}

impl<F> Handler for FnHandler<F>
where
    F: Fn(&Error) -> Action + Send + Sync,
{
    fn handle(&self, error: &Error) -> Action {
        (self.0)(error)
    }
}

/// Build a [`HandlerRef`] from a closure.
pub fn handler_fn<F>(f: F) -> HandlerRef
where
    F: Fn(&Error) -> Action + Send + Sync + 'static,
{
    Arc::new(FnHandler(f))
}

/// Apply `handler` to `error` and translate its decision into the action a
/// pipeline stage should take.
pub fn dispatch(handler: &dyn Handler, error: Error) -> Dispatch {
    match handler.handle(&error) {
        Action::Continue => Dispatch::Skip,
        Action::Stop => Dispatch::Stop,
        Action::Reraise => Dispatch::Yield(error),
    }
}

/// The outcome of running a [`Handler`] over an error, as consumed by stages.
#[derive(Debug)]
pub enum Dispatch {
    /// Drop the item and pull the next one.
    Skip,
    /// End the stream.
    Stop,
    /// Yield this error downstream.
    Yield(Error),
}
