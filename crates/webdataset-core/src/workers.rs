//! Which worker the calling thread is, and how that is discovered.
//!
//! A sharded pipeline needs to know two things: how many workers are reading
//! it, and which one this thread is. Everything else — splitting shards across
//! workers, deriving per-worker seeds — follows from that.
//!
//! How the answer is stored depends on what the target offers:
//!
//! | build | storage |
//! |---|---|
//! | `std` | a thread-local, set for the duration of [`with_worker`] |
//! | `no_std` + `threads` | a lock-free slot table, keyed by a host-supplied thread token |
//! | `no_std` without `threads` | a single global, which is all a single-threaded target needs |
//!
//! The middle row is the interesting one. Without the standard library there is
//! no portable way to ask "which thread am I on?", so the host has to say. Call
//! [`set_thread_id_hook`] once with something that identifies the current
//! execution context — a task id, a core id, a worker index — and per-worker
//! state starts working:
//!
//! ```
//! # #[cfg(feature = "threads")] {
//! use webdataset_core::workers::{set_thread_id_hook, with_worker};
//! use webdataset_core::worker_info;
//!
//! // On a real RTOS this would return the current task's id.
//! set_thread_id_hook(|| 0).ok();
//!
//! let info = with_worker(2, 4, worker_info);
//! assert_eq!((info.worker, info.num_workers), (2, 4));
//! # }
//! ```
//!
//! Without a hook the table is bypassed and the single global is used, so a
//! single-threaded `no_std` program needs no setup at all.

#[cfg(not(feature = "std"))]
use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(all(feature = "threads", not(feature = "std")))]
use crate::error::Error;
use crate::error::Result;

/// How many execution contexts can hold a worker identity at once in a
/// `no_std` build. Beyond this the global fallback is used.
#[cfg(all(feature = "threads", not(feature = "std")))]
pub const MAX_THREADS: usize = 64;

/// The sentinel for "no identity bound".
#[cfg(not(feature = "std"))]
const UNSET: usize = usize::MAX;

// ---------------------------------------------------------------------------
// std: a thread-local, which is exactly what this needs and costs nothing.
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
std::thread_local! {
    static CURRENT: core::cell::Cell<Option<(usize, usize)>> = const { core::cell::Cell::new(None) };
}

/// The worker identity bound to this thread, if any.
#[cfg(feature = "std")]
pub fn current() -> Option<(usize, usize)> {
    CURRENT.with(|slot| slot.get())
}

/// Bind an identity to this thread, returning the one it replaced.
#[cfg(feature = "std")]
pub fn replace(next: Option<(usize, usize)>) -> Option<(usize, usize)> {
    CURRENT.with(|slot| slot.replace(next))
}

// ---------------------------------------------------------------------------
// no_std without threads: one global is enough, and it is still atomic.
// ---------------------------------------------------------------------------

/// The worker identity bound to this program, if any.
#[cfg(not(any(feature = "std", feature = "threads")))]
pub fn current() -> Option<(usize, usize)> {
    read_global()
}

/// Bind an identity, returning the one it replaced.
#[cfg(not(any(feature = "std", feature = "threads")))]
pub fn replace(next: Option<(usize, usize)>) -> Option<(usize, usize)> {
    replace_global(next)
}

// ---------------------------------------------------------------------------
// no_std with threads: a slot table keyed by a host-supplied thread token.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "threads", not(feature = "std")))]
mod table {
    use super::*;
    use alloc::boxed::Box;
    use once_cell::race::OnceBox;

    /// Tells the library which execution context is running.
    pub type ThreadIdHook = Box<dyn Fn() -> usize + Send + Sync>;

    static HOOK: OnceBox<ThreadIdHook> = OnceBox::new();

    /// One bound identity: a token and the worker it maps to.
    pub struct Slot {
        pub token: AtomicUsize,
        pub worker: AtomicUsize,
        pub num_workers: AtomicUsize,
    }

    impl Slot {
        const fn new() -> Slot {
            Slot { token: AtomicUsize::new(UNSET), worker: AtomicUsize::new(UNSET), num_workers: AtomicUsize::new(1) }
        }
    }

    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY: Slot = Slot::new();
    pub static SLOTS: [Slot; MAX_THREADS] = [EMPTY; MAX_THREADS];

    /// Install the hook that identifies the current execution context.
    ///
    /// Can only be done once; a second call reports an error rather than
    /// silently changing how every existing binding is interpreted.
    pub fn set_hook(hook: impl Fn() -> usize + Send + Sync + 'static) -> Result<()> {
        HOOK.set(Box::new(Box::new(hook))).map_err(|_| Error::value("the thread id hook has already been set"))
    }

    /// The current context's token, if a hook is installed.
    pub fn token() -> Option<usize> {
        HOOK.get().map(|hook| hook())
    }

    /// Find the slot holding `token`.
    fn find(token: usize) -> Option<&'static Slot> {
        SLOTS.iter().find(|slot| slot.token.load(Ordering::Acquire) == token)
    }

    /// Claim a free slot for `token`.
    fn claim(token: usize) -> Option<&'static Slot> {
        SLOTS.iter().find(|slot| slot.token.compare_exchange(UNSET, token, Ordering::AcqRel, Ordering::Acquire).is_ok())
    }

    /// The identity bound to the current context.
    pub fn current() -> Option<(usize, usize)> {
        let Some(token) = token() else {
            return read_global();
        };
        let slot = find(token)?;
        match slot.worker.load(Ordering::Acquire) {
            UNSET => None,
            worker => Some((worker, slot.num_workers.load(Ordering::Acquire))),
        }
    }

    /// Bind `next` to the current context, returning what it replaced.
    pub fn replace(next: Option<(usize, usize)>) -> Option<(usize, usize)> {
        let Some(token) = token() else {
            // No hook: there is no way to tell contexts apart, so the single
            // global is the best available answer.
            return replace_global(next);
        };

        let slot = match find(token).or_else(|| claim(token)) {
            Some(slot) => slot,
            None => {
                // More live contexts than slots. Falling back keeps the
                // pipeline correct for one of them rather than wrong for all.
                log::warn!("more than {MAX_THREADS} worker threads; falling back to a shared identity");
                return replace_global(next);
            }
        };

        let previous = match slot.worker.load(Ordering::Acquire) {
            UNSET => None,
            worker => Some((worker, slot.num_workers.load(Ordering::Acquire))),
        };
        match next {
            Some((worker, num_workers)) => {
                slot.num_workers.store(num_workers.max(1), Ordering::Release);
                slot.worker.store(worker, Ordering::Release);
            }
            None => {
                slot.worker.store(UNSET, Ordering::Release);
                // Release the slot so a later context can reuse it.
                slot.token.store(UNSET, Ordering::Release);
            }
        }
        previous
    }
}

#[cfg(all(feature = "threads", not(feature = "std")))]
pub use table::{current, replace};

/// The global fallback, used when no per-context storage is available.
#[cfg(not(feature = "std"))]
static GLOBAL_FALLBACK: (AtomicUsize, AtomicUsize) = (AtomicUsize::new(UNSET), AtomicUsize::new(1));

#[cfg(not(feature = "std"))]
fn read_global() -> Option<(usize, usize)> {
    match GLOBAL_FALLBACK.0.load(Ordering::Acquire) {
        UNSET => None,
        worker => Some((worker, GLOBAL_FALLBACK.1.load(Ordering::Acquire))),
    }
}

#[cfg(not(feature = "std"))]
fn replace_global(next: Option<(usize, usize)>) -> Option<(usize, usize)> {
    let previous = read_global();
    let (worker, num_workers) = next.unwrap_or((UNSET, 1));
    GLOBAL_FALLBACK.1.store(num_workers.max(1), Ordering::Release);
    GLOBAL_FALLBACK.0.store(worker, Ordering::Release);
    previous
}

/// Install the hook that tells the library which execution context is running.
///
/// Only meaningful in a `no_std` build with the `threads` feature; with the
/// standard library a thread-local is used instead and this is a no-op that
/// reports success. Can only be set once.
#[cfg(any(feature = "std", not(feature = "threads")))]
pub fn set_thread_id_hook(_hook: impl Fn() -> usize + Send + Sync + 'static) -> Result<()> {
    Ok(())
}

/// Install the hook that tells the library which execution context is running.
///
/// See the [module documentation](self) for what the token should be.
#[cfg(all(feature = "threads", not(feature = "std")))]
pub fn set_thread_id_hook(hook: impl Fn() -> usize + Send + Sync + 'static) -> Result<()> {
    table::set_hook(hook)
}

/// Run `body` with this context presenting as worker `worker` of `num_workers`.
///
/// The identity is restored afterwards, so nesting works.
pub fn with_worker<T>(worker: usize, num_workers: usize, body: impl FnOnce() -> T) -> T {
    let previous = replace(Some((worker, num_workers)));
    let result = body();
    replace(previous);
    result
}

/// Bind a worker identity to this context until it is replaced.
pub fn set_worker(worker: usize, num_workers: usize) {
    replace(Some((worker, num_workers)));
}

/// Forget this context's worker identity.
pub fn clear_worker() {
    replace(None);
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::vec::Vec;

    #[test]
    fn binds_and_restores_an_identity() {
        assert_eq!(current(), None);
        with_worker(3, 8, || {
            assert_eq!(current(), Some((3, 8)));
            with_worker(1, 2, || assert_eq!(current(), Some((1, 2))));
            assert_eq!(current(), Some((3, 8)), "the outer binding is restored");
        });
        assert_eq!(current(), None);
    }

    #[test]
    fn set_and_clear_persist_beyond_a_scope() {
        set_worker(5, 6);
        assert_eq!(current(), Some((5, 6)));
        clear_worker();
        assert_eq!(current(), None);
    }

    #[cfg(feature = "std")]
    #[test]
    fn identities_do_not_leak_between_threads() {
        set_worker(1, 4);
        let seen = std::thread::spawn(current).join().expect("thread");
        assert_eq!(seen, None, "another thread has its own identity");
        assert_eq!(current(), Some((1, 4)));
        clear_worker();
    }

    #[cfg(feature = "std")]
    #[test]
    fn many_threads_keep_their_own_identity() {
        let handles: Vec<_> =
            std::vec::Vec::from_iter((0..16).map(|i| std::thread::spawn(move || with_worker(i, 16, current))));
        for (i, handle) in handles.into_iter().enumerate() {
            assert_eq!(handle.join().expect("thread"), Some((i, 16)));
        }
    }

    /// The `no_std` slot table, driven by real threads.
    ///
    /// The hook stands in for whatever a real host would provide — an RTOS task
    /// id, a core number, a worker index.
    #[cfg(all(feature = "threads", not(feature = "std")))]
    #[test]
    fn the_slot_table_keeps_threads_apart() {
        std::thread_local! {
            static TOKEN: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
        }
        set_thread_id_hook(|| TOKEN.with(|t| t.get())).expect("the hook is set once");
        assert!(set_thread_id_hook(|| 0).is_err(), "the hook cannot be replaced");

        let handles: Vec<_> = std::vec::Vec::from_iter((0..8).map(|i| {
            std::thread::spawn(move || {
                // A real host would derive this; the test just assigns one.
                TOKEN.with(|t| t.set(i + 1));
                let seen = with_worker(i, 8, current);
                (seen, current())
            })
        }));

        for (i, handle) in handles.into_iter().enumerate() {
            let (inside, after) = handle.join().expect("thread");
            assert_eq!(inside, Some((i, 8)), "thread {i} should see its own identity");
            assert_eq!(after, None, "the binding is released when the scope ends");
        }
    }

    /// Without a hook there is nothing to tell contexts apart, so the single
    /// global is used and shared — correct for a single-threaded program.
    #[cfg(all(not(feature = "threads"), not(feature = "std")))]
    #[test]
    fn without_threads_a_single_global_is_used() {
        set_worker(2, 3);
        assert_eq!(current(), Some((2, 3)));
        assert_eq!(std::thread::spawn(current).join().expect("thread"), Some((2, 3)));
        clear_worker();
    }
}
