//! The lock order of the backends, checked in debug builds.
//!
//! Locks are taken in this order, never against it:
//!
//! 1. the random number generator's state (`ops::random`);
//! 2. the CPU execution lock (`cpu::exec`);
//! 3. one ROCm device's execution lock (`rocm::exec`), never two at a time: copies between ROCm
//!    devices go through host memory, holding one device's lock and then the other's;
//! 4. allocator locks, which are leaves: nothing is locked while holding one.
//!
//! Every acquisition (not the reentrant ones) calls [`acquire`] first, which panics in debug
//! builds if the calling thread holds a lock that comes later in the order, before the
//! acquisition could deadlock. Release builds don't track anything.

#[cfg(debug_assertions)]
use std::cell::RefCell;

/// A lock in the order above.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Rank {
    Generator,
    CpuExec,
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    RocmExec(u32),
    #[allow(dead_code)]
    Allocator,
}

impl Rank {
    #[cfg(debug_assertions)]
    fn level(self) -> u8 {
        match self {
            Rank::Generator => 0,
            Rank::CpuExec => 1,
            Rank::RocmExec(_) => 2,
            Rank::Allocator => 3,
        }
    }
}

#[cfg(debug_assertions)]
thread_local! {
    static HELD: RefCell<Vec<Rank>> = const { RefCell::new(Vec::new()) };
}

/// Marks a lock of `rank` as held by this thread until the result is dropped.
///
/// # Panics
///
/// Panics in debug builds if this thread holds a lock that must not be held while taking it.
#[must_use = "the lock counts as held until this is dropped"]
pub(crate) fn acquire(rank: Rank) -> Held {
    #[cfg(not(debug_assertions))]
    let _ = rank;
    #[cfg(debug_assertions)]
    HELD.with_borrow_mut(|held| {
        if let Some(h) = held.iter().find(|h| h.level() >= rank.level()) {
            panic!(
                "internal error: lock order violation: taking {rank:?} while holding {h:?} \
                 (held: {held:?})"
            );
        }
        held.push(rank);
    });
    Held {
        #[cfg(debug_assertions)]
        rank,
    }
}

/// See [`acquire`].
#[derive(Debug)]
pub(crate) struct Held {
    #[cfg(debug_assertions)]
    rank: Rank,
}

impl Drop for Held {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        HELD.with_borrow_mut(|held| {
            let i = held.iter().rposition(|h| *h == self.rank);
            held.remove(i.expect("released lock was held"));
        });
    }
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;

    fn violation(first: Rank, second: Rank) -> bool {
        std::panic::catch_unwind(|| {
            let _a = acquire(first);
            let _b = acquire(second);
        })
        .is_err()
    }

    #[test]
    fn locks_follow_the_order() {
        use Rank::*;
        assert!(!violation(Generator, CpuExec));
        assert!(!violation(CpuExec, RocmExec(0)));
        assert!(!violation(Generator, RocmExec(1)));
        assert!(!violation(RocmExec(0), Allocator));
        assert!(violation(CpuExec, Generator));
        assert!(violation(RocmExec(0), CpuExec));
        assert!(violation(RocmExec(0), Generator));
        assert!(violation(RocmExec(0), RocmExec(1)));
        assert!(violation(Allocator, Allocator));
        assert!(violation(Allocator, CpuExec));
        // A failed acquisition leaves nothing behind.
        HELD.with_borrow(|held| assert!(held.is_empty()));
    }
}
