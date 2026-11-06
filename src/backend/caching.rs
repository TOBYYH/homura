//! A caching allocator for device memory, after a simplified PyTorch CUDA caching allocator.
//!
//! Requests of up to 1 MiB are rounded up to 512 bytes and cut from 2 MiB segments; larger ones
//! are rounded up to 2 MiB and get a segment of their own. Freed blocks stay cached, merged with
//! free neighbours of their segment, and are reused best fit (the smallest free block of the
//! request's pool that is large enough, split if larger). When the device has no memory left,
//! the cached segments that are entirely free are released and the request is tried once more.
//!
//! The device memory itself comes from a [`Memory`], so the bookkeeping is tested without a GPU.
//! Blocks are freed back to the cache as soon as their buffer is dropped: on a device with a
//! single stream, work that still uses the block was queued before, and work that reuses it is
//! queued after (stream-ordered reuse, as in PyTorch), so no events are needed. Segments are
//! released to the device only after waiting for the stream.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

use crate::backend::locks::{self, Rank};
use crate::storage::MemoryCounters;

/// The largest request served from the small pool.
pub(crate) const SMALL_MAX: usize = 1 << 20;
/// The rounding of small requests.
pub(crate) const SMALL_ROUND: usize = 512;
/// The size of the segments that small requests are cut from.
pub(crate) const SMALL_SEGMENT: usize = 2 << 20;
/// The rounding of large requests, and of their segments.
pub(crate) const LARGE_ROUND: usize = 2 << 20;

/// The device memory behind a [`CachingAllocator`]. Addresses are plain numbers here.
pub(crate) trait Memory {
    /// `size` bytes of device memory.
    fn malloc(&mut self, size: usize) -> Result<usize, MallocError>;
    /// Frees a segment from [`malloc`](Memory::malloc).
    fn free(&mut self, addr: usize) -> Result<(), String>;
    /// Waits until the work queued on the device, which may use the memory, has finished.
    fn synchronize(&mut self) -> Result<(), String>;
}

/// A failed [`Memory::malloc`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MallocError {
    /// The device has no memory left.
    OutOfMemory,
    /// Any other failure, with its message.
    Failed(String),
}

/// A failed request to a [`CachingAllocator`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PoolError {
    /// The device has no memory left, even after the cached segments were released.
    OutOfMemory,
    /// The device failed otherwise (allocating, freeing or waiting), with its message.
    Failed(String),
}

/// A block handed out by [`CachingAllocator::allocate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    pub addr: usize,
    /// The rounded size, which counts as allocated.
    pub size: usize,
}

/// A device's allocator: a [`Pool`] behind a mutex (in the lock order, only the memory counters
/// come after it), and its counters, which are read without the pool's mutex.
pub(crate) struct CachingAllocator<M> {
    pool: Mutex<Pool<M>>,
    counters: MemoryCounters,
}

impl<M: Memory> CachingAllocator<M> {
    pub(crate) fn new(memory: M) -> Self {
        CachingAllocator {
            pool: Mutex::new(Pool::new(memory)),
            counters: MemoryCounters::default(),
        }
    }

    fn with_pool<R>(&self, f: impl FnOnce(&mut Pool<M>, &MemoryCounters) -> R) -> R {
        let _rank = locks::acquire(Rank::Allocator);
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut pool, &self.counters)
    }

    /// A block of at least `nbytes` (which isn't 0) bytes.
    pub(crate) fn allocate(&self, nbytes: usize) -> Result<Block, PoolError> {
        self.with_pool(|pool, counters| pool.allocate(nbytes, counters))
    }

    /// Returns the block at `addr` to the cache.
    pub(crate) fn release(&self, addr: usize) {
        self.with_pool(|pool, counters| pool.release(addr, counters))
    }

    /// Frees the cached segments that are entirely free.
    pub(crate) fn empty_cache(&self) -> Result<(), PoolError> {
        self.with_pool(|pool, counters| pool.release_cached(counters))
    }

    pub(crate) fn counters(&self) -> &MemoryCounters {
        &self.counters
    }

    #[cfg(test)]
    pub(crate) fn with_memory<R>(&self, f: impl FnOnce(&mut M) -> R) -> R {
        self.with_pool(|pool, _| f(&mut pool.memory))
    }
}

/// The size a request of `nbytes` is rounded to, and whether it comes from the small pool.
pub(crate) fn rounded(nbytes: usize) -> Option<(usize, bool)> {
    let small = nbytes <= SMALL_MAX;
    let unit = if small { SMALL_ROUND } else { LARGE_ROUND };
    let size = nbytes.max(1).checked_next_multiple_of(unit)?;
    Some((size, small))
}

#[derive(Debug)]
struct BlockInfo {
    size: usize,
    free: bool,
    small: bool,
    /// The neighbouring blocks of the same segment, by address.
    prev: Option<usize>,
    next: Option<usize>,
}

/// The blocks of all segments, by address, and the free ones of each pool by (size, address).
pub(crate) struct Pool<M> {
    memory: M,
    blocks: HashMap<usize, BlockInfo>,
    free_small: BTreeSet<(usize, usize)>,
    free_large: BTreeSet<(usize, usize)>,
    /// Segment sizes by address.
    segments: HashMap<usize, usize>,
}

impl<M: Memory> Pool<M> {
    fn new(memory: M) -> Self {
        Pool {
            memory,
            blocks: HashMap::new(),
            free_small: BTreeSet::new(),
            free_large: BTreeSet::new(),
            segments: HashMap::new(),
        }
    }

    fn free_set(&mut self, small: bool) -> &mut BTreeSet<(usize, usize)> {
        if small {
            &mut self.free_small
        } else {
            &mut self.free_large
        }
    }

    fn allocate(&mut self, nbytes: usize, counters: &MemoryCounters) -> Result<Block, PoolError> {
        debug_assert!(nbytes > 0);
        let (size, small) = rounded(nbytes).ok_or(PoolError::OutOfMemory)?;
        let fit = self.free_set(small).range((size, 0)..).next().copied();
        let addr = match fit {
            Some((_, addr)) => addr,
            None => self.new_segment(if small { SMALL_SEGMENT } else { size }, small, counters)?,
        };
        self.take(addr, size);
        counters.add(size as u64);
        Ok(Block { addr, size })
    }

    /// A new free segment of `size` bytes, after releasing the cached ones if the device has no
    /// memory left.
    fn new_segment(
        &mut self,
        size: usize,
        small: bool,
        counters: &MemoryCounters,
    ) -> Result<usize, PoolError> {
        let addr = match self.memory.malloc(size) {
            Ok(addr) => addr,
            Err(MallocError::OutOfMemory) => {
                self.release_cached(counters)?;
                match self.memory.malloc(size) {
                    Ok(addr) => addr,
                    Err(MallocError::OutOfMemory) => return Err(PoolError::OutOfMemory),
                    Err(MallocError::Failed(m)) => return Err(PoolError::Failed(m)),
                }
            }
            Err(MallocError::Failed(m)) => return Err(PoolError::Failed(m)),
        };
        let block = BlockInfo {
            size,
            free: true,
            small,
            prev: None,
            next: None,
        };
        self.blocks.insert(addr, block);
        self.free_set(small).insert((size, addr));
        self.segments.insert(addr, size);
        counters.reserve(size as u64);
        Ok(addr)
    }

    /// Marks the free block at `addr` as used, splitting off the bytes past `size`.
    fn take(&mut self, addr: usize, size: usize) {
        let b = &self.blocks[&addr];
        let (old, small, next) = (b.size, b.small, b.next);
        self.free_set(small).remove(&(old, addr));
        if old > size {
            let rest = addr + size;
            self.blocks.insert(
                rest,
                BlockInfo {
                    size: old - size,
                    free: true,
                    small,
                    prev: Some(addr),
                    next,
                },
            );
            self.free_set(small).insert((old - size, rest));
            if let Some(n) = next {
                self.blocks.get_mut(&n).unwrap().prev = Some(rest);
            }
            let b = self.blocks.get_mut(&addr).unwrap();
            b.size = size;
            b.next = Some(rest);
        }
        self.blocks.get_mut(&addr).unwrap().free = false;
    }

    fn release(&mut self, addr: usize, counters: &MemoryCounters) {
        let b = self
            .blocks
            .get_mut(&addr)
            .expect("internal error: freeing a block the allocator doesn't know");
        assert!(!b.free, "internal error: freeing a block twice");
        b.free = true;
        let (mut addr, small) = (addr, b.small);
        counters.sub(b.size as u64);
        if let Some(next) = self.blocks[&addr].next.filter(|n| self.blocks[n].free) {
            self.merge(addr, next);
        }
        if let Some(prev) = self.blocks[&addr].prev.filter(|p| self.blocks[p].free) {
            self.merge(prev, addr);
            addr = prev;
        }
        let size = self.blocks[&addr].size;
        self.free_set(small).insert((size, addr));
    }

    /// Merges the free block `second` into the free block `first`, its predecessor. `first` may
    /// be in its free set; `second` (if it was free before this release) is.
    fn merge(&mut self, first: usize, second: usize) {
        let s = self.blocks.remove(&second).unwrap();
        self.free_set(s.small).remove(&(s.size, second));
        let f = self.blocks.get_mut(&first).unwrap();
        let (fsize, small) = (f.size, f.small);
        f.size += s.size;
        f.next = s.next;
        self.free_set(small).remove(&(fsize, first));
        if let Some(n) = s.next {
            self.blocks.get_mut(&n).unwrap().prev = Some(first);
        }
    }

    /// Frees the segments that are one free block, after waiting for the device's work.
    fn release_cached(&mut self, counters: &MemoryCounters) -> Result<(), PoolError> {
        let unused: Vec<(usize, usize)> = self
            .segments
            .iter()
            .filter(|&(addr, &size)| {
                let b = &self.blocks[addr];
                b.free && b.size == size
            })
            .map(|(&addr, &size)| (addr, size))
            .collect();
        if unused.is_empty() {
            return Ok(());
        }
        self.memory.synchronize().map_err(PoolError::Failed)?;
        for (addr, size) in unused {
            self.memory.free(addr).map_err(PoolError::Failed)?;
            let b = self.blocks.remove(&addr).unwrap();
            self.free_set(b.small).remove(&(size, addr));
            self.segments.remove(&addr);
            counters.unreserve(size as u64);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Memory with a capacity, handing out increasing addresses and recording what it did.
    #[derive(Default)]
    struct Fake {
        capacity: usize,
        used: usize,
        next: usize,
        live: HashMap<usize, usize>,
        log: Vec<String>,
        fail_mallocs: usize,
        fail_free: bool,
    }

    impl Fake {
        fn with_capacity(capacity: usize) -> Self {
            Fake {
                capacity,
                next: 1 << 40,
                ..Fake::default()
            }
        }
    }

    impl Memory for Fake {
        fn malloc(&mut self, size: usize) -> Result<usize, MallocError> {
            if self.fail_mallocs > 0 {
                self.fail_mallocs -= 1;
                return Err(MallocError::Failed("injected".into()));
            }
            if self.used + size > self.capacity {
                self.log.push(format!("oom {size}"));
                return Err(MallocError::OutOfMemory);
            }
            let addr = self.next;
            self.next += size + 4096;
            self.used += size;
            self.live.insert(addr, size);
            self.log.push(format!("malloc {size}"));
            Ok(addr)
        }

        fn free(&mut self, addr: usize) -> Result<(), String> {
            if self.fail_free {
                return Err("free failed".into());
            }
            let size = self.live.remove(&addr).expect("freeing unknown memory");
            self.used -= size;
            self.log.push(format!("free {size}"));
            Ok(())
        }

        fn synchronize(&mut self) -> Result<(), String> {
            self.log.push("sync".into());
            Ok(())
        }
    }

    const MIB: usize = 1 << 20;

    fn allocator(capacity: usize) -> CachingAllocator<Fake> {
        CachingAllocator::new(Fake::with_capacity(capacity))
    }

    fn stats(a: &CachingAllocator<Fake>) -> (u64, u64, u64) {
        let s = a.counters().stats();
        (s.allocated, s.reserved, s.peak)
    }

    fn log(a: &CachingAllocator<Fake>) -> Vec<String> {
        a.with_memory(|m| std::mem::take(&mut m.log))
    }

    #[test]
    fn requests_are_rounded() {
        assert_eq!(rounded(1), Some((512, true)));
        assert_eq!(rounded(512), Some((512, true)));
        assert_eq!(rounded(513), Some((1024, true)));
        assert_eq!(rounded(MIB), Some((MIB, true)));
        assert_eq!(rounded(MIB + 1), Some((2 * MIB, false)));
        assert_eq!(rounded(5 * MIB), Some((6 * MIB, false)));
        assert_eq!(rounded(usize::MAX), None);
        let a = allocator(64 * MIB);
        assert_eq!(a.allocate(usize::MAX), Err(PoolError::OutOfMemory));
    }

    #[test]
    fn small_requests_share_a_segment() {
        let a = allocator(64 * MIB);
        let b1 = a.allocate(100).unwrap();
        let b2 = a.allocate(1000).unwrap();
        let b3 = a.allocate(MIB).unwrap();
        assert_eq!((b1.size, b2.size, b3.size), (512, 1024, MIB));
        assert_eq!(b2.addr, b1.addr + 512);
        assert_eq!(b3.addr, b2.addr + 1024);
        assert_eq!(log(&a), ["malloc 2097152"]);
        assert_eq!(
            stats(&a),
            (
                512 + 1024 + MIB as u64,
                2 * MIB as u64,
                512 + 1024 + MIB as u64
            )
        );
        // The segment has MiB - 1536 bytes left: a request of a whole MiB needs a new one.
        let b4 = a.allocate(MIB).unwrap();
        assert_eq!(log(&a), ["malloc 2097152"]);
        assert!(b4.addr >= b1.addr + SMALL_SEGMENT);
        assert_eq!(stats(&a).1, 4 * MIB as u64);
    }

    #[test]
    fn large_requests_get_their_own_segment() {
        let a = allocator(64 * MIB);
        let b = a.allocate(3 * MIB).unwrap();
        assert_eq!(b.size, 4 * MIB);
        assert_eq!(log(&a), ["malloc 4194304"]);
        a.release(b.addr);
        // Reused, and split for a smaller large request.
        let c = a.allocate(2 * MIB).unwrap();
        assert_eq!((c.addr, c.size), (b.addr, 2 * MIB));
        let d = a.allocate(2 * MIB).unwrap();
        assert_eq!(d.addr, b.addr + 2 * MIB);
        assert!(log(&a).is_empty());
        // Small requests never come from the large pool.
        let e = a.allocate(512).unwrap();
        assert_eq!(log(&a), ["malloc 2097152"]);
        assert!(e.addr < b.addr || e.addr >= b.addr + 4 * MIB);
    }

    #[test]
    fn freed_blocks_are_reused_best_fit() {
        let a = allocator(64 * MIB);
        let blocks: Vec<Block> = [4096, 512, 1024, 512, 8192]
            .iter()
            .map(|&n| a.allocate(n).unwrap())
            .collect();
        // Free the 4 KiB block and the 1 KiB one: not neighbours, so not merged.
        a.release(blocks[0].addr);
        a.release(blocks[2].addr);
        // 600 bytes round to 1 KiB: the 1 KiB block fits best, not the 4 KiB one.
        assert_eq!(a.allocate(600).unwrap(), blocks[2]);
        // 2 KiB: only the 4 KiB block fits; it is split.
        let c = a.allocate(2048).unwrap();
        assert_eq!(
            c,
            Block {
                addr: blocks[0].addr,
                size: 2048
            }
        );
        let d = a.allocate(2048).unwrap();
        assert_eq!(d.addr, blocks[0].addr + 2048);
        assert_eq!(log(&a), ["malloc 2097152"]);
    }

    #[test]
    fn freed_neighbours_merge_and_whole_segments_are_released() {
        let a = allocator(64 * MIB);
        let small: Vec<Block> = (0..4).map(|_| a.allocate(512).unwrap()).collect();
        let large = a.allocate(2 * MIB).unwrap();
        assert_eq!(log(&a), ["malloc 2097152", "malloc 2097152"]);
        for i in [1, 3, 0] {
            a.release(small[i].addr);
        }
        // Still one block in use: the small segment stays.
        a.empty_cache().unwrap();
        assert!(log(&a).is_empty());
        a.release(small[2].addr);
        // The segment is one free block again: a request of all of it fits without malloc.
        let whole = a.allocate(MIB).unwrap();
        assert_eq!(whole.addr, small[0].addr);
        a.release(whole.addr);
        // The large segment is in use: only the small one is released, after waiting. The peak
        // was reached with the large block and the whole small segment's MiB.
        a.empty_cache().unwrap();
        assert_eq!(log(&a), ["sync", "free 2097152"]);
        assert_eq!(stats(&a), (2 * MIB as u64, 2 * MIB as u64, 3 * MIB as u64));
        a.release(large.addr);
        a.empty_cache().unwrap();
        assert_eq!(log(&a), ["sync", "free 2097152"]);
        assert_eq!((stats(&a).0, stats(&a).1), (0, 0));
        // Nothing to release: no wait either.
        a.empty_cache().unwrap();
        assert!(log(&a).is_empty());
    }

    #[test]
    fn out_of_memory_releases_the_cache_and_retries_once() {
        let a = allocator(8 * MIB);
        let x = a.allocate(4 * MIB).unwrap();
        let y = a.allocate(512).unwrap();
        a.release(x.addr);
        log(&a);
        // 6 MiB don't fit next to the cached 4 MiB and the small segment: the free 4 MiB
        // segment is released, then the request succeeds.
        let z = a.allocate(6 * MIB).unwrap();
        assert_eq!(z.size, 6 * MIB);
        assert_eq!(
            log(&a),
            ["oom 6291456", "sync", "free 4194304", "malloc 6291456"]
        );
        assert_eq!(stats(&a).1, 8 * MIB as u64);
        // Nothing left to release: out of memory, with the counters unchanged.
        let before = stats(&a);
        assert_eq!(a.allocate(4 * MIB), Err(PoolError::OutOfMemory));
        assert_eq!(log(&a), ["oom 4194304", "oom 4194304"]);
        assert_eq!(stats(&a), before);
        // Still usable.
        a.release(y.addr);
        a.release(z.addr);
        assert_eq!(a.allocate(4 * MIB).unwrap().addr, z.addr);
    }

    #[test]
    fn device_failures_are_reported() {
        let a = allocator(8 * MIB);
        a.with_memory(|m| m.fail_mallocs = 1);
        assert_eq!(a.allocate(512), Err(PoolError::Failed("injected".into())));
        assert_eq!(stats(&a), (0, 0, 0));
        let b = a.allocate(512).unwrap();
        a.release(b.addr);
        a.with_memory(|m| m.fail_free = true);
        assert_eq!(
            a.empty_cache(),
            Err(PoolError::Failed("free failed".into()))
        );
    }

    #[test]
    fn the_peak_follows_allocations_and_resets_to_the_current_value() {
        let a = allocator(64 * MIB);
        let x = a.allocate(3 * MIB).unwrap();
        let y = a.allocate(MIB).unwrap();
        a.release(x.addr);
        assert_eq!(stats(&a), (MIB as u64, 6 * MIB as u64, 5 * MIB as u64));
        a.counters().reset_peak();
        assert_eq!(stats(&a).2, MIB as u64);
        a.release(y.addr);
        assert_eq!(stats(&a), (0, 6 * MIB as u64, MIB as u64));
    }

    #[test]
    #[should_panic(expected = "freeing a block twice")]
    fn a_double_free_is_an_internal_error() {
        let a = allocator(8 * MIB);
        let b = a.allocate(512).unwrap();
        a.release(b.addr);
        a.release(b.addr);
    }

    #[test]
    fn blocks_are_freed_from_many_threads() {
        let a = std::sync::Arc::new(allocator(256 * MIB));
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let a = a.clone();
                std::thread::spawn(move || {
                    for i in 0..500 {
                        let b = a
                            .allocate(1 + (t * 7919 + i * 104_729) % (3 * MIB))
                            .unwrap();
                        a.release(b.addr);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(stats(&a).0, 0);
        a.empty_cache().unwrap();
        assert_eq!(stats(&a).1, 0);
        a.with_memory(|m| assert!(m.live.is_empty()));
    }
}
