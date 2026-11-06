//! Device memory: each ROCm device's caching allocator (`backend::caching`) over `hipMalloc`,
//! and the buffers it hands out.

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use super::context::{Context, exec, try_context};
use super::{call, check_index, device_count, ffi, mem_get_info};
use crate::Device;
use crate::backend::caching::{CachingAllocator, MallocError, Memory, PoolError};
use crate::device::MemoryStats;
use crate::storage::AllocError;

/// The device memory of `rocm:index`.
struct Hip {
    index: u32,
}

impl Memory for Hip {
    fn malloc(&mut self, size: usize) -> Result<usize, MallocError> {
        #[cfg(test)]
        if take_injected_malloc_failure() {
            return Err(MallocError::OutOfMemory);
        }
        let mut ptr = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid output location.
        match call(unsafe { ffi::hmr_malloc(self.index as i32, size as u64, &mut ptr) }) {
            Ok(()) => Ok(ptr as usize),
            Err(f) if f.status == ffi::HMR_OUT_OF_MEMORY => Err(MallocError::OutOfMemory),
            Err(f) => Err(MallocError::Failed(f.message)),
        }
    }

    fn free(&mut self, addr: usize) -> Result<(), String> {
        // SAFETY: `addr` came from `hmr_malloc` on this device, and no buffer uses it any more.
        call(unsafe { ffi::hmr_free(self.index as i32, addr as *mut c_void) })
            .map_err(|f| f.message)
    }

    fn synchronize(&mut self) -> Result<(), String> {
        // The allocator has memory, so the context exists: this only looks it up.
        try_context(self.index)
            .and_then(|ctx| ctx.try_synchronize())
            .map_err(|f| f.message)
    }
}

fn allocators() -> &'static [CachingAllocator<Hip>] {
    static ALLOCATORS: OnceLock<Vec<CachingAllocator<Hip>>> = OnceLock::new();
    ALLOCATORS.get_or_init(|| {
        (0..device_count() as u32)
            .map(|index| CachingAllocator::new(Hip { index }))
            .collect()
    })
}

/// The ROCm memory of a storage: a block of its device's caching allocator, zeroed when
/// allocated and returned to the cache when dropped. The device pointer is only kept as a raw
/// pointer: no Rust reference to device memory is ever formed.
#[derive(Debug)]
pub(crate) struct DeviceBuffer {
    /// The context whose stream uses the memory; it outlives the buffer.
    #[cfg_attr(not(test), allow(dead_code))]
    ctx: Arc<Context>,
    index: u32,
    /// Null for 0 bytes, which take no block.
    ptr: *mut u8,
    nbytes: usize,
    /// The block's rounded size, 0 without a block.
    size: usize,
}

// SAFETY: the buffer is a device address that any thread may use with HIP; its bytes are only
// accessed by work queued on the context's stream, inside the device's execution lock.
unsafe impl Send for DeviceBuffer {}
// SAFETY: see above.
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    #[cfg(test)]
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    #[cfg(test)]
    pub fn block_size(&self) -> usize {
        self.size
    }

    #[cfg(test)]
    pub fn context(&self) -> &Context {
        &self.ctx
    }
}

impl Drop for DeviceBuffer {
    /// Returns the block to the cache. Work that still uses it was queued on the stream before,
    /// and work that reuses it will be queued after.
    fn drop(&mut self) {
        if self.size > 0 {
            allocators()[self.index as usize].release(self.ptr as usize);
        }
    }
}

/// `nbytes` of zeroed memory on `rocm:index`, which exists, creating its context if needed.
///
/// # Panics
///
/// Panics for failures other than running out of memory (when creating the context, allocating
/// or zeroing).
#[track_caller]
pub(crate) fn allocate(index: u32, nbytes: usize) -> Result<DeviceBuffer, AllocError> {
    let out_of_memory = AllocError::OutOfMemory {
        device: Device::Rocm(index),
        requested: nbytes,
    };
    let ctx = match try_context(index) {
        Ok(ctx) => ctx,
        Err(f) if f.status == ffi::HMR_OUT_OF_MEMORY => return Err(out_of_memory),
        Err(f) => panic!("{}", f.message),
    };
    if nbytes == 0 {
        return Ok(DeviceBuffer {
            ctx,
            index,
            ptr: std::ptr::null_mut(),
            nbytes,
            size: 0,
        });
    }
    let block = match allocators()[index as usize].allocate(nbytes) {
        Ok(block) => block,
        Err(PoolError::OutOfMemory) => return Err(out_of_memory),
        Err(PoolError::Failed(m)) => panic!("{m}"),
    };
    let buffer = DeviceBuffer {
        ctx,
        index,
        ptr: block.addr as *mut u8,
        nbytes,
        size: block.size,
    };
    // New storage is zero, as on the CPU. The zeroing is queued before any work that uses the
    // buffer, and after any that used the block before.
    let zeroed = exec(index, || {
        #[cfg(test)]
        if take_injected_memset_failure() {
            return Err(super::Failure {
                status: ffi::HMR_OUT_OF_MEMORY,
                message: "hipMemsetAsync: out of memory (injected for testing)".into(),
            });
        }
        buffer.ctx.memset_zero(buffer.ptr, nbytes)
    });
    match zeroed {
        Ok(()) => Ok(buffer),
        // Queuing the fill can need memory too. Dropping the buffer returns the block to the
        // cache (review round 122, R146).
        Err(f) if f.status == ffi::HMR_OUT_OF_MEMORY => Err(out_of_memory),
        Err(f) => panic!("{}", f.message),
    }
}

/// The statistics of `rocm:index`'s allocator, without creating its context.
#[track_caller]
pub(crate) fn memory_stats(index: u32) -> MemoryStats {
    check_index(index);
    allocators()[index as usize].counters().stats()
}

#[track_caller]
pub(crate) fn reset_peak_memory_stats(index: u32) {
    check_index(index);
    allocators()[index as usize].counters().reset_peak();
}

/// Frees the cached memory of `rocm:index` that no buffer uses.
#[track_caller]
pub(crate) fn empty_cache(index: u32) {
    check_index(index);
    match allocators()[index as usize].empty_cache() {
        Ok(()) => {}
        Err(PoolError::Failed(m)) => panic!("{m}"),
        Err(PoolError::OutOfMemory) => unreachable!("releasing memory doesn't allocate"),
    }
}

/// The panic message for a request of `requested` bytes that `rocm:index` refused, after its
/// cache was released, in the style of PyTorch's.
#[track_caller]
pub(crate) fn out_of_memory_message(index: u32, requested: usize) -> String {
    const MIB: f64 = (1u64 << 20) as f64;
    const GIB: f64 = (1u64 << 30) as f64;
    let info = mem_get_info(index);
    let stats = memory_stats(index);
    format!(
        "ROCm out of memory. Tried to allocate {:.2} MiB. rocm:{index} has a total capacity of \
         {:.2} GiB of which {:.2} MiB is free. Of the allocated memory {:.2} MiB is allocated by \
         homura, and {:.2} MiB is reserved but unallocated.",
        requested as f64 / MIB,
        info.total as f64 / GIB,
        info.free as f64 / MIB,
        stats.allocated as f64 / MIB,
        stats.reserved.saturating_sub(stats.allocated) as f64 / MIB,
    )
}

#[cfg(test)]
thread_local! {
    /// `hipMalloc` calls on this thread that report running out of memory without trying.
    static INJECTED_MALLOC_FAILURES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Zeroings on this thread that report running out of memory without being queued.
    static INJECTED_MEMSET_FAILURES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Makes the next `n` segment allocations (`hipMalloc`) on this thread fail for lack of memory.
#[cfg(test)]
pub(super) fn inject_malloc_failures(n: u32) {
    INJECTED_MALLOC_FAILURES.set(n);
}

/// Makes the zeroing of the next `n` new buffers on this thread fail for lack of memory.
#[cfg(test)]
pub(super) fn inject_memset_failures(n: u32) {
    INJECTED_MEMSET_FAILURES.set(n);
}

#[cfg(test)]
fn take(failures: &'static std::thread::LocalKey<std::cell::Cell<u32>>) -> bool {
    let n = failures.get();
    if n > 0 {
        failures.set(n - 1);
    }
    n > 0
}

#[cfg(test)]
fn take_injected_malloc_failure() -> bool {
    take(&INJECTED_MALLOC_FAILURES)
}

#[cfg(test)]
fn take_injected_memset_failure() -> bool {
    take(&INJECTED_MEMSET_FAILURES)
}
