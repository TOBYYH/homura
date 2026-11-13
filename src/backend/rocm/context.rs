//! The registry of ROCm device contexts and their execution locks.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::{Failure, call, check_index, device_count, expect, ffi};
use crate::backend::locks::{self, Rank};

/// A device's execution context: its stream, created on first use and kept until the process
/// exits (destroying it at exit could race with the HIP runtime's own teardown, so the registry
/// is never dropped).
#[derive(Debug)]
pub(crate) struct Context {
    #[cfg_attr(not(test), allow(dead_code))]
    index: u32,
    raw: ffi::hmr_context_t,
    /// Serializes submissions to the stream, see [`exec`].
    exec: Mutex<()>,
}

// SAFETY: the context is a device index and a HIP stream, which HIP lets any thread use; the
// library doesn't mutate it after creation.
unsafe impl Send for Context {}
// SAFETY: see above.
unsafe impl Sync for Context {}

impl Context {
    /// Waits until all work queued on the stream has finished.
    #[track_caller]
    pub fn synchronize(&self) {
        expect(self.try_synchronize());
    }

    /// Like [`synchronize`](Context::synchronize), reporting a failure instead of panicking.
    pub fn try_synchronize(&self) -> Result<(), Failure> {
        // SAFETY: `raw` is a live context.
        call(unsafe { ffi::hmr_context_synchronize(self.raw) })
    }

    /// Queues setting `nbytes` of device memory at `ptr` to the byte `value` on the stream.
    /// Call it inside the device's execution lock.
    pub(super) fn memset(&self, ptr: *mut u8, value: u8, nbytes: usize) -> Result<(), Failure> {
        // SAFETY: `raw` is a live context, and the caller passes device memory of this device
        // that the bytes fit in.
        call(unsafe { ffi::hmr_memset(self.raw, ptr.cast(), value, nbytes as u64) })
    }

    /// Copies `dst.len()` bytes of device memory at `src` into `dst`, after the work queued
    /// before, and returns when they are there.
    pub(super) fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<(), Failure> {
        // SAFETY: `raw` is a live context; `src` holds `dst.len()` bytes of this device.
        call(unsafe {
            ffi::hmr_copy_to_host(
                self.raw,
                dst.as_mut_ptr().cast(),
                src.cast(),
                dst.len() as u64,
            )
        })
    }

    /// Copies `src` into device memory at `dst`, after the work queued before, and returns when
    /// the copy is complete.
    pub(super) fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<(), Failure> {
        // SAFETY: `raw` is a live context; `dst` holds `src.len()` bytes of this device.
        call(unsafe {
            ffi::hmr_copy_from_host(self.raw, dst.cast(), src.as_ptr().cast(), src.len() as u64)
        })
    }

    /// Queues `hmr_copy` (a strided copy with conversion). Call it inside the device's execution
    /// lock, with descriptions of this device's memory.
    pub(super) fn copy(
        &self,
        dst: &ffi::hmr_tensor_desc,
        src: &ffi::hmr_tensor_desc,
    ) -> Result<(), Failure> {
        // SAFETY: `raw` is a live context; the descriptions lie inside device memory of this
        // device, and `dst`'s elements are distinct and don't partially overlap `src`'s.
        call(unsafe { ffi::hmr_copy(self.raw, dst, src) })
    }

    /// Queues `hmr_fill`. Call it inside the device's execution lock.
    pub(super) fn fill(&self, dst: &ffi::hmr_tensor_desc, bits: u64) -> Result<(), Failure> {
        // SAFETY: `raw` is a live context; the description lies inside device memory of this
        // device, with distinct elements.
        call(unsafe { ffi::hmr_fill(self.raw, dst, bits) })
    }

    /// Queues copying `nbytes` from `src` to `dst`, device memory of this device that doesn't
    /// overlap. Call it inside the device's execution lock.
    pub(super) fn copy_device(
        &self,
        dst: *mut u8,
        src: *const u8,
        nbytes: usize,
    ) -> Result<(), Failure> {
        // SAFETY: `raw` is a live context; both ranges are device memory of this device.
        call(unsafe { ffi::hmr_copy_device(self.raw, dst.cast(), src.cast(), nbytes as u64) })
    }

    /// The library's handle, for the kernel wrappers that call the library themselves. Calls
    /// that queue work on the stream belong inside the device's execution lock.
    pub(super) fn raw(&self) -> ffi::hmr_context_t {
        self.raw
    }

    #[cfg(test)]
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl Drop for Context {
    /// Destroys the stream. Contexts in the registry live until the process exits (statics are
    /// never dropped), so this only runs for contexts that tests create outside it.
    fn drop(&mut self) {
        // SAFETY: `raw` is a live context, and nothing uses it after this.
        let _ = call(unsafe { ffi::hmr_context_destroy(self.raw) });
    }
}

type Slot = Mutex<Option<Arc<Context>>>;

fn registry() -> &'static [Slot] {
    static SLOTS: OnceLock<Vec<Slot>> = OnceLock::new();
    SLOTS.get_or_init(|| (0..device_count()).map(|_| Mutex::new(None)).collect())
}

/// The context of `rocm:index`, which exists, created on first use. A failed creation is not
/// cached: the next call tries again.
pub(crate) fn try_context(index: u32) -> Result<Arc<Context>, Failure> {
    get_or_create(&registry()[index as usize], index)
}

fn get_or_create(slot: &Slot, index: u32) -> Result<Arc<Context>, Failure> {
    let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ctx) = &*slot {
        return Ok(ctx.clone());
    }
    let mut raw = std::ptr::null_mut();
    // SAFETY: `raw` is a valid output location.
    call(unsafe { ffi::hmr_context_create(index as i32, &mut raw) })?;
    // The context owns the stream from here on: on a failure below, dropping it destroys the
    // stream, and the error is returned rather than raised (review round 120, R142).
    let ctx = Arc::new(Context {
        index,
        raw,
        exec: Mutex::new(()),
    });
    let arch = super::try_properties(index)?.arch;
    warn_if_untested(index, &arch);
    *slot = Some(ctx.clone());
    Ok(ctx)
}

/// The context of `rocm:index`.
///
/// # Panics
///
/// Panics if the device doesn't exist or its context can't be created.
#[track_caller]
pub(crate) fn context(index: u32) -> Arc<Context> {
    check_index(index);
    expect(try_context(index))
}

/// Warns once per process when a context is created on a GPU homura has only been compiled for.
fn warn_if_untested(index: u32, arch: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if arch != "gfx1201" && !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "warning: homura has not been tested on {arch} GPUs (rocm:{index}); only gfx1201 is \
             supported"
        );
    }
}

thread_local! {
    /// The devices whose execution lock this thread holds (at most one, see `locks`).
    static HOLDS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` while holding the execution lock of `rocm:index`. Reentrant per device: kernels
/// submitted from `f` run inside the same critical section, so a write and its version bump are
/// atomic as on the CPU.
///
/// # Panics
///
/// Panics if the device doesn't exist or its context can't be created, and in debug builds if
/// this thread holds a lock that comes after it in the lock order (`locks`).
#[track_caller]
pub(crate) fn exec<R>(index: u32, f: impl FnOnce() -> R) -> R {
    if HOLDS.with_borrow(|h| h.contains(&index)) {
        return f();
    }
    let ctx = context(index);
    let _rank = locks::acquire(Rank::RocmExec(index));
    let _guard = ctx.exec.lock().unwrap_or_else(|e| e.into_inner());
    struct Release(u32);
    impl Drop for Release {
        fn drop(&mut self) {
            HOLDS.with_borrow_mut(|h| h.retain(|&d| d != self.0));
        }
    }
    HOLDS.with_borrow_mut(|h| h.push(index));
    let _release = Release(index);
    f()
}

/// Creates a context in a slot of its own, outside the registry, for tests of the creation.
#[cfg(test)]
pub(super) fn create_in(slot: &Slot, index: u32) -> Result<Arc<Context>, Failure> {
    get_or_create(slot, index)
}

#[cfg(test)]
pub(super) type TestSlot = Slot;
