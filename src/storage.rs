//! Untyped tensor memory shared by all views of a tensor.

use std::alloc::{self, Layout};
use std::fs::File;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Device;
use crate::device::MemoryStats;

/// Alignment of CPU allocations, enough for any SIMD width.
const CPU_ALIGN: usize = 64;

/// A block of tensor memory together with its device and version counter.
#[derive(Debug)]
pub(crate) struct Storage {
    buffer: Buffer,
    device: Device,
    /// Bumped by every in-place write; shared by all views of this storage.
    version: AtomicU64,
}

#[derive(Debug)]
enum Buffer {
    Cpu(CpuBuffer),
    #[cfg(feature = "rocm")]
    Rocm(crate::backend::rocm::DeviceBuffer),
    /// `nbytes` bytes of a file mapping from `offset`, on the CPU.
    Mapped {
        map: Arc<FileMap>,
        offset: usize,
        nbytes: usize,
    },
    /// The meta device has no data, only a size.
    Meta {
        nbytes: usize,
    },
}

/// Zero-initialized, 64-byte aligned host memory.
#[derive(Debug)]
struct CpuBuffer {
    ptr: NonNull<u8>,
    nbytes: usize,
}

// SAFETY: `CpuBuffer` owns its allocation. All reads and writes go through raw pointers inside
// the CPU backend's global execution lock, so sharing it between threads can't race.
unsafe impl Send for CpuBuffer {}
// SAFETY: see above.
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    fn layout(nbytes: usize) -> Layout {
        Layout::from_size_align(nbytes.max(1), CPU_ALIGN).expect("allocation too large")
    }

    /// `nbytes` zeroed bytes, or `None` if the allocator refuses them.
    fn try_zeroed(nbytes: usize) -> Option<Self> {
        let layout = Self::layout(nbytes);
        // SAFETY: `layout` has a nonzero size.
        let ptr = NonNull::new(unsafe { alloc::alloc_zeroed(layout) })?;
        CPU_COUNTERS.add_unpooled(nbytes as u64);
        Some(CpuBuffer { ptr, nbytes })
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated in `try_zeroed` with the same layout.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), Self::layout(self.nbytes)) };
        CPU_COUNTERS.sub_unpooled(self.nbytes as u64);
    }
}

/// The memory statistics of a device (or of one ROCm device's allocator): bytes allocated to
/// tensors and homura's temporary buffers, bytes held from the device, and the peak of the
/// allocated bytes.
///
/// The three change together under a mutex of their own, held only for the arithmetic (the last
/// lock in `backend::locks`: nothing else is locked while it is held), so that
/// every update and every snapshot is consistent: a reset of the peak can't lose an allocation
/// that happens at the same time, and no snapshot shows the peak below the allocated bytes
/// (review round 122, R144).
#[derive(Debug, Default)]
pub(crate) struct MemoryCounters(std::sync::Mutex<MemoryStats>);

impl MemoryCounters {
    pub(crate) const fn new() -> Self {
        MemoryCounters(std::sync::Mutex::new(MemoryStats {
            allocated: 0,
            reserved: 0,
            peak: 0,
        }))
    }

    fn update(&self, f: impl FnOnce(&mut MemoryStats)) {
        let _rank = crate::backend::locks::acquire(crate::backend::locks::Rank::MemoryCounters);
        f(&mut self.0.lock().unwrap_or_else(|e| e.into_inner()));
    }

    // The caching allocator's updates: it reserves segments and allocates blocks at different
    // times, reserving before it allocates and releasing only segments with nothing allocated.
    #[cfg(any(feature = "rocm", test))]
    pub(crate) fn add(&self, n: u64) {
        self.update(|s| {
            s.allocated += n;
            s.peak = s.peak.max(s.allocated);
        });
    }

    #[cfg(any(feature = "rocm", test))]
    pub(crate) fn sub(&self, n: u64) {
        self.update(|s| s.allocated -= n);
    }

    #[cfg(any(feature = "rocm", test))]
    pub(crate) fn reserve(&self, n: u64) {
        self.update(|s| s.reserved += n);
    }

    #[cfg(any(feature = "rocm", test))]
    pub(crate) fn unreserve(&self, n: u64) {
        self.update(|s| s.reserved -= n);
    }

    /// Allocates and reserves `n` bytes in one update, for a device without a cache (the CPU), so
    /// that no snapshot shows them allocated but not reserved.
    pub(crate) fn add_unpooled(&self, n: u64) {
        self.update(|s| {
            s.reserved += n;
            s.allocated += n;
            s.peak = s.peak.max(s.allocated);
        });
    }

    /// Frees and releases `n` bytes in one update (the inverse of `add_unpooled`).
    pub(crate) fn sub_unpooled(&self, n: u64) {
        self.update(|s| {
            s.allocated -= n;
            s.reserved -= n;
        });
    }

    /// Resets the peak to the bytes allocated now.
    pub(crate) fn reset_peak(&self) {
        self.update(|s| s.peak = s.allocated);
    }

    pub(crate) fn stats(&self) -> MemoryStats {
        let _rank = crate::backend::locks::acquire(crate::backend::locks::Rank::MemoryCounters);
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The CPU's: the buffers homura allocates (file mappings don't count). Without a cache,
/// reserved equals allocated.
static CPU_COUNTERS: MemoryCounters = MemoryCounters::new();

pub(crate) fn cpu_memory_counters() -> &'static MemoryCounters {
    &CPU_COUNTERS
}

/// A request for memory that the device's allocator refused (a size that can't be represented
/// is an argument error instead, which panics before allocating).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AllocError {
    OutOfMemory { device: Device, requested: usize },
}

/// A private copy-on-write mapping of a whole file, shared by the storages of the tensors loaded
/// from it and unmapped when the last of them is dropped. Writes through it stay in this process
/// and never reach the file.
///
/// It is kept as an `MmapRaw`, which only hands out raw pointers: like `CpuBuffer`, the bytes are
/// read and written through pointers inside the CPU backend's execution lock or with exclusive
/// access (`Storage::cpu_bytes_mut`), never through a shared Rust reference to the whole mapping.
#[derive(Debug)]
pub(crate) struct FileMap(memmap2::MmapRaw);

impl FileMap {
    /// Maps all of `file` privately.
    ///
    /// # Safety
    ///
    /// The file must not be truncated or modified, by this or any other process, while the
    /// mapping is alive, that is while any storage made from it is.
    pub unsafe fn new(file: &File) -> std::io::Result<Arc<FileMap>> {
        // SAFETY: the caller keeps the file unchanged while the mapping lives.
        let map = unsafe { memmap2::MmapOptions::new().map_copy(file)? };
        Ok(Arc::new(FileMap(map.into())))
    }

    /// The size of the mapping: the size of the file when it was mapped.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// Zeroed ROCm memory from the device's caching allocator. Unit tests can make it return a
/// placeholder without data instead ([`fake_rocm`]), to reach the ROCm arms of the operations.
///
/// # Panics
///
/// Panics if the device doesn't exist, if homura was built without the `rocm` feature, and for
/// failures of the device other than running out of memory.
#[track_caller]
fn rocm_buffer(nbytes: usize, index: u32) -> Result<Buffer, AllocError> {
    #[cfg(test)]
    if fake_rocm::enabled() {
        return Ok(Buffer::Meta { nbytes });
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = nbytes;
        crate::device::without_rocm(index);
    }
    #[cfg(feature = "rocm")]
    {
        crate::backend::rocm::check_index(index);
        crate::backend::rocm::allocate(index, nbytes).map(Buffer::Rocm)
    }
}

impl Storage {
    /// Allocates `nbytes` of zeroed memory on `device`, or reports that the device's allocator
    /// refused them. This is the only primitive that allocates tensor memory.
    ///
    /// # Panics
    ///
    /// Panics if `nbytes` can't form a CPU allocation (the callers check sizes first, see
    /// `checked_nbytes`), for a ROCm device that doesn't exist or in a build without the `rocm`
    /// feature, and for device failures other than running out of memory.
    #[track_caller]
    pub fn try_new(nbytes: usize, device: Device) -> Result<Storage, AllocError> {
        #[cfg(test)]
        alloc_faults::check(nbytes, device)?;
        let out_of_memory = AllocError::OutOfMemory {
            device,
            requested: nbytes,
        };
        let buffer = match device {
            Device::Cpu => Buffer::Cpu(CpuBuffer::try_zeroed(nbytes).ok_or(out_of_memory)?),
            Device::Meta => Buffer::Meta { nbytes },
            Device::Rocm(i) => rocm_buffer(nbytes, i)?,
        };
        Ok(Storage {
            buffer,
            device,
            version: AtomicU64::new(0),
        })
    }

    /// Allocates `nbytes` of zeroed memory on `device`.
    ///
    /// # Panics
    ///
    /// Like [`try_new`](Storage::try_new), and if the device's allocator refuses the memory: on
    /// a ROCm device with a message that says how much memory there is and how much homura
    /// holds. On the CPU, the process aborts as for any failed allocation of Rust
    /// (`handle_alloc_error`).
    #[track_caller]
    pub fn new(nbytes: usize, device: Device) -> Storage {
        match Storage::try_new(nbytes, device) {
            Ok(s) => s,
            Err(e) => e.raise(),
        }
    }

    /// CPU storage for the `nbytes` bytes of `map` from `offset`, without copying them.
    ///
    /// # Safety
    ///
    /// No other storage may cover any of these bytes of `map`: a storage has exclusive access to
    /// its bytes (see [`Storage::cpu_bytes_mut`]).
    ///
    /// # Panics
    ///
    /// Panics if the bytes are not all inside the mapping.
    pub unsafe fn mapped(map: Arc<FileMap>, offset: usize, nbytes: usize) -> Storage {
        assert!(
            offset
                .checked_add(nbytes)
                .is_some_and(|end| end <= map.len()),
            "storage outside the file mapping"
        );
        Storage {
            buffer: Buffer::Mapped {
                map,
                offset,
                nbytes,
            },
            device: Device::Cpu,
            version: AtomicU64::new(0),
        }
    }

    /// Whether the data is in a file mapping rather than allocated memory.
    #[cfg(test)]
    pub fn is_mapped(&self) -> bool {
        matches!(self.buffer, Buffer::Mapped { .. })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn nbytes(&self) -> usize {
        match &self.buffer {
            Buffer::Cpu(b) => b.nbytes,
            #[cfg(feature = "rocm")]
            Buffer::Rocm(b) => b.nbytes(),
            Buffer::Mapped { nbytes, .. } | Buffer::Meta { nbytes } => *nbytes,
        }
    }

    /// Start of the host memory.
    ///
    /// # Panics
    ///
    /// Panics if the storage is not on the CPU.
    pub fn cpu_ptr(&self) -> *mut u8 {
        match &self.buffer {
            Buffer::Cpu(b) => b.ptr.as_ptr(),
            // SAFETY: `mapped` checked that `offset` is inside the mapping.
            Buffer::Mapped { map, offset, .. } => unsafe { map.0.as_mut_ptr().add(*offset) },
            Buffer::Meta { .. } => panic!("meta tensors have no data"),
            #[cfg(feature = "rocm")]
            Buffer::Rocm(_) => panic!("internal error: ROCm storage has no host memory"),
        }
    }

    /// The ROCm memory of this storage.
    ///
    /// # Panics
    ///
    /// Panics if the storage is not on a ROCm device (an internal error).
    #[cfg(all(feature = "rocm", test))]
    pub fn rocm_buffer(&self) -> &crate::backend::rocm::DeviceBuffer {
        match &self.buffer {
            Buffer::Rocm(b) => b,
            _ => panic!("internal error: not ROCm storage"),
        }
    }

    /// The host memory of a storage that no tensor uses yet, for filling it (from a file, for
    /// example) before it is shared. Exclusive access makes the CPU execution lock unnecessary.
    ///
    /// This is how raw bytes enter tensors, so the caller must uphold the invariant of `Bool`
    /// data before the storage gets a `Bool` tensor: every byte is 0 or 1. CPU kernels read it as
    /// Rust `bool`, for which any other byte is undefined behaviour.
    ///
    /// # Panics
    ///
    /// Panics if the storage is not on the CPU.
    pub fn cpu_bytes_mut(&mut self) -> &mut [u8] {
        let ptr = self.cpu_ptr();
        // SAFETY: an allocated buffer owns `nbytes` initialized (zeroed) bytes at `ptr`; a mapped
        // one has `nbytes` bytes of the file there, inside the mapping and covered by no other
        // storage (see `mapped`). `&mut self` rules out any other access while the slice lives.
        unsafe { std::slice::from_raw_parts_mut(ptr, self.nbytes()) }
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    pub fn bump_version(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// Starts the version of this new storage one past `previous`, the version of the storage it
    /// replaces as a tensor's data (see `Tensor::set_data`), so that a backward pass that saved
    /// the tensor before the replacement sees a version change. Only for a storage that no other
    /// tensor uses yet: the versions of shared storage only ever grow by one per write.
    pub fn continue_version(&self, previous: u64) {
        self.version.store(previous + 1, Ordering::Relaxed);
    }
}

impl AllocError {
    /// Panics (or, on the CPU, aborts) for a refused allocation from an entry point that can't
    /// return the error.
    #[track_caller]
    pub(crate) fn raise(self) -> ! {
        let AllocError::OutOfMemory { device, requested } = self;
        match device {
            Device::Rocm(_i) => {
                #[cfg(feature = "rocm")]
                panic!(
                    "{}",
                    crate::backend::rocm::out_of_memory_message(_i, requested)
                );
                #[cfg(not(feature = "rocm"))]
                unreachable!("ROCm allocations need the rocm feature");
            }
            _ => alloc::handle_alloc_error(CpuBuffer::layout(requested)),
        }
    }
}

/// Injected allocation failures for unit tests: "the n-th allocation on this thread from now
/// fails", for the CPU and for ROCm devices separately, so that tests cover a failure after
/// earlier allocations succeeded.
#[cfg(test)]
pub(crate) mod alloc_faults {
    use std::cell::Cell;

    use super::AllocError;
    use crate::Device;

    thread_local! {
        /// For the CPU and ROCm: the allocations left before the failing one, if any.
        static LEFT: Cell<[Option<u32>; 2]> = const { Cell::new([None, None]) };
    }

    fn slot(device: Device) -> Option<usize> {
        match device {
            Device::Cpu => Some(0),
            Device::Rocm(_) => Some(1),
            Device::Meta => None,
        }
    }

    /// Makes the `n`-th next allocation (1 for the next) on this thread on devices of the kind
    /// of `device` fail.
    pub(crate) fn fail_nth(device: Device, n: u32) {
        assert!(n >= 1);
        let i = slot(device).expect("meta allocations don't fail");
        let mut left = LEFT.get();
        left[i] = Some(n - 1);
        LEFT.set(left);
    }

    /// Clears the injected failures of this thread.
    pub(crate) fn clear() {
        LEFT.set([None, None]);
    }

    pub(super) fn check(nbytes: usize, device: Device) -> Result<(), AllocError> {
        let Some(i) = slot(device) else { return Ok(()) };
        let mut left = LEFT.get();
        match left[i] {
            Some(0) => {
                left[i] = None;
                LEFT.set(left);
                Err(AllocError::OutOfMemory {
                    device,
                    requested: nbytes,
                })
            }
            Some(n) => {
                left[i] = Some(n - 1);
                LEFT.set(left);
                Ok(())
            }
            None => Ok(()),
        }
    }
}

/// Placeholder ROCm storages for unit tests, which have no data: every kernel on them must
/// panic, which shows that operations reach the ROCm arm of the dispatch layer instead of
/// treating ROCm like the meta device.
#[cfg(test)]
pub(crate) mod fake_rocm {
    use std::cell::Cell;

    thread_local! {
        static ENABLED: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn enabled() -> bool {
        ENABLED.get()
    }

    /// Runs `f` with ROCm allocations on this thread returning placeholders.
    pub(crate) fn with<R>(f: impl FnOnce() -> R) -> R {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                ENABLED.set(false);
            }
        }
        ENABLED.set(true);
        let _reset = Reset;
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oom(device: Device, requested: usize) -> Result<(), AllocError> {
        Err(AllocError::OutOfMemory { device, requested })
    }

    #[test]
    fn injected_failures_hit_the_nth_allocation_of_their_kind() {
        alloc_faults::fail_nth(Device::Cpu, 2);
        assert!(Storage::try_new(8, Device::Cpu).is_ok());
        // Other kinds don't count.
        assert!(Storage::try_new(8, Device::Meta).is_ok());
        fake_rocm::with(|| assert!(Storage::try_new(8, Device::Rocm(0)).is_ok()));
        assert_eq!(
            Storage::try_new(24, Device::Cpu).map(drop),
            oom(Device::Cpu, 24)
        );
        assert!(Storage::try_new(24, Device::Cpu).is_ok());
        // ROCm devices count apart from the CPU, all of them together.
        alloc_faults::fail_nth(Device::Rocm(0), 1);
        assert!(Storage::try_new(8, Device::Cpu).is_ok());
        let failed = fake_rocm::with(|| Storage::try_new(16, Device::Rocm(3)).map(drop));
        assert_eq!(failed, oom(Device::Rocm(3), 16));
        fake_rocm::with(|| assert!(Storage::try_new(16, Device::Rocm(3)).is_ok()));
        alloc_faults::fail_nth(Device::Cpu, 1);
        alloc_faults::clear();
        assert!(Storage::try_new(8, Device::Cpu).is_ok());
    }

    #[test]
    fn cpu_buffers_count_in_the_cpu_statistics() {
        // Other tests allocate concurrently: only check what this thread's buffer adds while it
        // lives, through the peak, which never goes down without a reset.
        let s = Storage::new(1 << 20, Device::Cpu);
        assert!(cpu_memory_counters().stats().peak >= 1 << 20);
        drop(s);
        let mapped = MemoryCounters::new();
        mapped.add(10);
        mapped.reserve(10);
        mapped.add(5);
        mapped.sub(10);
        assert_eq!(
            mapped.stats(),
            MemoryStats {
                allocated: 5,
                reserved: 10,
                peak: 15
            }
        );
        mapped.reset_peak();
        assert_eq!(mapped.stats().peak, 5);
    }

    /// A reset racing with allocations never leaves the peak below what is allocated (review
    /// round 122, R144). One thread allocates and frees, the only one that changes `allocated`,
    /// and checks the peak after each allocation; others reset the peak all the time.
    #[test]
    fn a_concurrent_reset_keeps_the_peak_at_least_the_allocated_bytes() {
        let c = std::sync::Arc::new(MemoryCounters::new());
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let resetters: Vec<_> = (0..3)
            .map(|_| {
                let (c, done) = (c.clone(), done.clone());
                std::thread::spawn(move || {
                    while !done.load(Ordering::Relaxed) {
                        c.reset_peak();
                    }
                })
            })
            .collect();
        let mut low = 0;
        for _ in 0..200_000 {
            c.add(512);
            let s = c.stats();
            if s.peak < s.allocated {
                low += 1;
            }
            c.sub(512);
        }
        done.store(true, Ordering::Relaxed);
        for r in resetters {
            r.join().unwrap();
        }
        assert_eq!(
            low, 0,
            "the peak fell below the allocated bytes {low} times"
        );
    }

    #[test]
    fn a_cpu_snapshot_never_shows_more_allocated_than_reserved() {
        let c = std::sync::Arc::new(MemoryCounters::new());
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..3)
            .map(|_| {
                let (c, done) = (c.clone(), done.clone());
                std::thread::spawn(move || {
                    let mut over = 0u64;
                    while !done.load(Ordering::Relaxed) {
                        let s = c.stats();
                        if s.allocated > s.reserved {
                            over += 1;
                        }
                    }
                    over
                })
            })
            .collect();
        for _ in 0..200_000 {
            c.add_unpooled(512);
            c.sub_unpooled(512);
        }
        done.store(true, Ordering::Relaxed);
        let over: u64 = readers.into_iter().map(|r| r.join().unwrap()).sum();
        assert_eq!(
            over, 0,
            "a snapshot showed more allocated than reserved bytes {over} times"
        );
        assert_eq!(c.stats().allocated, 0);
        assert_eq!(c.stats().reserved, 0);
        assert_eq!(c.stats().peak, 512);
    }

    #[test]
    #[should_panic(expected = "allocation too large")]
    fn a_size_that_cant_form_a_cpu_layout_is_not_out_of_memory() {
        let _ = Storage::try_new(isize::MAX as usize, Device::Cpu);
    }
}
