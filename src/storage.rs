//! Untyped tensor memory shared by all views of a tensor.

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Device;

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

    fn zeroed(nbytes: usize) -> Self {
        let layout = Self::layout(nbytes);
        // SAFETY: `layout` has a nonzero size.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::handle_alloc_error(layout);
        };
        CpuBuffer { ptr, nbytes }
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated in `zeroed` with the same layout.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), Self::layout(self.nbytes)) };
    }
}

impl Storage {
    /// Allocates `nbytes` of zeroed memory on `device`.
    ///
    /// # Panics
    ///
    /// Panics for ROCm devices, which are not implemented yet.
    #[track_caller]
    pub fn new(nbytes: usize, device: Device) -> Storage {
        let buffer = match device {
            Device::Cpu => Buffer::Cpu(CpuBuffer::zeroed(nbytes)),
            Device::Meta => Buffer::Meta { nbytes },
            Device::Rocm(_) => panic!("the ROCm backend is not implemented yet"),
        };
        Storage {
            buffer,
            device,
            version: AtomicU64::new(0),
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn nbytes(&self) -> usize {
        match &self.buffer {
            Buffer::Cpu(b) => b.nbytes,
            Buffer::Meta { nbytes } => *nbytes,
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
            Buffer::Meta { .. } => panic!("meta tensors have no data"),
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
        match &mut self.buffer {
            // SAFETY: the buffer owns `nbytes` initialized (zeroed) bytes at `ptr`, and `&mut self`
            // rules out any other access while the slice lives.
            Buffer::Cpu(b) => unsafe { std::slice::from_raw_parts_mut(b.ptr.as_ptr(), b.nbytes) },
            Buffer::Meta { .. } => panic!("meta tensors have no data"),
        }
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
