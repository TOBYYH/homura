//! Untyped tensor memory shared by all views of a tensor.

use std::alloc::{self, Layout};
use std::fs::File;
use std::ptr::NonNull;
use std::sync::Arc;
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
