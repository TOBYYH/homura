//! The ROCm backend: safe wrappers around the C ABI of `libhomura_rocm.a` (`csrc/`).
//!
//! Every entry into the library goes through [`call`], which turns a failed status into a
//! [`Failure`] carrying the library's thread-local message, copied before anything else can
//! call into the library on this thread.

mod context;
mod copy;
mod elementwise;
#[allow(clippy::all, unsafe_op_in_unsafe_fn)]
mod ffi;
mod memory;
pub(crate) mod random;

use std::ffi::CStr;
use std::sync::OnceLock;

pub(crate) use context::{context, exec};
pub(crate) use copy::{download, read_bytes, read_host, try_copy, try_fill, upload, write_host};
pub(crate) use elementwise::{binary, binary_scalar, flip, fused, triangle, unary, where_};
pub(crate) use memory::{
    DeviceBuffer, allocate, empty_cache, memory_stats, out_of_memory_message,
    reset_peak_memory_stats,
};

use crate::device::{DeviceProperties, MemInfo};

/// The status of a failure for lack of memory.
pub(crate) const OUT_OF_MEMORY: ffi::hmr_status = ffi::HMR_OUT_OF_MEMORY;

/// A failed call into the ROCm layer.
#[derive(Debug)]
pub(crate) struct Failure {
    /// The `HMR_*` status; distinguishes running out of memory from other failures.
    pub status: ffi::hmr_status,
    pub message: String,
}

/// Checks the status of a call into the library.
fn call(status: ffi::hmr_status) -> Result<(), Failure> {
    if status == ffi::HMR_OK {
        return Ok(());
    }
    // SAFETY: the library returns a NUL-terminated string valid until this thread's next call
    // into it, which can only come after the copy.
    let message = unsafe { CStr::from_ptr(ffi::hmr_last_error()) }
        .to_string_lossy()
        .into_owned();
    Err(Failure { status, message })
}

/// Panics with the message of a failed call.
#[track_caller]
fn expect<T>(result: Result<T, Failure>) -> T {
    match result {
        Ok(v) => v,
        Err(f) => panic!("{}", f.message),
    }
}

/// The number of ROCm devices, counted once.
///
/// # Panics
///
/// Panics if the HIP runtime fails to count them (other than finding no device or driver).
#[track_caller]
pub(crate) fn device_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    if let Some(&n) = COUNT.get() {
        return n;
    }
    // Counted outside `get_or_init`, whose closure would hide the caller's location from a
    // panic; racing threads count the same devices.
    let mut n = 0;
    // SAFETY: `n` is a valid output location.
    expect(call(unsafe { ffi::hmr_device_count(&mut n) }));
    *COUNT.get_or_init(|| n as usize)
}

/// Panics unless `rocm:index` exists.
#[track_caller]
pub(crate) fn check_index(index: u32) {
    let n = device_count();
    if index as usize >= n {
        let s = if n == 1 { "" } else { "s" };
        panic!("rocm:{index} is not available: {n} ROCm device{s} found");
    }
}

/// The properties of `rocm:index`, which exists.
#[track_caller]
pub(crate) fn properties(index: u32) -> DeviceProperties {
    check_index(index);
    expect(try_properties(index))
}

/// The properties of `rocm:index`, or why the library couldn't query them.
fn try_properties(index: u32) -> Result<DeviceProperties, Failure> {
    // SAFETY: all-zero bytes are a valid `hmr_device_properties`.
    let mut p: ffi::hmr_device_properties = unsafe { std::mem::zeroed() };
    // SAFETY: `p` is a valid output location.
    call(unsafe { ffi::hmr_device_properties_get(index as i32, &mut p) })?;
    let text = |chars: &[std::ffi::c_char]| {
        // SAFETY: the library NUL-terminates both arrays, which `zeroed` already filled with NULs.
        unsafe { CStr::from_ptr(chars.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    Ok(DeviceProperties {
        name: text(&p.name),
        arch: text(&p.arch),
        // HIP counts workgroup processors on RDNA GPUs, each of which has two compute units
        // (rocminfo's "Compute Unit"). homura only supports RDNA (gfx12) GPUs.
        compute_units: 2 * p.multiprocessor_count as u32,
        warp_size: p.warp_size as u32,
        total_memory: p.total_memory,
    })
}

/// Free and total memory of `rocm:index`, which exists.
#[track_caller]
pub(crate) fn mem_get_info(index: u32) -> MemInfo {
    check_index(index);
    let (mut free, mut total) = (0, 0);
    // SAFETY: valid output locations.
    expect(call(unsafe {
        ffi::hmr_mem_get_info(index as i32, &mut free, &mut total)
    }));
    MemInfo { free, total }
}

/// Waits for the work queued on `rocm:index`, creating its context if needed.
#[track_caller]
pub(crate) fn synchronize(index: u32) {
    context(index).synchronize();
}

#[cfg(test)]
mod tests;
