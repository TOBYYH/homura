//! The backends and the dispatch layer in front of them.
//!
//! Operations call the free functions of this module, which have the signatures of the CPU
//! backend's kernels and pick the backend by device. The meta device has no data: callers return
//! before dispatching for meta tensors, which the dispatch functions treat as an internal error.
//! ROCm kernels are added step by step; until then the ROCm arm of a kernel panics. Moving data
//! (reading, writing, copying, filling and the transfers between devices), the elementwise
//! kernels, the reductions, indexing and pooling work on ROCm.

#[cfg(any(feature = "rocm", test))]
pub(crate) mod caching;
pub(crate) mod cpu;
pub(crate) mod locks;
#[cfg(feature = "rocm")]
pub(crate) mod rocm;

pub(crate) use cpu::Fused;

#[cfg(test)]
mod dispatch_tests;

use crate::ops::{BinaryOp, Pool2d, ReduceOp, UnaryOp};
use crate::storage::AllocError;
use crate::tensor::{Tensor, TensorData};
use crate::{Device, Element, Scalar};

/// Panics for a kernel the ROCm backend doesn't implement yet.
#[cold]
#[track_caller]
fn rocm_todo(kernel: &str, index: u32) -> ! {
    #[cfg(not(feature = "rocm"))]
    {
        let _ = kernel;
        crate::device::without_rocm(index)
    }
    #[cfg(feature = "rocm")]
    panic!("{kernel} is not implemented on ROCm yet (rocm:{index})")
}

/// Panics for a kernel called on meta tensors, which callers must not do.
#[cold]
#[track_caller]
fn meta(kernel: &str) -> ! {
    panic!("internal error: {kernel} dispatched for meta tensors, which have no data")
}

/// `match` on the device of `$data`: the CPU kernel `cpu::$kernel($args)`, or the panics above.
macro_rules! dispatch {
    ($data:expr, $kernel:ident($($arg:expr),* $(,)?)) => {
        match $data.device() {
            Device::Cpu => cpu::$kernel($($arg),*),
            Device::Rocm(i) => rocm_todo(stringify!($kernel), i),
            Device::Meta => meta(stringify!($kernel)),
        }
    };
}

/// [`dispatch`] for the kernels that the ROCm backend implements: `rocm::$kernel($args)`.
macro_rules! dispatch_rocm {
    ($data:expr, $kernel:ident($($arg:expr),* $(,)?)) => {
        match $data.device() {
            Device::Cpu => cpu::$kernel($($arg),*),
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => rocm::$kernel($($arg),*),
            #[cfg(not(feature = "rocm"))]
            Device::Rocm(i) => rocm_todo(stringify!($kernel), i),
            Device::Meta => meta(stringify!($kernel)),
        }
    };
}

/// Runs `f` while holding the execution lock of `device` (see `cpu::exec`), so that writes and
/// version bumps in `f` are atomic. Meta tensors have no data and need no lock.
#[track_caller]
pub(crate) fn exec<R>(device: Device, f: impl FnOnce() -> R) -> R {
    match device {
        Device::Cpu => cpu::exec(f),
        #[cfg(feature = "rocm")]
        Device::Rocm(i) => rocm::exec(i, f),
        #[cfg(not(feature = "rocm"))]
        Device::Rocm(i) => crate::device::without_rocm(i),
        Device::Meta => f(),
    }
}

#[track_caller]
pub(crate) fn write_host<T: Element>(dst: &TensorData, values: &[T]) {
    dispatch_rocm!(dst, write_host(dst, values))
}

#[track_caller]
pub(crate) fn read_bytes(src: &TensorData) -> Vec<u8> {
    dispatch_rocm!(src, read_bytes(src))
}

#[track_caller]
pub(crate) fn read_host<T: Element>(src: &TensorData) -> Vec<T> {
    dispatch_rocm!(src, read_host(src))
}

/// A device's failure to run a data operation, not reported yet. Operations inside the closure of
/// an execution lock return it, to be reported after the closure, where `#[track_caller]` still
/// names the caller: closures don't pass it on (review round 124, R149). Only ROCm devices fail.
#[cfg(feature = "rocm")]
#[derive(Debug)]
pub(crate) struct DeviceFailure {
    failure: rocm::Failure,
    device: Device,
    /// The bytes of the operation's device operand, reported if the device ran out of memory.
    requested: usize,
}

/// See the `rocm` variant: without ROCm, no device fails.
#[cfg(not(feature = "rocm"))]
#[derive(Debug)]
pub(crate) enum DeviceFailure {}

impl DeviceFailure {
    /// Running out of memory as the [`AllocError`] for the operand's bytes.
    ///
    /// # Panics
    ///
    /// Panics with the device's message for any other failure.
    #[track_caller]
    pub(crate) fn into_alloc_error(self) -> AllocError {
        #[cfg(feature = "rocm")]
        {
            if self.failure.status == rocm::OUT_OF_MEMORY {
                return AllocError::OutOfMemory {
                    device: self.device,
                    requested: self.requested,
                };
            }
            panic!("{}", self.failure.message)
        }
        #[cfg(not(feature = "rocm"))]
        match self {}
    }

    /// Panics for the failure: like a refused allocation for running out of memory, else with
    /// the device's message.
    #[track_caller]
    pub(crate) fn raise(self) -> ! {
        self.into_alloc_error().raise()
    }
}

/// `result` of a ROCm operation on `device_data` (the operand on the device).
#[cfg(feature = "rocm")]
fn rocm_result(
    result: Result<(), rocm::Failure>,
    device_data: &TensorData,
) -> Result<(), DeviceFailure> {
    result.map_err(|failure| DeviceFailure {
        failure,
        device: device_data.device(),
        requested: device_data.layout.num_elements() * device_data.dtype.size_in_bytes(),
    })
}

/// Copies `src` into `dst` on the same device, broadcasting and converting the dtype.
///
/// # Panics
///
/// Panics for any failure of the device, running out of memory included (see [`try_copy`]).
#[track_caller]
pub(crate) fn copy(dst: &TensorData, src: &TensorData) {
    if let Err(f) = copy_unreported(dst, src) {
        f.raise()
    }
}

/// [`copy`], returning the device's lack of memory instead of panicking for it.
#[track_caller]
pub(crate) fn try_copy(dst: &TensorData, src: &TensorData) -> Result<(), AllocError> {
    match copy_unreported(dst, src) {
        Ok(()) => Ok(()),
        Err(f) => Err(f.into_alloc_error()),
    }
}

/// [`copy`], returning every failure of the device unreported, for callers inside the closure of
/// an execution lock. Nothing is written when it fails.
#[track_caller]
pub(crate) fn copy_unreported(dst: &TensorData, src: &TensorData) -> Result<(), DeviceFailure> {
    match dst.device() {
        Device::Cpu => {
            cpu::copy(dst, src);
            Ok(())
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => rocm_result(rocm::try_copy(dst, src), dst),
        #[cfg(not(feature = "rocm"))]
        Device::Rocm(i) => rocm_todo("copy", i),
        Device::Meta => meta("copy"),
    }
}

/// Sets every element of `dst` to `value`.
///
/// # Panics
///
/// Panics for any failure of the device, running out of memory included (see [`try_fill`]).
#[track_caller]
pub(crate) fn fill(dst: &TensorData, value: Scalar) {
    if let Err(f) = fill_unreported(dst, value) {
        f.raise()
    }
}

/// [`fill`], returning the device's lack of memory instead of panicking for it.
#[track_caller]
pub(crate) fn try_fill(dst: &TensorData, value: Scalar) -> Result<(), AllocError> {
    match fill_unreported(dst, value) {
        Ok(()) => Ok(()),
        Err(f) => Err(f.into_alloc_error()),
    }
}

/// [`fill`], returning every failure of the device unreported, see [`copy_unreported`].
#[track_caller]
pub(crate) fn fill_unreported(dst: &TensorData, value: Scalar) -> Result<(), DeviceFailure> {
    match dst.device() {
        Device::Cpu => {
            cpu::fill(dst, value);
            Ok(())
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => rocm_result(rocm::try_fill(dst, value), dst),
        #[cfg(not(feature = "rocm"))]
        Device::Rocm(i) => rocm_todo("fill", i),
        Device::Meta => meta("fill"),
    }
}

/// `data` with its dtype and shape in new contiguous storage on `device`, another device than
/// its own (not meta), or why the memory couldn't be allocated. See [`try_transfer_into`].
#[track_caller]
pub(crate) fn try_transfer(data: &TensorData, device: Device) -> Result<TensorData, AllocError> {
    let dst = Tensor::try_empty_on(data.layout.shape.clone(), data.dtype, device)?.data();
    try_transfer_into(&dst, data)?;
    Ok(dst)
}

/// Copies `src` into `dst`, a contiguous tensor with its dtype and shape on another device (not
/// meta), or reports why memory for the copy couldn't be allocated: a contiguous copy of `src` on
/// its device, or the host staging between two ROCm devices. CPU and ROCm data move through
/// pageable host memory, synchronously; data moves between two ROCm devices through the host,
/// one device's execution lock at a time.
///
/// # Panics
///
/// Panics for failures of the devices other than running out of memory.
#[track_caller]
pub(crate) fn try_transfer_into(dst: &TensorData, src: &TensorData) -> Result<(), AllocError> {
    debug_assert!(dst.layout.is_contiguous());
    debug_assert_eq!(
        (&dst.layout.shape, dst.dtype),
        (&src.layout.shape, src.dtype)
    );
    match (src.device(), dst.device()) {
        (Device::Cpu, Device::Rocm(_)) | (Device::Rocm(_), Device::Cpu) => {
            host_device_copy(dst, &contiguous(src)?)
        }
        (Device::Rocm(_), Device::Rocm(_)) => {
            host_device_copy(dst, &try_transfer(src, Device::Cpu)?)
        }
        (from, to) => panic!("internal error: a transfer from {from} to {to}"),
    }
}

/// `data`, or a contiguous copy of it on its device.
#[track_caller]
fn contiguous(data: &TensorData) -> Result<TensorData, AllocError> {
    if data.layout.is_contiguous() {
        Ok(data.clone())
    } else {
        crate::ops::try_clone_data(data)
    }
}

/// Copies the contiguous `src` into the contiguous `dst` of the same dtype and size, one on the
/// CPU and the other on a ROCm device.
#[cfg(feature = "rocm")]
#[track_caller]
fn host_device_copy(dst: &TensorData, src: &TensorData) -> Result<(), AllocError> {
    let (result, device_data) = if dst.device() == Device::Cpu {
        (rocm::download(dst, src), src)
    } else {
        (rocm::upload(dst, src), dst)
    };
    // HIP stages pageable copies through buffers of its own, which it may fail to allocate.
    match rocm_result(result, device_data) {
        Ok(()) => Ok(()),
        Err(f) => Err(f.into_alloc_error()),
    }
}

#[cfg(not(feature = "rocm"))]
#[track_caller]
fn host_device_copy(dst: &TensorData, src: &TensorData) -> Result<(), AllocError> {
    let (kernel, device) = if dst.device() == Device::Cpu {
        ("download", src.device())
    } else {
        ("upload", dst.device())
    };
    match device {
        Device::Rocm(i) => rocm_todo(kernel, i),
        d => unreachable!("a transfer to {d}"),
    }
}

#[track_caller]
pub(crate) fn unary(op: UnaryOp, out: &TensorData, src: &TensorData) {
    dispatch_rocm!(out, unary(op, out, src))
}

#[track_caller]
pub(crate) fn binary(op: BinaryOp, out: &TensorData, a: &TensorData, b: &TensorData) {
    dispatch_rocm!(out, binary(op, out, a, b))
}

#[track_caller]
pub(crate) fn binary_scalar(
    op: BinaryOp,
    out: &TensorData,
    a: &TensorData,
    s: f64,
    scalar_first: bool,
) {
    dispatch_rocm!(out, binary_scalar(op, out, a, s, scalar_first))
}

#[track_caller]
pub(crate) fn fused(
    op: Fused,
    out: &TensorData,
    a: &TensorData,
    b: &TensorData,
    c: Option<&TensorData>,
) {
    dispatch_rocm!(out, fused(op, out, a, b, c))
}

#[track_caller]
pub(crate) fn where_(out: &TensorData, cond: &TensorData, x: &TensorData, y: &TensorData) {
    dispatch_rocm!(out, where_(out, cond, x, y))
}

#[track_caller]
pub(crate) fn softmax(out: &TensorData, src: &TensorData, dim: usize, log: bool) {
    dispatch_rocm!(out, softmax(out, src, dim, log))
}

/// How an indexing kernel treats the positions in its index tensor, a tensor of `I32` or `I64`
/// positions along a dimension of `size`. The kernels check them with the device's execution lock
/// held, then use the same values: a check and its use can't see different data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bounds {
    /// In `0..size`, else the kernel panics with PyTorch's message.
    NonNegative,
    /// In `-size..size`, negative positions counting from the end; else the kernel panics.
    AllowNegative,
    /// Positions an earlier step made and checked (advanced indexing): in `0..size`, unchecked.
    Valid,
}

/// The message of a position out of range.
#[track_caller]
pub(crate) fn out_of_bounds(value: i64, dim: usize, size: usize) -> String {
    format!("index {value} is out of bounds for dimension {dim} with size {size}")
}

#[track_caller]
pub(crate) fn index_select(
    out: &TensorData,
    src: &TensorData,
    dim: usize,
    index: &TensorData,
    bounds: Bounds,
) {
    dispatch_rocm!(out, index_select(out, src, dim, index, bounds))
}

/// Copies slice `i` of `src` along `dim` into slice `index[i]` of `out`, the last of repeated
/// positions winning: the overwriting `index_put`. `index` holds [`Bounds::Valid`] positions.
#[track_caller]
pub(crate) fn index_overwrite(out: &TensorData, dim: usize, index: &TensorData, src: &TensorData) {
    dispatch_rocm!(out, index_overwrite(out, dim, index, src))
}

/// The coordinates of the true elements of the bool tensor `mask` in row-major order: one `I64`
/// vector per dimension, on the device of `mask`.
#[track_caller]
pub(crate) fn mask_coordinates(mask: &TensorData) -> Vec<TensorData> {
    dispatch_rocm!(mask, mask_coordinates(mask))
}

#[track_caller]
pub(crate) fn index_add(
    out: &TensorData,
    dim: usize,
    index: &TensorData,
    src: &TensorData,
    bounds: Bounds,
) {
    dispatch_rocm!(out, index_add(out, dim, index, src, bounds))
}

/// `out[.., index[p], ..] += src[p]` for every position `p` of `index`, whose values are checked
/// to be positions along `dim` (non-negative).
#[track_caller]
pub(crate) fn scatter_add(out: &TensorData, dim: usize, index: &TensorData, src: &TensorData) {
    dispatch_rocm!(out, scatter_add(out, dim, index, src))
}

/// `out[p] = src[.., index[p], ..]`, the values of `index` checked to be positions along `dim`
/// (non-negative).
#[track_caller]
pub(crate) fn gather(out: &TensorData, src: &TensorData, dim: usize, index: &TensorData) {
    dispatch_rocm!(out, gather(out, src, dim, index))
}

/// The positions of advanced indexing: checks each index of `coords` (`None` for one without
/// values, on the meta device), in order, to lie in `-size..size` for its size in `sizes`, and
/// returns the first position out of range as `(k, value)` for index `k`. Without one, and if
/// `out` is given, writes to `out` (`I64`, the shape the indices broadcast to) the linear
/// position in the indexed subspace (row-major over `sizes`) of every element, negative
/// coordinates counting from the end. The indices are on the device of `out`, and are checked
/// and read with its execution lock held.
#[track_caller]
pub(crate) fn index_positions(
    out: Option<&TensorData>,
    coords: &[Option<TensorData>],
    sizes: &[usize],
) -> Option<(usize, i64)> {
    let first = out.or(coords.iter().flatten().next())?;
    dispatch_rocm!(first, index_positions(out, coords, sizes))
}

/// The first value of the integer tensor `index`, in row-major order, outside `low..high` and,
/// if `exempt` is given, not equal to it.
#[track_caller]
pub(crate) fn out_of_range(
    index: &TensorData,
    low: i64,
    high: i64,
    exempt: Option<i64>,
) -> Option<i64> {
    dispatch_rocm!(index, out_of_range(index, low, high, exempt))
}

#[track_caller]
pub(crate) fn triangle(out: &TensorData, src: &TensorData, diagonal: i64, upper: bool) {
    dispatch_rocm!(out, triangle(out, src, diagonal, upper))
}

#[track_caller]
pub(crate) fn flip(out: &TensorData, src: &TensorData, mask: &[bool]) {
    dispatch_rocm!(out, flip(out, src, mask))
}

#[track_caller]
pub(crate) fn unfold_backward(
    out: &TensorData,
    grad: &TensorData,
    dim: usize,
    size: usize,
    step: usize,
) {
    dispatch_rocm!(out, unfold_backward(out, grad, dim, size, step))
}

#[track_caller]
pub(crate) fn max_pool2d(out: &TensorData, indices: &TensorData, input: &TensorData, g: &Pool2d) {
    dispatch_rocm!(out, max_pool2d(out, indices, input, g))
}

#[track_caller]
pub(crate) fn max_pool2d_backward(out: &TensorData, grad: &TensorData, indices: &TensorData) {
    dispatch_rocm!(out, max_pool2d_backward(out, grad, indices))
}

#[track_caller]
pub(crate) fn adaptive_avg_pool2d(out: &TensorData, input: &TensorData) {
    dispatch_rocm!(out, adaptive_avg_pool2d(out, input))
}

#[track_caller]
pub(crate) fn adaptive_avg_pool2d_backward(out: &TensorData, grad: &TensorData) {
    dispatch_rocm!(out, adaptive_avg_pool2d_backward(out, grad))
}

#[track_caller]
pub(crate) fn reduce(op: ReduceOp, out: &TensorData, src: &TensorData, mask: &[bool]) {
    dispatch_rocm!(out, reduce(op, out, src, mask))
}

#[track_caller]
pub(crate) fn var(
    out: &TensorData,
    src: &TensorData,
    mask: &[bool],
    correction: f64,
    take_sqrt: bool,
) {
    dispatch_rocm!(out, var(out, src, mask, correction, take_sqrt))
}

#[track_caller]
pub(crate) fn arg_reduce(
    values: Option<&TensorData>,
    indices: &TensorData,
    src: &TensorData,
    dim: usize,
    is_max: bool,
) {
    dispatch_rocm!(indices, arg_reduce(values, indices, src, dim, is_max))
}

#[track_caller]
pub(crate) fn exclusive_cumprod(out: &TensorData, src: &TensorData, dim: usize, reverse: bool) {
    dispatch_rocm!(out, exclusive_cumprod(out, src, dim, reverse))
}

#[track_caller]
pub(crate) fn matmul(out: &TensorData, a: &TensorData, b: &TensorData) {
    dispatch!(out, matmul(out, a, b))
}
