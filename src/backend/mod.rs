//! The backends and the dispatch layer in front of them.
//!
//! Operations call the free functions of this module, which have the signatures of the CPU
//! backend's kernels and pick the backend by device. The meta device has no data: callers return
//! before dispatching for meta tensors, which the dispatch functions treat as an internal error.
//! ROCm kernels are added step by step; until then the ROCm arm of a kernel panics.

pub(crate) mod cpu;
pub(crate) mod locks;
#[cfg(feature = "rocm")]
pub(crate) mod rocm;

pub(crate) use cpu::Fused;

#[cfg(test)]
mod dispatch_tests;

use crate::ops::{BinaryOp, Pool2d, ReduceOp, UnaryOp};
use crate::tensor::TensorData;
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
    dispatch!(dst, write_host(dst, values))
}

#[track_caller]
pub(crate) fn read_bytes(src: &TensorData) -> Vec<u8> {
    dispatch!(src, read_bytes(src))
}

#[track_caller]
pub(crate) fn read_host<T: Element>(src: &TensorData) -> Vec<T> {
    dispatch!(src, read_host(src))
}

#[track_caller]
pub(crate) fn copy(dst: &TensorData, src: &TensorData) {
    dispatch!(dst, copy(dst, src))
}

#[track_caller]
pub(crate) fn fill(dst: &TensorData, value: Scalar) {
    dispatch!(dst, fill(dst, value))
}

#[track_caller]
pub(crate) fn unary(op: UnaryOp, out: &TensorData, src: &TensorData) {
    dispatch!(out, unary(op, out, src))
}

#[track_caller]
pub(crate) fn binary(op: BinaryOp, out: &TensorData, a: &TensorData, b: &TensorData) {
    dispatch!(out, binary(op, out, a, b))
}

#[track_caller]
pub(crate) fn binary_scalar(
    op: BinaryOp,
    out: &TensorData,
    a: &TensorData,
    s: f64,
    scalar_first: bool,
) {
    dispatch!(out, binary_scalar(op, out, a, s, scalar_first))
}

#[track_caller]
pub(crate) fn fused(
    op: Fused,
    out: &TensorData,
    a: &TensorData,
    b: &TensorData,
    c: Option<&TensorData>,
) {
    dispatch!(out, fused(op, out, a, b, c))
}

#[track_caller]
pub(crate) fn where_(out: &TensorData, cond: &TensorData, x: &TensorData, y: &TensorData) {
    dispatch!(out, where_(out, cond, x, y))
}

#[track_caller]
pub(crate) fn softmax(out: &TensorData, src: &TensorData, dim: usize, log: bool) {
    dispatch!(out, softmax(out, src, dim, log))
}

#[track_caller]
pub(crate) fn index_select(
    out: &TensorData,
    src: &TensorData,
    dim: usize,
    index: &TensorData,
    allow_negative: bool,
) {
    dispatch!(out, index_select(out, src, dim, index, allow_negative))
}

#[track_caller]
pub(crate) fn index_overwrite(out: &TensorData, dim: usize, index: &[usize], src: &TensorData) {
    dispatch!(out, index_overwrite(out, dim, index, src))
}

#[track_caller]
pub(crate) fn mask_coordinates(mask: &TensorData) -> Vec<Vec<i64>> {
    dispatch!(mask, mask_coordinates(mask))
}

#[track_caller]
pub(crate) fn index_add(out: &TensorData, dim: usize, index: &[usize], src: &TensorData) {
    dispatch!(out, index_add(out, dim, index, src))
}

#[track_caller]
pub(crate) fn scatter_add(
    out: &TensorData,
    dim: usize,
    index_shape: &[usize],
    index: &[usize],
    src: &TensorData,
) {
    dispatch!(out, scatter_add(out, dim, index_shape, index, src))
}

#[track_caller]
pub(crate) fn gather(out: &TensorData, src: &TensorData, dim: usize, index: &[usize]) {
    dispatch!(out, gather(out, src, dim, index))
}

#[track_caller]
pub(crate) fn triangle(out: &TensorData, src: &TensorData, diagonal: i64, upper: bool) {
    dispatch!(out, triangle(out, src, diagonal, upper))
}

#[track_caller]
pub(crate) fn flip(out: &TensorData, src: &TensorData, mask: &[bool]) {
    dispatch!(out, flip(out, src, mask))
}

#[track_caller]
pub(crate) fn unfold_backward(
    out: &TensorData,
    grad: &TensorData,
    dim: usize,
    size: usize,
    step: usize,
) {
    dispatch!(out, unfold_backward(out, grad, dim, size, step))
}

#[track_caller]
pub(crate) fn max_pool2d(out: &TensorData, indices: &TensorData, input: &TensorData, g: &Pool2d) {
    dispatch!(out, max_pool2d(out, indices, input, g))
}

#[track_caller]
pub(crate) fn max_pool2d_backward(out: &TensorData, grad: &TensorData, indices: &TensorData) {
    dispatch!(out, max_pool2d_backward(out, grad, indices))
}

#[track_caller]
pub(crate) fn adaptive_avg_pool2d(out: &TensorData, input: &TensorData) {
    dispatch!(out, adaptive_avg_pool2d(out, input))
}

#[track_caller]
pub(crate) fn adaptive_avg_pool2d_backward(out: &TensorData, grad: &TensorData) {
    dispatch!(out, adaptive_avg_pool2d_backward(out, grad))
}

#[track_caller]
pub(crate) fn reduce(op: ReduceOp, out: &TensorData, src: &TensorData, mask: &[bool]) {
    dispatch!(out, reduce(op, out, src, mask))
}

#[track_caller]
pub(crate) fn var(
    out: &TensorData,
    src: &TensorData,
    mask: &[bool],
    correction: f64,
    take_sqrt: bool,
) {
    dispatch!(out, var(out, src, mask, correction, take_sqrt))
}

#[track_caller]
pub(crate) fn arg_reduce(
    values: Option<&TensorData>,
    indices: &TensorData,
    src: &TensorData,
    dim: usize,
    is_max: bool,
) {
    dispatch!(indices, arg_reduce(values, indices, src, dim, is_max))
}

#[track_caller]
pub(crate) fn exclusive_cumprod(out: &TensorData, src: &TensorData, dim: usize, reverse: bool) {
    dispatch!(out, exclusive_cumprod(out, src, dim, reverse))
}

#[track_caller]
pub(crate) fn matmul(out: &TensorData, a: &TensorData, b: &TensorData) {
    dispatch!(out, matmul(out, a, b))
}
