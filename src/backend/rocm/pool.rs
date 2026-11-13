//! Pooling on ROCm (`csrc/pool.hip`): the counterparts of the CPU backend's `max_pool2d`,
//! `adaptive_avg_pool2d`, `adaptive_avg_pool2d_backward` and `unfold_backward`, with the CPU's
//! bits. (The backward of `max_pool2d`, which checks its indices, is in `index.rs`.)

use super::context::exec;
use super::copy::{buffer, device_index};
use super::index::full_desc;
use super::{call, expect, ffi};
use crate::Scalar;
use crate::ops::Pool2d;
use crate::tensor::TensorData;

/// Queues `launch` with the context of `out`'s device inside its execution lock, then reports a
/// failure.
#[track_caller]
fn run(kernel: &str, out: &TensorData, launch: impl FnOnce(ffi::hmr_context_t) -> ffi::hmr_status) {
    let ctx = buffer(kernel, out).context();
    expect(exec(device_index(out), || call(launch(ctx.raw()))));
}

/// The descriptions of `operands` with all their dimensions.
#[track_caller]
fn full<const N: usize>(kernel: &str, operands: [&TensorData; N]) -> [ffi::hmr_tensor_desc; N] {
    // A loop, not `map`: a closure would hide the caller's location.
    // SAFETY: all-zero bytes are a valid description; each is overwritten below.
    let mut out: [ffi::hmr_tensor_desc; N] = unsafe { std::mem::zeroed() };
    for (d, data) in out.iter_mut().zip(operands) {
        *d = expect(full_desc(kernel, data));
    }
    out
}

/// The geometry `g` for the C ABI: kernel size, stride, padding and dilation, each for the height
/// and the width.
pub(super) fn geometry_words(g: &Pool2d) -> [i64; 8] {
    [
        g.kernel[0],
        g.kernel[1],
        g.stride[0],
        g.stride[1],
        g.padding[0],
        g.padding[1],
        g.dilation[0],
        g.dilation[1],
    ]
    .map(|v| v as i64)
}

/// See `cpu::max_pool2d`.
#[track_caller]
pub(crate) fn max_pool2d(out: &TensorData, indices: &TensorData, input: &TensorData, g: &Pool2d) {
    if out.layout.num_elements() == 0 {
        return;
    }
    let [o, i, x] = full("max_pool2d", [out, indices, input]);
    let geometry = geometry_words(g);
    // SAFETY: the descriptions lie inside device memory of `out`'s device; `out` and `indices`
    // are new tensors.
    run("max_pool2d", out, |ctx| unsafe {
        ffi::hmr_max_pool2d(ctx, &o, &i, &x, geometry.as_ptr())
    });
}

/// See `cpu::adaptive_avg_pool2d`.
#[track_caller]
pub(crate) fn adaptive_avg_pool2d(out: &TensorData, input: &TensorData) {
    if out.layout.num_elements() == 0 {
        return;
    }
    let [o, x] = full("adaptive_avg_pool2d", [out, input]);
    // SAFETY: as in `max_pool2d`.
    run("adaptive_avg_pool2d", out, |ctx| unsafe {
        ffi::hmr_adaptive_avg_pool2d(ctx, &o, &x)
    });
}

/// See `cpu::adaptive_avg_pool2d_backward`. Every element of `out` is written.
#[track_caller]
pub(crate) fn adaptive_avg_pool2d_backward(out: &TensorData, grad: &TensorData) {
    if out.layout.num_elements() == 0 {
        return;
    }
    // An empty output size gives an empty gradient: no window covers an input element.
    if grad.layout.num_elements() == 0 {
        expect(super::copy::try_fill(out, Scalar::Int(0)));
        return;
    }
    let [o, g] = full("adaptive_avg_pool2d_backward", [out, grad]);
    // SAFETY: as in `max_pool2d`.
    run("adaptive_avg_pool2d_backward", out, |ctx| unsafe {
        ffi::hmr_adaptive_avg_pool2d_backward(ctx, &o, &g)
    });
}

/// See `cpu::unfold_backward`. Every element of `out` is written.
#[track_caller]
pub(crate) fn unfold_backward(
    out: &TensorData,
    grad: &TensorData,
    dim: usize,
    size: usize,
    step: usize,
) {
    let (lo, lg) = (&out.layout, &grad.layout);
    // Without elements on either side nothing is read, and the strides may be arbitrary.
    if lo.num_elements() == 0 {
        return;
    }
    // Windows of size 0 give an empty gradient: no window covers an input element.
    if lg.num_elements() == 0 {
        expect(super::copy::try_fill(out, Scalar::Int(0)));
        return;
    }
    if lo.shape.is_empty() {
        let window = TensorData {
            layout: lg.select(0, 0),
            ..grad.clone()
        };
        crate::backend::copy(out, &window);
        return;
    }
    let [o, g] = full("unfold_backward", [out, grad]);
    // SAFETY: as in `max_pool2d`.
    run("unfold_backward", out, |ctx| unsafe {
        ffi::hmr_unfold_backward(ctx, &o, &g, dim as i32, size as i64, step as i64)
    });
}
