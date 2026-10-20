//! Pooling kernels. They loop over the planes (all dimensions but the last two) and, in each, over
//! the output positions in row-major order, like PyTorch's CPU kernels for contiguous inputs, so
//! that sums are added in the same order. Inputs may have any strides; the outputs are new
//! contiguous tensors.

use super::{Float, FloatElement, IntElement, Plan, base, dispatch_float, dispatch_int, exec};
use crate::Element;
use crate::ops::{AdaptiveWindows, Pool2d};
use crate::tensor::{Layout, TensorData};
use crate::{Shape, Strides};

/// The plane dimensions of `l` (all but the last two) and their strides.
fn planes(l: &Layout) -> (Shape, Strides) {
    let nd = l.shape.len();
    (
        Shape::new(&l.shape[..nd - 2]),
        Strides::new(&l.strides[..nd - 2]),
    )
}

/// The height and width of `l`, and their strides.
fn spatial(l: &Layout) -> ([usize; 2], [usize; 2]) {
    let nd = l.shape.len();
    (
        [l.shape[nd - 2], l.shape[nd - 1]],
        [l.strides[nd - 2], l.strides[nd - 1]],
    )
}

/// `out` and `indices` (`I64`) get the maximum of each window of `input` and its position in the
/// plane (`h * width + w`), PyTorch's `max_pool2d_with_indices`. Each window looks only at the
/// input positions it covers, the first of them being the initial candidate; a value replaces
/// the candidate if it is greater or NaN, so ties keep the first and NaN the last, and padding is
/// never selected. Every window must cover an input position.
pub(crate) fn max_pool2d(out: &TensorData, indices: &TensorData, input: &TensorData, g: &Pool2d) {
    if out.dtype.is_floating_point() {
        dispatch_float!(out.dtype, T => {
            max_pool2d_typed::<T>(out, indices, input, g, T::down(<T as FloatElement>::C::of(f64::NEG_INFINITY)), |v, m| {
                let v = v.up();
                v > m.up() || v.is_nan()
            })
        })
    } else {
        dispatch_int!(out.dtype, T => {
            max_pool2d_typed::<T>(out, indices, input, g, <T as IntElement>::MIN, |v, m| v > m)
        })
    }
}

fn max_pool2d_typed<T: Element>(
    out: &TensorData,
    indices: &TensorData,
    input: &TensorData,
    g: &Pool2d,
    lowest: T,
    replaces: impl Fn(T, T) -> bool,
) {
    let (lo, lx, li) = (&out.layout, &input.layout, &indices.layout);
    if lo.num_elements() == 0 {
        return;
    }
    let ([height, width], [sh, sw]) = spatial(lx);
    let [out_h, out_w] = g.output;
    let (plane_shape, plane_strides) = planes(lx);
    let (_, out_strides) = planes(lo);
    let (_, index_strides) = planes(li);
    let rows: Vec<(usize, usize)> = (0..out_h).map(|o| g.window(0, o, height)).collect();
    let cols: Vec<(usize, usize)> = (0..out_w).map(|o| g.window(1, o, width)).collect();
    let [dh, dw] = g.dilation;
    let (px, po, pi) = (base::<T>(input), base::<T>(out), base::<i64>(indices));
    exec(|| {
        Plan::new(&plane_shape, [&plane_strides, &out_strides, &index_strides]).for_each(
            [lx.offset, lo.offset, li.offset],
            |[x, o, i]| {
                for (r, &(h0, h1)) in rows.iter().enumerate() {
                    for (c, &(w0, w1)) in cols.iter().enumerate() {
                        let (mut best, mut index) = (lowest, h0 * width + w0);
                        for ih in (h0..h1).step_by(dh) {
                            for iw in (w0..w1).step_by(dw) {
                                // SAFETY: `ih < height` and `iw < width` stay inside `input`.
                                let v = unsafe { *px.add(x + ih * sh + iw * sw) };
                                if replaces(v, best) {
                                    (best, index) = (v, ih * width + iw);
                                }
                            }
                        }
                        let k = r * out_w + c;
                        // SAFETY: `out` and `indices` are contiguous with `out_h * out_w`
                        // elements per plane.
                        unsafe {
                            *po.add(o + k) = best;
                            *pi.add(i + k) = index as i64;
                        }
                    }
                }
            },
        )
    });
}

/// The gradient of [`max_pool2d`]: `out`, zero-filled with the input's shape, gets each element
/// of `grad` added at the plane position `indices` holds for it, one at a time in output order,
/// in the dtype (`f16` and `bf16` round after each addition), like PyTorch's CPU kernel.
///
/// # Panics
///
/// Panics if an index is outside the plane.
pub(crate) fn max_pool2d_backward(out: &TensorData, grad: &TensorData, indices: &TensorData) {
    let (lo, lg, li) = (&out.layout, &grad.layout, &indices.layout);
    if lg.num_elements() == 0 {
        return;
    }
    let ([height, width], [oh_stride, ow_stride]) = spatial(lo);
    let ([out_h, out_w], [gh, gw]) = spatial(lg);
    let (_, [ih, iw]) = spatial(li);
    let (plane_shape, out_strides) = planes(lo);
    let (_, grad_strides) = planes(lg);
    let (_, index_strides) = planes(li);
    let size = height * width;
    dispatch_float!(out.dtype, T => {
        let (po, pg, pi) = (base::<T>(out), base::<T>(grad), base::<i64>(indices));
        exec(|| {
            Plan::new(&plane_shape, [&out_strides, &grad_strides, &index_strides]).for_each(
                [lo.offset, lg.offset, li.offset],
                |[o, g, i]| {
                    for r in 0..out_h {
                        for c in 0..out_w {
                            // SAFETY: `r < out_h` and `c < out_w` stay inside `grad` and
                            // `indices`.
                            let (v, index) =
                                unsafe { (*pg.add(g + r * gh + c * gw), *pi.add(i + r * ih + c * iw)) };
                            let index = usize::try_from(index)
                                .ok()
                                .filter(|&k| k < size)
                                .unwrap_or_else(|| panic!("max_pool2d index {index} is outside a plane of {size} elements"));
                            let at = o + index / width * oh_stride + index % width * ow_stride;
                            // SAFETY: the index is inside the plane, so `at` is inside `out`.
                            unsafe { *po.add(at) = T::down((*po.add(at)).up() + v.up()) };
                        }
                    }
                },
            )
        });
    })
}

/// `out` gets the mean of each window of `input` that [`AdaptiveWindows`] gives, PyTorch's
/// `_adaptive_avg_pool2d` CPU kernel: the window is summed in its computation type (`f32` for
/// `f16`, `bf16` and `f32`, `f64` for `f64`) in row-major order, divided by its height and then
/// by its width, and rounded once.
pub(crate) fn adaptive_avg_pool2d(out: &TensorData, input: &TensorData) {
    let (lo, lx) = (&out.layout, &input.layout);
    if lo.num_elements() == 0 {
        return;
    }
    let ([height, width], [sh, sw]) = spatial(lx);
    let [out_h, out_w] = [lo.shape[lo.shape.len() - 2], lo.shape[lo.shape.len() - 1]];
    let (rows, cols) = (
        AdaptiveWindows::new(height, out_h),
        AdaptiveWindows::new(width, out_w),
    );
    let (plane_shape, plane_strides) = planes(lx);
    let (_, out_strides) = planes(lo);
    dispatch_float!(out.dtype, T => {
        type C = <T as FloatElement>::C;
        let (px, po) = (base::<T>(input), base::<T>(out));
        exec(|| {
            Plan::new(&plane_shape, [&plane_strides, &out_strides]).for_each(
                [lx.offset, lo.offset],
                |[x, o]| {
                    for r in 0..out_h {
                        let (h0, h1) = rows.window(r);
                        for c in 0..out_w {
                            let (w0, w1) = cols.window(c);
                            let mut sum = C::ZERO;
                            for ih in h0..h1 {
                                for iw in w0..w1 {
                                    // SAFETY: the window lies inside `input`.
                                    sum += unsafe { *px.add(x + ih * sh + iw * sw) }.up();
                                }
                            }
                            let mean = sum / C::of((h1 - h0) as f64) / C::of((w1 - w0) as f64);
                            // SAFETY: `out` is contiguous with `out_h * out_w` elements per plane.
                            unsafe { *po.add(o + r * out_w + c) = T::down(mean) };
                        }
                    }
                },
            )
        });
    })
}

/// The gradient of [`adaptive_avg_pool2d`]: `out`, zero-filled with the input's shape, gets each
/// element of `grad` divided by its window's height and then width, in the dtype (half precision
/// rounds the window sizes and each quotient to the dtype, as `c10::Half / int64_t` does), added
/// to every position of the window, one at a time in output order and in the dtype, like
/// PyTorch's CPU kernel.
pub(crate) fn adaptive_avg_pool2d_backward(out: &TensorData, grad: &TensorData) {
    let (lo, lg) = (&out.layout, &grad.layout);
    if lg.num_elements() == 0 || lo.num_elements() == 0 {
        return;
    }
    let ([height, width], [oh_stride, ow_stride]) = spatial(lo);
    let ([out_h, out_w], [gh, gw]) = spatial(lg);
    let (rows, cols) = (
        AdaptiveWindows::new(height, out_h),
        AdaptiveWindows::new(width, out_w),
    );
    let (plane_shape, out_strides) = planes(lo);
    let (_, grad_strides) = planes(lg);
    dispatch_float!(out.dtype, T => {
        type C = <T as FloatElement>::C;
        // A window size as the dtype, then the computation type.
        let size = |n: usize| T::down(C::of(n as f64)).up();
        let (po, pg) = (base::<T>(out), base::<T>(grad));
        exec(|| {
            Plan::new(&plane_shape, [&out_strides, &grad_strides]).for_each(
                [lo.offset, lg.offset],
                |[o, g]| {
                    for r in 0..out_h {
                        let (h0, h1) = rows.window(r);
                        for c in 0..out_w {
                            let (w0, w1) = cols.window(c);
                            // SAFETY: `r < out_h` and `c < out_w` stay inside `grad`.
                            let v = unsafe { *pg.add(g + r * gh + c * gw) };
                            let delta = T::down(T::down(v.up() / size(h1 - h0)).up() / size(w1 - w0));
                            for ih in h0..h1 {
                                for iw in w0..w1 {
                                    let at = o + ih * oh_stride + iw * ow_stride;
                                    // SAFETY: the window lies inside `out`.
                                    unsafe { *po.add(at) = T::down((*po.add(at)).up() + delta.up()) };
                                }
                            }
                        }
                    }
                },
            )
        });
    })
}
