//! Kernels that don't fit the elementwise or reduction patterns.

use super::{Plan, base, copy, dispatch_all, dispatch_float, exec, read_host};
use crate::dtype::sealed::Sealed;
use crate::tensor::{Layout, TensorData};
use crate::{Shape, Strides};

/// Removes dimension `dim` from a layout, starting at index `i` along it.
fn select_layout(l: &Layout, dim: usize, i: usize) -> Layout {
    let mut shape = l.shape.clone();
    let mut strides = l.strides.clone();
    shape.remove(dim);
    strides.remove(dim);
    Layout {
        shape,
        strides,
        offset: l.offset_at(dim, i),
    }
}

/// Softmax (or log-softmax) of `src` along `dim` into `out`, which has the same shape. Computed in
/// `f64`, subtracting the maximum first.
pub(crate) fn softmax(out: &TensorData, src: &TensorData, dim: usize, log: bool) {
    let (ls, lo) = (&src.layout, &out.layout);
    assert_eq!(ls.shape, lo.shape);
    let scalar = ls.shape.is_empty();
    let (n, si, so) = if scalar {
        (1, 0, 0)
    } else {
        (ls.shape[dim], ls.strides[dim], lo.strides[dim])
    };
    let (ks, ko) = if scalar {
        (ls.clone(), lo.clone())
    } else {
        (select_layout(ls, dim, 0), select_layout(lo, dim, 0))
    };
    dispatch_float!(src.dtype, T => {
        let (ps, po) = (base::<T>(src), base::<T>(out));
        let outer = Plan::new(&ks.shape, [&ko.strides, &ks.strides]);
        exec(|| {
            outer.for_each([ko.offset, ks.offset], |[o, i]| {
                // SAFETY: `k < n` stays inside both layouts.
                let x = |k: usize| unsafe { *ps.add(i + k * si) }.to_f64();
                let max = (0..n).map(x).fold(f64::NEG_INFINITY, |m, v| if v > m { v } else { m });
                let sum: f64 = (0..n).map(|k| (x(k) - max).exp()).sum();
                let log_sum = sum.ln();
                for k in 0..n {
                    let v = if log { x(k) - max - log_sum } else { (x(k) - max).exp() / sum };
                    // SAFETY: as above.
                    unsafe { *po.add(o + k * so) = T::from_f64(v) };
                }
            })
        });
    })
}

/// `out` = the slices of `src` along `dim` picked by the 1-D integer tensor `index`. Negative
/// indices count from the end when `allow_negative` is set.
pub(crate) fn index_select(
    out: &TensorData,
    src: &TensorData,
    dim: usize,
    index: &TensorData,
    allow_negative: bool,
) {
    let indices: Vec<i64> = read_host(index);
    let size = src.layout.shape[dim] as i64;
    for (j, &i) in indices.iter().enumerate() {
        let wrapped = if i < 0 && allow_negative { i + size } else { i };
        if wrapped < 0 || wrapped >= size {
            panic!("index {i} is out of bounds for dimension {dim} with size {size}");
        }
        let dst = TensorData {
            storage: out.storage.clone(),
            layout: select_layout(&out.layout, dim, j),
            dtype: out.dtype,
        };
        let s = TensorData {
            storage: src.storage.clone(),
            layout: select_layout(&src.layout, dim, wrapped as usize),
            dtype: src.dtype,
        };
        copy(&dst, &s);
    }
}

/// Copies the upper (`triu`) or lower (`tril`) triangle of the last two dimensions of `src` into
/// `out` and zeroes the rest. Element `(r, c)` is kept when `c - r >= diagonal` (upper) or
/// `c - r <= diagonal` (lower).
pub(crate) fn triangle(out: &TensorData, src: &TensorData, diagonal: i64, upper: bool) {
    let (ls, lo) = (&src.layout, &out.layout);
    let nd = ls.shape.len();
    assert!(nd >= 2 && ls.shape == lo.shape);
    let (rows, cols) = (ls.shape[nd - 2], ls.shape[nd - 1]);
    let batch = Shape::new(&ls.shape[..nd - 2]);
    let (bs, bo) = (
        Strides::new(&ls.strides[..nd - 2]),
        Strides::new(&lo.strides[..nd - 2]),
    );
    let (rs, cs) = (ls.strides[nd - 2], ls.strides[nd - 1]);
    let (ro, co) = (lo.strides[nd - 2], lo.strides[nd - 1]);
    dispatch_all!(src.dtype, T => {
        let (ps, po) = (base::<T>(src), base::<T>(out));
        exec(|| {
            Plan::new(&batch, [&bo, &bs]).for_each([lo.offset, ls.offset], |[o, i]| {
                for r in 0..rows {
                    for c in 0..cols {
                        let d = c as i64 - r as i64;
                        let keep = if upper { d >= diagonal } else { d <= diagonal };
                        // SAFETY: `r < rows` and `c < cols` stay inside both layouts.
                        unsafe {
                            *po.add(o + r * ro + c * co) =
                                if keep { *ps.add(i + r * rs + c * cs) } else { T::default() };
                        }
                    }
                }
            })
        });
    })
}

/// Copies `src` into the contiguous tensor `out`, reversing the dimensions selected by `mask`.
pub(crate) fn flip(out: &TensorData, src: &TensorData, mask: &[bool]) {
    let l = &src.layout;
    assert!(out.layout.is_contiguous() && out.layout.shape == l.shape);
    let numel = l.num_elements();
    let nd = l.shape.len();
    dispatch_all!(src.dtype, T => {
        let (ps, po) = (base::<T>(src), base::<T>(out));
        exec(|| {
            let mut idx = vec![0usize; nd];
            for linear in 0..numel {
                let mut off = l.offset;
                for d in 0..nd {
                    let i = if mask[d] { l.shape[d] - 1 - idx[d] } else { idx[d] };
                    off += i * l.strides[d];
                }
                // SAFETY: every multi-index is inside the shape, so offsets are in bounds.
                unsafe { *po.add(out.layout.offset + linear) = *ps.add(off) };
                for d in (0..nd).rev() {
                    idx[d] += 1;
                    if idx[d] < l.shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        });
    });
}
