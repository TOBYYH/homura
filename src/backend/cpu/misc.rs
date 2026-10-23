//! Kernels that don't fit the elementwise or reduction patterns.

use super::{Float, FloatElement, Plan, base, copy, dispatch_all, dispatch_float, exec, read_host};
use crate::dtype::sealed::Sealed;
use crate::shape::contiguous_strides;
use crate::tensor::TensorData;
use crate::{Shape, Strides};

/// Softmax (or log-softmax) of `src` along `dim` into `out`, which has the same shape. Computed in
/// `f64`, subtracting the maximum first.
pub(crate) fn softmax(out: &TensorData, src: &TensorData, dim: usize, log: bool) {
    let (ls, lo) = (&src.layout, &out.layout);
    assert_eq!(ls.shape, lo.shape);
    if ls.num_elements() == 0 {
        // The other dimensions may be nonempty, with arbitrary strides.
        return;
    }
    let scalar = ls.shape.is_empty();
    let (n, si, so) = if scalar {
        (1, 0, 0)
    } else {
        (ls.shape[dim], ls.strides[dim], lo.strides[dim])
    };
    let (ks, ko) = if scalar {
        (ls.clone(), lo.clone())
    } else {
        (ls.select(dim, 0), lo.select(dim, 0))
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
#[track_caller]
pub(crate) fn index_select(
    out: &TensorData,
    src: &TensorData,
    dim: usize,
    index: &TensorData,
    allow_negative: bool,
) {
    let indices: Vec<i64> = read_host(index);
    let size = src.layout.shape[dim] as i64;
    // When the slices have no elements, their strides may be arbitrary and their offsets may not
    // even fit in `usize`: check the indices but don't build the slices.
    let copies = out.layout.num_elements() > 0;
    for (j, &i) in indices.iter().enumerate() {
        let wrapped = if i < 0 && allow_negative { i + size } else { i };
        if wrapped < 0 || wrapped >= size {
            panic!("index {i} is out of bounds for dimension {dim} with size {size}");
        }
        if !copies {
            continue;
        }
        let dst = TensorData {
            storage: out.storage.clone(),
            layout: out.layout.select(dim, j),
            dtype: out.dtype,
        };
        let s = TensorData {
            storage: src.storage.clone(),
            layout: src.layout.select(dim, wrapped as usize),
            dtype: src.dtype,
        };
        copy(&dst, &s);
    }
}

/// Copies slice `i` of `src` along `dim` into slice `index[i]` of `out`, in order, so that the
/// last of repeated positions wins: the overwriting `index_put`. `out` is new memory that shares
/// none with `src`, and the positions must be in range.
pub(crate) fn index_overwrite(out: &TensorData, dim: usize, index: &[usize], src: &TensorData) {
    // An empty source writes nothing, and its strides and offset may be arbitrary.
    if src.layout.num_elements() == 0 {
        return;
    }
    for (i, &j) in index.iter().enumerate() {
        let dst = TensorData {
            layout: out.layout.select(dim, j),
            ..out.clone()
        };
        let s = TensorData {
            layout: src.layout.select(dim, i),
            ..src.clone()
        };
        copy(&dst, &s);
    }
}

/// The positions of the true elements of the bool tensor `mask` in row-major order, as one
/// vector of coordinates per dimension: PyTorch's `nonzero`, transposed.
pub(crate) fn mask_coordinates(mask: &TensorData) -> Vec<Vec<i64>> {
    let shape = &mask.layout.shape;
    let mut coords = vec![vec![]; shape.len()];
    let mut index = vec![0usize; shape.len()];
    for set in read_host::<bool>(mask) {
        if set {
            for (c, &i) in coords.iter_mut().zip(&index) {
                c.push(i as i64);
            }
        }
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    coords
}

/// `out[o] += src[s]` for every pair of offsets that `pairs` reports, where `o` counts the
/// elements of the contiguous `out` from its first one and `s` is a storage offset of `src`, which
/// has the dtype of `out`. Like the reductions, each element of `out` is accumulated in `f64`
/// (floating point) or `i64` (integers, wrapping; for bools the sum is a logical or) and rounded
/// once at the end. The caller guarantees that every pair lies inside both tensors and that `out`
/// is a new tensor, which shares no memory with `src`.
fn add_at(out: &TensorData, src: &TensorData, pairs: impl Fn(&mut dyn FnMut(usize, usize))) {
    assert!(out.layout.is_contiguous() && out.dtype == src.dtype);
    let n = out.layout.num_elements();
    if n == 0 {
        return;
    }
    let first = out.layout.offset;
    dispatch_all!(out.dtype, T => {
        let (po, ps) = (base::<T>(out), base::<T>(src));
        exec(|| {
            // SAFETY: `out` is contiguous, so its `n` elements start at `first`, and nothing else
            // refers to them: `out` is new and shares no memory with `src`.
            let elements = unsafe { std::slice::from_raw_parts_mut(po.add(first), n) };
            // SAFETY: the caller guarantees that `s` is an offset inside `src`.
            let read = |s: usize| unsafe { *ps.add(s) };
            if out.dtype.is_floating_point() {
                let mut acc: Vec<f64> = elements.iter().map(|v| v.to_f64()).collect();
                pairs(&mut |o, s| acc[o] += read(s).to_f64());
                for (e, v) in elements.iter_mut().zip(acc) {
                    *e = T::from_f64(v);
                }
            } else {
                let mut acc: Vec<i64> = elements.iter().map(|v| v.to_i64()).collect();
                pairs(&mut |o, s| acc[o] = acc[o].wrapping_add(read(s).to_i64()));
                for (e, v) in elements.iter_mut().zip(acc) {
                    *e = T::from_i64(v);
                }
            }
        });
    });
}

/// Adds slice `i` of `src` along `dim` into slice `index[i]` of the contiguous `out`
/// (`torch.index_add`), see [`add_at`] for the precision. The positions must be in range.
pub(crate) fn index_add(out: &TensorData, dim: usize, index: &[usize], src: &TensorData) {
    // An empty source adds nothing, and its strides and offset may be arbitrary.
    if src.layout.num_elements() == 0 {
        return;
    }
    let out_strides = contiguous_strides(&out.layout.shape);
    let slice = |s: &[usize]| {
        let mut s = s.to_vec();
        s.remove(dim);
        s
    };
    let ls = &src.layout;
    let plan = Plan::new(
        &slice(&ls.shape),
        [&slice(&out_strides), &slice(&ls.strides)],
    );
    add_at(out, src, |f| {
        for (i, &j) in index.iter().enumerate() {
            let start = [j * out_strides[dim], ls.offset + i * ls.strides[dim]];
            plan.for_each(start, |[o, s]| f(o, s));
        }
    });
}

/// `out[.., index[p], ..] += src[p]`, with `index[p]` along `dim`, for every position `p` of the
/// index tensor, whose shape is `index_shape` and whose values `index` holds in row-major order
/// (`torch.scatter_add`). `out` is contiguous and `src` may be larger than the index; see
/// [`add_at`] for the precision. The positions must be in range.
pub(crate) fn scatter_add(
    out: &TensorData,
    dim: usize,
    index_shape: &[usize],
    index: &[usize],
    src: &TensorData,
) {
    if index.is_empty() {
        return;
    }
    let out_strides = contiguous_strides(&out.layout.shape);
    // The index value replaces the position along `dim`.
    let mut walk = out_strides.clone();
    walk[dim] = 0;
    let plan = Plan::new(index_shape, [&walk, &src.layout.strides]);
    add_at(out, src, |f| {
        let mut values = index.iter();
        plan.for_each([0, src.layout.offset], |[o, s]| {
            let j = values.next().expect("one value per position");
            f(o + j * out_strides[dim], s);
        });
    });
}

/// `out[p] = src[.., index[p], ..]`, with `index[p]` along `dim`, for every position `p` of the
/// contiguous `out`, which has the shape of the index tensor; `index` holds its values in
/// row-major order (`torch.gather`). The positions must be in range.
pub(crate) fn gather(out: &TensorData, src: &TensorData, dim: usize, index: &[usize]) {
    let (lo, ls) = (&out.layout, &src.layout);
    assert!(lo.is_contiguous() && lo.num_elements() == index.len());
    if index.is_empty() {
        return;
    }
    // The index value replaces the position along `dim`.
    let mut walk = ls.strides.clone();
    walk[dim] = 0;
    dispatch_all!(src.dtype, T => {
        let (po, ps) = (base::<T>(out), base::<T>(src));
        exec(|| {
            let mut values = index.iter();
            Plan::new(&lo.shape, [&lo.strides, &walk]).for_each([lo.offset, ls.offset], |[o, s]| {
                let j = values.next().expect("one value per position");
                // SAFETY: `o` is inside `out`. The index has no more positions than `src` in
                // every dimension but `dim`, and `j` is in range along `dim`, so the offset is
                // inside `src`.
                unsafe { *po.add(o) = *ps.add(s + j * ls.strides[dim]) };
            });
        });
    });
}

/// Copies the upper (`triu`) or lower (`tril`) triangle of the last two dimensions of `src` into
/// `out` and zeroes the rest. Element `(r, c)` is kept when `c - r >= diagonal` (upper) or
/// `c - r <= diagonal` (lower).
pub(crate) fn triangle(out: &TensorData, src: &TensorData, diagonal: i64, upper: bool) {
    let (ls, lo) = (&src.layout, &out.layout);
    let nd = ls.shape.len();
    assert!(nd >= 2 && ls.shape == lo.shape);
    if ls.num_elements() == 0 {
        // The batch dimensions may be nonempty, with arbitrary strides.
        return;
    }
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

/// The gradient of `x.unfold(dim, size, step)` for the upstream gradient `grad` (PyTorch's
/// `unfold_backward`): `out` has the shape of `x` and is zero-filled; each of its positions along
/// `dim` gets the elements of `grad` that view it. Where windows overlap (`step < size`), a
/// position adds the windows covering it to zero one at a time in window order, in the dtype
/// (`f16` and `bf16` round after each addition), like PyTorch's CPU kernel; otherwise it copies
/// the element of the one window covering it, keeping `-0.0`, like PyTorch's copy. A
/// zero-dimensional `x` is viewed by one window of `size` (0 or 1) elements. Floating point only.
pub(crate) fn unfold_backward(
    out: &TensorData,
    grad: &TensorData,
    dim: usize,
    size: usize,
    step: usize,
) {
    let (lo, lg) = (&out.layout, &grad.layout);
    // Without elements on either side nothing is written, and the strides may be arbitrary.
    if lo.num_elements() == 0 || lg.num_elements() == 0 {
        return;
    }
    if lo.shape.is_empty() {
        debug_assert_eq!(&lg.shape[..], [1]);
        copy(
            out,
            &TensorData {
                layout: lg.select(0, 0),
                ..grad.clone()
            },
        );
        return;
    }
    let last = lg.shape.len() - 1;
    debug_assert_eq!(last, lo.shape.len());
    let (n, windows) = (lo.shape[dim], lg.shape[dim]);
    let (od, gd, gl) = (lo.strides[dim], lg.strides[dim], lg.strides[last]);
    // The positions of all dimensions but `dim` (and the window's last one).
    let (ko, kg) = (lo.select(dim, 0), lg.select(last, 0).select(dim, 0));
    dispatch_float!(out.dtype, T => {
        let (po, pg) = (base::<T>(out), base::<T>(grad));
        exec(|| {
            Plan::new(&ko.shape, [&ko.strides, &kg.strides]).for_each(
                [ko.offset, kg.offset],
                |[o, g]| {
                    // SAFETY: `w < windows` and `j < size` stay inside `grad`.
                    let read = |w: usize, j: usize| unsafe { *pg.add(g + w * gd + j * gl) };
                    for i in 0..n {
                        let value = if step >= size {
                            let (w, j) = (i / step, i % step);
                            if w >= windows || j >= size {
                                continue;
                            }
                            read(w, j)
                        } else {
                            // The windows `first..=last` cover position `i`.
                            let first = if i >= size { (i - size) / step + 1 } else { 0 };
                            let last = (i / step).min(windows - 1);
                            (first..=last).fold(T::down(Float::ZERO), |acc, w| {
                                T::down(acc.up() + read(w, i - w * step).up())
                            })
                        };
                        // SAFETY: `i < n` stays inside `out`.
                        unsafe { *po.add(o + i * od) = value };
                    }
                },
            )
        });
    });
}
