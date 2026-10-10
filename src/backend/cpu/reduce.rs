//! Reduction kernels.

use super::{Plan, base, dispatch_all, dispatch_float, exec};
use crate::dtype::sealed::Sealed;
use crate::ops::ReduceOp;
use crate::shape::contiguous_strides;
use crate::tensor::{Layout, TensorData};
use crate::{DType, Element};

/// The kept and reduced dimensions of a layout.
struct Split {
    kept_shape: Vec<usize>,
    kept_strides: Vec<usize>,
    red_shape: Vec<usize>,
    red_strides: Vec<usize>,
}

impl Split {
    fn new(l: &Layout, mask: &[bool]) -> Split {
        let mut s = Split {
            kept_shape: vec![],
            kept_strides: vec![],
            red_shape: vec![],
            red_strides: vec![],
        };
        for ((&size, &stride), &reduced) in l.shape.iter().zip(l.strides.iter()).zip(mask) {
            if reduced {
                s.red_shape.push(size);
                s.red_strides.push(stride);
            } else {
                s.kept_shape.push(size);
                s.kept_strides.push(stride);
            }
        }
        s
    }
}

/// Folds the reduced dimensions of `src` into each element of the contiguous tensor `out`.
fn fold<S: Element, O: Element, A: Copy>(
    out: &TensorData,
    src: &TensorData,
    mask: &[bool],
    init: A,
    step: impl Fn(A, S) -> A,
    finish: impl Fn(A, usize) -> O,
) {
    let s = Split::new(&src.layout, mask);
    let count: usize = s.red_shape.iter().product();
    let out_strides = contiguous_strides(&s.kept_shape);
    assert!(out.layout.is_contiguous());
    assert_eq!(
        out.layout.num_elements(),
        s.kept_shape.iter().product::<usize>()
    );
    let (po, ps) = (base::<O>(out), base::<S>(src));
    let outer = Plan::new(&s.kept_shape, [&out_strides, &s.kept_strides]);
    let inner = Plan::new(&s.red_shape, [&s.red_strides]);
    exec(|| {
        outer.for_each([out.layout.offset, src.layout.offset], |[o, i]| {
            let mut acc = init;
            inner.for_each([i], |[j]| {
                // SAFETY: the source layout lies inside its storage.
                acc = step(acc, unsafe { *ps.add(j) });
            });
            // SAFETY: `out` is contiguous with one element per kept position.
            unsafe { *po.add(o) = finish(acc, count) };
        })
    });
}

/// Whether `x` replaces the current best value `acc`: the larger (or smaller) value wins, and the
/// first NaN wins and sticks, like PyTorch.
#[inline]
fn replaces<S: Element>(acc: Option<S>, x: S, is_max: bool) -> bool {
    #[allow(clippy::eq_op)]
    let is_nan = |v: S| v != v;
    match acc {
        None => true,
        Some(a) if is_nan(a) => false,
        Some(a) => is_nan(x) || if is_max { x > a } else { x < a },
    }
}

/// Reduces the dimensions selected by `mask` into `out`, which is contiguous. Output dtypes:
/// `Sum` and `Prod` give `I64` for integer and bool inputs; `All` and `Any` give `Bool` (or `U8`
/// for `U8` inputs); everything else keeps the input dtype. `Mean` requires floats and `Max`/`Min`
/// a nonempty reduction.
pub(crate) fn reduce(op: ReduceOp, out: &TensorData, src: &TensorData, mask: &[bool]) {
    let dtype = src.dtype;
    match op {
        ReduceOp::Sum | ReduceOp::Prod | ReduceOp::Mean if dtype.is_floating_point() => {
            dispatch_float!(dtype, T => match op {
                ReduceOp::Sum => fold::<T, T, f64>(out, src, mask, 0.0, |a, x| a + x.to_f64(), |a, _| T::from_f64(a)),
                ReduceOp::Prod => fold::<T, T, f64>(out, src, mask, 1.0, |a, x| a * x.to_f64(), |a, _| T::from_f64(a)),
                _ => fold::<T, T, f64>(out, src, mask, 0.0, |a, x| a + x.to_f64(), |a, n| T::from_f64(a / n as f64)),
            })
        }
        ReduceOp::Sum | ReduceOp::Prod => {
            assert_eq!(out.dtype, DType::I64);
            dispatch_all!(dtype, T => if op == ReduceOp::Sum {
                fold::<T, i64, i64>(out, src, mask, 0, |a, x| a.wrapping_add(x.to_i64()), |a, _| a)
            } else {
                fold::<T, i64, i64>(out, src, mask, 1, |a, x| a.wrapping_mul(x.to_i64()), |a, _| a)
            })
        }
        ReduceOp::Mean => panic!("mean requires a floating point input, got {dtype}"),
        ReduceOp::Max | ReduceOp::Min => {
            let is_max = op == ReduceOp::Max;
            dispatch_all!(dtype, T => fold::<T, T, Option<T>>(
                out, src, mask, None,
                |a, x| if replaces(a, x, is_max) { Some(x) } else { a },
                |a, _| a.expect("reduction over an empty dimension"),
            ))
        }
        ReduceOp::All | ReduceOp::Any => {
            let all = op == ReduceOp::All;
            dispatch_all!(dtype, T => {
                let step = move |a: bool, x: T| if all { a && x.to_bool() } else { a || x.to_bool() };
                if out.dtype == DType::U8 {
                    fold::<T, u8, bool>(out, src, mask, all, step, |a, _| a as u8)
                } else {
                    fold::<T, bool, bool>(out, src, mask, all, step, |a, _| a)
                }
            })
        }
    }
}

/// Variance (or standard deviation when `take_sqrt`) over the dimensions selected by `mask`,
/// dividing by `max(0, N - correction)`. Computed in `f64` with two passes.
pub(crate) fn var(
    out: &TensorData,
    src: &TensorData,
    mask: &[bool],
    correction: f64,
    take_sqrt: bool,
) {
    dispatch_float!(src.dtype, T => {
        let s = Split::new(&src.layout, mask);
        let count: usize = s.red_shape.iter().product();
        let out_strides = contiguous_strides(&s.kept_shape);
        assert!(out.layout.is_contiguous());
        let (po, ps) = (base::<T>(out), base::<T>(src));
        let outer = Plan::new(&s.kept_shape, [&out_strides, &s.kept_strides]);
        let inner = Plan::new(&s.red_shape, [&s.red_strides]);
        let divisor = (count as f64 - correction).max(0.0);
        exec(|| {
            outer.for_each([out.layout.offset, src.layout.offset], |[o, i]| {
                let mut sum = 0.0;
                // SAFETY: the source layout lies inside its storage.
                inner.for_each([i], |[j]| sum += unsafe { *ps.add(j) }.to_f64());
                let mean = sum / count as f64;
                let mut sq = 0.0;
                inner.for_each([i], |[j]| {
                    // SAFETY: as above.
                    let d = unsafe { *ps.add(j) }.to_f64() - mean;
                    sq += d * d;
                });
                let v = sq / divisor;
                // SAFETY: `out` is contiguous with one element per kept position.
                unsafe { *po.add(o) = T::from_f64(if take_sqrt { v.sqrt() } else { v }) };
            })
        });
    })
}

/// Index (and optionally value) of the maximum or minimum along `dim`. The first occurrence wins
/// ties, and the first NaN wins over everything. `values` (input dtype) and `indices` (`I64`) are
/// contiguous with the shape of `src` without `dim`.
pub(crate) fn arg_reduce(
    values: Option<&TensorData>,
    indices: &TensorData,
    src: &TensorData,
    dim: usize,
    is_max: bool,
) {
    let l = &src.layout;
    let (n, stride) = if l.shape.is_empty() {
        (1, 0)
    } else {
        (l.shape[dim], l.strides[dim])
    };
    assert!(
        n > 0,
        "cannot compute the arg reduction of an empty dimension"
    );
    let mut mask = vec![false; l.shape.len()];
    if !mask.is_empty() {
        mask[dim] = true;
    }
    let s = Split::new(l, &mask);
    let out_strides = contiguous_strides(&s.kept_shape);
    dispatch_all!(src.dtype, T => {
        let ps = base::<T>(src);
        let pi = base::<i64>(indices);
        let pv = values.map(base::<T>);
        let outer = Plan::new(&s.kept_shape, [&out_strides, &s.kept_strides]);
        exec(|| {
            outer.for_each([0, l.offset], |[o, i]| {
                let mut best: Option<T> = None;
                let mut best_index = 0;
                for k in 0..n {
                    // SAFETY: `k < n` stays inside the source layout.
                    let x = unsafe { *ps.add(i + k * stride) };
                    if replaces(best, x, is_max) {
                        best = Some(x);
                        best_index = k;
                    }
                }
                // SAFETY: outputs are contiguous with one element per kept position.
                unsafe {
                    *pi.add(indices.layout.offset + o) = best_index as i64;
                    if let (Some(pv), Some(v)) = (pv, values) {
                        *pv.add(v.layout.offset + o) = best.unwrap();
                    }
                }
            })
        });
    })
}
