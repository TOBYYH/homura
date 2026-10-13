//! Reduction kernels.

use super::{Plan, base, dispatch_all, dispatch_float, exec, walk_strides};
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
        // The kept dimensions of an empty source may be nonempty and are walked even though no
        // element is read.
        let strides = walk_strides(l);
        for ((&size, &stride), &reduced) in l.shape.iter().zip(strides.iter()).zip(mask) {
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
                // An empty reduction has no mean, whatever the correction.
                let v = if count == 0 { f64::NAN } else { sq / divisor };
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

/// A floating point element type and the type PyTorch's CPU kernels accumulate products of it in
/// (`acc_type`): `f32` for `f16` and `bf16`, `f64` for `f32` and `f64`.
trait Accumulate: Element {
    type Acc: Copy + std::ops::Mul<Output = Self::Acc>;
    const ONE: Self::Acc;
    fn widen(self) -> Self::Acc;
    /// Rounds an accumulated value to the element type.
    fn narrow(acc: Self::Acc) -> Self;
}

impl Accumulate for half::f16 {
    type Acc = f32;
    const ONE: f32 = 1.0;
    fn widen(self) -> f32 {
        self.to_f32()
    }
    fn narrow(acc: f32) -> Self {
        half::f16::from_f32(acc)
    }
}

impl Accumulate for half::bf16 {
    type Acc = f32;
    const ONE: f32 = 1.0;
    fn widen(self) -> f32 {
        self.to_f32()
    }
    fn narrow(acc: f32) -> Self {
        half::bf16::from_f32(acc)
    }
}

impl Accumulate for f32 {
    type Acc = f64;
    const ONE: f64 = 1.0;
    fn widen(self) -> f64 {
        self as f64
    }
    fn narrow(acc: f64) -> Self {
        acc as f32
    }
}

impl Accumulate for f64 {
    type Acc = f64;
    const ONE: f64 = 1.0;
    fn widen(self) -> f64 {
        self
    }
    fn narrow(acc: f64) -> Self {
        acc
    }
}

/// For each element of `src`, the product of the elements before it along `dim` (after it when
/// `reverse`), 1 for the first (last), written to the contiguous `out` of the same shape and
/// dtype. Like PyTorch's CPU `cumprod` of the shifted input, the product is accumulated one
/// element at a time in the accumulation type of the dtype, and each one is rounded to the dtype
/// as it is written: the accumulator itself is never rounded.
pub(crate) fn exclusive_cumprod(out: &TensorData, src: &TensorData, dim: usize, reverse: bool) {
    fn scan<T: Accumulate>(out: &TensorData, src: &TensorData, dim: usize, reverse: bool) {
        let shape = &src.layout.shape;
        let strides = walk_strides(&src.layout);
        let out_strides = contiguous_strides(shape);
        let others = |v: &[usize]| -> Vec<usize> {
            v.iter()
                .enumerate()
                .filter(|&(d, _)| d != dim)
                .map(|(_, &s)| s)
                .collect()
        };
        let (n, step, out_step) = (shape[dim], strides[dim], out_strides[dim]);
        let lines = Plan::new(&others(shape), [&others(&out_strides), &others(&strides)]);
        let (po, ps) = (base::<T>(out), base::<T>(src));
        exec(|| {
            lines.for_each([out.layout.offset, src.layout.offset], |[o, i]| {
                let mut acc = T::ONE;
                for k in 0..n {
                    let k = if reverse { n - 1 - k } else { k };
                    // SAFETY: `k < n` stays inside the line, which lies inside both layouts.
                    unsafe {
                        *po.add(o + k * out_step) = T::narrow(acc);
                        acc = acc * (*ps.add(i + k * step)).widen();
                    }
                }
            })
        });
    }
    assert!(out.layout.is_contiguous() && out.layout.shape == src.layout.shape);
    assert_eq!(out.dtype, src.dtype);
    match src.dtype {
        DType::F16 => scan::<half::f16>(out, src, dim, reverse),
        DType::BF16 => scan::<half::bf16>(out, src, dim, reverse),
        DType::F32 => scan::<f32>(out, src, dim, reverse),
        DType::F64 => scan::<f64>(out, src, dim, reverse),
        other => panic!("expected a floating point dtype, got {other}"),
    }
}
