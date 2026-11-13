//! Matrix multiplication, the outer product of `addr`, and the writeback of the scaled products.

use std::ops::{Add, Mul};

use super::num::{Float, FloatElement};
use super::{base, dispatch_float, dispatch_int, exec, walk_strides};
use crate::Element;
use crate::backend::{Gemm, Outer};
use crate::tensor::TensorData;

/// `out = gemm(a, b)` (see [`Gemm`]): `a` is `[B, M, K]`, `b` is `[B, K, N]`, any strides (a
/// stride of 0 broadcasts); `out` is `[B, M, N]`, or `[1, M, N]` when the batches are summed.
///
/// - Floats: each product accumulates in `f64` and is rounded to the computation type (`f32`
///   for `F16`, `BF16` and `F32`); summed batches add those in `f64`, in batch order, and round
///   once more. The writeback then follows the composite of PyTorch's CPU kernels with one
///   rounding per step in the computation type: `alpha * p` (skipped for `alpha == 1`), then,
///   unless `beta` is zero (when the input isn't read), `beta * c + alpha * p` (`beta * c`
///   skipped for `beta == 1`). The result is rounded to the dtype.
/// - Integers: wrapping `i64` throughout, truncated to the dtype once. Truncation is taken
///   modulo 2^bits, so this gives the values of truncating after every step.
#[track_caller]
pub(crate) fn gemm(out: &TensorData, g: &Gemm) {
    let dtype = g.a.dtype;
    if dtype.is_floating_point() {
        dispatch_float!(dtype, T => mm(out, g, &FloatSteps::<T>::new(g)))
    } else if dtype.is_integral() {
        dispatch_int!(dtype, T => mm(out, g, &IntSteps::<T>::new(g)))
    } else {
        panic!("matmul is not supported for {dtype} tensors")
    }
}

/// The arithmetic of [`mm`] for one dtype `T`: products accumulate in `Acc` (batch sums too) and
/// are rounded to `P`, the computation type, before the writeback.
trait Steps<T> {
    type Acc: Copy;
    type P: Copy;
    const ZERO: Self::Acc;
    fn mul_add(&self, acc: Self::Acc, x: T, y: T) -> Self::Acc;
    fn round(&self, acc: Self::Acc) -> Self::P;
    /// Adds a rounded product to a batch sum.
    fn add_batch(&self, sum: Self::Acc, p: Self::P) -> Self::Acc;
    /// Whether the writeback reads the input (it has one and `beta` isn't zero).
    fn reads_input(&self) -> bool;
    /// The output element of the product `p` and the input element `c` (if read).
    fn write(&self, p: Self::P, c: Option<T>) -> T;
}

/// `(alpha, beta)` in the computation type, if the product is scaled.
struct FloatSteps<T: FloatElement> {
    scale: Option<(T::C, T::C)>,
}

impl<T: FloatElement> FloatSteps<T> {
    fn new(g: &Gemm) -> Self {
        let scale = g
            .scale
            .as_ref()
            .map(|s| (T::C::of(s.alpha.to_f64()), T::C::of(s.beta.to_f64())));
        FloatSteps { scale }
    }
}

impl<T: FloatElement> Steps<T> for FloatSteps<T>
where
    T::C: PartialEq + Add<Output = T::C> + Mul<Output = T::C>,
{
    type Acc = f64;
    type P = T::C;
    const ZERO: f64 = 0.0;
    #[inline]
    fn mul_add(&self, acc: f64, x: T, y: T) -> f64 {
        acc + x.to_f64() * y.to_f64()
    }
    #[inline]
    fn round(&self, acc: f64) -> T::C {
        T::C::of(acc)
    }
    #[inline]
    fn add_batch(&self, sum: f64, p: T::C) -> f64 {
        sum + p.wide()
    }
    fn reads_input(&self) -> bool {
        self.scale.is_some_and(|(_, beta)| beta != T::C::ZERO)
    }
    #[inline]
    fn write(&self, p: T::C, c: Option<T>) -> T {
        let Some((alpha, beta)) = self.scale else {
            return T::down(p);
        };
        let a = if alpha == T::C::ONE { p } else { p * alpha };
        T::down(match c {
            None => a,
            Some(c) if beta == T::C::ONE => c.up() + a,
            Some(c) => c.up() * beta + a,
        })
    }
}

/// `(alpha, beta)`, if the product is scaled.
struct IntSteps<T> {
    scale: Option<(i64, i64)>,
    _t: std::marker::PhantomData<T>,
}

impl<T> IntSteps<T> {
    fn new(g: &Gemm) -> Self {
        let scale = g
            .scale
            .as_ref()
            .map(|s| (s.alpha.to_i64(), s.beta.to_i64()));
        IntSteps {
            scale,
            _t: std::marker::PhantomData,
        }
    }
}

impl<T: Element> Steps<T> for IntSteps<T> {
    type Acc = i64;
    type P = i64;
    const ZERO: i64 = 0;
    #[inline]
    fn mul_add(&self, acc: i64, x: T, y: T) -> i64 {
        acc.wrapping_add(x.to_i64().wrapping_mul(y.to_i64()))
    }
    #[inline]
    fn round(&self, acc: i64) -> i64 {
        acc
    }
    #[inline]
    fn add_batch(&self, sum: i64, p: i64) -> i64 {
        sum.wrapping_add(p)
    }
    fn reads_input(&self) -> bool {
        self.scale.is_some_and(|(_, beta)| beta != 0)
    }
    #[inline]
    fn write(&self, p: i64, c: Option<T>) -> T {
        let v = match self.scale {
            None => p,
            Some((alpha, beta)) => {
                let a = if alpha == 1 { p } else { p.wrapping_mul(alpha) };
                match c {
                    None => a,
                    Some(c) if beta == 1 => c.to_i64().wrapping_add(a),
                    Some(c) => c.to_i64().wrapping_mul(beta).wrapping_add(a),
                }
            }
        };
        T::from_i64(v)
    }
}

fn mm<T: Element, S: Steps<T>>(out: &TensorData, g: &Gemm, steps: &S) {
    let (a, b) = (g.a, g.b);
    let (la, lb, lo) = (&a.layout, &b.layout, &out.layout);
    let (batch, m, k) = (la.shape[0], la.shape[1], la.shape[2]);
    let n = lb.shape[2];
    assert_eq!(lb.shape[..2], [batch, k]);
    let out_batches = if g.sum_batches { 1 } else { batch };
    assert_eq!(lo.shape, [out_batches, m, n]);
    if lo.num_elements() == 0 {
        // Nothing to compute: don't allocate a row of `n` accumulators, which may be huge.
        return;
    }
    let input = g.scale.as_ref().filter(|_| steps.reads_input()).map(|s| {
        assert_eq!(s.input.layout.shape, lo.shape);
        (
            base::<T>(s.input),
            walk_strides(&s.input.layout),
            s.input.layout.offset,
        )
    });
    let (pa, pb, po) = (base::<T>(a), base::<T>(b), base::<T>(out));
    // With `k == 0` (or no batches) the operands have no elements and any strides; the
    // products are zero.
    let (sa, sb, so) = (walk_strides(la), walk_strides(lb), &lo.strides);
    let mut row = vec![S::ZERO; n];
    let mut sums = vec![S::ZERO; if g.sum_batches { n } else { 0 }];
    exec(|| {
        for ob in 0..out_batches {
            let batches = if g.sum_batches { 0..batch } else { ob..ob + 1 };
            let o0 = lo.offset + ob * so[0];
            for i in 0..m {
                sums.fill(S::ZERO);
                for bi in batches.clone() {
                    let a0 = la.offset + bi * sa[0];
                    let b0 = lb.offset + bi * sb[0];
                    row.fill(S::ZERO);
                    for kk in 0..k {
                        // SAFETY: `bi < batch`, `i < m`, `kk < k` and `j < n` stay inside the
                        // layouts.
                        let x = unsafe { *pa.add(a0 + i * sa[1] + kk * sa[2]) };
                        let brow = b0 + kk * sb[1];
                        for (j, acc) in row.iter_mut().enumerate() {
                            // SAFETY: as above.
                            *acc = steps.mul_add(*acc, x, unsafe { *pb.add(brow + j * sb[2]) });
                        }
                    }
                    if g.sum_batches {
                        for (sum, &acc) in sums.iter_mut().zip(&row) {
                            *sum = steps.add_batch(*sum, steps.round(acc));
                        }
                    }
                }
                let products = if g.sum_batches { &sums } else { &row };
                for (j, &acc) in products.iter().enumerate() {
                    let c = input.as_ref().map(|(pc, sc, off)| {
                        // SAFETY: the input has the output's shape.
                        unsafe { *pc.add(off + ob * sc[0] + i * sc[1] + j * sc[2]) }
                    });
                    // SAFETY: as above.
                    unsafe {
                        *po.add(o0 + i * so[1] + j * so[2]) = steps.write(steps.round(acc), c)
                    };
                }
            }
        }
    });
}

/// `out = beta * input + alpha * outer(v1, v2)` with the steps of [`Outer`]: `out` is `[n, m]`,
/// `v1` is `[n]`, `v2` is `[m]` and the input is a view of `[n, m]`, all with any strides.
#[track_caller]
pub(crate) fn addr(out: &TensorData, o: &Outer) {
    let dtype = o.v1.dtype;
    if dtype.is_floating_point() {
        dispatch_float!(dtype, T => outer(out, o, &FloatOuter::<T>::new(o)))
    } else if dtype.is_integral() {
        dispatch_int!(dtype, T => outer::<T, _>(out, o, &IntOuter::new(o)))
    } else {
        outer(out, o, &BoolOuter::new(o))
    }
}

/// The arithmetic of [`outer`] for one dtype `T`.
trait OuterSteps<T> {
    /// Whether the input is read (`beta` isn't zero).
    fn reads_input(&self) -> bool;
    /// `v1[i]` as it enters the products of row `i`.
    fn row(&self, x: T) -> T;
    /// The output element of the row value `a`, `y = v2[j]` and the input element `c` (if read).
    fn write(&self, a: T, y: T, c: Option<T>) -> T;
}

/// `(alpha, beta)` in the computation type.
struct FloatOuter<T: FloatElement> {
    alpha: T::C,
    beta: T::C,
}

impl<T: FloatElement> FloatOuter<T> {
    fn new(o: &Outer) -> Self {
        FloatOuter {
            alpha: T::C::of(o.alpha.to_f64()),
            beta: T::C::of(o.beta.to_f64()),
        }
    }
}

impl<T: FloatElement> OuterSteps<T> for FloatOuter<T>
where
    T::C: PartialEq + Add<Output = T::C> + Mul<Output = T::C>,
{
    fn reads_input(&self) -> bool {
        self.beta != T::C::ZERO
    }
    #[inline]
    fn row(&self, x: T) -> T {
        if self.alpha == T::C::ONE {
            x
        } else {
            T::down(x.up() * self.alpha)
        }
    }
    #[inline]
    fn write(&self, a: T, y: T, c: Option<T>) -> T {
        let p = T::down(a.up() * y.up());
        match c {
            None => p,
            Some(c) => {
                let c = if self.beta == T::C::ONE {
                    c
                } else {
                    T::down(c.up() * self.beta)
                };
                T::down(c.up() + p.up())
            }
        }
    }
}

/// `(alpha, beta)`: wrapping `i64` arithmetic, truncated to the dtype once (which gives the
/// values of truncating after every step).
struct IntOuter {
    alpha: i64,
    beta: i64,
}

impl IntOuter {
    fn new(o: &Outer) -> Self {
        IntOuter {
            alpha: o.alpha.to_i64(),
            beta: o.beta.to_i64(),
        }
    }
}

impl<T: Element> OuterSteps<T> for IntOuter {
    fn reads_input(&self) -> bool {
        self.beta != 0
    }
    #[inline]
    fn row(&self, x: T) -> T {
        x
    }
    #[inline]
    fn write(&self, a: T, y: T, c: Option<T>) -> T {
        let p = a.to_i64().wrapping_mul(y.to_i64());
        let p = if self.alpha == 1 {
            p
        } else {
            p.wrapping_mul(self.alpha)
        };
        T::from_i64(match c {
            None => p,
            Some(c) if self.beta == 1 => c.to_i64().wrapping_add(p),
            Some(c) => c.to_i64().wrapping_mul(self.beta).wrapping_add(p),
        })
    }
}

/// `(alpha, beta)` of bools: products are and, sums are or.
struct BoolOuter {
    alpha: bool,
    beta: bool,
}

impl BoolOuter {
    fn new(o: &Outer) -> Self {
        BoolOuter {
            alpha: o.alpha.to_bool(),
            beta: o.beta.to_bool(),
        }
    }
}

impl OuterSteps<bool> for BoolOuter {
    fn reads_input(&self) -> bool {
        self.beta
    }
    #[inline]
    fn row(&self, x: bool) -> bool {
        x && self.alpha
    }
    #[inline]
    fn write(&self, a: bool, y: bool, c: Option<bool>) -> bool {
        // `c` is only read with a true `beta`.
        c.unwrap_or(false) || (a && y)
    }
}

fn outer<T: Element, S: OuterSteps<T>>(out: &TensorData, o: &Outer, steps: &S) {
    let (l1, l2, lo) = (&o.v1.layout, &o.v2.layout, &out.layout);
    let (n, m) = (l1.shape[0], l2.shape[0]);
    assert_eq!(lo.shape, [n, m]);
    if lo.num_elements() == 0 {
        return;
    }
    let input = steps.reads_input().then(|| {
        assert_eq!(o.input.layout.shape, lo.shape);
        (
            base::<T>(o.input),
            &o.input.layout.strides,
            o.input.layout.offset,
        )
    });
    let (p1, p2, po) = (base::<T>(o.v1), base::<T>(o.v2), base::<T>(out));
    let (s1, s2, so) = (l1.strides[0], l2.strides[0], &lo.strides);
    exec(|| {
        for i in 0..n {
            // SAFETY: `i < n` and `j < m` stay inside the layouts, which have elements.
            let a = steps.row(unsafe { *p1.add(l1.offset + i * s1) });
            for j in 0..m {
                // SAFETY: as above.
                let y = unsafe { *p2.add(l2.offset + j * s2) };
                let c = input.map(|(pc, sc, off)| {
                    // SAFETY: the input has the output's shape.
                    unsafe { *pc.add(off + i * sc[0] + j * sc[1]) }
                });
                // SAFETY: as above.
                unsafe { *po.add(lo.offset + i * so[0] + j * so[1]) = steps.write(a, y, c) };
            }
        }
    });
}
