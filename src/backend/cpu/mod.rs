//! The CPU reference backend.
//!
//! Kernels are simple and favour correctness over speed. Every kernel runs inside one global
//! execution lock and accesses tensor memory through raw pointers, so a tensor may be both input
//! and output of an operation (`x.add_(&x)`) without deadlocks or data races. Callers are
//! responsible for passing layouts that lie inside their storage and, for in-place writes, for
//! removing partial overlap between inputs and the output.
//!
//! The lock makes each kernel atomic, not each operation: an in-place operation computes its
//! result and writes it back in separate kernels, so concurrent in-place operations on shared
//! storage can lose updates. Like PyTorch, homura leaves synchronizing those to the caller.

use std::cell::Cell;
use std::sync::Mutex;

use crate::dtype::sealed::Sealed;
use crate::ops::{BinaryOp, UnaryOp};
use crate::tensor::TensorData;
use crate::{DType, Element, Scalar};

mod iter;
mod linalg;
mod misc;
mod num;
mod reduce;

pub(crate) use iter::{Plan, walk_strides};
pub(crate) use linalg::matmul;
pub(crate) use misc::{
    flip, gather, index_add, index_overwrite, index_select, mask_coordinates, scatter_add, softmax,
    triangle, unfold_backward,
};
pub(crate) use num::{Float, FloatElement, IntElement};
pub(crate) use reduce::{arg_reduce, exclusive_cumprod, reduce, var};

static EXEC: Mutex<()> = Mutex::new(());

thread_local! {
    /// Whether this thread holds [`EXEC`].
    static HOLDS_EXEC: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` while holding the CPU execution lock. Reentrant: kernels called from `f` run inside
/// the same critical section, which lets a write and its version bump happen atomically.
pub(crate) fn exec<R>(f: impl FnOnce() -> R) -> R {
    if HOLDS_EXEC.get() {
        return f();
    }
    let _guard = EXEC.lock().unwrap_or_else(|e| e.into_inner());
    struct Release;
    impl Drop for Release {
        fn drop(&mut self) {
            HOLDS_EXEC.set(false);
        }
    }
    HOLDS_EXEC.set(true);
    let _release = Release;
    f()
}

/// Instantiates `$body` with `$T` bound to the element type of `$dtype`.
macro_rules! dispatch_all {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::DType::Bool => {
                type $T = bool;
                $body
            }
            $crate::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::DType::F16 => {
                type $T = half::f16;
                $body
            }
            $crate::DType::BF16 => {
                type $T = half::bf16;
                $body
            }
            $crate::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::DType::F64 => {
                type $T = f64;
                $body
            }
        }
    };
}

/// Like [`dispatch_all`] for floating point dtypes; panics on others.
macro_rules! dispatch_float {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::DType::F16 => {
                type $T = half::f16;
                $body
            }
            $crate::DType::BF16 => {
                type $T = half::bf16;
                $body
            }
            $crate::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::DType::F64 => {
                type $T = f64;
                $body
            }
            other => panic!("expected a floating point dtype, got {other}"),
        }
    };
}

/// Like [`dispatch_all`] for integer dtypes; panics on others.
macro_rules! dispatch_int {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::DType::I64 => {
                type $T = i64;
                $body
            }
            other => panic!("expected an integer dtype, got {other}"),
        }
    };
}

pub(crate) use {dispatch_all, dispatch_float, dispatch_int};

/// Typed pointer to the start of a tensor's storage.
#[inline]
fn base<T: Element>(d: &TensorData) -> *mut T {
    assert_eq!(d.dtype, T::DTYPE, "kernel instantiated for the wrong dtype");
    d.storage.cpu_ptr() as *mut T
}

fn check_shape(out: &TensorData, input: &TensorData) {
    assert_eq!(
        out.layout.shape, input.layout.shape,
        "kernel operands must have the same shape"
    );
}

/// `out[i] = f()` for every element of `out`.
pub(crate) fn map0<O: Element>(out: &TensorData, mut f: impl FnMut() -> O) {
    let po = base::<O>(out);
    let l = &out.layout;
    exec(|| {
        Plan::new(&l.shape, [&l.strides]).for_each([l.offset], |[o]| {
            // SAFETY: the layout lies inside the storage, so every offset is in bounds.
            unsafe { *po.add(o) = f() };
        })
    });
}

/// `out[i] = f(a[i])`. `a` must have the shape of `out`.
pub(crate) fn map1<A: Element, O: Element>(out: &TensorData, a: &TensorData, f: impl Fn(A) -> O) {
    check_shape(out, a);
    let (po, pa) = (base::<O>(out), base::<A>(a));
    let (lo, la) = (&out.layout, &a.layout);
    exec(|| {
        Plan::new(&lo.shape, [&lo.strides, &la.strides]).for_each(
            [lo.offset, la.offset],
            |[o, i]| {
                // SAFETY: both layouts lie inside their storage. `out` either doesn't overlap `a` or
                // is the same view, so each element is read before it is written.
                unsafe { *po.add(o) = f(*pa.add(i)) };
            },
        )
    });
}

/// `out[i] = f(a[i], b[i])`. Inputs must have the shape of `out`.
pub(crate) fn map2<A: Element, B: Element, O: Element>(
    out: &TensorData,
    a: &TensorData,
    b: &TensorData,
    f: impl Fn(A, B) -> O,
) {
    check_shape(out, a);
    check_shape(out, b);
    let (po, pa, pb) = (base::<O>(out), base::<A>(a), base::<B>(b));
    let (lo, la, lb) = (&out.layout, &a.layout, &b.layout);
    exec(|| {
        Plan::new(&lo.shape, [&lo.strides, &la.strides, &lb.strides]).for_each(
            [lo.offset, la.offset, lb.offset],
            |[o, i, j]| {
                // SAFETY: as in `map1`.
                unsafe { *po.add(o) = f(*pa.add(i), *pb.add(j)) };
            },
        )
    });
}

/// `out[i] = f(a[i], b[i], c[i])`. Inputs must have the shape of `out`.
pub(crate) fn map3<A: Element, B: Element, C: Element, O: Element>(
    out: &TensorData,
    a: &TensorData,
    b: &TensorData,
    c: &TensorData,
    f: impl Fn(A, B, C) -> O,
) {
    check_shape(out, a);
    check_shape(out, b);
    check_shape(out, c);
    let (po, pa, pb, pc) = (base::<O>(out), base::<A>(a), base::<B>(b), base::<C>(c));
    let (lo, la, lb, lc) = (&out.layout, &a.layout, &b.layout, &c.layout);
    exec(|| {
        Plan::new(
            &lo.shape,
            [&lo.strides, &la.strides, &lb.strides, &lc.strides],
        )
        .for_each(
            [lo.offset, la.offset, lb.offset, lc.offset],
            |[o, i, j, k]| {
                // SAFETY: as in `map1`.
                unsafe { *po.add(o) = f(*pa.add(i), *pb.add(j), *pc.add(k)) };
            },
        )
    });
}

/// Writes host values into a contiguous tensor.
pub(crate) fn write_host<T: Element>(dst: &TensorData, values: &[T]) {
    assert!(dst.layout.is_contiguous() && dst.layout.num_elements() == values.len());
    // The offset of an empty layout may lie outside the allocation; don't even form the pointer.
    if values.is_empty() {
        return;
    }
    let p = base::<T>(dst);
    exec(|| {
        // SAFETY: `dst` is contiguous with `values.len()` elements starting at its offset.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), p.add(dst.layout.offset), values.len())
        };
    });
}

/// Writes raw element bytes into a contiguous tensor.
///
/// `Bool` storage must only ever hold the bytes 0 and 1: kernels read it as Rust `bool`, for
/// which any other byte is undefined behaviour. This function is the only way raw bytes enter a
/// tensor, so it enforces that invariant.
pub(crate) fn write_bytes(dst: &TensorData, bytes: &[u8]) {
    let size = dst.dtype.size_in_bytes();
    assert!(dst.layout.is_contiguous() && dst.layout.num_elements() * size == bytes.len());
    assert!(
        dst.dtype != DType::Bool || bytes.iter().all(|&b| b <= 1),
        "bool tensor data must consist of the bytes 0 and 1"
    );
    if bytes.is_empty() {
        return;
    }
    let p = dst.storage.cpu_ptr();
    exec(|| {
        // SAFETY: `dst` is contiguous and covers exactly `bytes.len()` bytes from its offset.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                p.add(dst.layout.offset * size),
                bytes.len(),
            )
        };
    });
}

/// The raw element bytes of a contiguous tensor.
pub(crate) fn read_bytes(src: &TensorData) -> Vec<u8> {
    let size = src.dtype.size_in_bytes();
    assert!(src.layout.is_contiguous());
    let n = src.layout.num_elements() * size;
    if n == 0 {
        return vec![];
    }
    let p = src.storage.cpu_ptr();
    let mut out = vec![0u8; n];
    exec(|| {
        // SAFETY: `src` is contiguous and covers `n` bytes from its offset.
        unsafe {
            std::ptr::copy_nonoverlapping(p.add(src.layout.offset * size), out.as_mut_ptr(), n)
        };
    });
    out
}

/// Reads every element in row-major order, converting to `T`.
pub(crate) fn read_host<T: Element>(src: &TensorData) -> Vec<T> {
    let mut out = Vec::with_capacity(src.layout.num_elements());
    dispatch_all!(src.dtype, S => {
        let p = base::<S>(src);
        let l = &src.layout;
        exec(|| {
            Plan::new(&l.shape, [&l.strides]).for_each([l.offset], |[i]| {
                // SAFETY: the layout lies inside the storage.
                out.push(unsafe { *p.add(i) }.cast::<T>());
            })
        });
    });
    out
}

/// Copies `src` into `dst`, broadcasting and converting the dtype.
pub(crate) fn copy(dst: &TensorData, src: &TensorData) {
    let src = src.expand_to(&dst.layout.shape);
    if src.dtype == dst.dtype {
        dispatch_all!(src.dtype, T => map1::<T, T>(dst, &src, |x| x));
    } else {
        dispatch_all!(src.dtype, S => {
            dispatch_all!(dst.dtype, D => map1::<S, D>(dst, &src, |x| x.cast::<D>()))
        });
    }
}

/// Sets every element of `dst` to `value`.
pub(crate) fn fill(dst: &TensorData, value: Scalar) {
    dispatch_all!(dst.dtype, T => {
        let v = T::from_scalar(value);
        map0::<T>(dst, || v)
    });
}

/// `out = op(src)`. Float-only ops must be given floating point inputs; `out` has the dtype of
/// `src` except for `LogicalNot`, whose output is `Bool`.
pub(crate) fn unary(op: UnaryOp, out: &TensorData, src: &TensorData) {
    use UnaryOp::*;
    let dtype = src.dtype;
    match op {
        LogicalNot => dispatch_all!(dtype, T => map1::<T, bool>(out, src, |x| !x.to_bool())),
        Sign if dtype == DType::Bool => map1::<bool, bool>(out, src, |x| x),
        Neg | Abs | Relu | Sign if dtype.is_integral() => dispatch_int!(dtype, T => match op {
            Neg => map1::<T, T>(out, src, |x| x.wneg()),
            Abs => map1::<T, T>(out, src, |x| x.wabs()),
            Relu => map1::<T, T>(out, src, |x| if x > T::ZERO { x } else { T::ZERO }),
            _ => map1::<T, T>(out, src, |x| match x.cmp(&T::ZERO) {
                std::cmp::Ordering::Greater => T::ONE,
                std::cmp::Ordering::Less => T::ONE.wneg(),
                std::cmp::Ordering::Equal => T::ZERO,
            }),
        }),
        _ => dispatch_float!(dtype, T => {
            type C = <T as FloatElement>::C;
            macro_rules! float_map {
                (|$x:ident| $e:expr) => {
                    map1::<T, T>(out, src, |v: T| {
                        let $x: C = v.up();
                        T::down($e)
                    })
                };
            }
            let one = C::ONE;
            let half = C::of(0.5);
            match op {
                Neg => float_map!(|x| -x),
                Abs => float_map!(|x| x.abs()),
                Relu => float_map!(|x| if x.is_nan() || x > C::ZERO { x } else { C::ZERO }),
                Sign => float_map!(|x| {
                    // NaN and -0.0 give +0.0, as in PyTorch.
                    if x > C::ZERO {
                        one
                    } else if x < C::ZERO {
                        -one
                    } else {
                        C::ZERO
                    }
                }),
                Exp => float_map!(|x| x.exp()),
                Log => float_map!(|x| x.ln()),
                Sqrt => float_map!(|x| x.sqrt()),
                Rsqrt => float_map!(|x| one / x.sqrt()),
                Reciprocal => float_map!(|x| one / x),
                Sin => float_map!(|x| x.sin()),
                Cos => float_map!(|x| x.cos()),
                Erf => float_map!(|x| Float::erf(x)),
                Tanh => float_map!(|x| x.tanh()),
                Sigmoid => float_map!(|x| one / (one + (-x).exp())),
                // PyTorch's CPU kernel: min(x, 0) - log1p(e^-|x|).
                LogSigmoid => float_map!(|x| {
                    let min = if x.is_nan() || x < C::ZERO { x } else { C::ZERO };
                    min - (-x.abs()).exp().ln_1p()
                }),
                Silu => float_map!(|x| x / (one + (-x).exp())),
                Gelu => {
                    let frac_1_sqrt_2 = C::of(std::f64::consts::FRAC_1_SQRT_2);
                    float_map!(|x| half * x * (one + Float::erf(x * frac_1_sqrt_2)))
                }
                GeluTanh => {
                    let k = C::of((2.0 / std::f64::consts::PI).sqrt());
                    let c = C::of(0.044715);
                    float_map!(|x| half * x * (one + (k * (x + c * x * x * x)).tanh()))
                }
                LogicalNot => unreachable!(),
            }
        }),
    }
}

/// `out = op(a, b)`. `a` and `b` have the same (computation) dtype and the shape of `out`.
/// Comparisons and logical ops write `Bool`; the other ops write the computation dtype.
/// Logical ops expect `Bool` inputs.
// The comparisons are generic code that is also instantiated for `bool`.
#[allow(clippy::bool_comparison)]
pub(crate) fn binary(op: BinaryOp, out: &TensorData, a: &TensorData, b: &TensorData) {
    use BinaryOp::*;
    let dtype = a.dtype;
    debug_assert_eq!(dtype, b.dtype);
    match op {
        Greater | GreaterEqual | Less | LessEqual | Equal | NotEqual => {
            dispatch_all!(dtype, T => match op {
                Greater => map2::<T, T, bool>(out, a, b, |x, y| x > y),
                GreaterEqual => map2::<T, T, bool>(out, a, b, |x, y| x >= y),
                Less => map2::<T, T, bool>(out, a, b, |x, y| x < y),
                LessEqual => map2::<T, T, bool>(out, a, b, |x, y| x <= y),
                Equal => map2::<T, T, bool>(out, a, b, |x, y| x == y),
                _ => map2::<T, T, bool>(out, a, b, |x, y| x != y),
            })
        }
        LogicalAnd => map2::<bool, bool, bool>(out, a, b, |x, y| x && y),
        LogicalOr => map2::<bool, bool, bool>(out, a, b, |x, y| x || y),
        _ if dtype == DType::Bool => match op {
            Add | Maximum => map2::<bool, bool, bool>(out, a, b, |x, y| x || y),
            Mul | Minimum => map2::<bool, bool, bool>(out, a, b, |x, y| x && y),
            _ => panic!("{op:?} is not supported for bool tensors"),
        },
        _ if dtype.is_integral() => dispatch_int!(dtype, T => match op {
            Add => map2::<T, T, T>(out, a, b, |x, y| x.wadd(y)),
            Sub => map2::<T, T, T>(out, a, b, |x, y| x.wsub(y)),
            Mul => map2::<T, T, T>(out, a, b, |x, y| x.wmul(y)),
            Pow => map2::<T, T, T>(out, a, b, num::int_pow::<T>),
            Maximum => map2::<T, T, T>(out, a, b, |x, y| x.max(y)),
            Minimum => map2::<T, T, T>(out, a, b, |x, y| x.min(y)),
            _ => panic!("{op:?} is not supported for integer tensors"),
        }),
        _ => dispatch_float!(dtype, T => {
            macro_rules! float_map {
                (|$x:ident, $y:ident| $e:expr) => {
                    map2::<T, T, T>(out, a, b, |u: T, v: T| {
                        let ($x, $y) = (u.up(), v.up());
                        T::down($e)
                    })
                };
            }
            match op {
                Add => float_map!(|x, y| x + y),
                Sub => float_map!(|x, y| x - y),
                Mul => float_map!(|x, y| x * y),
                Div => float_map!(|x, y| x / y),
                Pow => float_map!(|x, y| x.powf(y)),
                Maximum => float_map!(|x, y| if x.is_nan() || x > y { x } else { y }),
                Minimum => float_map!(|x, y| if x.is_nan() || x < y { x } else { y }),
                _ => unreachable!(),
            }
        }),
    }
}

/// `out = a * s`, `a / s` or `s / a` (`scalar_first`) for `Mul` and `Div` of a floating point
/// `a`, with the scalar `s` converted to the computation type of `a`'s dtype instead of to the
/// dtype itself, as in PyTorch's CPU kernels: `f16` and `bf16` are multiplied or divided by the
/// scalar in `f32` and rounded once.
pub(crate) fn binary_scalar(
    op: BinaryOp,
    out: &TensorData,
    a: &TensorData,
    s: f64,
    scalar_first: bool,
) {
    dispatch_float!(a.dtype, T => {
        let s = <T as FloatElement>::C::of(s);
        match (op, scalar_first) {
            (BinaryOp::Mul, _) => map1::<T, T>(out, a, |x| T::down(x.up() * s)),
            (BinaryOp::Div, false) => map1::<T, T>(out, a, |x| T::down(x.up() / s)),
            (BinaryOp::Div, true) => map1::<T, T>(out, a, |x| T::down(s / x.up())),
            _ => unreachable!("{op:?} rounds its scalar to the dtype"),
        }
    })
}

/// The elementwise steps of the optimizers that PyTorch computes in one kernel, as its vectorized
/// CPU kernels (AVX2 build) do: in the computation type of the dtype (`f32` for `f16` and
/// `bf16`), rounding to the dtype once. Composing them from separate operations would round the
/// intermediate products, which can overflow or vanish. PyTorch's scalar loop for the elements
/// after the last full vector rounds `f16` / `bf16` scaled additions after each operation;
/// homura computes every element like the vectorized loop.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Fused {
    /// `a + alpha * b` (`add(b, alpha=)`), with one rounding: `fmadd(b, alpha, a)`, with `alpha`
    /// rounded to the dtype first.
    ScaledAdd(f64),
    /// `lerp(a, b, weight)` with a scalar weight: `fmadd(weight, b - a, a)` if `|weight| < 0.5`
    /// and `fmadd(weight - 1, b - a, b)` otherwise.
    Lerp(f64),
    /// `a + value * b * c` (`addcmul`), as `fmadd(value * b, c, a)`: the compiler fuses the last
    /// multiplication and the addition of PyTorch's kernel.
    Addcmul(f64),
    /// `a + value * b / c` (`addcdiv`), without fused multiply-adds.
    Addcdiv(f64),
}

/// `out = op(a, b)` or `op(a, b, c)` for floating point operands of the dtype and shape of
/// `out`; `c` is given exactly for `Addcmul` and `Addcdiv`.
pub(crate) fn fused(
    op: Fused,
    out: &TensorData,
    a: &TensorData,
    b: &TensorData,
    c: Option<&TensorData>,
) {
    dispatch_float!(out.dtype, T => {
        let of = <T as FloatElement>::C::of;
        match op {
            Fused::ScaledAdd(alpha) => {
                let alpha = T::down(of(alpha)).up();
                map2::<T, T, T>(out, a, b, |x, y| T::down(y.up().mul_add(alpha, x.up())));
            }
            Fused::Lerp(weight) => {
                let weight = of(weight);
                let small = weight.abs() < of(0.5);
                let coeff = if small { weight } else { weight - of(1.0) };
                map2::<T, T, T>(out, a, b, |x, y| {
                    let (x, y) = (x.up(), y.up());
                    T::down(coeff.mul_add(y - x, if small { x } else { y }))
                });
            }
            Fused::Addcmul(value) => {
                let value = of(value);
                let c = c.expect("addcmul has three operands");
                map3::<T, T, T, T>(out, a, b, c, |x, y, z| {
                    T::down((value * y.up()).mul_add(z.up(), x.up()))
                });
            }
            Fused::Addcdiv(value) => {
                let value = of(value);
                let c = c.expect("addcdiv has three operands");
                map3::<T, T, T, T>(out, a, b, c, |x, y, z| T::down(x.up() + value * y.up() / z.up()));
            }
        }
    })
}

/// `out = cond ? x : y`. `x` and `y` have the dtype of `out`, `cond` is `Bool`; all have the
/// shape of `out`.
pub(crate) fn where_(out: &TensorData, cond: &TensorData, x: &TensorData, y: &TensorData) {
    dispatch_all!(out.dtype, T => {
        map3::<bool, T, T, T>(out, cond, x, y, |c, a, b| if c { a } else { b })
    });
}
