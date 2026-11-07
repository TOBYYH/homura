//! Elementwise kernels on ROCm (`csrc/elementwise.hip`): the counterparts of the CPU backend's
//! `unary`, `binary`, `binary_scalar`, `fused`, `where_`, `triangle` and `flip`, with the same
//! arguments and the same results (bitwise, except for the device library's transcendental
//! functions, see `csrc/elementwise.hip`).
//!
//! Each kernel is queued inside the device's execution lock, and its failure is reported after
//! the lock's closure, where `#[track_caller]` still names the caller (review round 124, R149).

use super::context::exec;
use super::copy::{buffer, descs, descs_with_dims, device_index};
use super::{call, expect, ffi};
use crate::backend::Fused;
use crate::backend::cpu::{FloatElement, dispatch_float};
use crate::ops::{BinaryOp, UnaryOp};
use crate::tensor::TensorData;

fn unary_code(op: UnaryOp) -> i32 {
    use UnaryOp::*;
    match op {
        Neg => ffi::HMR_UNARY_NEG,
        Abs => ffi::HMR_UNARY_ABS,
        Sign => ffi::HMR_UNARY_SIGN,
        Relu => ffi::HMR_UNARY_RELU,
        Exp => ffi::HMR_UNARY_EXP,
        Log => ffi::HMR_UNARY_LOG,
        Sqrt => ffi::HMR_UNARY_SQRT,
        Rsqrt => ffi::HMR_UNARY_RSQRT,
        Reciprocal => ffi::HMR_UNARY_RECIPROCAL,
        Sin => ffi::HMR_UNARY_SIN,
        Cos => ffi::HMR_UNARY_COS,
        Erf => ffi::HMR_UNARY_ERF,
        Tanh => ffi::HMR_UNARY_TANH,
        Sigmoid => ffi::HMR_UNARY_SIGMOID,
        LogSigmoid => ffi::HMR_UNARY_LOG_SIGMOID,
        Silu => ffi::HMR_UNARY_SILU,
        Gelu => ffi::HMR_UNARY_GELU,
        GeluTanh => ffi::HMR_UNARY_GELU_TANH,
        LogicalNot => ffi::HMR_UNARY_LOGICAL_NOT,
    }
}

fn binary_code(op: BinaryOp) -> i32 {
    use BinaryOp::*;
    match op {
        Add => ffi::HMR_BINARY_ADD,
        Sub => ffi::HMR_BINARY_SUB,
        Mul => ffi::HMR_BINARY_MUL,
        Div => ffi::HMR_BINARY_DIV,
        Pow => ffi::HMR_BINARY_POW,
        Maximum => ffi::HMR_BINARY_MAXIMUM,
        Minimum => ffi::HMR_BINARY_MINIMUM,
        Greater => ffi::HMR_BINARY_GREATER,
        GreaterEqual => ffi::HMR_BINARY_GREATER_EQUAL,
        Less => ffi::HMR_BINARY_LESS,
        LessEqual => ffi::HMR_BINARY_LESS_EQUAL,
        Equal => ffi::HMR_BINARY_EQUAL,
        NotEqual => ffi::HMR_BINARY_NOT_EQUAL,
        LogicalAnd => ffi::HMR_BINARY_LOGICAL_AND,
        LogicalOr => ffi::HMR_BINARY_LOGICAL_OR,
    }
}

/// Queues `launch` with the context of `out`'s device inside its execution lock, then reports
/// a failure.
#[track_caller]
fn run(kernel: &str, out: &TensorData, launch: impl FnOnce(ffi::hmr_context_t) -> ffi::hmr_status) {
    let ctx = buffer(kernel, out).context();
    expect(exec(device_index(out), || call(launch(ctx.raw()))));
}

/// The descriptions of `operands` (see `copy::descs`), reporting too many dimensions here.
#[track_caller]
fn checked<const N: usize>(
    kernel: &str,
    operands: [&TensorData; N],
) -> Option<[ffi::hmr_tensor_desc; N]> {
    expect(descs(kernel, operands))
}

/// `out = op(src)`, as `cpu::unary`.
#[track_caller]
pub(crate) fn unary(op: UnaryOp, out: &TensorData, src: &TensorData) {
    check_shape(out, src);
    let Some([o, s]) = checked("unary", [out, src]) else {
        return;
    };
    // SAFETY: the descriptions lie inside device memory of `out`'s device (ops put both
    // operands there); `out` has distinct elements and is either `src` itself or doesn't
    // overlap it.
    run("unary", out, |ctx| unsafe {
        ffi::hmr_unary(ctx, unary_code(op), &o, &s)
    });
}

/// `out = op(a, b)`, as `cpu::binary`.
#[track_caller]
pub(crate) fn binary(op: BinaryOp, out: &TensorData, a: &TensorData, b: &TensorData) {
    check_shape(out, a);
    check_shape(out, b);
    let Some([o, x, y]) = checked("binary", [out, a, b]) else {
        return;
    };
    // SAFETY: as in `unary`.
    run("binary", out, |ctx| unsafe {
        ffi::hmr_binary(ctx, binary_code(op), &o, &x, &y)
    });
}

/// `out = a * s`, `a / s` or `s / a`, as `cpu::binary_scalar`.
#[track_caller]
pub(crate) fn binary_scalar(
    op: BinaryOp,
    out: &TensorData,
    a: &TensorData,
    s: f64,
    scalar_first: bool,
) {
    check_shape(out, a);
    assert!(
        matches!(op, BinaryOp::Mul | BinaryOp::Div),
        "{op:?} rounds its scalar to the dtype"
    );
    let Some([o, x]) = checked("binary_scalar", [out, a]) else {
        return;
    };
    // SAFETY: as in `unary`.
    run("binary_scalar", out, |ctx| unsafe {
        ffi::hmr_binary_scalar(ctx, binary_code(op), &o, &x, s, scalar_first as i32)
    });
}

/// `out = op(a, b)` or `op(a, b, c)`, as `cpu::fused`. The coefficients are computed here as the
/// CPU computes them, in the computation type of the dtype, and passed on exactly.
// The `as f64` casts are needed where the computation type is f32.
#[allow(clippy::unnecessary_cast)]
#[track_caller]
pub(crate) fn fused(
    op: Fused,
    out: &TensorData,
    a: &TensorData,
    b: &TensorData,
    c: Option<&TensorData>,
) {
    check_shape(out, a);
    check_shape(out, b);
    let (code, coefficient, small) = dispatch_float!(out.dtype, T => {
        type C = <T as FloatElement>::C;
        let of = |v: f64| v as C;
        match op {
            Fused::ScaledAdd(alpha) => {
                (ffi::HMR_FUSED_SCALED_ADD, T::down(of(alpha)).up() as f64, false)
            }
            Fused::Lerp(weight) => {
                let weight = of(weight);
                let small = weight.abs() < of(0.5);
                let coeff = if small { weight } else { weight - of(1.0) };
                (ffi::HMR_FUSED_LERP, coeff as f64, small)
            }
            Fused::Addcmul(value) => (ffi::HMR_FUSED_ADDCMUL, of(value) as f64, false),
            Fused::Addcdiv(value) => (ffi::HMR_FUSED_ADDCDIV, of(value) as f64, false),
        }
    });
    match c {
        Some(c) => {
            check_shape(out, c);
            let Some([o, x, y, z]) = checked("fused", [out, a, b, c]) else {
                return;
            };
            // SAFETY: as in `unary`.
            run("fused", out, |ctx| unsafe {
                ffi::hmr_fused(ctx, code, &o, &x, &y, &z, coefficient, small as i32)
            });
        }
        None => {
            let Some([o, x, y]) = checked("fused", [out, a, b]) else {
                return;
            };
            // SAFETY: as in `unary`; `c` may be null for the two-operand steps.
            run("fused", out, |ctx| unsafe {
                ffi::hmr_fused(
                    ctx,
                    code,
                    &o,
                    &x,
                    &y,
                    std::ptr::null(),
                    coefficient,
                    small as i32,
                )
            });
        }
    }
}

/// `out = cond ? x : y`, as `cpu::where_`.
#[track_caller]
pub(crate) fn where_(out: &TensorData, cond: &TensorData, x: &TensorData, y: &TensorData) {
    check_shape(out, cond);
    check_shape(out, x);
    check_shape(out, y);
    let Some([o, c, a, b]) = checked("where_", [out, cond, x, y]) else {
        return;
    };
    // SAFETY: as in `unary`.
    run("where_", out, |ctx| unsafe {
        ffi::hmr_where(ctx, &o, &c, &a, &b)
    });
}

/// The lower (`upper` false) or upper triangle of the matrices of `src` from `diagonal` on, as
/// `cpu::triangle`: the last two dimensions are kept as they are (the kernel needs the row and
/// column of each element), the batch dimensions are merged.
#[track_caller]
pub(crate) fn triangle(out: &TensorData, src: &TensorData, diagonal: i64, upper: bool) {
    let (ls, lo) = (&src.layout, &out.layout);
    let nd = ls.shape.len();
    assert!(nd >= 2 && ls.shape == lo.shape);
    if ls.num_elements() == 0 {
        return;
    }
    let batch = &ls.shape[..nd - 2];
    let merged = super::copy::merge_dims(batch, [&lo.strides[..nd - 2], &ls.strides[..nd - 2]])
        .expect("a nonempty shape");
    let mut dims: Vec<(usize, [i64; 2])> = merged
        .into_iter()
        .map(|(size, [o, s])| (size, [o as i64, s as i64]))
        .collect();
    for d in nd - 2..nd {
        dims.push((ls.shape[d], [lo.strides[d] as i64, ls.strides[d] as i64]));
    }
    let offsets = [lo.offset as i64, ls.offset as i64];
    let [o, s] = expect(descs_with_dims("triangle", [out, src], offsets, &dims));
    // SAFETY: as in `unary`.
    run("triangle", out, |ctx| unsafe {
        ffi::hmr_triangle(ctx, &o, &s, diagonal, upper as i32)
    });
}

/// Copies `src` into the contiguous `out`, reversing the dimensions selected by `mask`, as
/// `cpu::flip`: `src` is read through negative strides from the last element of each reversed
/// dimension.
#[track_caller]
pub(crate) fn flip(out: &TensorData, src: &TensorData, mask: &[bool]) {
    let l = &src.layout;
    assert!(out.layout.is_contiguous() && out.layout.shape == l.shape);
    if l.num_elements() == 0 {
        return;
    }
    let mut src_offset = l.offset as i64;
    let mut dims: Vec<(usize, [i64; 2])> = Vec::with_capacity(l.shape.len());
    let out_strides = crate::shape::contiguous_strides(&l.shape);
    for (d, &size) in l.shape.iter().enumerate() {
        if size == 1 {
            continue;
        }
        let mut s = l.strides[d] as i64;
        if mask[d] {
            src_offset += (size as i64 - 1) * s;
            s = -s;
        }
        let st = [out_strides[d] as i64, s];
        // Neighbours merge as in `Plan` when they are contiguous with each other in both
        // operands, which with signed strides also requires both to be reversed or neither.
        if let Some(outer) = dims.last_mut()
            && (0..2).all(|k| st[k].checked_mul(size as i64) == Some(outer.1[k]))
        {
            outer.0 *= size;
            outer.1 = st;
            continue;
        }
        dims.push((size, st));
    }
    let offsets = [out.layout.offset as i64, src_offset];
    let [o, s] = expect(descs_with_dims("flip", [out, src], offsets, &dims));
    // SAFETY: as in `unary`; every element `src` is read at lies inside its layout.
    run("flip", out, |ctx| unsafe { ffi::hmr_flip(ctx, &o, &s) });
}

#[track_caller]
fn check_shape(out: &TensorData, input: &TensorData) {
    assert_eq!(
        out.layout.shape, input.layout.shape,
        "kernel operands must have the same shape"
    );
}
