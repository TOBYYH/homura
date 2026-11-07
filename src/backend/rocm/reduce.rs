//! Reductions on ROCm (`csrc/reduce.hip`, plans in `csrc/reduce.hpp`): the counterparts of the
//! CPU backend's `reduce`, `var`, `arg_reduce`, `softmax` and `exclusive_cumprod`, with the same
//! arguments. Floating point sums and products are accumulated in f64 as on the CPU, in an order
//! that depends only on the shape and strides; the other reductions give the CPU's bits.
//!
//! A reduction is described by the kept dimensions (with the source's and the outputs' strides)
//! and the reduced ones, each merged where they are contiguous with each other. A plan that cuts
//! the reduced range into chunks needs a workspace for the partial results, allocated here before
//! the execution lock is taken, like any other allocation of an operation.

use super::context::exec;
use super::copy::{buffer, descs_with_dims, device_index, merge_dims};
use super::{call, expect, ffi};
use crate::backend::cpu::walk_strides;
use crate::ops::ReduceOp;
use crate::shape::contiguous_strides;
use crate::tensor::TensorData;
use crate::{DType, Shape, Tensor};

fn reduce_code(op: ReduceOp) -> i32 {
    match op {
        ReduceOp::Sum => ffi::HMR_REDUCE_SUM,
        ReduceOp::Prod => ffi::HMR_REDUCE_PROD,
        ReduceOp::Mean => ffi::HMR_REDUCE_MEAN,
        ReduceOp::Max => ffi::HMR_REDUCE_MAX,
        ReduceOp::Min => ffi::HMR_REDUCE_MIN,
        ReduceOp::All => ffi::HMR_REDUCE_ALL,
        ReduceOp::Any => ffi::HMR_REDUCE_ANY,
    }
}

/// The descriptions of a reduction of `src` over the dimensions in `mask`.
pub(super) struct Reduction {
    /// The merged kept dimensions: sizes, and the strides of the outputs and of the source.
    dims: Vec<(usize, [i64; 2])>,
    kept: ffi::hmr_tensor_desc,
    red: ffi::hmr_tensor_desc,
}

/// The descriptions of a reduction, or `None` when it has no outputs.
#[track_caller]
pub(super) fn describe(kernel: &str, src: &TensorData, mask: &[bool]) -> Option<Reduction> {
    let l = &src.layout;
    // The kept dimensions of an empty source may be nonempty; it isn't read.
    let strides = walk_strides(l);
    let (mut kept_shape, mut kept_strides, mut red_shape, mut red_strides) =
        (vec![], vec![], vec![], vec![]);
    for ((&size, &stride), &reduced) in l.shape.iter().zip(strides.iter()).zip(mask) {
        if reduced {
            red_shape.push(size);
            red_strides.push(stride);
        } else {
            kept_shape.push(size);
            kept_strides.push(stride);
        }
    }
    let out_strides = contiguous_strides(&kept_shape);
    let dims: Vec<(usize, [i64; 2])> =
        merge_dims(&kept_shape, [&out_strides[..], &kept_strides[..]])?
            .into_iter()
            .map(|(size, st)| (size, st.map(|s| s as i64)))
            .collect();
    let red_dims: Vec<(usize, [i64; 1])> = match merge_dims(&red_shape, [&red_strides]) {
        Some(d) => d
            .into_iter()
            .map(|(size, st)| (size, st.map(|s| s as i64)))
            .collect(),
        // Nothing to fold.
        None => vec![(0, [0])],
    };
    let src_dims: Vec<(usize, [i64; 1])> = dims.iter().map(|&(n, st)| (n, [st[1]])).collect();
    let [kept] = expect(descs_with_dims(kernel, [src], [l.offset as i64], &src_dims));
    let [red] = expect(descs_with_dims(kernel, [src], [0], &red_dims));
    Some(Reduction { dims, kept, red })
}

/// The plan of a reduction (`csrc/reduce.hpp`): lanes per output, chunks, elements per chunk.
#[cfg(test)]
pub(super) fn plan(r: &Reduction) -> (i64, i64, i64) {
    let (mut group, mut splits, mut chunk) = (0, 0, 0);
    // SAFETY: the descriptions are valid; the call only reads their sizes and strides.
    expect(call(unsafe {
        ffi::hmr_test_reduce_plan(&r.kept, &r.red, &mut group, &mut splits, &mut chunk)
    }));
    (group, splits, chunk)
}

/// The description of a contiguous output of `r`.
#[track_caller]
fn output(kernel: &str, r: &Reduction, out: &TensorData) -> ffi::hmr_tensor_desc {
    assert!(
        out.layout.is_contiguous(),
        "internal error: {kernel} into a strided tensor"
    );
    let dims: Vec<(usize, [i64; 1])> = r.dims.iter().map(|&(n, st)| (n, [st[0]])).collect();
    let [d] = expect(descs_with_dims(
        kernel,
        [out],
        [out.layout.offset as i64],
        &dims,
    ));
    d
}

/// The workspace of a reduction's plan on the device of `src`, if it needs one.
#[track_caller]
fn workspace(r: &Reduction, src: &TensorData, var: bool) -> Option<Tensor> {
    let mut bytes = 0i64;
    // SAFETY: the descriptions are valid; the call only reads their sizes and strides.
    expect(call(unsafe {
        ffi::hmr_reduce_workspace(&r.kept, &r.red, var as i32, &mut bytes)
    }));
    (bytes > 0).then(|| Tensor::empty_on(Shape::new(&[bytes as usize]), DType::U8, src.device()))
}

/// A workspace's pointer and size for the C ABI.
fn raw(kernel: &str, ws: &Option<Tensor>) -> (*mut std::ffi::c_void, i64) {
    match ws {
        Some(w) => {
            let data = w.data();
            (
                buffer(kernel, &data).ptr().cast(),
                data.layout.num_elements() as i64,
            )
        }
        None => (std::ptr::null_mut(), 0),
    }
}

/// Queues `launch` with the context of `data`'s device inside its execution lock, then reports
/// a failure.
#[track_caller]
fn run(
    kernel: &str,
    data: &TensorData,
    launch: impl FnOnce(ffi::hmr_context_t) -> ffi::hmr_status,
) {
    let ctx = buffer(kernel, data).context();
    expect(exec(device_index(data), || call(launch(ctx.raw()))));
}

/// Reduces the dimensions of `src` selected by `mask` into the contiguous `out`, as
/// `cpu::reduce` (output dtypes and empty reductions alike).
#[track_caller]
pub(crate) fn reduce(op: ReduceOp, out: &TensorData, src: &TensorData, mask: &[bool]) {
    let Some(r) = describe("reduce", src, mask) else {
        return;
    };
    let o = output("reduce", &r, out);
    let ws = workspace(&r, src, false);
    let (ptr, bytes) = raw("reduce", &ws);
    // SAFETY: the descriptions lie inside device memory of `out`'s device (ops put the source
    // there); `out` is a new tensor and the workspace holds `bytes` bytes of the same device.
    run("reduce", out, |ctx| unsafe {
        ffi::hmr_reduce(ctx, reduce_code(op), &o, &r.kept, &r.red, ptr, bytes)
    });
}

/// Variance (or standard deviation) over the dimensions selected by `mask`, as `cpu::var`.
#[track_caller]
pub(crate) fn var(
    out: &TensorData,
    src: &TensorData,
    mask: &[bool],
    correction: f64,
    take_sqrt: bool,
) {
    let Some(r) = describe("var", src, mask) else {
        return;
    };
    let o = output("var", &r, out);
    let ws = workspace(&r, src, true);
    let (ptr, bytes) = raw("var", &ws);
    // SAFETY: as in `reduce`.
    run("var", out, |ctx| unsafe {
        ffi::hmr_var(
            ctx,
            &o,
            &r.kept,
            &r.red,
            correction,
            take_sqrt as i32,
            ptr,
            bytes,
        )
    });
}

/// Index (and optionally value) of the maximum or minimum along `dim`, as `cpu::arg_reduce`.
#[track_caller]
pub(crate) fn arg_reduce(
    values: Option<&TensorData>,
    indices: &TensorData,
    src: &TensorData,
    dim: usize,
    is_max: bool,
) {
    let nd = src.layout.shape.len();
    let mask: Vec<bool> = (0..nd).map(|d| d == dim).collect();
    if nd > 0 {
        assert!(
            src.layout.shape[dim] > 0,
            "cannot compute the arg reduction of an empty dimension"
        );
    }
    let Some(r) = describe("arg_reduce", src, &mask) else {
        return;
    };
    let i = output("arg_reduce", &r, indices);
    let v = values.map(|v| output("arg_reduce", &r, v));
    let ws = workspace(&r, src, false);
    let (ptr, bytes) = raw("arg_reduce", &ws);
    let vp = v.as_ref().map_or(std::ptr::null(), |v| v as *const _);
    // SAFETY: as in `reduce`; `values`, when given, is another new tensor.
    run("arg_reduce", indices, |ctx| unsafe {
        ffi::hmr_arg_reduce(ctx, vp, &i, &r.kept, &r.red, is_max as i32, ptr, bytes)
    });
}

/// The descriptions of the lines of `out` and `src` along `dim` (the other dimensions, merged),
/// the line length, and the strides along the line; `None` without elements.
#[track_caller]
fn lines(
    kernel: &str,
    out: &TensorData,
    src: &TensorData,
    dim: usize,
    out_strides: &[usize],
) -> Option<([ffi::hmr_tensor_desc; 2], usize, usize, usize)> {
    let (ls, lo) = (&src.layout, &out.layout);
    assert_eq!(
        ls.shape, lo.shape,
        "kernel operands must have the same shape"
    );
    if ls.num_elements() == 0 {
        return None;
    }
    let shape = &ls.shape;
    let others = |v: &[usize]| -> Vec<usize> {
        v.iter()
            .enumerate()
            .filter(|&(d, _)| d != dim)
            .map(|(_, &s)| s)
            .collect()
    };
    let (n, so, si) = if shape.is_empty() {
        (1, 0, 0)
    } else {
        (shape[dim], out_strides[dim], ls.strides[dim])
    };
    let dims: Vec<(usize, [i64; 2])> =
        merge_dims(&others(shape), [&others(out_strides), &others(&ls.strides)])?
            .into_iter()
            .map(|(size, st)| (size, st.map(|s| s as i64)))
            .collect();
    let d = expect(descs_with_dims(
        kernel,
        [out, src],
        [lo.offset as i64, ls.offset as i64],
        &dims,
    ));
    Some((d, n, so, si))
}

/// Softmax (or log-softmax) of `src` along `dim` into `out`, as `cpu::softmax`. Few long rows
/// are split across blocks, with a workspace for their maxima, sums and partial results.
#[track_caller]
pub(crate) fn softmax(out: &TensorData, src: &TensorData, dim: usize, log: bool) {
    let out_strides = out.layout.strides.clone();
    let Some(([o, s], n, so, si)) = lines("softmax", out, src, dim, &out_strides) else {
        return;
    };
    let mut bytes = 0i64;
    // SAFETY: the description is valid; the call only reads its sizes and strides.
    expect(call(unsafe {
        ffi::hmr_softmax_workspace(&s, n as i64, si as i64, &mut bytes)
    }));
    let ws = (bytes > 0)
        .then(|| Tensor::empty_on(Shape::new(&[bytes as usize]), DType::U8, src.device()));
    let (ptr, bytes) = raw("softmax", &ws);
    // SAFETY: the descriptions lie inside device memory of `out`'s device; `out` is a new
    // tensor and the workspace holds `bytes` bytes of the same device.
    run("softmax", out, |ctx| unsafe {
        ffi::hmr_softmax(
            ctx, &o, &s, n as i64, so as i64, si as i64, log as i32, ptr, bytes,
        )
    });
}

/// The plan of a softmax of `src` along `dim` (`csrc/reduce.hip`): whether it splits its rows,
/// the lanes per row, and the elements a lane keeps in registers (0: none).
#[cfg(test)]
pub(super) fn softmax_plan(src: &TensorData, dim: usize) -> (bool, i32, i32) {
    let strides = src.layout.strides.clone();
    let ([_, s], n, _, si) = lines("test", src, src, dim, &strides).expect("elements");
    let (mut split, mut group, mut cache) = (0, 0, 0);
    // SAFETY: the description is valid; the call only reads its sizes and strides.
    expect(call(unsafe {
        ffi::hmr_test_softmax_plan(&s, n as i64, si as i64, &mut split, &mut group, &mut cache)
    }));
    (split != 0, group, cache)
}

/// The exclusive cumulative product of `src` along `dim` into the contiguous `out`, as
/// `cpu::exclusive_cumprod`.
#[track_caller]
pub(crate) fn exclusive_cumprod(out: &TensorData, src: &TensorData, dim: usize, reverse: bool) {
    assert!(out.layout.is_contiguous() && out.layout.shape == src.layout.shape);
    assert_eq!(out.dtype, src.dtype);
    let out_strides = contiguous_strides(&out.layout.shape);
    let Some(([o, s], n, so, si)) = lines("exclusive_cumprod", out, src, dim, &out_strides) else {
        return;
    };
    // SAFETY: as in `softmax`.
    run("exclusive_cumprod", out, |ctx| unsafe {
        ffi::hmr_exclusive_cumprod(ctx, &o, &s, n as i64, so as i64, si as i64, reverse as i32)
    });
}
