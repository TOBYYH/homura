//! Random sampling on ROCm (`csrc/random.hip`): the generator of `ops::random` evaluated on the
//! device. The caller reserves the Philox blocks of a call as the CPU does; these functions only
//! compute the samples of the reserved blocks.

use super::context::exec;
use super::copy::{buffer, descs, device_index};
use super::{call, expect, ffi};
use crate::tensor::TensorData;
use crate::{DType, Device, Tensor};

/// The description of a contiguous output: one flat dimension.
#[track_caller]
fn flat(kernel: &str, out: &TensorData) -> Option<ffi::hmr_tensor_desc> {
    assert!(
        out.layout.is_contiguous(),
        "internal error: {kernel} into a strided tensor"
    );
    expect(descs(kernel, [out])).map(|[d]| d)
}

/// Queues `launch` with the context of `out`'s device inside its execution lock, then reports
/// a failure.
#[track_caller]
fn run(kernel: &str, out: &TensorData, launch: impl FnOnce(ffi::hmr_context_t) -> ffi::hmr_status) {
    let ctx = buffer(kernel, out).context();
    expect(exec(device_index(out), || call(launch(ctx.raw()))));
}

/// Fills the contiguous float tensor `out` with uniform samples from `[from, to)` drawn from
/// the blocks from `offset` on (one block per 4 elements, per 2 for F64).
#[track_caller]
pub(crate) fn uniform(out: &TensorData, seed: u64, offset: u64, from: f64, to: f64) {
    let Some(d) = flat("uniform", out) else {
        return;
    };
    // SAFETY: the description lies inside device memory of `out`'s device, a new tensor.
    run("uniform", out, |ctx| unsafe {
        ffi::hmr_random_uniform(ctx, &d, seed, offset, from, to)
    });
}

/// Fills the contiguous float tensor `out` with normal samples drawn from the blocks from
/// `offset` on (one block per 2 elements).
#[track_caller]
pub(crate) fn normal(out: &TensorData, seed: u64, offset: u64, mean: f64, std: f64) {
    let Some(d) = flat("normal", out) else {
        return;
    };
    // SAFETY: as in `uniform`.
    run("normal", out, |ctx| unsafe {
        ffi::hmr_random_normal(ctx, &d, seed, offset, mean, std)
    });
}

/// The workspace of [`int_batch`] for batches of up to `blocks` blocks on `device`.
#[track_caller]
pub(crate) fn int_workspace(device: Device, blocks: usize) -> Tensor {
    let tiles = blocks.div_ceil(ffi::HMR_RANDOM_INT_TILE as usize);
    Tensor::uninit_on(crate::Shape::new(&[2 * tiles + 1]), DType::I64, device)
}

/// One batch of integer rejection sampling into the contiguous I64 tensor `out` (the elements
/// still to fill): the words of `blocks` blocks from `offset` on that are at least `threshold`,
/// in the CPU's order. Returns how many words were accepted, which it reads back (a
/// synchronization), unless `threshold` is 0: then every word is accepted, `out` is filled, and
/// `workspace` isn't needed.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub(crate) fn int_batch(
    out: &TensorData,
    seed: u64,
    offset: u64,
    blocks: u64,
    low: i64,
    range: u64,
    threshold: u64,
    workspace: Option<&Tensor>,
) -> Option<u64> {
    let d = flat("randint", out)?;
    let ws = match workspace {
        Some(w) if threshold != 0 => {
            let data = w.data();
            let tiles = (blocks as usize).div_ceil(ffi::HMR_RANDOM_INT_TILE as usize);
            assert!(
                2 * tiles < data.layout.num_elements() && data.layout.offset == 0,
                "internal error: the workspace of randint"
            );
            Some((data, tiles))
        }
        _ => {
            assert_eq!(threshold, 0, "internal error: randint without a workspace");
            None
        }
    };
    let ptr = ws.as_ref().map_or(std::ptr::null_mut(), |(data, _)| {
        buffer("randint", data).ptr()
    });
    // SAFETY: the description lies inside device memory of `out`'s device; the workspace, when
    // given, holds `2 * tiles + 1` words of the same device.
    run("randint", out, |ctx| unsafe {
        ffi::hmr_random_int(
            ctx,
            &d,
            seed,
            offset,
            blocks,
            low,
            range,
            threshold,
            ptr.cast(),
        )
    });
    let (data, tiles) = ws?;
    let count = Tensor::from_data(data)
        .narrow(0, 2 * tiles as i64, 1)
        .data();
    Some(super::read_host::<i64>(&count)[0] as u64)
}
