//! Indexing on ROCm (`csrc/index.hip`): the counterparts of the CPU backend's `index_select`,
//! `gather`, `index_overwrite`, `index_add`, `scatter_add`, `mask_coordinates`,
//! `index_positions` and `out_of_range`, and of `max_pool2d_backward`, a scatter of its gradient.
//!
//! The positions in an index tensor are checked on the device: the check kernels leave the first
//! position out of range (in row-major order of the index's logical positions) and its value in
//! a small status buffer, which is read back once (a synchronization). The check, the read-back
//! and the kernels that use the positions run in one critical section of the device's execution
//! lock, so they see the same index data. The messages are the CPU's, reported at the caller.
//!
//! Scatters that add group their elements by target with a stable radix sort on the device and
//! add each group in the CPU's order, so the results are the CPU's bits. The workspaces they need
//! are allocated here before the execution lock is taken.

use super::context::{Context, exec};
use super::copy::{buffer, descs_with_dims, device_index, merge_dims};
use super::{Failure, call, expect, ffi, workspace};
use crate::backend::cpu::walk_strides;
use crate::backend::{Bounds, out_of_bounds};
use crate::tensor::{Tensor, TensorData};
use crate::{DType, Device, Shape};

/// A check of an index tensor: its description and the range its values must lie in.
struct Check {
    index: ffi::hmr_tensor_desc,
    low: i64,
    high: i64,
    exempt: Option<i64>,
}

/// The description of `data` with its own dimensions merged where they are contiguous (the
/// row-major order of its elements doesn't change), or `None` if it has no elements.
#[track_caller]
fn flat_desc(kernel: &str, data: &TensorData) -> Result<Option<ffi::hmr_tensor_desc>, Failure> {
    let l = &data.layout;
    let Some(dims) = merge_dims(&l.shape, [&l.strides[..]]) else {
        return Ok(None);
    };
    let dims: Vec<(usize, [i64; 1])> = dims.into_iter().map(|(n, [s])| (n, [s as i64])).collect();
    descs_with_dims(kernel, [data], [l.offset as i64], &dims).map(|[d]| Some(d))
}

/// The description of `data` with all its dimensions, none merged: for the kernels that need the
/// coordinates themselves (the planes of pooling, the coordinates of a mask).
#[track_caller]
pub(super) fn full_desc(kernel: &str, data: &TensorData) -> Result<ffi::hmr_tensor_desc, Failure> {
    let l = &data.layout;
    let strides = walk_strides(l);
    let dims: Vec<(usize, [i64; 1])> = l
        .shape
        .iter()
        .zip(strides.iter())
        .map(|(&n, &s)| (n, [s as i64]))
        .collect();
    descs_with_dims(kernel, [data], [l.offset as i64], &dims).map(|[d]| d)
}

/// The descriptions of `operands`, walked over `shape` (which all share but along `dim`), with
/// the dimensions before `dim` and those after it merged as the CPU merges them, and `dim` kept
/// in between; and the position of `dim` among them. `shape` has elements.
#[track_caller]
fn around<const N: usize>(
    kernel: &str,
    operands: [&TensorData; N],
    shape: &[usize],
    dim: usize,
) -> Result<([ffi::hmr_tensor_desc; N], usize), Failure> {
    let strides: [&[usize]; N] = std::array::from_fn(|k| &operands[k].layout.strides[..]);
    let part = |range: std::ops::Range<usize>| {
        let st: [&[usize]; N] = std::array::from_fn(|k| &strides[k][range.clone()]);
        merge_dims(&shape[range.clone()], st).expect("a nonempty shape")
    };
    let mut dims: Vec<(usize, [i64; N])> = vec![];
    let to_i64 = |(n, st): (usize, [usize; N])| (n, st.map(|s| s as i64));
    dims.extend(part(0..dim).into_iter().map(to_i64));
    let at = dims.len();
    dims.push((shape[dim], std::array::from_fn(|k| strides[k][dim] as i64)));
    dims.extend(part(dim + 1..shape.len()).into_iter().map(to_i64));
    let offsets = operands.map(|d| d.layout.offset as i64);
    Ok((descs_with_dims(kernel, operands, offsets, &dims)?, at))
}

/// The address of a contiguous tensor's first element.
fn first_element<T>(kernel: &str, data: &TensorData) -> *mut T {
    // SAFETY: the offset lies inside the storage.
    unsafe {
        buffer(kernel, data)
            .ptr()
            .add(data.layout.offset * data.dtype.size_in_bytes())
            .cast()
    }
}

/// A workspace's pointer and size for the C ABI.
fn raw(kernel: &str, ws: &Option<Tensor>) -> (*mut std::ffi::c_void, i64) {
    match ws {
        Some(w) => {
            let data = w.data();
            (
                first_element(kernel, &data),
                data.layout.num_elements() as i64,
            )
        }
        None => (std::ptr::null_mut(), 0),
    }
}

/// The status buffer of `n` checks on `device`: two words per check.
#[track_caller]
fn status(n: usize, device: Device) -> Tensor {
    Tensor::empty_on(Shape::new(&[2 * n.max(1)]), DType::I64, device)
}

/// Inside the execution lock of `data`'s device: queues `checks` (each writing its slot of
/// `slots`), reads the slots back, and returns the first check that found a position out of
/// range with its value; without one, queues `then`. Checks of empty indices find nothing.
/// Reports failures of the device in the outer result, at the caller.
#[track_caller]
fn checked<R>(
    kernel: &str,
    data: &TensorData,
    slots: &Tensor,
    checks: &[Option<Check>],
    then: impl FnOnce(&Context) -> Result<R, Failure>,
) -> Result<Result<R, (usize, i64)>, Failure> {
    let ctx = buffer(kernel, data).context();
    let slots = slots.data();
    let base: *mut u64 = first_element(kernel, &slots);
    exec(device_index(data), || {
        for (k, check) in checks.iter().enumerate() {
            let Some(c) = check else { continue };
            // SAFETY: the description lies inside device memory of this device; slot `k` is two
            // words of the status buffer, which has two per check.
            call(unsafe {
                ffi::hmr_index_check(
                    ctx.raw(),
                    &c.index,
                    c.low,
                    c.high,
                    c.exempt.is_some() as i32,
                    c.exempt.unwrap_or(0),
                    base.add(2 * k),
                )
            })?;
        }
        if checks.iter().any(Option::is_some) {
            let mut bytes = vec![0u8; 16 * checks.len()];
            ctx.copy_to_host(&mut bytes, base.cast())?;
            let present: Vec<bool> = checks.iter().map(Option::is_some).collect();
            if let Some(found) = first_violation(&bytes, &present) {
                return Ok(Err(found));
            }
        }
        then(ctx).map(Ok)
    })
}

/// The first check, in order, among those `present`, whose slot in the status bytes `slots` (two
/// native-endian words per check: the first offending logical position, `u64::MAX` for none, and
/// its value) found a position out of range: its number and the value.
pub(super) fn first_violation(slots: &[u8], present: &[bool]) -> Option<(usize, i64)> {
    let word = |i: usize| {
        let mut w = [0u8; 8];
        w.copy_from_slice(&slots[8 * i..8 * i + 8]);
        u64::from_ne_bytes(w)
    };
    (0..present.len())
        .find_map(|k| (present[k] && word(2 * k) != u64::MAX).then(|| (k, word(2 * k + 1) as i64)))
}

/// The check of `index`'s positions along a dimension of `size` that `bounds` asks for.
#[track_caller]
fn bounds_check(
    kernel: &str,
    index: &TensorData,
    size: usize,
    bounds: Bounds,
) -> Result<Option<Check>, Failure> {
    let n = size as i64;
    let low = match bounds {
        Bounds::Valid => return Ok(None),
        Bounds::NonNegative => 0,
        Bounds::AllowNegative => -n,
    };
    Ok(flat_desc(kernel, index)?.map(|d| Check {
        index: d,
        low,
        high: n,
        exempt: None,
    }))
}

/// Runs an indexing kernel whose single index tensor `index` is checked against a dimension
/// `dim` of `size` (as `bounds` says), panicking with the CPU's message for a position out of
/// range.
#[track_caller]
fn run_checked(
    kernel: &str,
    data: &TensorData,
    index: &TensorData,
    dim: usize,
    size: usize,
    bounds: Bounds,
    then: impl FnOnce(ffi::hmr_context_t) -> ffi::hmr_status,
) {
    let check = expect(bounds_check(kernel, index, size, bounds));
    let slots = status(1, data.device());
    match expect(checked(kernel, data, &slots, &[check], |ctx| {
        call(then(ctx.raw()))
    })) {
        Ok(()) => {}
        Err((_, v)) => panic!("{}", out_of_bounds(v, dim, size)),
    }
}

/// See `cpu::index_select`.
#[track_caller]
pub(crate) fn index_select(
    out: &TensorData,
    src: &TensorData,
    dim: usize,
    index: &TensorData,
    bounds: Bounds,
) {
    let size = src.layout.shape[dim];
    let shape = out.layout.shape.clone();
    let descs = if shape.num_elements() == 0 {
        None
    } else {
        // The index walked along `dim` of out, its other strides 0.
        let mut strides = vec![0; shape.len()];
        strides[dim] = walk_strides(&index.layout).first().copied().unwrap_or(0);
        let walk = TensorData {
            layout: crate::tensor::Layout {
                shape: shape.clone(),
                strides: crate::Strides::new(&strides),
                offset: index.layout.offset,
            },
            ..index.clone()
        };
        let (mut d, at) = expect(around("index_select", [out, src, &walk], &shape, dim));
        d[1].sizes[at] = size as i64;
        Some((d, at))
    };
    run_checked("index_select", out, index, dim, size, bounds, |ctx| {
        let Some(([o, s, w], at)) = descs else {
            return ffi::HMR_OK;
        };
        // The 1-D index: its walk along `dim`.
        let mut i = w;
        i.ndim = 1;
        i.sizes[0] = w.sizes[at];
        i.strides[0] = w.strides[at];
        // SAFETY: the descriptions lie inside device memory of this device; the positions were
        // checked in the same critical section.
        unsafe {
            ffi::hmr_index_select(
                ctx,
                &o,
                &s,
                &i,
                at as i32,
                (bounds == Bounds::AllowNegative) as i32,
            )
        }
    });
}

/// See `cpu::gather`.
#[track_caller]
pub(crate) fn gather(out: &TensorData, src: &TensorData, dim: usize, index: &TensorData) {
    let size = src.layout.shape[dim];
    let shape = out.layout.shape.clone();
    let descs = if shape.num_elements() > 0 {
        Some(expect(around("gather", [out, src, index], &shape, dim)))
    } else {
        None
    };
    run_checked(
        "gather",
        out,
        index,
        dim,
        size,
        Bounds::NonNegative,
        |ctx| {
            let Some(([o, s, i], at)) = descs else {
                return ffi::HMR_OK;
            };
            // SAFETY: as in `index_select`.
            unsafe { ffi::hmr_gather(ctx, &o, &s, &i, at as i32) }
        },
    );
}

/// See `cpu::index_overwrite`.
#[track_caller]
pub(crate) fn index_overwrite(out: &TensorData, dim: usize, index: &TensorData, src: &TensorData) {
    let shape = src.layout.shape.clone();
    if shape.num_elements() == 0 {
        return;
    }
    assert!(
        index.layout.is_contiguous() && index.dtype == DType::I64,
        "internal error: index_overwrite with positions that aren't contiguous I64"
    );
    let size = out.layout.shape[dim];
    let ([s, mut o], at) = expect(around("index_overwrite", [src, out], &shape, dim));
    o.sizes[at] = size as i64;
    // One dimension even for a single position, as the kernel expects.
    let m = index.layout.num_elements();
    let [p] = expect(descs_with_dims(
        "index_overwrite",
        [index],
        [index.layout.offset as i64],
        &[(m, [1])],
    ));
    let mut bytes = 0;
    // SAFETY: a valid output location.
    expect(call(unsafe {
        ffi::hmr_index_overwrite_workspace(size as i64, &mut bytes)
    }));
    let ws = workspace(bytes, out.device());
    let (ptr, len) = raw("index_overwrite", &ws);
    let ctx = buffer("index_overwrite", out).context();
    expect(exec(device_index(out), || {
        // SAFETY: the descriptions and the workspace lie inside device memory of this device;
        // the positions are valid (`Bounds::Valid`).
        call(unsafe { ffi::hmr_index_overwrite(ctx.raw(), &o, &s, &p, at as i32, ptr, len) })
    }));
}

/// See `cpu::index_add`.
#[track_caller]
pub(crate) fn index_add(
    out: &TensorData,
    dim: usize,
    index: &TensorData,
    src: &TensorData,
    bounds: Bounds,
) {
    let size = out.layout.shape[dim];
    let m = index.layout.num_elements();
    let shape = out.layout.shape.clone();
    let work = src.layout.num_elements() > 0;
    let descs = if work {
        let ([o, mut s], at) = expect(around("index_add", [out, src], &shape, dim));
        s.sizes[at] = m as i64;
        Some(([o, s], at))
    } else {
        None
    };
    let index_desc = if work {
        expect(flat_desc("index_add", index))
    } else {
        None
    };
    let mut bytes = 0;
    if work {
        // SAFETY: a valid output location.
        expect(call(unsafe {
            ffi::hmr_index_add_workspace(m as i64, size as i64, &mut bytes)
        }));
    }
    let ws = workspace(bytes, out.device());
    let (ptr, len) = raw("index_add", &ws);
    run_checked("index_add", out, index, dim, size, bounds, |ctx| {
        let (Some(([o, s], at)), Some(i)) = (descs, index_desc) else {
            return ffi::HMR_OK;
        };
        // SAFETY: the descriptions and the workspace lie inside device memory of this device;
        // the positions were checked in the same critical section.
        unsafe { ffi::hmr_index_add(ctx, &o, &s, &i, at as i32, ptr, len) }
    });
}

/// See `cpu::scatter_add`.
#[track_caller]
pub(crate) fn scatter_add(out: &TensorData, dim: usize, index: &TensorData, src: &TensorData) {
    let size = out.layout.shape[dim];
    let shape = index.layout.shape.clone();
    let n = shape.num_elements();
    let descs = if n > 0 {
        let ([i, mut o, s], at) = expect(around("scatter_add", [index, out, src], &shape, dim));
        o.sizes[at] = size as i64;
        Some(([i, o, s], at))
    } else {
        None
    };
    let mut bytes = 0;
    // SAFETY: a valid output location.
    expect(call(unsafe {
        ffi::hmr_scatter_add_workspace(n as i64, &mut bytes)
    }));
    let ws = workspace(bytes, out.device());
    let (ptr, len) = raw("scatter_add", &ws);
    run_checked(
        "scatter_add",
        out,
        index,
        dim,
        size,
        Bounds::NonNegative,
        |ctx| {
            let Some(([i, o, s], at)) = descs else {
                return ffi::HMR_OK;
            };
            // SAFETY: as in `index_add`.
            unsafe { ffi::hmr_scatter_add(ctx, &o, &s, &i, at as i32, ptr, len) }
        },
    );
}

/// See `cpu::max_pool2d_backward`.
#[track_caller]
pub(crate) fn max_pool2d_backward(out: &TensorData, grad: &TensorData, indices: &TensorData) {
    let n = grad.layout.num_elements();
    if n == 0 {
        return;
    }
    assert!(
        out.layout.is_contiguous(),
        "internal error: max_pool2d_backward into a strided tensor"
    );
    let nd = out.layout.shape.len();
    let size = out.layout.shape[nd - 2] * out.layout.shape[nd - 1];
    let descs = expect(full_desc("max_pool2d_backward", out).and_then(|o| {
        Ok([
            o,
            full_desc("max_pool2d_backward", grad)?,
            full_desc("max_pool2d_backward", indices)?,
        ])
    }));
    let check = expect(flat_desc("max_pool2d_backward", indices)).map(|d| Check {
        index: d,
        low: 0,
        high: size as i64,
        exempt: None,
    });
    let mut bytes = 0;
    // SAFETY: a valid output location.
    expect(call(unsafe {
        ffi::hmr_max_pool2d_backward_workspace(n as i64, &mut bytes)
    }));
    let ws = workspace(bytes, out.device());
    let (ptr, len) = raw("max_pool2d_backward", &ws);
    let slots = status(1, out.device());
    let [o, g, i] = descs;
    let result = expect(checked(
        "max_pool2d_backward",
        out,
        &slots,
        &[check],
        |ctx| {
            // SAFETY: the descriptions and the workspace lie inside device memory of this
            // device; the indices were checked in the same critical section.
            call(unsafe { ffi::hmr_max_pool2d_backward(ctx.raw(), &o, &g, &i, ptr, len) })
        },
    ));
    if let Err((_, index)) = result {
        panic!("max_pool2d index {index} is outside a plane of {size} elements");
    }
}

/// See `cpu::mask_coordinates`: counts the true elements, reads their number back, then
/// allocates and writes the coordinates, all in one critical section, so that both passes see
/// the same mask.
#[track_caller]
pub(crate) fn mask_coordinates(mask: &TensorData) -> Vec<TensorData> {
    let nd = mask.layout.shape.len();
    let n = mask.layout.num_elements();
    let device = mask.device();
    if n == 0 {
        let mut coords = vec![];
        for _ in 0..nd {
            coords.push(Tensor::empty_on(Shape::new(&[0]), DType::I64, device).data());
        }
        return coords;
    }
    let desc = expect(full_desc("mask_coordinates", mask));
    let (mut bytes, mut total_word) = (0, 0);
    // SAFETY: valid output locations.
    expect(call(unsafe {
        ffi::hmr_mask_workspace(n as i64, &mut bytes, &mut total_word)
    }));
    let ws = workspace(bytes, device);
    let (ptr, len) = raw("mask_coordinates", &ws);
    let ctx = buffer("mask_coordinates", mask).context();
    let result = exec(device_index(mask), || -> Result<_, Failure> {
        // SAFETY: the description and the workspace lie inside device memory of this device.
        call(unsafe { ffi::hmr_mask_count(ctx.raw(), &desc, ptr, len) })?;
        let mut word = [0u8; 8];
        // SAFETY: word `total_word` lies inside the workspace.
        let total = unsafe { ptr.cast::<u64>().add(total_word as usize) };
        ctx.copy_to_host(&mut word, total.cast())?;
        let count = u64::from_ne_bytes(word) as usize;
        let coords = match Tensor::try_empty_on(Shape::new(&[nd, count]), DType::I64, device) {
            Ok(t) => t,
            Err(e) => return Ok(Err(e)),
        };
        let out = coords.data();
        let [o] = descs_with_dims(
            "mask_coordinates",
            [&out],
            [out.layout.offset as i64],
            &[(nd, [count as i64]), (count, [1])],
        )?;
        // SAFETY: as above; `out` is new device memory of this device.
        call(unsafe { ffi::hmr_mask_coordinates(ctx.raw(), &desc, ptr, len, &o) })?;
        Ok(Ok((coords, count)))
    });
    let (coords, count) = match expect(result) {
        Ok(c) => c,
        Err(e) => e.raise(),
    };
    debug_assert_eq!(coords.shape(), [nd, count]);
    (0..nd).map(|d| coords.select(0, d as i64).data()).collect()
}

/// See `backend::index_positions`: the indices are checked and, without a position out of
/// range, `out` written, in one critical section.
#[track_caller]
pub(crate) fn index_positions(
    out: Option<&TensorData>,
    coords: &[Option<TensorData>],
    sizes: &[usize],
) -> Option<(usize, i64)> {
    let first = out
        .or(coords.iter().flatten().next())
        .expect("a tensor on the device");
    // Loops, not closures, around the calls that may panic: a closure would hide the caller's
    // location.
    let mut checks: Vec<Option<Check>> = vec![];
    for (c, &size) in coords.iter().zip(sizes) {
        let desc = match c {
            Some(c) => expect(flat_desc("index_positions", c)),
            None => None,
        };
        checks.push(desc.map(|d| Check {
            index: d,
            low: -(size as i64),
            high: size as i64,
            exempt: None,
        }));
    }
    // The positions: one pass per index over the broadcast shape.
    let strides = crate::shape::contiguous_strides(sizes);
    let mut passes: Vec<([ffi::hmr_tensor_desc; 2], i64, i64)> = vec![];
    if let Some(o) = out
        && o.layout.num_elements() > 0
    {
        for (k, c) in coords.iter().enumerate() {
            let c = c.as_ref().expect("indices with values");
            let walked = c.expand_to(&o.layout.shape);
            let d = expect(super::copy::descs("index_positions", [o, &walked]))
                .expect("a nonempty shape");
            passes.push((d, sizes[k] as i64, strides[k] as i64));
        }
    }
    let slots = status(coords.len(), first.device());
    let result = expect(checked("index_positions", first, &slots, &checks, |ctx| {
        for (k, ([o, c], size, stride)) in passes.iter().enumerate() {
            // SAFETY: the descriptions lie inside device memory of this device; the coordinates
            // were checked in the same critical section.
            call(unsafe {
                ffi::hmr_index_positions(ctx.raw(), o, c, *size, *stride, (k == 0) as i32)
            })?;
        }
        Ok(())
    }));
    result.err()
}

/// See `cpu::out_of_range`.
#[track_caller]
pub(crate) fn out_of_range(
    index: &TensorData,
    low: i64,
    high: i64,
    exempt: Option<i64>,
) -> Option<i64> {
    let check = expect(flat_desc("out_of_range", index)).map(|d| Check {
        index: d,
        low,
        high,
        exempt,
    });
    let slots = status(1, index.device());
    expect(checked("out_of_range", index, &slots, &[check], |_| Ok(())))
        .err()
        .map(|(_, v)| v)
}
