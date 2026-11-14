//! Moving ROCm tensor data: copies and conversions on a device, fills, and the transfers between
//! a device and host memory.

use super::context::{Context, exec};
use super::{Failure, expect, ffi};
use crate::backend::cpu::dispatch_all;
use crate::dtype::sealed::Sealed;
use crate::tensor::TensorData;
use crate::{DType, Device, Element, Scalar};

/// The most dimensions of a kernel operand, after merging (`HMR_MAX_DIMS`, PyTorch's limit too).
pub(crate) const MAX_DIMS: usize = ffi::HMR_MAX_DIMS as usize;

pub(super) fn dtype_code(dtype: DType) -> i32 {
    match dtype {
        DType::Bool => ffi::HMR_BOOL,
        DType::U8 => ffi::HMR_U8,
        DType::I8 => ffi::HMR_I8,
        DType::I16 => ffi::HMR_I16,
        DType::I32 => ffi::HMR_I32,
        DType::I64 => ffi::HMR_I64,
        DType::F16 => ffi::HMR_F16,
        DType::BF16 => ffi::HMR_BF16,
        DType::F32 => ffi::HMR_F32,
        DType::F64 => ffi::HMR_F64,
    }
}

/// `shape` and the strides of each operand with the dimensions merged as the CPU's `Plan` merges
/// them: size-1 dimensions dropped, and neighbours that are contiguous with each other in every
/// operand joined. `None` for an empty shape. Operands with the same strides merge to one
/// dimension when they are contiguous.
#[track_caller]
pub(crate) fn merge_dims<const N: usize>(
    shape: &[usize],
    strides: [&[usize]; N],
) -> Option<Vec<(usize, [usize; N])>> {
    if shape.contains(&0) {
        return None;
    }
    let mut dims: Vec<(usize, [usize; N])> = Vec::with_capacity(shape.len());
    for (i, &size) in shape.iter().enumerate() {
        if size == 1 {
            continue;
        }
        let st: [usize; N] = std::array::from_fn(|k| strides[k][i]);
        if let Some(outer) = dims.last_mut()
            && (0..N).all(|k| st[k].checked_mul(size) == Some(outer.1[k]))
        {
            outer.0 *= size;
            outer.1 = st;
            continue;
        }
        dims.push((size, st));
    }
    Some(dims)
}

/// The kernel descriptions of `operands`, which have the same shape and live on one device, or
/// `None` if they have no elements. Fails with `HMR_INVALID` if more than [`MAX_DIMS`]
/// dimensions remain after merging (the CPU has no limit): the caller reports it where the
/// user's call site is known, like the device's failures.
///
/// # Panics
///
/// Panics (an internal error) for storage without device memory.
#[track_caller]
pub(super) fn descs<const N: usize>(
    kernel: &str,
    operands: [&TensorData; N],
) -> Result<Option<[ffi::hmr_tensor_desc; N]>, Failure> {
    let shape = &operands[0].layout.shape;
    let strides: [&[usize]; N] = std::array::from_fn(|k| &operands[k].layout.strides[..]);
    let Some(dims) = merge_dims(shape, strides) else {
        return Ok(None);
    };
    let dims: Vec<(usize, [i64; N])> = dims
        .into_iter()
        .map(|(size, st)| (size, st.map(|s| s as i64)))
        .collect();
    let offsets = operands.map(|d| d.layout.offset as i64);
    descs_with_dims(kernel, operands, offsets, &dims).map(Some)
}

/// The kernel descriptions of `operands` with the given element offsets and dimensions
/// (`(size, stride of each operand)`, outermost first). Fails like [`descs`] for more than
/// [`MAX_DIMS`] dimensions.
#[track_caller]
pub(super) fn descs_with_dims<const N: usize>(
    kernel: &str,
    operands: [&TensorData; N],
    offsets: [i64; N],
    dims: &[(usize, [i64; N])],
) -> Result<[ffi::hmr_tensor_desc; N], Failure> {
    if dims.len() > MAX_DIMS {
        return Err(Failure {
            status: ffi::HMR_INVALID,
            message: format!(
                "{kernel} on ROCm supports at most {MAX_DIMS} dimensions that can't be merged, \
                 got {}",
                dims.len()
            ),
        });
    }
    Ok(std::array::from_fn(|k| {
        let d = operands[k];
        // SAFETY: all-zero bytes are a valid description.
        let mut desc: ffi::hmr_tensor_desc = unsafe { std::mem::zeroed() };
        desc.data = buffer(kernel, d).ptr().cast();
        desc.offset = offsets[k];
        desc.dtype = dtype_code(d.dtype);
        desc.ndim = dims.len() as i32;
        for (j, (size, st)) in dims.iter().enumerate() {
            desc.sizes[j] = *size as i64;
            desc.strides[j] = st[k];
        }
        desc
    }))
}

/// The device memory of `data`'s storage.
///
/// # Panics
///
/// Panics (an internal error) for storage without device memory: the placeholders of the unit
/// tests (`storage::fake_rocm`), which reach kernels without being able to run them.
#[track_caller]
pub(super) fn buffer<'a>(kernel: &str, data: &'a TensorData) -> &'a super::DeviceBuffer {
    match data.storage.try_rocm_buffer() {
        Some(b) => b,
        None => panic!("internal error: {kernel} on ROCm storage without device memory"),
    }
}

pub(super) fn device_index(data: &TensorData) -> u32 {
    match data.device() {
        Device::Rocm(i) => i,
        d => panic!("internal error: a ROCm kernel got a {d} tensor"),
    }
}

/// The address of the first element of a contiguous `data`, and its size in bytes.
#[track_caller]
fn contiguous_bytes(kernel: &str, data: &TensorData) -> (*mut u8, usize) {
    assert!(data.layout.is_contiguous());
    let size = data.dtype.size_in_bytes();
    let b = buffer(kernel, data);
    // SAFETY: the layout lies inside the storage, so its offset is inside the buffer.
    let ptr = unsafe { b.ptr().add(data.layout.offset * size) };
    (ptr, data.layout.num_elements() * size)
}

/// Writes host values into a contiguous ROCm tensor, returning when they are on the device.
///
/// The failures of this and the other functions here are reported outside the closures of
/// `exec`, which don't pass `#[track_caller]` on (review round 124, R149).
#[track_caller]
pub(crate) fn write_host<T: Element>(dst: &TensorData, values: &[T]) {
    assert!(dst.layout.is_contiguous() && dst.layout.num_elements() == values.len());
    if values.is_empty() {
        return;
    }
    let (ptr, nbytes) = contiguous_bytes("write_host", dst);
    // SAFETY: `T` has no padding (it is a number or `bool`), so its bytes are initialized.
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), nbytes) };
    let ctx = buffer("write_host", dst).context();
    expect(exec(device_index(dst), || ctx.copy_from_host(ptr, bytes)));
}

/// The raw element bytes of a contiguous ROCm tensor.
#[track_caller]
pub(crate) fn read_bytes(src: &TensorData) -> Vec<u8> {
    let mut out = vec![0u8; src.layout.num_elements() * src.dtype.size_in_bytes()];
    if !out.is_empty() {
        read_into(src, &mut out);
    }
    out
}

/// Copies the bytes of a contiguous, nonempty ROCm tensor into `out`, which has their size.
#[track_caller]
fn read_into(src: &TensorData, out: &mut [u8]) {
    let (ptr, nbytes) = contiguous_bytes("read_host", src);
    assert_eq!(nbytes, out.len());
    let ctx = buffer("read_host", src).context();
    expect(exec(device_index(src), || ctx.copy_to_host(out, ptr)));
}

/// Every element in row-major order, converted to `T` on the host as the CPU's `read_host`
/// converts them.
#[track_caller]
pub(crate) fn read_host<T: Element>(src: &TensorData) -> Vec<T> {
    let n = src.layout.num_elements();
    if n == 0 {
        return vec![];
    }
    let contiguous;
    let src = if src.layout.is_contiguous() {
        src
    } else {
        contiguous = crate::ops::clone_data(src);
        &contiguous
    };
    let bytes = read_bytes(src);
    dispatch_all!(src.dtype, S => {
        let size = std::mem::size_of::<S>();
        (0..n)
            .map(|i| {
                // SAFETY: `bytes` holds `n` elements of `S` read from a tensor of that dtype;
                // `Bool` bytes are 0 or 1 (the invariant of `Bool` data).
                let v: S = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(i * size).cast()) };
                v.to_element::<T>()
            })
            .collect()
    })
}

/// Copies `src` into `dst` on the same device, broadcasting and converting the dtype as the CPU's
/// `copy` does, or returns why the device didn't queue the copy (nothing was written then). The
/// caller holds the device's execution lock or doesn't need it.
#[track_caller]
pub(crate) fn try_copy(dst: &TensorData, src: &TensorData) -> Result<(), Failure> {
    assert_eq!(
        dst.device(),
        src.device(),
        "internal error: copy between devices"
    );
    let src = src.expand_to(&dst.layout.shape);
    let Some([d, s]) = descs("copy", [dst, &src])? else {
        return Ok(());
    };
    let index = device_index(dst);
    let ctx = buffer("copy", dst).context();
    let flat = |x: &ffi::hmr_tensor_desc| x.ndim == 0 || (x.ndim == 1 && x.strides[0] == 1);
    exec(index, || {
        if dst.dtype == src.dtype && flat(&d) && flat(&s) {
            let size = dst.dtype.size_in_bytes();
            let n = dst.layout.num_elements() * size;
            // SAFETY: both offsets are inside their buffers (the layouts lie inside them).
            let (to, from) = unsafe {
                (
                    d.data.cast::<u8>().add(d.offset as usize * size),
                    s.data.cast::<u8>().add(s.offset as usize * size),
                )
            };
            ctx.copy_device(to, from, n)
        } else {
            ctx.copy(&d, &s)
        }
    })
}

/// `value` converted to `dtype` as the CPU's `fill` converts it, its bytes in the low bytes of
/// the result (as `hmr_fill` and `hmr_binary_constant` read it).
#[track_caller]
pub(crate) fn scalar_bits(value: Scalar, dtype: DType) -> u64 {
    dispatch_all!(dtype, T => {
        let v = T::from_scalar(value);
        let mut bits = [0u8; 8];
        // SAFETY: `T` is at most 8 bytes and has no padding.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&v as *const T).cast::<u8>(),
                bits.as_mut_ptr(),
                std::mem::size_of::<T>(),
            )
        };
        u64::from_le_bytes(bits)
    })
}

/// Sets every element of `dst` to `value`, converted as the CPU's `fill` converts it, or returns
/// why the device didn't queue the fill (nothing was written then).
#[track_caller]
pub(crate) fn try_fill(dst: &TensorData, value: Scalar) -> Result<(), Failure> {
    let Some([d]) = descs("fill", [dst])? else {
        return Ok(());
    };
    let bits = scalar_bits(value, dst.dtype);
    let ctx = buffer("fill", dst).context();
    exec(device_index(dst), || ctx.fill(&d, bits))
}

/// Copies the contiguous CPU tensor `src` into the contiguous ROCm tensor `dst` of the same
/// dtype and number of elements, holding the CPU's execution lock and then the device's.
#[track_caller]
pub(crate) fn upload(dst: &TensorData, src: &TensorData) -> Result<(), super::Failure> {
    assert!(src.layout.is_contiguous() && src.device() == Device::Cpu);
    assert_eq!(
        (src.dtype, src.layout.num_elements()),
        (dst.dtype, dst.layout.num_elements())
    );
    if src.layout.num_elements() == 0 {
        return Ok(());
    }
    let (ptr, nbytes) = contiguous_bytes("upload", dst);
    let ctx: &Context = buffer("upload", dst).context();
    crate::backend::cpu::exec(|| {
        // SAFETY: `src` is contiguous with `nbytes` bytes from its offset, read inside the CPU
        // lock.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                src.storage
                    .cpu_ptr()
                    .add(src.layout.offset * src.dtype.size_in_bytes()),
                nbytes,
            )
        };
        exec(device_index(dst), || ctx.copy_from_host(ptr, bytes))
    })
}

/// Copies the contiguous ROCm tensor `src` into the contiguous CPU tensor `dst` of the same
/// dtype and number of elements, holding the CPU's execution lock and then the device's.
#[track_caller]
pub(crate) fn download(dst: &TensorData, src: &TensorData) -> Result<(), super::Failure> {
    assert!(dst.layout.is_contiguous() && dst.device() == Device::Cpu);
    assert_eq!(
        (src.dtype, src.layout.num_elements()),
        (dst.dtype, dst.layout.num_elements())
    );
    if src.layout.num_elements() == 0 {
        return Ok(());
    }
    let (ptr, nbytes) = contiguous_bytes("download", src);
    let ctx: &Context = buffer("download", src).context();
    crate::backend::cpu::exec(|| {
        // SAFETY: `dst` is contiguous with `nbytes` bytes from its offset, written inside the CPU
        // lock; nothing else holds a reference to them.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                dst.storage
                    .cpu_ptr()
                    .add(dst.layout.offset * dst.dtype.size_in_bytes()),
                nbytes,
            )
        };
        exec(device_index(src), || ctx.copy_to_host(bytes, ptr))
    })
}
