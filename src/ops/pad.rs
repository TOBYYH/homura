//! Constant padding.

use super::clone_data;
use crate::autograd::formulas;
use crate::backend::cpu;
use crate::tensor::TensorData;
use crate::{Device, Scalar, Shape, Tensor};

/// `input` padded with `value` (PyTorch's `constant_pad_nd`, which `F.pad` calls in its default
/// mode). `pad` holds pairs (before, after) for the last dimensions, starting with the last one;
/// negative values crop. Both are checked like PyTorch, in its order: the length of `pad`, then
/// the crops dimension by dimension, before and after, from the first padded dimension.
///
/// The result is always new storage. With a positive pad it is filled with `value`, which is
/// converted like `fill_` (also when there are no elements), and the cropped input is copied
/// into it; otherwise it is a contiguous copy of the cropped input and `value` isn't used.
///
/// # Panics
///
/// Panics with PyTorch's messages if `pad` has odd length or more than two entries per
/// dimension, if a crop would leave a negative size, or if `value` overflows the dtype.
pub(crate) fn constant_pad_nd(input: &Tensor, pad: &[i64], value: Scalar) -> Tensor {
    let src = input.data();
    let shape = &src.layout.shape;
    let nd = shape.len();
    if !pad.len().is_multiple_of(2) {
        panic!("Padding length must be divisible by 2");
    }
    if pad.len() > 2 * nd {
        panic!(
            "Padding length should be less than or equal to two times the input dimension but \
             got padding length {} and input of dimension {nd}",
            pad.len()
        );
    }
    // Where the cropped input starts in `input` and in the result along each dimension, and its
    // size.
    let mut crop_start = vec![0; nd];
    let mut out_start = vec![0; nd];
    let mut crop_shape = shape.clone();
    let mut out_shape = shape.clone();
    let mut fills = false;
    for d in nd - pad.len() / 2..nd {
        let k = 2 * (nd - 1 - d);
        let (before, after) = (pad[k], pad[k + 1]);
        let mut len = shape[d] as i128;
        for p in [before, after] {
            if p < 0 {
                len += p as i128;
                if len < 0 {
                    panic!("narrow(): length must be non-negative.");
                }
            }
        }
        fills |= before > 0 || after > 0;
        crop_start[d] = (-before).max(0) as usize;
        out_start[d] = before.max(0) as usize;
        crop_shape[d] = len as usize;
        let padded = len + before.max(0) as i128 + after.max(0) as i128;
        out_shape[d] = usize::try_from(padded).unwrap_or_else(|_| {
            panic!("constant_pad_nd(): the padded size {padded} of dimension {d} is too large")
        });
    }
    let (dtype, device) = (src.dtype, src.device());
    // The cropped input, if it has elements: an empty layout may have arbitrary strides.
    let cropped = (crop_shape.num_elements() > 0).then(|| {
        let mut layout = src.layout.clone();
        for d in 0..nd {
            layout = layout.slice(d, crop_start[d], crop_shape[d], 1);
        }
        TensorData {
            layout,
            ..src.clone()
        }
    });
    let out = if !fills {
        match &cropped {
            Some(c) => Tensor::from_data(clone_data(c)),
            None => Tensor::empty_on(crop_shape, dtype, device),
        }
    } else {
        // Checks `value` before anything is written, also on an empty result.
        let out = crate::full_with(Shape::new(&out_shape), value, (dtype, device));
        if let (Some(c), Device::Cpu) = (&cropped, device) {
            let mut dst = out.data();
            for d in 0..nd {
                dst.layout = dst.layout.slice(d, out_start[d], crop_shape[d], 1);
            }
            cpu::copy(&dst, c);
        }
        out
    };
    formulas::constant_pad_nd(input, pad.to_vec(), &out);
    out
}
