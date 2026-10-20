use crate::{Scalar, Tensor};

/// Pads `input` with `value`, PyTorch's `F.pad` in its default `"constant"` mode. `pad` holds a
/// pair (before, after) for each of the last `pad.len() / 2` dimensions, starting with the last
/// one: `[1, 2]` adds one element before and two after along the last dimension,
/// `[1, 1, 0, 3]` also adds three rows at the end of the dimension before it. Negative values
/// crop instead, and the two may be mixed.
///
/// The result is always a new contiguous tensor. `value` is converted to the dtype like
/// [`fill_`](Tensor::fill_) does (floats are truncated for integer tensors), and only when
/// something is padded: a pad without positive entries never uses it. It is differentiable for
/// floating point inputs, recorded as `ConstantPadNdBackward0` like in PyTorch, whose gradient
/// pads with the negated `pad` and is differentiable in turn.
///
/// ```
/// use homura::nn::functional::pad;
///
/// let x = homura::tensor([[1.0f32, 2.0], [3.0, 4.0]]);
/// let y = pad(&x, &[1, 0, 0, 1], -1.0);
/// assert_eq!(y.shape(), [3, 3]);
/// assert_eq!(y.to_vec::<f32>(), [-1.0, 1.0, 2.0, -1.0, 3.0, 4.0, -1.0, -1.0, -1.0]);
/// assert_eq!(pad(&x, &[-1, 0], 0).to_vec::<f32>(), [2.0, 4.0]);
/// ```
///
/// Unlike PyTorch, a pad without positive entries also gives a contiguous result; PyTorch keeps
/// the strides of a non-contiguous input then. Values and gradients are the same.
///
/// # Panics
///
/// With PyTorch's messages: if `pad` has odd length or more than two entries per dimension of
/// `input`, if cropping would leave a negative size, or if `value` overflows the dtype.
pub fn pad(input: &Tensor, pad: &[i64], value: impl Into<Scalar>) -> Tensor {
    crate::ops::constant_pad_nd(input, pad, value.into())
}
