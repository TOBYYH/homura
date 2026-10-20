use crate::{DType, Tensor};

/// The padding of [`conv2d`], PyTorch's `padding` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv2dPadding {
    /// Zeros on both sides of the height and the width (`padding=(h, w)`).
    Explicit([usize; 2]),
    /// No padding (`padding="valid"`).
    Valid,
    /// Enough zeros for an output of the input's size (`padding="same"`), only with a stride of
    /// 1. Each dimension gets `dilation * (kernel - 1)` zeros, half before and the rest after.
    Same,
}

impl From<usize> for Conv2dPadding {
    fn from(p: usize) -> Self {
        Conv2dPadding::Explicit([p, p])
    }
}

impl From<[usize; 2]> for Conv2dPadding {
    fn from(p: [usize; 2]) -> Self {
        Conv2dPadding::Explicit(p)
    }
}

/// The options of [`conv2d`] with PyTorch's defaults; the pairs are for the height and then the
/// width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dOptions {
    /// The step between windows, `[1, 1]` by default.
    pub stride: [usize; 2],
    /// The zero padding, none by default.
    pub padding: Conv2dPadding,
    /// The step between the elements of a window, `[1, 1]` by default.
    pub dilation: [usize; 2],
    /// The number of groups the channels are split into, 1 by default: output channel group `i`
    /// sees only input channel group `i`.
    pub groups: usize,
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Conv2dOptions {
            stride: [1, 1],
            padding: Conv2dPadding::Explicit([0, 0]),
            dilation: [1, 1],
            groups: 1,
        }
    }
}

/// The 2-D convolution (cross-correlation) of `input`, `[N, C, H, W]` or `[C, H, W]`, with
/// `weight`, `[O, C / groups, kh, kw]`, plus `bias`, `[O]`: PyTorch's `F.conv2d`.
///
/// It is computed from differentiable operations, like an `im2col`: the padded input
/// ([`pad`](super::pad)) is unfolded into windows ([`Tensor::unfold`]), whose elements are
/// rearranged into rows and multiplied with the weight ([`Tensor::matmul`]), and the bias is
/// added. So its gradients of any order follow from those operations, and the graph holds their
/// nodes instead of PyTorch's single `ConvolutionBackward0`. `F16` and `BF16` are computed in
/// `F32` and rounded once at the end, like PyTorch, so that a bias can cancel products that
/// would overflow the dtype. The matrix product accumulates in `F64` (see
/// [`Tensor::matmul`]), so `F32` results can differ from PyTorch's in the last bits. Integer
/// convolutions wrap on overflow, like PyTorch's. An unbatched input is unsqueezed and the
/// result squeezed back, as in PyTorch.
///
/// ```
/// use homura::nn::functional::{Conv2dOptions, conv2d};
///
/// let x = homura::ones([1, 1, 3, 3]);
/// let w = homura::ones([1, 1, 2, 2]);
/// let options = Conv2dOptions { padding: 1.into(), ..Default::default() };
/// let y = conv2d(&x, &w, None, options);
/// assert_eq!(y.shape(), [1, 1, 4, 4]);
/// assert_eq!(y.to_vec::<f32>()[..4], [1.0, 2.0, 2.0, 1.0]);
/// ```
///
/// # Panics
///
/// With PyTorch's messages and in its order: if `bias` has another dtype than `input`, `input`
/// isn't 3- or 4-dimensional, `weight` isn't 4-dimensional, `groups` is zero, `Same` padding has
/// a stride other than 1, the stride or the dilation is zero, the shapes don't fit together,
/// the padded input is smaller than the kernel, or the kernel is empty; if `weight` has another
/// dtype; for bool tensors. Unlike PyTorch: for an input without channels (PyTorch returns an
/// output without channels), when the tensors are on different devices, and for `Valid` or
/// `Same` padding with a bias of another dtype.
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    options: Conv2dOptions,
) -> Tensor {
    let Conv2dOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options;
    let mut devices = vec![input.data(), weight.data()];
    devices.extend(bias.map(|b| b.data()));
    crate::ops::common_device(&devices);
    let dtype = input.dtype();
    if let Some(b) = bias
        && b.dtype() != dtype
    {
        panic!(
            "Input type ({dtype}) and bias type ({}) should be the same",
            b.dtype()
        );
    }
    let unbatched = match input.num_dims() {
        3 => true,
        4 => false,
        _ => panic!(
            "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: {:?}",
            &input.shape()[..]
        ),
    };
    let x = if unbatched {
        input.unsqueeze(0)
    } else {
        input.clone()
    };
    let w_shape = weight.shape();
    if w_shape.len() < 3 {
        panic!("weight should have at least three dimensions");
    }
    if groups == 0 {
        panic!("non-positive groups is not supported");
    }
    let x_shape = x.shape();
    if w_shape.len() != 4 {
        if padding == Conv2dPadding::Same {
            panic!(
                "Expected {k}-dimensional input for {k}-dimensional weight{:?}, but got \
                 4-dimensional input of size {:?} instead",
                &w_shape[..],
                &x_shape[..],
                k = w_shape.len()
            );
        }
        panic!(
            "expected stride to be a single integer value or a list of {} values to match the \
             convolution dimensions, but got stride={stride:?}",
            w_shape.len() - 2
        );
    }
    let [out_channels, group_channels, kh, kw] = [w_shape[0], w_shape[1], w_shape[2], w_shape[3]];
    // The zeros before and after the height and the width.
    let pads: [[usize; 2]; 2] = match padding {
        Conv2dPadding::Explicit([h, w]) => [[h, h], [w, w]],
        Conv2dPadding::Valid => [[0, 0], [0, 0]],
        Conv2dPadding::Same => {
            if stride != [1, 1] {
                panic!("padding='same' is not supported for strided convolutions");
            }
            [(kh, dilation[0]), (kw, dilation[1])].map(|(k, d)| {
                let total = d * k.saturating_sub(1);
                [total / 2, total - total / 2]
            })
        }
    };
    if stride.contains(&0) {
        panic!("non-positive stride is not supported");
    }
    if out_channels < groups {
        panic!(
            "Given groups={groups}, expected weight to be at least {groups} at dimension 0, but \
             got weight of size {:?} instead",
            &w_shape[..]
        );
    }
    if !out_channels.is_multiple_of(groups) {
        panic!(
            "Given groups={groups}, expected weight to be divisible by {groups} at dimension 0, \
             but got weight of size [{:?}] instead",
            &w_shape[..]
        );
    }
    let channels = x_shape[1];
    if Some(channels) != group_channels.checked_mul(groups) {
        panic!(
            "Given groups={groups}, weight of size {:?}, expected input{:?} to have {} channels, \
             but got {channels} channels instead",
            &w_shape[..],
            &x_shape[..],
            group_channels * groups
        );
    }
    if let Some(b) = bias
        && b.shape()[..] != [out_channels]
    {
        panic!(
            "Given weight of size {:?}, expected bias to be 1-dimensional with {out_channels} \
             elements, but got bias of size {:?} instead",
            &w_shape[..],
            &b.shape()[..]
        );
    }
    let mut padded = [0; 2];
    for d in 0..2 {
        let half = pads[d][0].min(pads[d][1]);
        if half > i64::MAX as usize / 2 {
            panic!(
                "Given padding={half} at dimension {d} , expected padding to be at most {}",
                i64::MAX / 2
            );
        }
        padded[d] = x_shape[2 + d] + pads[d][0] + pads[d][1];
    }
    // The kernel widened by the dilation; 0 for an empty kernel.
    let effective = [(kh, dilation[0]), (kw, dilation[1])]
        .map(|(k, d)| (d as i64 * (k as i64 - 1) + 1).max(0) as usize);
    if padded[0] < effective[0] || padded[1] < effective[1] {
        panic!(
            "Calculated padded input size per channel: ({} x {}). Kernel size: ({} x {}). \
             Kernel size can't be greater than actual input size",
            padded[0], padded[1], effective[0], effective[1]
        );
    }
    if channels == 0 {
        panic!(
            "conv2d(): the input of size {:?} has no channels, so the output would have no \
             channels either, unlike the weight of size {:?}; PyTorch returns such an output",
            &x_shape[..],
            &w_shape[..]
        );
    }
    if dilation.contains(&0) {
        panic!("dilation should be greater than zero, but got {dilation:?}");
    }
    if kh == 0 || kw == 0 {
        panic!(
            "kernel size should be greater than zero, but got kernel_height: {kh} kernel_width: {kw}"
        );
    }
    if weight.dtype() != dtype {
        panic!("expected scalar type {dtype} but found {}", weight.dtype());
    }
    if dtype == DType::Bool {
        panic!("conv2d is not implemented for bool tensors");
    }
    // Half precision is computed in F32 and rounded once.
    let half = matches!(dtype, DType::F16 | DType::BF16);
    let up = |t: &Tensor| if half { t.to(DType::F32) } else { t.clone() };
    let (x, w, b) = (up(&x), up(weight), bias.map(up));
    let out = convolve(&x, &w, b.as_ref(), stride, pads, dilation, groups);
    let out = if half { out.to(dtype) } else { out };
    if unbatched { out.squeeze_dim(0) } else { out }
}

/// The convolution of the checked 4-D `x` with `w` and `b`, all of one dtype: pad, unfold the
/// windows, take every `dilation`-th element of them, rearrange them into rows of
/// `C / groups * kh * kw` elements per group and multiply them with the weight's rows.
fn convolve(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    stride: [usize; 2],
    pads: [[usize; 2]; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Tensor {
    let ws = w.shape();
    let (out_channels, group_channels, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
    let n = x.size(0);
    let [[top, bottom], [left, right]] = pads;
    let x = if [top, bottom, left, right] == [0; 4] {
        x.clone()
    } else {
        let pad = [left, right, top, bottom].map(|p| p as i64);
        crate::ops::constant_pad_nd(x, &pad, 0.into())
    };
    let windows = x.unfold(2, (kh - 1) * dilation[0] + 1, stride[0]).unfold(
        3,
        (kw - 1) * dilation[1] + 1,
        stride[1],
    );
    let windows = if dilation == [1, 1] {
        windows
    } else {
        windows
            .slice(4, None, None, dilation[0] as i64)
            .slice(5, None, None, dilation[1] as i64)
    };
    // [N, C, OH, OW, kh, kw]
    let ws = windows.shape();
    let (oh, ow) = (ws[2], ws[3]);
    let to_i64 = |dims: &[usize]| dims.iter().map(|&d| d as i64).collect::<Vec<_>>();
    let rows = windows
        .reshape(to_i64(&[n, groups, group_channels, oh, ow, kh, kw]))
        .permute([0, 1, 3, 4, 2, 5, 6])
        .reshape(to_i64(&[n, groups, oh * ow, group_channels * kh * kw]));
    let per_group = out_channels / groups;
    let weights = w
        .reshape(to_i64(&[groups, per_group, group_channels * kh * kw]))
        .transpose(1, 2);
    // [N, groups, OH * OW, O / groups] -> [N, O, OH, OW]
    let out = rows
        .matmul(&weights)
        .permute([0, 1, 3, 2])
        .reshape(to_i64(&[n, out_channels, oh, ow]));
    match b {
        Some(b) => out.add(b.view(to_i64(&[1, out_channels, 1, 1]))),
        None => out,
    }
}
