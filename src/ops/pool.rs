//! Pooling: the operations behind `nn::functional::max_pool2d` and `adaptive_avg_pool2d`, and
//! their gradients, each recorded as a node with PyTorch's name. They take checked arguments;
//! the functions in `nn::functional` check them.

use crate::autograd::formulas::pool as formulas;
use crate::backend::cpu;
use crate::{DType, Device, Shape, Tensor};

/// The geometry of a 2-D pooling: kernel size, stride, padding and dilation for the height and
/// the width, and the output size they give.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pool2d {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub output: [usize; 2],
}

impl Pool2d {
    /// The input positions `first..end`, every `dilation`, that output position `o` of dimension
    /// `d` (0 for the height, 1 for the width) covers in an input dimension of `n`, the first one
    /// advanced past the padding by whole dilation steps like PyTorch's kernel. Empty
    /// (`first >= end`) for a window that covers only padding.
    pub fn window(&self, d: usize, o: usize, n: usize) -> (usize, usize) {
        let dilation = self.dilation[d] as i64;
        let start = (o * self.stride[d]) as i64 - self.padding[d] as i64;
        let end = (start + (self.kernel[d] as i64 - 1) * dilation + 1).min(n as i64);
        let first = if start < 0 {
            start + (-start + dilation - 1) / dilation * dilation
        } else {
            start
        };
        (first as usize, end.max(first) as usize)
    }

    /// The shape of the output for an input of `shape`.
    fn output_shape(&self, shape: &Shape) -> Shape {
        let nd = shape.len();
        shape[..nd - 2].iter().copied().chain(self.output).collect()
    }
}

/// The windows of an adaptive pooling of a dimension of `input` positions into `output`:
/// window `i` is `[floor(i * input / output), ceil((i + 1) * input / output))`, PyTorch's
/// `start_index` and `end_index`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AdaptiveWindows {
    input: usize,
    output: usize,
}

impl AdaptiveWindows {
    pub fn new(input: usize, output: usize) -> AdaptiveWindows {
        AdaptiveWindows { input, output }
    }

    /// The input positions `start..end` of window `i`.
    pub fn window(&self, i: usize) -> (usize, usize) {
        let (a, b, c) = (i, self.output, self.input);
        ((a / b) * c + ((a % b) * c) / b, 1 + ((a + 1) * c - 1) / b)
    }
}

/// The maxima of the windows of `input` and their positions in the plane (`I64`), PyTorch's
/// `max_pool2d_with_indices`; see `cpu::max_pool2d`. The values are recorded as
/// `MaxPool2DWithIndicesBackward0`, which saves `input` and the indices; the indices have no
/// history.
#[track_caller]
pub(crate) fn max_pool2d_with_indices(input: &Tensor, g: Pool2d) -> (Tensor, Tensor) {
    let x = input.data();
    let shape = g.output_shape(&x.layout.shape);
    let out = Tensor::empty_on(shape.clone(), x.dtype, x.device());
    let indices = Tensor::empty_on(shape, DType::I64, x.device());
    if x.device() == Device::Cpu {
        cpu::max_pool2d(&out.data(), &indices.data(), &x, &g);
    }
    formulas::max_pool2d(input, &indices, &out);
    (out, indices)
}

/// The gradient of [`max_pool2d_with_indices`] for the upstream gradient `grad`, PyTorch's
/// `max_pool2d_with_indices_backward`: a tensor of the shape of `input` that adds each element
/// of `grad` at its index. Recorded as `MaxPool2DWithIndicesBackwardBackward0`.
pub(crate) fn max_pool2d_backward(grad: &Tensor, input: &Tensor, indices: &Tensor) -> Tensor {
    let g = grad.data();
    let out = Tensor::empty_on(input.shape(), g.dtype, g.device());
    if g.device() == Device::Cpu {
        cpu::max_pool2d_backward(&out.data(), &g, &indices.data());
    }
    formulas::max_pool2d_backward(grad, input, indices, &out);
    out
}

/// The means of the adaptive windows of `input` for an output of `output` (PyTorch's
/// `_adaptive_avg_pool2d`); see `cpu::adaptive_avg_pool2d`. Recorded as
/// `AdaptiveAvgPool2DBackward0`, which saves `input`.
#[track_caller]
pub(crate) fn adaptive_avg_pool2d(input: &Tensor, output: [usize; 2]) -> Tensor {
    let x = input.data();
    let nd = x.layout.shape.len();
    let shape: Shape = x.layout.shape[..nd - 2]
        .iter()
        .copied()
        .chain(output)
        .collect();
    let out = Tensor::empty_on(shape, x.dtype, x.device());
    if x.device() == Device::Cpu {
        cpu::adaptive_avg_pool2d(&out.data(), &x);
    }
    formulas::adaptive_avg_pool2d(input, &out);
    out
}

/// The gradient of [`adaptive_avg_pool2d`] for the upstream gradient `grad`, PyTorch's
/// `_adaptive_avg_pool2d_backward`: a tensor of the shape of `input`. Recorded as
/// `AdaptiveAvgPool2DBackwardBackward0`.
pub(crate) fn adaptive_avg_pool2d_backward(grad: &Tensor, input: &Tensor) -> Tensor {
    let g = grad.data();
    let out = Tensor::empty_on(input.shape(), g.dtype, g.device());
    if g.device() == Device::Cpu {
        cpu::adaptive_avg_pool2d_backward(&out.data(), &g);
    }
    formulas::adaptive_avg_pool2d_backward(grad, input, &out);
    out
}
