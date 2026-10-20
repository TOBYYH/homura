//! Operations that rearrange or combine tensors into a new one. Their gradients rearrange the
//! upstream gradient the same way back, so they save no tensors.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::{Scalar, Tensor};

/// Records the backward node of `out = concat(inputs, dim)`, with `dim` already wrapped.
/// `lengths` holds the size of each input along `dim`, `None` for a one-dimensional empty input
/// that was skipped.
pub(crate) fn concat(
    inputs: &[Option<&Tensor>],
    dim: usize,
    lengths: Vec<Option<usize>>,
    out: &Tensor,
) {
    record(inputs, &[out], |_| CatBackward0 { dim, lengths });
}

/// Records the backward node of `out = x.flip(dims)`, with `dims` already wrapped.
pub(crate) fn flip(x: &Tensor, dims: Vec<i64>, out: &Tensor) {
    record(&[Some(x)], &[out], |_| FlipBackward0 { dims });
}

/// Records the backward node of `out = x.triu(diagonal)` (`upper`) or `x.tril(diagonal)`.
pub(crate) fn triangle(x: &Tensor, diagonal: i64, upper: bool, out: &Tensor) {
    record(&[Some(x)], &[out], |_| TriangleBackward { diagonal, upper });
}

/// Records the backward node of `out = constant_pad_nd(x, pad, value)`.
pub(crate) fn constant_pad_nd(x: &Tensor, pad: Vec<i64>, out: &Tensor) {
    record(&[Some(x)], &[out], |_| ConstantPadNdBackward0 { pad });
}

/// Records the backward node of `out = unfold_backward(grad, .., dim, size, step)`, the gradient
/// of an unfold, with `dim` already wrapped.
pub(crate) fn unfold_backward(grad: &Tensor, dim: usize, size: usize, step: usize, out: &Tensor) {
    record(&[Some(grad)], &[out], |_| UnfoldBackwardBackward0 {
        dim,
        size,
        step,
    });
}

/// `concat(inputs, dim)`: each input gets its part of the gradient, narrowed along `dim`, and a
/// skipped empty input an empty gradient (PyTorch's `cat_tensors_backward`). The engine converts
/// each part to its input's dtype.
struct CatBackward0 {
    dim: usize,
    lengths: Vec<Option<usize>>,
}

impl Node for CatBackward0 {
    fn name(&self) -> &'static str {
        "CatBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let mut start = 0;
        let mut result = Vec::with_capacity(self.lengths.len());
        for (i, &length) in self.lengths.iter().enumerate() {
            let Some(length) = length else {
                result.push(ctx.needs_input_grad(i).then(|| g.new_zeros([0])));
                continue;
            };
            result.push(
                ctx.needs_input_grad(i)
                    .then(|| g.narrow(self.dim as i64, start as i64, length)),
            );
            start += length;
        }
        result
    }
}

/// `x.flip(dims)`: the gradient flipped back along the same dimensions.
struct FlipBackward0 {
    dims: Vec<i64>,
}

impl Node for FlipBackward0 {
    fn name(&self) -> &'static str {
        "FlipBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).flip(self.dims.clone()))]
    }
}

/// `x.triu(diagonal)` or `x.tril(diagonal)`: the same triangle of the gradient. It selects
/// rather than multiplies, so non-finite gradients outside the triangle give zeros.
struct TriangleBackward {
    diagonal: i64,
    upper: bool,
}

impl Node for TriangleBackward {
    fn name(&self) -> &'static str {
        if self.upper {
            "TriuBackward0"
        } else {
            "TrilBackward0"
        }
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        vec![Some(if self.upper {
            g.triu(self.diagonal)
        } else {
            g.tril(self.diagonal)
        })]
    }
}

/// `constant_pad_nd(x, pad, value)`: the gradient padded with the negated `pad` (PyTorch's
/// `constant_pad_nd_backward`), which crops what was padded and pads what was cropped with zeros.
/// It is recorded like the forward, so its own derivative is a padding again.
struct ConstantPadNdBackward0 {
    pad: Vec<i64>,
}

impl Node for ConstantPadNdBackward0 {
    fn name(&self) -> &'static str {
        "ConstantPadNdBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        // The forward succeeded, so no entry is `i64::MIN`.
        let negated: Vec<i64> = self.pad.iter().map(|&p| -p).collect();
        vec![Some(crate::ops::constant_pad_nd(
            &only(grads),
            &negated,
            Scalar::Int(0),
        ))]
    }
}

/// `unfold_backward(grad, .., dim, size, step)`: linear in `grad`, so the gradient of `grad` is
/// the upstream gradient unfolded the same way.
struct UnfoldBackwardBackward0 {
    dim: usize,
    size: usize,
    step: usize,
}

impl Node for UnfoldBackwardBackward0 {
    fn name(&self) -> &'static str {
        "UnfoldBackwardBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).unfold(
            self.dim as i64,
            self.size,
            self.step,
        ))]
    }
}
