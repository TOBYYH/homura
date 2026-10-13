//! Operations that rearrange or combine tensors into a new one. Their gradients rearrange the
//! upstream gradient the same way back, so they save no tensors.

use super::only;
use crate::Tensor;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;

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
