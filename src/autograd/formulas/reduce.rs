//! Reductions.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::{Shape, Tensor};

/// Records the backward node of `out = x.sum()`.
pub(crate) fn sum(x: &Tensor, out: &Tensor) {
    let self_shape = x.shape();
    record(&[Some(x)], &[out], |_| SumBackward0 { self_shape });
}

/// Records the backward node of `out = x.sum_to_size(..)`, when it computed a new tensor.
pub(crate) fn sum_to_size(x: &Tensor, out: &Tensor) {
    let self_shape = x.shape();
    record(&[Some(x)], &[out], |_| SumToSizeBackward0 { self_shape });
}

/// `x.sum()`: the zero-dimensional gradient expanded to the shape of `x`.
struct SumBackward0 {
    self_shape: Shape,
}

impl Node for SumBackward0 {
    fn name(&self) -> &'static str {
        "SumBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).expand(&self.self_shape))]
    }
}

/// `x.sum_to_size(size)`: the gradient, of shape `size`, expanded back to the shape of `x`.
/// PyTorch records this operation as a `SumBackward1` with `keepdim`, followed by a
/// `ViewBackward0` when leading dimensions are removed; homura records one node.
struct SumToSizeBackward0 {
    self_shape: Shape,
}

impl Node for SumToSizeBackward0 {
    fn name(&self) -> &'static str {
        "SumToSizeBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).expand(&self.self_shape))]
    }
}
