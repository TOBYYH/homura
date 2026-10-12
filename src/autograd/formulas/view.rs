//! Views and reshapes.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::{Shape, Tensor};

/// Records the backward node of `out = x.view(..)`, also used by `reshape` when it can share
/// the storage.
pub(crate) fn view(x: &Tensor, out: &Tensor) {
    let self_shape = x.shape();
    record(&[Some(x)], &[out], |_| ViewBackward0 { self_shape });
}

/// Records the backward node of `out = x.unsafe_view(..)`, the reshape of a result an operation
/// has just allocated: a differentiable reshape whose output is not recorded as a view.
pub(crate) fn unsafe_view(x: &Tensor, out: &Tensor) {
    let self_shape = x.shape();
    record(&[Some(x)], &[out], |_| UnsafeViewBackward0 { self_shape });
}

/// Records the backward node of `out = x.expand(..)`.
pub(crate) fn expand(x: &Tensor, out: &Tensor) {
    let self_shape = x.shape();
    record(&[Some(x)], &[out], |_| ExpandBackward0 { self_shape });
}

/// `x.view(shape)`: the gradient reshaped back to the shape of `x`.
struct ViewBackward0 {
    self_shape: Shape,
}

impl Node for ViewBackward0 {
    fn name(&self) -> &'static str {
        "ViewBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).reshape(&self.self_shape))]
    }
}

/// `x.unsafe_view(shape)`: the gradient reshaped back to the shape of `x`.
struct UnsafeViewBackward0 {
    self_shape: Shape,
}

impl Node for UnsafeViewBackward0 {
    fn name(&self) -> &'static str {
        "UnsafeViewBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).reshape(&self.self_shape))]
    }
}

/// `x.expand(shape)`: the gradient summed back to the shape of `x`.
struct ExpandBackward0 {
    self_shape: Shape,
}

impl Node for ExpandBackward0 {
    fn name(&self) -> &'static str {
        "ExpandBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).sum_to_size(&self.self_shape))]
    }
}
