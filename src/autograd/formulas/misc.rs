//! Copies.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::{DType, Device, Tensor};

/// Records the backward node of `out = x.deep_clone()`.
pub(crate) fn deep_clone(x: &Tensor, out: &Tensor) {
    record(&[Some(x)], &[out], |_| CloneBackward0);
}

/// Records the backward node of `out = x.to(..)`, when it computed a new tensor. Conversions to
/// integer dtypes are not differentiable, and `record` skips them.
pub(crate) fn to_copy(x: &Tensor, out: &Tensor) {
    let (self_dtype, self_device) = (x.dtype(), x.device());
    record(&[Some(x)], &[out], |_| ToCopyBackward0 {
        self_dtype,
        self_device,
    });
}

/// `x.deep_clone()`: the gradient passes through unchanged.
struct CloneBackward0;

impl Node for CloneBackward0 {
    fn name(&self) -> &'static str {
        "CloneBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads))]
    }
}

/// `x.to(..)`: the gradient converted back to the dtype and device of `x`. A gradient coming
/// from the meta device can't be converted back, as with forward `to`.
struct ToCopyBackward0 {
    self_dtype: DType,
    self_device: Device,
}

impl Node for ToCopyBackward0 {
    fn name(&self) -> &'static str {
        "ToCopyBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).to((self.self_dtype, self.self_device)))]
    }
}
