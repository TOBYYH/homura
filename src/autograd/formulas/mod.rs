//! The backward formulas of the operations, grouped like `src/ops`.
//!
//! Each formula is a [`Node`] written with public `Tensor` operations, so that a backward pass
//! with `create_graph` records it in turn and higher-order derivatives come for free. The
//! functions here are the recording side: the operations call them after computing their
//! result, and they decide what to save. Operations whose formula isn't written yet record
//! [`NotImplemented`] through `record_unimplemented`.

mod addc;
mod binary;
mod clamp;
mod index;
mod lerp;
mod misc;
mod pow;
mod reduce;
mod unary;
mod view;

pub(crate) use addc::{addcdiv, addcmul};
pub(crate) use binary::{binary, where_};
pub(crate) use clamp::clamp;
pub(crate) use index::{
    gather, index, index_add, index_select, scatter_add, select_scatter, slice_scatter,
};
pub(crate) use lerp::lerp;
pub(crate) use misc::{deep_clone, to_copy};
pub(crate) use reduce::{extremum, extremum_dim, mean, prod, sum, sum_dim, sum_to_size, var_std};
pub(crate) use unary::unary;
pub(crate) use view::{as_strided, expand, select, slice, unsafe_view, view};

use super::node::{ApplyContext, Node};
use crate::Tensor;

/// Stands in for the formula of a differentiable operation whose derivative isn't implemented
/// yet. Recording it keeps the outputs requiring grad; running it fails with PyTorch's message
/// for missing derivatives instead of dropping the gradients. A backward pass doesn't run it
/// when no gradient reaches it or `autograd::grad` doesn't need what lies behind it.
pub(crate) struct NotImplemented {
    pub op: &'static str,
}

impl Node for NotImplemented {
    fn name(&self) -> &'static str {
        "NotImplemented"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, _grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        panic!("the derivative for '{}' is not implemented.", self.op)
    }
}

/// The gradient of the only output of an operation. Nodes run only with at least one gradient.
fn only(grads: Vec<Option<Tensor>>) -> Tensor {
    let [grad] = <[Option<Tensor>; 1]>::try_from(grads).expect("one gradient per output");
    grad.expect("nodes run with at least one gradient")
}
