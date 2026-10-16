//! The backward formulas of the operations, grouped like `src/ops`.
//!
//! Each formula is a [`Node`] written with public `Tensor` operations, so that a backward pass
//! with `create_graph` records it in turn and higher-order derivatives come for free. The
//! functions here are the recording side: the operations call them after computing their
//! result, and they decide what to save.

mod addc;
mod binary;
mod clamp;
mod index;
mod inplace;
mod lerp;
mod matmul;
mod misc;
mod pow;
mod reduce;
mod shape;
mod softmax;
mod unary;
mod view;

pub(crate) use addc::{addcdiv, addcmul};
pub(crate) use binary::{binary, where_};
pub(crate) use clamp::clamp;
pub(crate) use index::{
    gather, index, index_add, index_select, scatter_add, select_scatter, slice_scatter,
};
pub(crate) use inplace::{copy, overwrite};
pub(crate) use lerp::lerp;
pub(crate) use matmul::{ScaledProduct, matmul, scaled_product};
pub(crate) use misc::{deep_clone, to_copy};
pub(crate) use reduce::{extremum, extremum_dim, mean, prod, sum, sum_dim, sum_to_size, var_std};
pub(crate) use shape::{concat, flip, triangle};
pub(crate) use softmax::softmax;
pub(crate) use unary::unary;
pub(crate) use view::{
    Squeeze, alias, as_strided, expand, permute, select, slice, squeeze, transpose, unsafe_view,
    unsqueeze, view,
};

use crate::Tensor;

/// The gradient of the only output of an operation. Nodes run only with at least one gradient.
fn only(grads: Vec<Option<Tensor>>) -> Tensor {
    let [grad] = <[Option<Tensor>; 1]>::try_from(grads).expect("one gradient per output");
    grad.expect("nodes run with at least one gradient")
}
