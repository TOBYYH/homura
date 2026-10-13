//! Views and reshapes.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::tensor::Layout;
use crate::{DType, Device, Shape, Tensor};

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

/// Records the backward node of `out = x.permute(dims)`, with `dims` already wrapped.
pub(crate) fn permute(x: &Tensor, dims: Vec<usize>, out: &Tensor) {
    record(&[Some(x)], &[out], |_| PermuteBackward0 { dims });
}

/// Records the backward node of `out = x.transpose(dim0, dim1)`.
pub(crate) fn transpose(x: &Tensor, dim0: i64, dim1: i64, out: &Tensor) {
    record(&[Some(x)], &[out], |_| TransposeBackward0 { dim0, dim1 });
}

/// Which squeeze made a view, for the name of its node: like PyTorch's overloads, `squeeze()`,
/// `squeeze(dim)` and `squeeze(dims)` record `SqueezeBackward0`, `1` and `2`.
#[derive(Clone, Copy)]
pub(crate) enum Squeeze {
    All,
    Dim,
    Dims,
}

/// Records the backward node of a squeeze of `x` that removed the dimensions `removed`, in
/// increasing order.
pub(crate) fn squeeze(x: &Tensor, kind: Squeeze, removed: Vec<usize>, out: &Tensor) {
    record(&[Some(x)], &[out], |_| SqueezeBackward { kind, removed });
}

/// Records the backward node of `out = x.unsqueeze(dim)`, with `dim` already wrapped.
pub(crate) fn unsqueeze(x: &Tensor, dim: usize, out: &Tensor) {
    record(&[Some(x)], &[out], |_| UnsqueezeBackward0 { dim });
}

/// Records the backward node of `out`, a view of all of `x` (PyTorch's `alias`), which indexing
/// returns when no index item changes anything.
pub(crate) fn alias(x: &Tensor, out: &Tensor) {
    record(&[Some(x)], &[out], |_| AliasBackward0);
}

/// Records the backward node of `out = x.select(dim, index)`, with `index` already wrapped.
pub(crate) fn select(x: &Tensor, dim: usize, index: usize, out: &Tensor) {
    record(&[Some(x)], &[out], |_| SelectBackward0 {
        self_shape: x.shape(),
        dim,
        index,
    });
}

/// Records the backward node of `out`, the positions `start..end` with `step` of `x` along
/// `dim`, which slicing and `narrow` view. `start` and `end` are already clamped into the
/// dimension.
pub(crate) fn slice(x: &Tensor, dim: usize, start: usize, end: usize, step: usize, out: &Tensor) {
    record(&[Some(x)], &[out], |_| SliceBackward0 {
        self_shape: x.shape(),
        dim,
        start,
        end,
        step,
    });
}

/// Records the backward node of `out = x.as_strided(..)`.
pub(crate) fn as_strided(x: &Tensor, out: &Tensor) {
    record(&[Some(x)], &[out], |_| {
        let input = x.layout();
        AsStridedBackward0 {
            input_overlaps: input.may_overlap_internally(),
            input,
            output: out.layout(),
        }
    });
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

/// `x.permute(dims)`: the gradient permuted back with the inverse permutation.
struct PermuteBackward0 {
    dims: Vec<usize>,
}

impl Node for PermuteBackward0 {
    fn name(&self) -> &'static str {
        "PermuteBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let mut inverse = vec![0; self.dims.len()];
        for (i, &d) in self.dims.iter().enumerate() {
            inverse[d] = i as i64;
        }
        vec![Some(only(grads).permute(inverse))]
    }
}

/// `x.transpose(dim0, dim1)`: the gradient with the same dimensions swapped back.
struct TransposeBackward0 {
    dim0: i64,
    dim1: i64,
}

impl Node for TransposeBackward0 {
    fn name(&self) -> &'static str {
        "TransposeBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).transpose(self.dim0, self.dim1))]
    }
}

/// A squeeze: the removed dimensions inserted back into the gradient, in increasing order so
/// that each lands at its original position (PyTorch's `unsqueeze_to`).
struct SqueezeBackward {
    kind: Squeeze,
    removed: Vec<usize>,
}

impl Node for SqueezeBackward {
    fn name(&self) -> &'static str {
        match self.kind {
            Squeeze::All => "SqueezeBackward0",
            Squeeze::Dim => "SqueezeBackward1",
            Squeeze::Dims => "SqueezeBackward2",
        }
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        vec![Some(
            self.removed.iter().fold(g, |g, &d| g.unsqueeze(d as i64)),
        )]
    }
}

/// `x.unsqueeze(dim)`: the gradient with that dimension removed again.
struct UnsqueezeBackward0 {
    dim: usize,
}

impl Node for UnsqueezeBackward0 {
    fn name(&self) -> &'static str {
        "UnsqueezeBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).squeeze_dim(self.dim as i64))]
    }
}

/// A view of all of `x`: the gradient passes through unchanged.
struct AliasBackward0;

impl Node for AliasBackward0 {
    fn name(&self) -> &'static str {
        "AliasBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads))]
    }
}

/// `x.select(dim, index)`: the gradient written into that slice of zeros of the shape of `x`.
struct SelectBackward0 {
    self_shape: Shape,
    dim: usize,
    index: usize,
}

impl Node for SelectBackward0 {
    fn name(&self) -> &'static str {
        "SelectBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let zeros = g.new_zeros(&self.self_shape);
        vec![Some(zeros.select_scatter(
            &g,
            self.dim as i64,
            self.index as i64,
        ))]
    }
}

/// A slice of `x` (slicing or `narrow`): the gradient written into that slice of zeros of the
/// shape of `x`.
struct SliceBackward0 {
    self_shape: Shape,
    dim: usize,
    start: usize,
    end: usize,
    step: usize,
}

impl Node for SliceBackward0 {
    fn name(&self) -> &'static str {
        "SliceBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let zeros = g.new_zeros(&self.self_shape);
        vec![Some(zeros.slice_scatter(
            &g,
            self.dim as i64,
            Some(self.start as i64),
            Some(self.end as i64),
            self.step as i64,
        ))]
    }
}

/// `x.as_strided(..)`: each element of the gradient goes to the storage element it views, and
/// `x` collects what reaches its own elements. The gradient is summed by storage offset into a
/// buffer that covers both `x` and the output, which `x`'s layout then reads back. An input with
/// elements that overlap in memory would need its sums divided among them, which isn't supported
/// yet.
struct AsStridedBackward0 {
    input: Layout,
    output: Layout,
    input_overlaps: bool,
}

impl Node for AsStridedBackward0 {
    fn name(&self) -> &'static str {
        "AsStridedBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        if self.input_overlaps {
            panic!(
                "computing the gradient of as_strided for an input whose elements overlap in \
                 memory is not supported yet; deep_clone() the input first"
            );
        }
        let g = only(grads);
        let (Some((in_start, in_end)), Some((out_start, out_end))) =
            (self.input.span(), self.output.span())
        else {
            // Either side is empty: nothing reaches `x`.
            return vec![Some(g.new_zeros(&self.input.shape))];
        };
        let first = in_start.min(out_start);
        let len = in_end.max(out_end) - first;
        let index = storage_positions(&self.output, first, g.device());
        let sums = g.new_zeros([len]).index_add(0, &index, &g.reshape([-1]));
        vec![Some(sums.as_strided(
            &self.input.shape[..],
            &self.input.strides[..],
            in_start - first,
        ))]
    }
}

/// The storage offset of every element of the nonempty layout `l` relative to `first`, as an
/// `I64` vector on `device` in row-major order. Built with tensor operations, so that nothing
/// is enumerated on the meta device.
fn storage_positions(l: &Layout, first: usize, device: Device) -> Tensor {
    let nd = l.shape.len();
    let start = crate::full_with((), (l.offset - first) as i64, (DType::I64, device));
    let positions = (0..nd).fold(start, |positions, d| {
        let mut shape = vec![1i64; nd];
        shape[d] = l.shape[d] as i64;
        let steps = crate::arange_with(l.shape[d] as i64, (DType::I64, device));
        positions + steps.mul(l.strides[d] as i64).reshape(shape)
    });
    positions.reshape([-1])
}
