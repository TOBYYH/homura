//! Pooling, recorded as one node per operation like PyTorch, and their gradients, recorded as
//! nodes of their own. Like PyTorch's, the nodes save the pooling's input, so that modifying it
//! in place fails the backward pass, and the gradients' nodes give that input a zero gradient.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::{Shape, Tensor};

/// Records the backward node of `out`, the values of `max_pool2d_with_indices(x)` with the
/// positions `indices`.
pub(crate) fn max_pool2d(x: &Tensor, indices: &Tensor, out: &Tensor) {
    record(&[Some(x)], &[out], |b| MaxPool2DWithIndicesBackward0 {
        input: b.save(x),
        indices: b.save(indices),
    });
}

/// Records the backward node of `out = max_pool2d_backward(grad, input, indices)`.
pub(crate) fn max_pool2d_backward(grad: &Tensor, input: &Tensor, indices: &Tensor, out: &Tensor) {
    record(&[Some(grad), Some(input)], &[out], |b| {
        MaxPool2DWithIndicesBackwardBackward0 {
            indices: b.save(indices),
            self_shape: input.shape(),
        }
    });
}

/// Records the backward node of `out = adaptive_avg_pool2d(x, ..)`, the windowed means.
pub(crate) fn adaptive_avg_pool2d(x: &Tensor, out: &Tensor) {
    record(&[Some(x)], &[out], |b| AdaptiveAvgPool2DBackward0 {
        input: b.save(x),
    });
}

/// Records the backward node of `out = adaptive_avg_pool2d_backward(grad, input)`.
pub(crate) fn adaptive_avg_pool2d_backward(grad: &Tensor, input: &Tensor, out: &Tensor) {
    let shape = grad.shape();
    let nd = shape.len();
    record(&[Some(grad), Some(input)], &[out], |_| {
        AdaptiveAvgPool2DBackwardBackward0 {
            output: [shape[nd - 2], shape[nd - 1]],
            self_shape: input.shape(),
        }
    });
}

/// `max_pool2d_with_indices(x)`: each element of the gradient goes to the position it selected,
/// see `ops::max_pool2d_backward`.
struct MaxPool2DWithIndicesBackward0 {
    input: SavedId,
    indices: SavedId,
}

impl Node for MaxPool2DWithIndicesBackward0 {
    fn name(&self) -> &'static str {
        "MaxPool2DWithIndicesBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let (x, indices) = (ctx.unpack(self.input), ctx.unpack(self.indices));
        vec![Some(crate::ops::max_pool2d_backward(
            &only(grads),
            &x,
            &indices,
        ))]
    }
}

/// `max_pool2d_backward(grad, input, indices)`: linear in `grad`, whose gradient picks the
/// upstream gradient at the indices (PyTorch's `max_pool_double_backward`); the input gets
/// zeros.
struct MaxPool2DWithIndicesBackwardBackward0 {
    indices: SavedId,
    self_shape: Shape,
}

impl Node for MaxPool2DWithIndicesBackwardBackward0 {
    fn name(&self) -> &'static str {
        "MaxPool2DWithIndicesBackwardBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let r = only(grads);
        // Unpacked whichever gradients are needed, so that modifying the indices in place fails
        // like in PyTorch, also for the input's.
        let indices = ctx.unpack(self.indices);
        let dgrad = ctx.needs_input_grad(0).then(|| {
            let shape = indices.shape();
            if shape.num_elements() == 0 {
                return r.new_zeros(&shape);
            }
            let nd = shape.len();
            let rows: Vec<i64> = shape[..nd - 2]
                .iter()
                .map(|&n| n as i64)
                .chain([-1])
                .collect();
            r.reshape(rows.clone())
                .gather(-1, &indices.view(rows))
                .view(&shape)
        });
        let dinput = ctx
            .needs_input_grad(1)
            .then(|| r.new_zeros(&self.self_shape));
        vec![dgrad, dinput]
    }
}

/// `adaptive_avg_pool2d(x, ..)`: each element of the gradient is spread evenly over its window,
/// see `ops::adaptive_avg_pool2d_backward`.
struct AdaptiveAvgPool2DBackward0 {
    input: SavedId,
}

impl Node for AdaptiveAvgPool2DBackward0 {
    fn name(&self) -> &'static str {
        "AdaptiveAvgPool2DBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let x = ctx.unpack(self.input);
        vec![Some(crate::ops::adaptive_avg_pool2d_backward(
            &only(grads),
            &x,
        ))]
    }
}

/// `adaptive_avg_pool2d_backward(grad, input)`: linear in `grad`, whose gradient pools the
/// upstream gradient to the size of `grad` again; the input gets zeros.
struct AdaptiveAvgPool2DBackwardBackward0 {
    output: [usize; 2],
    self_shape: Shape,
}

impl Node for AdaptiveAvgPool2DBackwardBackward0 {
    fn name(&self) -> &'static str {
        "AdaptiveAvgPool2DBackwardBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let r = only(grads);
        let dgrad = ctx
            .needs_input_grad(0)
            .then(|| crate::ops::adaptive_avg_pool2d(&r, self.output));
        let dinput = ctx
            .needs_input_grad(1)
            .then(|| r.new_zeros(&self.self_shape));
        vec![dgrad, dinput]
    }
}
