//! Softmax and log-softmax, each recorded as one node rather than as the operations that compute
//! it, and their first derivatives, recorded as nodes of their own, like PyTorch.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::{DType, Tensor};

/// Records the backward node of `out = x.softmax(dim)`, or `x.log_softmax(dim)` (`log`), with
/// `dim` already wrapped. The derivative is a function of the result, so only the result is
/// saved.
#[track_caller]
pub(crate) fn softmax(x: &Tensor, dim: usize, log: bool, out: &Tensor) {
    record(&[Some(x)], &[out], |b| SoftmaxBackward {
        result: b.save_output(out, 0),
        dim: dim as i64,
        log,
    });
}

/// `x.softmax(dim)` or `x.log_softmax(dim)`: the gradient is [`backward_data`] of `g` and the
/// saved result.
struct SoftmaxBackward {
    result: SavedId,
    dim: i64,
    log: bool,
}

impl Node for SoftmaxBackward {
    fn name(&self) -> &'static str {
        if self.log {
            "LogSoftmaxBackward0"
        } else {
            "SoftmaxBackward0"
        }
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let y = ctx.unpack(self.result);
        vec![Some(backward_data(&g, &y, self.dim, self.log))]
    }
}

/// The gradient of the input of softmax for upstream gradient `g` and result `y` (PyTorch's
/// `_softmax_backward_data`), or of log-softmax (`_log_softmax_backward_data`): softmax
/// `y * (g - sum(g * y))`, log-softmax `g - exp(y) * sum(g)`, with the sums along `dim`, as in
/// PyTorch's backward kernels. A row whose result is NaN (all `-inf`, or containing `inf` or NaN)
/// gets NaN, and so does a row whose sum is infinite or NaN, as in PyTorch.
///
/// Half precision is converted to `f32` like the reference's kernels: the products and the
/// final combination are computed there and rounded to the dtype once, while the sums accumulate
/// in `f64` like every CPU reduction (the reference accumulates in `f32`). Rounding the products
/// and the sum to the dtype first would move the gradients by many units in the last place.
///
/// When `g` or `y` requires grad (a backward pass with `create_graph`), the result is recorded as
/// a node of its own whose derivatives follow the reference's formulas and their order of
/// operations, rather than differentiating the operations above.
#[track_caller]
fn backward_data(g: &Tensor, y: &Tensor, dim: i64, log: bool) -> Tensor {
    let grad = {
        let _no_grad = crate::autograd::no_grad();
        let dtype = y.dtype();
        let (g, y) = if matches!(dtype, DType::F16 | DType::BF16) {
            (g.to(DType::F32), y.to(DType::F32))
        } else {
            (g.clone(), y.clone())
        };
        let grad = if log {
            g.sub(y.exp().mul(g.sum_dim(dim, true)))
        } else {
            y.mul(g.sub(g.mul(&y).sum_dim(dim, true)))
        };
        grad.to(dtype)
    };
    // The reference saves both whichever gradients are needed later.
    record(&[Some(g), Some(y)], &[&grad], |b| BackwardDataBackward {
        grad_output: b.save(g),
        output: b.save(y),
        dim,
        log,
    });
    grad
}

/// [`backward_data`] of the upstream gradient `g` and the result `y`, for upstream gradient `r`,
/// with derivatives.yaml's formulas in their order of operations, computed in the dtype. Softmax:
/// `g` gets `backward_data(r, y)` and `y` gets `softmax_double_backward`,
/// `g * r - sum(y * g) * r - g * sum(y * r)`. Log-softmax: `g` gets `r - sum(r * exp(y))` and
/// `y` gets `-sum(g) * exp(y) * r`. A different order of the same products can overflow to
/// infinity or NaN where these stay finite, and the other way around.
struct BackwardDataBackward {
    grad_output: SavedId,
    output: SavedId,
    dim: i64,
    log: bool,
}

impl Node for BackwardDataBackward {
    fn name(&self) -> &'static str {
        if self.log {
            "LogSoftmaxBackwardDataBackward0"
        } else {
            "SoftmaxBackwardDataBackward0"
        }
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let r = only(grads);
        let g = ctx.unpack(self.grad_output);
        let y = ctx.unpack(self.output);
        let dim = self.dim;
        let (dg, dy) = if self.log {
            let exp = y.exp();
            (
                ctx.needs_input_grad(0)
                    .then(|| r.sub(r.mul(&exp).sum_dim(dim, true))),
                ctx.needs_input_grad(1)
                    .then(|| g.sum_dim(dim, true).neg().mul(&exp).mul(&r)),
            )
        } else {
            (
                ctx.needs_input_grad(0)
                    .then(|| backward_data(&r, &y, dim, false)),
                ctx.needs_input_grad(1).then(|| {
                    let gy = y.mul(&g).sum_dim(dim, true);
                    let ry = y.mul(&r).sum_dim(dim, true);
                    g.mul(&r).sub(gy.mul(&r)).sub(g.mul(ry))
                }),
            )
        };
        vec![dg, dy]
    }
}
