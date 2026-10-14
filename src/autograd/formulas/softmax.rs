//! Softmax and log-softmax, each recorded as one node rather than as the operations that compute
//! it, like PyTorch.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::{DType, Tensor};

/// Records the backward node of `out = x.softmax(dim)`, or `x.log_softmax(dim)` (`log`), with
/// `dim` already wrapped. The derivative is a function of the result, so only the result is
/// saved.
pub(crate) fn softmax(x: &Tensor, dim: usize, log: bool, out: &Tensor) {
    record(&[Some(x)], &[out], |b| SoftmaxBackward {
        result: b.save_output(out, 0),
        dim: dim as i64,
        log,
    });
}

/// Softmax: `y * (g - sum(g * y))`; log-softmax: `g - exp(y) * sum(g)`, with the sums along
/// `dim` and `y` the saved result, as in PyTorch's backward kernels. A row whose result is NaN
/// (all `-inf`, or containing `inf` or NaN) gets NaN, and so does a row whose sum is infinite
/// or NaN, as in PyTorch.
///
/// Half precision is computed in `f32` and rounded once, like the reference's kernels; rounding
/// the products and the sum to the dtype first would move the gradients by many units in the
/// last place. Written with public operations, the formula gives the reference's second
/// derivatives (`_softmax_backward_data` and `softmax_double_backward`) when differentiated.
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
        let dtype = y.dtype();
        let (g, y) = if matches!(dtype, DType::F16 | DType::BF16) {
            (g.to(DType::F32), y.to(DType::F32))
        } else {
            (g, y)
        };
        let grad = if self.log {
            g.sub(y.exp().mul(g.sum_dim(self.dim, true)))
        } else {
            y.mul(g.sub(g.mul(&y).sum_dim(self.dim, true)))
        };
        vec![Some(grad.to(dtype))]
    }
}
