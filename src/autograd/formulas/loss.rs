//! The losses whose reference is a single node. Their results are computed by the loss
//! functions in `nn::functional`; the nodes compute PyTorch's gradient formulas in its order of
//! operations.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::nn::functional::Reduction;
use crate::{DType, Tensor, where_};

/// Records `out = mse_loss(input, target)` for an `input` and `target` of the same shape.
#[track_caller]
pub(crate) fn mse_loss(input: &Tensor, target: &Tensor, reduction: Reduction, out: &Tensor) {
    record(&[Some(input), Some(target)], &[out], |b| MseLossBackward0 {
        input: b.save(input),
        target: b.save(target),
        reduction,
    });
}

/// `mse_loss`: PyTorch's `mse_loss_backward`, `norm * (input - target) * g` with `norm` 2, or
/// `2 / n` for the mean of `n` elements; the target's gradient swaps the two.
struct MseLossBackward0 {
    input: SavedId,
    target: SavedId,
    reduction: Reduction,
}

impl Node for MseLossBackward0 {
    fn name(&self) -> &'static str {
        "MseLossBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let (input, target) = (ctx.unpack(self.input), ctx.unpack(self.target));
        let norm = match self.reduction {
            Reduction::Mean => 2.0 / input.num_elements() as f64,
            _ => 2.0,
        };
        let grad = |a: &Tensor, b: &Tensor| a.sub(b).mul(norm).mul(&g);
        vec![
            ctx.needs_input_grad(0).then(|| grad(&input, &target)),
            ctx.needs_input_grad(1).then(|| grad(&target, &input)),
        ]
    }
}

/// What the gradient of `nll_loss` needs besides the saved tensors.
pub(crate) struct NllLoss {
    pub reduction: Reduction,
    pub ignore_index: i64,
    /// The dimension of the classes, 0 for a 1-D input and 1 otherwise.
    pub class_dim: usize,
    /// Whether the input has more than two dimensions, which PyTorch computes with
    /// `nll_loss2d` (`NllLoss2DBackward0`).
    pub spatial: bool,
}

/// Records `out = nll_loss(input, target)`. `weight` has one element per class, in any shape
/// (saved as the caller's tensor, so that its version is checked), and `total_weight` is the
/// divisor of the mean, both of the input's dtype.
#[track_caller]
pub(crate) fn nll_loss(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    total_weight: &Tensor,
    args: NllLoss,
    out: &Tensor,
) {
    record(&[Some(input)], &[out], |b| NllLossBackward0 {
        input: b.save(input),
        target: b.save(target),
        weight: weight.map(|w| b.save(w)),
        total_weight: b.save(total_weight),
        args,
    });
}

/// `nll_loss`: PyTorch's `nll_loss_backward`. Each position's input at its target class gets
/// `-w * g`, with `g / total_weight` for the mean; the other inputs, and all inputs of a position
/// whose target is `ignore_index`, get zero, even where `g` is infinite or NaN. The gradient is
/// linear in `g`, through a scatter into zeros. As in PyTorch, it depends on the input with a zero
/// derivative, instead of not at all.
struct NllLossBackward0 {
    input: SavedId,
    target: SavedId,
    weight: Option<SavedId>,
    total_weight: SavedId,
    args: NllLoss,
}

impl Node for NllLossBackward0 {
    fn name(&self) -> &'static str {
        if self.args.spatial {
            "NllLoss2DBackward0"
        } else {
            "NllLossBackward0"
        }
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let NllLoss {
            reduction,
            ignore_index,
            class_dim,
            ..
        } = self.args;
        // Zeros that depend on the input with a zero derivative, PyTorch's
        // `self: zeros_like(grad)` for `nll_loss_backward`; never NaN, even where the input is.
        let input = ctx.unpack(self.input);
        let never = crate::full_with([], false, input.device());
        let zeros = where_(&never, &input, 0.0);
        let target = ctx.unpack(self.target).to(DType::I64);
        let weight = self.weight.map(|w| ctx.unpack(w));
        // The gradient of each position, before the weight.
        let grad = match reduction {
            Reduction::Mean => g.div(ctx.unpack(self.total_weight)).neg(),
            _ => g.neg(),
        };
        if input.shape()[class_dim] == 0 {
            // No classes: every target is ignored and nothing is scattered. The gradient is still
            // added, expanded to the empty input, so that it depends on `g` like PyTorch's (its
            // derivative with respect to `g` is the loss of `g`: zeros, or NaN for the mean, whose
            // total weight is zero).
            // Unreduced losses of a batch have a gradient per position, without the class
            // dimension; the others are 0-D.
            let grad = if reduction == Reduction::None && class_dim == 1 {
                grad.unsqueeze(1)
            } else {
                grad
            };
            return vec![Some(zeros.add(grad.expand(input.shape())))];
        }
        let valid = target.not_equal(ignore_index);
        let index = where_(&valid, &target, 0i64);
        let grad = match weight {
            Some(w) => w
                .reshape([-1])
                .index_select(0, &index.reshape([-1]))
                .reshape(index.shape())
                .mul(&grad),
            None => grad.expand(index.shape()),
        };
        let grad = where_(&valid, &grad, 0.0);
        let (index, grad) = if class_dim == 0 {
            (index.reshape([1]), grad.reshape([1]))
        } else {
            (index.unsqueeze(1), grad.unsqueeze(1))
        };
        vec![Some(zeros.scatter_add(class_dim as i64, &index, &grad))]
    }
}

/// Records `out = binary_cross_entropy_with_logits(input, target)`; `weight` and `pos_weight`
/// broadcast to the input's shape.
#[track_caller]
pub(crate) fn binary_cross_entropy_with_logits(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    pos_weight: Option<&Tensor>,
    reduction: Reduction,
    out: &Tensor,
) {
    record(&[Some(input), Some(target)], &[out], |b| {
        BinaryCrossEntropyWithLogitsBackward0 {
            input: b.save(input),
            target: b.save(target),
            weight: weight.map(|w| b.save(w)),
            pos_weight: pos_weight.map(|p| b.save(p)),
            reduction,
        }
    });
}

/// `binary_cross_entropy_with_logits`: PyTorch's `binary_cross_entropy_with_logits_backward` and
/// `binary_cross_entropy_with_logits_target_backward`. With `t = pos_weight * target`, the
/// input gets `((t + 1 - target) * σ(input) - t) * g`, or `(σ(input) - target) * g` without
/// `pos_weight`; the target gets `(log σ(-input) - log σ(input) * pos_weight) * g`, or
/// `-input * g`. Both are then multiplied by the weight and, for the mean, divided by the number
/// of elements.
struct BinaryCrossEntropyWithLogitsBackward0 {
    input: SavedId,
    target: SavedId,
    weight: Option<SavedId>,
    pos_weight: Option<SavedId>,
    reduction: Reduction,
}

impl Node for BinaryCrossEntropyWithLogitsBackward0 {
    fn name(&self) -> &'static str {
        "BinaryCrossEntropyWithLogitsBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let (input, target) = (ctx.unpack(self.input), ctx.unpack(self.target));
        let weight = self.weight.map(|w| ctx.unpack(w));
        let pos_weight = self.pos_weight.map(|p| ctx.unpack(p));
        let finish = |grad: Tensor| {
            let grad = match &weight {
                Some(w) => grad.mul(w),
                None => grad,
            };
            match self.reduction {
                Reduction::Mean => grad.div(input.num_elements() as i64),
                _ => grad,
            }
        };
        let grad_input = ctx.needs_input_grad(0).then(|| {
            let grad = match &pos_weight {
                Some(p) => {
                    let t = p.mul(&target);
                    t.add(1.0).sub(&target).mul(input.sigmoid()).sub(&t)
                }
                None => input.sigmoid().sub(&target),
            };
            finish(grad.mul(&g))
        });
        let grad_target = ctx.needs_input_grad(1).then(|| {
            let grad = match &pos_weight {
                Some(p) => input
                    .neg()
                    .log_sigmoid()
                    .sub(input.log_sigmoid().mul(p))
                    .mul(&g),
                None => input.neg().mul(&g),
            };
            finish(grad)
        });
        vec![grad_input, grad_target]
    }
}
