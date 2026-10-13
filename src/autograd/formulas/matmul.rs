//! Matrix products: `matmul`, and the scaled products `addmm`, `addmv`, `addr`, `baddbmm` and
//! `addbmm`. Each operand is saved for the other one's gradient, as in PyTorch.

use super::only;
use crate::autograd::node::{ApplyContext, Node, NodeBuilder};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::{Scalar, Shape, Tensor};

/// Saves each of two operands for the other one's gradient: the first is needed when the second
/// has an edge (input `first + 1`), and the other way around.
fn cross_save(
    builder: &mut NodeBuilder,
    first: usize,
    a: &Tensor,
    b: &Tensor,
) -> [Option<SavedId>; 2] {
    [
        builder.needs_grad(first + 1).then(|| builder.save(a)),
        builder.needs_grad(first).then(|| builder.save(b)),
    ]
}

/// Records the backward node of `out = a.matmul(b)`.
pub(crate) fn matmul(a: &Tensor, b: &Tensor, out: &Tensor) {
    record(&[Some(a), Some(b)], &[out], |builder| MatmulBackward {
        operands: cross_save(builder, 0, a, b),
        shapes: [a.shape(), b.shape()],
    });
}

/// The last two dimensions swapped.
fn mt(t: &Tensor) -> Tensor {
    t.transpose(-2, -1)
}

/// `Σ_i x_iᵀ @ y_i` over the batch dimensions of `x` (`[batch..., r, p]`) and `y`
/// (`[batch..., r, q]`), which are the same, computed as one product of the flattened operands.
fn folded(x: &Tensor, y: &Tensor) -> Tensor {
    x.flatten(0, -2).transpose(0, 1).matmul(&y.flatten(0, -2))
}

/// `a.matmul(b)`. The gradients are `g @ bᵀ` and `aᵀ @ g` in the full broadcast batch shape,
/// summed to each operand's shape, with 1-D operands as a row (`a`) or a column (`b`) and the
/// dimension they dropped restored in `g`. When only one operand has batch dimensions, the
/// other's gradient is folded into a single product over the batch and the rows instead, like
/// PyTorch's folding into `mm`: the weight `[d, e]` of `[B, S, d] @ [d, e]` gets
/// `x.flatten(0, -2)ᵀ @ g.flatten(0, -2)` without a `[B, d, e]` intermediate.
struct MatmulBackward {
    /// `a`, saved when `b` has an edge, and `b`, saved when `a` has one.
    operands: [Option<SavedId>; 2],
    shapes: [Shape; 2],
}

impl Node for MatmulBackward {
    fn name(&self) -> &'static str {
        // The reference's node where it records a single one, its name for the matmul formula
        // where it records views around `mm`, `mv` or `bmm`.
        match (self.shapes[0].len(), self.shapes[1].len()) {
            (1, 1) => "DotBackward0",
            (2, 1) => "MvBackward0",
            (2, 2) => "MmBackward0",
            _ => "MatmulBackward0",
        }
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let mut g = only(grads);
        let [a, b] = self.operands.map(|id| id.map(|id| ctx.unpack(id)));
        let [sa, sb] = &self.shapes;
        let (da, db) = (sa.len(), sb.len());
        if db == 1 {
            g = g.unsqueeze(-1);
        }
        if da == 1 {
            g = g.unsqueeze(-2);
        }
        let grad_a = ctx.needs_input_grad(0).then(|| {
            let b = b.as_ref().expect("saved for a's gradient");
            let b = if db == 1 { b.unsqueeze(1) } else { b.clone() };
            let grad = if da <= 2 && db > 2 {
                // Σ_i g_i @ b_iᵀ
                folded(&mt(&g), &mt(&b))
            } else {
                let full = g.matmul(&mt(&b));
                if da == 1 {
                    full.sum_to_size([1, sa[0]])
                } else {
                    full.sum_to_size(sa)
                }
            };
            if da == 1 { grad.reshape(sa) } else { grad }
        });
        let grad_b = ctx.needs_input_grad(1).then(|| {
            let a = a.as_ref().expect("saved for b's gradient");
            let a = if da == 1 { a.unsqueeze(0) } else { a.clone() };
            let grad = if db <= 2 && da > 2 {
                folded(&a, &g)
            } else {
                let full = mt(&a).matmul(&g);
                if db == 1 {
                    full.sum_to_size([sb[0], 1])
                } else {
                    full.sum_to_size(sb)
                }
            };
            if db == 1 { grad.reshape(sb) } else { grad }
        });
        vec![grad_a, grad_b]
    }
}

/// The product that a scaled product adds to `beta * input`.
#[derive(Clone, Copy)]
pub(crate) enum ScaledProduct {
    /// `addmm`: `mat1 @ mat2`.
    Mm,
    /// `addmv`: `mat @ vec`.
    Mv,
    /// `addr`: the outer product of `vec1` and `vec2`.
    Outer,
    /// `baddbmm`: `batch1 @ batch2`, batch by batch.
    Bmm,
    /// `addbmm`: the sum of `batch1[i] @ batch2[i]`.
    SumBmm,
}

/// Records the backward node of `out = beta * input + alpha * product(x, y)`, with the inputs as
/// the operation used them (`addr` promotes them first).
pub(crate) fn scaled_product(
    product: ScaledProduct,
    input: &Tensor,
    x: &Tensor,
    y: &Tensor,
    beta: Scalar,
    alpha: Scalar,
    out: &Tensor,
) {
    record(&[Some(input), Some(x), Some(y)], &[out], |builder| {
        ScaledProductBackward {
            product,
            factors: cross_save(builder, 1, x, y),
            beta,
            alpha,
            input_shape: input.shape(),
        }
    });
}

/// `t * s`, or `t` itself when `s` is one: PyTorch's `maybe_multiply`. A floating point scalar
/// stays unrounded in f32 for half precision `t`, as in the reference.
fn scale(t: Tensor, s: Scalar) -> Tensor {
    if s.to_f64() == 1.0 { t } else { t.mul(s) }
}

/// `beta * input + alpha * product(x, y)` (PyTorch's `addmm`, `addmv`, `addr`, `baddbmm` and
/// `addbmm` formulas): the input gets `g * beta`, even when `beta` is zero and the value ignored
/// the input, so it gets zeros where `g` is finite and NaN where it isn't. The factors get the
/// product of `g` with the other factor, then times `alpha`, in that order, so that `alpha`
/// doesn't make `g` overflow or vanish before the product.
struct ScaledProductBackward {
    product: ScaledProduct,
    /// `x`, saved when `y` has an edge, and `y`, saved when `x` has one.
    factors: [Option<SavedId>; 2],
    beta: Scalar,
    alpha: Scalar,
    input_shape: Shape,
}

impl Node for ScaledProductBackward {
    fn name(&self) -> &'static str {
        match self.product {
            ScaledProduct::Mm => "AddmmBackward0",
            ScaledProduct::Mv => "AddmvBackward0",
            ScaledProduct::Outer => "AddrBackward0",
            ScaledProduct::Bmm => "BaddbmmBackward0",
            ScaledProduct::SumBmm => "AddbmmBackward0",
        }
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        use ScaledProduct::*;
        let g = only(grads);
        let [x, y] = self.factors.map(|id| id.map(|id| ctx.unpack(id)));
        let grad_input = ctx
            .needs_input_grad(0)
            .then(|| scale(g.clone(), self.beta).sum_to_size(&self.input_shape));
        let grad_x = ctx.needs_input_grad(1).then(|| {
            let y = y.as_ref().expect("saved for x's gradient");
            let product = match self.product {
                // `addbmm`'s `g` broadcasts over the batch of `y`.
                Mm | Bmm | SumBmm => g.matmul(&mt(y)),
                Mv => g.unsqueeze(1).mul(y.unsqueeze(0)),
                Outer => g.matmul(y),
            };
            scale(product, self.alpha)
        });
        let grad_y = ctx.needs_input_grad(2).then(|| {
            let x = x.as_ref().expect("saved for y's gradient");
            let product = match self.product {
                Mm | Mv | Bmm | SumBmm => mt(x).matmul(&g),
                Outer => mt(&g).matmul(x),
            };
            scale(product, self.alpha)
        });
        vec![grad_input, grad_x, grad_y]
    }
}
