//! Elementwise unary operations.

use super::only;
use crate::Tensor;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::{record, record_unimplemented};
use crate::autograd::saved::SavedId;
use crate::ops::UnaryOp;

/// Records the backward node of `out = op(x)`.
pub(crate) fn unary(op: UnaryOp, x: &Tensor, out: &Tensor) {
    use UnaryOp::*;
    let inputs = [Some(x)];
    let name = match op {
        Neg => return record(&inputs, &[out], |_| NegBackward0),
        Exp => {
            return record(&inputs, &[out], |b| ExpBackward0 {
                result: b.save_output(out, 0),
            });
        }
        Erf => return record(&inputs, &[out], |b| ErfBackward0 { x: b.save(x) }),
        Sign => return record(&inputs, &[out], |_| SignBackward0),
        LogicalNot => return,
        Abs => "abs",
        Relu => "relu",
        Log => "log",
        Sqrt => "sqrt",
        Rsqrt => "rsqrt",
        Reciprocal => "reciprocal",
        Sin => "sin",
        Cos => "cos",
        Tanh => "tanh",
        Sigmoid => "sigmoid",
        Silu => "silu",
        Gelu | GeluTanh => "gelu",
    };
    record_unimplemented(name, &inputs, out);
}

/// `-x`: the gradient is `-g`.
struct NegBackward0;

impl Node for NegBackward0 {
    fn name(&self) -> &'static str {
        "NegBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).neg())]
    }
}

/// `e^x`: the gradient is `g * e^x`, computed from the saved result.
struct ExpBackward0 {
    result: SavedId,
}

impl Node for ExpBackward0 {
    fn name(&self) -> &'static str {
        "ExpBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).mul(ctx.unpack(self.result)))]
    }
}

/// `erf(x)`: the gradient is `2/√π · e^(−x²) · g`, computed from the saved input in PyTorch's
/// order of operations.
struct ErfBackward0 {
    x: SavedId,
}

impl Node for ErfBackward0 {
    fn name(&self) -> &'static str {
        "ErfBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let x = ctx.unpack(self.x);
        // `x * x` equals PyTorch's `x.pow(2)` and only needs the multiplication's formula.
        let derivative = x.mul(&x).neg().exp().mul(2.0 / std::f64::consts::PI.sqrt());
        vec![Some(derivative.mul(only(grads)))]
    }
}

/// `sign(x)`: the gradient is zero, even where `g` is infinite or NaN, as in PyTorch.
struct SignBackward0;

impl Node for SignBackward0 {
    fn name(&self) -> &'static str {
        "SignBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(crate::zeros_like(&only(grads)))]
    }
}
