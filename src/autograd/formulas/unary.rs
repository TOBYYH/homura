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
