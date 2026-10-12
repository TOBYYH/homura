//! Elementwise binary operations.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::{record, record_unimplemented};
use crate::autograd::saved::SavedId;
use crate::ops::{BinaryOp, Operand};
use crate::{Scalar, Shape, Tensor};

/// Records the backward node of `out = op(lhs, rhs)`.
///
/// As in PyTorch, a scalar operand has no edge, and a scalar on the left of `+` or `*` is
/// recorded on the right (`2 * x` is `x * 2`), while `scalar - x` is `RsubBackward1` with a
/// single edge.
pub(crate) fn binary(op: BinaryOp, lhs: &Operand, rhs: &Operand, out: &Tensor) {
    use BinaryOp::*;
    let tensor = |o: &Operand| match o {
        Operand::Tensor(t) => Some(t.clone()),
        Operand::Scalar(_) => None,
    };
    let unimplemented = match op {
        Add | Sub | Mul => None,
        Div => Some("div"),
        Pow => Some("pow"),
        Maximum => Some("maximum"),
        Minimum => Some("minimum"),
        // Comparisons and logical operations have bool results.
        _ => return,
    };
    if let Some(name) = unimplemented {
        let (a, b) = (tensor(lhs), tensor(rhs));
        return record_unimplemented(name, &[a.as_ref(), b.as_ref()], out);
    }
    let (first, second) = match (lhs, rhs) {
        (Operand::Scalar(s), Operand::Tensor(t)) => {
            if op == Sub {
                return record(&[Some(t)], &[out], |_| RsubBackward1);
            }
            (t, Operand::Scalar(*s))
        }
        (Operand::Tensor(t), other) => (t, other.clone()),
        (Operand::Scalar(_), Operand::Scalar(_)) => unreachable!("a tensor operand is required"),
    };
    let other = tensor(&second);
    let inputs = [Some(first), other.as_ref()];
    let shapes = [
        first.shape(),
        other.as_ref().map_or_else(Shape::scalar, Tensor::shape),
    ];
    match op {
        Add => record(&inputs, &[out], |_| AddBackward0 { shapes }),
        Sub => record(&inputs, &[out], |_| SubBackward0 { shapes }),
        _ => record(&inputs, &[out], |b| {
            // The gradient of each input is `g` times the other operand.
            let first_factor = b.needs_grad(0).then(|| match &second {
                Operand::Tensor(t) => Factor::Saved(b.save(t)),
                Operand::Scalar(s) => Factor::Scalar(*s),
            });
            let second_factor = b.needs_grad(1).then(|| Factor::Saved(b.save(first)));
            MulBackward0 {
                factors: [first_factor, second_factor],
                shapes,
            }
        }),
    }
}

/// `a + b`: each input gets `g`, summed to its shape.
struct AddBackward0 {
    shapes: [Shape; 2],
}

impl Node for AddBackward0 {
    fn name(&self) -> &'static str {
        "AddBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        (0..2)
            .map(|i| {
                ctx.needs_input_grad(i)
                    .then(|| g.sum_to_size(&self.shapes[i]))
            })
            .collect()
    }
}

/// `a - b`: `a` gets `g` and `b` gets `-g`, summed to their shapes.
struct SubBackward0 {
    shapes: [Shape; 2],
}

impl Node for SubBackward0 {
    fn name(&self) -> &'static str {
        "SubBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        vec![
            ctx.needs_input_grad(0)
                .then(|| g.sum_to_size(&self.shapes[0])),
            ctx.needs_input_grad(1)
                .then(|| g.neg().sum_to_size(&self.shapes[1])),
        ]
    }
}

/// `scalar - x`: the gradient is `-g`.
struct RsubBackward1;

impl Node for RsubBackward1 {
    fn name(&self) -> &'static str {
        "RsubBackward1"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).neg())]
    }
}

/// What the gradient of one factor of a product is multiplied by: the other factor.
enum Factor {
    Saved(SavedId),
    Scalar(Scalar),
}

/// `a * b`: `a` gets `g * b` and `b` gets `g * a`, summed to their shapes. Each factor is saved
/// only when the other one needs a gradient.
struct MulBackward0 {
    factors: [Option<Factor>; 2],
    shapes: [Shape; 2],
}

impl Node for MulBackward0 {
    fn name(&self) -> &'static str {
        "MulBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        (0..2)
            .map(|i| {
                if !ctx.needs_input_grad(i) {
                    return None;
                }
                let product = match self.factors[i].as_ref().expect("saved for an edge") {
                    Factor::Saved(id) => g.mul(ctx.unpack(*id)),
                    Factor::Scalar(s) => g.mul(*s),
                };
                Some(product.sum_to_size(&self.shapes[i]))
            })
            .collect()
    }
}
