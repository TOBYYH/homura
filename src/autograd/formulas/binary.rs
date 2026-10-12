//! Elementwise binary and ternary operations.

use super::only;
use crate::autograd::node::{ApplyContext, Node, NodeBuilder};
use crate::autograd::record::{record, record_unimplemented};
use crate::autograd::saved::SavedId;
use crate::ops::{BinaryOp, Operand, binary_op};
use crate::{DType, Device, Scalar, Shape, Tensor};

/// The tensor of an operand, `None` for a scalar.
fn tensor(o: &Operand) -> Option<Tensor> {
    match o {
        Operand::Tensor(t) => Some(t.clone()),
        Operand::Scalar(_) => None,
    }
}

/// The shape of an operand; a scalar has no dimensions.
fn shape(o: &Operand) -> Shape {
    match o {
        Operand::Tensor(t) => t.shape(),
        Operand::Scalar(_) => Shape::scalar(),
    }
}

/// Records the backward node of `out = op(lhs, rhs)`.
///
/// As in PyTorch, a scalar operand has no edge, and a scalar on the left of `+` or `*` is
/// recorded on the right (`2 * x` is `x * 2`), while `scalar - x` is `RsubBackward1` with a
/// single edge.
pub(crate) fn binary(op: BinaryOp, lhs: &Operand, rhs: &Operand, out: &Tensor) {
    use BinaryOp::*;
    let unimplemented = match op {
        Add | Sub | Mul => None,
        Div => return div(lhs, rhs, out),
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
    let shapes = [first.shape(), shape(&second)];
    match op {
        Add => record(&inputs, &[out], |_| AddBackward0 { shapes }),
        Sub => record(&inputs, &[out], |_| SubBackward0 { shapes }),
        _ => record(&inputs, &[out], |b| {
            // The gradient of each input is `g` times the other operand.
            let first_factor = b.needs_grad(0).then(|| save_operand(b, &second));
            let second_factor = b.needs_grad(1).then(|| b.save(first));
            MulBackward0 {
                factors: [first_factor, second_factor],
                scalar_factor: matches!(second, Operand::Scalar(_)),
                shapes,
            }
        }),
    }
}

/// Records the backward node of `out = lhs / rhs`. Unlike `+` and `*`, a scalar dividend stays
/// on the left: `s / x` is a `DivBackward0` whose first input, the scalar, has no edge, like
/// PyTorch's `torch.div(torch.tensor(s), x)`. (PyTorch's Python operator computes
/// `x.reciprocal() * s` instead, and records two nodes.)
fn div(lhs: &Operand, rhs: &Operand, out: &Tensor) {
    let (a, b) = (tensor(lhs), tensor(rhs));
    let shapes = [shape(lhs), shape(rhs)];
    record(&[a.as_ref(), b.as_ref()], &[out], |builder| {
        // As in PyTorch, the divisor is saved for both gradients, the dividend only for the
        // divisor's.
        let dividend = builder.needs_grad(1).then(|| save_operand(builder, lhs));
        let divisor = save_operand(builder, rhs);
        DivBackward0 {
            dividend,
            divisor,
            scalars: [lhs, rhs].map(|o| matches!(o, Operand::Scalar(_))),
            shapes,
        }
    });
}

/// Records the backward node of `out = where_(condition, x, y)`. The condition has no edge, and
/// neither has a scalar `x` or `y`.
pub(crate) fn where_(condition: &Tensor, x: &Operand, y: &Operand, out: &Tensor) {
    let (tx, ty) = (tensor(x), tensor(y));
    let shapes = [shape(x), shape(y)];
    record(&[tx.as_ref(), ty.as_ref()], &[out], |b| WhereBackward0 {
        condition: b.save(condition),
        shapes,
    });
}

/// Saves an operand for a formula. As in PyTorch, a scalar is saved as a tensor, so it is
/// released with the graph like any other operand. That tensor keeps the scalar's own type and
/// value on the CPU, like PyTorch's wrapped numbers, and becomes a scalar again for the
/// arithmetic ([`scalar_of`]).
fn save_operand(b: &mut NodeBuilder, o: &Operand) -> SavedId {
    match o {
        Operand::Tensor(t) => b.save(t),
        Operand::Scalar(s) => {
            let dtype = match s {
                Scalar::Bool(_) => DType::Bool,
                Scalar::Int(_) => DType::I64,
                Scalar::Float(_) => DType::F64,
            };
            let data = crate::ops::scalar_data(*s, dtype, Device::Cpu);
            b.save(&Tensor::from_data(data))
        }
    }
}

/// The scalar saved as a tensor by [`save_operand`].
fn scalar_of(saved: &Tensor) -> Scalar {
    match saved.dtype() {
        DType::Bool => Scalar::Bool(saved.item()),
        DType::I64 => Scalar::Int(saved.item()),
        _ => Scalar::Float(saved.item()),
    }
}

/// An operand saved by [`save_operand`], as the operand it was.
fn operand_of(saved: &Tensor, scalar: bool) -> Operand {
    if scalar {
        Operand::Scalar(scalar_of(saved))
    } else {
        Operand::Tensor(saved.clone())
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

/// `a * b`: `a` gets `g * b` and `b` gets `g * a`, summed to their shapes. Each factor is saved
/// only when the other one needs a gradient. When `b` is a scalar, `scalar_factor` is set and
/// its saved tensor is multiplied as a scalar.
struct MulBackward0 {
    factors: [Option<SavedId>; 2],
    scalar_factor: bool,
    shapes: [Shape; 2],
}

impl Node for MulBackward0 {
    fn name(&self) -> &'static str {
        "MulBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        // As in PyTorch, every saved factor is unpacked, and so checked, even when the gradient
        // that needs it isn't computed in this pass.
        let factors = self.factors.map(|id| id.map(|id| ctx.unpack(id)));
        (0..2)
            .map(|i| {
                if !ctx.needs_input_grad(i) {
                    return None;
                }
                let factor = factors[i].as_ref().expect("saved for an edge");
                let product = g.mul(operand_of(factor, i == 0 && self.scalar_factor));
                Some(product.sum_to_size(&self.shapes[i]))
            })
            .collect()
    }
}

/// `a / b`: `a` gets `g / b` and `b` gets `-g * ((a / b) / b)`, summed to their shapes, as
/// PyTorch's `div_tensor_self_backward` and `div_tensor_other_backward` compute them. Either
/// operand may be a scalar (`scalars`), saved as a tensor and divided as a scalar again.
struct DivBackward0 {
    /// Saved when the divisor needs a gradient.
    dividend: Option<SavedId>,
    divisor: SavedId,
    scalars: [bool; 2],
    shapes: [Shape; 2],
}

impl Node for DivBackward0 {
    fn name(&self) -> &'static str {
        "DivBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        // As in PyTorch, both saved operands are unpacked, and so checked, even when only the
        // dividend's gradient, which doesn't need the dividend, is computed.
        let divisor = ctx.unpack(self.divisor);
        let dividend = self.dividend.map(|id| ctx.unpack(id));
        vec![
            ctx.needs_input_grad(0).then(|| {
                g.div(operand_of(&divisor, self.scalars[1]))
                    .sum_to_size(&self.shapes[0])
            }),
            ctx.needs_input_grad(1).then(|| {
                // The divisor has an edge, so it is a tensor. A scalar dividend is divided as the
                // public `scalar / tensor` operator does.
                let dividend = dividend.as_ref().expect("saved for the divisor's gradient");
                let quotient = binary_op(
                    BinaryOp::Div,
                    operand_of(dividend, self.scalars[0]),
                    Operand::Tensor(divisor.clone()),
                );
                g.neg()
                    .mul(quotient.div(&divisor))
                    .sum_to_size(&self.shapes[1])
            }),
        ]
    }
}

/// `where_(condition, x, y)`: `x` gets `g` where the condition holds and `y` elsewhere, and each
/// gets zeros at the other positions, even where `g` is infinite or NaN: the gradients are
/// selected rather than multiplied by a mask, as in PyTorch. Each is then summed to its input's
/// shape.
struct WhereBackward0 {
    condition: SavedId,
    shapes: [Shape; 2],
}

impl Node for WhereBackward0 {
    fn name(&self) -> &'static str {
        "WhereBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let condition = ctx.unpack(self.condition);
        vec![
            ctx.needs_input_grad(0)
                .then(|| crate::where_(&condition, &g, 0.0).sum_to_size(&self.shapes[0])),
            ctx.needs_input_grad(1)
                .then(|| crate::where_(&condition, 0.0, &g).sum_to_size(&self.shapes[1])),
        ]
    }
}
