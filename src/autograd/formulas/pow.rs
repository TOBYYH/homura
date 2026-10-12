//! Powers with scalar and tensor exponents.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::ops::Operand;
use crate::{Scalar, Shape, Tensor, where_};

pub(super) fn pow(base: &Operand, exponent: &Operand, out: &Tensor) {
    let Operand::Tensor(base) = base else {
        unreachable!("the public pow method has a tensor base")
    };
    match exponent {
        Operand::Scalar(exponent) => record(&[Some(base)], &[out], |b| PowBackward0 {
            base: b.save(base),
            exponent: *exponent,
        }),
        Operand::Tensor(exponent) => {
            record(&[Some(base), Some(exponent)], &[out], |b| PowBackward1 {
                base: b.save(base),
                exponent: b.save(exponent),
                // The reference saves the result even with no exponent edge; modifying it
                // must invalidate a backward pass that only requests the base's gradient.
                result: b.save_output(out, 0),
                shapes: [base.shape(), exponent.shape()],
            })
        }
    }
}

struct PowBackward0 {
    base: SavedId,
    exponent: Scalar,
}

impl Node for PowBackward0 {
    fn name(&self) -> &'static str {
        "PowBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let base = ctx.unpack(self.base);
        let exponent = self.exponent.to_f64();
        // PyTorch returns a constant zero, disconnected from both base and g, for a scalar
        // exponent of zero. Still unpack the saved base for release and version checks.
        vec![Some(if exponent == 0.0 {
            crate::zeros_like(&base)
        } else {
            g.mul(base.pow(exponent - 1.0).mul(exponent))
        })]
    }
}

struct PowBackward1 {
    base: SavedId,
    exponent: SavedId,
    result: SavedId,
    shapes: [Shape; 2],
}

impl Node for PowBackward1 {
    fn name(&self) -> &'static str {
        "PowBackward1"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let base = ctx.unpack(self.base);
        let exponent = ctx.unpack(self.exponent);
        let result = ctx.unpack(self.result);
        vec![
            ctx.needs_input_grad(0).then(|| {
                let product = g.mul(exponent.mul(base.pow(exponent.sub(1))));
                where_(&exponent.equal(0), 0.0, product).sum_to_size(&self.shapes[0])
            }),
            ctx.needs_input_grad(1).then(|| {
                let derivative = result.mul(base.to(result.dtype()).log());
                let zero = base.equal(0).logical_and(exponent.greater_equal(0));
                // The multiplication stays outside the selection: a zero derivative with an
                // infinite upstream gradient is NaN in the reference.
                g.mul(where_(&zero, 0.0, derivative))
                    .sum_to_size(&self.shapes[1])
            }),
        ]
    }
}
