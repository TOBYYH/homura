//! Scaled products and quotients follow the reference's derivative evaluation order.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::dtype::ResultType;
use crate::{Scalar, Shape, Tensor};

#[track_caller]
pub(crate) fn addcmul(x: &Tensor, a: &Tensor, b: &Tensor, value: Scalar, out: &Tensor) {
    record(&[Some(x), Some(a), Some(b)], &[out], |builder| {
        AddcmulBackward0 {
            factors: [
                builder.needs_grad(1).then(|| builder.save(b)),
                builder.needs_grad(2).then(|| builder.save(a)),
            ],
            value,
            shapes: [x.shape(), a.shape(), b.shape()],
        }
    });
}

#[track_caller]
pub(crate) fn addcdiv(x: &Tensor, a: &Tensor, b: &Tensor, value: Scalar, out: &Tensor) {
    record(&[Some(x), Some(a), Some(b)], &[out], |builder| {
        AddcdivBackward0 {
            numerator: builder.needs_grad(2).then(|| builder.save(a)),
            // The reference saves the divisor even if only x needs a gradient.
            divisor: builder.save(b),
            value,
            shapes: [x.shape(), a.shape(), b.shape()],
        }
    });
}

struct AddcmulBackward0 {
    factors: [Option<SavedId>; 2],
    value: Scalar,
    shapes: [Shape; 3],
}

impl Node for AddcmulBackward0 {
    fn name(&self) -> &'static str {
        "AddcmulBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let factors = self.factors.map(|id| id.map(|id| ctx.unpack(id)));
        let mut result = vec![
            ctx.needs_input_grad(0)
                .then(|| g.sum_to_size(&self.shapes[0])),
        ];
        for (i, factor) in factors.iter().enumerate() {
            result.push(ctx.needs_input_grad(i + 1).then(|| {
                // Scaling before multiplication by g avoids an intermediate overflow or
                // underflow that may disappear in the final mathematical result.
                g.mul(
                    factor
                        .as_ref()
                        .expect("saved for the other input")
                        .mul(self.value),
                )
                .sum_to_size(&self.shapes[i + 1])
            }));
        }
        result
    }
}

struct AddcdivBackward0 {
    numerator: Option<SavedId>,
    divisor: SavedId,
    value: Scalar,
    shapes: [Shape; 3],
}

impl Node for AddcdivBackward0 {
    fn name(&self) -> &'static str {
        "AddcdivBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let divisor = ctx.unpack(self.divisor);
        let numerator = self.numerator.map(|id| ctx.unpack(id));
        vec![
            ctx.needs_input_grad(0)
                .then(|| g.sum_to_size(&self.shapes[0])),
            ctx.needs_input_grad(1).then(|| {
                // C++ Scalar / Tensor in the reference rounds the coefficient to the
                // division's dtype first, including f16/bf16. Our public scalar / tensor
                // overload instead keeps the unrounded scalar in f32 opmath for half types.
                // A zero-dimensional tensor reproduces the reference without reading data
                // on Meta. Retain floating promotion for homura's integer-divisor extension.
                let mut rt = ResultType::new();
                rt.tensor(divisor.dtype(), divisor.num_dims())
                    .scalar(self.value);
                let dtype = rt.result();
                let dtype = if dtype.is_floating_point() {
                    dtype
                } else {
                    crate::default_dtype()
                };
                let value = crate::full_with((), self.value, (dtype, divisor.device()));
                g.mul(value.div(&divisor)).sum_to_size(&self.shapes[1])
            }),
            ctx.needs_input_grad(2).then(|| {
                let numerator = numerator
                    .as_ref()
                    .expect("saved for the divisor's gradient");
                let derivative = numerator.mul(self.value).div(divisor.mul(&divisor));
                g.neg().mul(derivative).sum_to_size(&self.shapes[2])
            }),
        ]
    }
}
