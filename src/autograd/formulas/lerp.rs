//! Lerp's direct derivative avoids contamination from an inactive forward branch.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::ops::Operand;
use crate::{Scalar, Shape, Tensor};

pub(crate) fn lerp(x: &Tensor, end: &Tensor, weight: &Operand, out: &Tensor) {
    let shapes = [x.shape(), end.shape()];
    match weight {
        Operand::Scalar(weight) => record(&[Some(x), Some(end)], &[out], |_| LerpBackward0 {
            weight: *weight,
            shapes,
        }),
        Operand::Tensor(weight) => record(&[Some(x), Some(end), Some(weight)], &[out], |b| {
            LerpBackward1 {
                endpoints: b.needs_grad(2).then(|| [b.save(x), b.save(end)]),
                // Saved and checked even if only the weight's own gradient is requested,
                // matching the reference's observable version and release semantics.
                weight: b.save(weight),
                shapes: [x.shape(), end.shape(), weight.shape()],
            }
        }),
    }
}

struct LerpBackward0 {
    weight: Scalar,
    shapes: [Shape; 2],
}

impl Node for LerpBackward0 {
    fn name(&self) -> &'static str {
        "LerpBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        vec![
            ctx.needs_input_grad(0).then(|| {
                g.mul(1.0 - self.weight.to_f64())
                    .sum_to_size(&self.shapes[0])
            }),
            ctx.needs_input_grad(1)
                .then(|| g.mul(self.weight).sum_to_size(&self.shapes[1])),
        ]
    }
}

struct LerpBackward1 {
    endpoints: Option<[SavedId; 2]>,
    weight: SavedId,
    shapes: [Shape; 3],
}

impl Node for LerpBackward1 {
    fn name(&self) -> &'static str {
        "LerpBackward1"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let endpoints = self.endpoints.map(|ids| ids.map(|id| ctx.unpack(id)));
        let weight = ctx.unpack(self.weight);
        vec![
            ctx.needs_input_grad(0)
                .then(|| g.mul(1.0 - &weight).sum_to_size(&self.shapes[0])),
            ctx.needs_input_grad(1)
                .then(|| g.mul(&weight).sum_to_size(&self.shapes[1])),
            ctx.needs_input_grad(2).then(|| {
                let [x, end] = endpoints.as_ref().expect("saved for the weight");
                g.mul(end.sub(x)).sum_to_size(&self.shapes[2])
            }),
        ]
    }
}
