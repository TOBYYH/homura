//! Clamp has its own gradient rules, distinct from maximum followed by minimum.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::ops::Operand;
use crate::{DType, Scalar, Shape, Tensor, where_};

fn tensor(o: Option<&Operand>) -> Option<&Tensor> {
    match o {
        Some(Operand::Tensor(t)) => Some(t),
        _ => None,
    }
}

/// Records scalar bounds with strict inequalities, tensor bounds with tie splitting. Mixed
/// bounds use the tensor rule, rounding the scalar at its corresponding forward stage: `dtypes`
/// are the dtypes that the `min` and `max` steps computed in. They are passed in rather than
/// taken from `out`, which `clamp_` makes the target of the write, whose dtype can be narrower.
#[track_caller]
pub(crate) fn clamp(
    x: &Tensor,
    min: Option<&Operand>,
    max: Option<&Operand>,
    dtypes: [DType; 2],
    out: &Tensor,
) {
    let bounds = [min, max];
    let tensor_rule = bounds.iter().any(|o| tensor(*o).is_some());
    let name = match (min.is_some(), max.is_some(), tensor_rule) {
        (true, true, true) => "ClampBackward0",
        (true, true, false) => "ClampBackward1",
        (true, false, true) => "ClampMinBackward1",
        (true, false, false) => "ClampMinBackward0",
        (false, true, true) => "ClampMaxBackward1",
        (false, true, false) => "ClampMaxBackward0",
        _ => unreachable!(),
    };
    let mut inputs = [Some(x), tensor(min), tensor(max)];
    // Preserve the public edge counts of scalar, single-bound and two-bound nodes.
    let (count, edges) = if !tensor_rule {
        (1, [Some(0), None, None])
    } else if min.is_none() {
        inputs.swap(1, 2);
        (2, [Some(0), None, Some(1)])
    } else if max.is_none() {
        (2, [Some(0), Some(1), None])
    } else {
        (3, [Some(0), Some(1), Some(2)])
    };
    record(&inputs[..count], &[out], |b| ClampBackward {
        name,
        x: b.save(x),
        bounds: bounds.map(|o| tensor(o).map(|t| b.save(t))),
        scalars: bounds.map(|o| match o {
            Some(Operand::Scalar(s)) => Some(*s),
            _ => None,
        }),
        dtypes,
        shapes: [
            x.shape(),
            tensor(min).map_or_else(Shape::scalar, Tensor::shape),
            tensor(max).map_or_else(Shape::scalar, Tensor::shape),
        ],
        tensor_rule,
        edges,
    });
}

struct ClampBackward {
    name: &'static str,
    x: SavedId,
    bounds: [Option<SavedId>; 2],
    scalars: [Option<Scalar>; 2],
    dtypes: [DType; 2],
    shapes: [Shape; 3],
    tensor_rule: bool,
    edges: [Option<usize>; 3],
}

impl Node for ClampBackward {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let needs = |i: usize| self.edges[i].is_some_and(|edge| ctx.needs_input_grad(edge));
        let x = ctx.unpack(self.x);
        let saved = self.bounds.map(|id| id.map(|id| ctx.unpack(id)));
        if !self.tensor_rule {
            let active = match self.scalars {
                [Some(min), Some(max)] => x.greater(min).logical_and(x.less(max)),
                [Some(min), None] => x.greater(min),
                [None, Some(max)] => x.less(max),
                _ => unreachable!(),
            };
            return vec![needs(0).then(|| where_(&active, &g, 0.0).sum_to_size(&self.shapes[0]))];
        }
        let bounds: [Option<Tensor>; 2] = std::array::from_fn(|i| {
            saved[i].clone().or_else(|| {
                self.scalars[i].map(|s| {
                    Tensor::from_data(crate::ops::scalar_data(s, self.dtypes[i], x.device()))
                })
            })
        });
        let half_at = |tie: Tensor| where_(&tie, g.div(2.0), &g);
        let local: [Option<Tensor>; 3] = match (&bounds[0], &bounds[1]) {
            (Some(min), Some(max)) => {
                let lt = min.less(max);
                let eq = min.equal(max);
                let dx = needs(0).then(|| {
                    let tie = x.equal(min).logical_or(x.equal(max)).logical_and(&lt);
                    half_at(tie).masked_fill(&x.less(min).logical_or(x.greater(max)), 0.0)
                });
                let dmin = needs(1).then(|| {
                    let active = lt
                        .logical_and(x.less_equal(min))
                        .logical_or(eq.logical_and(x.less(min)));
                    where_(&active, half_at(x.equal(min)), 0.0)
                });
                let dmax = needs(2).then(|| {
                    let active = max
                        .less(min)
                        .logical_or(lt.logical_and(x.greater_equal(max)))
                        .logical_or(eq.logical_and(x.greater(max)));
                    where_(&active, half_at(x.equal(max).logical_and(&lt)), 0.0)
                });
                [dx, dmin, dmax]
            }
            (Some(min), None) => {
                let split = half_at(x.equal(min));
                [
                    needs(0).then(|| split.masked_fill(&x.less(min), 0.0)),
                    needs(1).then(|| split.masked_fill(&x.greater(min), 0.0)),
                    None,
                ]
            }
            (None, Some(max)) => {
                let split = half_at(x.equal(max));
                [
                    needs(0).then(|| split.masked_fill(&x.greater(max), 0.0)),
                    None,
                    needs(2).then(|| split.masked_fill(&x.less(max), 0.0)),
                ]
            }
            _ => unreachable!(),
        };
        local
            .into_iter()
            .enumerate()
            .filter(|(i, _)| self.edges[*i].is_some())
            .map(|(i, d)| d.map(|d| d.sum_to_size(&self.shapes[i])))
            .collect()
    }
}
