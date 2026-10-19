//! The node of `nn::functional::embedding`, which is a single node in PyTorch.

use super::only;
use crate::Tensor;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;

/// Records `out = embedding(weight, indices)`: the rows of the 2-D `weight` at `indices`.
pub(crate) fn embedding(weight: &Tensor, indices: &Tensor, padding_idx: Option<i64>, out: &Tensor) {
    record(&[Some(weight)], &[out], |b| EmbeddingBackward0 {
        indices: b.save(indices),
        num_weights: weight.shape()[0],
        padding_idx,
    });
}

/// `embedding`: PyTorch's `embedding_dense_backward`, which adds the gradient of each position
/// to the row of its index, except for positions whose index is `padding_idx`: their row gets
/// nothing, even where `g` is infinite or NaN. The gradient is linear in `g` and doesn't depend
/// on the weights, like PyTorch's `embedding_dense_double_backward`.
struct EmbeddingBackward0 {
    indices: SavedId,
    num_weights: usize,
    padding_idx: Option<i64>,
}

impl Node for EmbeddingBackward0 {
    fn name(&self) -> &'static str {
        "EmbeddingBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let indices = ctx.unpack(self.indices).reshape([-1]);
        let dim = g.shape()[g.num_dims() - 1];
        let g = g.reshape([indices.num_elements() as i64, dim as i64]);
        let g = match self.padding_idx {
            Some(p) => g.masked_fill(&indices.equal(p).unsqueeze(1), 0.0),
            None => g,
        };
        let zeros = g.new_zeros([self.num_weights, dim]);
        vec![Some(zeros.index_add(0, &indices, &g))]
    }
}
