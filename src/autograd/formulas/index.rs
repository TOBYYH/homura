//! Indexing, gathering and scattering. These operations come in pairs whose formulas use each
//! other (`index_select` and `index_add`, `gather` and `scatter_add`, `select` and
//! `select_scatter`, slicing and `slice_scatter`), so that every derivative can be
//! differentiated again. Index tensors have no gradient: they are saved, never given an edge.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::{DType, Shape, Tensor};

/// Records the backward node of `out = x.index_select(dim, index)`.
pub(crate) fn index_select(x: &Tensor, dim: usize, index: &Tensor, out: &Tensor) {
    record(&[Some(x)], &[out], |b| IndexSelectBackward0 {
        self_shape: x.shape(),
        dim,
        index: b.save(index),
    });
}

/// Records the backward node of indexing `x` with the integer tensor `index` along `dim`, whose
/// result `out` has the positions along `dim` flattened (the caller reshapes it afterwards).
/// Unlike `index_select`, negative positions count from the end.
pub(crate) fn index(x: &Tensor, dim: usize, index: &Tensor, out: &Tensor) {
    record(&[Some(x)], &[out], |b| IndexBackward0 {
        self_shape: x.shape(),
        dim,
        index: b.save(index),
    });
}

/// Records the backward node of `out = x.index_add(dim, index, source)`. As in PyTorch, the index
/// is saved only when `source` needs a gradient.
pub(crate) fn index_add(x: &Tensor, dim: usize, index: &Tensor, source: &Tensor, out: &Tensor) {
    record(&[Some(x), Some(source)], &[out], |b| IndexAddBackward0 {
        dim,
        index: b.needs_grad(1).then(|| b.save(index)),
    });
}

/// Records the backward node of `out = x.gather(dim, index)`.
pub(crate) fn gather(x: &Tensor, dim: usize, index: &Tensor, out: &Tensor) {
    record(&[Some(x)], &[out], |b| GatherBackward0 {
        self_shape: x.shape(),
        dim,
        index: b.save(index),
    });
}

/// Records the backward node of `out = x.scatter_add(dim, index, src)`. As in PyTorch, the index
/// is saved only when `src` needs a gradient.
pub(crate) fn scatter_add(x: &Tensor, dim: usize, index: &Tensor, src: &Tensor, out: &Tensor) {
    record(&[Some(x), Some(src)], &[out], |b| ScatterAddBackward0 {
        dim,
        index: b.needs_grad(1).then(|| b.save(index)),
    });
}

/// Records the backward node of `out = x.select_scatter(src, dim, index)`, with `index` already
/// wrapped.
pub(crate) fn select_scatter(x: &Tensor, src: &Tensor, dim: usize, index: usize, out: &Tensor) {
    record(&[Some(x), Some(src)], &[out], |_| SelectScatterBackward0 {
        src_shape: src.shape(),
        dim,
        index,
    });
}

/// Records the backward node of `out = x.slice_scatter(src, dim, ..)`, with `start` and `end`
/// already clamped into the dimension.
pub(crate) fn slice_scatter(
    x: &Tensor,
    src: &Tensor,
    dim: usize,
    start: usize,
    end: usize,
    step: usize,
    out: &Tensor,
) {
    record(&[Some(x), Some(src)], &[out], |_| SliceScatterBackward0 {
        src_shape: src.shape(),
        dim,
        start,
        end,
        step,
    });
}

/// `x.index_select(dim, index)`: the gradient added into zeros of the shape of `x` at the
/// positions it came from, so that repeated positions add up.
struct IndexSelectBackward0 {
    self_shape: Shape,
    dim: usize,
    index: SavedId,
}

impl Node for IndexSelectBackward0 {
    fn name(&self) -> &'static str {
        "IndexSelectBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let index = ctx.unpack(self.index);
        let g = only(grads);
        let zeros = g.new_zeros(&self.self_shape);
        vec![Some(zeros.index_add(self.dim as i64, &index, &g))]
    }
}

/// Indexing with an integer tensor: like `IndexSelectBackward0`, after flattening the index and
/// wrapping its negative positions.
struct IndexBackward0 {
    self_shape: Shape,
    dim: usize,
    index: SavedId,
}

impl Node for IndexBackward0 {
    fn name(&self) -> &'static str {
        "IndexBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let index = ctx.unpack(self.index).reshape([-1]).to(DType::I64);
        let size = self.self_shape[self.dim] as i64;
        let index = crate::where_(&index.less(0), &index + size, &index);
        let g = only(grads);
        let zeros = g.new_zeros(&self.self_shape);
        vec![Some(zeros.index_add(self.dim as i64, &index, &g))]
    }
}

/// `x.index_add(dim, index, source)`: `x` gets `g`, and `source` gets the slices of `g` it was
/// added to.
struct IndexAddBackward0 {
    dim: usize,
    index: Option<SavedId>,
}

impl Node for IndexAddBackward0 {
    fn name(&self) -> &'static str {
        "IndexAddBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        // Unpacked, and so checked, even when this pass doesn't need the gradient of `source`.
        let index = self.index.map(|id| ctx.unpack(id));
        let g = only(grads);
        let source = ctx.needs_input_grad(1).then(|| {
            let index = index.as_ref().expect("saved for an edge");
            g.index_select(self.dim as i64, index)
        });
        vec![ctx.needs_input_grad(0).then_some(g), source]
    }
}

/// `x.gather(dim, index)`: the gradient added into zeros of the shape of `x` at the positions it
/// came from, so that repeated positions add up.
struct GatherBackward0 {
    self_shape: Shape,
    dim: usize,
    index: SavedId,
}

impl Node for GatherBackward0 {
    fn name(&self) -> &'static str {
        "GatherBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let index = ctx.unpack(self.index);
        let g = only(grads);
        let zeros = g.new_zeros(&self.self_shape);
        vec![Some(zeros.scatter_add(self.dim as i64, &index, &g))]
    }
}

/// `x.scatter_add(dim, index, src)`: `x` gets `g`, and `src` gets the elements of `g` it was
/// added to. As in PyTorch, that gradient has the shape of the index, so a backward pass fails
/// when `src` is larger than the index.
struct ScatterAddBackward0 {
    dim: usize,
    index: Option<SavedId>,
}

impl Node for ScatterAddBackward0 {
    fn name(&self) -> &'static str {
        "ScatterAddBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        // Unpacked, and so checked, even when this pass doesn't need the gradient of `src`.
        let index = self.index.map(|id| ctx.unpack(id));
        let g = only(grads);
        let src = ctx.needs_input_grad(1).then(|| {
            let index = index.as_ref().expect("saved for an edge");
            g.gather(self.dim as i64, index)
        });
        vec![ctx.needs_input_grad(0).then_some(g), src]
    }
}

/// `x.select_scatter(src, dim, index)`: `x` gets `g` with the overwritten slice zeroed, and `src`
/// gets that slice of `g`.
struct SelectScatterBackward0 {
    src_shape: Shape,
    dim: usize,
    index: usize,
}

impl Node for SelectScatterBackward0 {
    fn name(&self) -> &'static str {
        "SelectScatterBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let (dim, index) = (self.dim as i64, self.index as i64);
        vec![
            ctx.needs_input_grad(0)
                .then(|| g.select_scatter(&g.new_zeros(&self.src_shape), dim, index)),
            ctx.needs_input_grad(1).then(|| g.select(dim, index)),
        ]
    }
}

/// `x.slice_scatter(src, dim, ..)`: `x` gets `g` with the overwritten slice zeroed, and `src`
/// gets that slice of `g`.
struct SliceScatterBackward0 {
    src_shape: Shape,
    dim: usize,
    start: usize,
    end: usize,
    step: usize,
}

impl Node for SliceScatterBackward0 {
    fn name(&self) -> &'static str {
        "SliceScatterBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let (start, end, step) = (
            Some(self.start as i64),
            Some(self.end as i64),
            self.step as i64,
        );
        vec![
            ctx.needs_input_grad(0).then(|| {
                let zeros = g.new_zeros(&self.src_shape);
                g.slice_scatter(&zeros, self.dim as i64, start, end, step)
            }),
            ctx.needs_input_grad(1)
                .then(|| g.slice(self.dim, start, end, step)),
        ]
    }
}
