//! Reductions.

use super::only;
use crate::autograd::is_grad_enabled;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::ops::exclusive_cumprod;
use crate::shape::dims_mask;
use crate::{Device, Shape, Tensor};

/// The shape that the gradient of a reduction of `shape` over the dimensions in `mask` is
/// reshaped to before it is broadcast back, with the reduced dimensions put back as 1s, like
/// PyTorch's `unsqueeze`s; `None` with `keepdim`, which kept them.
fn unreduced(shape: &Shape, mask: &[bool], keepdim: bool) -> Option<Shape> {
    if keepdim {
        return None;
    }
    let kept: Vec<usize> = shape
        .iter()
        .zip(mask)
        .map(|(&n, &reduced)| if reduced { 1 } else { n })
        .collect();
    Some(Shape::new(&kept))
}

/// The number of elements of `shape` in the dimensions in `mask`.
fn reduced_count(shape: &Shape, mask: &[bool]) -> usize {
    shape
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(&n, _)| n)
        .product()
}

/// The mask of the single dimension `dim` among `num_dims`.
fn single_dim(dim: usize, num_dims: usize) -> Vec<bool> {
    (0..num_dims).map(|d| d == dim).collect()
}

/// Records the backward node of `out = x.sum()`.
pub(crate) fn sum(x: &Tensor, out: &Tensor) {
    let self_shape = x.shape();
    record(&[Some(x)], &[out], |_| SumBackward0 { self_shape });
}

/// Records the backward node of `out = x.sum_dim(dims, keepdim)`.
pub(crate) fn sum_dim(x: &Tensor, dims: &[i64], keepdim: bool, out: &Tensor) {
    record(&[Some(x)], &[out], |_| {
        let self_shape = x.shape();
        let mask = dims_mask(dims, self_shape.len());
        SumBackward1 {
            unreduced: unreduced(&self_shape, &mask, keepdim),
            self_shape,
        }
    });
}

/// Records the backward node of `out = x.sum_to_size(..)`, when it computed a new tensor.
pub(crate) fn sum_to_size(x: &Tensor, out: &Tensor) {
    let self_shape = x.shape();
    record(&[Some(x)], &[out], |_| SumToSizeBackward0 { self_shape });
}

/// Records the backward node of `out = x.mean()`, or of `out = x.mean_dim(dims, keepdim)` when
/// `dims` is `Some`.
pub(crate) fn mean(x: &Tensor, dims: Option<&[i64]>, keepdim: bool, out: &Tensor) {
    record(&[Some(x)], &[out], |_| {
        let self_shape = x.shape();
        let Some(dims) = dims else {
            return MeanBackward {
                name: "MeanBackward0",
                unreduced: None,
                count: self_shape.num_elements(),
                self_shape,
            };
        };
        let mask = dims_mask(dims, self_shape.len());
        MeanBackward {
            name: "MeanBackward1",
            unreduced: unreduced(&self_shape, &mask, keepdim),
            count: reduced_count(&self_shape, &mask),
            self_shape,
        }
    });
}

/// Records the backward node of `out = x.prod()`, or of `out = x.prod_dim(dim, keepdim)` with
/// `dim` already wrapped when `dim` is `Some`.
pub(crate) fn prod(x: &Tensor, dim: Option<usize>, keepdim: bool, out: &Tensor) {
    record(&[Some(x)], &[out], |b| ProdBackward {
        x: b.save(x),
        result: b.save_output(out, 0),
        reduced: dim.map(|d| {
            let shape = x.shape();
            (d, unreduced(&shape, &single_dim(d, shape.len()), keepdim))
        }),
    });
}

/// Records the backward node of `out = x.max()` or `out = x.min()`.
pub(crate) fn extremum(x: &Tensor, is_max: bool, out: &Tensor) {
    record(&[Some(x)], &[out], |b| ExtremumBackward {
        name: if is_max {
            "MaxBackward1"
        } else {
            "MinBackward1"
        },
        x: b.save(x),
        result: b.save_output(out, 0),
    });
}

/// Records the backward node of the `values` of `x.max_dim(dim, keepdim)` or
/// `x.min_dim(dim, keepdim)`, with `dim` already wrapped. The `indices` are saved, but are not an
/// output of the node: they never require grad.
pub(crate) fn extremum_dim(
    x: &Tensor,
    dim: usize,
    keepdim: bool,
    is_max: bool,
    values: &Tensor,
    indices: &Tensor,
) {
    record(&[Some(x)], &[values], |b| {
        let self_shape = x.shape();
        ExtremumDimBackward {
            name: if is_max {
                "MaxBackward0"
            } else {
                "MinBackward0"
            },
            indices: b.save(indices),
            dim,
            unreduced: unreduced(&self_shape, &single_dim(dim, self_shape.len()), keepdim),
            self_shape,
        }
    });
}

/// Records the backward node of `out = x.var(..)` or `out = x.std(..)` (`std`), over `dims`
/// (`None` for the whole tensor). An empty list of dimensions, like a zero-dimensional input,
/// reduces the whole tensor, and gets the whole tensor's gradient.
pub(crate) fn var_std(
    x: &Tensor,
    dims: Option<&[i64]>,
    correction: i64,
    keepdim: bool,
    std: bool,
    out: &Tensor,
) {
    record(&[Some(x)], &[out], |b| {
        let shape = x.shape();
        let dims = dims.filter(|d| !d.is_empty() && !shape.is_empty());
        let reduced = dims.map(|dims| {
            let mask = dims_mask(dims, shape.len());
            VarDims {
                dims: dims.to_vec(),
                count: reduced_count(&shape, &mask),
                unreduced: unreduced(&shape, &mask, keepdim),
            }
        });
        VarBackward {
            name: if std { "StdBackward0" } else { "VarBackward0" },
            x: b.save(x),
            result: std.then(|| b.save_output(out, 0)),
            reduced,
            correction,
        }
    });
}

/// `x.sum()`: the zero-dimensional gradient expanded to the shape of `x`.
struct SumBackward0 {
    self_shape: Shape,
}

impl Node for SumBackward0 {
    fn name(&self) -> &'static str {
        "SumBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).expand(&self.self_shape))]
    }
}

/// `x.sum_dim(dims, keepdim)`: the gradient, with the reduced dimensions put back when they
/// were removed, expanded to the shape of `x`.
struct SumBackward1 {
    self_shape: Shape,
    unreduced: Option<Shape>,
}

impl Node for SumBackward1 {
    fn name(&self) -> &'static str {
        "SumBackward1"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = match &self.unreduced {
            Some(kept) => only(grads).reshape(kept),
            None => only(grads),
        };
        vec![Some(g.expand(&self.self_shape))]
    }
}

/// `x.sum_to_size(size)`: the gradient, of shape `size`, expanded back to the shape of `x`.
/// PyTorch records this operation as a `SumBackward1` with `keepdim`, followed by a
/// `ViewBackward0` when leading dimensions are removed; homura records one node.
struct SumToSizeBackward0 {
    self_shape: Shape,
}

impl Node for SumToSizeBackward0 {
    fn name(&self) -> &'static str {
        "SumToSizeBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).expand(&self.self_shape))]
    }
}

/// `x.mean()` (`MeanBackward0`) and `x.mean_dim(dims, keepdim)` (`MeanBackward1`): the gradient
/// of the sum, divided by the number of elements each mean is taken over.
struct MeanBackward {
    name: &'static str,
    self_shape: Shape,
    unreduced: Option<Shape>,
    count: usize,
}

impl Node for MeanBackward {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = match &self.unreduced {
            Some(kept) => only(grads).reshape(kept),
            None => only(grads),
        };
        vec![Some(g.expand(&self.self_shape).div(self.count as i64))]
    }
}

/// `x.prod()` (`ProdBackward0`) and `x.prod_dim(dim, keepdim)` (`ProdBackward1`), as PyTorch's
/// `prod_backward` computes them. Without zeros in `x`, `g * (result / x)`. With zeros, `g`
/// times the product of all other elements of the slice, in every slice; a whole product with
/// more than one zero gives zeros instead, outside grad mode. These products aren't
/// differentiable yet, so with `create_graph` (grad mode) such inputs are rejected. On the meta
/// device, where values are unknown, the gradient is the one without zeros.
struct ProdBackward {
    x: SavedId,
    result: SavedId,
    /// The dimension of `prod_dim`, and its shape kept when it was removed. `None` for `prod`.
    reduced: Option<(usize, Option<Shape>)>,
}

impl Node for ProdBackward {
    fn name(&self) -> &'static str {
        if self.reduced.is_some() {
            "ProdBackward1"
        } else {
            "ProdBackward0"
        }
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let x = ctx.unpack(self.x);
        let result = ctx.unpack(self.result);
        if x.num_dims() == 0 {
            return vec![Some(g)];
        }
        let (g, result) = match &self.reduced {
            Some((_, Some(kept))) => (g.reshape(kept), result.reshape(kept)),
            _ => (g, result),
        };
        let zeros = if x.device() == Device::Meta {
            0
        } else {
            x.equal(0.0).sum().item::<i64>()
        };
        if zeros == 0 {
            return vec![Some(g.mul(result.div(&x)))];
        }
        if self.reduced.is_none() && zeros > 1 && !is_grad_enabled() {
            return vec![Some(crate::zeros_like(&x))];
        }
        let (flat, dim) = match self.reduced {
            Some((dim, _)) => (x.clone(), dim),
            None => (x.reshape([-1]), 0),
        };
        if flat.shape()[dim] == 1 {
            return vec![Some(g.reshape(x.shape()))];
        }
        if is_grad_enabled() {
            panic!(
                "computing the gradient of prod with create_graph for an input that contains \
                 zeros is not supported yet"
            );
        }
        let products = g.mul(products_of_the_others(&flat, dim));
        vec![Some(products.reshape(x.shape()))]
    }
}

/// The product of all elements but one along `dim` of `x`, for each element, as PyTorch's
/// `prod_safe_zeros_backward` computes it without dividing: the product of the elements before
/// it times the product of those after it, each accumulated in order and rounded to the dtype of
/// `x` like PyTorch's `cumprod`. Correct with zeros and infinities, but computed by a kernel that
/// isn't recorded, unlike the other formulas: this gradient is first-order only.
fn products_of_the_others(x: &Tensor, dim: usize) -> Tensor {
    let before = exclusive_cumprod(x, dim, false);
    let after = exclusive_cumprod(x, dim, true);
    before.mul(after)
}

/// `x.max()` (`MaxBackward1`) and `x.min()` (`MinBackward1`): `g` divided evenly among the
/// positions that hold the result, selected rather than multiplied, so that the others get
/// zeros even for an infinite or NaN `g` (PyTorch's `evenly_distribute_backward`). A NaN result
/// is shared by every NaN of `x`.
struct ExtremumBackward {
    name: &'static str,
    x: SavedId,
    result: SavedId,
}

impl Node for ExtremumBackward {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let x = ctx.unpack(self.x);
        let result = ctx.unpack(self.result);
        let both_nan = x.not_equal(&x).logical_and(result.not_equal(&result));
        let mask = x.equal(&result).logical_or(&both_nan);
        let share = g.div(mask.sum());
        vec![Some(g.new_zeros(x.shape()).masked_fill(&mask, &share))]
    }
}

/// The values of `x.max_dim(dim, keepdim)` (`MaxBackward0`) and `x.min_dim(dim, keepdim)`
/// (`MinBackward0`): `g` at the returned indices, zeros elsewhere.
struct ExtremumDimBackward {
    name: &'static str,
    self_shape: Shape,
    dim: usize,
    unreduced: Option<Shape>,
    indices: SavedId,
}

impl Node for ExtremumDimBackward {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let indices = ctx.unpack(self.indices);
        let (g, indices) = match &self.unreduced {
            Some(kept) => (g.reshape(kept), indices.reshape(kept)),
            None => (g, indices),
        };
        let zeros = g.new_zeros(&self.self_shape);
        vec![Some(zeros.scatter_add(self.dim as i64, &indices, &g))]
    }
}

/// The dimensions of a variance taken over some dimensions, not the whole tensor.
struct VarDims {
    dims: Vec<i64>,
    /// The number of elements each variance is taken over.
    count: usize,
    unreduced: Option<Shape>,
}

/// `x.var(..)` (`VarBackward0`) and `x.std(..)` (`StdBackward0`), as PyTorch's `var_backward`
/// and `std_backward` compute them. The standard deviation first turns `g` into the gradient of
/// the variance, `g / (2 * result)`, zero where the result is zero. Over the whole tensor, with
/// `n - correction` degrees of freedom, that gradient becomes `2 / dof * g * (x - mean)`, or for
/// `dof <= 0`, `g * Inf`, NaN where `x` equals the mean. Over some dimensions, it is always
/// `2 / (n - correction) * g * (x - mean)`: infinite for `n == correction`, finite for
/// `n < correction`.
struct VarBackward {
    name: &'static str,
    x: SavedId,
    /// The saved result of `std`.
    result: Option<SavedId>,
    reduced: Option<VarDims>,
    correction: i64,
}

impl Node for VarBackward {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let mut g = only(grads);
        let x = ctx.unpack(self.x);
        if let Some(result) = self.result.map(|id| ctx.unpack(id)) {
            g = g.div(result.mul(2.0)).masked_fill(&result.equal(0.0), 0.0);
        }
        let correction = self.correction as f64;
        let Some(reduced) = &self.reduced else {
            let mean = x.mean();
            let dof = x.num_elements() as f64 - correction;
            if dof <= 0.0 {
                let infinite =
                    crate::full_like(&x, f64::INFINITY).masked_fill(&x.equal(&mean), f64::NAN);
                return vec![Some(g.mul(infinite))];
            }
            return vec![Some(g.mul(2.0 / dof).mul(x.sub(&mean)))];
        };
        if let Some(kept) = &reduced.unreduced {
            g = g.reshape(kept);
        }
        let mean = x.mean_dim(&reduced.dims[..], true);
        let scale = 2.0 / (reduced.count as f64 - correction);
        vec![Some(g.mul(scale).mul(x.sub(&mean)))]
    }
}
