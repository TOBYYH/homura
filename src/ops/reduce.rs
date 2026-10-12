//! Reductions.

use super::{ReduceOp, reshape_data};
use crate::autograd::record_unimplemented;
use crate::backend::cpu;
use crate::shape::{Dims, IntoShape, dims_mask, normalize_dim};
use crate::{DType, Device, Shape, Tensor};

/// The shape left after reducing the dimensions in `mask`.
fn reduced_shape(shape: &Shape, mask: &[bool], keepdim: bool) -> Shape {
    let mut out = vec![];
    for (d, &n) in shape.iter().enumerate() {
        if !mask[d] {
            out.push(n);
        } else if keepdim {
            out.push(1);
        }
    }
    Shape::new(&out)
}

fn reduce_op(op: ReduceOp, x: &Tensor, dims: &[i64], keepdim: bool, name: &str) -> Tensor {
    let src = x.data();
    let dtype = src.dtype;
    let mask = dims_mask(dims, src.layout.shape.len());
    let out_dtype = match op {
        ReduceOp::Sum | ReduceOp::Prod if !dtype.is_floating_point() => DType::I64,
        ReduceOp::Mean if !dtype.is_floating_point() => panic!(
            "{name}(): could not infer output dtype. Input dtype must be either a floating point \
             or complex dtype. Got: {dtype}"
        ),
        ReduceOp::All | ReduceOp::Any if dtype != DType::U8 => DType::Bool,
        _ => dtype,
    };
    let reduces_empty = src
        .layout
        .shape
        .iter()
        .zip(&mask)
        .any(|(&n, &m)| m && n == 0);
    if matches!(op, ReduceOp::Max | ReduceOp::Min) && reduces_empty {
        panic!(
            "{name}(): Expected reduction dim to be specified for input.numel() == 0. Specify \
             the reduction dim with the 'dim' argument."
        );
    }
    let out = Tensor::empty_on(
        reduced_shape(&src.layout.shape, &mask, keepdim),
        out_dtype,
        src.device(),
    );
    if src.device() == Device::Cpu {
        cpu::reduce(op, &out.data(), &src, &mask);
    }
    out
}

/// Index of the maximum (or minimum) along `dim`, or over all elements when `dim` is `None`.
fn arg_op(
    x: &Tensor,
    dim: Option<i64>,
    keepdim: bool,
    is_max: bool,
    values: bool,
    name: &str,
) -> (Option<Tensor>, Tensor) {
    let src = x.data();
    let (src, dim, keepdim) = match dim {
        Some(d) => (src, d, keepdim),
        None => {
            let n = src.layout.num_elements();
            (reshape_data(&src, Shape::new(&[n])), 0, false)
        }
    };
    let nd = src.layout.shape.len();
    let dim = normalize_dim(dim, nd);
    let size = if nd == 0 { 1 } else { src.layout.shape[dim] };
    if size == 0 {
        panic!("{name}(): Expected reduction dim {dim} to have non-zero size.");
    }
    let mut mask = vec![false; nd];
    if nd > 0 {
        mask[dim] = true;
    }
    let shape = reduced_shape(&src.layout.shape, &mask, keepdim);
    let device = src.device();
    let vals = values.then(|| Tensor::empty_on(shape.clone(), src.dtype, device));
    let indices = Tensor::empty_on(shape, DType::I64, device);
    if device == Device::Cpu {
        let vd = vals.as_ref().map(Tensor::data);
        cpu::arg_reduce(vd.as_ref(), &indices.data(), &src, dim, is_max);
    }
    (vals, indices)
}

fn var_op(x: &Tensor, dims: &[i64], correction: i64, keepdim: bool, take_sqrt: bool) -> Tensor {
    let src = x.data();
    if !src.dtype.is_floating_point() {
        panic!(
            "std and var only support floating point dtypes, got {}",
            src.dtype
        );
    }
    let mask = dims_mask(dims, src.layout.shape.len());
    let out = Tensor::empty_on(
        reduced_shape(&src.layout.shape, &mask, keepdim),
        src.dtype,
        src.device(),
    );
    if src.device() == Device::Cpu {
        cpu::var(&out.data(), &src, &mask, correction as f64, take_sqrt);
    }
    out
}

impl Tensor {
    /// Sum of all elements. Integer and bool inputs give `I64`.
    pub fn sum(&self) -> Tensor {
        let out = reduce_op(ReduceOp::Sum, self, &[], false, "sum");
        crate::autograd::formulas::sum(self, &out);
        out
    }

    /// Sum over `dims` (all dimensions if empty).
    pub fn sum_dim(&self, dims: impl Dims, keepdim: bool) -> Tensor {
        let out = reduce_op(ReduceOp::Sum, self, &dims.into_dims(), keepdim, "sum");
        record_unimplemented("sum_dim", &[Some(self)], &out);
        out
    }

    /// Sums `self` down to `size`, the reverse of broadcasting a tensor of shape `size` to the
    /// shape of `self` (`torch.Tensor.sum_to_size`): leading dimensions that `size` lacks, and
    /// dimensions where `size` has 1 and `self` doesn't, are summed. Returns `self` itself when
    /// nothing is summed; otherwise integer and bool inputs give `I64`, like
    /// [`sum`](Tensor::sum).
    ///
    /// ```
    /// let x = homura::ones([2, 3]);
    /// assert_eq!(x.sum_to_size([3]).to_vec::<f32>(), [2.0, 2.0, 2.0]);
    /// assert_eq!(x.sum_to_size([2, 1]).to_vec::<f32>(), [3.0, 3.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `size` doesn't broadcast to the shape of `self`.
    pub fn sum_to_size(&self, size: impl IntoShape) -> Tensor {
        let size = size.into_shape();
        let shape = self.shape();
        let lead = shape.len().checked_sub(size.len());
        let expandable = lead.is_some_and(|lead| {
            size.iter()
                .zip(&shape[lead..])
                .all(|(&s, &n)| s == n || s == 1)
        });
        let Some(lead) = lead.filter(|_| expandable) else {
            panic!("size {{{size:?}}} is not expandable to size {{{shape:?}}}.");
        };
        let dims: Vec<i64> = (0..shape.len())
            .filter(|&d| d < lead || (size[d - lead] == 1 && shape[d] != 1))
            .map(|d| d as i64)
            .collect();
        if dims.is_empty() {
            return self.clone();
        }
        // Summed with kept dimensions, so the result has the elements of `size` in order.
        let summed = reduce_op(ReduceOp::Sum, self, &dims, true, "sum");
        let out = Tensor::from_data(reshape_data(&summed.data(), size));
        crate::autograd::formulas::sum_to_size(self, &out);
        out
    }

    /// Mean of all elements. Requires a floating point input.
    pub fn mean(&self) -> Tensor {
        let out = reduce_op(ReduceOp::Mean, self, &[], false, "mean");
        record_unimplemented("mean", &[Some(self)], &out);
        out
    }

    /// Mean over `dims` (all dimensions if empty).
    pub fn mean_dim(&self, dims: impl Dims, keepdim: bool) -> Tensor {
        let out = reduce_op(ReduceOp::Mean, self, &dims.into_dims(), keepdim, "mean");
        record_unimplemented("mean_dim", &[Some(self)], &out);
        out
    }

    /// Product of all elements. Integer and bool inputs give `I64`.
    pub fn prod(&self) -> Tensor {
        let out = reduce_op(ReduceOp::Prod, self, &[], false, "prod");
        record_unimplemented("prod", &[Some(self)], &out);
        out
    }

    /// Product along `dim`.
    pub fn prod_dim(&self, dim: i64, keepdim: bool) -> Tensor {
        let out = reduce_op(ReduceOp::Prod, self, &[dim], keepdim, "prod");
        record_unimplemented("prod_dim", &[Some(self)], &out);
        out
    }

    /// Largest element (NaN if there is any).
    pub fn max(&self) -> Tensor {
        let out = reduce_op(ReduceOp::Max, self, &[], false, "max");
        record_unimplemented("max", &[Some(self)], &out);
        out
    }

    /// Largest values along `dim` and their indices (the first one on ties).
    pub fn max_dim(&self, dim: i64, keepdim: bool) -> (Tensor, Tensor) {
        let (v, i) = arg_op(self, Some(dim), keepdim, true, true, "max");
        let v = v.unwrap();
        record_unimplemented("max_dim", &[Some(self)], &v);
        (v, i)
    }

    /// Smallest element (NaN if there is any).
    pub fn min(&self) -> Tensor {
        let out = reduce_op(ReduceOp::Min, self, &[], false, "min");
        record_unimplemented("min", &[Some(self)], &out);
        out
    }

    /// Smallest values along `dim` and their indices (the first one on ties).
    pub fn min_dim(&self, dim: i64, keepdim: bool) -> (Tensor, Tensor) {
        let (v, i) = arg_op(self, Some(dim), keepdim, false, true, "min");
        let v = v.unwrap();
        record_unimplemented("min_dim", &[Some(self)], &v);
        (v, i)
    }

    /// Index of the largest element in the flattened tensor.
    pub fn argmax(&self) -> Tensor {
        arg_op(self, None, false, true, false, "argmax").1
    }

    /// Indices of the largest values along `dim` (the first one on ties).
    pub fn argmax_dim(&self, dim: i64, keepdim: bool) -> Tensor {
        arg_op(self, Some(dim), keepdim, true, false, "argmax").1
    }

    /// Index of the smallest element in the flattened tensor.
    pub fn argmin(&self) -> Tensor {
        arg_op(self, None, false, false, false, "argmin").1
    }

    /// Indices of the smallest values along `dim` (the first one on ties).
    pub fn argmin_dim(&self, dim: i64, keepdim: bool) -> Tensor {
        arg_op(self, Some(dim), keepdim, false, false, "argmin").1
    }

    /// Variance of all elements, dividing by `N - correction` (`correction = 1` is Bessel's
    /// correction, PyTorch's default).
    pub fn var(&self, correction: i64) -> Tensor {
        let out = var_op(self, &[], correction, false, false);
        record_unimplemented("var", &[Some(self)], &out);
        out
    }

    /// Variance over `dims` (all dimensions if empty).
    pub fn var_dim(&self, dims: impl Dims, correction: i64, keepdim: bool) -> Tensor {
        let out = var_op(self, &dims.into_dims(), correction, keepdim, false);
        record_unimplemented("var_dim", &[Some(self)], &out);
        out
    }

    /// Standard deviation of all elements; see [`var`](Tensor::var).
    pub fn std(&self, correction: i64) -> Tensor {
        let out = var_op(self, &[], correction, false, true);
        record_unimplemented("std", &[Some(self)], &out);
        out
    }

    /// Standard deviation over `dims` (all dimensions if empty).
    pub fn std_dim(&self, dims: impl Dims, correction: i64, keepdim: bool) -> Tensor {
        let out = var_op(self, &dims.into_dims(), correction, keepdim, true);
        record_unimplemented("std_dim", &[Some(self)], &out);
        out
    }

    /// Whether all elements are nonzero, as a zero-dimensional `Bool` tensor (`U8` for `U8`
    /// inputs, like PyTorch).
    pub fn all(&self) -> Tensor {
        reduce_op(ReduceOp::All, self, &[], false, "all")
    }

    /// Whether all elements along `dims` are nonzero.
    pub fn all_dim(&self, dims: impl Dims, keepdim: bool) -> Tensor {
        reduce_op(ReduceOp::All, self, &dims.into_dims(), keepdim, "all")
    }

    /// Whether any element is nonzero.
    pub fn any(&self) -> Tensor {
        reduce_op(ReduceOp::Any, self, &[], false, "any")
    }

    /// Whether any element along `dims` is nonzero.
    pub fn any_dim(&self, dims: impl Dims, keepdim: bool) -> Tensor {
        reduce_op(ReduceOp::Any, self, &dims.into_dims(), keepdim, "any")
    }
}
