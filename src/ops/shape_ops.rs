//! Operations that rearrange or combine tensors.

use std::borrow::Borrow;

use super::common_device;
use crate::backend::cpu;
use crate::dtype::ResultType;
use crate::shape::{Dims, IntoShape, dims_mask, normalize_dim};
use crate::{Device, Shape, Tensor};

/// Concatenates tensors along `dim` (`torch.cat`). All tensors must have the same shape except
/// along `dim`; the dtype is promoted across all of them. One-dimensional empty tensors are
/// skipped, like in PyTorch.
///
/// ```
/// let a = homura::tensor([[1, 2]]);
/// let b = homura::tensor([[3, 4]]);
/// assert_eq!(homura::concat(&[&a, &b], 0).shape(), [2, 2]);
/// ```
pub fn concat<T: Borrow<Tensor>>(tensors: &[T], dim: i64) -> Tensor {
    if tensors.is_empty() {
        panic!("concat(): expected a non-empty list of Tensors");
    }
    let all: Vec<_> = tensors.iter().map(|t| t.borrow().data()).collect();
    let device = common_device(&all);
    let mut rt = ResultType::new();
    for t in &all {
        rt.tensor(t.dtype, t.layout.shape.len());
    }
    let dtype = rt.result();
    let parts: Vec<_> = all.iter().filter(|t| t.layout.shape != [0]).collect();
    let Some(first) = parts.first() else {
        return Tensor::empty_on(Shape::new(&[0]), dtype, device);
    };
    let nd = first.layout.shape.len();
    if nd == 0 {
        panic!("zero-dimensional tensor (at position 0) cannot be concatenated");
    }
    let dim = normalize_dim(dim, nd);
    let mut shape = first.layout.shape.clone();
    shape[dim] = 0;
    for (i, t) in parts.iter().enumerate() {
        let s = &t.layout.shape;
        if s.len() != nd {
            panic!(
                "Tensors must have same number of dimensions: got {nd} and {}",
                s.len()
            );
        }
        for d in 0..nd {
            if d != dim && s[d] != first.layout.shape[d] {
                panic!(
                    "Sizes of tensors must match except in dimension {dim}. Expected size {} but \
                     got size {} for tensor number {i} in the list.",
                    first.layout.shape[d], s[d]
                );
            }
        }
        shape[dim] += s[dim];
    }
    let out = Tensor::empty_on(shape, dtype, device);
    if device == Device::Cpu {
        let mut offset = 0;
        for t in parts {
            let n = t.layout.shape[dim];
            let dst = out.narrow(dim as i64, offset as i64, n).data();
            cpu::copy(&dst, t);
            offset += n;
        }
    }
    out
}

/// Stacks tensors of the same shape along a new dimension `dim` (`torch.stack`).
pub fn stack<T: Borrow<Tensor>>(tensors: &[T], dim: i64) -> Tensor {
    if tensors.is_empty() {
        panic!("stack expects a non-empty TensorList");
    }
    let first = tensors[0].borrow().shape();
    for (i, t) in tensors.iter().enumerate() {
        let s = t.borrow().shape();
        if s != first {
            panic!(
                "stack expects each tensor to be equal size, but got {first:?} at entry 0 and {s:?} at entry {i}"
            );
        }
    }
    let dim = normalize_dim(dim, first.len() + 1) as i64;
    let parts: Vec<Tensor> = tensors.iter().map(|t| t.borrow().unsqueeze(dim)).collect();
    concat(&parts, dim)
}

impl Tensor {
    /// Splits into views of `split_size` elements along `dim`; the last one may be smaller.
    pub fn split(&self, split_size: usize, dim: i64) -> Vec<Tensor> {
        let d = normalize_dim(dim, self.num_dims());
        let size = self.shape()[d];
        if split_size == 0 {
            if size == 0 {
                return vec![self.clone()];
            }
            panic!(
                "split_size can only be 0 if dimension size is 0, but got dimension size of {size}"
            );
        }
        let count = size.div_ceil(split_size).max(1);
        (0..count)
            .map(|i| {
                let start = i * split_size;
                self.narrow(dim, start as i64, split_size.min(size - start))
            })
            .collect()
    }

    /// Splits into views with the given sizes along `dim`; the sizes must add up to the size of
    /// `dim`.
    pub fn split_with_sizes(&self, split_sizes: &[usize], dim: i64) -> Vec<Tensor> {
        let d = normalize_dim(dim, self.num_dims());
        let size = self.shape()[d];
        let total: usize = split_sizes.iter().sum();
        if total != size {
            panic!(
                "split_with_sizes expects split_sizes to sum exactly to {size} (input tensor's \
                 size at dimension {dim}), but got split_sizes={split_sizes:?}"
            );
        }
        let mut start = 0;
        split_sizes
            .iter()
            .map(|&n| {
                let t = self.narrow(dim, start as i64, n);
                start += n;
                t
            })
            .collect()
    }

    /// Splits into `chunks` views of (almost) equal size along `dim`. Like PyTorch, fewer chunks
    /// may be returned when the size is not divisible.
    pub fn chunk(&self, chunks: usize, dim: i64) -> Vec<Tensor> {
        if chunks == 0 {
            panic!("chunk expects `chunks` to be greater than 0, got: 0");
        }
        let size = self.size(dim);
        self.split(size.div_ceil(chunks), dim)
    }

    fn triangle(&self, diagonal: i64, upper: bool) -> Tensor {
        let src = self.data();
        if src.layout.shape.len() < 2 {
            panic!(
                "{}: input tensor must have at least 2 dimensions",
                if upper { "triu" } else { "tril" }
            );
        }
        let out = Tensor::empty_on(src.layout.shape.clone(), src.dtype, src.device());
        if src.device() == Device::Cpu {
            cpu::triangle(&out.data(), &src, diagonal, upper);
        }
        out
    }

    /// Keeps elements on and above the `diagonal`-th diagonal of the last two dimensions and
    /// zeroes the rest (`diagonal = 0` is the main diagonal, positive values are above it).
    pub fn triu(&self, diagonal: i64) -> Tensor {
        self.triangle(diagonal, true)
    }

    /// Keeps elements on and below the `diagonal`-th diagonal of the last two dimensions and
    /// zeroes the rest.
    pub fn tril(&self, diagonal: i64) -> Tensor {
        self.triangle(diagonal, false)
    }

    /// Tiles the tensor `repeats[i]` times along each dimension. `repeats` may have more entries
    /// than `self` has dimensions; leading dimensions are added.
    pub fn repeat(&self, repeats: impl IntoShape) -> Tensor {
        let repeats = repeats.into_shape();
        let shape = self.shape();
        if repeats.len() < shape.len() {
            panic!(
                "Number of dimensions of repeat dims can not be smaller than number of dimensions \
                 of tensor"
            );
        }
        let padded: Vec<usize> = std::iter::repeat_n(1, repeats.len() - shape.len())
            .chain(shape.iter().copied())
            .collect();
        // View as [1, s0, 1, s1, ...], expand to [r0, s0, r1, s1, ...] and merge each pair.
        let mut interleaved = vec![];
        let mut expanded = vec![];
        let mut result = vec![];
        for (&r, &s) in repeats.iter().zip(&padded) {
            interleaved.extend([1, s as i64]);
            expanded.extend([r as i64, s as i64]);
            result.push((r * s) as i64);
        }
        self.reshape(interleaved).expand(expanded).reshape(result)
    }

    /// Reverses the order of elements along `dims`, returning a copy.
    pub fn flip(&self, dims: impl Dims) -> Tensor {
        let src = self.data();
        let dims = dims.into_dims();
        let mask = if dims.is_empty() {
            vec![false; src.layout.shape.len()]
        } else {
            dims_mask(&dims, src.layout.shape.len())
        };
        let out = Tensor::empty_on(src.layout.shape.clone(), src.dtype, src.device());
        if src.device() == Device::Cpu {
            cpu::flip(&out.data(), &src, &mask);
        }
        out
    }
}
