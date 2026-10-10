//! Operations that return views sharing the storage of the input.

use super::{Layout, Tensor};
use crate::shape::{
    Dims, IntoShape, IntoShapeSpec, compute_view_strides, dims_mask, infer_size, normalize_dim,
};
use crate::{Shape, Strides};

impl Tensor {
    /// A view with a different shape (one dimension may be `-1`). The memory layout must allow
    /// it without copying; [`reshape`](Tensor::reshape) copies when needed.
    pub fn view(&self, shape: impl IntoShapeSpec) -> Tensor {
        let l = self.layout();
        let shape = infer_size(&shape.into_shape_spec(), l.num_elements());
        let Some(strides) = compute_view_strides(&l.shape, &l.strides, &shape) else {
            panic!(
                "view size is not compatible with input tensor's size and stride (at least one \
                 dimension spans across two contiguous subspaces). Use .reshape(...) instead."
            );
        };
        self.with_layout(Layout {
            shape,
            strides,
            offset: l.offset,
        })
    }

    /// A tensor with a different shape (one dimension may be `-1`): a view when the memory
    /// layout allows it, otherwise a contiguous copy.
    pub fn reshape(&self, shape: impl IntoShapeSpec) -> Tensor {
        let l = self.layout();
        let shape = infer_size(&shape.into_shape_spec(), l.num_elements());
        match compute_view_strides(&l.shape, &l.strides, &shape) {
            Some(strides) => self.with_layout(Layout {
                shape,
                strides,
                offset: l.offset,
            }),
            None => self.deep_clone().view(shape),
        }
    }

    /// Reorders the dimensions: dimension `i` of the result is dimension `dims[i]` of `self`.
    pub fn permute(&self, dims: impl Dims) -> Tensor {
        let l = self.layout();
        let dims = dims.into_dims();
        let n = l.shape.len();
        if dims.len() != n {
            panic!(
                "permute(): number of dimensions in the tensor input does not match the length of \
                 the desired ordering of dimensions i.e. input.dim() = {n} is not equal to \
                 len(dims) = {}",
                dims.len()
            );
        }
        let mut seen = vec![false; n];
        let mut shape = vec![0; n];
        let mut strides = vec![0; n];
        for (i, &d) in dims.iter().enumerate() {
            let d = normalize_dim(d, n);
            if seen[d] {
                panic!("permute(): duplicate dims are not allowed.");
            }
            seen[d] = true;
            shape[i] = l.shape[d];
            strides[i] = l.strides[d];
        }
        self.with_layout(Layout {
            shape: Shape::new(&shape),
            strides: Strides::new(&strides),
            offset: l.offset,
        })
    }

    /// Swaps two dimensions.
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Tensor {
        let mut l = self.layout();
        let n = l.shape.len();
        if n == 0 {
            normalize_dim(dim0, n);
            normalize_dim(dim1, n);
            return self.with_layout(l);
        }
        let (a, b) = (normalize_dim(dim0, n), normalize_dim(dim1, n));
        l.shape.swap(a, b);
        l.strides.swap(a, b);
        self.with_layout(l)
    }

    /// A view broadcast to `shape`: size-1 dimensions may grow and new leading dimensions may be
    /// added; `-1` keeps a size. No memory is copied.
    pub fn expand(&self, shape: impl IntoShapeSpec) -> Tensor {
        let l = self.layout();
        let spec = shape.into_shape_spec();
        let (n, m) = (spec.len(), l.shape.len());
        if n < m {
            panic!(
                "expand: the number of sizes provided ({n}) must be greater or equal to the \
                 number of dimensions in the tensor ({m})"
            );
        }
        let mut target = vec![0; n];
        for i in 0..n {
            target[i] = if spec[i] == -1 {
                if i < n - m {
                    panic!(
                        "The expanded size of the tensor (-1) isn't allowed in a leading, \
                         non-existing dimension {i}"
                    );
                }
                l.shape[i - (n - m)]
            } else if spec[i] < 0 {
                panic!("invalid shape dimension {}", spec[i]);
            } else {
                spec[i] as usize
            };
        }
        self.with_layout(l.expand_to(&Shape::new(&target)))
    }

    /// [`expand`](Tensor::expand) to the shape of `other`.
    pub fn expand_as(&self, other: &Tensor) -> Tensor {
        self.expand(other.shape())
    }

    /// Removes all dimensions of size 1.
    pub fn squeeze(&self) -> Tensor {
        let l = self.layout();
        let keep: Vec<usize> = (0..l.shape.len()).filter(|&d| l.shape[d] != 1).collect();
        self.keep_dims(&l, &keep)
    }

    /// Removes the given dimensions if they have size 1 (others are left alone, like PyTorch).
    pub fn squeeze_dim(&self, dims: impl Dims) -> Tensor {
        let l = self.layout();
        let mask = dims_mask(&dims.into_dims(), l.shape.len());
        let keep: Vec<usize> = (0..l.shape.len())
            .filter(|&d| !(mask[d] && l.shape[d] == 1))
            .collect();
        self.keep_dims(&l, &keep)
    }

    fn keep_dims(&self, l: &Layout, keep: &[usize]) -> Tensor {
        self.with_layout(Layout {
            shape: keep.iter().map(|&d| l.shape[d]).collect(),
            strides: keep.iter().map(|&d| l.strides[d]).collect(),
            offset: l.offset,
        })
    }

    /// Inserts a dimension of size 1 at position `dim` (`-1` appends at the end).
    pub fn unsqueeze(&self, dim: i64) -> Tensor {
        let mut l = self.layout();
        let d = normalize_dim(dim, l.shape.len() + 1);
        // The stride of a size-1 dimension is never used to address memory; fall back to 1 in
        // the unlikely case that PyTorch's choice doesn't fit.
        let stride = if d < l.shape.len() {
            l.shape[d].checked_mul(l.strides[d]).unwrap_or(1)
        } else {
            1
        };
        l.shape.insert(d, 1);
        l.strides.insert(d, stride);
        self.with_layout(l)
    }

    /// The `length` elements along `dim` starting at `start` (negative values count from the
    /// end).
    pub fn narrow(&self, dim: i64, start: i64, length: usize) -> Tensor {
        let mut l = self.layout();
        if l.shape.is_empty() {
            panic!("narrow() cannot be applied to a 0-dim tensor.");
        }
        let d = normalize_dim(dim, l.shape.len());
        let size = l.shape[d] as i64;
        if start < -size || start > size {
            panic!(
                "start out of range (expected to be in range of [{}, {size}], but got {start})",
                -size
            );
        }
        let start = if start < 0 { start + size } else { start } as usize;
        if start
            .checked_add(length)
            .is_none_or(|end| end > size as usize)
        {
            panic!("start ({start}) + length ({length}) exceeds dimension size ({size}).");
        }
        l.offset = l.offset_at(d, start);
        l.shape[d] = length;
        self.with_layout(l)
    }

    /// The slice at `index` along `dim`, with that dimension removed.
    pub fn select(&self, dim: i64, index: i64) -> Tensor {
        let mut l = self.layout();
        if l.shape.is_empty() {
            panic!("select() cannot be applied to a 0-dim tensor.");
        }
        let d = normalize_dim(dim, l.shape.len());
        let size = l.shape[d] as i64;
        if index < -size || index >= size {
            panic!(
                "select(): index {index} out of range for tensor of size {:?} at dimension {d}",
                l.shape
            );
        }
        let index = if index < 0 { index + size } else { index } as usize;
        l.offset = l.offset_at(d, index);
        l.shape.remove(d);
        l.strides.remove(d);
        self.with_layout(l)
    }

    /// A Python-style slice `start:end:step` along `dim`: bounds may be negative and are clamped
    /// to the size.
    pub(crate) fn slice(
        &self,
        dim: usize,
        start: Option<i64>,
        end: Option<i64>,
        step: i64,
    ) -> Tensor {
        if step <= 0 {
            panic!("slice step must be positive");
        }
        let mut l = self.layout();
        let size = l.shape[dim] as i64;
        let clamp = |v: i64| (if v < 0 { v + size } else { v }).clamp(0, size);
        let start = clamp(start.unwrap_or(0));
        let end = clamp(end.unwrap_or(size));
        let len = if end > start {
            (end - start - 1) / step + 1
        } else {
            0
        };
        l.offset = l.offset_at(dim, start as usize);
        l.shape[dim] = len as usize;
        // With fewer than two elements the stride is never used, and a huge step may not fit.
        if len > 1 {
            l.strides[dim] = l.strides[dim]
                .checked_mul(step as usize)
                .expect("slice stride overflows usize");
        }
        self.with_layout(l)
    }

    /// Merges dimensions `start_dim..=end_dim` into one.
    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Tensor {
        let shape = self.shape();
        if shape.is_empty() {
            normalize_dim(start_dim, 0);
            normalize_dim(end_dim, 0);
            return self.reshape([1]);
        }
        let (s, e) = (
            normalize_dim(start_dim, shape.len()),
            normalize_dim(end_dim, shape.len()),
        );
        if s > e {
            panic!("flatten() has invalid args: start_dim cannot come after end_dim");
        }
        let mut spec: Vec<i64> = shape[..s].iter().map(|&d| d as i64).collect();
        spec.push(shape[s..=e].iter().product::<usize>() as i64);
        spec.extend(shape[e + 1..].iter().map(|&d| d as i64));
        self.reshape(spec)
    }

    /// A view with an explicit shape, strides and storage offset (in elements).
    ///
    /// # Panics
    ///
    /// Panics if the view would reach outside the storage.
    pub fn as_strided(
        &self,
        size: impl IntoShape,
        stride: impl IntoShape,
        storage_offset: usize,
    ) -> Tensor {
        let (shape, stride) = (size.into_shape(), stride.into_shape());
        if shape.len() != stride.len() {
            panic!("mismatch in length of strides and shape");
        }
        // `with_layout` checks that the view lies inside the storage.
        self.with_layout(Layout {
            shape,
            strides: Strides::new(&stride),
            offset: storage_offset,
        })
    }

    /// `self` if it is already contiguous, otherwise a contiguous copy.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            self.clone()
        } else {
            self.deep_clone()
        }
    }
}
