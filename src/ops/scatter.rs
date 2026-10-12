//! Gathering and scattering: operations that read or write elements at positions given by an
//! index tensor or a slice. Each is the backward of another indexing operation: `index_add` of
//! `index_select`, `gather` and `scatter_add` of each other, `select_scatter` of `select` and
//! `slice_scatter` of slicing.

use super::{clone_data, common_device};
use crate::autograd::formulas;
use crate::backend::cpu;
use crate::shape::normalize_dim;
use crate::tensor::{Layout, TensorData, select_position, slice_bounds};
use crate::{DType, Device, Shape, Strides, Tensor};

/// Panics unless `index` is an `i32` or `i64` tensor, the index dtypes of PyTorch's indexing
/// operations.
fn check_index_dtype(op: &str, index: &TensorData) {
    if !matches!(index.dtype, DType::I32 | DType::I64) {
        panic!(
            "{op}(): Expected dtype i32 or i64 for index, got {}",
            index.dtype
        );
    }
}

/// The values of `index` in row-major order, checked to be positions along a dimension `dim` of
/// `size`. Negative values are out of range, as in PyTorch's `index_add`, `gather` and
/// `scatter_add`. The meta device has no values to check: the result is empty there.
fn positions(index: &TensorData, dim: usize, size: usize) -> Vec<usize> {
    if index.device() != Device::Cpu {
        return vec![];
    }
    cpu::read_host::<i64>(index)
        .into_iter()
        .map(|i| {
            if !(0..size as i64).contains(&i) {
                panic!("index {i} is out of bounds for dimension {dim} with size {size}");
            }
            i as usize
        })
        .collect()
}

/// `data` with a zero-dimensional layout seen as a one-dimensional one of one element, as
/// PyTorch's `gather` and `scatter_add` treat scalars.
fn at_least_1d(data: &TensorData) -> TensorData {
    if !data.layout.shape.is_empty() {
        return data.clone();
    }
    TensorData {
        layout: Layout {
            shape: Shape::new(&[1]),
            strides: Strides::new(&[1]),
            offset: data.layout.offset,
        },
        ..data.clone()
    }
}

impl Tensor {
    /// Adds the slices of `source` along `dim` into the slices of `self` that `index` picks
    /// (`torch.index_add`): slice `i` of `source` is added to slice `index[i]` of `self`, and
    /// repeated positions add up.
    ///
    /// `index` is a vector (or a zero-dimensional tensor) of `i32` or `i64` positions, none
    /// negative. `source` has the dtype of `self` and its shape, except along `dim`, where its
    /// size is the number of positions. Each element of the result is summed in `f64` for
    /// floating point dtypes, like the reductions, and rounded once (PyTorch rounds after every
    /// addition for one-dimensional `f16` and `bf16` tensors). Integer sums wrap around, and
    /// bool sums are a logical or.
    ///
    /// ```
    /// let x = homura::zeros([3, 2]);
    /// let source = homura::tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    /// let y = x.index_add(0, &homura::tensor([2i64, 0, 2]), &source);
    /// assert_eq!(y.to_vec::<f32>(), [3.0, 4.0, 0.0, 0.0, 6.0, 8.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero-dimensional, like [`index_select`](Tensor::index_select), if the
    /// shapes or dtypes don't match as described, or if a position is out of range.
    pub fn index_add(&self, dim: i64, index: &Tensor, source: &Tensor) -> Tensor {
        let (data, idx, src) = (self.data(), index.data(), source.data());
        let device = common_device([&data, &idx, &src]);
        if idx.layout.shape.len() > 1 {
            panic!("index_add(): Index is supposed to be a vector");
        }
        check_index_dtype("index_add", &idx);
        let (shape, src_shape) = (&data.layout.shape, &src.layout.shape);
        let nd = shape.len();
        if nd == 0 {
            panic!("index_add() cannot be applied to a 0-dim tensor.");
        }
        let d = normalize_dim(dim, nd);
        if data.dtype != src.dtype {
            panic!(
                "index_add(): self ({}) and source ({}) must have the same scalar type",
                data.dtype, src.dtype
            );
        }
        let count = idx.layout.num_elements();
        let src_size = match src_shape.len() {
            0 => 1,
            n if d < n => src_shape[d],
            n => panic!(
                "index_add(): Indexing dim {d} is out of bounds of the source tensor with dim {n}"
            ),
        };
        if count != src_size {
            panic!(
                "index_add(): Number of indices ({count}) should be equal to source.size(dim): \
                 ({src_size}), for dim: {d}"
            );
        }
        if src_shape.len() != nd || (0..nd).any(|k| k != d && src_shape[k] != shape[k]) {
            panic!(
                "source tensor shape must match self tensor shape, excluding the specified \
                 dimension. Got self.shape = {shape:?} source.shape = {src_shape:?}"
            );
        }
        let positions = positions(&idx, d, shape[d]);
        let out = Tensor::from_data(clone_data(&data));
        if device == Device::Cpu {
            cpu::index_add(&out.data(), d, &positions, &src);
        }
        formulas::index_add(self, d, index, source, &out);
        out
    }

    /// In-place [`index_add`](Tensor::index_add).
    pub fn index_add_(&self, dim: i64, index: &Tensor, source: &Tensor) -> &Self {
        crate::autograd::check_inplace(self, &[Some(source)]);
        let result = self.index_add(dim, index, source);
        self.assign_result(&result);
        self
    }

    /// Picks the elements along `dim` at the positions in `index` (`torch.gather`): for
    /// `dim = 1`, `out[i][j][k] = self[i][index[i][j][k]][k]`.
    ///
    /// `index` is an `i32` or `i64` tensor with as many dimensions as `self` and no larger than
    /// `self` in the other dimensions; its values are positions along `dim`, none negative. The
    /// result has the shape of `index` and the dtype of `self`. As in PyTorch, zero-dimensional
    /// tensors count as one-dimensional ones of one element.
    ///
    /// ```
    /// let x = homura::tensor([[1.0, 2.0], [3.0, 4.0]]);
    /// let y = x.gather(1, &homura::tensor([[1i64, 1], [0, 1]]));
    /// assert_eq!(y.to_vec::<f32>(), [2.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the shapes or dtypes don't match as described, or if a position is out of
    /// range.
    pub fn gather(&self, dim: i64, index: &Tensor) -> Tensor {
        let (data, idx) = (self.data(), index.data());
        let device = common_device([&data, &idx]);
        check_index_dtype("gather", &idx);
        let d = normalize_dim(dim, data.layout.shape.len());
        let (src, index_1d) = (at_least_1d(&data), at_least_1d(&idx));
        let (shape, index_shape) = (&src.layout.shape, &index_1d.layout.shape);
        if index_shape.len() != shape.len() {
            panic!("Index tensor must have the same number of dimensions as input tensor");
        }
        if let Some(k) = (0..shape.len()).find(|&k| k != d && index_shape[k] > shape[k]) {
            panic!(
                "Size does not match at dimension {k} expected index {:?} to be no larger than \
                 self {:?} apart from dimension {d}",
                idx.layout.shape, data.layout.shape
            );
        }
        let positions = positions(&idx, d, shape[d]);
        let out = Tensor::empty_on(idx.layout.shape.clone(), data.dtype, device);
        if device == Device::Cpu {
            cpu::gather(&at_least_1d(&out.data()), &src, d, &positions);
        }
        formulas::gather(self, d, index, &out);
        out
    }

    /// Adds the elements of `src` into `self` along `dim` at the positions in `index`
    /// (`torch.scatter_add`), the reverse of [`gather`](Tensor::gather): for `dim = 1`,
    /// `out[i][index[i][j][k]][k] += src[i][j][k]`, and repeated positions add up.
    ///
    /// `index` is an `i32` or `i64` tensor with as many dimensions as `self` and `src`, no larger
    /// than `src` and no larger than `self` in the other dimensions; its values are positions
    /// along `dim`, none negative. Only the part of `src` that the index covers is used. `src`
    /// has the dtype of `self`. Sums are computed as in [`index_add`](Tensor::index_add), and
    /// zero-dimensional tensors count as one-dimensional ones of one element.
    ///
    /// ```
    /// let x = homura::zeros([3]);
    /// let y = x.scatter_add(0, &homura::tensor([2i64, 0, 2]), &homura::tensor([1.0, 2.0, 3.0]));
    /// assert_eq!(y.to_vec::<f32>(), [2.0, 0.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the shapes or dtypes don't match as described, or if a position is out of
    /// range.
    pub fn scatter_add(&self, dim: i64, index: &Tensor, src: &Tensor) -> Tensor {
        let (data, idx, source) = (self.data(), index.data(), src.data());
        let device = common_device([&data, &idx, &source]);
        check_index_dtype("scatter_add", &idx);
        if data.dtype != source.dtype {
            panic!("scatter_add(): Expected self.dtype to be equal to src.dtype");
        }
        let d = normalize_dim(dim, data.layout.shape.len());
        let (index_1d, source_1d) = (at_least_1d(&idx), at_least_1d(&source));
        let shape = &at_least_1d(&data).layout.shape;
        let (index_shape, src_shape) = (&index_1d.layout.shape, &source_1d.layout.shape);
        if index_shape.len() != shape.len() {
            panic!("Index tensor must have the same number of dimensions as self tensor");
        }
        if src_shape.len() != shape.len() {
            panic!("Index tensor must have the same number of dimensions as src tensor");
        }
        if (0..shape.len())
            .any(|k| (k != d && index_shape[k] > shape[k]) || index_shape[k] > src_shape[k])
        {
            panic!(
                "Expected index {:?} to be no larger than self {:?} apart from dimension {d} and \
                 to be no larger size than src {:?}",
                idx.layout.shape, data.layout.shape, source.layout.shape
            );
        }
        let positions = positions(&idx, d, shape[d]);
        let out = Tensor::from_data(clone_data(&data));
        if device == Device::Cpu {
            let o = at_least_1d(&out.data());
            cpu::scatter_add(&o, d, index_shape, &positions, &source_1d);
        }
        formulas::scatter_add(self, d, index, src, &out);
        out
    }

    /// In-place [`scatter_add`](Tensor::scatter_add).
    pub fn scatter_add_(&self, dim: i64, index: &Tensor, src: &Tensor) -> &Self {
        crate::autograd::check_inplace(self, &[Some(src)]);
        let result = self.scatter_add(dim, index, src);
        self.assign_result(&result);
        self
    }

    /// A copy of `self` with the slice that `self.select(dim, index)` views replaced by `src`
    /// (`torch.select_scatter`), the reverse of [`select`](Tensor::select). `src` must have the
    /// shape of that slice; it is converted to the dtype of `self`.
    ///
    /// ```
    /// let x = homura::zeros([2, 2]);
    /// let y = x.select_scatter(&homura::tensor([1.0, 2.0]), 1, -1);
    /// assert_eq!(y.to_vec::<f32>(), [0.0, 1.0, 0.0, 2.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics like `select` does, or if `src` doesn't have the shape of the slice.
    pub fn select_scatter(&self, src: &Tensor, dim: i64, index: i64) -> Tensor {
        let (data, source) = (self.data(), src.data());
        let device = common_device([&data, &source]);
        let (d, i) = select_position(&data.layout.shape, dim, index);
        let out = self.scatter_into(&source, device, |l| l.select(d, i));
        formulas::select_scatter(self, src, d, i, &out);
        out
    }

    /// A copy of `self` with the Python-style slice `start:end:step` along `dim` replaced by `src`
    /// (`torch.slice_scatter`), the reverse of slicing with [`index`](Tensor::index): negative
    /// bounds count from the end, bounds out of range are clamped, and `None` stands for the
    /// start or the end of the dimension. `src` must have the shape of the slice; it is converted
    /// to the dtype of `self`.
    ///
    /// ```
    /// let x = homura::zeros([5]);
    /// let y = x.slice_scatter(&homura::ones([2]), 0, Some(1), None, 2);
    /// assert_eq!(y.to_vec::<f32>(), [0.0, 1.0, 0.0, 1.0, 0.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero-dimensional, if `dim` is out of range, if `step` is not
    /// positive, or if `src` doesn't have the shape of the slice.
    pub fn slice_scatter(
        &self,
        src: &Tensor,
        dim: i64,
        start: Option<i64>,
        end: Option<i64>,
        step: i64,
    ) -> Tensor {
        let (data, source) = (self.data(), src.data());
        let device = common_device([&data, &source]);
        let shape = &data.layout.shape;
        if shape.is_empty() {
            panic!("slice() cannot be applied to a 0-dim tensor.");
        }
        let d = normalize_dim(dim, shape.len());
        let (start, end, len) = slice_bounds(shape[d], start, end, step);
        let step = step as usize;
        let out = self.scatter_into(&source, device, |l| l.slice(d, start, len, step));
        formulas::slice_scatter(self, src, d, start, end, step, &out);
        out
    }

    /// A contiguous copy of `self` with the part that `part` selects from a layout overwritten
    /// by `src`, which must have its shape: the forward pass of `select_scatter` and
    /// `slice_scatter`.
    fn scatter_into(
        &self,
        src: &TensorData,
        device: Device,
        part: impl Fn(&Layout) -> Layout,
    ) -> Tensor {
        let data = self.data();
        let expected = part(&Layout::contiguous(data.layout.shape.clone())).shape;
        if src.layout.shape != expected {
            panic!(
                "expected src to have a size equal to the slice of self. src size = {:?}, slice \
                 size = {expected:?}",
                src.layout.shape
            );
        }
        let out = Tensor::from_data(clone_data(&data));
        if device == Device::Cpu {
            let o = out.data();
            let target = TensorData {
                layout: part(&o.layout),
                ..o
            };
            cpu::copy(&target, src);
        }
        out
    }
}
