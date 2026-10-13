//! Indexing with `Tensor::index`, the counterpart of Python's `x[...]`.

use std::iter::StepBy;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use super::Tensor;
use crate::backend::cpu;
use crate::{DType, Device};

/// Inserts a new dimension of size 1, like `None` in Python indexing.
#[derive(Clone, Copy, Debug)]
pub struct NewAxis;

/// Stands for as many full slices as needed, like `...` in Python indexing.
#[derive(Clone, Copy, Debug)]
pub struct Ellipsis;

/// One item of an index expression.
#[derive(Clone, Debug)]
pub enum TensorIndex {
    /// Selects one position and removes the dimension. Negative values count from the end.
    Select(i64),
    /// A Python slice `start:end:step`. Missing bounds mean the start or end of the dimension.
    Slice {
        /// First position (inclusive).
        start: Option<i64>,
        /// Last position (exclusive).
        end: Option<i64>,
        /// Distance between positions; must be positive.
        step: i64,
    },
    /// Inserts a dimension of size 1.
    NewAxis,
    /// Expands to as many full slices as needed.
    Ellipsis,
    /// Picks positions given by an integer tensor; the result is a copy.
    Tensor(Tensor),
}

/// Types usable as one item of an index expression: `i64`, ranges of `i64`, `(a..b).step_by(n)`,
/// [`NewAxis`], [`Ellipsis`] and integer tensors.
pub trait IndexItem {
    /// Converts into a [`TensorIndex`].
    fn into_index(self) -> TensorIndex;
}

impl IndexItem for TensorIndex {
    fn into_index(self) -> TensorIndex {
        self
    }
}

impl IndexItem for i64 {
    fn into_index(self) -> TensorIndex {
        TensorIndex::Select(self)
    }
}

impl IndexItem for NewAxis {
    fn into_index(self) -> TensorIndex {
        TensorIndex::NewAxis
    }
}

impl IndexItem for Ellipsis {
    fn into_index(self) -> TensorIndex {
        TensorIndex::Ellipsis
    }
}

impl IndexItem for &Tensor {
    fn into_index(self) -> TensorIndex {
        TensorIndex::Tensor(self.clone())
    }
}

impl IndexItem for Tensor {
    fn into_index(self) -> TensorIndex {
        TensorIndex::Tensor(self)
    }
}

fn slice(start: Option<i64>, end: Option<i64>) -> TensorIndex {
    TensorIndex::Slice {
        start,
        end,
        step: 1,
    }
}

impl IndexItem for RangeFull {
    fn into_index(self) -> TensorIndex {
        slice(None, None)
    }
}

impl IndexItem for Range<i64> {
    fn into_index(self) -> TensorIndex {
        slice(Some(self.start), Some(self.end))
    }
}

impl IndexItem for RangeFrom<i64> {
    fn into_index(self) -> TensorIndex {
        slice(Some(self.start), None)
    }
}

impl IndexItem for RangeTo<i64> {
    fn into_index(self) -> TensorIndex {
        slice(None, Some(self.end))
    }
}

/// An inclusive end of `-1` means "through the last element", and so does `i64::MAX`, which
/// lies past the end of every dimension.
fn inclusive_end(end: i64) -> Option<i64> {
    if end == -1 { None } else { end.checked_add(1) }
}

impl IndexItem for RangeInclusive<i64> {
    fn into_index(self) -> TensorIndex {
        slice(Some(*self.start()), inclusive_end(*self.end()))
    }
}

impl IndexItem for RangeToInclusive<i64> {
    fn into_index(self) -> TensorIndex {
        slice(None, inclusive_end(self.end))
    }
}

/// Recovers `start:end:step` from a stepped range by looking at the positions it yields.
fn stepped<I: Iterator<Item = i64> + Clone>(it: StepBy<I>) -> TensorIndex {
    let mut probe = it.clone();
    let Some(first) = probe.next() else {
        return TensorIndex::Slice {
            start: Some(0),
            end: Some(0),
            step: 1,
        };
    };
    let Some(second) = probe.next() else {
        return TensorIndex::Slice {
            start: Some(first),
            end: inclusive_end(first),
            step: 1,
        };
    };
    let step = second - first;
    let end = match it.size_hint() {
        (n, Some(m)) if n == m => inclusive_end(first + (n as i64 - 1) * step),
        _ => None,
    };
    TensorIndex::Slice {
        start: Some(first),
        end,
        step,
    }
}

impl IndexItem for StepBy<Range<i64>> {
    fn into_index(self) -> TensorIndex {
        stepped(self)
    }
}

impl IndexItem for StepBy<RangeInclusive<i64>> {
    fn into_index(self) -> TensorIndex {
        stepped(self)
    }
}

impl IndexItem for StepBy<RangeFrom<i64>> {
    fn into_index(self) -> TensorIndex {
        stepped(self)
    }
}

/// Whether the slice `start:end:step` covers all of a dimension of `size`, judged like PyTorch
/// by the bounds as given: starting at 0 (not at `-size`) and ending at or past the end, with a
/// step of 1.
fn covers(size: usize, start: Option<i64>, end: Option<i64>, step: i64) -> bool {
    step == 1 && start.unwrap_or(0) == 0 && end.is_none_or(|end| end >= size as i64)
}

/// A full index expression: a single [`IndexItem`] or a tuple of them.
pub trait IndexSpec {
    /// Converts into a list of [`TensorIndex`] items.
    fn into_indices(self) -> Vec<TensorIndex>;
}

impl<T: IndexItem> IndexSpec for T {
    fn into_indices(self) -> Vec<TensorIndex> {
        vec![self.into_index()]
    }
}

impl IndexSpec for Vec<TensorIndex> {
    fn into_indices(self) -> Vec<TensorIndex> {
        self
    }
}

macro_rules! tuple_spec {
    ($($name:ident),+) => {
        impl<$($name: IndexItem),+> IndexSpec for ($($name,)+) {
            #[allow(non_snake_case)]
            fn into_indices(self) -> Vec<TensorIndex> {
                let ($($name,)+) = self;
                vec![$($name.into_index()),+]
            }
        }
    };
}

tuple_spec!(A);
tuple_spec!(A, B);
tuple_spec!(A, B, C);
tuple_spec!(A, B, C, D);
tuple_spec!(A, B, C, D, E);
tuple_spec!(A, B, C, D, E, F);
tuple_spec!(A, B, C, D, E, F, G);
tuple_spec!(A, B, C, D, E, F, G, H);

impl Tensor {
    /// Indexes like Python's `x[...]`: integers select (and remove) a dimension, ranges slice,
    /// `(a..b).step_by(n)` slices with a step, [`NewAxis`] inserts a dimension, [`Ellipsis`]
    /// skips dimensions, and an integer tensor picks positions along its dimension.
    ///
    /// The result is a view of `self`, except when a tensor index is used. Slices follow Python
    /// semantics: negative bounds count from the end and out-of-range bounds are clamped.
    ///
    /// As in PyTorch, when there are several items, a slice that covers its whole dimension
    /// (start 0 or omitted, end omitted or at least the size, step 1) is skipped instead of
    /// recorded in the backward graph, and an expression that changes nothing, such as
    /// `(.., ..)` or [`Ellipsis`], gives a new view of all of `self`. A lone slice is always
    /// recorded; unlike Python, a one-item tuple counts as a lone item.
    ///
    /// ```
    /// use homura::{Ellipsis, NewAxis};
    /// let x = homura::arange(24).view([2, 3, 4]);
    /// assert_eq!(x.index((.., 1..3)).shape(), [2, 2, 4]);
    /// assert_eq!(x.index((Ellipsis, -1)).shape(), [2, 3]);
    /// assert_eq!(x.index((0, NewAxis)).shape(), [1, 3, 4]);
    /// assert_eq!(x.index((.., .., (0..4).step_by(2))).shape(), [2, 3, 2]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics on out-of-range positions, on more indices than dimensions, on more than one
    /// ellipsis, and (for now) on more than one tensor index or boolean masks.
    pub fn index(&self, spec: impl IndexSpec) -> Tensor {
        let items = spec.into_indices();
        let consumed = items
            .iter()
            .filter(|i| !matches!(i, TensorIndex::NewAxis | TensorIndex::Ellipsis))
            .count();
        let nd = self.num_dims();
        if consumed > nd {
            panic!("too many indices for tensor of dimension {nd}");
        }
        if items
            .iter()
            .filter(|i| matches!(i, TensorIndex::Ellipsis))
            .count()
            > 1
        {
            panic!("an index can only have a single ellipsis ('...')");
        }
        // As in PyTorch, among several items a slice that covers its whole dimension is skipped
        // rather than recorded, while a lone slice is always a slice.
        let lone = items.len() == 1;
        let mut result = self.clone();
        let mut dim = 0;
        let mut tensor_index = None;
        for item in items {
            match item {
                TensorIndex::Select(i) => result = result.select(dim as i64, i),
                TensorIndex::Slice { start, end, step } => {
                    if lone || !covers(result.shape()[dim], start, end, step) {
                        result = result.slice(dim, start, end, step);
                    }
                    dim += 1;
                }
                TensorIndex::NewAxis => {
                    result = result.unsqueeze(dim as i64);
                    dim += 1;
                }
                TensorIndex::Ellipsis => dim += nd - consumed,
                TensorIndex::Tensor(t) => {
                    if tensor_index.is_some() {
                        panic!("indexing with more than one tensor is not supported yet");
                    }
                    tensor_index = Some((dim, t));
                    dim += 1;
                }
            }
        }
        match tensor_index {
            // Nothing applied, as with `x[...]` or `x[:, :]`: still a new view, like PyTorch's.
            None if result.is_same(self) => self.alias(),
            None => result,
            Some((d, idx)) => result.take_along(d, &idx),
        }
    }

    /// Picks positions along `dim` with an integer tensor of any shape; the dimension is replaced
    /// by the shape of `index`. Negative positions count from the end.
    fn take_along(&self, dim: usize, index: &Tensor) -> Tensor {
        let idx_shape = index.shape();
        if !index.dtype().is_integral() {
            panic!(
                "tensors used as indices must be integer tensors, got {}",
                index.dtype()
            );
        }
        let flat = index.reshape([-1]).to(DType::I64).contiguous();
        let selected = self.select_positions(dim, &flat, true);
        crate::autograd::formulas::index(self, dim, index, &selected);
        let shape = self.shape();
        let new_shape: Vec<i64> = shape[..dim]
            .iter()
            .chain(idx_shape.iter())
            .chain(shape[dim + 1..].iter())
            .map(|&d| d as i64)
            .collect();
        selected.unsafe_view(new_shape)
    }

    /// The slices along `dim` at the positions in the flat `i64` tensor `index`, without
    /// recording anything for autograd: the callers record their own node.
    fn select_positions(&self, dim: usize, index: &Tensor, allow_negative: bool) -> Tensor {
        let (src, idx) = (self.data(), index.data());
        let device = crate::ops::common_device([&src, &idx]);
        let mut shape = src.layout.shape.clone();
        shape[dim] = idx.layout.num_elements();
        let out = Tensor::empty_on(shape, src.dtype, device);
        if device == Device::Cpu {
            cpu::index_select(&out.data(), &src, dim, &idx, allow_negative);
        }
        out
    }

    /// The slices along `dim` at the positions in the 1-D integer tensor `index`
    /// (`torch.index_select`).
    pub fn index_select(&self, dim: i64, index: &Tensor) -> Tensor {
        if index.num_dims() > 1 {
            panic!("index_select(): Index is supposed to be a vector");
        }
        if !matches!(index.dtype(), DType::I64 | DType::I32) {
            panic!(
                "index_select(): Expected dtype i32 or i64 for index, got {}",
                index.dtype()
            );
        }
        let nd = self.num_dims();
        if nd == 0 {
            panic!("index_select() cannot be applied to a 0-dim tensor.");
        }
        let d = crate::shape::normalize_dim(dim, nd);
        let flat = index.reshape([-1]).to(DType::I64).contiguous();
        let out = self.select_positions(d, &flat, false);
        crate::autograd::formulas::index_select(self, d, index, &out);
        out
    }
}
