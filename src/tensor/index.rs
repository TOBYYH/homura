//! Indexing with `Tensor::index`, the counterpart of Python's `x[...]`.

use std::iter::StepBy;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use super::Tensor;
use crate::backend::cpu;
use crate::ops::{META_VALUES, check_index_dtype, check_index_put_target};
use crate::{DType, Device, Shape};

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
    /// Picks positions given by an `i64` or `i32` tensor, or the elements where a `bool` mask is
    /// set (advanced indexing); the result is a copy. A zero-dimensional integer tensor selects
    /// like [`Select`](TensorIndex::Select).
    Tensor(Tensor),
}

/// Types usable as one item of an index expression: `i64`, ranges of `i64`, `(a..b).step_by(n)`,
/// [`NewAxis`], [`Ellipsis`], `bool` and index tensors.
pub trait IndexItem {
    /// Converts into a [`TensorIndex`].
    #[track_caller]
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

/// Like Python's `x[True]` and `x[False]`: inserts a dimension of size 1 and keeps it (`true`) or
/// empties it (`false`). Converts to a zero-dimensional `bool` tensor on the CPU, whatever the
/// default device.
impl IndexItem for bool {
    fn into_index(self) -> TensorIndex {
        TensorIndex::Tensor(Tensor::from_host(&[self], Shape::scalar(), Device::Cpu))
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
    #[track_caller]
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

/// The number of dimensions of the indexed tensor that `t`, an item of an index expression,
/// stands for: as many as a mask has, one for an integer tensor.
fn index_dims(t: &Tensor) -> usize {
    if t.dtype() == DType::Bool {
        t.num_dims()
    } else {
        1
    }
}

/// The value of a zero-dimensional index tensor, which indexing consumes. Meta tensors have none.
#[track_caller]
fn index_value(t: &Tensor) -> i64 {
    let data = t.data();
    if data.device() != Device::Cpu {
        panic!("{META_VALUES}");
    }
    cpu::read_host::<i64>(&data)[0]
}

impl Tensor {
    /// Indexes like Python's `x[...]`: integers select (and remove) a dimension, ranges slice,
    /// `(a..b).step_by(n)` slices with a step, [`NewAxis`] inserts a dimension, [`Ellipsis`]
    /// skips dimensions, and tensors index with positions or masks (advanced indexing).
    ///
    /// Basic indexing (integers, slices, [`NewAxis`], [`Ellipsis`] and zero-dimensional integer
    /// tensors, which select like integers) gives a view of `self`. Any other index tensor makes
    /// it advanced indexing, which gives a copy. Slices follow Python semantics: negative bounds
    /// count from the end and out-of-range bounds are clamped.
    ///
    /// As in PyTorch, when there are several items, a slice that covers its whole dimension
    /// (start 0 or omitted, end omitted or at least the size, step 1) is skipped instead of
    /// recorded in the backward graph, and an expression that changes nothing, such as
    /// `(.., ..)` or [`Ellipsis`], gives a new view of all of `self`. A lone slice is always
    /// recorded; unlike Python, a one-item tuple counts as a lone item. The check uses the slice
    /// an item converts to: a `step_by` range that yields fewer than two positions can't reveal
    /// its step and becomes a slice with step 1, so `(0..1).step_by(2)` on a dimension of size 1
    /// is skipped where Python's `0:1:2` is recorded (values and gradients are the same). Use
    /// [`TensorIndex::Slice`] to keep the step.
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
    /// # Advanced indexing
    ///
    /// Tensor items follow PyTorch (and mostly NumPy):
    ///
    /// - An `i64` or `i32` tensor picks positions along its dimension, negative ones counting
    ///   from the end; the dimension is replaced by the shape of the tensor. A zero-dimensional
    ///   one selects a position like an integer, giving a view.
    /// - A `bool` mask with k dimensions covers the next k dimensions, whose sizes it must have,
    ///   and picks the elements where it is set, in row-major order: those k dimensions are
    ///   replaced by one dimension, the number of set elements. A `bool` item (or a
    ///   zero-dimensional mask) inserts a dimension of size 1 and keeps it (`true`) or empties it
    ///   (`false`).
    /// - Several tensors are broadcast together, and the broadcast shape replaces the dimensions
    ///   they index. If those dimensions are adjacent, it takes their place; otherwise it comes
    ///   first. Adjacency is judged after the integers and the zero-dimensional integer tensors
    ///   have removed their dimensions, so `x[:, i, 0, j]` keeps `i` and `j` adjacent (unlike
    ///   NumPy), while any slice between them, even a skipped full slice, and [`NewAxis`] separate
    ///   them; an [`Ellipsis`] separates them only if it stands for at least one dimension.
    ///
    /// Basic items first make a view, which the tensors index in one step: the backward graph
    /// holds one `IndexBackward0` on top of the views' nodes. That node saves the index tensors
    /// as given, so modifying one in place before the backward pass is an error (but not a
    /// zero-dimensional integer or bool tensor, whose value has been used up).
    ///
    /// ```
    /// use homura::tensor;
    /// let x = homura::arange(24).view([2, 3, 4]);
    /// let (i, j) = (tensor([0i64, 2]), tensor([1i64, 3]));
    /// assert_eq!(x.index((.., &i, &j)).shape(), [2, 2]);
    /// assert_eq!(x.index((&i.index(..1), .., &j)).shape(), [2, 3]);
    /// let mask = x.index((0, .., 0)).greater(3);
    /// assert_eq!(x.index((.., &mask)).shape(), [2, 2, 4]);
    /// assert_eq!(x.index(true).shape(), [1, 2, 3, 4]);
    /// ```
    ///
    /// Index tensors may be on the CPU or on the device of `self`. On the meta device, integer
    /// tensors index by shape only; masks and zero-dimensional index tensors on the meta device
    /// are rejected, since the result depends on their values. Positions in CPU tensors are
    /// checked even where PyTorch skips the check (for a meta `self`, or an empty result).
    ///
    /// # Panics
    ///
    /// Panics on out-of-range positions, on more indices than dimensions, on more than one
    /// ellipsis, on index tensors of another dtype (PyTorch also accepts `u8` masks and
    /// zero-dimensional `i8` and `i16` tensors), on masks of the wrong shape, on indices that
    /// don't broadcast together, and on meta tensors as described.
    #[track_caller]
    pub fn index(&self, spec: impl IndexSpec) -> Tensor {
        let items = spec.into_indices();
        for item in &items {
            if let TensorIndex::Tensor(t) = item {
                check_index_dtype(t.dtype());
            }
        }
        let consumed: usize = items
            .iter()
            .map(|i| match i {
                TensorIndex::NewAxis | TensorIndex::Ellipsis => 0,
                TensorIndex::Tensor(t) => index_dims(t),
                _ => 1,
            })
            .sum();
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
        // The entries for the advanced indexing core, like PyTorch's: the first tensor at the
        // position of the dimension it indexes (after `None`s for the dimensions before it),
        // then one entry per item that keeps dimensions, `None` for each of those dimensions.
        let mut entries: Vec<Option<Tensor>> = vec![];
        let skip = |entries: &mut Vec<Option<Tensor>>, n: usize| {
            if !entries.is_empty() {
                entries.resize(entries.len() + n, None);
            }
        };
        for item in items {
            match item {
                TensorIndex::Select(i) => result = result.select(dim as i64, i),
                TensorIndex::Slice { start, end, step } => {
                    if lone || !covers(result.shape()[dim], start, end, step) {
                        result = result.slice(dim, start, end, step);
                    }
                    dim += 1;
                    skip(&mut entries, 1);
                }
                TensorIndex::NewAxis => {
                    result = result.unsqueeze(dim as i64);
                    dim += 1;
                    skip(&mut entries, 1);
                }
                TensorIndex::Ellipsis => {
                    dim += nd - consumed;
                    skip(&mut entries, nd - consumed);
                }
                TensorIndex::Tensor(t) => {
                    let t = if t.num_dims() > 0 {
                        t
                    } else if t.dtype() != DType::Bool {
                        result = result.select(dim as i64, index_value(&t));
                        continue;
                    } else {
                        // Like PyTorch's `boolToIndexingTensor`.
                        let keep = index_value(&t) != 0;
                        result = result.unsqueeze(dim as i64);
                        let positions: &[i64] = if keep { &[0] } else { &[] };
                        let shape = Shape::new(&[positions.len()]);
                        Tensor::from_host(positions, shape, self.device())
                    };
                    if entries.is_empty() {
                        entries.resize(dim, None);
                    }
                    dim += index_dims(&t);
                    entries.push(Some(t));
                }
            }
        }
        while matches!(entries.last(), Some(None)) {
            entries.pop();
        }
        if !entries.is_empty() {
            result.index_tensors(&entries)
        } else if result.is_same(self) {
            // Nothing applied, as with `x[...]` or `x[:, :]`: still a new view, like PyTorch's.
            self.alias()
        } else {
            result
        }
    }

    /// Writes `values` into the elements of `self` that advanced indexing with `indices` picks,
    /// in place (`index_put_`, Python's `x[indices] = values` or, with `accumulate`,
    /// `x[indices] += values` where repeated positions add up). See
    /// [`index_put`](Tensor::index_put) for the indices and values.
    ///
    /// ```
    /// use homura::tensor;
    /// let x = homura::zeros([2, 3]);
    /// x.index_put_((.., &tensor([2i64, 0])), &tensor([1.0, 2.0]), false); // x[:, [2, 0]] = ..
    /// assert_eq!(x.to_vec::<f32>(), [2.0, 0.0, 1.0, 2.0, 0.0, 1.0]);
    /// let mask = x.greater(1.5);
    /// x.index_put_(&mask, &tensor(-1.0), false); // x[mask] = -1
    /// assert_eq!(x.to_vec::<f32>(), [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
    /// ```
    ///
    /// Python's `x[0, idx] = v` indexes a view first; this works where the
    /// [in-place rules](crate::autograd#in-place-operations) allow writing into a view, such as
    /// updating a parameter in no-grad mode: `x.index(0).index_put_((&idx,), &v, false)`. Where
    /// gradients are needed, index with tensors only, broadcasting together:
    /// `x.index_put_((&tensor([0i64]), &idx), &v, false)`.
    ///
    /// # Panics
    ///
    /// Panics like [`index_put`](Tensor::index_put), under the in-place rules, and also if the
    /// elements of `self` can't be proven to be distinct memory locations (PyTorch warns) or if
    /// `values` or an index tensor may share memory with `self`. Every check runs before the
    /// write, so a call that panics changes nothing.
    #[track_caller]
    pub fn index_put_(&self, indices: impl IndexSpec, values: &Tensor, accumulate: bool) -> &Self {
        let entries = self.put_entries(indices.into_indices());
        let begun = self.begin_in_place(&[Some(values)]);
        let computed = {
            check_index_put_target(self, &entries, values);
            (self.index_put_tensors(&entries, values, accumulate), ())
        };
        self.finish_in_place(begun, computed, false, |original, ()| {
            crate::autograd::formulas::index_put(original, &entries, values, accumulate, self)
        })
    }

    /// A copy of `self` with `values` written into the elements that advanced indexing with
    /// `indices` picks (`torch.index_put`): copied, or added with `accumulate`.
    ///
    /// `indices` holds index tensors, as in [`index`](Tensor::index), full slices `..` for
    /// dimensions to keep, and at most one [`Ellipsis`]: it corresponds to PyTorch's list of
    /// optional tensors, `..` standing for `None`. At least one tensor is needed. A
    /// zero-dimensional integer tensor indexes like any other (it doesn't select), and
    /// zero-dimensional masks are not accepted.
    ///
    /// `values` has the dtype and device of `self` and broadcasts to the shape of
    /// `self.index(indices)`, without dropping leading dimensions of size 1. Without
    /// `accumulate`, which of several values for a repeated position is written is unspecified,
    /// as in PyTorch (the CPU backend happens to write the last; don't rely on it). With it,
    /// repeated positions add up, summed like [`index_add`](Tensor::index_add): in `f64` for
    /// floating point dtypes and rounded once, which can differ from PyTorch's sums, rounded
    /// after each addition, by more than the last bit (`2048 + 1 - 2048` is 1 here and 0 in
    /// PyTorch in `f16`). Integer sums wrap around, and bool sums are a logical or.
    ///
    /// Unlike the in-place [`index_put_`](Tensor::index_put_), `values` and the indices may
    /// share memory with `self`. The result is new memory. On the CPU it is contiguous with its
    /// dimensions in the order the indexing works in, so it is not contiguous when the indexed
    /// dimensions aren't adjacent; it doesn't keep the strides of `self`. For autograd, the node
    /// is `IndexPutBackward0`: `self` gets the gradient with the written positions zeroed (unless
    /// `accumulate`), and `values` the gradient at those positions, summed to its shape.
    ///
    /// ```
    /// use homura::tensor;
    /// let x = homura::zeros([4]);
    /// let y = x.index_put((&tensor([1i64, 3, 1]),), &tensor(1.0), true);
    /// assert_eq!(y.to_vec::<f32>(), [0.0, 2.0, 0.0, 1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics on items other than tensors, `..` and one [`Ellipsis`], without a tensor, on
    /// zero-dimensional masks, if the indices are invalid as in `index`, or if `values` doesn't
    /// have the dtype, device or a shape as described.
    #[track_caller]
    pub fn index_put(&self, indices: impl IndexSpec, values: &Tensor, accumulate: bool) -> Tensor {
        let entries = self.put_entries(indices.into_indices());
        self.index_put_tensors(&entries, values, accumulate)
    }

    /// The entries for the advanced indexing core from the items of `index_put` and
    /// `index_put_`: tensors as they are, `..` as `None`, an ellipsis as `None` for every
    /// dimension it stands for, without trailing `None`s.
    #[track_caller]
    fn put_entries(&self, items: Vec<TensorIndex>) -> Vec<Option<Tensor>> {
        let mut entries: Vec<Option<Option<Tensor>>> = vec![];
        let mut ellipses = 0;
        for item in items {
            match item {
                TensorIndex::Tensor(t) => {
                    check_index_dtype(t.dtype());
                    if t.num_dims() == 0 && t.dtype() == DType::Bool {
                        panic!(
                            "index_put(): zero-dimensional bool masks are not supported; index \
                             with integer tensors instead (index(true) and index(false) give \
                             copies, which writes don't reach)"
                        );
                    }
                    entries.push(Some(Some(t)));
                }
                TensorIndex::Slice {
                    start: None,
                    end: None,
                    step: 1,
                } => entries.push(Some(None)),
                TensorIndex::Ellipsis => {
                    ellipses += 1;
                    entries.push(None);
                }
                other => panic!(
                    "index_put(): indices can only be tensors, full slices (..) and Ellipsis, got \
                     {other:?}; use index() to take a view first"
                ),
            }
        }
        if ellipses > 1 {
            panic!("an index can only have a single ellipsis ('...')");
        }
        let consumed: usize = entries
            .iter()
            .flatten()
            .map(|e| e.as_ref().map_or(1, index_dims))
            .sum();
        let nd = self.num_dims();
        if consumed > nd {
            panic!("too many indices for tensor of dimension {nd} (got {consumed})");
        }
        let mut out = vec![];
        for entry in entries {
            match entry {
                Some(e) => out.push(e),
                None => out.resize(out.len() + nd - consumed, None),
            }
        }
        while matches!(out.last(), Some(None)) {
            out.pop();
        }
        if out.is_empty() {
            panic!("index_put(): at least one index must be a tensor");
        }
        out
    }

    /// The slices along `dim` at the positions in the flat `i64` tensor `index`, without
    /// recording anything for autograd: the callers record their own node.
    #[track_caller]
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
    #[track_caller]
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
