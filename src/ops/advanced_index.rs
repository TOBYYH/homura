//! Advanced indexing: indexing with several integer tensors or boolean masks at once, ATen's
//! `index.Tensor` and `index_put`. [`Tensor::index`] and [`Tensor::index_put`] parse their index
//! expressions into a list of optional index tensors, one entry per dimension (`None` keeps the
//! dimension), and call the core operations here, which their backward formulas use in turn.
//!
//! The core follows ATen's `make_info`: masks are expanded into one coordinate vector per
//! dimension they cover, the indices are broadcast together, and when the indexed dimensions are
//! not adjacent they are moved to the front of the result. Both operations fold the target, with
//! its dimensions in that order, into `[before, indexed, after]` and pick or write slices along
//! the middle dimension at the linear positions that the indices give in the indexed subspace.

use super::binary::{check_no_overlap, check_writable};
use super::{clone_data, common_device, reshape_data};
use crate::autograd::formulas;
use crate::backend::cpu;
use crate::shape::{checked_num_elements, contiguous_strides};
use crate::tensor::{Layout, TensorData};
use crate::{DType, Device, Shape, Tensor};

/// Panics unless `dtype` is a dtype that indexing accepts: `i64` or `i32` positions, or a `bool`
/// mask. PyTorch also takes `u8` tensors, as deprecated masks.
#[track_caller]
pub(crate) fn check_index_dtype(dtype: DType) {
    if !matches!(dtype, DType::I64 | DType::I32 | DType::Bool) {
        let hint = if dtype == DType::U8 {
            "; use a bool tensor for a mask"
        } else {
            ""
        };
        panic!("tensors used as indices must be i64, i32 or bool tensors, got {dtype}{hint}");
    }
}

/// The message for indexing that needs the values of a meta tensor.
pub(crate) const META_VALUES: &str = "indexing with a bool mask or a zero-dimensional bool or \
    integer tensor on the meta device is not supported: the result depends on its values, which \
    a meta tensor doesn't have";

/// One index of the core after masks are expanded.
struct Coords {
    shape: Shape,
    /// The positions in row-major order; `None` for an index on the meta device.
    values: Option<Vec<i64>>,
}

/// The work out of an advanced indexing expression, before anything is read or written.
struct AdvancedIndex {
    /// The dimensions of the target in the order the operation works in: the indexed ones moved
    /// to the front when they aren't adjacent.
    perm: Vec<usize>,
    /// The shape of the target with its dimensions in the order of `perm`.
    shape: Shape,
    /// The position of the first indexed dimension in `shape`; the others follow it.
    first: usize,
    /// One index per indexed dimension.
    coords: Vec<Coords>,
    /// The shape the indices broadcast to.
    broadcast: Shape,
    /// The shape of the indexing result: the dimensions before the indexed ones, the broadcast
    /// shape and the dimensions after them.
    out_shape: Shape,
}

/// The broadcast shape of `a` and `b`, or `None` if they don't broadcast.
fn broadcast(a: &[usize], b: &[usize]) -> Option<Shape> {
    let n = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        if i < n - s.len() {
            1
        } else {
            s[i - (n - s.len())]
        }
    };
    (0..n)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (x, y) if x == y || y == 1 => Some(x),
            (1, y) => Some(y),
            _ => None,
        })
        .collect()
}

impl AdvancedIndex {
    /// Checks `indices` against the target `target` and works out the shapes, as ATen's
    /// `make_info` does: index dtypes, then devices (an index is on the CPU or on the device of
    /// the target), mask shapes, the number of indices, broadcasting, and indices into empty
    /// dimensions. The positions are checked later, by [`positions`](Self::positions) or
    /// [`check_bounds`](Self::check_bounds).
    #[track_caller]
    fn new(target: &TensorData, indices: &[Option<Tensor>]) -> AdvancedIndex {
        let device = target.device();
        let shape = &target.layout.shape;
        let nd = shape.len();
        let tensors = || indices.iter().flatten();
        for t in tensors() {
            check_index_dtype(t.dtype());
        }
        for t in tensors() {
            let d = t.device();
            if d != Device::Cpu && d != device {
                panic!(
                    "indices should be either on cpu or on the same device as the indexed tensor \
                     ({device})"
                );
            }
        }
        let count: usize = indices
            .iter()
            .map(|entry| match entry {
                Some(t) if t.dtype() == DType::Bool => t.num_dims(),
                _ => 1,
            })
            .sum();
        if count > nd {
            panic!("too many indices for tensor of dimension {nd} (got {count})");
        }
        let mut slots: Vec<Option<Coords>> = Vec::with_capacity(nd);
        for entry in indices {
            let Some(t) = entry else {
                slots.push(None);
                continue;
            };
            let data = t.data();
            if t.dtype() != DType::Bool {
                let values = (data.device() == Device::Cpu).then(|| cpu::read_host::<i64>(&data));
                slots.push(Some(Coords {
                    shape: data.layout.shape.clone(),
                    values,
                }));
                continue;
            }
            let mask_shape = &data.layout.shape;
            if mask_shape.is_empty() || data.device() != Device::Cpu {
                panic!("{META_VALUES}");
            }
            let start = slots.len();
            for (j, &n) in mask_shape.iter().enumerate() {
                if n != shape[start + j] {
                    panic!(
                        "The shape of the mask {mask_shape:?} at index {j} does not match the \
                         shape of the indexed tensor {shape:?} at index {}",
                        start + j
                    );
                }
            }
            for values in cpu::mask_coordinates(&data) {
                slots.push(Some(Coords {
                    shape: Shape::new(&[values.len()]),
                    values: Some(values),
                }));
            }
        }
        while matches!(slots.last(), Some(None)) {
            slots.pop();
        }
        let defined: Vec<usize> = (0..slots.len()).filter(|&k| slots[k].is_some()).collect();
        let (Some(&first), Some(&last)) = (defined.first(), defined.last()) else {
            panic!("advanced indexing needs at least one index tensor");
        };
        let adjacent = (first..=last).all(|k| slots[k].is_some());
        slots.resize_with(nd, || None);
        let (perm, first): (Vec<usize>, usize) = if adjacent {
            ((0..nd).collect(), first)
        } else {
            let rest = (0..nd).filter(|&k| slots[k].is_none());
            (defined.iter().copied().chain(rest).collect(), 0)
        };
        let shape: Shape = perm.iter().map(|&k| shape[k]).collect();
        let coords: Vec<Coords> = perm.iter().filter_map(|&k| slots[k].take()).collect();
        let broadcast = coords
            .iter()
            .try_fold(Shape::scalar(), |acc, c| broadcast(&acc, &c.shape));
        // `match` / `let else` rather than closures on the paths that panic: `#[track_caller]`
        // doesn't pass through closures.
        let Some(broadcast) = broadcast else {
            let shapes: Vec<String> = coords.iter().map(|c| format!("{:?}", c.shape)).collect();
            panic!(
                "shape mismatch: indexing tensors could not be broadcast together with shapes {}",
                shapes.join(", ")
            )
        };
        let n = coords.len();
        let Some(num_positions) = checked_num_elements(&broadcast) else {
            panic!("the indices broadcast to a shape {broadcast:?} too large")
        };
        if num_positions > 0 && shape[first..first + n].contains(&0) {
            panic!("index is out of bounds for dimension with size 0");
        }
        let out_shape: Shape = shape[..first]
            .iter()
            .chain(broadcast.iter())
            .chain(shape[first + n..].iter())
            .copied()
            .collect();
        AdvancedIndex {
            perm,
            shape,
            first,
            coords,
            broadcast,
            out_shape,
        }
    }

    /// The sizes of the indexed dimensions.
    fn indexed_sizes(&self) -> &[usize] {
        &self.shape[self.first..self.first + self.coords.len()]
    }

    /// The target folded into `[before, indexed, after]`: the products of the sizes of the
    /// dimensions before, of and after the indexed ones.
    fn fold(&self) -> (usize, usize, usize) {
        let n = self.coords.len();
        let product = |s: &[usize]| s.iter().product::<usize>();
        (
            product(&self.shape[..self.first]),
            product(self.indexed_sizes()),
            product(&self.shape[self.first + n..]),
        )
    }

    /// Panics unless every position of the indices on the CPU is in range: from `-size` to
    /// `size - 1` along its dimension. Unlike PyTorch, this holds even when the result is empty
    /// and for meta targets. Indices on the meta device have no positions to check.
    #[track_caller]
    fn check_bounds(&self) {
        for (k, (c, &size)) in self.coords.iter().zip(self.indexed_sizes()).enumerate() {
            let size = size as i64;
            let bad = c.values.iter().flatten().find(|&&v| v < -size || v >= size);
            if let Some(v) = bad {
                panic!("index {v} is out of bounds for dimension {k} with size {size}");
            }
        }
    }

    /// The linear position in the indexed subspace (row-major over the indexed dimensions) for
    /// every element of the broadcast shape, in row-major order, after
    /// [`check_bounds`](Self::check_bounds). All indices must be on the CPU.
    #[track_caller]
    fn positions(&self) -> Vec<usize> {
        self.check_bounds();
        let sizes = self.indexed_sizes();
        let strides = contiguous_strides(sizes);
        let shape = &self.broadcast;
        let walks: Vec<_> = self
            .coords
            .iter()
            .map(|c| Layout::contiguous(c.shape.clone()).expand_to(shape).strides)
            .collect();
        let values: Vec<&[i64]> = self
            .coords
            .iter()
            .map(|c| c.values.as_deref().expect("indices on the CPU"))
            .collect();
        let total = shape.num_elements();
        let mut out = Vec::with_capacity(total);
        let mut index = vec![0usize; shape.len()];
        for _ in 0..total {
            let mut position = 0;
            for k in 0..values.len() {
                let offset: usize = index.iter().zip(walks[k].iter()).map(|(i, s)| i * s).sum();
                let v = values[k][offset];
                let wrapped = if v < 0 { v + sizes[k] as i64 } else { v } as usize;
                position += wrapped * strides[k];
            }
            out.push(position);
            for d in (0..index.len()).rev() {
                index[d] += 1;
                if index[d] < shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
        out
    }

    /// `data` with its dimensions in the order of `perm`.
    fn permuted(&self, data: &TensorData) -> TensorData {
        let l = &data.layout;
        TensorData {
            layout: Layout {
                shape: self.perm.iter().map(|&k| l.shape[k]).collect(),
                strides: self.perm.iter().map(|&k| l.strides[k]).collect(),
                offset: l.offset,
            },
            ..data.clone()
        }
    }

    /// The indexing result for a CPU target: a new contiguous tensor of `out_shape`.
    #[track_caller]
    fn gather(&self, target: &TensorData) -> Tensor {
        let out = Tensor::empty_on(self.out_shape.clone(), target.dtype, Device::Cpu);
        if self.out_shape.num_elements() == 0 {
            self.check_bounds();
            return out;
        }
        let positions: Vec<i64> = self.positions().into_iter().map(|p| p as i64).collect();
        let (before, indexed, after) = self.fold();
        let src = reshape_data(
            &self.permuted(target),
            Shape::new(&[before, indexed, after]),
        );
        let o = out.data();
        let folded = TensorData {
            layout: Layout::contiguous(Shape::new(&[before, positions.len(), after])),
            ..o
        };
        let index = Tensor::from_host(&positions, Shape::new(&[positions.len()]), Device::Cpu);
        cpu::index_select(&folded, &src, 1, &index.data(), false);
        out
    }

    /// Panics unless `values` can be written into the indexing result of `target`: it has the
    /// dtype and device of the target and broadcasts to `out_shape` without dropping leading
    /// dimensions of size 1.
    #[track_caller]
    fn check_values(&self, target: &TensorData, values: &TensorData) {
        if values.dtype != target.dtype {
            panic!(
                "Index put requires the source and destination dtypes match, got {} for the \
                 destination and {} for the source.",
                target.dtype, values.dtype
            );
        }
        common_device([target, values]);
        let (v, o) = (&values.layout.shape, &self.out_shape);
        let fits = v.len() <= o.len()
            && v.iter()
                .zip(&o[o.len() - v.len()..])
                .all(|(&a, &b)| a == b || a == 1);
        if !fits {
            panic!(
                "shape mismatch: value tensor of shape {v:?} cannot be broadcast to indexing \
                 result of shape {o:?}"
            );
        }
    }

    /// A copy of the CPU tensor `target` with `values` written at the indexed positions: added
    /// with `accumulate`, otherwise copied, with the last of repeated positions winning. The
    /// work happens on a contiguous copy with the dimensions in the order of `perm`, which the
    /// result sees in the original order: new storage, possibly not contiguous.
    #[track_caller]
    fn scatter(&self, target: &TensorData, values: &TensorData, accumulate: bool) -> Tensor {
        let work = clone_data(&self.permuted(target));
        if self.out_shape.num_elements() == 0 {
            self.check_bounds();
        } else {
            let positions = self.positions();
            let (before, indexed, after) = self.fold();
            let folded = TensorData {
                layout: Layout::contiguous(Shape::new(&[before, indexed, after])),
                ..work.clone()
            };
            let src = reshape_data(
                &values.expand_to(&self.out_shape),
                Shape::new(&[before, positions.len(), after]),
            );
            if accumulate {
                cpu::index_add(&folded, 1, &positions, &src);
            } else {
                cpu::index_overwrite(&folded, 1, &positions, &src);
            }
        }
        let l = &work.layout;
        let nd = self.perm.len();
        let (mut shape, mut strides) = (vec![0; nd], vec![0; nd]);
        for (k, &d) in self.perm.iter().enumerate() {
            shape[d] = l.shape[k];
            strides[d] = l.strides[k];
        }
        Tensor::from_data(TensorData {
            layout: Layout {
                shape: Shape::new(&shape),
                strides: crate::Strides::new(&strides),
                offset: l.offset,
            },
            ..work
        })
    }
}

/// The checks that only the in-place `index_put_` makes, on the tensors as the caller gave
/// them, before anything is copied: the elements of `target` are distinct memory locations, and
/// neither `values` nor an index may share memory with it. The out-of-place `index_put` writes
/// into a copy and accepts all of these.
#[track_caller]
pub(crate) fn check_index_put_target(target: &Tensor, indices: &[Option<Tensor>], values: &Tensor) {
    let dst = target.data();
    check_writable(&dst);
    let others = std::iter::once(values).chain(indices.iter().flatten());
    check_no_overlap(&dst, others.map(|t| t.data()));
}

impl Tensor {
    /// ATen's `index.Tensor`: `self` indexed with one optional index tensor per leading
    /// dimension (`None` keeps the dimension), as described in the module documentation. The
    /// result is new memory, never a view, and records `IndexBackward0`, which saves the index
    /// tensors as given.
    #[track_caller]
    pub(crate) fn index_tensors(&self, indices: &[Option<Tensor>]) -> Tensor {
        let data = self.data();
        let info = AdvancedIndex::new(&data, indices);
        let out = if data.device() == Device::Cpu {
            info.gather(&data)
        } else {
            info.check_bounds();
            Tensor::empty_on(info.out_shape.clone(), data.dtype, data.device())
        };
        formulas::index(self, indices, &out);
        out
    }

    /// ATen's `index_put`: a copy of `self` with `values` written where
    /// [`index_tensors`](Tensor::index_tensors) would read, added with `accumulate`. Records
    /// `IndexPutBackward0`.
    #[track_caller]
    pub(crate) fn index_put_tensors(
        &self,
        indices: &[Option<Tensor>],
        values: &Tensor,
        accumulate: bool,
    ) -> Tensor {
        let (data, vals) = (self.data(), values.data());
        let info = AdvancedIndex::new(&data, indices);
        info.check_values(&data, &vals);
        let out = if data.device() == Device::Cpu {
            info.scatter(&data, &vals, accumulate)
        } else {
            info.check_bounds();
            Tensor::empty_on(data.layout.shape.clone(), data.dtype, data.device())
        };
        formulas::index_put(self, indices, values, accumulate, &out);
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Tensor};

    fn idx(values: &[i64]) -> Option<Tensor> {
        Some(Tensor::from_slice(values, [values.len()]))
    }

    #[test]
    fn non_adjacent_scatter_writes_back_in_the_original_order() {
        // Indexed dimensions 0 and 2 move to the front; the work copy is contiguous in that
        // order and the result sees it in the original one.
        let x = crate::zeros([2, 3, 4]);
        let indices = [idx(&[0, 1]), None, idx(&[1, 2])];
        let values = crate::arange(6.0).view([2, 3]);
        for accumulate in [false, true] {
            let y = x.index_put_tensors(&indices, &values, accumulate);
            assert_eq!(y.shape(), [2, 3, 4]);
            assert_eq!(y.stride(), [12, 1, 3]);
            assert!(!y.is_contiguous() && y.view_info().is_none());
            let expected = crate::zeros([2, 3, 4]);
            for (n, (i, j)) in [(0, 1), (1, 2)].into_iter().enumerate() {
                for k in 0..3 {
                    expected
                        .select(0, i)
                        .select(0, k)
                        .select(0, j)
                        .fill_((n as i64 * 3 + k) as f64);
                }
            }
            assert_eq!(y.to_vec::<f32>(), expected.to_vec::<f32>());
        }
    }

    #[test]
    fn gather_and_scatter_read_strided_targets() {
        // A transposed slice with an offset: the gather reads it through its layout and the
        // scatter copies it before writing.
        let base = crate::arange(30.0).view([5, 6]);
        let x = base.narrow(0, 1, 3).transpose(0, 1);
        assert!(x.storage_offset() > 0 && !x.is_contiguous());
        let indices = [None, idx(&[2, 0])];
        let picked = x.index_tensors(&indices);
        assert_eq!(picked.shape(), [6, 2]);
        let columns = crate::stack(&[&x.index((.., 2)), &x.index((.., 0))], 1);
        assert_eq!(picked.to_vec::<f32>(), columns.to_vec::<f32>());
        let y = x.index_put_tensors(&indices, &crate::full([6, 2], -1.0), false);
        assert_eq!(base.to_vec::<f32>(), crate::arange(30.0).to_vec::<f32>());
        let expected = x.deep_clone();
        expected.select(1, 2).fill_(-1.0);
        expected.select(1, 0).fill_(-1.0);
        assert_eq!(y.to_vec::<f32>(), expected.to_vec::<f32>());
    }

    #[test]
    fn positions_wrap_negative_values_and_broadcast() {
        let x = crate::arange(12).view([3, 4]).to(DType::I64);
        let y = x.index_tensors(&[
            Some(Tensor::from_slice(&[-1i64, 0], [2, 1])),
            idx(&[3, -4, 1]),
        ]);
        assert_eq!(y.shape(), [2, 3]);
        assert_eq!(y.to_vec::<i64>(), [11, 8, 9, 3, 0, 1]);
    }

    #[test]
    fn graphs_release_saved_indices() {
        let x = crate::zeros([3]);
        x.requires_grad_(true);
        let index = idx(&[2, 0]).unwrap();
        let weak = index.downgrade();
        let y = x.index((&index,)).sum();
        drop(index);
        assert!(weak.upgrade().is_some(), "saved by the graph");
        y.backward();
        assert!(weak.upgrade().is_none(), "released by the backward pass");
    }
}
