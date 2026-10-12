//! The [`Tensor`] type.

use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use crate::autograd::{AutogradMeta, ViewInfo};
use crate::backend::cpu;
use crate::shape::{contiguous_strides, is_contiguous};
use crate::storage::Storage;
use crate::{DType, Device, Element, Shape, Strides};

mod convert;
mod create;
mod fmt;
mod index;
mod view;

pub use create::*;
pub use index::{Ellipsis, IndexItem, IndexSpec, NewAxis, TensorIndex};

/// An n-dimensional array with reference semantics, like a PyTorch tensor.
///
/// `Clone` copies the handle, not the data: the clone refers to the same tensor, and in-place
/// operations through either handle are visible through both, just like `y = x` in Python. Use
/// [`Tensor::deep_clone`] for PyTorch's `x.clone()`.
///
/// Programming errors (mismatched shapes, dtypes or devices) panic with PyTorch's error messages.
///
/// # Threads
///
/// `Tensor` is `Send + Sync`. Reading tensors from several threads and operating on distinct
/// tensors in parallel need no synchronization. In-place operations are not atomic, though: if
/// several threads mutate tensors that share storage, or one mutates while another reads, the
/// results are unspecified (updates can be lost) but memory-safe. As in PyTorch, synchronize such
/// accesses yourself, for example with a `Mutex`.
#[derive(Clone)]
pub struct Tensor(Arc<TensorImpl>);

struct TensorImpl {
    data: RwLock<TensorData>,
    /// `None` for tensors that don't take part in autograd. Lock it only briefly: never run a
    /// tensor operation or take another tensor's lock while holding it.
    autograd: Mutex<Option<AutogradMeta>>,
    /// Set when the tensor is created as a view of another one; never changes afterwards, so it
    /// is read without a lock.
    view: Option<ViewInfo>,
}

/// A weak handle to a tensor, for tests that check when tensors are destroyed.
#[cfg(test)]
pub(crate) struct WeakTensor(std::sync::Weak<TensorImpl>);

#[cfg(test)]
impl WeakTensor {
    pub(crate) fn upgrade(&self) -> Option<Tensor> {
        self.0.upgrade().map(Tensor)
    }
}

/// How a tensor's elements are placed in its storage.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    pub shape: Shape,
    pub strides: Strides,
    /// Offset of the first element, in elements.
    pub offset: usize,
}

/// A snapshot of a tensor's data: storage, layout and dtype.
#[derive(Clone, Debug)]
pub(crate) struct TensorData {
    pub storage: Arc<Storage>,
    pub layout: Layout,
    pub dtype: DType,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tensor>();
};

impl Layout {
    pub fn contiguous(shape: Shape) -> Layout {
        let strides = contiguous_strides(&shape);
        Layout {
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.shape, &self.strides)
    }

    /// This layout with a dimension of size 1 inserted at position `dim` (at most the number of
    /// dimensions).
    pub fn unsqueeze(&self, dim: usize) -> Layout {
        let mut l = self.clone();
        // The stride of a size-1 dimension is never used to address memory; fall back to 1 in
        // the unlikely case that PyTorch's choice doesn't fit.
        let stride = if dim < l.shape.len() {
            l.shape[dim].checked_mul(l.strides[dim]).unwrap_or(1)
        } else {
            1
        };
        l.shape.insert(dim, 1);
        l.strides.insert(dim, stride);
        l
    }

    /// Broadcasts this layout to `shape` by giving new and size-1 dimensions a stride of 0.
    pub fn expand_to(&self, shape: &Shape) -> Layout {
        let n = shape.len();
        let m = self.shape.len();
        assert!(
            m <= n,
            "cannot broadcast shape {:?} to {:?}",
            self.shape,
            shape
        );
        let mut strides = vec![0; n];
        for i in 0..m {
            let (src, dst) = (self.shape[i], shape[n - m + i]);
            if src == dst {
                strides[n - m + i] = self.strides[i];
            } else if src != 1 {
                panic!(
                    "The expanded size of the tensor ({dst}) must match the existing size ({src}) \
                     at non-singleton dimension {}. Target sizes: {:?}. Tensor sizes: {:?}",
                    n - m + i,
                    shape,
                    self.shape
                );
            }
        }
        Layout {
            shape: shape.clone(),
            strides: Strides::new(&strides),
            offset: self.offset,
        }
    }

    /// One past the largest element offset this layout touches (0 for an empty layout, which
    /// touches no memory), or `None` if that doesn't fit in `usize`.
    fn checked_end(&self) -> Option<usize> {
        if crate::shape::checked_num_elements(&self.shape)? == 0 {
            return Some(0);
        }
        self.shape
            .iter()
            .zip(self.strides.iter())
            .try_fold(self.offset, |acc, (&n, &s)| {
                acc.checked_add((n - 1).checked_mul(s)?)
            })?
            .checked_add(1)
    }

    /// The range of element offsets this layout can touch, or `None` if it has no elements.
    ///
    /// # Panics
    ///
    /// Panics if the range doesn't fit in `usize`; layouts of existing tensors always fit.
    pub fn span(&self) -> Option<(usize, usize)> {
        let end = self
            .checked_end()
            .expect("layout reaches beyond the addressable range");
        (end > 0).then_some((self.offset, end))
    }

    /// Panics unless every element of this layout, with elements of `item_size` bytes, lies
    /// inside a storage of `nbytes` bytes and the number of elements is representable.
    pub fn check_bounds(&self, item_size: usize, nbytes: usize) {
        let needed = self
            .checked_end()
            .and_then(|end| end.checked_mul(item_size));
        if needed.is_none_or(|n| n > nbytes) {
            panic!(
                "setStorage: sizes {:?}, strides {:?}, storage offset {}, and itemsize \
                 {item_size} requiring a storage size of {} are out of bounds for storage of size \
                 {nbytes}",
                self.shape,
                self.strides,
                self.offset,
                needed.map_or("more than usize::MAX".to_string(), |n| n.to_string()),
            );
        }
    }

    /// The storage offset of index `i` along dimension `dim`.
    ///
    /// # Panics
    ///
    /// Panics if the offset doesn't fit in `usize`, which only happens for an index one past
    /// the end of a dimension with an unusually large stride.
    pub fn offset_at(&self, dim: usize, i: usize) -> usize {
        i.checked_mul(self.strides[dim])
            .and_then(|o| o.checked_add(self.offset))
            .expect("storage offset overflows usize")
    }

    /// Whether distinct indices of this layout might refer to the same storage element.
    ///
    /// Conservative: `false` means the elements are proven distinct. Dimensions of size greater
    /// than 1 are sorted by stride; the layout is distinct if each stride exceeds the largest
    /// offset reachable through the dimensions with smaller strides. This accepts contiguous and
    /// permuted tensors, stepped slices and empty layouts, but rejects some exotic `as_strided`
    /// layouts that happen to be distinct, such as sizes `[3, 2]` with strides `[2, 3]`.
    pub fn may_overlap_internally(&self) -> bool {
        if self.shape.contains(&0) {
            return false;
        }
        let mut dims: Vec<(usize, usize)> = self
            .shape
            .iter()
            .zip(self.strides.iter())
            .filter(|&(&n, _)| n > 1)
            .map(|(&n, &s)| (s, n))
            .collect();
        dims.sort_unstable();
        // The largest offset (relative to the first element) reachable so far.
        let mut reach: usize = 0;
        for (stride, n) in dims {
            if stride <= reach {
                return true;
            }
            // Fits: the layout of an existing tensor spans less than `usize::MAX`.
            reach += (n - 1) * stride;
        }
        false
    }
}

impl TensorData {
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Whether `self` and `other` may refer to the same memory without being the exact same view.
    pub fn partially_overlaps(&self, other: &TensorData) -> bool {
        if !Arc::ptr_eq(&self.storage, &other.storage) {
            return false;
        }
        let (Some((a0, a1)), Some((b0, b1))) = (self.layout.span(), other.layout.span()) else {
            return false;
        };
        let disjoint = a1 * self.dtype.size_in_bytes() <= b0 * other.dtype.size_in_bytes()
            || b1 * other.dtype.size_in_bytes() <= a0 * self.dtype.size_in_bytes();
        let same = self.dtype == other.dtype
            && self.layout.offset == other.layout.offset
            && self.layout.shape == other.layout.shape
            && self.layout.strides == other.layout.strides;
        !disjoint && !same
    }

    /// The same data broadcast to `shape`.
    pub fn expand_to(&self, shape: &Shape) -> TensorData {
        TensorData {
            storage: self.storage.clone(),
            layout: self.layout.expand_to(shape),
            dtype: self.dtype,
        }
    }
}

impl Tensor {
    pub(crate) fn from_data(data: TensorData) -> Tensor {
        Tensor::from_parts(data, None, None)
    }

    pub(crate) fn from_parts(
        data: TensorData,
        autograd: Option<AutogradMeta>,
        view: Option<ViewInfo>,
    ) -> Tensor {
        Tensor(Arc::new(TensorImpl {
            data: RwLock::new(data),
            autograd: Mutex::new(autograd),
            view,
        }))
    }

    /// The autograd state, locked. See `TensorImpl::autograd` for the locking rules.
    pub(crate) fn autograd_meta(&self) -> MutexGuard<'_, Option<AutogradMeta>> {
        self.0.autograd.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// How this tensor is a view of another one, if it is.
    pub(crate) fn view_info(&self) -> Option<&ViewInfo> {
        self.0.view.as_ref()
    }

    /// A snapshot of the storage, layout and dtype.
    pub(crate) fn data(&self) -> TensorData {
        self.0
            .data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// A new contiguous, zero-filled tensor.
    ///
    /// # Panics
    ///
    /// Panics if the size in bytes exceeds `isize::MAX`, on every device.
    pub(crate) fn empty_on(shape: Shape, dtype: DType, device: Device) -> Tensor {
        let nbytes = shape
            .num_elements()
            .checked_mul(dtype.size_in_bytes())
            .filter(|&n| n <= isize::MAX as usize)
            .unwrap_or_else(|| {
                panic!("a tensor of shape {shape:?} and dtype {dtype} would be too large")
            });
        let storage = Arc::new(Storage::new(nbytes, device));
        Tensor::from_data(TensorData {
            storage,
            layout: Layout::contiguous(shape),
            dtype,
        })
    }

    /// A view of this tensor with `layout`, as the public view operations create it: unlike
    /// [`with_layout`](Tensor::with_layout), the result records its root base (see `ViewInfo`),
    /// so that it is known to share storage with it.
    ///
    /// # Panics
    ///
    /// Panics if the layout reaches outside the storage.
    pub(crate) fn make_view(&self, layout: Layout) -> Tensor {
        crate::autograd::view_of(self, layout)
    }

    /// A new tensor that shares this tensor's storage with a different layout, without being
    /// recorded as a view of it: for an operation that reshapes a result it has just allocated,
    /// see `unsafe_view`. The public view operations use [`make_view`](Tensor::make_view).
    ///
    /// # Panics
    ///
    /// Panics if the layout reaches outside the storage. `make_view` checks the same, so the CPU
    /// kernels can rely on all layouts lying inside their storage.
    pub(crate) fn with_layout(&self, layout: Layout) -> Tensor {
        let data = self.data();
        layout.check_bounds(data.dtype.size_in_bytes(), data.storage.nbytes());
        Tensor::from_data(TensorData {
            storage: data.storage,
            layout,
            dtype: data.dtype,
        })
    }

    /// Creates a tensor from host data with an exact dtype.
    pub(crate) fn from_host<T: Element>(values: &[T], shape: Shape, device: Device) -> Tensor {
        assert_eq!(
            values.len(),
            shape.num_elements(),
            "{} values can't fill a tensor of shape {:?}",
            values.len(),
            shape
        );
        let t = Tensor::empty_on(shape, T::DTYPE, device);
        if device == Device::Cpu {
            cpu::write_host(&t.data(), values);
        }
        t
    }

    pub(crate) fn layout(&self) -> Layout {
        self.0
            .data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .layout
            .clone()
    }

    /// The size of each dimension.
    pub fn shape(&self) -> Shape {
        self.0
            .data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .layout
            .shape
            .clone()
    }

    /// The size of dimension `dim` (negative values count from the end).
    pub fn size(&self, dim: i64) -> usize {
        let shape = self.shape();
        shape[crate::shape::normalize_dim(dim, shape.len())]
    }

    /// The strides of each dimension, in elements.
    pub fn stride(&self) -> Strides {
        self.layout().strides
    }

    /// Offset of the first element in the underlying storage, in elements.
    pub fn storage_offset(&self) -> usize {
        self.layout().offset
    }

    /// Number of dimensions (PyTorch's `dim()`).
    pub fn num_dims(&self) -> usize {
        self.shape().len()
    }

    /// Total number of elements (PyTorch's `numel()`).
    pub fn num_elements(&self) -> usize {
        self.shape().num_elements()
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.0.data.read().unwrap_or_else(|e| e.into_inner()).dtype
    }

    /// The device holding the data.
    pub fn device(&self) -> Device {
        self.0
            .data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .device()
    }

    /// Whether the elements are laid out in row-major order without gaps.
    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Whether the dtype is a floating point type.
    pub fn is_floating_point(&self) -> bool {
        self.dtype().is_floating_point()
    }

    /// Size of one element in bytes.
    pub fn element_size(&self) -> usize {
        self.dtype().size_in_bytes()
    }

    /// Whether `self` and `other` are handles to the same tensor (Python's `x is y`).
    pub fn is_same(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    #[cfg(test)]
    pub(crate) fn downgrade(&self) -> WeakTensor {
        WeakTensor(Arc::downgrade(&self.0))
    }

    /// The version counter of the storage, bumped by every in-place operation.
    pub(crate) fn version(&self) -> u64 {
        self.0
            .data
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .storage
            .version()
    }

    /// Copies the data into a new contiguous tensor (PyTorch's `x.clone()`). The copy is
    /// connected to the graph: gradients flow through it to `self`.
    pub fn deep_clone(&self) -> Tensor {
        let out = Tensor::from_data(crate::ops::clone_data(&self.data()));
        crate::autograd::formulas::deep_clone(self, &out);
        out
    }
}
