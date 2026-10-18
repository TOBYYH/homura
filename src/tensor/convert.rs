//! Moving data between tensors, dtypes, devices and the host.

use super::{Tensor, TensorData, TensorOptions};
use crate::backend::cpu;
use crate::ops::{cast_data, clone_data};
use crate::{DType, Device, Element};

impl Tensor {
    fn host_data<T: Element>(&self, what: &str) -> crate::tensor::TensorData {
        let data = self.data();
        let readable = data.dtype == T::DTYPE
            || T::DTYPE == DType::F32 && matches!(data.dtype, DType::F16 | DType::BF16);
        if !readable {
            panic!("{what}: can't read a {} tensor as {}", data.dtype, T::DTYPE);
        }
        match data.device() {
            Device::Cpu => data,
            Device::Meta => panic!("{what}: meta tensors have no data"),
            d => panic!("{what}: reading data from {d} is not supported yet"),
        }
    }

    /// All elements in row-major order. This is a synchronization point.
    ///
    /// # Panics
    ///
    /// Panics if `T` doesn't match the dtype; `f16` and `bf16` tensors may also be read as `f32`.
    pub fn to_vec<T: Element>(&self) -> Vec<T> {
        cpu::read_host(&self.host_data::<T>("to_vec"))
    }

    /// The value of a single-element tensor. This is a synchronization point.
    ///
    /// # Panics
    ///
    /// Panics if the tensor doesn't have exactly one element or if `T` doesn't match the dtype
    /// (as in [`to_vec`](Tensor::to_vec)).
    pub fn item<T: Element>(&self) -> T {
        let n = self.num_elements();
        if n != 1 {
            panic!("a Tensor with {n} elements cannot be converted to Scalar");
        }
        cpu::read_host(&self.host_data::<T>("item"))[0]
    }

    /// Converts to another dtype and/or device: `x.to(DType::F16)`, `x.to(Device::Meta)` or
    /// `x.to((DType::F16, device))`. Returns `self` (a handle to the same tensor) when nothing
    /// changes.
    pub fn to(&self, options: impl Into<TensorOptions>) -> Tensor {
        let o = options.into();
        let data = self.data();
        let dtype = o.dtype.unwrap_or(data.dtype);
        let device = o.device.unwrap_or(data.device());
        if dtype == data.dtype && device == data.device() {
            return self.clone();
        }
        let out = Tensor::from_data(convert_data(&data, dtype, device));
        crate::autograd::formulas::to_copy(self, &out);
        out
    }

    /// Replaces the storage, layout and dtype of this tensor with `data`, which has the same
    /// shape (PyTorch's `x.data = new`), for `Module::to`. Every handle to the tensor sees the
    /// new data; its `requires_grad` and `.grad` stay. Views and `detach` aliases created before
    /// keep the old storage.
    ///
    /// Graphs recorded before can't use the new data by accident: the storage of `data` must be
    /// new (no other tensor uses it), and its version starts one past the old storage's, so a
    /// graph that saved this tensor fails its version check; the cached accumulator is dropped,
    /// and the old one, which an old graph still holds, refuses gradients whose metadata no
    /// longer match (see `AccumulateGrad`); views of the old storage become stale.
    ///
    /// # Panics
    ///
    /// Panics (an internal error) if the tensor has a history or is a view, or if the shapes
    /// differ: callers check these first.
    pub(crate) fn set_data(&self, data: TensorData) {
        assert!(
            self.view_info().is_none() && self.is_leaf(),
            "internal error: set_data on a view or a tensor with a history"
        );
        let old = {
            let mut guard = self.0.data.write().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                guard.layout.shape, data.layout.shape,
                "internal error: set_data with another shape"
            );
            data.storage.continue_version(guard.storage.version());
            std::mem::replace(&mut *guard, data)
        };
        drop(old);
        crate::autograd::forget_accumulator(self);
    }
}

/// Why converting data from device `from` to device `to` is impossible, or `None` if it works.
pub(crate) fn conversion_error(from: Device, to: Device) -> Option<String> {
    match (from, to) {
        (_, Device::Meta) | (Device::Cpu, Device::Cpu) => None,
        (Device::Meta, _) => Some("Cannot copy out of meta tensor; no data!".into()),
        (from, to) => Some(format!("copying from {from} to {to} is not supported yet")),
    }
}

/// `data` converted to `dtype` on `device`, in new contiguous storage that nothing else uses.
///
/// # Panics
///
/// Panics with [`conversion_error`]'s message if the devices don't allow the conversion.
pub(crate) fn convert_data(data: &TensorData, dtype: DType, device: Device) -> TensorData {
    if let Some(e) = conversion_error(data.device(), device) {
        panic!("{e}");
    }
    if device == Device::Meta {
        Tensor::empty_on(data.layout.shape.clone(), dtype, device).data()
    } else if dtype == data.dtype {
        clone_data(data)
    } else {
        cast_data(data, dtype)
    }
}
