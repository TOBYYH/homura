//! Moving data between tensors, dtypes, devices and the host.

use super::{Tensor, TensorData, TensorOptions};
use crate::backend;
use crate::ops::{try_cast_data, try_clone_data};
use crate::storage::AllocError;
use crate::{DType, Device, Element};

impl Tensor {
    #[track_caller]
    fn host_data<T: Element>(&self, what: &str) -> crate::tensor::TensorData {
        let data = self.data();
        let readable = data.dtype == T::DTYPE
            || T::DTYPE == DType::F32 && matches!(data.dtype, DType::F16 | DType::BF16);
        if !readable {
            panic!("{what}: can't read a {} tensor as {}", data.dtype, T::DTYPE);
        }
        match data.device() {
            Device::Meta => panic!("{what}: meta tensors have no data"),
            _ => data,
        }
    }

    /// All elements in row-major order. This is a synchronization point.
    ///
    /// # Panics
    ///
    /// Panics if `T` doesn't match the dtype; `f16` and `bf16` tensors may also be read as `f32`.
    #[track_caller]
    pub fn to_vec<T: Element>(&self) -> Vec<T> {
        backend::read_host(&self.host_data::<T>("to_vec"))
    }

    /// The value of a single-element tensor. This is a synchronization point.
    ///
    /// # Panics
    ///
    /// Panics if the tensor doesn't have exactly one element or if `T` doesn't match the dtype
    /// (as in [`to_vec`](Tensor::to_vec)).
    #[track_caller]
    pub fn item<T: Element>(&self) -> T {
        let n = self.num_elements();
        if n != 1 {
            panic!("a Tensor with {n} elements cannot be converted to Scalar");
        }
        backend::read_host(&self.host_data::<T>("item"))[0]
    }

    /// Converts to another dtype and/or device: `x.to(DType::F16)`, `x.to(Device::Meta)` or
    /// `x.to((DType::F16, device))`. Returns `self` (a handle to the same tensor) when nothing
    /// changes.
    ///
    /// Between the CPU and a ROCm device the data moves through pageable host memory and the
    /// call returns when it has arrived; between two ROCm devices it moves through the host.
    ///
    /// # Panics
    ///
    /// Panics if a device can't hold the result, with the message of [`Error::OutOfMemory`]
    /// (the CPU aborts instead, as for any failed allocation), and for conversions out of a meta
    /// tensor. [`try_to`](Tensor::try_to) returns the out-of-memory error instead.
    ///
    /// [`Error::OutOfMemory`]: crate::Error::OutOfMemory
    #[track_caller]
    pub fn to(&self, options: impl Into<TensorOptions>) -> Tensor {
        match self.convert(options.into()) {
            Ok(t) => t,
            Err(e) => e.raise(),
        }
    }

    /// [`to`](Tensor::to), returning [`Error::OutOfMemory`](crate::Error::OutOfMemory) if a
    /// device refuses the memory of the conversion: the result's, a contiguous copy on the
    /// source device or the host staging between devices, or the memory of a ROCm device's
    /// first use. The tensor is unchanged when it fails.
    ///
    /// # Panics
    ///
    /// Panics like [`to`](Tensor::to) for everything but running out of memory.
    #[track_caller]
    pub fn try_to(&self, options: impl Into<TensorOptions>) -> crate::Result<Tensor> {
        self.convert(options.into()).map_err(|e| e.into_error())
    }

    #[track_caller]
    fn convert(&self, o: TensorOptions) -> Result<Tensor, AllocError> {
        let data = self.data();
        let dtype = o.dtype.unwrap_or(data.dtype);
        let device = o.device.unwrap_or(data.device());
        if dtype == data.dtype && device == data.device() {
            return Ok(self.clone());
        }
        let out = Tensor::from_data(try_convert_data(&data, dtype, device)?);
        crate::autograd::formulas::to_copy(self, &out);
        Ok(out)
    }

    /// Replaces the storage, layout and dtype of this tensor with `data`, which has the same
    /// shape (PyTorch's `x.data = new`), for `Module::to`. Every handle to the tensor sees the
    /// new data; its `requires_grad` and `.grad` stay. Views and `detach` aliases created before
    /// keep the old storage.
    ///
    /// Graphs recorded before can't use the new data by accident: the replacement increments
    /// the tensor's data generation, which a graph that saved this tensor, the accumulator an
    /// old graph holds (the cached one is dropped) and views created before compare with the
    /// value they recorded (see `AccumulateGrad`, `SavedTensor` and the staleness of views). The
    /// storage of `data` must be new (no other tensor uses it); its version also starts one past
    /// the old storage's.
    ///
    /// # Panics
    ///
    /// Panics (an internal error) if the tensor has a history or is a view, or if the shapes
    /// differ: callers check these first.
    pub(crate) fn set_data(&self, data: TensorData) {
        self.replace_leaf_data(data, true);
    }

    /// Replaces the storage, layout and dtype of this tensor with `data`, which has the same
    /// shape and may share its storage with other tensors, for loading by assignment
    /// (`load_state_dict(assign=True)`). Like [`set_data`](Tensor::set_data), except that the
    /// version of the storage of `data` is left alone: the storage belongs to the source too, and
    /// changing its version would fail or revive the version checks of the source's graphs. Old
    /// graphs and views of this tensor are refused by the data generation alone.
    ///
    /// # Panics
    ///
    /// Like [`set_data`](Tensor::set_data).
    pub(crate) fn assign_data(&self, data: TensorData) {
        self.replace_leaf_data(data, false);
    }

    fn replace_leaf_data(&self, data: TensorData, continue_version: bool) {
        assert!(
            self.view_info().is_none() && self.is_leaf(),
            "internal error: replacing the data of a view or a tensor with a history"
        );
        let old = self.replace_data(data, continue_version);
        drop(old);
        crate::autograd::forget_accumulator(self);
    }
}

/// Why converting data from device `from` to device `to` is impossible, or `None` if it works:
/// only data out of a meta tensor can't be converted.
pub(crate) fn conversion_error(from: Device, to: Device) -> Option<String> {
    match (from, to) {
        (_, Device::Meta) => None,
        (Device::Meta, _) => Some("Cannot copy out of meta tensor; no data!".into()),
        _ => None,
    }
}

/// `data` converted to `dtype` on `device`, in new contiguous storage that nothing else uses.
///
/// # Panics
///
/// Panics with [`conversion_error`]'s message if the devices don't allow the conversion, and if
/// a device refuses the memory (see [`try_convert_data`]).
#[track_caller]
pub(crate) fn convert_data(data: &TensorData, dtype: DType, device: Device) -> TensorData {
    match try_convert_data(data, dtype, device) {
        Ok(d) => d,
        Err(e) => e.raise(),
    }
}

/// [`convert_data`], returning the refusal of any memory it needs: the result, a contiguous
/// copy on the source device, the host staging. Between devices the dtype changes on the CPU,
/// before uploading or after downloading, with the CPU's conversions (the GPU's are the same).
#[track_caller]
pub(crate) fn try_convert_data(
    data: &TensorData,
    dtype: DType,
    device: Device,
) -> Result<TensorData, AllocError> {
    if let Some(e) = conversion_error(data.device(), device) {
        panic!("{e}");
    }
    let from = data.device();
    if device == Device::Meta {
        return Ok(Tensor::try_empty_on(data.layout.shape.clone(), dtype, device)?.data());
    }
    if from == device {
        return if dtype == data.dtype {
            try_clone_data(data)
        } else {
            try_cast_data(data, dtype)
        };
    }
    // The result first: a request that the target device can't hold fails before any staging.
    let out = Tensor::try_empty_on(data.layout.shape.clone(), dtype, device)?.data();
    if from == Device::Cpu {
        // `try_cast_data` keeps `data` when the dtype doesn't change.
        backend::try_transfer_into(&out, &try_cast_data(data, dtype)?)?;
    } else if device == Device::Cpu && dtype == data.dtype {
        backend::try_transfer_into(&out, data)?;
    } else if device == Device::Cpu {
        // Downloaded as it is, converted by the CPU's copy.
        backend::copy(&out, &backend::try_transfer(data, Device::Cpu)?);
    } else {
        // Between two ROCm devices through the host, converted there.
        let host = try_cast_data(&backend::try_transfer(data, Device::Cpu)?, dtype)?;
        backend::try_transfer_into(&out, &host)?;
    }
    Ok(out)
}
