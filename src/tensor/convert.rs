//! Moving data between tensors, dtypes, devices and the host.

use super::{Tensor, TensorOptions};
use crate::backend::cpu;
use crate::ops::cast_data;
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
        let out = match (data.device(), device) {
            (_, Device::Meta) => Tensor::empty_on(data.layout.shape.clone(), dtype, device),
            (Device::Meta, _) => panic!("Cannot copy out of meta tensor; no data!"),
            (Device::Cpu, Device::Cpu) => Tensor::from_data(cast_data(&data, dtype)),
            (from, to) => panic!("copying from {from} to {to} is not supported yet"),
        };
        crate::autograd::formulas::to_copy(self, &out);
        out
    }
}
