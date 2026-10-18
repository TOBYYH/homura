//! homura: a PyTorch-like tensor computing framework written in Rust, with hardware backends built
//! on the vendors' latest official development tools.
//!
//! The design follows PyTorch: [`Tensor`] has reference semantics, operations broadcast and
//! promote dtypes like PyTorch, and tensor methods use PyTorch's names. Everything else follows
//! Rust conventions.
//!
//! ```
//! use homura::{DType, Tensor};
//!
//! let x = homura::tensor([[1., 2.], [3., 4.]]);
//! let w = homura::ones([2, 3]);
//! let y = x.matmul(&w).relu() * 0.5;
//! assert_eq!(y.shape(), [2, 3]);
//! assert_eq!(y.sum().item::<f32>(), 15.0);
//! ```
//!
//! [`autograd`] computes gradients like PyTorch's autograd, including gradients of gradients:
//! operations on tensors that require grad record a backward graph, which
//! [`Tensor::backward`] or [`autograd::grad`] runs. [`nn`] has PyTorch's module machinery:
//! parameters, state dicts, loading and conversion of module trees; [`optim`] has optimizers that
//! update the parameters.
//!
//! This project is in early development: only the CPU reference backend and the meta device are
//! implemented, and the API is not stable yet.

// `#[derive(nn::Module)]` in this crate, and code shared with the integration tests
// (`src/autograd/test_support`), name the crate `homura`.
extern crate self as homura;

// The README's examples run as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub mod autograd;
mod backend;
mod defaults;
mod device;
mod dtype;
mod error;
mod io;
pub mod nn;
mod ops;
pub mod optim;
mod shape;
mod storage;
mod tensor;

pub use half::{bf16, f16};

pub use autograd::{GradModeGuard, enable_grad, is_grad_enabled, no_grad, set_grad_enabled};
pub use defaults::{
    DeviceScope, default_device, default_dtype, device_scope, set_default_device, set_default_dtype,
};
pub use device::Device;
pub use dtype::{DType, Element, Scalar, can_cast, promote_types};
pub use error::{Error, Result};
pub use io::{
    load_safetensors, load_safetensors_from_bytes, save_safetensors, save_safetensors_to_bytes,
};
pub use ops::{
    GeluApproximate, Number, Operand, concat, manual_seed, rand, rand_like, rand_with, randint,
    randint_with, randn, randn_like, randn_with, stack, where_,
};
pub use shape::{Dims, IntoShape, IntoShapeSpec, Shape, Strides};
pub use tensor::{
    Ellipsis, IndexItem, IndexSpec, NestedData, NewAxis, Tensor, TensorIndex, TensorOptions,
    arange, arange_step, arange_step_with, arange_with, eye, eye_with, full, full_like, full_with,
    linspace, linspace_with, ones, ones_like, ones_with, tensor, tensor_with, zeros, zeros_like,
    zeros_with,
};
