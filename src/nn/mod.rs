//! Neural network modules, like PyTorch's `torch.nn`.
//!
//! A model is a struct whose fields are its [`Parameter`]s, buffers
//! ([`Tensor`](crate::Tensor)s), submodules and hyperparameters. It implements [`Module`], which
//! lists those members, and [`Member`], which lets it be a field of another module; it gets
//! PyTorch's module methods from `Module`:
//! [`parameters`](Module::parameters), [`state_dict`](Module::state_dict),
//! [`load_state_dict`](Module::load_state_dict), [`to`](Module::to), [`train`](Module::train),
//! [`eval`](Module::eval) and the others. They all take `&self`. The forward computation is an
//! ordinary method of the struct.
//!
//! ```
//! use homura::nn::{Member, Module, Parameter, Visitor};
//! use homura::{DType, Tensor};
//!
//! struct Mlp {
//!     hidden: Linear,
//!     output: Linear,
//! }
//!
//! struct Linear {
//!     weight: Parameter,
//!     bias: Parameter,
//! }
//!
//! impl Linear {
//!     fn new(inputs: usize, outputs: usize) -> Linear {
//!         Linear {
//!             weight: Parameter::new(homura::randn([outputs, inputs])),
//!             bias: Parameter::new(homura::zeros([outputs])),
//!         }
//!     }
//!
//!     fn forward(&self, x: &Tensor) -> Tensor {
//!         x.matmul(&self.weight.transpose(0, 1)) + &*self.bias
//!     }
//! }
//!
//! impl Mlp {
//!     fn forward(&self, x: &Tensor) -> Tensor {
//!         self.output.forward(&self.hidden.forward(x).relu())
//!     }
//! }
//!
//! impl Module for Linear {
//!     fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
//!         self.weight.visit_as("weight", v);
//!         self.bias.visit_as("bias", v);
//!     }
//! }
//!
//! impl Module for Mlp {
//!     fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
//!         self.hidden.visit_as("hidden", v);
//!         self.output.visit_as("output", v);
//!     }
//! }
//!
//! // How each reports itself as a field of another module: as a submodule.
//! impl Member for Linear {
//!     fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
//!         v.module(name, self, None);
//!     }
//! }
//!
//! impl Member for Mlp {
//!     fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
//!         v.module(name, self, None);
//!     }
//! }
//!
//! let model = Mlp { hidden: Linear::new(4, 8), output: Linear::new(8, 2) };
//! let keys: Vec<String> = model.named_parameters().into_iter().map(|(k, _)| k).collect();
//! assert_eq!(keys, ["hidden.weight", "hidden.bias", "output.weight", "output.bias"]);
//!
//! model.to(DType::F64);
//! let y = model.forward(&homura::ones_with([3, 4], DType::F64));
//! y.sum().backward();
//! assert_eq!(model.output.bias.grad().unwrap().to_vec::<f64>(), [3.0, 3.0]);
//! ```
//!
//! # Coming from PyTorch
//!
//! | PyTorch | homura |
//! | --- | --- |
//! | `nn.Parameter(t)`, `nn.Parameter(t, requires_grad=False)` | [`Parameter::new(t)`](Parameter::new), [`Parameter::with_requires_grad(t, false)`](Parameter::with_requires_grad) |
//! | `self.w = nn.Parameter(..)` in `__init__` | a `Parameter` field, reported by [`Module::visit`] (see [`Member`]) |
//! | `self.register_buffer("b", t)`, `persistent=False` | a [`Tensor`](crate::Tensor) field; [`Visitor::buffer`] with `persistent = false` |
//! | `nn.ModuleList`, `nn.ModuleDict` | `Vec<T>`, `BTreeMap<String, T>` fields |
//! | `self.training` | a [`Training`] field: `self.training.get()` |
//! | the same submodule assigned twice | an `Arc<T>` in both fields |
//! | `model.state_dict()`, `model.load_state_dict(sd, strict)` | [`model.state_dict()`](Module::state_dict), [`model.load_state_dict(&sd, strict)`](Module::load_state_dict), returning a `Result` |
//! | `model.to(torch.float64)`, `model.to("meta")` | [`model.to(DType::F64)`](Module::to), `model.to(Device::Meta)` |
//!
//! [`Module`] describes how members are named, ordered and deduplicated.
//!
//! # Differences from PyTorch
//!
//! - Submodules are values: a submodule is shared only through an `Arc`. Traversals list a
//!   shared submodule once, like PyTorch lists the same object once; the state dict has its
//!   members under every path, also like PyTorch.
//! - `BTreeMap` containers order their members by key, not by insertion like `nn.ModuleDict`.
//! - [`Module::load_state_dict`] and [`Module::to`] check everything they can before they
//!   change anything, so that their errors leave the module unchanged; PyTorch's
//!   `load_state_dict` writes the matching entries before it reports errors. This covers errors
//!   in their inputs: a panic for other reasons (out of memory) or another thread changing the
//!   module at the same time can still leave it partly changed.
//! - After [`Module::to`], graphs recorded before can't be run any more for the converted
//!   tensors; PyTorch runs them, and can write gradients of the old dtype into the converted
//!   parameters. The `.grad` of a converted tensor is replaced by a converted copy, where
//!   PyTorch converts the gradient tensor itself; one that has a history must be cleared before
//!   converting.
//! - Parameters are not a tensor subclass: [`Parameter`] derefs to
//!   [`Tensor`](crate::Tensor), so `&param` works where a `&Tensor` is expected, and `&*param`
//!   where an operator needs the tensor (`&*param * 2.0`).

mod module;
#[cfg(test)]
mod module_tests;
mod parameter;
mod state_dict;
mod training;

pub use module::{Member, Module, SharedId, Visitor};
pub use parameter::Parameter;
pub use state_dict::{LoadError, LoadReport, StateDict, StateDictIter};
pub use training::Training;
