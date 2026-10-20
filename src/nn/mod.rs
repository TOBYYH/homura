//! Neural network modules, like PyTorch's `torch.nn`.
//!
//! A model is a struct whose fields are its [`Parameter`]s, buffers
//! ([`Tensor`](crate::Tensor)s), submodules and hyperparameters. `#[derive(Module)]` implements
//! [`Module`] for it, which lists those members by field name, and [`Member`], which lets it be a
//! field of another module. It gets PyTorch's module methods from `Module`:
//! [`parameters`](Module::parameters), [`state_dict`](Module::state_dict),
//! [`load_state_dict`](Module::load_state_dict), [`to`](Module::to), [`train`](Module::train),
//! [`eval`](Module::eval) and the others. They all take `&self`. The forward computation is an
//! ordinary method of the struct.
//!
//! ```
//! use homura::nn::{Linear, Module};
//! use homura::{DType, Tensor};
//!
//! #[derive(Module)]
//! struct Mlp {
//!     hidden: Linear,
//!     output: Linear,
//! }
//!
//! impl Mlp {
//!     fn new() -> Mlp {
//!         Mlp { hidden: Linear::new(4, 8), output: Linear::new(8, 2) }
//!     }
//!
//!     fn forward(&self, x: &Tensor) -> Tensor {
//!         self.output.forward(&self.hidden.forward(x).relu())
//!     }
//! }
//!
//! let model = Mlp::new();
//! let keys: Vec<String> = model.named_parameters().into_iter().map(|(k, _)| k).collect();
//! assert_eq!(keys, ["hidden.weight", "hidden.bias", "output.weight", "output.bias"]);
//!
//! model.to(DType::F64);
//! let y = model.forward(&homura::ones_with([3, 4], DType::F64));
//! y.sum().backward();
//! let bias = model.output.bias.as_ref().unwrap();
//! assert_eq!(bias.grad().unwrap().to_vec::<f64>(), [3.0, 3.0]);
//! ```
//!
//! # Deriving `Module`
//!
//! The type of each field decides what it is (see [`Member`]): a `Parameter` is a parameter, a
//! `Tensor` a buffer, a [`Training`] flag the module's train / eval mode, a struct that derives
//! `Module` a submodule; `Option`, `Vec`, `BTreeMap`, `Box` and `Arc` of those are what they
//! hold, and numbers, strings and other plain values are hyperparameters, which are not listed.
//! A unit struct is a module without members, such as an activation. Attributes change what a
//! field is:
//!
//! ```
//! use std::rc::Rc;
//!
//! use homura::nn::{Module, Parameter, Training};
//! use homura::Tensor;
//!
//! #[derive(Module)]
//! struct Norm {
//!     #[module(rename = "weight")]
//!     scale: Parameter,
//!     /// A buffer that is not part of the state dict (`persistent=False`); `to` converts it.
//!     #[module(non_persistent)]
//!     cache: Option<Tensor>,
//!     /// Not a member: its type doesn't implement `Member`.
//!     #[module(skip)]
//!     log: Rc<Vec<String>>,
//!     /// Only modules whose computation depends on the mode need the flag.
//!     training: Training,
//!     eps: f64,
//! }
//!
//! let norm = Norm {
//!     scale: Parameter::new(homura::ones([4])),
//!     cache: Some(homura::zeros([4])),
//!     log: Rc::default(),
//!     training: Training::default(),
//!     eps: 1e-5,
//! };
//! assert_eq!(norm.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
//! assert_eq!(norm.named_buffers()[0].0, "cache");
//! norm.eval();
//! assert!(!norm.training.get());
//! ```
//!
//! Generic structs derive `Module` too; the implementations require `Member` for the fields whose
//! types refer to a type parameter, except fields that contain the struct itself (the children
//! of a `Tree<T>`). Implement `Module` by hand for structs that are recursive through another
//! generic struct. With homura renamed in `Cargo.toml`, give its path with
//! `#[module(crate = "path")]` on the struct. To report members in another way, implement
//! `Module` and `Member` by hand (see [`Module`]).
//!
//! # Layers
//!
//! [`Linear`], [`Embedding`], [`LayerNorm`], [`RmsNorm`], [`Dropout`] and the activations
//! [`Relu`], [`Gelu`], [`Silu`], [`Tanh`], [`Sigmoid`] and [`Softmax`] are modules with a
//! `forward` method and implement [`Forward`]; [`Sequential`] applies such layers one after
//! another. [`functional`] has their computations as functions of tensors, and [`init`] the
//! initializations that layers use, with PyTorch's formulas. [`Dropout`] depends on the mode:
//! it has a [`Training`] flag, which [`Module::eval`] clears.
//!
//! The losses are functions in [`functional`]: [`mse_loss`](functional::mse_loss),
//! [`nll_loss`](functional::nll_loss), [`cross_entropy`](functional::cross_entropy) and
//! [`binary_cross_entropy_with_logits`](functional::binary_cross_entropy_with_logits), with a
//! [`Reduction`](functional::Reduction).
//!
//! # Coming from PyTorch
//!
//! | PyTorch | homura |
//! | --- | --- |
//! | `nn.Parameter(t)`, `nn.Parameter(t, requires_grad=False)` | [`Parameter::new(t)`](Parameter::new), [`Parameter::with_requires_grad(t, false)`](Parameter::with_requires_grad) |
//! | `class Net(nn.Module)` | `#[derive(Module)] struct Net` |
//! | `self.w = nn.Parameter(..)` in `__init__` | a `Parameter` field |
//! | `self.register_buffer("b", t)`, `persistent=False` | a [`Tensor`](crate::Tensor) field, `#[module(non_persistent)]` |
//! | `nn.Sequential(a, b)` | [`Sequential::new().append(a).append(b)`](Sequential) |
//! | `nn.Linear(4, 3)`, `nn.Linear(4, 3, bias=False)` | [`Linear::new(4, 3)`](Linear::new), [`Linear::new(4, 3).bias(false)`](Linear::bias) |
//! | `nn.ReLU()`, `nn.GELU(approximate="tanh")`, `nn.Softmax(dim=-1)` | [`Relu`], [`Gelu::new().approximate(GeluApproximate::Tanh)`](Gelu::approximate), [`Softmax::new(-1)`](Softmax::new) |
//! | `F.linear(x, w, b)`, `F.relu(x)` | [`functional::linear(&x, &w, Some(&b))`](functional::linear), [`functional::relu(&x)`](functional::relu) or `x.relu()` |
//! | `F.cross_entropy(x, t)`, `F.cross_entropy(x, t, weight=w, label_smoothing=0.1)` | [`functional::cross_entropy(&x, &t, Default::default())`](functional::cross_entropy), `cross_entropy(&x, &t, CrossEntropyOptions { weight: Some(w), label_smoothing: 0.1, ..Default::default() })` |
//! | `F.nll_loss(x, t)`, `F.mse_loss(x, y, reduction="sum")` | [`functional::nll_loss(&x, &t, None, -100, Reduction::Mean)`](functional::nll_loss), [`functional::mse_loss(&x, &y, Reduction::Sum)`](functional::mse_loss) |
//! | `F.binary_cross_entropy_with_logits(x, y, pos_weight=p)` | [`functional::binary_cross_entropy_with_logits(&x, &y, None, Reduction::Mean, Some(&p))`](functional::binary_cross_entropy_with_logits) |
//! | `nn.Embedding(10, 4, padding_idx=0)`, `F.embedding(i, w)` | [`Embedding::new(10, 4).padding_idx(0)`](Embedding::padding_idx), [`functional::embedding(&i, &w, None)`](functional::embedding) |
//! | `nn.LayerNorm(4)`, `nn.LayerNorm(4, elementwise_affine=False)`, `F.layer_norm(x, [4])` | [`LayerNorm::new([4])`](LayerNorm::new), [`LayerNorm::new([4]).elementwise_affine(false)`](LayerNorm::elementwise_affine), [`functional::layer_norm(&x, [4], None, None, 1e-5)`](functional::layer_norm) |
//! | `nn.RMSNorm(4, eps=1e-6)`, `F.rms_norm(x, [4])` | [`RmsNorm::new([4]).eps(1e-6)`](RmsNorm::eps), [`functional::rms_norm(&x, [4], None, None)`](functional::rms_norm) |
//! | `F.pad(x, (1, 1, 0, 2), value=v)` | [`functional::pad(&x, &[1, 1, 0, 2], v)`](functional::pad) |
//! | `F.conv2d(x, w, b, stride=2, padding=1)` | [`functional::conv2d(&x, &w, Some(&b), Conv2dOptions { stride: [2, 2], padding: 1.into(), ..Default::default() })`](functional::conv2d) |
//! | `F.max_pool2d(x, 2)`, `F.max_pool2d(x, 3, 2, 1, return_indices=True)` | [`functional::max_pool2d(&x, MaxPool2dOptions::new([2, 2]))`](functional::max_pool2d), [`functional::max_pool2d_with_indices(&x, MaxPool2dOptions { stride: Some([2, 2]), padding: [1, 1], ..MaxPool2dOptions::new([3, 3]) })`](functional::max_pool2d_with_indices) |
//! | `F.adaptive_avg_pool2d(x, (1, 1))` | [`functional::adaptive_avg_pool2d(&x, [1, 1])`](functional::adaptive_avg_pool2d) |
//! | `nn.Dropout(0.1)`, `F.dropout(x, 0.1, training)` | [`Dropout::new(0.1)`](Dropout::new), [`functional::dropout(&x, 0.1, training)`](functional::dropout) |
//! | `nn.init.kaiming_uniform_(w, a=math.sqrt(5))` | [`init::kaiming_uniform_(&w, 5f64.sqrt(), FanMode::FanIn, Nonlinearity::LeakyRelu)`](init::kaiming_uniform_) |
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
//! - Initializations, layer constructors and dropout draw from homura's random number
//!   generator, whose numbers differ from PyTorch's for the same seed; load a state dict to start
//!   from PyTorch's values.
//! - [`Embedding`] has no `max_norm`, `scale_grad_by_freq` or `sparse`. Its gradient adds the
//!   rows of repeated indices in `F64` and rounds once, where PyTorch rounds after each addition;
//!   [`layer_norm`](functional::layer_norm) computes its statistics with `F64` sums too, and
//!   adds the rows of the weight and bias gradients without rounding them to half precision in
//!   between. With create graph, the gradients of the gradients of a layer norm are recorded as
//!   ordinary operations, where PyTorch has a `NativeLayerNormBackwardBackward0` node; they are
//!   also right for a bias without a weight, which PyTorch gets wrong.
//! - Parameters are not a tensor subclass: [`Parameter`] derefs to
//!   [`Tensor`](crate::Tensor), so `&param` works where a `&Tensor` is expected, and `&*param`
//!   where an operator needs the tensor (`&*param * 2.0`).
//! - [`pad`](functional::pad) has only the constant mode, and always returns a contiguous
//!   tensor; PyTorch keeps the strides of a non-contiguous input when nothing is padded.
//! - [`conv2d`](functional::conv2d) records the operations it is computed with (padding,
//!   unfolding, a matrix product) instead of PyTorch's `ConvolutionBackward0`, and its `F32`
//!   matrix product accumulates in `F64`. It rejects inputs without channels, tensors on
//!   different devices, and a bias of another dtype also with `Valid` or `Same` padding, which
//!   PyTorch accepts.
//! - [`max_pool2d`](functional::max_pool2d) rejects windows that cover only padding (possible
//!   with a dilation above 1), where PyTorch returns `-inf` with an index outside the input,
//!   and bool inputs also on the meta device. An [`adaptive_avg_pool2d`](functional::adaptive_avg_pool2d)
//!   with an output size of 0 has a gradient of zeros, where PyTorch's backward pass fails, and
//!   integer inputs are rejected also where nothing is computed. `output_size` has no `None`
//!   entries: pass the input's size.
//! - The losses have no warnings ([`mse_loss`](functional::mse_loss) broadcasts silently) and
//!   panic on the first target out of range. [`nll_loss`](functional::nll_loss) differentiates a
//!   single sample with a `[1]` target, which PyTorch's backward rejects, and computes `U8`
//!   targets for inputs of more than two dimensions, which PyTorch's CPU kernel rejects.

#[cfg(doctest)]
mod derive_errors;
pub mod functional;
pub mod init;
mod layers;
mod module;
#[cfg(test)]
mod module_tests;
mod parameter;
mod state_dict;
mod training;

/// Derives [`Module`](trait@Module) and [`Member`]: see [Deriving `Module`](self#deriving-module).
pub use homura_derive::Module;
pub use layers::{
    Dropout, Embedding, Forward, Gelu, LayerNorm, Linear, Relu, RmsNorm, Sequential, Sigmoid, Silu,
    Softmax, Tanh,
};
pub use module::{Member, Module, SharedId, Visitor};
pub use parameter::Parameter;
pub use state_dict::{LoadError, LoadReport, StateDict, StateDictIter};
pub use training::Training;

/// What `#[derive(Module)]` expands to uses; not part of the API.
#[doc(hidden)]
pub mod __private {
    pub use super::module::NonPersistent;
}
