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
//! `#[module(version = N)]` on the struct sets the [`version`](Module::version) of its state dict
//! (PyTorch's `_version`), and `#[module(added_in = N)]` on a zero-dimensional integer buffer
//! says which version added it, so that loading an older state dict without it keeps its value
//! ([`Module::added_in`]), as [`BatchNorm2d`] does for its `num_batches_tracked`.
//!
//! `#[module(extra_repr = path)]` on the struct gives the function, of the signature
//! `fn(&Self) -> String`, that implements [`Module::extra_repr`]: the settings that
//! [`Module::display`] prints, as PyTorch's `extra_repr()` does.
//!
//! ```
//! use homura::nn::{Linear, Module};
//!
//! #[derive(Module)]
//! #[module(extra_repr = Self::describe)]
//! struct Block {
//!     proj: Linear,
//!     scale: f64,
//! }
//!
//! impl Block {
//!     fn describe(&self) -> String {
//!         format!("scale={}", self.scale)
//!     }
//! }
//!
//! let block = Block { proj: Linear::new(2, 2), scale: 0.5 };
//! let text = "\
//! Block(
//!   scale=0.5
//!   (proj): Linear(in_features=2, out_features=2, bias=True)
//! )";
//! assert_eq!(block.display().to_string(), text);
//! ```
//!
//! # Layers
//!
//! [`Linear`], [`Embedding`], [`LayerNorm`], [`RmsNorm`], [`Dropout`], the activations
//! [`Relu`], [`Gelu`], [`Silu`], [`Tanh`], [`Sigmoid`] and [`Softmax`], and the CNN layers
//! [`Conv2d`], [`BatchNorm2d`], [`MaxPool2d`], [`AdaptiveAvgPool2d`] and [`Flatten`] are modules
//! with a `forward` method and implement [`Forward`]; [`Sequential`] applies such layers one
//! after another. [`functional`] has their computations as functions of tensors, and [`init`] the
//! initializations that layers use, with PyTorch's formulas. [`Dropout`] and [`BatchNorm2d`]
//! depend on the mode: they have a [`Training`] flag, which [`Module::eval`] clears. Options are
//! set one after another, starting from PyTorch's defaults: `Conv2d::new(3, 8, 3).padding(1)`;
//! sizes of 2-D layers are a number or a pair ([`IntoPair`]).
//!
//! ```
//! use homura::nn::{
//!     AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, Module, Relu, Sequential,
//! };
//!
//! let model = Sequential::new()
//!     .append(Conv2d::new(3, 8, 3).padding(1))
//!     .append(BatchNorm2d::new(8))
//!     .append(Relu)
//!     .append(MaxPool2d::new(2))
//!     .append(Conv2d::new(8, 16, 3).padding(1))
//!     .append(AdaptiveAvgPool2d::new(1))
//!     .append(Flatten::default())
//!     .append(Linear::new(16, 4));
//! assert_eq!(model.forward(&homura::randn([4, 3, 8, 8])).shape(), [4, 4]);
//! assert_eq!(model.state_dict().len(), 11);
//! model.eval();
//! ```
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
//! | `nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False)` | [`Conv2d::new(3, 8, 3).stride(2).padding(1).bias(false)`](Conv2d) |
//! | `nn.BatchNorm2d(8, momentum=None, affine=False)` | [`BatchNorm2d::new(8).momentum(None).affine(false)`](BatchNorm2d) |
//! | `nn.MaxPool2d(3, 2, 1)`, `nn.MaxPool2d(2, return_indices=True)` | [`MaxPool2d::new(3).stride(2).padding(1)`](MaxPool2d), [`MaxPool2d::new(2).forward_with_indices(&x)`](MaxPool2d::forward_with_indices) |
//! | `nn.AdaptiveAvgPool2d(1)`, `nn.Flatten()` | [`AdaptiveAvgPool2d::new(1)`](AdaptiveAvgPool2d::new), [`Flatten::default()`](Flatten) |
//! | `F.batch_norm(x, rm, rv, w, b, training=True)` | [`functional::batch_norm(&x, Some(&rm), Some(&rv), Some(&w), Some(&b), BatchNormOptions { training: true, ..Default::default() })`](functional::batch_norm) |
//! | `nn.Dropout(0.1)`, `F.dropout(x, 0.1, training)` | [`Dropout::new(0.1)`](Dropout::new), [`functional::dropout(&x, 0.1, training)`](functional::dropout) |
//! | `nn.init.kaiming_uniform_(w, a=math.sqrt(5))` | [`init::kaiming_uniform_(&w, 5f64.sqrt(), FanMode::FanIn, Nonlinearity::LeakyRelu)`](init::kaiming_uniform_) |
//! | `nn.ModuleList`, `nn.ModuleDict` | `Vec<T>`, `BTreeMap<String, T>` fields |
//! | `self.training` | a [`Training`] field: `self.training.get()` |
//! | the same submodule assigned twice | an `Arc<T>` in both fields |
//! | `model.state_dict()`, `model.load_state_dict(sd)`, `model.load_state_dict(sd, strict=False)` | [`model.state_dict()`](Module::state_dict), [`model.load_state_dict(&sd)`](Module::load_state_dict), [`model.load_state_dict_with(&sd, LoadOptions::default().strict(false))`](Module::load_state_dict_with), returning a `Result` |
//! | `with torch.device("meta"): model = Net()`, `model.load_state_dict(sd, assign=True)` | [`let model = homura::with_device(Device::Meta, Net::new)`](crate::with_device), [`model.load_state_dict_with(&sd, LoadOptions::default().assign(true))`](LoadOptions::assign) |
//! | `model.to(torch.float64)`, `model.to("meta")` | [`model.to(DType::F64)`](Module::to), `model.to(Device::Meta)` |
//! | `consume_prefix_in_state_dict_if_present(sd, "module.")`, `model.state_dict(prefix="x.")`, `sd._metadata` | [`sd.consume_prefix("module.")`](StateDict::consume_prefix), [`sd.add_prefix("x.")`](StateDict::add_prefix), [`sd.versions()`](StateDict::versions) |
//! | `_version = 2` in the class, a hook for old state dicts in `_load_from_state_dict` | `#[module(version = 2)]`, `#[module(added_in = 2)]` on the buffer ([`Module::added_in`]) |
//! | `print(model)`, `def extra_repr(self)` | [`println!("{}", model.display())`](Module::display), `#[module(extra_repr = Self::describe)]` |
//! | `sum(p.numel() for p in model.parameters())` | [`model.num_parameters()`](Module::num_parameters) |
//!
//! [`Module`] describes how members are named, ordered and deduplicated.
//!
//! # Differences from PyTorch
//!
//! - Submodules are values: a submodule is shared only through an `Arc`. Traversals list a
//!   shared submodule once, like PyTorch lists the same object once; the state dict has its
//!   members under every path, also like PyTorch.
//! - `BTreeMap` containers order their members by key, not by insertion like `nn.ModuleDict`.
//! - [`Module::display`] prints homura's type names: `Relu`, `Gelu`, `Silu` and `RmsNorm` for
//!   PyTorch's `ReLU`, `GELU`, `SiLU` and `RMSNorm`, `ModuleList` and `ModuleDict` for `Vec`,
//!   arrays and `BTreeMap`, `None` for a `None` root (`None()`). The sizes of [`MaxPool2d`] and
//!   [`AdaptiveAvgPool2d`] are printed as one number when the height and the width are equal,
//!   where PyTorch prints what it was given (`(1, 1)` or `1`); [`Dropout`] prints
//!   `inplace=False`, and floats are printed as Python prints them (`eps=1e-05`).
//! - [`Module::load_state_dict`] and [`Module::to`] check everything they can before they
//!   change anything, so that their errors leave the module unchanged; PyTorch's
//!   `load_state_dict` writes the matching entries before it reports errors. This covers errors
//!   in their inputs: a panic for other reasons (out of memory) or another thread changing the
//!   module at the same time can still leave it partly changed.
//! - After [`Module::to`], graphs recorded before can't be run any more for the converted
//!   tensors, also after converting back; PyTorch runs them, and can write gradients of the old
//!   dtype into the converted parameters. The `.grad` of a converted tensor is replaced by a converted copy, where
//!   PyTorch converts the gradient tensor itself; one that has a history must be cleared before
//!   converting.
//! - Loading with [`assign`](LoadOptions::assign) keeps the parameters and buffers and gives them
//!   the data of the entries, where PyTorch replaces them with new ones (a buffer with the entry
//!   itself). Handles taken before, such as an optimizer's, see the new data, a shared parameter
//!   stays shared, and graphs and views of the old data are refused, which in PyTorch keep the
//!   old parameter. A parameter keeps whether it requires grad (as in PyTorch) and loses its
//!   `.grad`.
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
//! - [`batch_norm`](functional::batch_norm) increments the versions of the running statistics
//!   when it updates them, so a graph that saved them in evaluation mode can't run backward
//!   after a training pass; PyTorch's update counts as no modification, and the old graph uses
//!   the new values. It checks every argument before it writes them, where PyTorch updates them
//!   before it checks the dtypes of the parameters, and rejects running statistics to update
//!   that are expanded or share memory with the input, the parameters or each other, which
//!   PyTorch writes and reads in its order. It takes parameters and statistics of any
//!   shape with `C` elements, whose gradients have their shapes (PyTorch takes statistics of
//!   one dimension only, and its backward pass fails for other shapes), and checks inputs
//!   without elements like others (PyTorch skips the checks after `eps` for them). Its sums
//!   accumulate in `F64`, and its gradients of gradients are recorded as operations, which are
//!   also right for a bias without a weight, where PyTorch's `NativeBatchNormBackwardBackward0`
//!   fails.
//! - The options of [`Conv2d`] and [`BatchNorm2d`] are set after construction, so homura checks
//!   each when it is set, with PyTorch's messages; [`Conv2d::groups`] gives the layer new
//!   parameters, which PyTorch sets only when it creates the layer. [`Conv2d`] has no
//!   `padding_mode`, [`MaxPool2d`] returns the indices from
//!   [`forward_with_indices`](MaxPool2d::forward_with_indices) instead of a `return_indices`
//!   option, and [`AdaptiveAvgPool2d`] has no `None` sizes.
//! - A buffer added in a later version ([`Module::added_in`], `num_batches_tracked` of
//!   [`BatchNorm2d`]) that is on the meta device gets a 0 of its own dtype when an old state
//!   dict is assigned, where PyTorch's `BatchNorm2d` uses an `I64` 0 whatever the buffer's dtype;
//!   only zero-dimensional integer buffers can be added this way. [`StateDict::consume_prefix`]
//!   renames the version of the module named by the prefix without its final `.`, where PyTorch
//!   compares with the prefix without any `.` (they differ for a prefix such as `a.b.`).
//! - The losses have no warnings ([`mse_loss`](functional::mse_loss) broadcasts silently) and
//!   panic on the first target out of range. [`nll_loss`](functional::nll_loss) differentiates a
//!   single sample with a `[1]` target, which PyTorch's backward rejects, and computes `U8`
//!   targets for inputs of more than two dimensions, which PyTorch's CPU kernel rejects.

#[cfg(doctest)]
mod derive_errors;
mod display;
pub mod functional;
pub mod init;
mod layers;
mod module;
#[cfg(test)]
mod module_tests;
mod parameter;
mod state_dict;
mod training;

pub use display::ModuleDisplay;
pub(crate) use display::{
    python_bool, python_float, python_option_float, python_size, python_tuple,
};
/// Derives [`Module`](trait@Module) and [`Member`]: see [Deriving `Module`](self#deriving-module).
pub use homura_derive::Module;
pub use layers::{
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout, Embedding, Flatten, Forward, Gelu, IntoPair,
    LayerNorm, Linear, MaxPool2d, Relu, RmsNorm, Sequential, Sigmoid, Silu, Softmax, Tanh,
};
pub use module::{Member, Module, SharedId, Visitor};
pub use parameter::Parameter;
pub use state_dict::{
    LoadError, LoadOptions, LoadReport, ModuleVersions, StateDict, StateDictIter,
};
pub use training::Training;

/// What `#[derive(Module)]` expands to uses; not part of the API.
#[doc(hidden)]
pub mod __private {
    pub use super::module::{AddedIn, NonPersistent, added_in_field};
    pub use std::string::String;
}
