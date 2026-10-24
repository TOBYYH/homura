//! The [`Module`], [`Member`] and [`Visitor`] traits and the traversal of module trees.

use std::collections::{BTreeMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;

use super::display::short_type_name;
use super::{LoadError, LoadOptions, LoadReport, ModuleDisplay, Parameter, StateDict, Training};
use crate::nn::functional::Reduction;
use crate::tensor::{TensorData, checked_nbytes, conversion_error, convert_data};
use crate::{DType, Device, GeluApproximate, Tensor, TensorOptions, bf16, f16};

/// The identity of a submodule shared through an [`Arc`]: the address of the `Arc`'s value,
/// which stays unique while the traversal holds the module tree. Traversals that list each
/// module once, such as [`Module::named_modules`], skip a submodule whose identity they have
/// seen; plain (not shared) submodules have none and are always visited.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SharedId(usize);

/// Receives the members of a module from [`Module::visit`] and [`Member::visit_as`], each with
/// its name in the module. The references live as long as the module tree, so a visitor can
/// keep them.
///
/// The traversals of [`Module`]'s provided methods are visitors; implement one to walk a module
/// tree in another way.
pub trait Visitor<'a> {
    /// A learnable parameter.
    fn parameter(&mut self, name: &str, parameter: &'a Parameter);
    /// A buffer: a tensor that is part of the module's state without being learned, such as the
    /// running mean of batch normalization. `persistent` buffers are part of the
    /// [`state_dict`](Module::state_dict); others are not, but [`Module::to`] still converts them.
    fn buffer(&mut self, name: &str, buffer: &'a Tensor, persistent: bool);
    /// A submodule, with the identity it is shared under if it lives in an [`Arc`]
    /// (see [`SharedId`]).
    fn module(&mut self, name: &str, module: &'a dyn Module, shared: Option<SharedId>);
    /// The train / eval flag of the module. It has no name.
    fn training(&mut self, training: &'a Training);
}

/// A value that can be a field of a module: it reports what it is to a [`Visitor`] when the
/// module reports its members. The field's type decides:
///
/// | Field type | Member |
/// | --- | --- |
/// | [`Parameter`] | a parameter |
/// | [`Tensor`] | a persistent buffer (report a non-persistent one with [`Visitor::buffer`]) |
/// | [`Training`] | the module's train / eval flag |
/// | `Option<T>`, `Box<T>` | what `T` is, or nothing for `None` |
/// | `Arc<T>` | what `T` is; a submodule is shared under the `Arc`'s identity ([`SharedId`]) |
/// | `Vec<T>`, `[T; N]` | a submodule whose members are named `0`, `1`, ... (`nn.ModuleList`) |
/// | `BTreeMap<String, T>` | a submodule whose members are named by their keys, in key order |
/// | a [`Module`] | a submodule |
/// | numbers, `bool`, `char`, `String`, `&str`, [`DType`], [`Device`], `PhantomData` | nothing (a hyperparameter) |
///
/// A container whose elements report nothing, such as `Vec<usize>` or `Vec<Box<usize>>`,
/// reports nothing either (see [`is_plain_value`](Member::is_plain_value)); an empty container of
/// modules is still a submodule, like an empty `nn.ModuleList`. `Box` and `Arc` hold sized values
/// or a `dyn Module` (also with `Send` and `Sync`), which is never a plain value.
///
/// Tensors and parameters are members but not modules, so the methods of [`Module`] never
/// shadow tensor methods of the same name: `Box<Tensor>::to` is [`Tensor::to`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't be a member of a module",
    label = "`{Self}` doesn't implement `Member`",
    note = "the fields of a module are parameters, buffers, submodules and hyperparameters; mark \
            other fields `#[module(skip)]`"
)]
pub trait Member {
    /// Reports this value to `v` as a member named `name` of a module: a module calls
    /// `v.module(name, self, None)`; the other types report what the table above says.
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>);

    /// Whether this type is a plain value that never reports anything, such as a number: a
    /// container of such values is not reported as a submodule, so that a hyperparameter such
    /// as a `Vec<usize>` doesn't appear in [`named_modules`](Module::named_modules). `false` by
    /// default; `Option`, `Box`, `Arc` and the containers have the value of their element type.
    fn is_plain_value() -> bool
    where
        Self: Sized,
    {
        false
    }
}

/// A neural network module, like PyTorch's `nn.Module`: a value whose parameters, buffers and
/// submodules can be traversed, saved, loaded and converted. Every method takes `&self`.
///
/// A module only describes its members. The forward computation is an ordinary method of the
/// type, with any arguments (`model.forward(&x)`).
///
/// # Implementing `Module`
///
/// `#[derive(Module)]` implements `Module` and [`Member`] for a struct, reporting its fields (see
/// [Deriving `Module`](super#deriving-module)). By hand, [`visit`](Module::visit) reports each
/// member of the module by calling [`visit_as`](Member::visit_as) on it with its name; the
/// member's type decides what it is (see [`Member`]). As a member of another module, a module
/// reports itself as a submodule.
///
/// ```
/// use homura::nn::{Member, Module, Parameter, Visitor};
///
/// struct Linear {
///     weight: Parameter,
///     bias: Option<Parameter>,
/// }
///
/// impl Module for Linear {
///     fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
///         self.weight.visit_as("weight", v);
///         self.bias.visit_as("bias", v);
///     }
/// }
///
/// impl Member for Linear {
///     fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
///         v.module(name, self, None);
///     }
/// }
///
/// let linear = Linear { weight: Parameter::new(homura::zeros([2, 3])), bias: None };
/// assert_eq!(linear.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
/// ```
///
/// Besides such structs, `Box` and `Arc` of a module, `Option` of a module and the containers
/// (`Vec`, arrays, `BTreeMap`) of any members are modules.
///
/// # Traversal
///
/// The keys of members are their names joined by `.` along the path from the module, as in
/// PyTorch: `blocks.0.weight`. Each module lists its own parameters first, then its own buffers,
/// then the members of its submodules in the order `visit` reports them, whatever the order of
/// the fields.
///
/// - A shared parameter or buffer (the same tensor reported twice) and a submodule shared
///   through an [`Arc`] are listed once by `named_parameters`, `named_buffers`, `named_modules`
///   and `named_children`, under the first name found, as PyTorch lists the same object once.
///   [`state_dict`](Module::state_dict) lists every path, so a shared tensor appears under each
///   key, as in PyTorch. Plain submodules are values, never shared, and always listed.
/// - Names must be nonempty and must not contain `.`, and the members of a module must have
///   distinct names. Traversals panic otherwise, with PyTorch's messages.
/// - Module trees must not have cycles: a traversal panics if a shared submodule contains
///   itself. A module whose `visit` reports itself recurses until the stack overflows.
///
/// # Printing
///
/// [`display`](Module::display) prints the tree as PyTorch's `print(model)` does, with each
/// module's [`type_name`](Module::type_name) and [`extra_repr`](Module::extra_repr); derived
/// modules set `extra_repr` with `#[module(extra_repr = path)]`, hand-written ones override the
/// methods.
///
/// ```
/// use homura::nn::{Linear, Module, Relu, Sequential};
///
/// let model = Sequential::new().append(Linear::new(4, 8)).append(Relu);
/// assert_eq!(
///     model.display().to_string(),
///     "Sequential(\n  (0): Linear(in_features=4, out_features=8, bias=True)\n  (1): Relu()\n)"
/// );
/// assert_eq!(model.num_parameters(), 40);
/// ```
pub trait Module: Member {
    /// Reports each member of this module to `v` (by calling `visit_as` on it with its name).
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>);

    /// The settings of this module for its [`display`](Module::display), PyTorch's
    /// `extra_repr()`: `in_features=4, out_features=8, bias=True` for a [`Linear`](super::Linear).
    /// Empty by default; it may have several lines. `Box`, `Arc` and `Option` give the text of
    /// the module they hold.
    fn extra_repr(&self) -> String {
        String::new()
    }

    /// The name of this module for its [`display`](Module::display), PyTorch's class name: the
    /// name of its type without the path and generic arguments, `Block` for a
    /// `my_crate::Block<Linear>`, also through a `dyn Module`. `Box`, `Arc` and `Option` give the
    /// name of the module they hold (`None` for none); `Vec` and arrays are `ModuleList`,
    /// `BTreeMap` is `ModuleDict`.
    fn type_name(&self) -> &'static str {
        short_type_name(std::any::type_name::<Self>())
    }

    /// The number of elements of the parameters of the module tree, each parameter once, like
    /// `sum(p.numel() for p in model.parameters())`: frozen parameters count, buffers don't.
    #[track_caller]
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.num_elements()).sum()
    }

    /// The module tree as PyTorch prints it (`print(model)`), for formatting with `{}`: see
    /// [`ModuleDisplay`]. A submodule shared through an `Arc` appears under each of its names.
    ///
    /// ```
    /// use homura::nn::{Linear, Module};
    ///
    /// #[derive(Module)]
    /// struct Mlp {
    ///     hidden: Linear,
    ///     output: Linear,
    /// }
    ///
    /// let model = Mlp { hidden: Linear::new(4, 8), output: Linear::new(8, 2) };
    /// let text = "\
    /// Mlp(
    ///   (hidden): Linear(in_features=4, out_features=8, bias=True)
    ///   (output): Linear(in_features=8, out_features=2, bias=True)
    /// )";
    /// assert_eq!(model.display().to_string(), text);
    /// ```
    ///
    /// # Panics
    ///
    /// Like the traversals: if names are invalid or the tree has a cycle through shared
    /// submodules.
    #[track_caller]
    fn display(&self) -> ModuleDisplay<'_>
    where
        Self: Sized,
    {
        ModuleDisplay::new(self)
    }

    /// The parameters of the module tree with their keys, each tensor once
    /// (`named_parameters()`).
    #[track_caller]
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = vec![];
        walk(self, true, &mut |item| {
            if let Item::Parameter(name, p) = item {
                out.push((name, p.clone()));
            }
        });
        out
    }

    /// The parameters of the module tree, each tensor once (`parameters()`).
    #[track_caller]
    fn parameters(&self) -> Vec<Parameter> {
        self.named_parameters()
            .into_iter()
            .map(|(_, p)| p)
            .collect()
    }

    /// The buffers of the module tree with their keys, each tensor once, including
    /// non-persistent ones (`named_buffers()`).
    #[track_caller]
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![];
        walk(self, true, &mut |item| {
            if let Item::Buffer(name, t, _) = item {
                out.push((name, t.clone()));
            }
        });
        out
    }

    /// The buffers of the module tree, each tensor once (`buffers()`).
    #[track_caller]
    fn buffers(&self) -> Vec<Tensor> {
        self.named_buffers().into_iter().map(|(_, t)| t).collect()
    }

    /// The modules of the tree with their keys, starting with this one under the key `""`, then
    /// each submodule before its own submodules; a shared submodule once (`named_modules()`).
    #[track_caller]
    fn named_modules(&self) -> Vec<(String, &dyn Module)> {
        let mut out = vec![];
        walk(self, true, &mut |item| {
            if let Item::Module(name, m) = item {
                out.push((name, m));
            }
        });
        out
    }

    /// The direct submodules with their names; a shared submodule once (`named_children()`).
    #[track_caller]
    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        let mut members = Members::default();
        self.visit(&mut members);
        check_names(&members.0);
        let mut seen = HashSet::new();
        members
            .0
            .into_iter()
            .filter_map(|(name, member)| match member {
                Entry::Module(m, shared) if shared.is_none_or(|id| seen.insert(id)) => {
                    Some((name, m))
                }
                _ => None,
            })
            .collect()
    }

    /// The direct submodules; a shared submodule once (`children()`).
    #[track_caller]
    fn children(&self) -> Vec<&dyn Module> {
        self.named_children().into_iter().map(|(_, m)| m).collect()
    }

    /// The parameters and persistent buffers of the module tree by key (`state_dict()`): every
    /// path, so a shared tensor appears under each of its keys.
    ///
    /// The values share the storage of the parameters and buffers, so they show later updates;
    /// they are new tensors without history that don't require grad, like PyTorch's. Use
    /// [`Tensor::deep_clone`] on them for a snapshot.
    #[track_caller]
    fn state_dict(&self) -> StateDict {
        entries(self)
            .into_iter()
            .map(|(key, t)| (key, t.detach()))
            .collect()
    }

    /// Copies the entries of `state_dict` into the parameters and persistent buffers with the
    /// same keys (`load_state_dict(state_dict)`, strict as PyTorch's default), converting them to
    /// the dtype of each target. The parameters and buffers stay the same tensors, so optimizers
    /// holding them stay valid. A tensor with several keys (a shared parameter) gets the entry of
    /// its last key.
    ///
    /// Missing and unexpected keys are errors, so the returned report has empty lists;
    /// [`load_state_dict_with`](Module::load_state_dict_with) takes [`LoadOptions`] to allow
    /// them. Entries whose shape differs from the target's, that come from a device the target
    /// can't copy from (the meta device), or whose target can't be written (its elements
    /// overlap) are errors in any case.
    ///
    /// Every entry is checked before anything is written: when loading fails, the module is
    /// unchanged. (PyTorch writes the entries that match before it reports errors.)
    ///
    /// ```
    /// use homura::nn::{Module, Parameter};
    ///
    /// let layers = vec![Parameter::new(homura::zeros([2]))];
    /// let mut sd = layers.state_dict();
    /// sd.insert("0", homura::tensor([1.0f64, 2.0]));
    /// layers.load_state_dict(&sd).unwrap();
    /// assert_eq!(layers[0].to_vec::<f32>(), [1.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`LoadError`] listing every problem found; nothing has been written then.
    #[track_caller]
    fn load_state_dict(&self, state_dict: &StateDict) -> Result<LoadReport, LoadError> {
        load(self, state_dict, LoadOptions::default())
    }

    /// [`load_state_dict`](Module::load_state_dict) with options
    /// (`load_state_dict(state_dict, strict=False)`): with
    /// [`strict(false)`](LoadOptions::strict), missing and unexpected keys are listed in the
    /// returned report instead of being errors.
    ///
    /// ```
    /// use homura::nn::{LoadOptions, Module, Parameter};
    ///
    /// let layers = vec![Parameter::new(homura::zeros([2]))];
    /// let mut sd = layers.state_dict();
    /// sd.insert("1", homura::tensor([3.0]));
    /// assert!(layers.load_state_dict(&sd).is_err());
    /// let report = layers
    ///     .load_state_dict_with(&sd, LoadOptions::default().strict(false))
    ///     .unwrap();
    /// assert_eq!(report.unexpected_keys, ["1"]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`LoadError`] listing every problem found; nothing has been written then.
    #[track_caller]
    fn load_state_dict_with(
        &self,
        state_dict: &StateDict,
        options: LoadOptions,
    ) -> Result<LoadReport, LoadError> {
        load(self, state_dict, options)
    }

    /// Converts the parameters and buffers of the module tree in place to another dtype and/or
    /// device (`module.to(...)`): `model.to(DType::F64)`, `model.to(Device::Meta)`. The
    /// parameters and buffers stay the same tensors (PyTorch's `param.data = ...`), so optimizers
    /// holding them stay valid. A `.grad` is converted into a new tensor, which replaces it: a
    /// gradient tensor shared with another tensor doesn't change. (PyTorch converts the gradient
    /// tensor itself.)
    ///
    /// A dtype converts only floating point parameters and buffers (an integer buffer such as a
    /// count keeps its dtype); a device moves all of them. Tensors that already have the target
    /// dtype and device are left alone.
    ///
    /// Graphs recorded before the conversion can't be run any more for the converted tensors
    /// (their backward passes panic instead of writing gradients of the old dtype or device), and
    /// views created before it keep the old data and can't be used for gradient computation;
    /// `detach` aliases keep the old data too. Converting back to the original dtype and device
    /// lets an old graph that didn't save the tensor run again. PyTorch lets old graphs run.
    ///
    /// Every tensor is checked before anything is converted: when `to` panics for the reasons
    /// below, the module is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not floating point, if a tensor or its `.grad` can't be moved to
    /// the device (out of the meta device, which has no data) or would be too large, if a tensor
    /// to convert is a view or has a history, or if its `.grad` has a history (clear it with
    /// [`set_grad(None)`](Tensor::set_grad) first; PyTorch converts such gradients).
    #[track_caller]
    fn to(&self, options: impl Into<TensorOptions>)
    where
        Self: Sized,
    {
        convert(self, options.into());
    }

    /// Sets every [`Training`] flag in the module tree to training mode (`train()`).
    #[track_caller]
    fn train(&self) {
        set_training(self, true);
    }

    /// Sets every [`Training`] flag in the module tree to evaluation mode (`eval()`).
    #[track_caller]
    fn eval(&self) {
        set_training(self, false);
    }

    /// Clears the gradient of every parameter (`zero_grad()`, which sets them to `None`).
    #[track_caller]
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.set_grad(None);
        }
    }

    /// Sets whether every parameter requires grad (`requires_grad_(flag)`).
    ///
    /// # Panics
    ///
    /// Like [`Tensor::requires_grad_`], for each parameter in turn.
    #[track_caller]
    fn requires_grad_(&self, requires_grad: bool) {
        for p in self.parameters() {
            p.requires_grad_(requires_grad);
        }
    }
}

/// A member reported to [`Members`].
enum Entry<'a> {
    Parameter(&'a Parameter),
    Buffer(&'a Tensor, bool),
    Module(&'a dyn Module, Option<SharedId>),
    Training(&'a Training),
}

/// Collects the members of one module, in the order they are reported.
#[derive(Default)]
struct Members<'a>(Vec<(String, Entry<'a>)>);

impl<'a> Visitor<'a> for Members<'a> {
    fn parameter(&mut self, name: &str, parameter: &'a Parameter) {
        self.0.push((name.into(), Entry::Parameter(parameter)));
    }

    fn buffer(&mut self, name: &str, buffer: &'a Tensor, persistent: bool) {
        self.0
            .push((name.into(), Entry::Buffer(buffer, persistent)));
    }

    fn module(&mut self, name: &str, module: &'a dyn Module, shared: Option<SharedId>) {
        self.0.push((name.into(), Entry::Module(module, shared)));
    }

    fn training(&mut self, training: &'a Training) {
        self.0.push((String::new(), Entry::Training(training)));
    }
}

/// What a traversal finds, with its key.
pub(super) enum Item<'a> {
    Parameter(String, &'a Parameter),
    Buffer(String, &'a Tensor, bool),
    Module(String, &'a dyn Module),
    Training(&'a Training),
}

/// Calls `emit` with every member of the tree under `root`, in the documented order: for each
/// module, its parameters, its buffers and its flag, then each submodule followed by its own
/// members. `root` is entered as a member named `""`.
///
/// With `dedup`, a parameter, buffer or shared submodule found before is skipped.
///
/// # Panics
///
/// Panics if names are invalid or the tree has a cycle through shared submodules.
#[track_caller]
pub(super) fn walk<'a, M: Member + ?Sized>(
    root: &'a M,
    dedup: bool,
    emit: &mut dyn FnMut(Item<'a>),
) {
    let mut members = Members::default();
    root.visit_as("", &mut members);
    Walk {
        dedup,
        parameters: HashSet::new(),
        buffers: HashSet::new(),
        modules: HashSet::new(),
        path: vec![],
        emit,
    }
    .level("", members.0, true);
}

struct Walk<'a, 'e> {
    dedup: bool,
    parameters: HashSet<usize>,
    buffers: HashSet<usize>,
    modules: HashSet<SharedId>,
    /// The shared modules on the path to the current one, to detect cycles.
    path: Vec<SharedId>,
    emit: &'e mut dyn FnMut(Item<'a>),
}

impl<'a> Walk<'a, '_> {
    /// Visits the members of the module with key `prefix`. The names of the root's members (the
    /// root itself, normally) are not checked: the root is named `""`.
    #[track_caller]
    fn level(&mut self, prefix: &str, members: Vec<(String, Entry<'a>)>, root: bool) {
        if !root {
            check_names(&members);
        }
        let key = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };
        for (name, member) in &members {
            if let Entry::Parameter(p) = *member
                && (!self.dedup || self.parameters.insert(p.id()))
            {
                (self.emit)(Item::Parameter(key(name), p));
            }
        }
        for (name, member) in &members {
            if let Entry::Buffer(t, persistent) = *member
                && (!self.dedup || self.buffers.insert(t.id()))
            {
                (self.emit)(Item::Buffer(key(name), t, persistent));
            }
        }
        for (_, member) in &members {
            if let Entry::Training(t) = *member {
                (self.emit)(Item::Training(t));
            }
        }
        for (name, member) in members {
            let Entry::Module(module, shared) = member else {
                continue;
            };
            let key = key(&name);
            if let Some(id) = shared {
                if self.path.contains(&id) {
                    panic!("the module tree has a cycle: module {key:?} contains itself");
                }
                if self.dedup && !self.modules.insert(id) {
                    continue;
                }
                self.path.push(id);
            }
            (self.emit)(Item::Module(key.clone(), module));
            let mut inner = Members::default();
            module.visit(&mut inner);
            self.level(&key, inner.0, false);
            if shared.is_some() {
                self.path.pop();
            }
        }
    }
}

/// Panics with PyTorch's messages if a name is empty or contains `.`, or if two members share a
/// name.
#[track_caller]
fn check_names(members: &[(String, Entry<'_>)]) {
    let mut seen = HashSet::new();
    for (name, member) in members {
        let kind = match member {
            Entry::Parameter(_) => "parameter",
            Entry::Buffer(..) => "buffer",
            Entry::Module(..) => "module",
            Entry::Training(_) => continue,
        };
        if name.is_empty() {
            panic!("{kind} name can't be empty string \"\"");
        }
        if name.contains('.') {
            match member {
                Entry::Module(..) => panic!("module name can't contain \".\", got: {name}"),
                _ => panic!("{kind} name can't contain \".\""),
            }
        }
        if !seen.insert(name.as_str()) {
            panic!("attribute '{name}' already exists");
        }
    }
}

/// The parameters and persistent buffers of the tree by key, every path.
#[track_caller]
fn entries<M: Member + ?Sized>(module: &M) -> Vec<(String, &Tensor)> {
    let mut out = vec![];
    walk(module, false, &mut |item| match item {
        Item::Parameter(key, p) => out.push((key, p.as_tensor())),
        Item::Buffer(key, t, true) => out.push((key, t)),
        _ => {}
    });
    out
}

#[track_caller]
fn set_training<M: Member + ?Sized>(module: &M, training: bool) {
    walk(module, true, &mut |item| {
        if let Item::Training(t) = item {
            t.set(training);
        }
    });
}

#[track_caller]
fn load<M: Member + ?Sized>(
    module: &M,
    state_dict: &StateDict,
    options: LoadOptions,
) -> Result<LoadReport, LoadError> {
    let strict = options.is_strict();
    let targets = entries(module);
    let keys: HashSet<&str> = targets.iter().map(|(k, _)| k.as_str()).collect();
    let missing_keys: Vec<String> = targets
        .iter()
        .filter(|(k, _)| !state_dict.contains_key(k))
        .map(|(k, _)| k.clone())
        .collect();
    let unexpected_keys: Vec<String> = state_dict
        .keys()
        .filter(|k| !keys.contains(k))
        .map(String::from)
        .collect();
    let mut errors = vec![];
    for (key, target) in &targets {
        let Some(source) = state_dict.get(key) else {
            continue;
        };
        let (dst, src) = (target.data(), source.data());
        let (shape, source_shape) = (&dst.layout.shape, &src.layout.shape);
        if shape != source_shape {
            errors.push(format!(
                "size mismatch for {key}: copying a param with shape {source_shape:?} from \
                 checkpoint, the shape in current model is {shape:?}."
            ));
        } else if let Some(e) = crate::ops::copy_error(&dst, &src) {
            errors.push(format!(
                "While copying the parameter named \"{key}\", whose dimensions in the model are \
                 {shape:?} and whose dimensions in the checkpoint are {source_shape:?}, an \
                 exception occurred: {e}"
            ));
        }
    }
    if !errors.is_empty() || strict && !(missing_keys.is_empty() && unexpected_keys.is_empty()) {
        let (missing_keys, unexpected_keys) = if strict {
            (missing_keys, unexpected_keys)
        } else {
            (vec![], vec![])
        };
        return Err(LoadError {
            missing_keys,
            unexpected_keys,
            errors,
        });
    }
    let _guard = crate::no_grad();
    for (key, target) in &targets {
        if let Some(source) = state_dict.get(key) {
            target.copy_(source);
        }
    }
    Ok(LoadReport {
        missing_keys,
        unexpected_keys,
    })
}

#[track_caller]
fn convert<M: Member + ?Sized>(module: &M, options: TensorOptions) {
    if let Some(dtype) = options.dtype
        && !dtype.is_floating_point()
    {
        panic!("Module::to only accepts floating point dtypes, but got desired dtype={dtype}");
    }
    // Each tensor once, under the first key it is found with.
    let mut seen = HashSet::new();
    let mut tensors = vec![];
    walk(module, true, &mut |item| match item {
        Item::Parameter(key, p) if seen.insert(p.id()) => {
            tensors.push(("parameter", key, p.as_tensor()));
        }
        Item::Buffer(key, t, _) if seen.insert(t.id()) => tensors.push(("buffer", key, t)),
        _ => {}
    });
    // Check every tensor that changes, and its gradient, before converting any: every error
    // the conversion can run into except running out of memory.
    let mut changes = vec![];
    for (kind, key, t) in tensors {
        let data = t.data();
        let dtype = match options.dtype {
            Some(dtype) if data.dtype.is_floating_point() => dtype,
            _ => data.dtype,
        };
        let device = options.device.unwrap_or(data.device());
        if (dtype, device) == (data.dtype, data.device()) {
            continue;
        }
        // The check returns the reason, so that the panic happens here, in a function with
        // `#[track_caller]`.
        let grad = match check_change(t, &data, device, dtype) {
            Ok(grad) => grad,
            Err(why) => panic!("Module::to can't convert {kind} {key:?}: {why}"),
        };
        changes.push((t, data, dtype, device, grad));
    }
    for (t, data, dtype, device, grad) in changes {
        let grad = grad.map(|(grad, requires_grad)| {
            let g = Tensor::from_data(convert_data(&grad, dtype, device));
            g.requires_grad_(requires_grad);
            g
        });
        crate::autograd::replace_leaf_data(t, convert_data(&data, dtype, device), grad);
    }
}

/// Why the tensor `t` with the data `data` can't be converted in place to `dtype` on `device`, or
/// the data and `requires_grad` of its `.grad`, which is converted with it. Takes the `.grad` now:
/// a tensor converted earlier may be this one's `.grad`.
fn check_change(
    t: &Tensor,
    data: &TensorData,
    device: Device,
    dtype: DType,
) -> Result<Option<(TensorData, bool)>, String> {
    check_conversion(data, device, dtype)?;
    if t.view_info().is_some() || !t.is_leaf() {
        return Err(
            "only a tensor that isn't a view and has no history can be converted in place".into(),
        );
    }
    let Some(g) = t.grad() else {
        return Ok(None);
    };
    if !g.is_leaf() {
        return Err(
            "its .grad has a history; clear it with set_grad(None) before converting the module"
                .into(),
        );
    }
    let grad = g.data();
    check_conversion(&grad, device, dtype)?;
    Ok(Some((grad, g.requires_grad())))
}

/// Why `data` can't be converted to `dtype` on `device`, except running out of memory.
fn check_conversion(data: &TensorData, device: Device, dtype: DType) -> Result<(), String> {
    if let Some(e) = conversion_error(data.device(), device) {
        return Err(e);
    }
    checked_nbytes(&data.layout.shape, dtype).map(drop)
}

/// Reports the submodules that `T` reports under the identity of the `Arc` holding it.
struct SharedVisitor<'v, 'a> {
    inner: &'v mut dyn Visitor<'a>,
    id: SharedId,
}

impl<'a> Visitor<'a> for SharedVisitor<'_, 'a> {
    fn parameter(&mut self, name: &str, parameter: &'a Parameter) {
        self.inner.parameter(name, parameter);
    }

    fn buffer(&mut self, name: &str, buffer: &'a Tensor, persistent: bool) {
        self.inner.buffer(name, buffer, persistent);
    }

    fn module(&mut self, name: &str, module: &'a dyn Module, shared: Option<SharedId>) {
        // An inner `Arc` has already given its own identity.
        self.inner.module(name, module, shared.or(Some(self.id)));
    }

    fn training(&mut self, training: &'a Training) {
        self.inner.training(training);
    }
}

/// How `#[module(non_persistent)]` fields report themselves: as buffers that are not part of
/// the state dict.
#[diagnostic::on_unimplemented(
    message = "`#[module(non_persistent)]` needs a `Tensor` or `Option<Tensor>` field, not \
               `{Self}`",
    label = "not a buffer"
)]
pub trait NonPersistent {
    fn visit_non_persistent<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>);
}

impl NonPersistent for Tensor {
    fn visit_non_persistent<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.buffer(name, self, false);
    }
}

impl NonPersistent for Option<Tensor> {
    fn visit_non_persistent<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        if let Some(buffer) = self {
            v.buffer(name, buffer, false);
        }
    }
}

impl Member for Parameter {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.parameter(name, self);
    }
}

impl Member for Tensor {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.buffer(name, self, true);
    }
}

impl Member for Training {
    fn visit_as<'a>(&'a self, _: &str, v: &mut dyn Visitor<'a>) {
        v.training(self);
    }
}

impl<T: Member> Member for Option<T> {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        if let Some(inner) = self {
            inner.visit_as(name, v);
        }
    }

    fn is_plain_value() -> bool {
        T::is_plain_value()
    }
}

impl<T: Module> Module for Option<T> {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        if let Some(inner) = self {
            inner.visit(v);
        }
    }

    fn extra_repr(&self) -> String {
        self.as_ref().map(T::extra_repr).unwrap_or_default()
    }

    fn type_name(&self) -> &'static str {
        self.as_ref().map_or("None", T::type_name)
    }
}

impl<T: Member> Member for Box<T> {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        (**self).visit_as(name, v);
    }

    fn is_plain_value() -> bool {
        T::is_plain_value()
    }
}

impl<T: Module> Module for Box<T> {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        (**self).visit(v);
    }

    fn extra_repr(&self) -> String {
        (**self).extra_repr()
    }

    fn type_name(&self) -> &'static str {
        (**self).type_name()
    }
}

impl<T: Member> Member for Arc<T> {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        visit_shared(self, name, v);
    }

    fn is_plain_value() -> bool {
        T::is_plain_value()
    }
}

impl<T: Module> Module for Arc<T> {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        (**self).visit(v);
    }

    fn extra_repr(&self) -> String {
        (**self).extra_repr()
    }

    fn type_name(&self) -> &'static str {
        (**self).type_name()
    }
}

/// Reports the value of `shared` as a member named `name`, a submodule under the identity of the
/// `Arc`: the address of its value, which can't be reused while the tree, which holds the `Arc`,
/// is borrowed.
fn visit_shared<'a, T: Member + ?Sized>(shared: &'a Arc<T>, name: &str, v: &mut dyn Visitor<'a>) {
    let id = SharedId(Arc::as_ptr(shared) as *const () as usize);
    (**shared).visit_as(name, &mut SharedVisitor { inner: v, id });
}

/// `Box` and `Arc` of trait objects, which the sized implementations don't cover. A module is
/// never a plain value.
macro_rules! dyn_modules {
    ($($t:ty),* $(,)?) => {
        $(
            impl<'m> Member for Box<$t> {
                fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
                    (**self).visit_as(name, v);
                }
            }

            impl<'m> Module for Box<$t> {
                fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
                    (**self).visit(v);
                }

                fn extra_repr(&self) -> String {
                    (**self).extra_repr()
                }

                fn type_name(&self) -> &'static str {
                    (**self).type_name()
                }
            }

            impl<'m> Member for Arc<$t> {
                fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
                    visit_shared(self, name, v);
                }
            }

            impl<'m> Module for Arc<$t> {
                fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
                    (**self).visit(v);
                }

                fn extra_repr(&self) -> String {
                    (**self).extra_repr()
                }

                fn type_name(&self) -> &'static str {
                    (**self).type_name()
                }
            }
        )*
    };
}

dyn_modules!(
    dyn Module + 'm,
    dyn Module + Send + 'm,
    dyn Module + Sync + 'm,
    dyn Module + Send + Sync + 'm,
);

/// `Vec`, arrays and `BTreeMap`: submodules unless their elements are plain values.
macro_rules! containers {
    ($([$($generics:tt)*] $t:ty),* $(,)?) => {
        $(
            impl<$($generics)*> Member for $t {
                fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
                    if !T::is_plain_value() {
                        v.module(name, self, None);
                    }
                }

                fn is_plain_value() -> bool {
                    T::is_plain_value()
                }
            }
        )*
    };
}

containers!(
    [T: Member] Vec<T>,
    [T: Member, const N: usize] [T; N],
    [T: Member] BTreeMap<String, T>,
);

impl<T: Member> Module for Vec<T> {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        visit_elements(self, v);
    }

    fn type_name(&self) -> &'static str {
        "ModuleList"
    }
}

impl<T: Member, const N: usize> Module for [T; N] {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        visit_elements(self, v);
    }

    fn type_name(&self) -> &'static str {
        "ModuleList"
    }
}

fn visit_elements<'a, T: Member>(elements: &'a [T], v: &mut dyn Visitor<'a>) {
    if !T::is_plain_value() {
        for (i, element) in elements.iter().enumerate() {
            element.visit_as(&i.to_string(), v);
        }
    }
}

impl<T: Member> Module for BTreeMap<String, T> {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        if !T::is_plain_value() {
            for (key, value) in self {
                value.visit_as(key, v);
            }
        }
    }

    fn type_name(&self) -> &'static str {
        "ModuleDict"
    }
}

impl<T: ?Sized> Member for PhantomData<T> {
    fn visit_as<'a>(&'a self, _: &str, _: &mut dyn Visitor<'a>) {}

    fn is_plain_value() -> bool {
        true
    }
}

/// Hyperparameters: values that report nothing.
macro_rules! plain_values {
    ($($t:ty),* $(,)?) => {
        $(
            impl Member for $t {
                fn visit_as<'a>(&'a self, _: &str, _: &mut dyn Visitor<'a>) {}

                fn is_plain_value() -> bool {
                    true
                }
            }
        )*
    };
}

plain_values!(
    bool,
    char,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    f32,
    f64,
    f16,
    bf16,
    String,
    &str,
    (),
    DType,
    Device,
    GeluApproximate,
    Reduction,
);
