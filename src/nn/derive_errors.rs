//! What `#[derive(Module)]` rejects, as `compile_fail` documentation tests from a caller's crate.
//! homura-derive's tests check the messages of the macro's own errors and where they point.

/// The struct the examples below change, which compiles:
///
/// ```
/// use homura::Tensor;
/// use homura::nn::{Module, Parameter};
///
/// #[derive(Module)]
/// struct Norm {
///     #[module(rename = "weight")]
///     scale: Parameter,
///     #[module(non_persistent)]
///     cache: Tensor,
///     #[module(skip)]
///     log: std::rc::Rc<u8>,
/// }
/// ```
///
/// A field that isn't a member, without `skip`:
///
/// ```compile_fail,E0277
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// struct Norm {
///     scale: Parameter,
///     log: std::rc::Rc<u8>,
/// }
/// ```
///
/// `non_persistent` on a field that isn't a `Tensor` or `Option<Tensor>`:
///
/// ```compile_fail,E0277
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// struct Norm {
///     #[module(non_persistent)]
///     scale: Parameter,
/// }
/// ```
///
/// A generic module whose field isn't a member for the type argument:
///
/// ```compile_fail,E0277
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// struct Wrap<M> {
///     inner: M,
/// }
///
/// let wrap = Wrap { inner: std::rc::Rc::new(1u8) };
/// wrap.state_dict();
/// ```
///
/// Errors of the macro: an invalid key, the same key twice, an unknown attribute, `skip` with
/// another attribute, a tuple struct and an enum.
///
/// ```compile_fail
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// struct Norm {
///     #[module(rename = "w.0")]
///     scale: Parameter,
/// }
/// ```
///
/// ```compile_fail
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// struct Norm {
///     #[module(rename = "bias")]
///     scale: Parameter,
///     bias: Parameter,
/// }
/// ```
///
/// ```compile_fail
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// struct Norm {
///     #[module(persistent)]
///     scale: Parameter,
/// }
/// ```
///
/// ```compile_fail
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// struct Norm {
///     #[module(skip, rename = "weight")]
///     scale: Parameter,
/// }
/// ```
///
/// ```compile_fail
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// struct Norm(Parameter);
/// ```
///
/// ```compile_fail
/// # use homura::nn::Module;
/// #[derive(Module)]
/// enum Activation {
///     Relu,
///     Tanh,
/// }
/// ```
///
/// `extra_repr` with a value that isn't a path, twice, or with a function of another signature:
///
/// ```compile_fail
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(extra_repr = "describe")]
/// struct Norm;
/// ```
///
/// ```compile_fail
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(extra_repr = Self::describe, extra_repr = Self::describe)]
/// struct Norm;
///
/// impl Norm {
///     fn describe(&self) -> String {
///         String::new()
///     }
/// }
/// ```
///
/// ```compile_fail,E0308
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(extra_repr = Self::describe)]
/// struct Norm;
///
/// impl Norm {
///     fn describe(&self) -> &'static str {
///         "norm"
///     }
/// }
/// ```
///
/// ```compile_fail,E0425
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(extra_repr = describe)]
/// struct Norm;
/// ```
///
/// Versions and `added_in`, which compile here:
///
/// ```
/// use homura::Tensor;
/// use homura::nn::{Module, Parameter};
///
/// #[derive(Module)]
/// #[module(version = 2)]
/// struct Counter {
///     scale: Parameter,
///     #[module(added_in = 2, rename = "count")]
///     steps: Option<Tensor>,
/// }
/// ```
///
/// `added_in` on a field that isn't a buffer in the state dict: a parameter, a submodule, a
/// container, a type parameter, a non-persistent buffer or a skipped field.
///
/// ```compile_fail,E0277
/// # use homura::nn::{Module, Parameter};
/// #[derive(Module)]
/// #[module(version = 2)]
/// struct Counter {
///     #[module(added_in = 2)]
///     scale: Parameter,
/// }
/// ```
///
/// ```compile_fail,E0277
/// # use homura::nn::{Linear, Module};
/// #[derive(Module)]
/// #[module(version = 2)]
/// struct Counter {
///     #[module(added_in = 2)]
///     head: Linear,
/// }
/// ```
///
/// ```compile_fail,E0277
/// # use homura::Tensor;
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(version = 2)]
/// struct Counter {
///     #[module(added_in = 2)]
///     counts: Vec<Tensor>,
/// }
/// ```
///
/// ```compile_fail,E0277
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(version = 2)]
/// struct Counter<T> {
///     #[module(added_in = 2)]
///     count: T,
/// }
/// ```
///
/// ```compile_fail
/// # use homura::Tensor;
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(version = 2)]
/// struct Counter {
///     #[module(non_persistent, added_in = 2)]
///     count: Tensor,
/// }
/// ```
///
/// ```compile_fail
/// # use homura::Tensor;
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(version = 2)]
/// struct Counter {
///     #[module(skip, added_in = 2)]
///     count: Tensor,
/// }
/// ```
///
/// `added_in` later than the module's version (1 by default), and versions that aren't positive
/// integers:
///
/// ```compile_fail
/// # use homura::Tensor;
/// # use homura::nn::Module;
/// #[derive(Module)]
/// struct Counter {
///     #[module(added_in = 2)]
///     count: Tensor,
/// }
/// ```
///
/// ```compile_fail
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(version = 0)]
/// struct Counter;
/// ```
///
/// ```compile_fail
/// # use homura::nn::Module;
/// #[derive(Module)]
/// #[module(version = "2")]
/// struct Counter;
/// ```
struct DeriveErrors;
