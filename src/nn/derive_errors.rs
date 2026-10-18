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
struct DeriveErrors;
