//! Procedural macros for [homura](https://docs.rs/homura): `#[derive(Module)]`, which homura
//! re-exports as `homura::nn::Module`. Depend on `homura` and use it from there; this crate is
//! an implementation detail, and its version always matches homura's.

mod module;

use proc_macro::TokenStream;

/// Derives `homura::nn::Module` and `homura::nn::Member` for a struct: its named fields are the
/// members of the module, and their names are the keys in the state dict. The type of each field
/// decides what it is: a `Parameter`, a buffer (`Tensor`), a submodule, the `Training` flag, or a
/// hyperparameter that is not listed. A unit struct is a module without members.
///
/// The fields are reported in declaration order. Each field's type must implement `Member`;
/// for a field whose type refers to a type parameter of the struct or to `Self`, the
/// implementations require it (`where Vec<M>: Member`). A field whose type contains the struct
/// itself, as `Self` or by its name without a path (the `Vec<Tree<T>>` of a `Tree<T>`), is not a
/// bound, which would need itself; its type must be a member whenever the other fields are.
/// Recursion through another generic struct or a type alias isn't recognized: implement one of
/// the structs by hand, with bounds on its type parameters.
///
/// The generated methods name their parameters `__visitor` and `__name` (with a number added if
/// the struct has generic parameters of those names), so constants of those names in scope
/// break the expansion.
///
/// Field attributes:
///
/// - `#[module(rename = "key")]`: the key of the field, instead of its name.
/// - `#[module(non_persistent)]`: the field, a `Tensor` or `Option<Tensor>`, is a buffer that is
///   not part of the state dict (PyTorch's `persistent=False`); `to` still converts it.
/// - `#[module(skip)]`: the field is not a member at all, and its type needs no `Member`
///   implementation.
/// - `#[module(added_in = N)]`: the field, a zero-dimensional integer buffer (`Tensor` or
///   `Option<Tensor>`, not `non_persistent`), was added in version `N` of the module, which must
///   not be later than the struct's `version`; implements `Module::added_in`, so that loading an
///   older state dict without it keeps its value.
///
/// Struct attributes:
///
/// - `#[module(crate = "path")]`: the path of the homura crate for a dependency renamed in
///   `Cargo.toml`; it is `::homura` by default.
/// - `#[module(extra_repr = path)]`: the function, of the signature `fn(&Self) -> String`
///   (usually an inherent method, `Self::describe`), that implements `Module::extra_repr`, the
///   settings that `Module::display` prints.
/// - `#[module(version = N)]`: the version of the module's state dict, an integer from 1
///   (PyTorch's `_version`, 1 by default), which implements `Module::version`.
///
/// See the documentation of `homura::nn` for examples.
#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    module::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
