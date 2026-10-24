//! The expansion of `#[derive(Module)]`.

use std::collections::{HashMap, HashSet};

use proc_macro2::{Ident, Span, TokenStream, TokenTree};
use quote::{ToTokens, quote, quote_spanned};
use syn::ext::IdentExt;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Data, DeriveInput, Error, Expr, ExprPath, Fields, GenericParam, Lifetime, LitStr,
    Path, Result, Type, TypeMacro, TypePath, parse_quote,
};

/// How a field reports itself.
enum Kind {
    /// Through its `Member` implementation.
    Member,
    /// As a buffer that is not part of the state dict.
    NonPersistent,
}

/// A field that is a member of the module.
struct Field<'a> {
    ident: &'a Ident,
    ty: &'a Type,
    key: LitStr,
    kind: Kind,
}

/// The implementations of `Module` and `Member` for `input`, or the errors in it: all of the
/// errors in fields and their attributes, or the first error in the struct's own attributes or
/// shape.
pub fn expand(input: &DeriveInput) -> Result<TokenStream> {
    let Container { krate, extra_repr } = container(&input.attrs)?;
    let fields: Vec<&syn::Field> = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => fields.named.iter().collect(),
            Fields::Unit => vec![],
            Fields::Unnamed(fields) => {
                return Err(Error::new(
                    fields.paren_token.span.join(),
                    "`#[derive(Module)]` needs named fields, whose names are the keys of the \
                     members; a tuple struct has none",
                ));
            }
        },
        Data::Enum(data) => {
            return Err(Error::new(
                data.enum_token.span,
                "`#[derive(Module)]` supports structs, not enums: the members of a module are the \
                 fields of a struct",
            ));
        }
        Data::Union(data) => {
            return Err(Error::new(
                data.union_token.span,
                "`#[derive(Module)]` supports structs, not unions: the members of a module are the \
                 fields of a struct",
            ));
        }
    };

    let mut errors: Option<Error> = None;
    let mut add_error = |e: Error| match &mut errors {
        Some(errors) => errors.combine(e),
        None => errors = Some(e),
    };
    let mut members = vec![];
    let mut keys: HashMap<String, &Ident> = HashMap::new();
    for field in fields {
        match member(field) {
            Ok(Some(member)) => {
                let key = member.key.value();
                if let Some(other) = keys.get(&key) {
                    add_error(Error::new(
                        member.key.span(),
                        format!("the key `{key}` is already the key of field `{other}`"),
                    ));
                } else {
                    keys.insert(key, member.ident);
                }
                members.push(member);
            }
            Ok(None) => {}
            Err(e) => add_error(e),
        }
    }
    if let Some(errors) = errors {
        return Err(errors);
    }

    // Fields whose type depends on a type parameter need their trait as a bound; the compiler
    // checks the others where they are visited. A type that contains the struct itself
    // (`Vec<Tree<T>>`) would make the bound need itself: checked where it is visited, it needs
    // the struct's implementation, whose bounds are those of the other fields.
    let parameters: HashSet<String> = input
        .generics
        .type_params()
        .map(|p| p.ident.unraw().to_string())
        .collect();
    let name = &input.ident;
    let mut generics = input.generics.clone();
    // The implementations declare the struct's parameters again. Their names have been linted
    // at the struct; declared at the derive, they aren't linted again, without overriding the
    // caller's lint levels.
    for parameter in &mut generics.params {
        let ident = match parameter {
            GenericParam::Type(p) => &mut p.ident,
            GenericParam::Const(p) => &mut p.ident,
            GenericParam::Lifetime(_) => continue,
        };
        ident.set_span(Span::call_site().located_at(ident.span()));
    }
    let predicates = &mut generics.make_where_clause().predicates;
    for member in &members {
        let uses = uses(member.ty, &parameters, &name.unraw().to_string());
        if uses.parameters && !uses.recursive {
            let ty = member.ty;
            predicates.push(match member.kind {
                Kind::Member => parse_quote!(#ty: #krate::nn::Member),
                Kind::NonPersistent => parse_quote!(#ty: #krate::nn::__private::NonPersistent),
            });
        }
    }
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    // Names of the methods' lifetime and parameters that the struct's generics don't use: a
    // const parameter `__visitor` can't be shadowed. (Constants of that name in scope can't
    // either; nothing here can see them.)
    let taken: HashSet<String> = input
        .generics
        .params
        .iter()
        .map(|p| match p {
            GenericParam::Lifetime(p) => p.lifetime.ident.unraw().to_string(),
            GenericParam::Type(p) => p.ident.unraw().to_string(),
            GenericParam::Const(p) => p.ident.unraw().to_string(),
        })
        .collect();
    let fresh = |base: &str| {
        (0..)
            .map(|i| match i {
                0 => base.to_string(),
                i => format!("{base}{i}"),
            })
            .find(|name| !taken.contains(name))
            .expect("an unused name")
    };
    let lifetime = Lifetime::new(&format!("'{}", fresh("__a")), Span::call_site());
    let visitor = Ident::new(&fresh("__visitor"), Span::call_site());
    let name_parameter = Ident::new(&fresh("__name"), Span::call_site());

    let calls = members.iter().map(|member| {
        let Field { ident, key, .. } = member;
        // Resolved at the derive, but reported at the field's type when it doesn't implement
        // the trait.
        let span = Span::call_site().located_at(member.ty.span());
        match member.kind {
            Kind::Member => quote_spanned! {span=>
                #krate::nn::Member::visit_as(&self.#ident, #key, #visitor);
            },
            Kind::NonPersistent => quote_spanned! {span=>
                #krate::nn::__private::NonPersistent::visit_non_persistent(
                    &self.#ident, #key, #visitor,
                );
            },
        }
    });
    // Resolved at the derive, but a function of another signature is reported at the path.
    let extra_repr = extra_repr.map(|path| {
        let span = Span::call_site().located_at(path.span());
        quote_spanned! {span=>
            fn extra_repr(&self) -> #krate::nn::__private::String {
                #path(self)
            }
        }
    });
    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics #krate::nn::Module for #name #type_generics #where_clause {
            fn visit<#lifetime>(
                &#lifetime self,
                #visitor: &mut dyn #krate::nn::Visitor<#lifetime>,
            ) {
                #(#calls)*
            }

            #extra_repr
        }

        #[automatically_derived]
        impl #impl_generics #krate::nn::Member for #name #type_generics #where_clause {
            fn visit_as<#lifetime>(
                &#lifetime self,
                #name_parameter: &str,
                #visitor: &mut dyn #krate::nn::Visitor<#lifetime>,
            ) {
                #krate::nn::Visitor::module(
                    #visitor,
                    #name_parameter,
                    self,
                    ::core::option::Option::None,
                );
            }
        }
    })
}

/// What the struct's attributes say.
struct Container {
    /// The path of the homura crate: `#[module(crate = "path")]`, or `::homura`.
    krate: Path,
    /// The function that implements `Module::extra_repr`: `#[module(extra_repr = path)]`.
    extra_repr: Option<ExprPath>,
}

fn container(attrs: &[Attribute]) -> Result<Container> {
    let mut krate: Option<Path> = None;
    let mut extra_repr: Option<ExprPath> = None;
    for attr in module_attrs(attrs) {
        attr.parse_nested_meta(|meta| {
            let duplicate = || {
                meta.error(format!(
                    "duplicate `module` attribute `{}`",
                    name(&meta.path)
                ))
            };
            if meta.path.is_ident("crate") {
                let value: LitStr = meta.value()?.parse()?;
                if krate.is_some() {
                    return Err(duplicate());
                }
                krate = Some(value.parse()?);
            } else if meta.path.is_ident("extra_repr") {
                let value: Expr = meta.value()?.parse()?;
                if extra_repr.is_some() {
                    return Err(duplicate());
                }
                match value {
                    Expr::Path(path) if path.attrs.is_empty() => extra_repr = Some(path),
                    other => {
                        return Err(Error::new(
                            other.span(),
                            "`extra_repr` takes the path of a function `fn(&Self) -> String`, \
                             such as `Self::describe`",
                        ));
                    }
                }
            } else {
                return Err(meta.error(
                    "unknown `module` attribute for a struct; expected `crate` or `extra_repr`",
                ));
            }
            Ok(())
        })?;
    }
    Ok(Container {
        krate: krate.unwrap_or_else(|| parse_quote!(::homura)),
        extra_repr,
    })
}

/// The member that `field` is, or `None` if it is skipped.
fn member(field: &syn::Field) -> Result<Option<Field<'_>>> {
    let ident = field.ident.as_ref().expect("a named field");
    let mut rename: Option<LitStr> = None;
    let mut skip = false;
    // Where the attributes other than `skip` are, for errors.
    let mut others: Vec<(Span, &str)> = vec![];
    let mut non_persistent = false;
    for attr in module_attrs(&field.attrs) {
        attr.parse_nested_meta(|meta| {
            let duplicate = || {
                meta.error(format!(
                    "duplicate `module` attribute `{}`",
                    name(&meta.path)
                ))
            };
            if meta.path.is_ident("rename") {
                let key: LitStr = meta.value()?.parse()?;
                if rename.is_some() {
                    return Err(duplicate());
                }
                let value = key.value();
                if value.is_empty() {
                    return Err(Error::new(key.span(), "a key can't be empty"));
                }
                if value.contains('.') {
                    return Err(Error::new(
                        key.span(),
                        format!(
                            "a key can't contain \".\", which separates the names in a path: \
                             `{value}`"
                        ),
                    ));
                }
                others.push((meta.path.span(), "rename"));
                rename = Some(key);
            } else if meta.path.is_ident("skip") {
                if skip {
                    return Err(duplicate());
                }
                skip = true;
            } else if meta.path.is_ident("non_persistent") {
                if non_persistent {
                    return Err(duplicate());
                }
                others.push((meta.path.span(), "non_persistent"));
                non_persistent = true;
            } else {
                return Err(meta.error(
                    "unknown `module` attribute for a field; expected `rename`, `skip` or \
                     `non_persistent`",
                ));
            }
            Ok(())
        })?;
    }
    if skip {
        return match others.first() {
            Some(&(span, other)) => Err(Error::new(
                span,
                format!("a field with `skip` is not a member, so it can't have `{other}`"),
            )),
            None => Ok(None),
        };
    }
    let key = rename.unwrap_or_else(|| LitStr::new(&ident.unraw().to_string(), ident.span()));
    Ok(Some(Field {
        ident,
        ty: &field.ty,
        key,
        kind: if non_persistent {
            Kind::NonPersistent
        } else {
            Kind::Member
        },
    }))
}

fn module_attrs(attrs: &[Attribute]) -> impl Iterator<Item = &Attribute> {
    attrs.iter().filter(|attr| attr.path().is_ident("module"))
}

fn name(path: &Path) -> String {
    path.to_token_stream().to_string()
}

/// What the type of a field refers to.
#[derive(Default)]
struct Uses {
    /// A type parameter of the struct, or `Self`: its trait is a bound.
    parameters: bool,
    /// The struct itself, as `Self` or by its name without a path (`Vec<Tree<T>>`).
    recursive: bool,
}

fn uses(ty: &Type, parameters: &HashSet<String>, name: &str) -> Uses {
    let mut analysis = Analysis {
        parameters,
        name,
        uses: Uses::default(),
    };
    analysis.visit_type(ty);
    analysis.uses
}

/// Finds the type paths in a type that refer to a type parameter or to the struct. Names are
/// compared without raw prefixes; a path with more than one segment (`child::Tree<T>`) is
/// another type, unless it starts with a type parameter (`T::Output`).
struct Analysis<'a> {
    parameters: &'a HashSet<String>,
    name: &'a str,
    uses: Uses,
}

impl<'ast> Visit<'ast> for Analysis<'_> {
    fn visit_type_path(&mut self, ty: &'ast TypePath) {
        if let Some(qself) = &ty.qself {
            // A projection out of a type (`<Self as Trait>::Item`) depends on the parameters of
            // that type, but is not that type.
            let inner = uses(&qself.ty, self.parameters, self.name);
            self.uses.parameters |= inner.parameters || inner.recursive;
            for segment in &ty.path.segments {
                self.visit_path_arguments(&segment.arguments);
            }
            return;
        }
        let path = &ty.path;
        if path.leading_colon.is_none()
            && let Some(first) = path.segments.first()
        {
            let first = first.ident.unraw().to_string();
            let single = path.segments.len() == 1;
            if first == "Self" {
                if single {
                    self.uses.recursive = true;
                } else {
                    self.uses.parameters = true;
                }
            } else if self.parameters.contains(&first) {
                self.uses.parameters = true;
            } else if single && first == self.name {
                self.uses.recursive = true;
            }
        }
        visit::visit_type_path(self, ty);
    }

    fn visit_type_macro(&mut self, ty: &'ast TypeMacro) {
        // What the macro expands to is unknown: a bound if it names a parameter or `Self`.
        if mentions(ty.mac.tokens.clone(), self.parameters) {
            self.uses.parameters = true;
        }
    }
}

/// Whether `tokens` contain one of the type parameters `names` or `Self`.
fn mentions(tokens: TokenStream, names: &HashSet<String>) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(ident) => {
            let ident = ident.unraw().to_string();
            ident == "Self" || names.contains(&ident)
        }
        TokenTree::Group(group) => mentions(group.stream(), names),
        _ => false,
    })
}

#[cfg(test)]
#[path = "module_tests.rs"]
mod tests;
