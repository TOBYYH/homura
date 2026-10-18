//! The expansion of `#[derive(Module)]`: its errors, with where they point, and the bounds and
//! paths it generates. Whether the generated code compiles and works is tested by homura's
//! tests, which use the macro.

use super::expand;

/// The errors for the struct `source`, with the line and column (from 1) they start at.
fn errors(source: &str) -> Vec<(String, usize, usize)> {
    let input = syn::parse_str(source).expect("a valid struct");
    let error = expand(&input).expect_err("expected errors");
    error
        .into_iter()
        .map(|e| {
            let start = e.span().start();
            (e.to_string(), start.line, start.column + 1)
        })
        .collect()
}

fn error(source: &str) -> (String, usize, usize) {
    let mut errors = errors(source);
    assert_eq!(errors.len(), 1, "{errors:?}");
    errors.remove(0)
}

/// The expansion, as a string of tokens.
fn expanded(source: &str) -> String {
    let input = syn::parse_str(source).expect("a valid struct");
    expand(&input).expect("an expansion").to_string()
}

#[test]
fn only_structs_with_named_fields_or_none_are_modules() {
    assert_eq!(
        error("struct Pair(u8, u8);"),
        (
            "`#[derive(Module)]` needs named fields, whose names are the keys of the members; a \
             tuple struct has none"
                .into(),
            1,
            12
        )
    );
    assert_eq!(
        error("pub enum Layer { A, B }"),
        (
            "`#[derive(Module)]` supports structs, not enums: the members of a module are the \
             fields of a struct"
                .into(),
            1,
            5
        )
    );
    assert_eq!(
        error("union Bits { a: u32, b: f32 }"),
        (
            "`#[derive(Module)]` supports structs, not unions: the members of a module are the \
             fields of a struct"
                .into(),
            1,
            1
        )
    );
    // Unit structs and empty structs are modules without members.
    for source in ["struct Relu;", "struct Relu {}"] {
        let expansion = expanded(source);
        assert!(expansion.contains("impl :: homura :: nn :: Module for Relu"));
        assert!(expansion.contains("impl :: homura :: nn :: Member for Relu"));
    }
}

#[test]
fn unknown_and_duplicate_attributes_point_at_the_attribute() {
    let field = "unknown `module` attribute for a field; expected `rename`, `skip` or \
                 `non_persistent`";
    assert_eq!(
        error("struct M {\n    #[module(skip, persistent)]\n    a: u8,\n}"),
        (field.into(), 2, 20)
    );
    assert_eq!(
        error("struct M {\n    #[module(crate = \"h\")]\n    a: u8,\n}"),
        (field.into(), 2, 14)
    );
    let container = "unknown `module` attribute for a struct; expected `crate`";
    assert_eq!(
        error("#[module(skip)]\nstruct M {\n    a: u8,\n}"),
        (container.into(), 1, 10)
    );
    assert_eq!(
        error("#[module(crate = \"a\")]\n#[module(crate = \"b\")]\nstruct M;"),
        ("duplicate `module` attribute `crate`".into(), 2, 10)
    );
    assert_eq!(
        error(
            "struct M {\n    #[module(rename = \"x\")]\n    #[module(rename = \"y\")]\n    a: u8,\n}"
        ),
        ("duplicate `module` attribute `rename`".into(), 3, 14)
    );
    assert_eq!(
        error("struct M {\n    #[module(skip, skip)]\n    a: u8,\n}"),
        ("duplicate `module` attribute `skip`".into(), 2, 20)
    );
    assert_eq!(
        error("struct M {\n    #[module(non_persistent, non_persistent)]\n    a: u8,\n}"),
        (
            "duplicate `module` attribute `non_persistent`".into(),
            2,
            30
        )
    );
    // syn's own errors for malformed attributes.
    assert_eq!(error("struct M {\n    #[module]\n    a: u8,\n}").1, 2);
    assert_eq!(
        error("struct M {\n    #[module(rename = x)]\n    a: u8,\n}").1,
        2
    );
    assert_eq!(error("#[module(crate = \"no path\")]\nstruct M;").1, 1);
}

#[test]
fn skipped_fields_take_no_other_attributes() {
    let message =
        |other: &str| format!("a field with `skip` is not a member, so it can't have `{other}`");
    assert_eq!(
        error("struct M {\n    #[module(rename = \"b\", skip)]\n    a: u8,\n}"),
        (message("rename"), 2, 14)
    );
    assert_eq!(
        error("struct M {\n    #[module(skip)]\n    #[module(non_persistent)]\n    a: u8,\n}"),
        (message("non_persistent"), 3, 14)
    );
}

#[test]
fn keys_must_be_valid_and_distinct() {
    assert_eq!(
        error("struct M {\n    #[module(rename = \"\")]\n    a: u8,\n}"),
        ("a key can't be empty".into(), 2, 23)
    );
    assert_eq!(
        error("struct M {\n    #[module(rename = \"a.b\")]\n    a: u8,\n}"),
        (
            "a key can't contain \".\", which separates the names in a path: `a.b`".into(),
            2,
            23
        )
    );
    // The later field is wrong, whether or not it is renamed.
    assert_eq!(
        error("struct M {\n    a: u8,\n    #[module(rename = \"a\")]\n    b: u8,\n}"),
        ("the key `a` is already the key of field `a`".into(), 3, 23)
    );
    assert_eq!(
        error("struct M {\n    #[module(rename = \"b\")]\n    a: u8,\n    b: u8,\n}"),
        ("the key `b` is already the key of field `a`".into(), 4, 5)
    );
    // A skipped field has no key.
    expanded(
        "struct M {\n    #[module(skip)]\n    a: u8,\n    #[module(rename = \"a\")]\n    b: u8,\n}",
    );
}

#[test]
fn every_wrong_field_is_reported() {
    let errors = errors(
        "struct M {\n    #[module(rename = \"\")]\n    a: u8,\n    #[module(skipped)]\n    b: u8,\n    \
         #[module(rename = \"a\")]\n    c: u8,\n    d: u8,\n    #[module(rename = \"d\")]\n    e: u8,\n}",
    );
    let places: Vec<_> = errors
        .iter()
        .map(|&(_, line, column)| (line, column))
        .collect();
    // The empty key leaves `a` without a key, so `c` can have it.
    assert_eq!(places, [(2, 23), (4, 14), (9, 23)]);
}

#[test]
fn keys_are_names_without_raw_prefixes() {
    let expansion = expanded(
        "struct M {\n    r#type: Parameter,\n    #[module(rename = \"r#x\")]\n    y: Parameter,\n}",
    );
    assert!(expansion.contains("visit_as (& self . r#type , \"type\" , __visitor)"));
    // A rename is the key as written.
    assert!(expansion.contains("visit_as (& self . y , \"r#x\" , __visitor)"));
}

#[test]
fn fields_are_visited_in_order_by_their_kind() {
    let expansion = expanded(
        "struct M {\n    w: Parameter,\n    #[module(skip)]\n    cache: Rc<u8>,\n    \
         #[module(non_persistent)]\n    #[module(rename = \"mask\")]\n    m: Option<Tensor>,\n    \
         training: Training,\n}",
    );
    let w = expansion
        .find(":: homura :: nn :: Member :: visit_as (& self . w , \"w\" , __visitor) ;")
        .unwrap();
    let m = expansion
        .find(
            ":: homura :: nn :: __private :: NonPersistent :: visit_non_persistent (& self . m , \
             \"mask\" , __visitor ,) ;",
        )
        .unwrap();
    let training = expansion
        .find(":: homura :: nn :: Member :: visit_as (& self . training , \"training\" , __visitor) ;")
        .unwrap();
    assert!(w < m && m < training);
    assert!(!expansion.contains("cache"));
    // No bounds without type parameters.
    assert!(!expansion.contains("where"));
}

#[test]
fn fields_that_mention_type_parameters_are_bounds() {
    let expansion = expanded(
        "struct M<'a, T: Clone, U, const N: usize> where U: Default {\n    a: Vec<T>,\n    \
         b: [Linear; N],\n    c: &'a str,\n    d: Wrap<Option<(U, T)>>,\n    #[module(skip)]\n    \
         e: T,\n    #[module(non_persistent)]\n    f: Option<U>,\n}",
    );
    let header = "impl < 'a , T : Clone , U , const N : usize > :: homura :: nn :: Module for M < \
                  'a , T , U , N > where U : Default , Vec < T > : :: homura :: nn :: Member , \
                  Wrap < Option < (U , T) > > : :: homura :: nn :: Member , Option < U > : :: \
                  homura :: nn :: __private :: NonPersistent {";
    assert!(expansion.contains(header), "{expansion}");
    assert!(expansion.contains(&header.replace(":: nn :: Module for", ":: nn :: Member for")));
}

#[test]
fn the_crate_path_can_be_given() {
    let expansion =
        expanded("#[module(crate = \"hm\")]\nstruct M<T> {\n    a: T,\n    b: Parameter,\n}");
    assert!(
        expansion
            .contains("impl < T > hm :: nn :: Module for M < T > where T : hm :: nn :: Member")
    );
    assert!(expansion.contains("hm :: nn :: Visitor :: module (__visitor"));
    assert!(!expansion.contains("homura"));
    let expansion = expanded("#[module(crate = \"::a::b\")]\nstruct M;");
    assert!(expansion.contains(":: a :: b :: nn :: Member for M"));
}

#[test]
fn fields_that_mention_the_struct_are_not_bounds() {
    let expansion = expanded(
        "struct Tree<T> {\n    value: T,\n    children: Vec<Tree<T>>,\n    parent: Option<Box<Self>>,\n}",
    );
    assert!(
        expansion.contains("Module for Tree < T > where T : :: homura :: nn :: Member {"),
        "{expansion}"
    );
}

#[test]
fn bounds_follow_type_paths_not_spellings() {
    let bounds = |source: &str| {
        let expansion = expanded(source);
        let start = expansion.find("where").expect("bounds") + "where".len();
        let end = start + expansion[start..].find('{').unwrap();
        expansion[start..end].trim().to_string()
    };
    // Another type with the struct's name, and paths that only start like the struct.
    assert_eq!(
        bounds("struct Wrap<T> {\n    inner: child::Wrap<T>,\n    outer: ::Wrap<T>,\n}"),
        "child :: Wrap < T > : :: homura :: nn :: Member , :: Wrap < T > : :: homura :: nn :: \
         Member"
    );
    // A raw type parameter is the same parameter as its name without the prefix.
    assert_eq!(
        bounds("struct Wrap<r#T> {\n    inner: T,\n    other: Vec<r#T>,\n}"),
        "T : :: homura :: nn :: Member , Vec < r#T > : :: homura :: nn :: Member"
    );
    // Projections out of the struct and associated types of parameters are bounds.
    assert_eq!(
        bounds("struct P<T: Value> {\n    a: <Self as Value>::Item,\n    b: T::Output,\n}"),
        "< Self as Value > :: Item : :: homura :: nn :: Member , T :: Output : :: homura :: nn \
         :: Member"
    );
    // A macro that names a parameter; a type that only has the parameter's name as a later
    // path segment is another type.
    assert_eq!(
        bounds("struct M<T> {\n    a: boxed!(T),\n    b: other::T,\n    c: [u8; 2],\n}"),
        "boxed ! (T) : :: homura :: nn :: Member"
    );
    // The struct itself, directly or through `Self`, inside other types.
    assert!(
        !expanded("struct Tree<T> {\n    a: Option<Box<Self>>,\n    b: Vec<r#Tree<T>>,\n}")
            .contains("where")
    );
}

#[test]
fn generated_names_avoid_the_generic_parameters() {
    let expansion = expanded(
        "struct M<'__a, '__a1, const __visitor: usize, __name> {\n    a: [u8; __visitor],\n    \
         b: __name,\n}",
    );
    assert!(expansion.contains("fn visit < '__a2 > (& '__a2 self , __visitor1 : & mut dyn"));
    assert!(expansion.contains("visit_as (& self . a , \"a\" , __visitor1)"));
    assert!(
        expansion.contains("fn visit_as < '__a2 > (& '__a2 self , __name1 : & str , __visitor1")
    );
    assert!(expansion.contains("module (__visitor1 , __name1 , self ,"));
    // The caller's lint levels stay: no lint attributes besides `automatically_derived`.
    assert!(!expansion.contains("allow"));
    assert_eq!(expansion.matches("# [").count(), 2);
}
