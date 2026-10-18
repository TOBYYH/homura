//! `#[derive(Module)]` under lint levels that a caller's crate may forbid: the expansion must not
//! override them. tests/nn_derive.rs has the unconventional generic names, allowed only at their
//! structs, under `deny`.
#![forbid(non_upper_case_globals, non_camel_case_types, non_snake_case)]

use homura::nn::{Module, Parameter};

#[derive(Module)]
struct Model {
    weight: Parameter,
}

#[derive(Module)]
struct Generic<'a, M, const N: usize> {
    layers: [M; N],
    label: &'a str,
}

#[test]
fn modules_derive_under_forbidden_lints() {
    let model = Model {
        weight: Parameter::new(homura::zeros([1])),
    };
    let generic = Generic {
        layers: [Model {
            weight: Parameter::new(homura::zeros([1])),
        }],
        label: "generic",
    };
    assert_eq!(model.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
    assert_eq!(
        generic.state_dict().keys().collect::<Vec<_>>(),
        ["layers.0.weight"]
    );
    assert_eq!(generic.label, "generic");
}
