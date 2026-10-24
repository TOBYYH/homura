//! Version counters around `Module::to` and failed loads, which the public API doesn't show, and
//! `#[derive(Module)]` inside the crate.

use super::{Member, Module, Parameter, StateDict, Visitor};
use crate::Tensor;
use crate::{DType, Device};

fn bumped(values: &[f32], writes: usize) -> Parameter {
    let p = Parameter::new(crate::tensor(values.to_vec()));
    let _guard = crate::no_grad();
    for _ in 0..writes {
        p.add_(0.0);
    }
    p
}

#[test]
fn conversion_continues_the_version_in_new_storage() {
    let w = bumped(&[1.0, 2.0], 3);
    w.set_grad(Some(crate::ones([2])));
    let grad = w.grad().unwrap();
    let alias = w.detach();
    assert_eq!((w.version(), grad.version()), (3, 0));
    [w.clone()].to(DType::F64);
    // The gradient was replaced by a new tensor; the old one is untouched.
    assert_eq!(
        (w.version(), grad.version(), w.grad().unwrap().version()),
        (4, 0, 0)
    );
    // The old storage keeps its version; the alias doesn't share the new one.
    assert_eq!(alias.version(), 3);
    {
        let _guard = crate::no_grad();
        w.add_(1.0);
    }
    assert_eq!((w.version(), alias.version()), (5, 3));
    [w.clone()].to(Device::Meta);
    assert_eq!(w.version(), 6);
}

#[test]
fn failed_loads_and_conversions_keep_every_version() {
    let a = bumped(&[1.0, 2.0], 1);
    let b = bumped(&[1.0, 2.0], 2);
    let modules: Vec<Parameter> = vec![a.clone(), b.clone()];
    let versions = |ps: &[&Parameter]| ps.iter().map(|p| p.version()).collect::<Vec<_>>();
    let before = versions(&[&a, &b]);

    let sd: StateDict = [
        ("0", crate::ones([2])),
        ("1", crate::zeros_with([2], Device::Meta)),
    ]
    .into_iter()
    .collect();
    assert!(modules.load_state_dict(&sd).is_err());
    assert_eq!(versions(&[&a, &b]), before);

    let x = crate::ones([2]);
    x.requires_grad_(true);
    b.set_grad(Some(&x * 1.0));
    let failed = std::panic::catch_unwind(|| modules.to(DType::F64));
    assert!(failed.is_err());
    assert_eq!(versions(&[&a, &b]), before);
    assert_eq!(a.dtype(), DType::F32);
    b.set_grad(None);

    // A target too large for the dtype is found before anything is converted, on the meta
    // device without allocating.
    let mixed = Mixed {
        a: Parameter::new(crate::ones_with([1], Device::Meta)),
        huge: crate::zeros_with([isize::MAX as usize / 2], (DType::F16, Device::Meta)),
    };
    let before = (mixed.a.version(), mixed.huge.version());
    let failed = std::panic::catch_unwind(|| mixed.to(DType::F64));
    assert!(failed.is_err());
    assert_eq!((mixed.a.version(), mixed.huge.version()), before);
    assert_eq!(
        (mixed.a.dtype(), mixed.huge.dtype()),
        (DType::F32, DType::F16)
    );

    // A successful load bumps each target once per key.
    let sd: StateDict = [("0", crate::ones([2])), ("1", crate::ones([2]))]
        .into_iter()
        .collect();
    modules.load_state_dict(&sd).unwrap();
    assert_eq!(versions(&[&a, &b]), [2, 3]);
}

struct Mixed {
    a: Parameter,
    huge: Tensor,
}

impl Module for Mixed {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        self.a.visit_as("a", v);
        self.huge.visit_as("huge", v);
    }
}

impl Member for Mixed {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}

/// The expansion names the crate `::homura`, which `extern crate self` declares here.
#[derive(super::Module)]
struct Derived {
    weight: Parameter,
    #[module(non_persistent)]
    cache: Tensor,
}

#[test]
fn modules_derive_inside_the_crate() {
    let m = Derived {
        weight: Parameter::new(crate::zeros([1])),
        cache: crate::zeros([1]),
    };
    assert_eq!(m.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
    assert_eq!(m.named_buffers()[0].0, "cache");
}
