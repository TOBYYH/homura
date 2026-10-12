//! Binary and ternary gradients: pinned PyTorch boundary semantics, broadcasting, higher
//! derivatives, saved values and metadata-only backward passes.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::{DType, Device, Tensor};

fn leaf(values: &[f64], shape: impl homura::IntoShape) -> Tensor {
    let x = Tensor::from_slice(values, shape);
    x.requires_grad_(true);
    x
}

fn gradients(out: &Tensor, inputs: &[&Tensor], g: &Tensor, graph: bool) -> Vec<Tensor> {
    autograd::grad(
        &[out],
        inputs,
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            create_graph: graph,
            ..Default::default()
        },
    )
    .into_iter()
    .map(Option::unwrap)
    .collect()
}

fn same(t: &Tensor, expected: &[f64]) {
    let actual = t.to(DType::F64).to_vec::<f64>();
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a == e || (a.is_nan() && e.is_nan()),
            "{actual:?} vs {expected:?}"
        );
    }
}

fn panic_message(f: impl FnOnce()) -> String {
    let p = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap()
}

#[test]
fn powers_at_zero_and_non_finite_values() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    let x = leaf(&[0., 0., 0., 0., -2., nan, inf], [7]);
    let e = leaf(&[0., 1., 2., -1., 2., 0., 0.], [7]);
    let g = Tensor::from_slice(&[inf, 2., 3., 1., 1., nan, inf], [7]);
    let ds = gradients(&x.pow(&e), &[&x, &e], &g, false);
    same(&ds[0], &[0., 2., 0., -inf, -4., 0., 0.]);
    same(&ds[1], &[nan, 0., 0., -inf, nan, nan, inf]);
    let x = leaf(&[0., -2., nan, inf], [4]);
    let g = Tensor::from_slice(&[inf, nan, inf, nan], [4]);
    let dx = gradients(&x.pow(0.), &[&x], &g, true).remove(0);
    same(&dx, &[0.; 4]);
    assert!(
        !dx.requires_grad(),
        "scalar exponent zero is a constant gradient"
    );
    let e = leaf(&[-1., 0., 1., nan], [4]);
    same(
        &gradients(
            &Tensor::from_slice(&[0f64], ()).pow(&e),
            &[&e],
            &homura::ones_like(&e),
            false,
        )[0],
        &[-inf, 0., 0., nan],
    );
}

#[test]
fn extrema_split_ties_and_select_losing_non_finite_gradients() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    let x = leaf(&[1., 2., 3., nan, 1., inf, -inf, 0.], [8]);
    let y = leaf(&[2., 2., 1., 1., nan, inf, -inf, -0.], [8]);
    let g = Tensor::from_slice(&[inf, 4., nan, 3., 5., 6., -inf, nan], [8]);
    let ds = gradients(&x.maximum(&y), &[&x, &y], &g, false);
    same(&ds[0], &[0., 2., nan, 3., 5., 3., -inf, nan]);
    same(&ds[1], &[inf, 2., 0., 3., 5., 3., -inf, nan]);
    let ds = gradients(&x.minimum(&y), &[&x, &y], &g, false);
    same(&ds[0], &[inf, 2., 0., 3., 5., 3., -inf, nan]);
    same(&ds[1], &[0., 2., nan, 3., 5., 3., -inf, nan]);
}

#[test]
fn scalar_clamp_excludes_the_bounds_and_nan() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    let x = leaf(&[-2., -1., 0., 1., 2., nan, inf, -inf], [8]);
    let g = Tensor::from_slice(&[inf, nan, 3., inf, nan, 6., 7., 8.], [8]);
    same(
        &gradients(&x.clamp(-1., 1.), &[&x], &g, false)[0],
        &[0., 0., 3., 0., 0., 0., 0., 0.],
    );
    same(
        &gradients(&x.clamp_min(-1.), &[&x], &g, false)[0],
        &[0., 0., 3., inf, nan, 0., 7., 0.],
    );
    same(
        &gradients(&x.clamp_max(1.), &[&x], &g, false)[0],
        &[inf, nan, 3., 0., 0., 0., 0., 8.],
    );
    for (lo, hi) in [(1., 1.), (2., -1.), (nan, 1.), (-1., nan)] {
        same(&gradients(&x.clamp(lo, hi), &[&x], &g, false)[0], &[0.; 8]);
    }
}

#[test]
fn tensor_clamp_handles_ordered_equal_reversed_and_nan_bounds() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    let x = leaf(&[-2., -1., 0., 1., 2., 0., -1., 1., 0., nan, 0., 0.], [12]);
    let lo = leaf(
        &[-1., -1., -1., -1., -1., 0., 0., 0., 2., 2., nan, -1.],
        [12],
    );
    let hi = leaf(&[1., 1., 1., 1., 1., 0., 0., 0., -2., -2., 1., nan], [12]);
    let g = Tensor::from_slice(&[inf, 4., 3., 6., nan, 5., 7., 8., 9., 10., 11., 12.], [12]);
    let ds = gradients(&x.clamp(&lo, &hi), &[&x, &lo, &hi], &g, false);
    same(&ds[0], &[0., 2., 3., 3., 0., 5., 0., 0., 0., 10., 11., 12.]);
    same(&ds[1], &[inf, 2., 0., 0., 0., 0., 7., 0., 0., 0., 0., 0.]);
    same(&ds[2], &[0., 0., 0., 3., nan, 0., 0., 8., 9., 10., 0., 0.]);
    let x = leaf(&[-1., 0., 1., nan, 1.], [5]);
    let b = leaf(&[0., 0., 0., 0., nan], [5]);
    let g = Tensor::from_slice(&[inf, 4., nan, 3., 5.], [5]);
    let ds = gradients(&x.clamp_min(&b), &[&x, &b], &g, false);
    same(&ds[0], &[0., 2., nan, 3., 5.]);
    same(&ds[1], &[inf, 2., 0., 3., 5.]);
    let ds = gradients(&x.clamp_max(&b), &[&x, &b], &g, false);
    same(&ds[0], &[inf, 2., 0., 3., 5.]);
    same(&ds[1], &[0., 2., nan, 3., 5.]);
}

#[test]
fn mixed_clamp_keeps_the_effective_scalar_bound_and_forward_dtype() {
    // The scalar minimum rounds to f16 before the f64 maximum widens the result. Using
    // its original f64 value in backward would miss the tie at 1.
    let x = Tensor::from_slice(&[0.5f64, 1., 2.], [3]).to(DType::F16);
    x.requires_grad_(true);
    let hi = leaf(&[3.; 3], [3]);
    let out = x.clamp(1.0002, &hi);
    assert_eq!(out.dtype(), DType::F64);
    same(&out, &[1., 1., 2.]);
    let ds = gradients(&out, &[&x, &hi], &homura::ones_like(&out), false);
    same(&ds[0], &[0., 0.5, 1.]);
    same(&ds[1], &[0.; 3]);
    // An integer input rounds the scalar minimum before a floating maximum promotes it.
    let x = Tensor::from_slice(&[0i64, 1, 2], [3]);
    let hi = leaf(&[0.5, 2., 3.], [3]);
    let out = x.clamp(1i64, &hi);
    same(&out, &[0.5, 1., 2.]);
    same(
        &gradients(&out, &[&hi], &homura::ones_like(&out), false)[0],
        &[1., 0., 0.],
    );
    let x = leaf(&[0., 1., 2.], [3]);
    let lo = leaf(&[-1.; 3], [3]);
    let out = x.clamp(&lo, 1.);
    let ds = gradients(&out, &[&x, &lo], &homura::ones_like(&out), false);
    same(&ds[0], &[1., 0.5, 0.]);
    same(&ds[1], &[0.; 3]);
}

#[test]
fn binary_and_ternary_formulas_pass_both_gradient_checks() {
    let x = leaf(&[0.7, 1.2, 1.8, 2.3, 0.9, 1.5], [3, 2])
        .detach()
        .transpose(0, 1);
    x.requires_grad_(true);
    let y = leaf(&[0.4, 1.4, 2.], [3]);
    let z = leaf(&[0.2, 1.7], [2, 1]);
    let s = leaf(&[0.3], ());
    type F = fn(&[Tensor]) -> Vec<Tensor>;
    let cases: &[(&str, F)] = &[
        ("pow", |t| vec![t[0].pow(&t[1]), t[1].pow(&t[2])]),
        ("scalar powers", |t| {
            vec![
                t[0].pow(2.5),
                Tensor::from_slice(&[1.7f64], ()).pow(&t[0]),
                t[0].pow(0.),
                t[0].pow(1.),
            ]
        }),
        ("extrema", |t| {
            vec![
                t[0].maximum(&t[1]),
                t[0].minimum(&t[2]),
                t[0].maximum(1.1),
                t[0].minimum(1.1),
            ]
        }),
        ("scalar clamp", |t| {
            vec![
                t[0].clamp(0.8, 1.9),
                t[0].clamp_min(1.1),
                t[0].clamp_max(1.1),
            ]
        }),
        ("tensor clamp", |t| {
            vec![
                t[0].clamp(&t[1], &t[2]),
                t[0].clamp_min(&t[1]),
                t[0].clamp_max(&t[2]),
            ]
        }),
        ("mixed clamp", |t| {
            vec![t[0].clamp(0.8, &t[1]), t[0].clamp(&t[2], 1.9)]
        }),
        ("masked_fill", |t| {
            vec![
                t[0].masked_fill(&homura::tensor([true, false, true]), &t[3]),
                t[0].masked_fill(&homura::tensor([true, false, true]), 0.5),
            ]
        }),
        ("addcmul and addcdiv", |t| {
            vec![
                t[0].addcmul(&t[1], &t[2], 1.7),
                t[0].addcdiv(&t[1], &t[2], -0.3),
            ]
        }),
        ("lerp", |t| {
            vec![
                t[0].lerp(&t[1], &t[2]),
                t[0].lerp(&t[1], 0.2),
                t[0].lerp(&t[1], 0.8),
            ]
        }),
        ("zero dimensions", |t| {
            vec![
                t[3].pow(&t[1]),
                t[3].maximum(&t[1]),
                t[3].clamp(&t[1], &t[2]),
            ]
        }),
    ];
    for (name, f) in cases {
        let inputs = [&x, &y, &z, &s];
        gradcheck(f, &inputs, GradcheckOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        gradgradcheck(f, &inputs, GradcheckOptions::default())
            .unwrap_or_else(|e| panic!("{name} second order: {e}"));
    }
}

#[test]
fn empty_and_meta_inputs_keep_broadcast_gradient_shapes() {
    for (device, n) in [(Device::Cpu, 0), (Device::Meta, 1usize << 40)] {
        let x = homura::zeros_with([n, 3], (DType::F32, device));
        let y = homura::ones_with([3], (DType::F64, device));
        let z = homura::ones_with((), (DType::F32, device));
        for t in [&x, &y, &z] {
            t.requires_grad_(true);
        }
        for out in [
            x.pow(&y),
            x.maximum(&y),
            x.minimum(&y),
            x.clamp(&z, &y),
            x.clamp_min(&y),
            x.clamp_max(&z),
            x.clamp(0., &y),
            x.clamp(&z, 2.),
            x.addcmul(&y, &z, 0.5),
            x.addcdiv(&y, &z, 0.5),
            x.lerp(&y, &z),
        ] {
            let ds = autograd::grad(
                &[&out],
                &[&x, &y, &z],
                GradOptions {
                    grad_outputs: vec![Some(homura::ones_like(&out))],
                    allow_unused: true,
                    create_graph: true,
                    ..Default::default()
                },
            );
            for (input, d) in [&x, &y, &z].into_iter().zip(ds) {
                if let Some(d) = d {
                    assert_eq!(
                        (d.shape(), d.dtype(), d.device()),
                        (input.shape(), input.dtype(), device)
                    );
                    if device == Device::Cpu {
                        same(&d, &vec![0.; input.num_elements()]);
                    }
                } else {
                    assert!(!std::ptr::eq(input, &x));
                }
            }
        }
    }
}

#[test]
fn new_nodes_check_and_release_saved_operands() {
    type F = fn(&Tensor, &Tensor) -> Tensor;
    let cases: &[(&str, F)] = &[
        ("PowBackward0", |x, _| x.pow(0.)),
        ("PowBackward0", |x, _| x.pow(2.)),
        ("PowBackward1", |x, y| x.pow(y)),
        ("PowBackward1", |x, _| {
            Tensor::from_slice(&[2f64], ()).pow(x)
        }),
        ("MaximumBackward0", |x, y| x.maximum(y)),
        ("MinimumBackward0", |x, y| x.minimum(y)),
        ("ClampBackward1", |x, _| x.clamp(0., 3.)),
        ("ClampBackward0", |x, y| x.clamp(y, 3.)),
        ("ClampMinBackward0", |x, _| x.clamp_min(0.)),
        ("ClampMinBackward1", |x, y| x.clamp_min(y)),
        ("ClampMaxBackward0", |x, _| x.clamp_max(3.)),
        ("ClampMaxBackward1", |x, y| x.clamp_max(y)),
    ];
    for (name, f) in cases {
        let x = leaf(&[1., 2.], [2]);
        let y = leaf(&[0.5, 1.5], [2]);
        let out = f(&x, &y);
        assert_eq!(out.grad_fn().unwrap().name(), *name);
        let loss = out.sum();
        loss.backward();
        assert!(
            panic_message(|| loss.backward())
                .starts_with("Trying to backward through the graph a second time"),
            "{name}"
        );
        let x = leaf(&[1., 2.], [2]);
        let y = leaf(&[0.5, 1.5], [2]);
        let loss = f(&x, &y).sum();
        {
            let _guard = homura::no_grad();
            x.add_(1.);
        }
        // Scalar-base powers save the exponent/result rather than x as a base, but x is the
        // saved exponent, so a change is detected there too.
        assert!(
            panic_message(|| loss.backward()).contains("modified by an inplace operation"),
            "{name}"
        );
    }
}

#[test]
fn lerp_uses_direct_gradients_for_non_finite_weights() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    let x = leaf(&[0., inf, nan, 1., -inf], [5]);
    let y = leaf(&[inf, 0., 1., nan, 2.], [5]);
    let w = leaf(&[0., 1., inf, -inf, nan], [5]);
    let g = Tensor::from_slice(&[1., inf, nan, -inf, 0.], [5]);
    let ds = gradients(&x.lerp(&y, &w), &[&x, &y, &w], &g, false);
    same(&ds[0], &[1., nan, nan, -inf, nan]);
    same(&ds[1], &[0., inf, nan, inf, nan]);
    same(&ds[2], &[inf, -inf, nan, nan, nan]);
    // Scalar weights do not save endpoints; tensor weights save them only for dw.
    for (name, weight) in [
        ("LerpBackward0", homura::Operand::from(0.2)),
        ("LerpBackward1", homura::Operand::from(&w)),
    ] {
        let out = x.lerp(&y, weight);
        assert_eq!(out.grad_fn().unwrap().name(), name);
        assert_eq!(
            out.grad_fn().unwrap().next_functions().len(),
            if name == "LerpBackward0" { 2 } else { 3 }
        );
    }
    let x = leaf(&[1., 2.], [2]);
    let y = leaf(&[3., 4.], [2]);
    let loss = x.lerp(&y, 0.2).sum();
    {
        let _guard = homura::no_grad();
        x.add_(1.);
        y.add_(1.);
    }
    loss.backward();
    loss.backward();
    let w = leaf(&[0.2, 0.8], [2]);
    let loss = x.lerp(&y, &w).sum();
    {
        let _guard = homura::no_grad();
        y.add_(1.);
    }
    assert!(panic_message(|| loss.backward()).contains("modified by an inplace operation"));
}

#[test]
fn requested_edges_still_validate_all_saved_values() {
    // Even when only dx is requested, pow unpacks its saved result (saved for de) and
    // clamp unpacks both tensor bounds. This follows the engine's existing saved-value rule.
    let x = leaf(&[1., 2.], [2]);
    let e = leaf(&[2., 3.], [2]);
    let out = x.pow(&e);
    {
        let _guard = homura::no_grad();
        out.add_(1.);
    }
    assert!(
        panic_message(|| {
            drop(gradients(&out, &[&x], &homura::ones_like(&out), false));
        })
        .contains("modified by an inplace operation")
    );
    let x = leaf(&[1., 2.], [2]);
    let lo = leaf(&[0., 0.], [2]);
    let hi = leaf(&[3., 3.], [2]);
    let out = x.clamp(&lo, &hi);
    {
        let _guard = homura::no_grad();
        hi.add_(1.);
    }
    assert!(
        panic_message(|| {
            drop(gradients(&out, &[&lo], &homura::ones_like(&out), false));
        })
        .contains("modified by an inplace operation")
    );
    // Even with no exponent edge, the reference saves and checks the result.
    let x = leaf(&[1., 2.], [2]);
    let e = Tensor::from_slice(&[2f64, 3.], [2]);
    let out = x.pow(&e);
    {
        let _guard = homura::no_grad();
        out.add_(1.);
    }
    assert!(
        panic_message(|| {
            drop(gradients(&out, &[&x], &homura::ones_like(&out), false));
        })
        .contains("modified by an inplace operation")
    );
    // The reference checks the weight even when only dw needs gradients.
    let x = Tensor::from_slice(&[1f64, 2.], [2]);
    let end = Tensor::from_slice(&[3f64, 5.], [2]);
    let w = leaf(&[0.2, 0.8], [2]);
    let out = x.lerp(&end, &w);
    {
        let _guard = homura::no_grad();
        w.add_(1.);
    }
    assert!(
        panic_message(|| {
            drop(gradients(&out, &[&w], &homura::ones_like(&out), false));
        })
        .contains("modified by an inplace operation")
    );
}

#[test]
fn clamp_edges_and_zero_combination_gradients_remain_observable() {
    let x = leaf(&[1., 2.], [2]);
    let y = leaf(&[3., 4.], [2]);
    for (out, edges) in [
        (x.clamp(0., 3.), 1),
        (x.clamp_min(0.), 1),
        (x.clamp_max(3.), 1),
        (x.clamp_min(&y), 2),
        (x.clamp_max(&y), 2),
        (x.clamp(&y, &y), 3),
    ] {
        assert_eq!(out.grad_fn().unwrap().next_functions().len(), edges);
    }
    let value = leaf(&[0.3], ());
    let mask = homura::tensor([false, false]);
    let ds = gradients(
        &x.masked_fill(&mask, &value),
        &[&x, &value],
        &homura::ones_like(&x),
        false,
    );
    same(&ds[0], &[1., 1.]);
    same(&ds[1], &[0.]);
    for out in [x.addcmul(&y, &value, 0.), x.addcdiv(&y, &value, 0.)] {
        let ds = gradients(&out, &[&x, &y, &value], &homura::ones_like(&out), false);
        same(&ds[0], &[1., 1.]);
        same(&ds[1], &[0., 0.]);
        same(&ds[2], &[0.]);
    }
    let x = leaf(&[1., 2.], [2]);
    let _guard = homura::no_grad();
    for out in [
        x.pow(2.),
        x.maximum(&y),
        x.clamp(&y, 5.),
        x.lerp(&y, &value),
    ] {
        assert!(!out.requires_grad());
        assert!(out.grad_fn().is_none());
    }
}
