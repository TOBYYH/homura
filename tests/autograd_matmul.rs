//! Gradients of `matmul` and the scaled products `addmm`, `addmv`, `addr`, `baddbmm` and
//! `addbmm`: node names and edges, values against an independent broadcast-and-sum product for
//! vectors, broadcast batches and non-contiguous inputs, second derivatives, what is saved, the
//! reference's order of scaling (the product first, then `alpha`), `beta = 0`, and
//! metadata-only and empty inputs. Run in debug and release profiles.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::autograd::{self, GradFn, GradOptions};
use homura::{DType, Device, Scalar, Tensor};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

/// Small deterministic values of `shape` in `F64`, different for each `seed`.
fn values(shape: &[usize], seed: usize) -> Tensor {
    let n: usize = shape.iter().product();
    let v = (0..n)
        .map(|i| ((i * 7 + seed * 5) % 11) as f64 / 4.0 - 1.25)
        .collect();
    Tensor::from_vec(v, shape)
}

fn leaf_of(t: Tensor) -> Tensor {
    t.requires_grad_(true);
    t
}

fn leaf(shape: &[usize], seed: usize) -> Tensor {
    leaf_of(values(shape, seed))
}

fn f32_leaf(v: &[f32], shape: impl homura::IntoShape) -> Tensor {
    leaf_of(Tensor::from_slice(v, shape))
}

fn grads(out: &Tensor, inputs: &[&Tensor], g: &Tensor, graph: bool) -> Vec<Tensor> {
    let options = GradOptions {
        grad_outputs: vec![Some(g.clone())],
        create_graph: graph,
        ..Default::default()
    };
    autograd::grad(&[out], inputs, options)
        .into_iter()
        .map(Option::unwrap)
        .collect()
}

fn edge_names(f: &GradFn) -> Vec<Option<&'static str>> {
    f.next_functions()
        .iter()
        .map(|e| e.as_ref().map(|(f, _)| f.name()))
        .collect()
}

/// The values of `t` as `f64`, with NaN mapped to a marker so that vectors compare exactly.
fn exact(t: &Tensor) -> Vec<String> {
    t.to(DType::F64)
        .to_vec::<f64>()
        .iter()
        .map(|v| format!("{v:?}"))
        .collect()
}

fn assert_close(actual: &Tensor, expected: &Tensor, what: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{what}");
    let (a, e) = (actual.to_vec::<f64>(), expected.to_vec::<f64>());
    for (x, y) in a.iter().zip(&e) {
        assert!(
            (x - y).abs() <= 1e-12 * y.abs().max(1.0),
            "{what}: {a:?} vs {e:?}"
        );
    }
}

/// `matmul` built from operations with earlier formulas: a broadcast product summed over the
/// inner dimension, with 1-D operands as a row (left) or column (right).
fn reference_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (da, db) = (a.num_dims(), b.num_dims());
    let a2 = if da == 1 { a.unsqueeze(0) } else { a.clone() };
    let b2 = if db == 1 { b.unsqueeze(1) } else { b.clone() };
    let mut out = (a2.unsqueeze(-1) * b2.unsqueeze(-3)).sum_dim(-2, false);
    if db == 1 {
        out = out.squeeze_dim(-1);
    }
    if da == 1 {
        out = out.squeeze_dim(-1 - (db > 1) as i64);
    }
    out
}

/// Pairs of operand shapes: dot, matrix-vector both ways, matrix product, batches on either or
/// both sides, batch dimensions that broadcast both ways (so the batch can't be merged without a
/// copy), different batch ranks, and zero-length dimensions.
const SHAPES: &[(&[usize], &[usize])] = &[
    (&[3], &[3]),
    (&[2, 3], &[3]),
    (&[3], &[3, 4]),
    (&[2, 3], &[3, 4]),
    (&[5, 2, 3], &[3, 4]),
    (&[2, 3], &[5, 3, 4]),
    (&[5, 2, 3], &[5, 3, 4]),
    (&[2, 1, 3, 4], &[1, 5, 4, 6]),
    (&[3], &[5, 3, 4]),
    (&[5, 2, 3], &[3]),
    (&[2, 5, 2, 3], &[5, 3, 4]),
    (&[2, 2, 3], &[3, 1]),
    (&[1, 3], &[2, 3, 4]),
    (&[0, 2, 3], &[3, 4]),
    (&[2, 0], &[0, 3]),
    (&[2, 3], &[3, 0]),
    (&[0], &[0]),
];

// ---------------------------------------------------------------- matmul

#[test]
fn matmul_records_one_node_named_after_the_reference() {
    // PyTorch decomposes `matmul`: a single `DotBackward0`, `MvBackward0` or `MmBackward0` for
    // 1-D x 1-D, 2-D x 1-D and 2-D x 2-D; views around `mm`, `mv` or `bmm` otherwise. homura
    // records one node, named like the reference's where that is a single node, and
    // `MatmulBackward0` (the reference's name for its own matmul formula) for the other shapes.
    let cases: &[(&[usize], &[usize], &str)] = &[
        (&[3], &[3], "DotBackward0"),
        (&[2, 3], &[3], "MvBackward0"),
        (&[2, 3], &[3, 4], "MmBackward0"),
        (&[3], &[3, 4], "MatmulBackward0"),
        (&[5, 2, 3], &[3, 4], "MatmulBackward0"),
        (&[2, 3], &[5, 3, 4], "MatmulBackward0"),
        (&[5, 2, 3], &[5, 3, 4], "MatmulBackward0"),
        (&[5, 2, 3], &[3], "MatmulBackward0"),
    ];
    for &(sa, sb, name) in cases {
        let (a, b) = (leaf(sa, 0), leaf(sb, 1));
        let f = a.matmul(&b).grad_fn().unwrap();
        assert_eq!(f.name(), name, "{sa:?} @ {sb:?}");
        assert_eq!(
            edge_names(&f),
            [Some("AccumulateGrad"), Some("AccumulateGrad")]
        );
        // An operand that doesn't require grad has no edge.
        let f = a.matmul(&b.detach()).grad_fn().unwrap();
        assert_eq!(edge_names(&f), [Some("AccumulateGrad"), None]);
    }
    // Integer products are not recorded.
    let i = homura::tensor([[1i64, 2], [3, 4]]);
    assert!(!i.matmul(&i).requires_grad());
}

#[test]
fn matmul_gradients_match_the_broadcast_product() {
    for (seed, &(sa, sb)) in SHAPES.iter().enumerate() {
        let what = format!("{sa:?} @ {sb:?}");
        let (a, b) = (leaf(sa, seed), leaf(sb, seed + 1));
        let out = a.matmul(&b);
        let expected = reference_matmul(&a, &b);
        assert_close(&out, &expected, &what);
        let g = values(&out.shape(), seed + 2);
        let actual = grads(&out, &[&a, &b], &g, false);
        let wanted = grads(&expected, &[&a, &b], &g, false);
        for (i, (x, y)) in actual.iter().zip(&wanted).enumerate() {
            assert_close(x, y, &format!("{what}: input {i}"));
        }
    }
}

#[test]
fn matmul_gradients_of_non_contiguous_operands() {
    // A transposed left operand, a stepped right one, and broadcast (stride 0) batches.
    let a = leaf(&[5, 3], 0);
    let b = leaf(&[5, 10], 1);
    let at = a.transpose(0, 1);
    let bs = b.index((.., (0..).step_by(2)));
    let e = leaf(&[1, 2, 3], 2);
    let ee = e.expand([4, 2, 3]);
    for (what, x, y) in [
        ("transposed", &at, &bs),
        ("expanded", &ee, &bs.narrow(0, 0, 3)),
    ] {
        let out = x.matmul(y);
        let expected = reference_matmul(x, y);
        assert_close(&out, &expected, what);
        let g = values(&out.shape(), 3).transpose(-1, -2).contiguous();
        let g = g.transpose(-1, -2);
        let inputs = [&a, &b, &e];
        let options = |g: &Tensor| GradOptions {
            grad_outputs: vec![Some(g.clone())],
            allow_unused: true,
            ..Default::default()
        };
        let actual = autograd::grad(&[&out], &inputs, options(&g));
        let wanted = autograd::grad(&[&expected], &inputs, options(&g));
        for (x, y) in actual.iter().zip(&wanted) {
            match (x, y) {
                (Some(x), Some(y)) => assert_close(x, y, what),
                (x, y) => assert_eq!(x.is_some(), y.is_some(), "{what}"),
            }
        }
    }
}

#[test]
fn matmul_differentiates_again() {
    for (seed, &(sa, sb)) in SHAPES.iter().enumerate() {
        let what = format!("{sa:?} @ {sb:?}");
        let (a, b) = (leaf(sa, seed), leaf(sb, seed + 1));
        let out = a.matmul(&b);
        let expected = reference_matmul(&a, &b);
        let g = leaf(&out.shape(), seed + 2);
        // Differentiates <grad_a, h_a> + <grad_b, h_b> with respect to a, b and g.
        let first = grads(&out, &[&a, &b], &g, true);
        let wanted = grads(&expected, &[&a, &b], &g, true);
        let h: Vec<Tensor> = first
            .iter()
            .enumerate()
            .map(|(i, t)| values(&t.shape(), seed + 3 + i))
            .collect();
        let dot = |gs: &[Tensor]| (&gs[0] * &h[0]).sum() + (&gs[1] * &h[1]).sum();
        let one = homura::ones_with([], DType::F64);
        let actual = grads(&dot(&first), &[&a, &b, &g], &one, false);
        let wanted = grads(&dot(&wanted), &[&a, &b, &g], &one, false);
        for (i, (x, y)) in actual.iter().zip(&wanted).enumerate() {
            assert_close(x, y, &format!("{what}: second derivative {i}"));
        }
    }
}

#[test]
fn an_unbatched_operand_gets_one_folded_product() {
    // Like the reference, the gradient of an operand without batch dimensions is one matrix
    // product over the batch and rows together, not a product per batch element summed
    // afterwards: for `[B, S, d] @ [d, e]` the weight's gradient is `x.flatten(0, -2)ᵀ @ g`. With
    // create_graph, that is an `mm` of the reshaped tensors.
    let x = leaf(&[2, 3, 4], 0);
    let w = leaf(&[4, 5], 1);
    let out = x.matmul(&w);
    let g = values(&out.shape(), 2);
    let gw = grads(&out, &[&w], &g, true).remove(0);
    assert_eq!(gw.grad_fn().unwrap().name(), "MmBackward0");
    let flat = x
        .detach()
        .flatten(0, 1)
        .transpose(0, 1)
        .matmul(&g.flatten(0, 1));
    assert_close(&gw, &flat, "weight");
    // The same on the left: `[d, e] @ [B, e, f]`.
    let w = leaf(&[3, 4], 3);
    let y = leaf(&[2, 4, 5], 4);
    let out = w.matmul(&y);
    let g = values(&out.shape(), 5);
    let gw = grads(&out, &[&w], &g, true).remove(0);
    assert_eq!(gw.grad_fn().unwrap().name(), "MmBackward0");
    assert_close(
        &gw,
        &grads(&reference_matmul(&w, &y), &[&w], &g, false)[0],
        "left",
    );
    // Batched on both sides: products in the full batch shape, summed to the operand's shape.
    let (a, b) = (leaf(&[1, 3, 4], 6), leaf(&[2, 4, 5], 7));
    let out = a.matmul(&b);
    let ga = grads(&out, &[&a], &values(&out.shape(), 8), true).remove(0);
    assert_ne!(ga.grad_fn().unwrap().name(), "MmBackward0");
}

#[test]
fn matmul_saves_each_operand_for_the_other() {
    // Measured on the pinned PyTorch: `mm` saves `self` when `mat2` needs a gradient and `mat2`
    // when `self` does, so only the saved one is checked for in-place changes.
    let a = leaf(&[2, 3], 0);
    let b = values(&[3, 4], 1);
    let y = a.matmul(&b).sum();
    {
        let _guard = homura::no_grad();
        b.add_(1.0);
    }
    assert!(panic_message(|| y.backward()).contains("modified by an inplace operation"));
    let b = values(&[3, 4], 1);
    let y = a.matmul(&b).sum();
    {
        let _guard = homura::no_grad();
        a.add_(1.0);
    }
    y.backward();
    // A second pass needs the released operand.
    let y = a.matmul(&b).sum();
    y.backward();
    assert!(
        panic_message(|| y.backward())
            .starts_with("Trying to backward through the graph a second time")
    );
}

#[test]
fn matmul_gradients_on_meta_and_empty_inputs() {
    for device in [Device::Cpu, Device::Meta] {
        for (sa, sb) in [
            (vec![2, 0, 3], vec![3, 4]),
            (vec![2, 3], vec![3, 0]),
            (vec![0], vec![0]),
            (vec![5, 0], vec![2, 0, 3]),
        ]
        .into_iter()
        .chain((device == Device::Meta).then(|| (vec![1 << 40, 3], vec![3, 2])))
        {
            let what = format!("{sa:?} @ {sb:?} on {device}");
            let a = leaf_of(homura::zeros_with(sa.as_slice(), device));
            let b = leaf_of(homura::zeros_with(sb.as_slice(), device));
            let out = a.matmul(&b);
            let g = leaf_of(homura::ones_with(out.shape(), device));
            let found = grads(&out, &[&a, &b], &g, true);
            for (input, grad) in [&a, &b].iter().zip(&found) {
                assert_eq!(grad.shape(), input.shape(), "{what}");
                assert_eq!(grad.device(), device, "{what}");
                if device == Device::Cpu {
                    assert!(grad.to_vec::<f32>().iter().all(|&v| v == 0.0), "{what}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------- scaled products

/// An operation with the shapes of its input and its two factors.
type Case = (&'static str, Vec<usize>, Vec<usize>, Vec<usize>);

/// The five scaled products with operands of matching shapes; `input` broadcasts to the result.
fn scaled_products() -> Vec<Case> {
    vec![
        ("addmm", vec![3], vec![2, 4], vec![4, 3]),
        ("addmv", vec![2], vec![2, 4], vec![4]),
        ("addr", vec![1, 3], vec![2], vec![3]),
        ("baddbmm", vec![2, 3], vec![5, 2, 4], vec![5, 4, 3]),
        ("addbmm", vec![2, 1], vec![5, 2, 4], vec![5, 4, 3]),
    ]
}

fn scaled(op: &str, x: &Tensor, a: &Tensor, b: &Tensor, beta: Scalar, alpha: Scalar) -> Tensor {
    match op {
        "addmm" => x.addmm(a, b, beta, alpha),
        "addmv" => x.addmv(a, b, beta, alpha),
        "addr" => x.addr(a, b, beta, alpha),
        "baddbmm" => x.baddbmm(a, b, beta, alpha),
        "addbmm" => x.addbmm(a, b, beta, alpha),
        _ => unreachable!(),
    }
}

/// The same scaled product from `mul`, `add` and the broadcast product.
fn reference_scaled(op: &str, x: &Tensor, a: &Tensor, b: &Tensor, beta: f64, alpha: f64) -> Tensor {
    let product = match op {
        "addr" => a.unsqueeze(1) * b.unsqueeze(0),
        "addbmm" => reference_matmul(a, b).sum_dim(0, false),
        _ => reference_matmul(a, b),
    };
    x * beta + product * alpha
}

#[test]
fn scaled_products_record_their_own_nodes() {
    // Measured on the pinned PyTorch: one node per operation, with an edge per tensor input.
    for (op, sx, sa, sb) in scaled_products() {
        let (x, a, b) = (leaf(&sx, 0), leaf(&sa, 1), leaf(&sb, 2));
        let out = scaled(op, &x, &a, &b, 2.into(), 3.into());
        let f = out.grad_fn().unwrap();
        let mut name = format!("{op}Backward0");
        name[..1].make_ascii_uppercase();
        assert_eq!(f.name(), name);
        assert_eq!(edge_names(&f), [Some("AccumulateGrad"); 3]);
        let out = scaled(op, &x.detach(), &a, &b.detach(), 2.into(), 3.into());
        assert_eq!(
            edge_names(&out.grad_fn().unwrap()),
            [None, Some("AccumulateGrad"), None]
        );
    }
}

#[test]
fn scaled_product_gradients_match_their_composites() {
    for (beta, alpha) in [
        (1.0, 1.0),
        (2.0, 3.0),
        (-0.5, 1.0),
        (1.0, -0.25),
        (0.0, 2.0),
    ] {
        for (seed, (op, sx, sa, sb)) in scaled_products().into_iter().enumerate() {
            let what = format!("{op}(beta {beta}, alpha {alpha})");
            let (x, a, b) = (leaf(&sx, seed), leaf(&sa, seed + 1), leaf(&sb, seed + 2));
            let out = scaled(op, &x, &a, &b, beta.into(), alpha.into());
            let expected = reference_scaled(op, &x, &a, &b, beta, alpha);
            assert_close(&out, &expected, &what);
            let g = leaf(&out.shape(), seed + 3);
            let actual = grads(&out, &[&x, &a, &b], &g, true);
            let wanted = grads(&expected, &[&x, &a, &b], &g, true);
            for (i, (p, q)) in actual.iter().zip(&wanted).enumerate() {
                assert_close(p, q, &format!("{what}: input {i}"));
            }
            // Second derivatives, including with respect to the upstream gradient.
            let h: Vec<Tensor> = (0..3).map(|i| values(&actual[i].shape(), i)).collect();
            let dot = |gs: &[Tensor]| {
                (0..3)
                    .map(|i| (&gs[i] * &h[i]).sum())
                    .reduce(|s, t| s + t)
                    .unwrap()
            };
            let one = homura::ones_with([], DType::F64);
            let inputs = [&x, &a, &b, &g];
            let options = GradOptions {
                grad_outputs: vec![Some(one)],
                allow_unused: true,
                retain_graph: Some(true),
                ..Default::default()
            };
            let second = autograd::grad(&[&dot(&actual)], &inputs, options.clone());
            let wanted = autograd::grad(&[&dot(&wanted)], &inputs, options);
            for (i, (p, q)) in second.iter().zip(&wanted).enumerate() {
                let zeros = || homura::zeros_with(inputs[i].shape(), DType::F64);
                let (p, q) = (
                    p.clone().unwrap_or_else(zeros),
                    q.clone().unwrap_or_else(zeros),
                );
                assert_close(&p, &q, &format!("{what}: second derivative {i}"));
            }
        }
    }
}

#[test]
fn beta_zero_gives_the_input_the_upstream_gradient_times_zero() {
    // The input is ignored by the value, but its gradient is `g * 0` summed to its shape, as
    // PyTorch's `maybe_multiply(grad, beta)` computes it: zeros where `g` is finite, NaN where
    // it isn't. A NaN in the ignored input doesn't matter.
    let x = f32_leaf(&[f32::NAN, 1., 2.], [3]);
    let a = f32_leaf(&[1., 2.], [2, 1]);
    let b = f32_leaf(&[3., 4., 5.], [1, 3]);
    let g = Tensor::from_slice(&[f32::INFINITY, 1., f32::NAN, 2., 3., 4.], [2, 3]);
    for beta in [Scalar::Float(0.0), Scalar::Int(0), Scalar::Bool(false)] {
        let out = x.addmm(&a, &b, beta, 1);
        assert_eq!(out.to_vec::<f32>(), [3., 4., 5., 6., 8., 10.]);
        assert_eq!(out.grad_fn().unwrap().name(), "AddmmBackward0");
        let [gx, ga, gb] = <[Tensor; 3]>::try_from(grads(&out, &[&x, &a, &b], &g, false)).unwrap();
        assert_eq!(
            exact(&gx),
            exact(&Tensor::from_slice(&[f32::NAN, 0., f32::NAN], [3]))
        );
        assert_eq!(exact(&ga), ["NaN", "38.0"]);
        assert_eq!(exact(&gb), ["inf", "7.0", "NaN"]);
    }
    // Every operation, including the input's broadcasting.
    for (op, sx, sa, sb) in scaled_products() {
        let (x, a, b) = (leaf(&sx, 0), leaf(&sa, 1), leaf(&sb, 2));
        let out = scaled(op, &x, &a, &b, 0.into(), 1.into());
        let g = homura::full_with(out.shape(), 2.0, DType::F64);
        let gx = grads(&out, &[&x], &g, false).remove(0);
        assert_eq!(gx.shape(), x.shape(), "{op}");
        assert!(gx.to_vec::<f64>().iter().all(|&v| v == 0.0), "{op}");
    }
}

#[test]
fn alpha_scales_the_product_of_the_gradient() {
    // The reference multiplies `g @ mat2ᵀ` by alpha, not `g * alpha` by `mat2ᵀ`. Pinned values
    // in F32: with alpha 0 an overflowing product gives NaN (the composite would give 0), with a
    // tiny alpha it gives Inf, and with a huge alpha a tiny product stays finite (the composite
    // would overflow `g * alpha` first).
    let g = Tensor::from_slice(&[1e10f32], [1, 1]);
    let one = |v: f32| f32_leaf(&[v], [1, 1]);
    let cases: [(f32, f64, &str); 3] = [
        (1e30, 0.0, "NaN"),
        (1e30, 1e-30, "inf"),
        (1e-30, 1e30, "1e10"),
    ];
    for (v, alpha, expected) in cases {
        let (x, a, b) = (one(0.0), one(v), one(v));
        let out = x.addmm(&a, &b, 1, alpha);
        let found = grads(&out, &[&x, &a, &b], &g, false);
        assert_eq!(exact(&found[0]), ["10000000000.0"]);
        for grad in &found[1..] {
            let value = grad.to_vec::<f32>()[0];
            match expected {
                "NaN" => assert!(value.is_nan(), "alpha {alpha}: {value}"),
                "inf" => assert_eq!(value, f32::INFINITY, "alpha {alpha}"),
                _ => assert!((value - 1e10).abs() <= 1e4, "alpha {alpha}: {value}"),
            }
        }
    }
    // The same order in every operation: alpha 0 and an overflowing product.
    for (op, sx, sa, sb) in scaled_products() {
        let big = |shape: &[usize]| leaf_of(homura::full_with(shape, 1e30, DType::F32));
        let (x, a, b) = (big(&sx), big(&sa), big(&sb));
        let out = scaled(op, &x, &a, &b, 1.into(), 0.into());
        let g = homura::full_with(out.shape(), 1e10, DType::F32);
        for grad in &grads(&out, &[&a, &b], &g, false) {
            assert!(grad.to_vec::<f32>().iter().all(|v| v.is_nan()), "{op}");
        }
    }
}

#[test]
fn half_precision_coefficients_scale_in_f32() {
    // `g * alpha` with alpha a scalar keeps alpha unrounded in f32 for F16, like the reference's
    // `maybe_multiply`: 1000 * 1.0004 = 1000.5, while alpha rounded to F16 first (1.0) would
    // give 1000.
    let x = leaf_of(homura::zeros_with([1, 1], DType::F16));
    let a = leaf_of(homura::ones_with([1, 1], DType::F16));
    let b = homura::ones_with([1, 1], DType::F16);
    let out = x.addmm(&a, &b, 1.0004, 1.0004);
    let g = homura::full_with([1, 1], 1000.0, DType::F16);
    let [gx, ga] = <[Tensor; 2]>::try_from(grads(&out, &[&x, &a], &g, false)).unwrap();
    assert_eq!(exact(&gx), ["1000.5"]);
    assert_eq!(exact(&ga), ["1000.5"]);
}

#[test]
fn scaled_products_save_the_factors_for_each_other() {
    for (op, sx, sa, sb) in scaled_products() {
        // Only the input needs a gradient: nothing is saved, so the graph can be run twice and
        // changing the factors afterwards doesn't matter.
        let (x, a, b) = (leaf(&sx, 0), values(&sa, 1), values(&sb, 2));
        let y = scaled(op, &x, &a, &b, 2.into(), 3.into()).sum();
        {
            let _guard = homura::no_grad();
            a.add_(1.0);
        }
        y.backward();
        y.backward();
        // The first factor needs a gradient: the second one is saved and checked.
        let (a, b) = (leaf(&sa, 1), values(&sb, 2));
        let y = scaled(op, &x, &a, &b, 2.into(), 3.into()).sum();
        {
            let _guard = homura::no_grad();
            a.add_(1.0);
        }
        y.backward();
        let y = scaled(op, &x, &a, &b, 2.into(), 3.into()).sum();
        {
            let _guard = homura::no_grad();
            b.add_(1.0);
        }
        let message = panic_message(|| y.backward());
        assert!(
            message.contains("modified by an inplace operation"),
            "{op}: {message}"
        );
    }
}

#[test]
fn addr_differentiates_promoted_inputs() {
    // homura extension: PyTorch's forward promotes mixed dtypes too, but its backward then fails
    // ("addmv input tensors must have the same dtype"). Here the promoted copies are recorded,
    // so each input gets its gradient in its own dtype.
    let x = leaf(&[2, 3], 0);
    let v1 = leaf_of(values(&[2], 1).to(DType::F32));
    let v2 = leaf_of(values(&[3], 2).to(DType::F32));
    let out = x.addr(&v1, &v2, 2, 3);
    assert_eq!(out.dtype(), DType::F64);
    let f = out.grad_fn().unwrap();
    assert_eq!(f.name(), "AddrBackward0");
    assert_eq!(
        edge_names(&f),
        [
            Some("AccumulateGrad"),
            Some("ToCopyBackward0"),
            Some("ToCopyBackward0")
        ]
    );
    let g = values(&[2, 3], 3);
    let found = grads(&out, &[&x, &v1, &v2], &g, false);
    let (w1, w2) = (v1.detach().to(DType::F64), v2.detach().to(DType::F64));
    let expected = [
        &g * 2.0,
        g.matmul(&w2) * 3.0,
        g.transpose(0, 1).matmul(&w1) * 3.0,
    ];
    for (i, (p, q)) in found.iter().zip(&expected).enumerate() {
        assert_eq!(p.dtype(), [DType::F64, DType::F32, DType::F32][i]);
        assert_close(&p.to(DType::F64), q, &format!("input {i}"));
    }
}

#[test]
fn scaled_product_gradients_on_meta_and_empty_inputs() {
    for device in [Device::Cpu, Device::Meta] {
        // `k == 0`: the product is all zeros, its factors are empty.
        let cases: Vec<Case> = vec![
            ("addmm", vec![2, 3], vec![2, 0], vec![0, 3]),
            ("addmv", vec![2], vec![2, 0], vec![0]),
            ("addr", vec![0, 3], vec![0], vec![3]),
            ("baddbmm", vec![0, 2, 3], vec![0, 2, 4], vec![0, 4, 3]),
            ("addbmm", vec![2, 3], vec![0, 2, 4], vec![0, 4, 3]),
        ];
        let huge = (device == Device::Meta)
            .then(|| ("addmm", vec![1 << 40], vec![2, 3], vec![3, 1 << 40]));
        for (op, sx, sa, sb) in cases.into_iter().chain(huge) {
            let what = format!("{op} on {device}");
            let t = |shape: &[usize]| leaf_of(homura::ones_with(shape, device));
            let (x, a, b) = (t(&sx), t(&sa), t(&sb));
            let out = scaled(op, &x, &a, &b, 2.into(), 3.into());
            let g = leaf_of(homura::ones_with(out.shape(), device));
            let found = grads(&out, &[&x, &a, &b], &g, true);
            for (input, grad) in [&x, &a, &b].iter().zip(&found) {
                assert_eq!(grad.shape(), input.shape(), "{what}");
                assert_eq!(grad.device(), device, "{what}");
            }
        }
    }
}
