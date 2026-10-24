//! Gradients of `softmax` and `log_softmax`: one node each that saves only the result (pinned
//! PyTorch names, edges and saved-tensor checks), first and second derivatives against the
//! composite `exp(x - max) / sum` for every dimension, non-contiguous inputs and upstream
//! gradients, rows with infinite and NaN values, half precision products in `f32` like the
//! reference's kernels, and zero-dimensional, empty and metadata-only inputs. The first
//! derivative is a node of its own, as in PyTorch, whose derivatives follow the reference's order
//! of operations: second derivatives against the pinned PyTorch (`second_grads.safetensors`),
//! including large finite upstream gradients, and that node's saved tensors. Run in debug and
//! release profiles.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden, assert_close_tol, tolerance};
use homura::autograd::{self, GradFn, GradOptions};
use homura::{DType, Device, Tensor};

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
            (x - y).abs() <= 1e-12 + 1e-12 * y.abs(),
            "{what}: {a:?} vs {e:?}"
        );
    }
}

fn softmax(x: &Tensor, dim: i64, log: bool) -> Tensor {
    if log {
        x.log_softmax(dim)
    } else {
        x.softmax(dim)
    }
}

/// The same function built from other differentiable operations. The maximum is detached: the
/// result doesn't depend on the shift, so neither do its derivatives.
fn composite(x: &Tensor, dim: i64, log: bool) -> Tensor {
    let shifted = x - &x.max_dim(dim, true).values.detach();
    let e = shifted.exp();
    if log {
        &shifted - &e.sum_dim(dim, true).log()
    } else {
        &e / &e.sum_dim(dim, true)
    }
}

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

/// Inputs of several shapes with the dimension to normalize over: every dimension, negative
/// ones, a dimension of size one, and non-contiguous inputs.
fn cases() -> Vec<(&'static str, Tensor, i64)> {
    vec![
        ("vector", values(&[5], 0), 0),
        ("rows", values(&[2, 3], 1), 1),
        ("columns", values(&[2, 3], 2), 0),
        ("middle", values(&[2, 3, 4], 3), -2),
        ("last of three", values(&[2, 3, 4], 4), -1),
        ("size one", values(&[3, 1, 2], 5), 1),
        ("transposed", values(&[4, 3], 6).transpose(0, 1), 1),
        ("transposed columns", values(&[4, 3], 7).transpose(0, 1), 0),
        (
            "stepped",
            values(&[3, 8], 8).index((.., (0..).step_by(2))),
            1,
        ),
        ("expanded", values(&[1, 3], 9).expand([2, 3]), 0),
    ]
}

/// A non-contiguous upstream gradient of `shape`: the first of each pair of elements of a wider
/// tensor.
fn upstream(shape: &[usize], seed: usize) -> Tensor {
    let wide = [shape, &[2]].concat();
    values(&wide, seed).mul(1.5).select(-1, 0)
}

// ---------------------------------------------------------------- nodes

#[test]
fn nodes_save_only_the_result_under_pytorch_names() {
    let x = leaf_of(values(&[2, 3], 0).to(DType::F32));
    let y = x.softmax(1);
    let f = y.grad_fn().unwrap();
    assert_eq!(f.name(), "SoftmaxBackward0");
    assert_eq!(edge_names(&f), [Some("AccumulateGrad")]);
    let y = x.mul(2.0).log_softmax(-1);
    let f = y.grad_fn().unwrap();
    assert_eq!(f.name(), "LogSoftmaxBackward0");
    assert_eq!(edge_names(&f), [Some("MulBackward0")]);
    // Nothing is recorded without an input that requires grad, or in no-grad mode.
    assert!(x.detach().softmax(0).grad_fn().is_none());
    {
        let _no_grad = homura::no_grad();
        assert!(x.log_softmax(0).grad_fn().is_none());
    }
    // The input is not saved: changing it afterwards leaves the gradient intact.
    for log in [false, true] {
        let x = leaf_of(Tensor::from_slice(&[0f32, 0.], [2]));
        let y = softmax(&x, 0, log);
        {
            let _no_grad = homura::no_grad();
            x.add_(5.0);
        }
        let g = Tensor::from_slice(&[1f32, 3.], [2]);
        let expected: [f32; 2] = if log { [-1., 1.] } else { [-0.5, 0.5] };
        assert_eq!(grads(&y, &[&x], &g, false)[0].to_vec::<f32>(), expected);
    }
}

#[test]
fn the_saved_result_is_checked() {
    // Measured on the pinned PyTorch: "…, which is output 0 of Softmax, is at version 1; …"
    for (log, operation) in [(false, "Softmax"), (true, "LogSoftmax")] {
        let x = leaf_of(Tensor::from_slice(&[1f32, 2.], [2]));
        let y = softmax(&x, 0, log);
        {
            let _no_grad = homura::no_grad();
            y.mul_(2.0);
        }
        let message = panic_message(|| y.sum().backward());
        assert!(
            message.contains(&format!(
                "[f32 [2]], which is output 0 of {operation}, is at version 1; expected version \
                 0 instead."
            )),
            "{message}"
        );
        // Without retain_graph, the saved result is freed by the first backward pass.
        let y = softmax(&x, 0, log);
        let s = y.sum();
        s.backward();
        let message = panic_message(|| s.backward());
        assert!(
            message.starts_with("Trying to backward through the graph a second time"),
            "{message}"
        );
    }
}

// ---------------------------------------------------------------- values

#[test]
fn first_derivatives_match_the_composite() {
    for (name, x, dim) in cases() {
        for log in [false, true] {
            let what = format!("{name}, log {log}");
            let x = leaf_of(x.deep_clone());
            let g = upstream(&x.shape(), 3);
            assert!(g.num_elements() < 2 || !g.is_contiguous(), "{what}");
            let actual = grads(&softmax(&x, dim, log), &[&x], &g, false).remove(0);
            let expected = grads(&composite(&x, dim, log), &[&x], &g, false).remove(0);
            assert_close(&actual, &expected, &what);
        }
    }
}

#[test]
fn second_derivatives_match_the_composite() {
    for (name, x, dim) in cases() {
        for log in [false, true] {
            let what = format!("{name}, log {log}");
            let x = leaf_of(x.deep_clone());
            let g = leaf_of(upstream(&x.shape(), 4));
            let r = upstream(&x.shape(), 5);
            let second = |f: &dyn Fn(&Tensor) -> Tensor| {
                let first = grads(&f(&x), &[&x], &g, true).remove(0);
                grads(&first, &[&x, &g], &r, false)
            };
            let actual = second(&|x| softmax(x, dim, log));
            let expected = second(&|x| composite(x, dim, log));
            assert_close(&actual[0], &expected[0], &format!("{what}, d/dx"));
            assert_close(&actual[1], &expected[1], &format!("{what}, d/dg"));
        }
    }
}

#[test]
fn rows_with_infinities_and_nan_follow_the_reference() {
    // Measured on the pinned PyTorch, in F32 and F64. Each input row is repeated for five
    // upstream gradient rows. Masked positions (`-inf`) of a row that keeps a finite element get
    // `-0.0` from softmax and `g` from log-softmax; an infinite or NaN gradient spreads through
    // the row's sum; a row without a finite element is NaN.
    let g_rows = [
        [1., 2., 4.],
        [1., INF, 2.],
        [INF, 1., 2.],
        [NAN, 1., 2.],
        [1., -INF, 2.],
    ];
    let nan_rows = [[NAN; 3]; 5];
    let soft_masked = [
        [-0.0, -0.5, 0.5],
        [NAN, NAN, -INF],
        [NAN, NAN, NAN],
        [NAN, NAN, NAN],
        [NAN, NAN, INF],
    ];
    let log_middle = [
        [1., -5., 4.],
        [NAN, NAN, NAN],
        [NAN, -INF, NAN],
        [NAN, NAN, NAN],
        [NAN, NAN, NAN],
    ];
    let log_first = [
        [-6., 2., 4.],
        [-INF, NAN, NAN],
        [NAN, NAN, NAN],
        [NAN, NAN, NAN],
        [INF, NAN, NAN],
    ];
    let cases: [(bool, [f64; 3], [[f64; 3]; 5]); 8] = [
        (false, [-INF, -INF, -INF], nan_rows),
        (false, [-INF, 0., 0.], soft_masked),
        (false, [INF, 0., 0.], nan_rows),
        (false, [NAN, 0., 0.], nan_rows),
        (true, [-INF, -INF, -INF], nan_rows),
        (true, [-INF, 0., -INF], log_middle),
        (true, [INF, 0., -INF], nan_rows),
        (true, [0., -INF, -INF], log_first),
    ];
    let g: Vec<f64> = g_rows.concat();
    for dtype in [DType::F32, DType::F64] {
        for (log, row, expected) in cases {
            let what = format!("{dtype}, log {log}, {row:?}");
            let x = leaf_of(Tensor::from_vec([row; 5].concat(), [5, 3]).to(dtype));
            let g = Tensor::from_vec(g.clone(), [5, 3]).to(dtype);
            let actual = grads(&softmax(&x, 1, log), &[&x], &g, false).remove(0);
            assert_eq!(actual.dtype(), dtype, "{what}");
            let expected = Tensor::from_vec(expected.concat(), [5, 3]);
            assert_eq!(exact(&actual), exact(&expected), "{what}");
            // The same rows as columns, normalized along the first dimension.
            let xt = leaf_of(x.detach().transpose(0, 1).deep_clone());
            let gt = g.transpose(0, 1);
            let actual = grads(&softmax(&xt, 0, log), &[&xt], &gt, false).remove(0);
            assert_eq!(
                exact(&actual),
                exact(&expected.transpose(0, 1)),
                "{what}, columns"
            );
        }
    }
}

#[test]
fn half_precision_is_computed_in_f32_and_rounded_once() {
    // Measured on the pinned PyTorch, whose kernels convert to `f32`, sum there and round each
    // gradient once; homura's sums accumulate in `f64` and are rounded to `f32`, which is exact
    // here. Rounding the sum to the dtype first would give -128 (F16) or -16 (BF16) for softmax
    // and -4 for log-softmax.
    let cases = [
        (DType::F16, 2048., [383.75, -127.9375, -127.9375, -127.9375]),
        (DType::BF16, 256., [47.75, -15.9375, -15.9375, -15.9375]),
    ];
    for (dtype, big, expected) in cases {
        let g = Tensor::from_slice(&[big, 1., 1., 1.], [4]).to(dtype);
        let x = leaf_of(homura::zeros([4]).to(dtype));
        let actual = grads(&x.softmax(0), &[&x], &g, false).remove(0);
        assert_eq!(actual.dtype(), dtype);
        assert_eq!(actual.to(DType::F64).to_vec::<f64>(), expected, "{dtype}");
        let x = leaf_of(Tensor::from_slice(&[0., -INF, -INF, -INF], [4]).to(dtype));
        let actual = grads(&x.log_softmax(0), &[&x], &g, false).remove(0);
        assert_eq!(
            actual.to(DType::F64).to_vec::<f64>(),
            [-3., 1., 1., 1.],
            "{dtype}"
        );
        // Along the first of two dimensions.
        let x = leaf_of(homura::zeros([4, 2]).to(dtype));
        let g = Tensor::from_slice(&[big, 1., 1., big, 1., 1., 1., 1.], [4, 2]).to(dtype);
        let actual = grads(&x.softmax(0), &[&x], &g, false).remove(0);
        let [a, b, ..] = expected;
        assert_eq!(
            actual.to(DType::F64).to_vec::<f64>(),
            [a, b, b, a, b, b, b, b],
            "{dtype}"
        );
    }
}

// ---------------------------------------------------------------- shapes

#[test]
fn zero_dimensional_inputs_get_a_zero_gradient() {
    // The result is 1 (softmax) or 0 (log-softmax) whatever the input, so the gradient is zero;
    // PyTorch gives 0 for both with an upstream gradient of 3.
    for (log, dim) in [(false, 0), (false, -1), (true, 0), (true, -1)] {
        let x = leaf_of(Tensor::from_slice(&[2.5f32], []));
        let y = softmax(&x, dim, log);
        assert_eq!(y.shape(), []);
        let g = Tensor::from_slice(&[3f32], []);
        let gx = grads(&y, &[&x], &g, false).remove(0);
        assert_eq!((gx.shape(), gx.to_vec::<f32>()), ([].into(), vec![0.]));
    }
}

#[test]
fn empty_and_metadata_only_inputs() {
    for log in [false, true] {
        // Empty along the normalized dimension, and along another one.
        for (shape, dim) in [([2, 0], 1), ([0, 3], 1), ([0, 3], 0)] {
            let x = leaf_of(homura::zeros(shape));
            let y = softmax(&x, dim, log);
            assert_eq!(y.shape(), shape);
            let gx = grads(&y, &[&x], &homura::ones(shape), false).remove(0);
            assert_eq!(gx.shape(), shape);
        }
        // Huge metadata-only tensors: the gradient has the input's shape, dtype and device.
        let x = leaf_of(homura::zeros_with([1 << 40, 3], Device::Meta));
        let y = softmax(&x, 1, log);
        let g = homura::ones_with([1 << 40, 3], Device::Meta);
        let gx = grads(&y, &[&x], &g, false).remove(0);
        assert_eq!(gx.shape(), [1 << 40, 3]);
        assert_eq!((gx.dtype(), gx.device()), (DType::F32, Device::Meta));
        let x = leaf_of(homura::zeros_with([3, 1 << 40], Device::Meta).to(DType::F16));
        let g = homura::ones_with([3, 1 << 40], Device::Meta).to(DType::F16);
        let gx = grads(&softmax(&x, 0, log), &[&x], &g, false).remove(0);
        assert_eq!((gx.shape(), gx.dtype()), ([3, 1 << 40].into(), DType::F16));
    }
}

// ---------------------------------------------------------------- second derivatives

/// The first derivative of `softmax(x, dim)` (or `log_softmax`) for upstream gradient `g`, with
/// its graph.
fn first(x: &Tensor, dim: i64, log: bool, g: &Tensor) -> Tensor {
    grads(&softmax(x, dim, log), &[x], g, true).remove(0)
}

/// `t` with the same values in a non-contiguous (column-major) layout.
fn scattered(t: &Tensor) -> Tensor {
    t.transpose(0, 1).contiguous().transpose(0, 1)
}

#[test]
fn second_derivatives_match_pytorch() {
    // Each case requests the second derivatives with respect to the input x, the result y and
    // the upstream gradient g of the first pass, with non-contiguous g and r; "large" cases have
    // finite g near the top of the dtype's range, where the order of operations decides between
    // NaN, infinities and finite values (review round 26, R40).
    let golden = Golden::load("second_grads.safetensors");
    let cases = golden.cases();
    assert_eq!(cases.len(), 32);
    for name in &cases {
        let c = Case::new(&golden, name);
        let op = name.split_once('.').unwrap().1;
        let log = op.starts_with("log_softmax");
        let dim = if op.contains("_last_") { -1 } else { 0 };
        let x = leaf_of(c.input("x"));
        let g = leaf_of(scattered(&c.input("g")));
        let r = scattered(&c.input("r"));
        assert!(!g.is_contiguous() && !r.is_contiguous());
        let y = softmax(&x, dim, log);
        let a = grads(&y, &[&x], &g, true).remove(0);
        let (rtol, atol) = match x.dtype() {
            DType::F32 => (1e-5, 1e-5),
            DType::F64 => (1e-10, 1e-10),
            dtype => tolerance(dtype),
        };
        let [dx, dy, dg] = <[_; 3]>::try_from(grads(&a, &[&x, &y, &g], &r, false)).unwrap();
        for (what, actual) in [("a", &a), ("dx", &dx), ("dy", &dy), ("dg", &dg)] {
            c.check_with(what, actual, |actual, expected| {
                assert_close_tol(actual, expected, rtol, atol, &format!("{name}/{what}"))
            });
        }
        c.assert_all_used();
    }
}

#[test]
fn second_derivatives_follow_the_reference_order() {
    // Measured on the pinned PyTorch with x = [0, 0] and equal finite elements of g; the first
    // derivative is zero. Softmax's derivative with respect to its result is
    // `g * r - sum(y * g) * r - g * sum(y * r)`, which overflows to NaN where `r` is nonzero;
    // log-softmax's is `-sum(g) * exp(y) * r`, finite although `r * sum(g)` would overflow.
    for (dtype, big) in [
        (DType::F16, 60000.),
        (DType::BF16, 3e38),
        (DType::F32, 3e38),
        (DType::F64, 1e308),
    ] {
        let x = leaf_of(homura::zeros([2]).to(dtype));
        let g = leaf_of(Tensor::from_slice(&[big, big], [2]).to(dtype));
        let big = g.to(DType::F64).to_vec::<f64>()[0];
        let y = x.softmax(0);
        let a = grads(&y, &[&x], &g, true).remove(0);
        assert_eq!(exact(&a), exact(&homura::zeros([2])), "{dtype}");
        let r = Tensor::from_slice(&[0., 2.], [2]).to(dtype);
        let [dx, dy, dg] = <[_; 3]>::try_from(grads(&a, &[&x, &y, &g], &r, false)).unwrap();
        let expected = |v: [f64; 2]| exact(&Tensor::from_slice(&v, [2]));
        assert_eq!(exact(&dx), expected([NAN, NAN]), "{dtype}");
        assert_eq!(exact(&dy), expected([-big, NAN]), "{dtype}");
        assert_eq!(exact(&dg), expected([-0.5, 0.5]), "{dtype}");
    }
    for (dtype, big) in [
        (DType::F16, 30000.),
        (DType::BF16, 1e38),
        (DType::F32, 1e38),
        (DType::F64, 5e307),
    ] {
        let x = leaf_of(homura::zeros([2]).to(dtype));
        let g = leaf_of(Tensor::from_slice(&[big, big], [2]).to(dtype));
        let big = g.to(DType::F64).to_vec::<f64>()[0];
        let y = x.log_softmax(0);
        let a = grads(&y, &[&x], &g, true).remove(0);
        let r = Tensor::from_slice(&[2., 0.], [2]).to(dtype);
        let [dx, dy, dg] = <[_; 3]>::try_from(grads(&a, &[&x, &y, &g], &r, false)).unwrap();
        let expected = |v: [f64; 2]| exact(&Tensor::from_slice(&v, [2]));
        assert_eq!(exact(&dx), expected([-big, big]), "{dtype}");
        assert_eq!(exact(&dy), expected([-2. * big, -0.0]), "{dtype}");
        assert_eq!(exact(&dg), expected([1., -1.]), "{dtype}");
    }
}

#[test]
fn the_first_derivative_saves_the_upstream_gradient_and_the_result() {
    // Measured on the pinned PyTorch: `SoftmaxBackwardDataBackward0` (or
    // `LogSoftmaxBackwardDataBackward0`) with an edge to g and one to the softmax node, through
    // the result, saves both whichever gradients are requested later.
    for (log, name, forward) in [
        (false, "SoftmaxBackwardDataBackward0", "SoftmaxBackward0"),
        (
            true,
            "LogSoftmaxBackwardDataBackward0",
            "LogSoftmaxBackward0",
        ),
    ] {
        let x = leaf_of(Tensor::from_slice(&[0f32, 0.], [2]));
        let g = leaf_of(Tensor::from_slice(&[1f32, 1.], [2]));
        let a = first(&x, 0, log, &g);
        let f = a.grad_fn().unwrap();
        assert_eq!(f.name(), name);
        assert_eq!(edge_names(&f), [Some("AccumulateGrad"), Some(forward)]);
        // The third pass: softmax's gradient of g is the same node again, log-softmax's a
        // difference; the gradient of x goes through the first derivative's node.
        let r = leaf_of(Tensor::from_slice(&[0f32, 2.], [2]));
        let options = GradOptions {
            grad_outputs: vec![Some(r.clone())],
            create_graph: true,
            ..Default::default()
        };
        let second = autograd::grad(&[&a], &[&x, &g], options);
        let second_names: Vec<_> = second
            .iter()
            .map(|t| t.as_ref().unwrap().grad_fn().unwrap().name())
            .collect();
        let dg_name = if log { "SubBackward0" } else { name };
        assert_eq!(second_names, [name, dg_name]);
        let constant = first(&x, 0, log, &g.detach());
        assert_eq!(
            edge_names(&constant.grad_fn().unwrap()),
            [None, Some(forward)]
        );
        // Without create_graph, nothing is recorded.
        let plain = grads(&softmax(&x, 0, log), &[&x], &g, false).remove(0);
        assert!(plain.grad_fn().is_none() && !plain.requires_grad());
        // Modifying g fails the second pass, even when only the gradient of g is requested.
        let r = Tensor::from_slice(&[0f32, 2.], [2]);
        for request in ["x", "g"] {
            let g = leaf_of(Tensor::from_slice(&[1f32, 1.], [2]));
            let a = first(&x, 0, log, &g);
            {
                let _no_grad = homura::no_grad();
                g.mul_(2.0);
            }
            let message = panic_message(|| {
                grads(&a, &[if request == "x" { &x } else { &g }], &r, false);
            });
            assert!(
                message.contains("[f32 [2]] is at version 1; expected version 0 instead."),
                "{message}"
            );
        }
        // Without retain_graph, the second pass frees what the node saved.
        let a = first(&x, 0, log, &g);
        grads(&a, &[&g], &r, false);
        let message = panic_message(|| {
            grads(&a, &[&g], &r, false);
        });
        assert!(
            message.starts_with("Trying to backward through the graph a second time"),
            "{message}"
        );
    }
}

#[test]
fn the_second_pass_runs_only_what_the_requested_gradients_need() {
    // The input of softmax is a product whose saved factor is then modified: the gradient of g
    // doesn't need the softmax node or what lies behind it, the gradient of x does.
    for log in [false, true] {
        let x = leaf_of(Tensor::from_slice(&[0f64, 0.], [2]));
        let c = Tensor::from_slice(&[1f64, 1.], [2]);
        let z = &x * &c;
        let g = leaf_of(Tensor::from_slice(&[1f64, 3.], [2]));
        let a = grads(&softmax(&z, 0, log), &[&z], &g, true).remove(0);
        {
            let _no_grad = homura::no_grad();
            c.add_(1.0);
        }
        let r = Tensor::from_slice(&[0f64, 2.], [2]);
        let retain = GradOptions {
            grad_outputs: vec![Some(r.clone())],
            retain_graph: Some(true),
            ..Default::default()
        };
        let dg = autograd::grad(&[&a], &[&g], retain).remove(0).unwrap();
        let expected: [f64; 2] = if log { [-1., 1.] } else { [-0.5, 0.5] };
        assert_eq!(dg.to_vec::<f64>(), expected);
        let message = panic_message(|| {
            grads(&a, &[&x], &r, false);
        });
        assert!(
            message.contains("modified by an inplace operation"),
            "{message}"
        );
    }
}
