//! Gradients of `softmax` and `log_softmax`: one node each that saves only the result (pinned
//! PyTorch names, edges and saved-tensor checks), first and second derivatives against the
//! composite `exp(x - max) / sum` for every dimension, non-contiguous inputs and upstream
//! gradients, rows with infinite and NaN values, half precision computed in `f32` like the
//! reference's kernels, and zero-dimensional, empty and metadata-only inputs. Run in debug and
//! release profiles.

use std::panic::{AssertUnwindSafe, catch_unwind};

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
    let shifted = x - &x.max_dim(dim, true).0.detach();
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
    // gradient once. Rounding the sum to the dtype first would give -128 (F16) or -16 (BF16)
    // for softmax and -4 for log-softmax.
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
