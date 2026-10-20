//! M2 step 8b: `nn::functional::conv2d`. Outputs and gradients are compared with
//! `tests/golden/m2_conv_pool.safetensors` from `tools/golden/gen_m2_conv_pool.py`; the reference
//! behavior encoded in the other tests is asserted by `tools/golden/check_m2_conv_pool.py`, in
//! the same order.
//!
//! The floating point tolerances scale with the magnitude of what is summed: an element of a
//! convolution is compared within `ulps` units in the last place of its dtype plus `k * ε` times
//! the same element of `conv2d(|x|, |w|) + |b|` (in F64), and its gradients likewise with the
//! absolute values. `tools/golden/measure_m2_tolerances.py` measures how much of these the errors
//! use (with `HOMURA_TOLERANCE_LOG` set, this test writes its measurements there).

mod common;

use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden, assert_close_tol};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::nn::functional::{Conv2dOptions, Conv2dPadding, conv2d};
use homura::{DType, Device, Tensor};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn assert_panics(f: impl FnOnce(), message: &str) {
    assert_eq!(panic_message(f), message);
}

/// The node names of `t`'s graph, depth first, as `(depth, name)`.
fn chain(t: &Tensor) -> Vec<(usize, &'static str)> {
    fn walk(f: &autograd::GradFn, depth: usize, out: &mut Vec<(usize, &'static str)>) {
        out.push((depth, f.name()));
        for (next, _) in f.next_functions().into_iter().flatten() {
            walk(&next, depth + 1, out);
        }
    }
    let mut out = vec![];
    if let Some(f) = t.grad_fn() {
        walk(&f, 0, &mut out);
    }
    out
}

fn names(t: &Tensor) -> Vec<&'static str> {
    chain(t).into_iter().map(|(_, n)| n).collect()
}

const ACC: &str = "AccumulateGrad";

/// Deterministic values of both signs, between 0.3 and 1.3 in magnitude.
fn constant(shape: &[usize], seed: usize) -> Tensor {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let magnitude = 0.3 + ((i * 7 + seed * 3) % 11) as f64 / 10.0;
            if (i + seed).is_multiple_of(3) {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect();
    Tensor::from_vec(values, shape)
}

fn var(shape: &[usize], seed: usize) -> Tensor {
    let t = constant(shape, seed);
    t.requires_grad_(true);
    t
}

fn f64s(values: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape)
}

fn vals(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

fn ints(t: &Tensor) -> Vec<i64> {
    t.to(DType::I64).to_vec::<i64>()
}

fn grads(out: &Tensor, inputs: &[&Tensor], g: &Tensor, create_graph: bool) -> Vec<Option<Tensor>> {
    autograd::grad(
        &[out],
        inputs,
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            create_graph,
            allow_unused: true,
            ..Default::default()
        },
    )
}

fn opts(stride: [usize; 2], padding: impl Into<Conv2dPadding>) -> Conv2dOptions {
    Conv2dOptions {
        stride,
        padding: padding.into(),
        ..Default::default()
    }
}

fn pair(c: &Case, name: &str) -> [usize; 2] {
    let v = c.input(name).to_vec::<i64>();
    [v[0] as usize, v[1] as usize]
}

fn has(c: &Case, name: &str) -> bool {
    c.golden.names(c.name).iter().any(|n| n == name)
}

fn case_options(c: &Case) -> Conv2dOptions {
    let padding = if has(c, "opt.valid") {
        c.input("opt.valid");
        Conv2dPadding::Valid
    } else if has(c, "opt.same") {
        c.input("opt.same");
        Conv2dPadding::Same
    } else {
        Conv2dPadding::Explicit(pair(c, "opt.padding"))
    };
    Conv2dOptions {
        stride: pair(c, "opt.stride"),
        padding,
        dilation: pair(c, "opt.dilation"),
        groups: c.input("opt.groups").item::<i64>() as usize,
    }
}

/// `t`, transposed in its last two dimensions if the case stores `flag`.
fn maybe_transposed(c: &Case, flag: &str, t: &Tensor) -> Tensor {
    if has(c, flag) {
        c.input(flag);
        t.transpose(-2, -1)
    } else {
        t.clone()
    }
}

/// The convolution of the golden case `c` with the given input tensors.
fn evaluate(c: &Case, x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let x = maybe_transposed(c, "opt.transpose", x);
    let w = maybe_transposed(c, "opt.transpose_w", w);
    conv2d(&x, &w, b, case_options(c))
}

/// How a dtype's results are compared: `ulps` units in the last place of the expected value plus
/// `k` times the F32 epsilon of the magnitude bound (F64: its own epsilon).
fn allowance(dtype: DType) -> (f64, f64) {
    let f32_eps = f32::EPSILON as f64 / 2.0;
    match dtype {
        DType::F64 => (0.0, 64.0 * f64::EPSILON),
        DType::F32 => (2.0, 8.0 * f32_eps),
        // PyTorch's own half precision convolution is within 1.24 units of an F32 one rounded
        // (M2.2 #84); its derived gradients are F32 gradients rounded.
        DType::F16 | DType::BF16 => (1.5, 8.0 * f32_eps),
        _ => (0.0, 0.0),
    }
}

/// The spacing of `dtype` relative to a value: an upper bound of one unit in the last place.
fn unit(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 2f64.powi(-10),
        DType::BF16 => 2f64.powi(-7),
        DType::F32 => 2f64.powi(-23),
        _ => 2f64.powi(-52),
    }
}

/// The largest errors of a comparison: absolute, relative to the expected value (for nonzero
/// ones), relative to the magnitude bound, and the fraction of the allowance used.
#[derive(Default)]
struct Errors {
    abs: f64,
    rel: f64,
    norm: f64,
    used: f64,
}

/// Asserts `|actual - expected| <= ulps * unit * |expected| + k * bound` elementwise (NaN and
/// infinities must match) and returns the largest errors.
fn assert_within(actual: &Tensor, expected: &Tensor, bound: &Tensor, what: &str) -> Errors {
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{what}: dtype");
    let dtype = expected.dtype();
    let (ulps, k) = allowance(dtype);
    let (a, e, m) = (vals(actual), vals(expected), vals(bound));
    let mut errors = Errors::default();
    for i in 0..a.len() {
        let (x, y) = (a[i], e[i]);
        if y.is_nan() || x.is_nan() || y.is_infinite() || x.is_infinite() {
            assert!(
                (x.is_nan() && y.is_nan()) || x == y,
                "{what}: element {i}: got {x:e}, expected {y:e}"
            );
            continue;
        }
        let allowed = ulps * unit(dtype) * y.abs() + k * m[i] + f64::MIN_POSITIVE;
        let error = (x - y).abs();
        assert!(
            error <= allowed,
            "{what}: element {i}: got {x:e}, expected {y:e} (error {error:e}, allowed {allowed:e})"
        );
        errors.abs = errors.abs.max(error);
        if y != 0.0 {
            errors.rel = errors.rel.max(error / y.abs());
        }
        if m[i] > 0.0 {
            errors.norm = errors.norm.max(error / m[i]);
        }
        errors.used = errors.used.max(error / allowed);
    }
    errors
}

/// The magnitude bounds of the convolution of the case `c` and of its gradients: the same
/// convolution of `|x|`, `|w|` and `|b|` in F64 and its gradients for `|g|`.
fn bounds(c: &Case, args: &[(&str, Tensor)], g: Option<&Tensor>) -> Vec<Tensor> {
    let abs: Vec<Tensor> = args
        .iter()
        .map(|(_, t)| {
            let t = t.detach().to(DType::F64).abs();
            t.requires_grad_(true);
            t
        })
        .collect();
    let b = abs.get(2);
    let out = evaluate(c, &abs[0], &abs[1], b);
    let mut result = vec![out.detach()];
    if let Some(g) = g {
        let inputs: Vec<&Tensor> = abs.iter().collect();
        let g = g.detach().to(DType::F64).abs();
        result.extend(
            grads(&out, &inputs, &g, false)
                .into_iter()
                .map(Option::unwrap),
        );
    }
    result
}

/// Appends the errors of `tensor` of `case` as a line of JSON to the file that
/// `HOMURA_TOLERANCE_LOG` names, if it is set.
fn log_measurement(case: &str, tensor: &str, errors: &Errors) {
    let Ok(path) = std::env::var("HOMURA_TOLERANCE_LOG") else {
        return;
    };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("can't open the tolerance log");
    let Errors {
        abs,
        rel,
        norm,
        used,
    } = errors;
    writeln!(
        file,
        r#"{{"case": "{case}", "tensor": "{tensor}", "abs": {abs:e}, "rel": {rel:e}, "norm": {norm:e}, "used": {used:e}}}"#
    )
    .unwrap();
}

#[test]
fn conv2d_matches_pytorch() {
    let golden = Golden::load("m2_conv_pool.safetensors");
    let cases: Vec<String> = golden
        .cases()
        .into_iter()
        .filter(|c| c.starts_with("conv_"))
        .collect();
    // 17 cases in f64 and f32, 5 in f16 and bf16, 2 in i64.
    assert_eq!(cases.len(), 2 * 17 + 2 * 5 + 2);
    let mut second_order = 0;
    for name in &cases {
        let c = Case::new(&golden, name);
        let keys: Vec<&str> = ["x", "w", "b"].into_iter().filter(|k| has(&c, k)).collect();
        let args: Vec<(&str, Tensor)> = keys.iter().map(|&k| (k, c.input(k))).collect();
        let floating = args[0].1.is_floating_point();
        for (_, t) in &args {
            t.requires_grad_(floating);
        }
        let b = args.get(2).map(|(_, t)| t);
        let out = evaluate(&c, &args[0].1, &args[1].1, b);
        if !floating {
            c.check_with("out", &out, |a, e| {
                assert_eq!(ints(a), ints(e), "{name}/out")
            });
            c.assert_all_used();
            continue;
        }
        let second = has(&c, "grad2_g");
        let g = c.input("g");
        g.requires_grad_(second);
        let bound = bounds(&c, &args, Some(&g));
        c.check_with("out", &out, |a, e| {
            let errors = assert_within(a, e, &bound[0], &format!("{name}/out"));
            log_measurement(name, "out", &errors);
        });
        let inputs: Vec<&Tensor> = args.iter().map(|(_, t)| t).collect();
        let first = grads(&out, &inputs, &g, second);
        for (i, (k, _)) in args.iter().enumerate() {
            let key = format!("grad_{k}");
            c.check_with(&key, first[i].as_ref().unwrap(), |a, e| {
                let errors = assert_within(a, e, &bound[1 + i], &format!("{name}/{key}"));
                log_measurement(name, &key, &errors);
            });
        }
        if second {
            second_order += 1;
            let mut total = homura::zeros_with([], DType::F64);
            for (i, (k, _)) in args.iter().enumerate() {
                let gg = c.input(&format!("gg_{k}"));
                total = total.add(first[i].as_ref().unwrap().mul(&gg).sum());
            }
            let mut variables = inputs.clone();
            variables.push(&g);
            let second_grads = autograd::grad(
                &[&total],
                &variables,
                GradOptions {
                    allow_unused: true,
                    ..Default::default()
                },
            );
            for (k, t) in keys.iter().chain(&["g"]).zip(&second_grads) {
                let key = format!("grad2_{k}");
                if has(&c, &key) {
                    c.check_with(&key, t.as_ref().unwrap(), |a, e| {
                        assert_close_tol(a, e, 1e-11, 1e-11, &format!("{name}/{key}"))
                    });
                } else {
                    // The bias has no second derivative, as in PyTorch.
                    assert!(t.is_none(), "{name}/{key}");
                }
            }
        }
        c.assert_all_used();
    }
    assert_eq!(second_order, 6);
}

#[test]
fn conv2d_records_its_operations() {
    // PyTorch records one ConvolutionBackward0; homura records the operations of its composite.
    let (x, w, b) = (var(&[2, 3, 5, 5], 0), var(&[4, 3, 3, 3], 1), var(&[4], 2));
    assert_eq!(
        chain(&conv2d(&x, &w, Some(&b), Conv2dOptions::default())),
        [
            (0, "AddBackward0"),
            (1, "ViewBackward0"),
            (2, "PermuteBackward0"),
            (3, "MatmulBackward0"),
            (4, "UnsafeViewBackward0"),
            (5, "CloneBackward0"),
            (6, "PermuteBackward0"),
            (7, "ViewBackward0"),
            (8, "UnfoldBackward0"),
            (9, "UnfoldBackward0"),
            (10, ACC),
            (4, "TransposeBackward0"),
            (5, "ViewBackward0"),
            (6, ACC),
            (1, "ViewBackward0"),
            (2, ACC)
        ]
    );
    let unbatched = names(&conv2d(&x.select(0, 0), &w, None, Conv2dOptions::default()));
    assert_eq!(unbatched[0], "SqueezeBackward1");
    assert!(unbatched.contains(&"UnsqueezeBackward0"));
    let same = Conv2dOptions {
        padding: Conv2dPadding::Same,
        ..Default::default()
    };
    let even = var(&[4, 3, 2, 2], 3);
    // Any padding is a constant pad (PyTorch records one only for asymmetric "same" padding).
    assert!(names(&conv2d(&x, &even, None, same)).contains(&"ConstantPadNdBackward0"));
    assert!(names(&conv2d(&x, &w, None, same)).contains(&"ConstantPadNdBackward0"));
    let valid = Conv2dOptions {
        padding: Conv2dPadding::Valid,
        ..Default::default()
    };
    assert!(!names(&conv2d(&x, &w, None, valid)).contains(&"ConstantPadNdBackward0"));
    let dilated = Conv2dOptions {
        dilation: [2, 1],
        ..Default::default()
    };
    assert!(names(&conv2d(&x, &w, None, dilated)).contains(&"SliceBackward0"));
    let half = conv2d(
        &x.detach().to(DType::F16),
        &w.detach().to(DType::F16),
        None,
        Conv2dOptions::default(),
    );
    assert!(half.grad_fn().is_none());
    let xh = x.detach().to(DType::F16);
    xh.requires_grad_(true);
    let half = names(&conv2d(
        &xh,
        &w.detach().to(DType::F16),
        None,
        Conv2dOptions::default(),
    ));
    assert_eq!(half[0], "ToCopyBackward0");
    assert_eq!(half.iter().filter(|&&n| n == "ToCopyBackward0").count(), 2);
    let _no_grad = homura::no_grad();
    assert!(
        conv2d(&x, &w, Some(&b), Conv2dOptions::default())
            .grad_fn()
            .is_none()
    );
}

#[test]
fn conv2d_checks_its_arguments() {
    let (x, w, b) = (
        constant(&[2, 3, 5, 5], 0),
        constant(&[4, 3, 3, 3], 1),
        constant(&[4], 2),
    );
    let d = Conv2dOptions::default();
    let with = |f: fn(&mut Conv2dOptions)| {
        let mut o = d;
        f(&mut o);
        o
    };
    let plane = x.select(0, 0).select(0, 0);
    let bias_type = "Input type (f64) and bias type (f32) should be the same";
    let b32 = b.to(DType::F32);
    assert_panics(|| drop(conv2d(&x, &w, Some(&b32), d)), bias_type);
    assert_panics(|| drop(conv2d(&plane, &w, Some(&b32), d)), bias_type);
    // PyTorch's "valid" and "same" paths don't compare the bias dtype; homura does.
    let valid = with(|o| o.padding = Conv2dPadding::Valid);
    assert_panics(|| drop(conv2d(&x, &w, Some(&b32), valid)), bias_type);
    let groups0 = with(|o| o.groups = 0);
    assert_panics(
        || drop(conv2d(&plane, &w, None, groups0)),
        "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [5, 5]",
    );
    let w3 = w.select(0, 0);
    assert_panics(
        || drop(conv2d(&x, &w3.select(0, 0), None, groups0)),
        "weight should have at least three dimensions",
    );
    let groups = "non-positive groups is not supported";
    assert_panics(|| drop(conv2d(&x, &w3, None, groups0)), groups);
    let stride0 = with(|o| o.stride = [0, 0]);
    assert_panics(
        || {
            drop(conv2d(
                &x,
                &w,
                None,
                Conv2dOptions {
                    groups: 0,
                    ..stride0
                },
            ));
        },
        groups,
    );
    let same = with(|o| o.padding = Conv2dPadding::Same);
    assert_panics(
        || {
            drop(conv2d(
                &x,
                &w,
                None,
                Conv2dOptions {
                    groups: 0,
                    stride: [0, 0],
                    ..same
                },
            ));
        },
        groups,
    );
    assert_panics(
        || drop(conv2d(&x, &w3, None, stride0)),
        "expected stride to be a single integer value or a list of 1 values to match the \
         convolution dimensions, but got stride=[0, 0]",
    );
    assert_panics(
        || drop(conv2d(&x, &w.unsqueeze(0), None, d)),
        "expected stride to be a single integer value or a list of 3 values to match the \
         convolution dimensions, but got stride=[1, 1]",
    );
    assert_panics(
        || drop(conv2d(&x, &w3, None, same)),
        "Expected 3-dimensional input for 3-dimensional weight[3, 3, 3], but got 4-dimensional \
         input of size [2, 3, 5, 5] instead",
    );
    let strided_same = "padding='same' is not supported for strided convolutions";
    for stride in [[1, 2], [0, 0]] {
        assert_panics(
            || drop(conv2d(&x, &w, None, Conv2dOptions { stride, ..same })),
            strided_same,
        );
    }
    let stride = "non-positive stride is not supported";
    assert_panics(
        || {
            drop(conv2d(
                &x,
                &w,
                None,
                Conv2dOptions {
                    groups: 2,
                    ..stride0
                },
            ))
        },
        stride,
    );
    assert_panics(
        || {
            drop(conv2d(
                &x,
                &w,
                None,
                Conv2dOptions {
                    dilation: [0, 0],
                    ..stride0
                },
            ))
        },
        stride,
    );
    let dilation0 = with(|o| o.dilation = [0, 0]);
    assert_panics(
        || drop(conv2d(&x, &w.narrow(0, 0, 0), None, dilation0)),
        "Given groups=1, expected weight to be at least 1 at dimension 0, but got weight of size \
         [0, 3, 3, 3] instead",
    );
    let g4 = with(|o| o.groups = 4);
    assert_panics(
        || {
            drop(conv2d(
                &homura::ones([1, 4, 5, 5]),
                &homura::ones([1, 1, 3, 3]),
                None,
                g4,
            ))
        },
        "Given groups=4, expected weight to be at least 4 at dimension 0, but got weight of size \
         [1, 1, 3, 3] instead",
    );
    let g2_same = Conv2dOptions { groups: 2, ..same };
    assert_panics(
        || {
            drop(conv2d(
                &homura::ones([1, 4, 5, 5]),
                &homura::ones([3, 2, 3, 3]),
                None,
                g2_same,
            ))
        },
        "Given groups=2, expected weight to be divisible by 2 at dimension 0, but got weight of \
         size [[3, 2, 3, 3]] instead",
    );
    let two = x.narrow(1, 0, 2);
    assert_panics(
        || drop(conv2d(&two, &w, Some(&b.narrow(0, 0, 3)), d)),
        "Given groups=1, weight of size [4, 3, 3, 3], expected input[2, 2, 5, 5] to have 3 \
         channels, but got 2 channels instead",
    );
    assert_panics(
        || {
            drop(conv2d(
                &two.select(0, 0),
                &w.to(DType::F32),
                None,
                dilation0,
            ))
        },
        "Given groups=1, weight of size [4, 3, 3, 3], expected input[1, 2, 5, 5] to have 3 \
         channels, but got 2 channels instead",
    );
    assert_panics(
        || drop(conv2d(&x, &w, Some(&b.unsqueeze(0)), d)),
        "Given weight of size [4, 3, 3, 3], expected bias to be 1-dimensional with 4 elements, \
         but got bias of size [1, 4] instead",
    );
    assert_panics(
        || drop(conv2d(&x, &w, Some(&b.narrow(0, 0, 2)), dilation0)),
        "Given weight of size [4, 3, 3, 3], expected bias to be 1-dimensional with 4 elements, \
         but got bias of size [2] instead",
    );
    let huge = with(|o| o.padding = Conv2dPadding::Explicit([1 << 62, 0]));
    assert_panics(
        || drop(conv2d(&x, &w, None, huge)),
        "Given padding=4611686018427387904 at dimension 0 , expected padding to be at most \
         4611686018427387903",
    );
    let small = x.narrow(2, 0, 2).narrow(3, 0, 2);
    assert_panics(
        || drop(conv2d(&small, &w.to(DType::F32), None, d)),
        "Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). Kernel size \
         can't be greater than actual input size",
    );
    assert_panics(
        || drop(conv2d(&x, &w, None, with(|o| o.dilation = [3, 3]))),
        "Calculated padded input size per channel: (5 x 5). Kernel size: (7 x 7). Kernel size \
         can't be greater than actual input size",
    );
    let flat = homura::ones_with([1, 3, 0, 5], DType::F64);
    assert_panics(
        || drop(conv2d(&flat, &w, None, with(|o| o.padding = 1.into()))),
        "Calculated padded input size per channel: (2 x 7). Kernel size: (3 x 3). Kernel size \
         can't be greater than actual input size",
    );
    // "same" pads first, so a kernel larger than the input is accepted.
    assert_eq!(conv2d(&small, &w, None, same).shape(), [2, 4, 2, 2]);
    assert_panics(
        || drop(conv2d(&x, &w.to(DType::F32), None, dilation0)),
        "dilation should be greater than zero, but got [0, 0]",
    );
    assert_panics(
        || {
            drop(conv2d(
                &x,
                &w,
                None,
                Conv2dOptions {
                    dilation: [0, 1],
                    ..same
                },
            ))
        },
        "dilation should be greater than zero, but got [0, 1]",
    );
    assert_panics(
        || {
            drop(conv2d(
                &x,
                &homura::ones_with([4, 3, 0, 3], DType::F64),
                None,
                d,
            ))
        },
        "kernel size should be greater than zero, but got kernel_height: 0 kernel_width: 3",
    );
    // One message for every dtype (PyTorch's F32 and half precision paths word it otherwise).
    assert_panics(
        || drop(conv2d(&x, &w.to(DType::F32), None, d)),
        "expected scalar type f64 but found f32",
    );
    assert_panics(
        || drop(conv2d(&x.to(DType::I64), &w, None, d)),
        "expected scalar type i64 but found f64",
    );
    assert_panics(
        || drop(conv2d(&x.to(DType::F32), &w, None, d)),
        "expected scalar type f32 but found f64",
    );
    let bools = homura::ones_with([1, 1, 3, 3], DType::Bool);
    let bool_kernel = homura::ones_with([1, 1, 2, 2], DType::Bool);
    assert_panics(
        || {
            drop(conv2d(
                &bools,
                &bool_kernel,
                None,
                with(|o| o.padding = 1.into()),
            ))
        },
        "conv2d is not implemented for bool tensors",
    );
    // Rejected on the meta device too, where PyTorch accepts dilation 0.
    let (xm, wm) = (x.to(Device::Meta), w.to(Device::Meta));
    assert_panics(
        || drop(conv2d(&xm, &wm, None, dilation0)),
        "dilation should be greater than zero, but got [0, 0]",
    );
}

#[test]
fn conv2d_dtypes_and_devices() {
    let ones = |shape: &[usize], dtype: DType| homura::ones_with(shape, dtype);
    for dtype in [DType::U8, DType::I8, DType::I16, DType::I32, DType::I64] {
        let y = conv2d(
            &ones(&[1, 1, 3, 3], dtype),
            &ones(&[1, 1, 2, 2], dtype),
            None,
            Conv2dOptions::default(),
        );
        assert_eq!(y.dtype(), dtype);
        assert_eq!(ints(&y), [4, 4, 4, 4], "{dtype}");
    }
    // Integer products and sums wrap.
    let y = conv2d(
        &homura::full_with([1, 1, 1, 2], 200, DType::U8),
        &ones(&[1, 1, 1, 2], DType::U8),
        None,
        Conv2dOptions::default(),
    );
    assert_eq!(ints(&y), [144]);
    let y = conv2d(
        &homura::full_with([1, 1, 1, 2], 1i64 << 62, DType::I64),
        &ones(&[1, 1, 1, 2], DType::I64),
        None,
        Conv2dOptions::default(),
    );
    assert_eq!(ints(&y), [i64::MIN]);
    let (x, w, b) = (
        homura::ones_with([2, 3, 5, 5], (DType::F64, Device::Meta)),
        homura::ones_with([4, 3, 3, 3], (DType::F64, Device::Meta)),
        homura::ones_with([4], (DType::F64, Device::Meta)),
    );
    let options = Conv2dOptions {
        stride: [2, 2],
        padding: 2.into(),
        dilation: [2, 2],
        groups: 1,
    };
    let y = conv2d(&x, &w, Some(&b), options);
    assert_eq!(
        (y.shape().to_vec(), y.device()),
        (vec![2, 4, 3, 3], Device::Meta)
    );
    x.requires_grad_(true);
    w.requires_grad_(true);
    let y = conv2d(&x, &w, Some(&b), options);
    let g = grads(
        &y.sum(),
        &[&x, &w],
        &homura::ones_with([], (DType::F64, Device::Meta)),
        false,
    );
    assert_eq!(g[0].as_ref().unwrap().shape(), [2, 3, 5, 5]);
    assert_eq!(g[1].as_ref().unwrap().device(), Device::Meta);
    // PyTorch accepts a meta input with CPU parameters, and a meta weight of another dtype;
    // homura requires one device and one dtype.
    let cpu_w = constant(&[4, 3, 3, 3], 1);
    assert_panics(
        || drop(conv2d(&x, &cpu_w, None, Conv2dOptions::default())),
        "Expected all tensors to be on the same device, but found at least two devices, meta and \
         cpu!",
    );
    assert_panics(
        || {
            drop(conv2d(
                &x,
                &w.to(DType::F32),
                None,
                Conv2dOptions::default(),
            ))
        },
        "expected scalar type f64 but found f32",
    );
    assert_panics(
        || {
            drop(conv2d(
                &x,
                &w,
                Some(&b.to(DType::F32)),
                Conv2dOptions::default(),
            ))
        },
        "Input type (f64) and bias type (f32) should be the same",
    );
}

#[test]
fn conv2d_degenerate_shapes() {
    let x = homura::zeros_with([0, 3, 5, 5], DType::F64);
    x.requires_grad_(true);
    let w = var(&[4, 3, 3, 3], 1);
    let y = conv2d(&x, &w, None, Conv2dOptions::default());
    assert_eq!(y.shape(), [0, 4, 3, 3]);
    let gw = grads(
        &y.sum(),
        &[&w],
        &homura::tensor_with(1.0, DType::F64),
        false,
    );
    assert_eq!(vals(gw[0].as_ref().unwrap()), [0.0; 108]);
    // Without input channels PyTorch returns an output without channels; homura panics.
    let message = "conv2d(): the input of size [1, 0, 5, 5] has no channels, so the output would \
                   have no channels either, unlike the weight of size [4, 0, 3, 3]; PyTorch \
                   returns such an output";
    let empty_w = homura::ones_with([4, 0, 3, 3], DType::F64);
    for x in [
        homura::ones_with([1, 0, 5, 5], DType::F64),
        homura::ones_with([0, 5, 5], DType::F64),
    ] {
        assert_panics(
            || drop(conv2d(&x, &empty_w, None, Conv2dOptions::default())),
            message,
        );
    }
    let x = constant(&[2, 3, 5, 5], 0);
    let w = constant(&[4, 3, 3, 3], 1);
    let one = w.narrow(2, 0, 1).narrow(3, 0, 1);
    assert_eq!(
        conv2d(&x, &one, None, opts([2, 2], 0)).shape(),
        [2, 4, 3, 3]
    );
    let corner = x.narrow(2, 0, 3).narrow(3, 0, 3);
    assert_eq!(
        conv2d(&corner, &w, None, opts([1, 1], 0)).shape(),
        [2, 4, 1, 1]
    );
    assert_eq!(
        conv2d(&x, &w, None, opts([1, 1], Conv2dPadding::Same)).shape(),
        [2, 4, 5, 5]
    );
    assert_eq!(
        conv2d(&x, &w, None, opts([1, 1], Conv2dPadding::Valid)).shape(),
        [2, 4, 3, 3]
    );
    let mixed = Conv2dOptions {
        stride: [2, 1],
        padding: [1, 0].into(),
        dilation: [1, 2],
        groups: 1,
    };
    assert_eq!(conv2d(&x, &w, None, mixed).shape(), [2, 4, 3, 1]);
    // "same" pads asymmetrically for an even effective kernel: the extra zeros go after.
    let ones = homura::ones_with([1, 1, 4, 4], DType::F64);
    let y = conv2d(
        &ones,
        &homura::ones_with([1, 1, 2, 2], DType::F64),
        None,
        opts([1, 1], Conv2dPadding::Same),
    );
    assert_eq!(
        vals(&y),
        [
            4., 4., 4., 2., 4., 4., 4., 2., 4., 4., 4., 2., 2., 2., 2., 1.
        ]
    );
    // Kernel height 2 with dilation 3: one row before, two after.
    let tall = Conv2dOptions {
        padding: Conv2dPadding::Same,
        dilation: [3, 1],
        ..Default::default()
    };
    let y = conv2d(
        &ones,
        &homura::ones_with([1, 1, 2, 3], DType::F64),
        None,
        tall,
    );
    assert_eq!(
        vals(&y),
        [
            2., 3., 3., 2., 4., 6., 6., 4., 2., 3., 3., 2., 2., 3., 3., 2.
        ]
    );
}

#[test]
fn conv2d_half_precision_rounds_once() {
    // Half precision is computed in F32 and rounded once, so a bias can cancel what the products
    // would overflow or round away in the dtype (review round 64, R77).
    let full = |v: f64, shape: &[usize]| homura::full_with(shape, v, DType::F16);
    let y = conv2d(
        &full(60000.0, &[1, 1, 1, 1]),
        &full(2.0, &[1, 1, 1, 1]),
        Some(&full(-60000.0, &[1])),
        Conv2dOptions::default(),
    );
    assert_eq!(vals(&y), [60000.0]);
    for (dtype, expected) in [
        (DType::F16, 0.0005002021789550781),
        (DType::BF16, 0.000_499_725_341_796_875),
    ] {
        let x = homura::ones_with([1, 2, 1, 1], dtype);
        x.requires_grad_(true);
        let w = f64s(&[1.0, 0.0005], &[1, 2, 1, 1]).to(dtype);
        w.requires_grad_(true);
        let b = f64s(&[-1.0], &[1]).to(dtype);
        b.requires_grad_(true);
        let y = conv2d(&x, &w, Some(&b), Conv2dOptions::default());
        assert_eq!(vals(&y), [expected], "{dtype}");
        // The gradients are F32 gradients rounded once: g * w, g * x and g.
        let g = f64s(&[3.0], &[1, 1, 1, 1]).to(dtype);
        let all = grads(&y, &[&x, &w, &b], &g, false);
        let rounded = |v: &[f64]| vals(&f64s(v, &[v.len()]).to(DType::F32).mul(3.0).to(dtype));
        let w_rounded = vals(&w);
        assert_eq!(
            vals(all[0].as_ref().unwrap()),
            rounded(&w_rounded),
            "{dtype}"
        );
        assert_eq!(vals(all[1].as_ref().unwrap()), [3.0, 3.0]);
        assert_eq!(vals(all[2].as_ref().unwrap()), [3.0]);
    }
}

#[test]
fn conv2d_gradients() {
    let x = var(&[2, 4, 5, 4], 0);
    let w = var(&[4, 2, 2, 3], 1);
    let b = var(&[4], 2);
    let cases = [
        Conv2dOptions::default(),
        Conv2dOptions {
            stride: [2, 1],
            padding: [1, 2].into(),
            dilation: [2, 1],
            groups: 2,
        },
        Conv2dOptions {
            padding: Conv2dPadding::Same,
            dilation: [1, 2],
            groups: 2,
            ..Default::default()
        },
    ];
    for (i, options) in cases.into_iter().enumerate() {
        let groups = options.groups;
        let f = move |xs: &[Tensor]| {
            // A transposed (non-contiguous) input, and a weight matching the groups.
            let x = xs[0].transpose(-2, -1);
            let w = xs[1].narrow(1, 0, 4 / groups);
            vec![conv2d(&x, &w, Some(&xs[2]), options)]
        };
        let w = if groups == 1 {
            var(&[4, 4, 3, 2], 1)
        } else {
            w.clone()
        };
        gradcheck(f, &[&x, &w, &b], GradcheckOptions::default())
            .unwrap_or_else(|e| panic!("case {i}: {e:?}"));
        gradgradcheck(f, &[&x, &w, &b], GradcheckOptions::default())
            .unwrap_or_else(|e| panic!("case {i}: {e:?}"));
    }
    // Unbatched.
    let f = |xs: &[Tensor]| vec![conv2d(&xs[0], &xs[1], None, opts([1, 1], 1))];
    let (x, w) = (var(&[2, 3, 3], 3), var(&[2, 2, 2, 2], 4));
    gradcheck(f, &[&x, &w], GradcheckOptions::default()).unwrap();
    gradgradcheck(f, &[&x, &w], GradcheckOptions::default()).unwrap();
}
