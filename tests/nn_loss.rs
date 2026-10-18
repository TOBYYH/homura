//! M2 step 3b: the losses of `nn::functional` and `Tensor::log_sigmoid`. Outputs and gradients
//! are compared with `tests/golden/m2_loss.safetensors` from `tools/golden/gen_m2_loss.py`; the
//! reference behavior encoded in the other tests is asserted by `tools/golden/check_m2_loss.py`,
//! in the same order.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden, assert_close_tol, tolerance};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::nn::functional::{
    CrossEntropyOptions, Reduction, binary_cross_entropy_with_logits, cross_entropy, logsigmoid,
    mse_loss, nll_loss,
};
use homura::{DType, Device, Tensor};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
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

/// A variable of `shape` with deterministic values of both signs, between 0.3 and 1.3 in
/// magnitude.
fn var(shape: &[usize], seed: usize) -> Tensor {
    let x = constant(shape, seed);
    x.requires_grad_(true);
    x
}

/// The values of [`var`] without requiring grad.
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

/// Positive class weights `0.5, 0.75, 1.0, ...` in f64.
fn weights(n: usize) -> Tensor {
    Tensor::from_vec((0..n).map(|i| 0.5 + 0.25 * i as f64).collect(), [n])
}

fn f64s(values: &[f64]) -> Tensor {
    Tensor::from_vec(values.to_vec(), [values.len()])
}

fn as_f64(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

fn reduction(code: i64) -> Reduction {
    match code {
        0 => Reduction::None,
        1 => Reduction::Mean,
        2 => Reduction::Sum,
        _ => panic!("unknown reduction {code}"),
    }
}

/// Evaluates the loss of a golden case on `inputs`, keyed like the stored tensors.
fn evaluate(function: &str, c: &Case, inputs: &dyn Fn(&str) -> Option<Tensor>) -> Tensor {
    let x = inputs("x").unwrap();
    let target = || inputs("target").unwrap();
    let r = || reduction(c.input("opt.reduction").item::<i64>());
    match function {
        "mse" => mse_loss(&x, &target(), r()),
        "nll" => nll_loss(
            &x,
            &target(),
            inputs("weight").as_ref(),
            c.input("opt.ignore_index").item::<i64>(),
            r(),
        ),
        "ce" => cross_entropy(
            &x,
            &target(),
            CrossEntropyOptions {
                weight: inputs("weight"),
                ignore_index: c.input("opt.ignore_index").item::<i64>(),
                reduction: r(),
                label_smoothing: c.input("opt.label_smoothing").item::<f64>(),
            },
        ),
        "bce" => binary_cross_entropy_with_logits(
            &x,
            &target(),
            inputs("weight").as_ref(),
            r(),
            inputs("pos_weight").as_ref(),
        ),
        "logsigmoid" => {
            assert_eq!(r(), Reduction::None);
            logsigmoid(&x)
        }
        _ => panic!("unknown function {function}"),
    }
}

/// Tolerances `(rtol, atol)` for comparing with the golden tensor `expected`. homura's sums
/// accumulate in f64 and round once, PyTorch's in the dtype; half precision allows two rounding
/// steps. Only the input gradient of cross_entropy (`cancels`) is a difference of similar terms,
/// `softmax(x) * Σ target * weight - target * weight` for probability targets, where both libraries
/// round the terms: there `atol` also allows two rounding steps of the tensor's largest element.
/// Against an f64 evaluation of the same half precision inputs, homura's largest error there is
/// smaller than PyTorch's.
fn golden_tolerance(expected: &Tensor, cancels: bool) -> (f64, f64) {
    let dtype = expected.dtype();
    match dtype {
        DType::F16 | DType::BF16 => {
            let (rtol, atol) = tolerance(dtype);
            let rtol = 2.0 * rtol;
            if !cancels {
                return (rtol, atol);
            }
            let largest = as_f64(expected)
                .into_iter()
                .filter(|v| v.is_finite())
                .fold(0.0, |m: f64, v| m.max(v.abs()));
            (rtol, rtol * largest + atol)
        }
        _ => tolerance(dtype),
    }
}

#[test]
fn losses_match_pytorch() {
    let golden = Golden::load("m2_loss.safetensors");
    let cases = golden.cases();
    assert_eq!(cases.len(), 2 * 73 + 2 * 10);
    let mut second_order = 0;
    for name in &cases {
        let c = Case::new(&golden, name);
        let function = name.split('_').next().unwrap();
        let stored = golden.names(name);
        let keys: Vec<&str> = ["x", "target", "weight", "pos_weight"]
            .into_iter()
            .filter(|k| stored.iter().any(|n| n == k))
            .collect();
        let inputs: Vec<(&str, Tensor)> = keys.iter().map(|&k| (k, c.input(k))).collect();
        let differentiable: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|k| stored.contains(&format!("grad_{k}")))
            .collect();
        for (k, t) in &inputs {
            t.requires_grad_(differentiable.contains(k));
        }
        let lookup = |k: &str| inputs.iter().find(|(n, _)| *n == k).map(|(_, t)| t.clone());
        let out = evaluate(function, &c, &lookup);
        let check = |key: &str, t: &Tensor| {
            c.check_with(key, t, |a, e| {
                let cancels = function == "ce" && key == "grad_x";
                let (rtol, atol) = golden_tolerance(e, cancels);
                assert_close_tol(a, e, rtol, atol, &format!("{name}/{key}"))
            })
        };
        check("out", &out);
        let second = stored.iter().any(|n| n.starts_with("gg_"));
        let g = c.input("g");
        g.requires_grad_(second);
        let variables: Vec<Tensor> = differentiable.iter().map(|k| lookup(k).unwrap()).collect();
        let variable_refs: Vec<&Tensor> = variables.iter().collect();
        let grads = autograd::grad(
            &[&out],
            &variable_refs,
            GradOptions {
                grad_outputs: vec![Some(g.clone())],
                create_graph: second,
                ..Default::default()
            },
        );
        let grads: Vec<Tensor> = grads.into_iter().map(Option::unwrap).collect();
        for (k, grad) in differentiable.iter().zip(&grads) {
            check(&format!("grad_{k}"), grad);
        }
        if second {
            second_order += 1;
            let total = differentiable
                .iter()
                .zip(&grads)
                .map(|(k, grad)| (grad * &c.input(&format!("gg_{k}"))).sum())
                .reduce(|a, b| a + b)
                .unwrap();
            let mut all = variable_refs.clone();
            all.push(&g);
            let second_grads = autograd::grad(
                &[&total],
                &all,
                GradOptions {
                    allow_unused: true,
                    ..Default::default()
                },
            );
            for (k, grad) in differentiable.iter().chain(&["g"]).zip(&second_grads) {
                let key = format!("grad2_{k}");
                match grad {
                    Some(grad) if stored.contains(&key) => check(&key, grad),
                    // The reference's None (nll_loss's gradient doesn't depend on the input):
                    // homura's None or zeros.
                    Some(grad) => assert!(
                        as_f64(grad).iter().all(|&v| v == 0.0),
                        "{name}/{key} {grad:?}"
                    ),
                    None => assert!(!stored.contains(&key), "{name}/{key}"),
                }
            }
        }
        c.assert_all_used();
    }
    assert_eq!(second_order, 12);
}

#[test]
fn losses_record_the_reference_nodes() {
    let acc = "AccumulateGrad";
    let x = var(&[3, 4], 0);
    let t = homura::tensor([1i64, 0, 3]);
    let w = weights(4);
    let p = constant(&[3, 4], 1);
    assert_eq!(
        chain(&mse_loss(&x, &p, Reduction::Mean)),
        [(0, "MseLossBackward0"), (1, acc)]
    );
    // PyTorch's broadcast_tensors expands the inputs whose shape changes.
    assert_eq!(
        chain(&mse_loss(
            &var(&[3, 1], 1),
            &var(&[1, 4], 2),
            Reduction::Mean
        )),
        [
            (0, "MseLossBackward0"),
            (1, "ExpandBackward0"),
            (2, acc),
            (1, "ExpandBackward0"),
            (2, acc)
        ]
    );
    assert_eq!(
        chain(&nll_loss(&x, &t, Some(&w), -100, Reduction::Mean)),
        [(0, "NllLossBackward0"), (1, acc)]
    );
    // PyTorch views a 3-D input as 4-D first (a ViewBackward0 below its NllLoss2DBackward0).
    let spatial_target = homura::zeros_with([2, 3], DType::I64);
    assert_eq!(
        chain(&nll_loss(
            &var(&[2, 4, 3], 0),
            &spatial_target,
            None,
            -100,
            Reduction::Mean
        )),
        [(0, "NllLoss2DBackward0"), (1, acc)]
    );
    let spatial_target = homura::zeros_with([2, 3, 2], DType::I64);
    assert_eq!(
        chain(&nll_loss(
            &var(&[2, 4, 3, 2], 0),
            &spatial_target,
            None,
            -100,
            Reduction::Mean
        )),
        [(0, "NllLoss2DBackward0"), (1, acc)]
    );
    assert_eq!(
        chain(&cross_entropy(&x, &t, CrossEntropyOptions::default())),
        [
            (0, "NllLossBackward0"),
            (1, "LogSoftmaxBackward0"),
            (2, acc)
        ]
    );
    // As the reference, with homura's names for products and quotients with a scalar
    // (MulBackward1 / DivBackward1 in PyTorch) and for masked_fill (MaskedFillBackward0).
    let smoothing = CrossEntropyOptions {
        weight: Some(w.clone()),
        label_smoothing: 0.1,
        ..Default::default()
    };
    assert_eq!(
        chain(&cross_entropy(&x, &t, smoothing)),
        [
            (0, "AddBackward0"),
            (1, "MulBackward0"),
            (2, "NllLossBackward0"),
            (3, "LogSoftmaxBackward0"),
            (4, acc),
            (1, "MulBackward0"),
            (2, "DivBackward0"),
            (3, "SumBackward0"),
            (4, "WhereBackward0"),
            (5, "NegBackward0"),
            (6, "SumBackward1"),
            (7, "MulBackward0"),
            (8, "LogSoftmaxBackward0"),
            (9, acc)
        ]
    );
    let (pt, pw) = (var(&[3, 4], 1), var(&[4], 2));
    let probabilities = CrossEntropyOptions {
        weight: Some(pw.clone()),
        ..Default::default()
    };
    assert_eq!(
        chain(&cross_entropy(&x, &pt, probabilities)),
        [
            (0, "DivBackward0"),
            (1, "NegBackward0"),
            (2, "SumBackward0"),
            (3, "MulBackward0"),
            (4, "MulBackward0"),
            (5, "LogSoftmaxBackward0"),
            (6, acc),
            (5, acc),
            (4, "ViewBackward0"),
            (5, acc)
        ]
    );
    // The mean of no positions: PyTorch's -(..).sum().fill_(NaN).
    let empty = CrossEntropyOptions::default();
    let p0 = homura::zeros_with([0, 4], DType::F64);
    assert_eq!(
        chain(&cross_entropy(&var(&[0, 4], 0), &p0, empty)),
        [
            (0, "NegBackward0"),
            (1, "FillBackward2"),
            (2, "SumBackward0"),
            (3, "MulBackward0"),
            (4, "LogSoftmaxBackward0"),
            (5, acc)
        ]
    );
    assert_eq!(
        chain(&binary_cross_entropy_with_logits(
            &x,
            &pt,
            Some(&w),
            Reduction::Mean,
            Some(&w)
        )),
        [
            (0, "BinaryCrossEntropyWithLogitsBackward0"),
            (1, acc),
            (1, acc)
        ]
    );
    assert_eq!(
        chain(&logsigmoid(&x)),
        [(0, "LogSigmoidBackward0"), (1, acc)]
    );
    assert!(
        cross_entropy(&x.detach(), &t, CrossEntropyOptions::default())
            .grad_fn()
            .is_none()
    );
    let _guard = homura::no_grad();
    let loss = binary_cross_entropy_with_logits(&x, &pt, None, Reduction::Mean, None);
    assert!(loss.grad_fn().is_none() && !loss.requires_grad());
}

#[test]
fn loss_gradients_pass_gradcheck() {
    let options = GradcheckOptions::default;
    let check = |what: &str, f: &dyn Fn(&[Tensor]) -> Vec<Tensor>, inputs: &[&Tensor]| {
        gradcheck(f, inputs, options()).unwrap_or_else(|e| panic!("{what}: {e}"));
        gradgradcheck(f, inputs, options()).unwrap_or_else(|e| panic!("{what}: {e}"));
    };
    let (x, y) = (var(&[3, 4], 0), var(&[3, 4], 1));
    for r in [Reduction::None, Reduction::Mean, Reduction::Sum] {
        check("mse", &|t| vec![mse_loss(&t[0], &t[1], r)], &[&x, &y]);
        check(
            "bce",
            &|t| {
                vec![binary_cross_entropy_with_logits(
                    &t[0], &t[1], None, r, None,
                )]
            },
            &[&x, &y],
        );
    }
    check(
        "mse broadcast",
        &|t| vec![mse_loss(&t[0], &t[1], Reduction::Mean)],
        &[&var(&[3, 1], 0), &var(&[1, 4], 1)],
    );
    let (w, pw) = (weights(4), weights(4).add(0.5));
    let column = weights(3).reshape([3, 1]);
    check(
        "bce weights",
        &|t| {
            let loss = binary_cross_entropy_with_logits(
                &t[0],
                &t[1],
                Some(&column),
                Reduction::Sum,
                Some(&pw),
            );
            vec![loss]
        },
        &[&x, &y],
    );
    let target = homura::tensor([1i64, -100, 3]);
    for r in [Reduction::None, Reduction::Mean] {
        check(
            "nll",
            &|t| vec![nll_loss(&t[0], &target, Some(&w), -100, r)],
            &[&x],
        );
        let options = || CrossEntropyOptions {
            weight: Some(w.clone()),
            reduction: r,
            label_smoothing: 0.2,
            ..Default::default()
        };
        check(
            "cross_entropy",
            &|t| vec![cross_entropy(&t[0], &target, options())],
            &[&x],
        );
        // The probabilities and their weight are variables too.
        check(
            "cross_entropy probabilities",
            &|t| {
                let options = CrossEntropyOptions {
                    weight: Some(t[2].clone()),
                    reduction: r,
                    label_smoothing: 0.1,
                    ..Default::default()
                };
                vec![cross_entropy(&t[0], &t[1], options)]
            },
            &[&x, &y, &var(&[4], 2)],
        );
    }
    let spatial = homura::tensor([[0i64, 3, 1], [2, -100, 0]]);
    check(
        "nll 3-D",
        &|t| vec![nll_loss(&t[0], &spatial, None, -100, Reduction::Mean)],
        &[&var(&[2, 4, 3], 0)],
    );
    let wide = constant(&[3, 4], 3).mul(4.0);
    wide.requires_grad_(true);
    check("log_sigmoid", &|t| vec![t[0].log_sigmoid()], &[&wide]);
}

#[test]
fn log_sigmoid_edges() {
    let v = f64s(&[40.0, 0.0, -800.0, 800.0]);
    v.requires_grad_(true);
    let out = v.log_sigmoid();
    assert_eq!(
        out.to_vec::<f64>(),
        [-4.248354255291589e-18, -2f64.ln(), -800.0, 0.0]
    );
    let g = autograd::grad(
        &[&out.sum()],
        &[&v],
        GradOptions {
            create_graph: true,
            ..Default::default()
        },
    );
    let g = g[0].clone().unwrap();
    let expected = [4.248354255291589e-18, 0.5, 1.0, 0.0];
    for (a, e) in as_f64(&g).into_iter().zip(expected) {
        assert!((a - e).abs() <= 1e-15 * e.abs(), "{a} {e}");
    }
    // PyTorch's log_sigmoid_double_backward, also at zero.
    let gg = autograd::grad(&[&g.sum()], &[&v], GradOptions::default());
    assert_eq!(as_f64(gg[0].as_ref().unwrap())[1], -0.25);
    // In half precision, σ(-x) is rounded to the dtype like PyTorch's buffer e^-|x|: subnormal
    // here, and the gradient matches PyTorch's exactly.
    let x16 = homura::tensor_with([15.0, 17.0], DType::F16);
    x16.requires_grad_(true);
    let g16 = homura::full_with([2], 1000.0, DType::F16);
    let grads = autograd::grad(
        &[&x16.log_sigmoid()],
        &[&x16],
        GradOptions {
            grad_outputs: vec![Some(g16)],
            ..Default::default()
        },
    );
    assert_eq!(
        as_f64(grads[0].as_ref().unwrap()),
        [0.0002980232238769531, 5.9604644775390625e-05]
    );
    // -4.25e-18 rounds to a negative zero in f16.
    let half = homura::tensor_with(40.0, DType::F16)
        .log_sigmoid()
        .item::<f32>();
    assert!(half == 0.0 && half.is_sign_negative());
    assert_eq!(logsigmoid(&v).to_vec::<f64>(), out.to_vec::<f64>());
}

#[test]
fn losses_on_meta_tensors() {
    let meta = |shape: &[usize], dtype: DType| {
        let t = homura::zeros_with(shape, (dtype, Device::Meta));
        if dtype.is_floating_point() {
            t.requires_grad_(true);
        }
        t
    };
    let shapes_of_grads = |out: &Tensor, inputs: &[&Tensor]| -> Vec<Vec<usize>> {
        autograd::grad(&[&out.sum()], inputs, GradOptions::default())
            .into_iter()
            .map(|g| {
                let g = g.unwrap();
                assert_eq!(g.device(), Device::Meta);
                g.shape().to_vec()
            })
            .collect()
    };
    let (x, y) = (meta(&[3, 1], DType::F32), meta(&[1, 4], DType::F32));
    let out = mse_loss(&x, &y, Reduction::None);
    assert_eq!(out.shape().to_vec(), [3, 4]);
    assert_eq!(shapes_of_grads(&out, &[&x, &y]), [vec![3, 1], vec![1, 4]]);
    assert_eq!(
        mse_loss(&x, &y, Reduction::Mean).shape().to_vec(),
        [0usize; 0]
    );
    // Meta targets have no values to check.
    let x = meta(&[3, 4], DType::F32);
    let t = meta(&[3], DType::I64);
    let w = meta(&[4], DType::F32);
    w.requires_grad_(false);
    let out = nll_loss(&x, &t, Some(&w), -100, Reduction::None);
    assert_eq!(out.shape().to_vec(), [3]);
    assert_eq!(shapes_of_grads(&out, &[&x]), [vec![3, 4]]);
    let x4 = meta(&[2, 3, 2, 2], DType::F32);
    let out = nll_loss(
        &x4,
        &meta(&[2, 2, 2], DType::U8),
        None,
        -100,
        Reduction::None,
    );
    assert_eq!(out.shape().to_vec(), [2, 2, 2]);
    assert_eq!(shapes_of_grads(&out, &[&x4]), [vec![2, 3, 2, 2]]);
    let smoothing = CrossEntropyOptions {
        weight: Some(w.clone()),
        label_smoothing: 0.1,
        ..Default::default()
    };
    let out = cross_entropy(&x, &t, smoothing);
    assert_eq!(out.shape().to_vec(), [0usize; 0]);
    assert_eq!(shapes_of_grads(&out, &[&x]), [vec![3, 4]]);
    let (p, pw) = (meta(&[3, 4], DType::F32), meta(&[4], DType::F32));
    let probabilities = CrossEntropyOptions {
        weight: Some(pw.clone()),
        reduction: Reduction::None,
        ..Default::default()
    };
    let out = cross_entropy(&x, &p, probabilities);
    assert_eq!(out.shape().to_vec(), [3]);
    assert_eq!(
        shapes_of_grads(&out, &[&x, &p, &pw]),
        [vec![3, 4], vec![3, 4], vec![4]]
    );
    let column = meta(&[3, 1], DType::F32);
    column.requires_grad_(false);
    let out = binary_cross_entropy_with_logits(&x, &p, Some(&column), Reduction::Mean, Some(&w));
    assert_eq!(out.shape().to_vec(), [0usize; 0]);
    assert_eq!(shapes_of_grads(&out, &[&x, &p]), [vec![3, 4], vec![3, 4]]);
    let out = x.log_sigmoid();
    assert_eq!(shapes_of_grads(&out, &[&x]), [vec![3, 4]]);
}

#[test]
fn loss_dtypes_follow_pytorch() {
    let (f32s, f64s) = (homura::ones([2]), homura::ones_with([2], DType::F64));
    let bce = |x: &Tensor, y: &Tensor, w: Option<&Tensor>| {
        binary_cross_entropy_with_logits(x, y, w, Reduction::Mean, w).dtype()
    };
    // `(1 - target).mul_(input)` keeps the target's dtype.
    assert_eq!(bce(&f32s, &f64s.mul(0.5), None), DType::F64);
    assert_eq!(bce(&f64s, &f32s.mul(0.5), None), DType::F32);
    assert_eq!(bce(&f32s, &f32s.mul(0.5), Some(&f64s)), DType::F32);
    let longs = homura::ones_with([2], DType::I64);
    assert_eq!(mse_loss(&f32s, &longs, Reduction::Mean).dtype(), DType::F32);
    assert_eq!(mse_loss(&f32s, &f64s, Reduction::Mean).dtype(), DType::F64);
    let probabilities =
        |x: &Tensor, p: &Tensor| cross_entropy(x, p, CrossEntropyOptions::default()).dtype();
    let (x32, x64) = (homura::ones([2, 3]), homura::ones_with([2, 3], DType::F64));
    assert_eq!(probabilities(&x64, &x32.div(3)), DType::F64);
    assert_eq!(probabilities(&x32, &x64.div(3)), DType::F64);
    for half in [DType::F16, DType::BF16] {
        let x = homura::ones_with([2, 3], half);
        let loss = cross_entropy(&x, &homura::tensor([0i64, 2]), Default::default());
        assert_eq!(loss.dtype(), half);
    }
    let t = homura::tensor([0i64, 1]);
    let w32 = homura::ones([3]);
    assert_eq!(
        panic_message(|| {
            nll_loss(&x64, &t, Some(&w32), -100, Reduction::Mean);
        }),
        "expected scalar type f64 but found f32"
    );
    let options = CrossEntropyOptions {
        weight: Some(homura::ones_with([3], DType::F64)),
        ..Default::default()
    };
    assert_eq!(
        panic_message(|| {
            cross_entropy(&x32, &t, options);
        }),
        "expected scalar type f32 but found f64"
    );
}

#[test]
fn nll_loss_and_cross_entropy_reject_invalid_arguments() {
    let x = constant(&[3, 4], 0);
    let t = homura::tensor([1i64, 0, 3]);
    let nll = |x: &Tensor, t: &Tensor, w: Option<&Tensor>| {
        panic_message(|| {
            nll_loss(x, t, w, -100, Reduction::Mean);
        })
    };
    assert_eq!(
        nll(&homura::tensor(1.0), &homura::tensor(0i64), None),
        "Expected 1 or more dimensions (got 0)"
    );
    assert_eq!(
        nll(&x, &homura::tensor([1i64, 0]), None),
        "Expected input batch_size (3) to match target batch_size (2)."
    );
    assert_eq!(
        nll(&x, &homura::tensor(2i64), None),
        "Expected input batch_size (3) to match target batch_size (0)."
    );
    assert_eq!(
        nll(&x.select(0, 0), &homura::tensor([2i64, 1]), None),
        "For 1D input, 1D target must have size 1, but got target size: 2"
    );
    assert_eq!(
        nll(&x, &t.reshape([3, 1]), None),
        "0D or 1D target tensor expected, multi-target not supported"
    );
    for (dtype, name) in [
        (DType::I32, "i32"),
        (DType::Bool, "bool"),
        (DType::F32, "f32"),
    ] {
        assert_eq!(
            nll(&x, &t.to(dtype), None),
            format!("expected target dtype to be i64 or u8, but got {name}")
        );
    }
    assert_eq!(
        nll(&x, &t, Some(&weights(3))),
        "weight tensor should be defined either for all 4 classes or no classes but got weight \
         tensor of shape: [3]"
    );
    assert_eq!(
        nll(&x, &t, Some(&weights(4).reshape([1, 4]))),
        "weight tensor should be defined either for all 4 classes or no classes but got weight \
         tensor of shape: [1, 4]"
    );
    let x3 = constant(&[2, 4, 3], 0);
    assert_eq!(
        nll(&x3, &homura::zeros_with([2, 2], DType::I64), None),
        "Expected target size [2, 3], got [2, 2]"
    );
    assert_eq!(
        nll(&x3, &homura::zeros_with([2], DType::I64), None),
        "Expected target size [2, 3], got [2]"
    );
    assert_eq!(
        nll(
            &x3,
            &homura::zeros_with([2, 3], DType::I64),
            Some(&weights(3))
        ),
        "weight tensor should be defined either for all or no classes"
    );
    let x4 = constant(&[2, 4, 2, 3], 0);
    assert_eq!(
        nll(&x4, &homura::zeros_with([2, 6], DType::I64), None),
        "only batches of spatial targets supported (3D tensors) but got targets of dimension: 2"
    );
    assert_eq!(
        nll(&x4, &homura::zeros_with([2, 2, 2], DType::I64), None),
        "size mismatch (got input: [2, 4, 2, 3] , target: [2, 2, 2]"
    );
    assert_eq!(
        nll(&x4, &homura::zeros_with([2, 2, 3], DType::I32), None),
        "expected target dtype to be i64 or u8, but got i32"
    );
    // The 2-D implementation of more dimensions only counts the weight's elements.
    let spatial = homura::zeros_with([2, 2, 3], DType::I64);
    let loss = nll_loss(
        &x4,
        &spatial,
        Some(&weights(4).reshape([1, 4])),
        -100,
        Reduction::Mean,
    );
    assert!(loss.item::<f64>().is_finite());
    for (target, bad) in [([1i64, 4, 0], 4), ([1, -1, 0], -1)] {
        assert_eq!(
            nll(&x, &homura::tensor(target), None),
            format!("Target {bad} is out of bounds.")
        );
    }
    // homura reports the first bad target; PyTorch checks targets in parallel without
    // reduction.
    assert_eq!(
        panic_message(|| {
            nll_loss(
                &x,
                &homura::tensor([1i64, -2, 5]),
                None,
                -100,
                Reduction::None,
            );
        }),
        "Target -2 is out of bounds."
    );
    assert_eq!(
        nll(&x, &homura::tensor([1u8, 255, 0]), None),
        "Target 255 is out of bounds."
    );
    assert_eq!(
        nll(&x4, &homura::full_with([2, 2, 3], 9, DType::I64), None),
        "Target 9 is out of bounds."
    );
    nll_loss(
        &x,
        &homura::tensor([1u8, 255, 0]),
        None,
        255,
        Reduction::Mean,
    );
    assert_eq!(
        nll(&homura::ones_with([3, 4], DType::I64), &t, None),
        "nll_loss is not implemented for i64 tensors"
    );
    let ce = |x: &Tensor, t: &Tensor, options: CrossEntropyOptions| {
        panic_message(|| {
            cross_entropy(x, t, options);
        })
    };
    assert_eq!(
        ce(
            &x,
            &homura::ones_with([3, 4], DType::I64),
            Default::default()
        ),
        "Expected floating point type for target with class probabilities, got i64"
    );
    assert_eq!(
        ce(
            &homura::tensor(1.0),
            &homura::tensor(0i64),
            Default::default()
        ),
        "Expected floating point type for target with class probabilities, got i64"
    );
    let probabilities = x.softmax(1);
    let ignore = |ignore_index| CrossEntropyOptions {
        ignore_index,
        ..Default::default()
    };
    assert_eq!(
        ce(&x, &probabilities, ignore(0)),
        "ignore_index is not supported for floating point target"
    );
    cross_entropy(&x, &probabilities, ignore(-1));
    let weighted = |label_smoothing| CrossEntropyOptions {
        weight: Some(weights(3)),
        label_smoothing,
        ..Default::default()
    };
    assert_eq!(
        ce(&x, &probabilities, weighted(0.0)),
        "cross_entropy: weight tensor should be defined either for all 4 classes or no classes \
         but got weight tensor of shape: [3]"
    );
    assert_eq!(
        ce(&x, &t, weighted(0.1)),
        "weight tensor should be defined either for all 4 classes or no classes but got weight \
         tensor of shape: [3]"
    );
    assert_eq!(
        ce(
            &homura::ones_with([3, 4], DType::I64),
            &t,
            Default::default()
        ),
        "log_softmax is not implemented for i64 tensors"
    );
}

#[test]
fn bce_and_mse_reject_invalid_arguments() {
    let x = constant(&[3, 4], 0);
    let y = constant(&[3, 4], 1).abs();
    let bce = |x: &Tensor, y: &Tensor, w: Option<&Tensor>, pw: Option<&Tensor>| {
        panic_message(|| {
            binary_cross_entropy_with_logits(x, y, w, Reduction::Mean, pw);
        })
    };
    assert_eq!(
        bce(&x, &y.select(0, 0), None, None),
        "Target size ([4]) must be the same as input size ([3, 4])"
    );
    let big = constant(&[2, 3, 4], 2);
    assert_eq!(
        bce(&x, &y, Some(&big), None),
        "output with shape [3, 4] doesn't match the broadcast shape [2, 3, 4]"
    );
    assert_eq!(
        bce(&x, &y, None, Some(&big)),
        "output with shape [3, 4] doesn't match the broadcast shape [2, 3, 4]"
    );
    assert_eq!(
        bce(&x, &y, None, Some(&weights(5))),
        "The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 1"
    );
    let ones = homura::ones([3]);
    assert_eq!(
        bce(&ones, &homura::ones_with([3], DType::I64), None, None),
        "result type f32 can't be cast to the desired output type i64"
    );
    // `1 - target` rejects a bool target. PyTorch's C++ `1 - target` fills a bool tensor with 1
    // first, so its message is the one for two bool tensors.
    assert_eq!(
        bce(&ones, &homura::ones_with([3], DType::Bool), None, None),
        "Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to \
         invert a mask, use `logical_not()` instead."
    );
    let longs = homura::ones_with([3], DType::I64);
    assert_eq!(
        bce(&longs, &longs, None, None),
        "LogSigmoid is not implemented for i64 tensors"
    );
    assert_eq!(
        panic_message(|| {
            logsigmoid(&longs);
        }),
        "LogSigmoid is not implemented for i64 tensors"
    );
    assert_eq!(
        panic_message(|| {
            mse_loss(&ones, &homura::ones([4]), Reduction::Mean);
        }),
        "The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 0"
    );
    for (dtype, name) in [(DType::I64, "i64"), (DType::Bool, "bool")] {
        let t = homura::ones_with([3], dtype);
        assert_eq!(
            panic_message(|| {
                mse_loss(&t, &t, Reduction::Mean);
            }),
            format!("mse_loss is not implemented for {name} tensors")
        );
    }
}

#[test]
fn constant_weights_reject_requires_grad() {
    let x = var(&[3, 4], 0);
    let t = homura::tensor([1i64, 0, 3]);
    let y = constant(&[3, 4], 1).abs();
    let w = weights(4);
    w.requires_grad_(true);
    let message = |function: &str, argument: &str| {
        format!(
            "The function '{function}' is not differentiable with respect to argument \
             '{argument}'. This input cannot have requires_grad True."
        )
    };
    let nll = |x: &Tensor, t: &Tensor| {
        panic_message(|| {
            nll_loss(x, t, Some(&w), -100, Reduction::Mean);
        })
    };
    assert_eq!(nll(&x, &t), message("nll_loss_forward", "weight"));
    // Before the shape checks of the reference's nll_loss_forward.
    assert_eq!(
        nll(&x, &t.reshape([3, 1])),
        message("nll_loss_forward", "weight")
    );
    let spatial = homura::zeros_with([2, 3], DType::I64);
    assert_eq!(
        nll(&var(&[2, 4, 3], 0), &spatial),
        message("nll_loss2d_forward", "weight")
    );
    for label_smoothing in [0.0, 0.1] {
        let options = CrossEntropyOptions {
            weight: Some(w.clone()),
            label_smoothing,
            ..Default::default()
        };
        assert_eq!(
            panic_message(|| {
                cross_entropy(&x, &t, options);
            }),
            message("nll_loss_forward", "weight")
        );
    }
    let bce = |weight: Option<&Tensor>, pos_weight: Option<&Tensor>| {
        panic_message(|| {
            binary_cross_entropy_with_logits(&x, &y, weight, Reduction::Mean, pos_weight);
        })
    };
    let function = "binary_cross_entropy_with_logits";
    assert_eq!(bce(Some(&w), None), message(function, "weight"));
    assert_eq!(bce(None, Some(&w)), message(function, "pos_weight"));
    {
        let _guard = homura::no_grad();
        nll_loss(&x, &t, Some(&w), -100, Reduction::Mean);
        binary_cross_entropy_with_logits(&x, &y, Some(&w), Reduction::Mean, Some(&w));
    }
    // The weight of probability targets is a variable.
    let options = CrossEntropyOptions {
        weight: Some(w.clone()),
        ..Default::default()
    };
    let loss = cross_entropy(&x, &x.detach().softmax(1), options);
    let grads = autograd::grad(&[&loss], &[&w], GradOptions::default());
    assert_eq!(grads[0].as_ref().unwrap().shape().to_vec(), [4]);
}

#[test]
fn label_smoothing_outside_the_unit_interval() {
    let x = constant(&[3, 4], 0);
    let t = homura::tensor([1i64, 0, 3]);
    let probabilities = x.softmax(1);
    let ce = |t: &Tensor, label_smoothing| {
        let options = CrossEntropyOptions {
            label_smoothing,
            ..Default::default()
        };
        cross_entropy(&x, t, options)
    };
    // Negative and NaN amounts mean no smoothing, as in PyTorch.
    for smoothing in [-0.5, f64::NAN] {
        for target in [&t, &probabilities] {
            assert_eq!(
                ce(target, smoothing).item::<f64>(),
                ce(target, 0.0).item::<f64>()
            );
        }
    }
    for target in [&t, &probabilities] {
        assert_eq!(
            panic_message(|| {
                ce(target, 1.5);
            }),
            "label_smoothing must be between 0.0 and 1.0. Got: 1.5"
        );
    }
}

#[test]
fn single_samples_with_a_one_element_target() {
    // A 1-D input takes a 0-D target or a [1] one, and the loss is 0-D also without reduction.
    // PyTorch's backward rejects the [1] target ("size mismatch (got input: [4], target: [1])");
    // homura computes the same gradient as for the 0-D target.
    let x = var(&[4], 0);
    let w = weights(4);
    for reduction in [Reduction::None, Reduction::Mean, Reduction::Sum] {
        let losses: Vec<Tensor> = [homura::tensor([2i64]), homura::tensor(2i64)]
            .iter()
            .map(|t| nll_loss(&x, t, Some(&w), -100, reduction))
            .collect();
        assert_eq!(losses[0].shape().to_vec(), [0usize; 0]);
        assert_eq!(losses[0].item::<f64>(), losses[1].item::<f64>());
        let grads: Vec<Vec<f64>> = losses
            .iter()
            .map(|l| {
                as_f64(
                    autograd::grad(&[l], &[&x], GradOptions::default())[0]
                        .as_ref()
                        .unwrap(),
                )
            })
            .collect();
        assert_eq!(grads[0], grads[1]);
        // `-w[2] * g`, divided by the total weight `w[2]` for the mean.
        let expected = match reduction {
            Reduction::Mean => -1.0,
            _ => -w.to_vec::<f64>()[2],
        };
        assert_eq!(grads[0], [0.0, 0.0, expected, 0.0]);
    }
}

#[test]
fn saved_tensors_follow_the_version_rules() {
    let modified = |f: &dyn Fn() -> (Tensor, Box<dyn Fn()>)| {
        let (loss, modify) = f();
        modify();
        let message = panic_message(|| loss.backward());
        assert!(
            message.contains("modified by an inplace operation"),
            "{message}"
        );
    };
    let x = var(&[3, 4], 0);
    modified(&|| {
        let t = homura::tensor([1i64, 0, 3]);
        let loss = nll_loss(&x, &t, None, -100, Reduction::Mean);
        (
            loss,
            Box::new(move || {
                t.add_(0);
            }),
        )
    });
    modified(&|| {
        let w = weights(4);
        let loss = nll_loss(
            &x,
            &homura::tensor([1i64, 0, 3]),
            Some(&w),
            -100,
            Reduction::Mean,
        );
        (
            loss,
            Box::new(move || {
                w.mul_(1);
            }),
        )
    });
    modified(&|| {
        let y = constant(&[3, 4], 1);
        let loss = mse_loss(&x, &y, Reduction::Mean);
        (
            loss,
            Box::new(move || {
                y.add_(1);
            }),
        )
    });
    modified(&|| {
        let w = weights(4);
        let y = constant(&[3, 4], 1).abs();
        let loss = binary_cross_entropy_with_logits(&x, &y, Some(&w), Reduction::Mean, None);
        (
            loss,
            Box::new(move || {
                w.add_(1);
            }),
        )
    });
    let y = constant(&[3, 4], 1).abs();
    let loss = binary_cross_entropy_with_logits(&x, &y, None, Reduction::Mean, None);
    loss.backward();
    let message = panic_message(|| loss.backward());
    assert!(
        message.contains("backward through the graph a second time"),
        "{message}"
    );
}

#[test]
fn reduction_is_a_hyperparameter_of_modules() {
    use homura::nn::Module;

    #[derive(Module)]
    struct Loss {
        reduction: Reduction,
        smoothing: f64,
    }

    let loss = Loss {
        reduction: Reduction::Sum,
        smoothing: 0.1,
    };
    assert!(loss.named_parameters().is_empty() && loss.state_dict().is_empty());
    assert_eq!((loss.reduction, loss.smoothing), (Reduction::Sum, 0.1));
    assert_eq!(Reduction::default(), Reduction::Mean);
}

/// The first derivative of `x.log_sigmoid()` for upstream gradient `g`, with `create_graph`.
fn log_sigmoid_grad(x: &Tensor, g: &Tensor, create_graph: bool) -> Tensor {
    let grads = autograd::grad(
        &[&x.log_sigmoid()],
        &[x],
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            create_graph,
            ..Default::default()
        },
    );
    grads[0].clone().unwrap()
}

fn assert_relative(actual: f64, expected: f64, what: &str) {
    assert!(
        expected != 0.0 && actual != 0.0,
        "{what}: {actual} {expected}"
    );
    assert!(
        (actual - expected).abs() <= 1e-6 * expected.abs(),
        "{what}: {actual} {expected}"
    );
}

#[test]
fn log_sigmoid_gradients_follow_the_reference_buffer() {
    // R59: PyTorch's buffer e^-|x| stays representable where σ(-x) = 1 / (1 + e^x) underflows,
    // and a large upstream gradient brings the product back into the normal range.
    for (dtype, x, g, expected) in [
        (DType::F32, 90.0, 1e38, 0.08194008469581604),
        (DType::F64, 710.0, 1e308, 0.44762862256751296),
    ] {
        let x = homura::tensor_with([x], dtype);
        x.requires_grad_(true);
        let g = homura::tensor_with([g], dtype);
        for create_graph in [false, true] {
            let grad = log_sigmoid_grad(&x, &g, create_graph);
            assert_relative(as_f64(&grad)[0], expected, &format!("{dtype:?}"));
        }
    }
    // The second derivative is PyTorch's log_sigmoid_double_backward, `r * g * (σ(x) - 1) *
    // σ(x)`, not the derivative of a saturated σ(-x).
    let x = f64s(&[-40.0, 0.0]);
    x.requires_grad_(true);
    let g = f64s(&[1e18, 1.0]);
    let grad = log_sigmoid_grad(&x, &g, true);
    assert_eq!(as_f64(&grad), [1e18, 0.5]);
    assert_eq!(
        chain(&grad)[0],
        (0, "LogSigmoidBackwardBackward0"),
        "the first derivative is a node of its own"
    );
    let second = autograd::grad(&[&grad.sum()], &[&x], GradOptions::default());
    let second = as_f64(second[0].as_ref().unwrap());
    assert_relative(second[0], -4.248354255291589, "second derivative at -40");
    assert_eq!(second[1], -0.25);
    // The derivative with respect to the upstream gradient is the first derivative again.
    g.requires_grad_(true);
    let grad = log_sigmoid_grad(&x, &g, true);
    let dg = autograd::grad(&[&grad.sum()], &[&g], GradOptions::default());
    let first = as_f64(&log_sigmoid_grad(&x, &homura::ones_like(&g), false));
    assert_eq!(as_f64(dg[0].as_ref().unwrap()), first);
    // ... from the buffer as well: nonzero where σ(-x) underflows.
    let x = f64s(&[710.0]);
    x.requires_grad_(true);
    let g = f64s(&[1.0]);
    g.requires_grad_(true);
    let grad = log_sigmoid_grad(&x, &g, true);
    let dg = autograd::grad(
        &[&grad],
        &[&g],
        GradOptions {
            grad_outputs: vec![Some(f64s(&[1e308]))],
            ..Default::default()
        },
    );
    let dg = as_f64(dg[0].as_ref().unwrap())[0];
    assert_relative(
        dg,
        0.44762862256751296,
        "derivative with respect to g at 710",
    );
    // In half precision, the buffer is rounded to the dtype and the rest computed in f32.
    let x = homura::tensor_with([1.0], DType::BF16);
    x.requires_grad_(true);
    let grad = log_sigmoid_grad(&x, &homura::tensor_with([1000.0], DType::BF16), false);
    assert_eq!(as_f64(&grad), [268.0]);
    // A strided input gives the values of a contiguous one.
    let wide = constant(&[4, 6], 2).mul(30.0);
    let strided = wide.index((.., (1..6).step_by(2)));
    assert!(!strided.is_contiguous());
    strided.requires_grad_(true);
    let dense = strided.contiguous().detach();
    dense.requires_grad_(true);
    let g = constant(&[4, 3], 5).mul(1e300);
    assert_eq!(
        as_f64(&log_sigmoid_grad(&strided, &g, true)),
        as_f64(&log_sigmoid_grad(&dense, &g, true))
    );
    // Meta: shapes of both derivatives.
    let x = homura::zeros_with([2, 3], (DType::F32, Device::Meta));
    x.requires_grad_(true);
    let grad = log_sigmoid_grad(
        &x,
        &homura::ones_with([2, 3], (DType::F32, Device::Meta)),
        true,
    );
    let second = autograd::grad(&[&grad.sum()], &[&x], GradOptions::default());
    assert_eq!(second[0].as_ref().unwrap().shape().to_vec(), [2, 3]);
    // The saved input is version checked and released after the backward pass.
    let x = constant(&[3], 1).mul(2.0);
    x.requires_grad_(true);
    let y = x.mul(1.0);
    let out = y.log_sigmoid().sum();
    y.add_(1.0);
    let message = panic_message(|| out.backward());
    assert!(
        message.contains("modified by an inplace operation"),
        "{message}"
    );
    let out = x.log_sigmoid().sum();
    out.backward();
    let message = panic_message(|| out.backward());
    assert!(
        message.contains("backward through the graph a second time"),
        "{message}"
    );
}

/// The derivatives of `sum(dx)` for the first derivative `dx` of `nll_loss` with respect to `x`,
/// computed with an upstream gradient of ones that requires grad: with respect to `x` and to
/// that upstream gradient, both required to exist (`allow_unused` is false).
fn nll_second_derivatives(
    x: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> (Tensor, Tensor) {
    let out = nll_loss(x, target, weight, -100, reduction);
    let g = homura::ones_with(out.shape(), (out.dtype(), out.device()));
    g.requires_grad_(true);
    let dx = autograd::grad(
        &[&out],
        &[x],
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            create_graph: true,
            ..Default::default()
        },
    );
    let dx = dx[0].clone().unwrap();
    let second = autograd::grad(&[&dx.sum()], &[x, &g], GradOptions::default());
    (second[0].clone().unwrap(), second[1].clone().unwrap())
}

/// The derivative with respect to the upstream gradient when no target counts: PyTorch's
/// `nll_loss` of it, zeros of the loss's shape, a zero sum, or a NaN mean (a zero total weight).
fn assert_empty_upstream_derivative(dg: &Tensor, out: &Tensor, reduction: Reduction) {
    assert_eq!(dg.shape(), out.shape());
    let values = as_f64(dg);
    match reduction {
        Reduction::Mean => assert!(values[0].is_nan(), "{values:?}"),
        _ => assert!(values.iter().all(|&v| v == 0.0), "{values:?}"),
    }
}

#[test]
fn nll_loss_without_classes() {
    // R60: with zero classes, every target must be ignored; nothing is gathered or scattered.
    let cases: [(&[usize], &[usize]); 3] =
        [(&[0], &[]), (&[2, 0], &[2]), (&[2, 0, 2, 2], &[2, 2, 2])];
    let mut groups = 0;
    for (shape, target_shape) in cases {
        let target = homura::full_with(target_shape, -100, DType::I64);
        for reduction in [Reduction::None, Reduction::Mean, Reduction::Sum] {
            for weight in [None, Some(homura::zeros_with([0], DType::F64))] {
                let x = homura::zeros_with(shape, DType::F64);
                x.requires_grad_(true);
                let out = nll_loss(&x, &target, weight.as_ref(), -100, reduction);
                let values = as_f64(&out);
                match reduction {
                    Reduction::None => {
                        let expected = if shape.len() == 1 {
                            vec![]
                        } else {
                            target_shape.to_vec()
                        };
                        assert_eq!(out.shape().to_vec(), expected);
                        assert!(values.iter().all(|&v| v == 0.0));
                    }
                    Reduction::Mean => assert!(values[0].is_nan()),
                    Reduction::Sum => assert_eq!(values, [0.0]),
                }
                let g = homura::full_with(out.shape(), f64::INFINITY, DType::F64);
                let grads = autograd::grad(
                    &[&out],
                    &[&x],
                    GradOptions {
                        grad_outputs: vec![Some(g)],
                        create_graph: true,
                        ..Default::default()
                    },
                );
                assert_eq!(grads[0].as_ref().unwrap().shape().to_vec(), shape);
                // R63: the gradient depends on the input with a zero derivative and on the
                // upstream gradient like PyTorch's, also without classes.
                let (dx, dg) = nll_second_derivatives(&x, &target, weight.as_ref(), reduction);
                assert_eq!(dx.shape().to_vec(), shape);
                assert_empty_upstream_derivative(&dg, &out, reduction);
                groups += 1;
            }
        }
    }
    assert_eq!(groups, 18);
    // The same derivatives with classes, every target ignored or no positions, and on Meta.
    let controls: [(&[usize], Tensor); 2] = [
        (&[2, 3], homura::full_with([2], -100, DType::I64)),
        (&[0, 3], homura::zeros_with([0], DType::I64)),
    ];
    for (shape, target) in controls {
        for reduction in [Reduction::None, Reduction::Mean, Reduction::Sum] {
            let x = homura::zeros_with(shape, DType::F64);
            x.requires_grad_(true);
            let out = nll_loss(&x, &target, None, -100, reduction);
            let (dx, dg) = nll_second_derivatives(&x, &target, None, reduction);
            assert_eq!(dx.shape().to_vec(), shape);
            assert_empty_upstream_derivative(&dg, &out, reduction);
        }
    }
    let meta = homura::zeros_with([2, 0], (DType::F32, Device::Meta));
    meta.requires_grad_(true);
    let target = homura::zeros_with([2], (DType::I64, Device::Meta));
    let (dx, dg) = nll_second_derivatives(&meta, &target, None, Reduction::None);
    assert_eq!(
        (dx.shape().to_vec(), dg.shape().to_vec()),
        (vec![2, 0], vec![2])
    );
    assert_eq!(dg.device(), Device::Meta);
    // Through cross_entropy, also with label smoothing, which divides by zero classes (NaN).
    let x = homura::zeros_with([2, 0], DType::F64);
    x.requires_grad_(true);
    let target = homura::full_with([2], -100, DType::I64);
    let plain = cross_entropy(
        &x,
        &target,
        CrossEntropyOptions {
            reduction: Reduction::Sum,
            ..Default::default()
        },
    );
    assert_eq!(plain.item::<f64>(), 0.0);
    let smoothing = CrossEntropyOptions {
        weight: Some(homura::zeros_with([0], DType::F64)),
        label_smoothing: 0.1,
        reduction: Reduction::None,
        ..Default::default()
    };
    let smoothed = cross_entropy(&x, &target, smoothing);
    assert!(as_f64(&smoothed).iter().all(|v| v.is_nan()));
    let mean = CrossEntropyOptions {
        weight: Some(homura::zeros_with([0], DType::F64)),
        label_smoothing: 0.1,
        ..Default::default()
    };
    let loss = cross_entropy(&x, &target, mean);
    assert!(loss.item::<f64>().is_nan());
    let grads = autograd::grad(&[&loss], &[&x], GradOptions::default());
    assert_eq!(grads[0].as_ref().unwrap().shape().to_vec(), [2, 0]);
    // A target that isn't ignored is still out of range, and Meta inputs only have shapes.
    assert_eq!(
        panic_message(|| {
            nll_loss(
                &x,
                &homura::tensor([0i64, -100]),
                None,
                -100,
                Reduction::Mean,
            );
        }),
        "Target 0 is out of bounds."
    );
    // The check runs in no-grad mode, which the panic leaves.
    assert!(homura::is_grad_enabled());
    let meta = homura::zeros_with([2, 0], (DType::F32, Device::Meta));
    let target = homura::zeros_with([2], (DType::I64, Device::Meta));
    assert_eq!(
        nll_loss(&meta, &target, None, -100, Reduction::None)
            .shape()
            .to_vec(),
        [2]
    );
}

#[test]
fn spatial_weights_are_version_checked_in_any_layout() {
    // R61: a weight of more dimensions is saved as the caller's tensor, not as a flattened copy.
    let run = |weight: &Tensor, modify: &dyn Fn(), retain: bool| {
        let x = homura::ones_with([1, 4, 1, 1], DType::F64);
        x.requires_grad_(true);
        let target = homura::zeros_with([1, 1, 1], DType::I64);
        let loss = nll_loss(&x, &target, Some(weight), -100, Reduction::Mean);
        if retain {
            autograd::backward(
                &[&loss],
                &[],
                autograd::BackwardOptions {
                    retain_graph: Some(true),
                    ..Default::default()
                },
            );
        }
        modify();
        catch_unwind(AssertUnwindSafe(|| {
            as_f64(
                &autograd::grad(&[&loss], &[&x], GradOptions::default())[0]
                    .clone()
                    .unwrap(),
            )
        }))
        .map_err(|p| p.downcast_ref::<String>().cloned().unwrap_or_default())
    };
    let base = || Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    for transposed in [false, true] {
        let make = || {
            if transposed {
                base().transpose(0, 1)
            } else {
                base()
            }
        };
        let w = make();
        assert_eq!(w.is_contiguous(), !transposed);
        assert_eq!(run(&w, &|| {}, false).unwrap(), [-1.0, 0.0, 0.0, 0.0]);
        for retain in [false, true] {
            let w = make();
            let error = run(
                &w,
                &|| {
                    w.add_(1.0);
                },
                retain,
            )
            .unwrap_err();
            assert!(
                error.contains("modified by an inplace operation"),
                "{error}"
            );
        }
        // Modifying the base of a transposed weight, or another view of it.
        let b = base();
        let w = if transposed {
            b.transpose(0, 1)
        } else {
            b.reshape([2, 2])
        };
        let error = run(
            &w,
            &|| {
                b.select(0, 1).mul_(2.0);
            },
            false,
        )
        .unwrap_err();
        assert!(
            error.contains("modified by an inplace operation"),
            "{error}"
        );
    }
}

#[test]
fn spatial_byte_targets_extend_pytorch() {
    // PyTorch's spatial NLL accepts U8 targets in its checks but its CPU kernel rejects them
    // ("expected scalar type Long but found Byte"); homura computes them like I64 targets.
    let cases: [(&[usize], &[usize]); 2] = [(&[2, 3, 2], &[2, 2]), (&[2, 3, 2, 2], &[2, 2, 2])];
    for (shape, target_shape) in cases {
        let n: usize = target_shape.iter().product();
        let classes: Vec<i64> = (0..n as i64).map(|i| i % 3).collect();
        let long = Tensor::from_vec(classes, target_shape);
        let byte = long.to(DType::U8);
        for reduction in [Reduction::None, Reduction::Mean, Reduction::Sum] {
            let x = var(shape, 1);
            let w = weights(3);
            let results: Vec<(Vec<f64>, Vec<f64>)> = [&long, &byte]
                .iter()
                .map(|t| {
                    let out = nll_loss(&x, t, Some(&w), -100, reduction);
                    let g = constant(&out.shape(), 2);
                    let grads = autograd::grad(
                        &[&out],
                        &[&x],
                        GradOptions {
                            grad_outputs: vec![Some(g)],
                            ..Default::default()
                        },
                    );
                    (as_f64(&out), as_f64(grads[0].as_ref().unwrap()))
                })
                .collect();
            assert_eq!(results[0], results[1], "{shape:?} {reduction:?}");
        }
    }
}

#[test]
fn bool_operands_of_subtraction_are_rejected() {
    // R62: PyTorch rejects a bool operand before type promotion, also a single one.
    let single = "Subtraction, the `-` operator, with a bool tensor is not supported. If you are \
                  trying to invert a mask, use `logical_not()` instead.";
    let two = "Subtraction, the `-` operator, with two bool tensors is not supported. Use \
               `not_equal()` (logical xor) instead.";
    let mask = homura::tensor([true, false]);
    let longs = homura::tensor([2i64, 3]);
    let floats = homura::tensor([2.0f32, 3.0]);
    let sub = |f: &dyn Fn() -> Tensor| {
        panic_message(|| {
            f();
        })
    };
    assert_eq!(sub(&|| mask.sub(1)), single);
    assert_eq!(sub(&|| 1i64 - &mask), single);
    assert_eq!(sub(&|| longs.sub(&mask)), single);
    assert_eq!(sub(&|| floats.sub(&mask)), single);
    assert_eq!(sub(&|| &mask - 1.5), single);
    assert_eq!(sub(&|| longs.sub(true)), single);
    assert_eq!(sub(&|| mask.sub(&mask)), two);
    assert_eq!(sub(&|| mask.sub(true)), two);
    assert_eq!(sub(&|| homura::zeros_with([0], DType::Bool).sub(1)), single);
    // PyTorch's meta kernels skip the check; a program that fails on the CPU fails on Meta too.
    let meta = homura::zeros_with([2], (DType::Bool, Device::Meta));
    assert_eq!(sub(&|| meta.sub(1)), single);
    // In place, nothing changes: data, version and history. A graph that saved `y` before still
    // runs, so its version didn't move.
    let target = longs.deep_clone();
    assert_eq!(
        panic_message(|| {
            target.sub_(&mask);
        }),
        single
    );
    assert_eq!(target.to_vec::<i64>(), [2, 3]);
    let x = var(&[2], 0);
    let y = x.mul(2.0);
    let saved = y.mul(&y).sum();
    assert_eq!(
        panic_message(|| {
            y.sub_(&mask);
        }),
        single
    );
    assert_eq!(chain(&y)[0], (0, "MulBackward0"));
    saved.backward();
    assert_eq!(as_f64(&x.grad().unwrap()), as_f64(&x.mul(8.0)));
    // Integer and floating point subtraction are unchanged.
    assert_eq!(longs.sub(1).to_vec::<i64>(), [1, 2]);
    assert_eq!((1i64 - &longs).to_vec::<i64>(), [-1, -2]);
    assert_eq!(floats.sub(&longs).to_vec::<f32>(), [0.0, 0.0]);
}
