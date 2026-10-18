//! M2 step 3a: `nn::init`, `nn::functional::linear` with `nn::Linear`, and the activation layers.
//! Outputs and gradients are compared with `tests/golden/m2_layers.safetensors` from
//! `tools/golden/gen_m2_layers.py`; the reference behavior encoded in the other tests is
//! asserted by `tools/golden/check_m2_layers.py`, in the same order.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden, assert_close_tol, tolerance};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::nn::functional::{self as f, linear};
use homura::nn::init::{self, FanMode, Nonlinearity};
use homura::nn::{
    Forward, Gelu, Linear, Module, Relu, Sequential, Sigmoid, Silu, Softmax, StateDict, Tanh,
};
use homura::{DType, Device, GeluApproximate, Shape, Tensor};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// A variable of `shape` with deterministic values of both signs, between 0.3 and 1.3 in
/// magnitude.
fn var(shape: &[usize], seed: usize) -> Tensor {
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
    let x = Tensor::from_vec(values, shape);
    x.requires_grad_(true);
    x
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

fn as_f64(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

#[test]
fn init_gains_and_fans_follow_pytorch() {
    use Nonlinearity::*;
    for nl in [
        Linear,
        Conv1d,
        Conv2d,
        Conv3d,
        ConvTranspose1d,
        ConvTranspose2d,
        ConvTranspose3d,
        Sigmoid,
    ] {
        assert_eq!(init::calculate_gain(nl, None), 1.0, "{nl:?}");
    }
    assert_eq!(init::calculate_gain(Tanh, None), 5.0 / 3.0);
    assert_eq!(init::calculate_gain(Relu, None), 2f64.sqrt());
    assert_eq!(init::calculate_gain(Relu, Some(3.0)), 2f64.sqrt());
    assert_eq!(
        init::calculate_gain(LeakyRelu, None),
        (2.0 / (1.0 + 0.01f64.powi(2))).sqrt()
    );
    assert_eq!(
        init::calculate_gain(LeakyRelu, Some(0.2)),
        (2.0f64 / 1.04).sqrt()
    );
    assert_eq!(
        init::calculate_gain(LeakyRelu, Some(5f64.sqrt())),
        0.5773502691896257
    );
    assert_eq!(init::calculate_gain(Selu, None), 0.75);
    let fans = |shape: &[usize]| init::calculate_fan_in_and_fan_out(&homura::zeros(shape));
    assert_eq!(fans(&[5, 3, 2, 2]), (12, 20));
    assert_eq!(fans(&[5, 3]), (3, 5));
    assert_eq!(fans(&[5, 0, 2]), (0, 10));
    assert_eq!(
        panic_message(|| {
            let _ = fans(&[3]);
        }),
        "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
    );
}

/// Checks the samples of a symmetric distribution: inside `[-bound, bound]` if given, and with
/// the mean and standard deviation expected, within loose margins for the sample size.
fn check_samples(t: &Tensor, bound: Option<f64>, std: f64, what: &str) {
    let v = as_f64(t);
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    assert!(mean.abs() < 0.1 * std, "{what}: mean {mean}, std {std}");
    assert!(
        (sd / std - 1.0).abs() < 0.08,
        "{what}: std {sd}, expected {std}"
    );
    if let Some(bound) = bound {
        let max = v.iter().fold(0f64, |m, x| m.max(x.abs()));
        assert!(max <= bound, "{what}: {max} > {bound}");
        assert!(max > 0.95 * bound, "{what}: {max} far below {bound}");
    }
}

#[test]
fn init_samples_lie_within_the_bounds() {
    homura::manual_seed(7);
    let (fan_in, fan_out) = (50.0f64, 64.0f64);
    let w = homura::zeros_with([64, 50], DType::F64);
    let returned = init::kaiming_uniform_(&w, 0.0, FanMode::FanOut, Nonlinearity::Relu);
    assert!(returned.is_same(&w));
    let bound = 3f64.sqrt() * (2f64.sqrt() / fan_out.sqrt());
    check_samples(&w, Some(bound), bound / 3f64.sqrt(), "kaiming_uniform_");
    init::kaiming_normal_(&w, 0.0, FanMode::FanIn, Nonlinearity::Tanh);
    check_samples(&w, None, 5.0 / 3.0 / fan_in.sqrt(), "kaiming_normal_");
    init::kaiming_normal_(&w, 0.2, FanMode::FanOut, Nonlinearity::LeakyRelu);
    let gain = (2.0f64 / 1.04).sqrt();
    check_samples(&w, None, gain / fan_out.sqrt(), "kaiming_normal_ leaky");
    init::xavier_uniform_(&w, 2.0);
    let a = 2.0 * (6.0 / (fan_in + fan_out)).sqrt();
    check_samples(&w, Some(a), a / 3f64.sqrt(), "xavier_uniform_");
    init::xavier_normal_(&w, 0.5);
    check_samples(
        &w,
        None,
        0.5 * (2.0 / (fan_in + fan_out)).sqrt(),
        "xavier_normal_",
    );
    init::kaiming_uniform_(&w, 5f64.sqrt(), FanMode::FanIn, Nonlinearity::LeakyRelu);
    let bound = 1.0 / fan_in.sqrt();
    check_samples(
        &w,
        Some(bound),
        bound / 3f64.sqrt(),
        "kaiming_uniform_ a=√5",
    );
    init::uniform_(&w, -3.0, -1.0);
    let v = as_f64(&w);
    assert!(v.iter().all(|x| (-3.0..-1.0).contains(x)));
    init::normal_(&w, 2.0, 0.5);
    check_samples(&(&w - 2.0), None, 0.5, "normal_");

    // Other dtypes keep their dtype; half precision rounds the bounds by up to half a step.
    for (dtype, step) in [
        (DType::F32, 0.0),
        (DType::F16, 2f64.powi(-11)),
        (DType::BF16, 2f64.powi(-8)),
    ] {
        let w = homura::zeros_with([64, 50], dtype);
        init::kaiming_uniform_(&w, 0.0, FanMode::FanIn, Nonlinearity::Relu);
        assert_eq!(w.dtype(), dtype);
        let bound = 3f64.sqrt() * (2f64.sqrt() / fan_in.sqrt());
        check_samples(
            &w,
            Some(bound * (1.0 + step)),
            bound / 3f64.sqrt(),
            &format!("{dtype}"),
        );
        init::xavier_normal_(&w, 1.0);
        assert_eq!(w.dtype(), dtype);
    }

    // `nn::Linear` samples from U(-1/√in, 1/√in); the weights' bound rounds just below 0.5
    // (0.49999999999999994) for 4 inputs, like the reference's.
    let layer = Linear::new(4, 3);
    for p in [&*layer.weight, layer.bias.as_deref().unwrap()] {
        assert!(as_f64(p).iter().all(|x| x.abs() <= 0.5));
    }
    let big = Linear::new(400, 300);
    check_samples(&big.weight, Some(0.05), 0.05 / 3f64.sqrt(), "Linear weight");
    let bias = as_f64(big.bias.as_ref().unwrap());
    let max = bias.iter().fold(0f64, |m, x| m.max(x.abs()));
    assert!(max <= 0.05 && max > 0.9 * 0.05, "Linear bias: {max}");
}

#[test]
fn init_edge_cases() {
    // Kaiming returns tensors without elements before computing the fans; Xavier doesn't.
    type Init = fn(&Tensor) -> &Tensor;
    let kaiming: [Init; 2] = [
        |t| init::kaiming_uniform_(t, 0.0, FanMode::FanIn, Nonlinearity::LeakyRelu),
        |t| init::kaiming_normal_(t, 0.0, FanMode::FanIn, Nonlinearity::LeakyRelu),
    ];
    let xavier: [Init; 2] = [
        |t| init::xavier_uniform_(t, 1.0),
        |t| init::xavier_normal_(t, 1.0),
    ];
    let fewer = "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions";
    for f in kaiming {
        assert_eq!(f(&homura::zeros([0])).shape(), [0]);
        assert_eq!(f(&homura::zeros([3, 0])).shape(), [3, 0]);
        assert_eq!(
            panic_message(|| {
                let _ = f(&homura::zeros([3]));
            }),
            fewer
        );
        assert_eq!(
            panic_message(|| {
                let _ = f(&homura::zeros([]));
            }),
            fewer
        );
    }
    for f in xavier {
        assert_eq!(
            panic_message(|| {
                let _ = f(&homura::zeros([0]));
            }),
            fewer
        );
        assert_eq!(f(&homura::zeros([3, 0])).shape(), [3, 0]);
        assert_eq!(
            panic_message(|| {
                let _ = f(&homura::zeros([0, 0]));
            }),
            "division by zero: fan_in + fan_out is 0 for a tensor of shape [0, 0]"
        );
    }
    // Sampling needs a floating point tensor; filling takes any dtype, with `fill_`'s rules.
    let ints = || homura::zeros_with([3, 4], DType::I64);
    let float_only = "random floating point sampling is not implemented for i64 tensors";
    for f in kaiming.into_iter().chain(xavier) {
        assert_eq!(
            panic_message(|| {
                let _ = f(&ints());
            }),
            float_only
        );
    }
    assert_eq!(
        panic_message(|| {
            let _ = init::uniform_(&ints(), 0.0, 1.0);
        }),
        float_only
    );
    assert_eq!(
        panic_message(|| {
            let _ = init::normal_(&ints(), 0.0, 1.0);
        }),
        float_only
    );
    let t = homura::zeros_with([3], DType::I64);
    assert_eq!(init::constant_(&t, -2.7).to_vec::<i64>(), [-2, -2, -2]);
    let t = homura::zeros_with([3], DType::Bool);
    assert_eq!(init::constant_(&t, 0.5).to_vec::<bool>(), [true; 3]);
    let t = homura::zeros_with([2], DType::I64);
    assert_eq!(init::ones_(&t).to_vec::<i64>(), [1, 1]);
    let t = homura::ones_with([2], DType::Bool);
    assert_eq!(init::zeros_(&t).to_vec::<bool>(), [false, false]);
    assert_eq!(
        panic_message(|| {
            let _ = init::uniform_(&homura::zeros([3]), 2.0, 1.0);
        }),
        "uniform_ expects to return a [from, to) range, but found from=2 > to=1"
    );
    assert_eq!(
        panic_message(|| {
            let _ = init::normal_(&homura::zeros([3]), 0.0, -1.0);
        }),
        "normal expects std >= 0.0, but found std -1"
    );

    // Leaves that require grad are filled in grad mode too, without history; grad mode stays on.
    let leaf = homura::zeros([3, 4]);
    leaf.requires_grad_(true);
    assert!(homura::is_grad_enabled());
    init::kaiming_uniform_(&leaf, 0.0, FanMode::FanIn, Nonlinearity::LeakyRelu);
    init::constant_(&leaf, 1.5);
    assert!(leaf.is_leaf() && leaf.grad_fn().is_none() && leaf.requires_grad());
    assert!(homura::is_grad_enabled());
    assert_eq!(leaf.to_vec::<f32>(), [1.5; 12]);

    // On the meta device, filling does nothing.
    let meta = homura::zeros_with([3, 4], Device::Meta);
    let all: [Init; 8] = [
        kaiming[0],
        kaiming[1],
        xavier[0],
        xavier[1],
        |t| init::uniform_(t, 0.0, 1.0),
        |t| init::normal_(t, 0.0, 1.0),
        init::ones_,
        init::zeros_,
    ];
    for f in all {
        assert_eq!(f(&meta).device(), Device::Meta);
    }
}

#[test]
fn linear_layers_initialize_like_pytorch() {
    let layer = Linear::new(4, 3);
    assert_eq!(
        layer.state_dict().keys().collect::<Vec<_>>(),
        ["weight", "bias"]
    );
    assert_eq!(layer.weight.shape(), [3, 4]);
    assert_eq!(layer.bias.as_ref().unwrap().shape(), [3]);
    assert!(layer.weight.requires_grad() && layer.weight.is_leaf());
    assert_eq!((layer.in_features, layer.out_features), (4, 3));
    assert_eq!(
        format!("{layer:?}"),
        "Linear { in_features: 4, out_features: 3, bias: true, .. }"
    );
    let layer = layer.bias(false);
    assert!(layer.bias.is_none());
    assert_eq!(layer.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
    assert_eq!(
        format!("{layer:?}"),
        "Linear { in_features: 4, out_features: 3, bias: false, .. }"
    );
    // A bias added again follows the weights' dtype and is initialized.
    layer.to(DType::F64);
    let layer = layer.bias(true);
    let bias = layer.bias.as_ref().unwrap();
    assert_eq!((bias.shape(), bias.dtype()), ([3].into(), DType::F64));
    assert!(bias.requires_grad() && bias.is_leaf());
    assert!(as_f64(bias).iter().all(|x| x.abs() <= 0.5) && as_f64(bias).iter().any(|&x| x != 0.0));
    // `bias(true)` keeps an existing bias.
    let same = bias.clone();
    let layer = layer.bias(true);
    assert!(layer.bias.as_ref().unwrap().is_same(&same));

    // `reset_parameters` draws new values in place.
    let before = layer.weight.deep_clone();
    layer.reset_parameters();
    assert!(as_f64(&layer.weight) != as_f64(&before));

    // Zero inputs give a zero bias and outputs; zero outputs work too.
    let layer = Linear::new(0, 3);
    assert_eq!(layer.weight.shape(), [3, 0]);
    assert_eq!(layer.bias.as_ref().unwrap().to_vec::<f32>(), [0.0; 3]);
    assert_eq!(
        layer.forward(&homura::zeros([2, 0])).to_vec::<f32>(),
        [0.0; 6]
    );
    let layer = Linear::new(3, 0);
    assert_eq!(layer.weight.shape(), [0, 3]);
    assert_eq!(layer.bias.as_ref().unwrap().shape(), [0]);
    assert_eq!(layer.forward(&homura::ones([2, 3])).shape(), [2, 0]);

    // Layers follow the default device.
    let layer = {
        let _meta = homura::device_scope(Device::Meta);
        Linear::new(3, 2)
    };
    assert_eq!(layer.weight.device(), Device::Meta);
    assert_eq!(layer.bias.as_ref().unwrap().device(), Device::Meta);
    let y = layer.forward(&homura::zeros_with([5, 3], Device::Meta));
    assert_eq!((y.shape(), y.device()), ([5, 2].into(), Device::Meta));

    // The layer is a `Forward` and a member of other modules.
    let model = Sequential::new()
        .append(Linear::new(4, 8))
        .append(Relu)
        .append(Linear::new(8, 2));
    assert_eq!(
        model.state_dict().keys().collect::<Vec<_>>(),
        ["0.weight", "0.bias", "2.weight", "2.bias"]
    );
    let layer: &dyn Forward = &Linear::new(4, 2);
    assert_eq!(layer.forward(&homura::ones([3, 4])).shape(), [3, 2]);
}

/// The input of a golden `linear` case: `x`, or `x_t` transposed in its last two dimensions.
fn golden_input(c: &Case) -> Tensor {
    if c.golden.names(c.name).iter().any(|n| n == "x_t") {
        c.input("x_t").transpose(-1, -2)
    } else {
        c.input("x")
    }
}

/// The bias of a golden `linear` case, if it has one: `b`, or column 0 of `b_s`, a 1-D bias
/// with stride 2.
fn golden_bias(c: &Case) -> Option<Tensor> {
    let names = c.golden.names(c.name);
    if names.iter().any(|n| n == "b_s") {
        let b = c.input("b_s").select(-1, 0);
        assert!(!b.is_contiguous());
        Some(b)
    } else {
        names.iter().any(|n| n == "b").then(|| c.input("b"))
    }
}

/// Tolerances `(rtol, atol)` for the golden comparisons: homura's matrix products accumulate in
/// f64 and round once, PyTorch's CPU kernels accumulate f32 and half precision products in f32.
fn golden_tolerance(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::F32 => (1e-5, 1e-5),
        DType::F16 | DType::BF16 => {
            let (rtol, atol) = tolerance(dtype);
            (2.0 * rtol, atol)
        }
        _ => tolerance(dtype),
    }
}

#[test]
fn linear_matches_pytorch() {
    let golden = Golden::load("m2_layers.safetensors");
    let cases: Vec<String> = golden
        .cases()
        .into_iter()
        .filter(|c| c.starts_with("linear_"))
        .collect();
    assert_eq!(cases.len(), 2 * 23 + 2 * 11 + 3);
    let mut second_order = 0;
    for name in &cases {
        let c = Case::new(&golden, name);
        let x = golden_input(&c);
        let w = c.input("w");
        let b = golden_bias(&c);
        let (rtol, atol) = golden_tolerance(w.dtype());
        let check = |key: &str, t: &Tensor| {
            c.check_with(key, t, |a, e| {
                assert_close_tol(a, e, rtol, atol, &format!("{name}/{key}"))
            })
        };
        if !w.is_floating_point() {
            check("out", &linear(&x, &w, b.as_ref()));
            c.assert_all_used();
            continue;
        }
        let mut inputs = vec![&x, &w];
        inputs.extend(b.as_ref());
        for t in &inputs {
            t.requires_grad_(true);
        }
        let out = linear(&x, &w, b.as_ref());
        check("out", &out);
        let keys = &["x", "w", "b"][..inputs.len()];
        let second = golden.names(name).iter().any(|n| n == "gg_x");
        let g = c.input("g");
        g.requires_grad_(second);
        let grads = autograd::grad(
            &[&out],
            &inputs,
            GradOptions {
                grad_outputs: vec![Some(g.clone())],
                create_graph: second,
                ..Default::default()
            },
        );
        let grads: Vec<Tensor> = grads.into_iter().map(Option::unwrap).collect();
        for (key, grad) in keys.iter().zip(&grads) {
            check(&format!("grad_{key}"), grad);
        }
        if second {
            // The second derivatives of sum(grad_k * gg_k), also with respect to `g`; those that
            // are None in the reference (the bias's) are not stored.
            second_order += 1;
            let total = keys
                .iter()
                .zip(&grads)
                .map(|(key, grad)| (grad * &c.input(&format!("gg_{key}"))).sum())
                .reduce(|a, b| a + b)
                .unwrap();
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
            let stored = golden.names(name);
            for (key, grad) in keys.iter().chain(&["g"]).zip(&second_grads) {
                assert_eq!(second_grads.len(), keys.len() + 1);
                let key = format!("grad2_{key}");
                match grad {
                    Some(grad) if stored.contains(&key) => check(&key, grad),
                    // Only the bias's second derivative is None, homura's or zero.
                    Some(grad) => {
                        assert_eq!(key, "grad2_b", "{name}");
                        assert!(as_f64(grad).iter().all(|&v| v == 0.0), "{name}/{key}");
                    }
                    None => assert!(!stored.contains(&key), "{name}/{key}"),
                }
            }
        }
        c.assert_all_used();
    }
    assert_eq!(second_order, 7);
}

#[test]
fn linear_records_the_reference_nodes() {
    // As the reference, with TransposeBackward0 for its TBackward0.
    let w = var(&[3, 4], 1);
    let b = var(&[3], 2);
    assert_eq!(
        chain(&linear(&var(&[2, 4], 0), &w, Some(&b))),
        [
            (0, "AddmmBackward0"),
            (1, "AccumulateGrad"),
            (1, "AccumulateGrad"),
            (1, "TransposeBackward0"),
            (2, "AccumulateGrad"),
        ]
    );
    for shape in [&[4][..], &[5, 2, 4], &[2, 3, 2, 4]] {
        assert_eq!(
            chain(&linear(&var(shape, 0), &w, Some(&b))),
            [
                (0, "ViewBackward0"),
                (1, "AddmmBackward0"),
                (2, "AccumulateGrad"),
                (2, "ViewBackward0"),
                (3, "AccumulateGrad"),
                (2, "TransposeBackward0"),
                (3, "AccumulateGrad"),
            ],
            "{shape:?}"
        );
    }
    assert_eq!(
        chain(&linear(&var(&[2, 4], 0), &w, None)),
        [
            (0, "MmBackward0"),
            (1, "AccumulateGrad"),
            (1, "TransposeBackward0"),
            (2, "AccumulateGrad"),
        ]
    );
    // A transposed 3-D input, and a 2-D bias with a 4-D input: matmul, then the bias added in
    // place. homura records the matmul as one node where the reference has views around `mm`.
    let xt = var(&[2, 4, 5], 0).detach().transpose(1, 2);
    xt.requires_grad_(true);
    assert_eq!(
        chain(&linear(&xt, &w, Some(&b))),
        [
            (0, "AddBackward0"),
            (1, "MatmulBackward0"),
            (2, "AccumulateGrad"),
            (2, "TransposeBackward0"),
            (3, "AccumulateGrad"),
            (1, "AccumulateGrad"),
        ]
    );
    // Only a contiguous bias with a single dimension that isn't 1 takes the flattened addmm
    // (review round 47, R56); a 2-D input always takes the addmm.
    let strided = || {
        let b = var(&[3, 2], 2).detach().select(1, 0);
        b.requires_grad_(true);
        b
    };
    let top = |x: &[usize], b: &Tensor| chain(&linear(&var(x, 0), &w, Some(b)))[0];
    for (x, b, node) in [
        (&[2, 3, 2, 4][..], var(&[2, 3], 3), "AddBackward0"),
        (&[2, 5, 4], var(&[5, 3], 3), "AddBackward0"),
        (&[2, 5, 4], var(&[2, 5, 3], 3), "AddBackward0"),
        (&[2, 5, 4], var(&[], 3), "AddBackward0"),
        (&[2, 5, 4], strided(), "AddBackward0"),
        (&[4], strided(), "AddBackward0"),
        (&[2, 3, 2, 4], var(&[1, 3], 3), "ViewBackward0"),
        (&[2, 4], strided(), "AddmmBackward0"),
        (&[2, 4], var(&[2, 3], 3), "AddmmBackward0"),
    ] {
        assert_eq!(top(x, &b), (0, node), "{x:?} {:?}", b.shape());
    }
    // As in the reference, a bias that squeezes to 1-D still has to broadcast in the addmm: a
    // `[1, 1, 3]` bias doesn't fit its 2-D result.
    assert_eq!(
        panic_message(|| {
            let _ = top(&[2, 5, 4], &var(&[1, 1, 3], 3));
        }),
        "cannot broadcast shape [1, 1, 3] to [10, 3]"
    );
    // Without anything that requires grad, nothing is recorded.
    let out = linear(
        &homura::ones([2, 4]),
        &homura::ones([3, 4]),
        Some(&homura::ones([3])),
    );
    assert!(!out.requires_grad() && out.grad_fn().is_none());
}

/// The shapes of a `linear` call: the input's, whether it is transposed, the weight's and the
/// bias's, and whether the bias is a 1-D view with stride 2.
type LinearShapes = (
    &'static [usize],
    bool,
    &'static [usize],
    Option<&'static [usize]>,
    bool,
);

#[test]
fn linear_gradients_pass_gradcheck() {
    let options = GradcheckOptions::default;
    let cases: [LinearShapes; 14] = [
        (&[2, 4], false, &[3, 4], Some(&[3]), false),
        (&[2, 4], true, &[3, 4], Some(&[3]), false),
        (&[4], false, &[3, 4], Some(&[3]), false),
        (&[2, 3, 4], false, &[3, 4], Some(&[3]), false),
        (&[2, 3, 4], true, &[3, 4], Some(&[3]), false),
        (&[2, 1, 2, 4], false, &[3, 4], Some(&[2, 1, 1, 3]), false),
        (&[2, 3, 2, 4], false, &[3, 4], Some(&[1, 3]), false),
        (&[2, 3, 4], false, &[3, 4], Some(&[3, 3]), false),
        (&[2, 3, 4], false, &[3, 4], Some(&[]), false),
        (&[2, 3, 4], false, &[3, 4], Some(&[3]), true),
        (&[4], false, &[3, 4], Some(&[3]), true),
        (&[2, 4], false, &[3, 4], None, false),
        (&[2, 3, 4], false, &[3, 4], None, false),
        (&[2, 4], false, &[4], None, false),
    ];
    for (i, (x_shape, transposed, w_shape, b_shape, strided)) in cases.into_iter().enumerate() {
        let x = if transposed {
            let mut s = x_shape.to_vec();
            let n = s.len();
            s.swap(n - 1, n - 2);
            let t = var(&s, 0).detach().transpose(-1, -2);
            t.requires_grad_(true);
            t
        } else {
            var(x_shape, 0)
        };
        let w = var(w_shape, 1);
        let mut inputs = vec![&x, &w];
        let b = b_shape.map(|s| {
            if strided {
                let mut s = s.to_vec();
                s.push(2);
                let b = var(&s, 2).detach().select(-1, 0);
                b.requires_grad_(true);
                b
            } else {
                var(s, 2)
            }
        });
        inputs.extend(b.as_ref());
        let f = |t: &[Tensor]| vec![linear(&t[0], &t[1], t.get(2))];
        gradcheck(f, &inputs, options()).unwrap_or_else(|e| panic!("case {i}: {e}"));
        gradgradcheck(f, &inputs, options()).unwrap_or_else(|e| panic!("case {i}: {e}"));
    }
}

#[test]
fn linear_on_meta_and_empty_inputs() {
    let meta = |shape: &[usize]| {
        let t = homura::zeros_with(shape, (DType::F64, Device::Meta));
        t.requires_grad_(true);
        t
    };
    let w = meta(&[3, 4]);
    let b = meta(&[3]);
    for (x_shape, bias, out_shape) in [
        (&[2, 4][..], true, &[2, 3][..]),
        (&[4], true, &[3]),
        (&[2, 5, 4], true, &[2, 5, 3]),
        (&[2, 5, 4], false, &[2, 5, 3]),
        (&[0, 4], true, &[0, 3]),
    ] {
        let x = meta(x_shape);
        let y = linear(&x, &w, bias.then_some(&b));
        assert_eq!(
            (y.shape().to_vec(), y.device()),
            (out_shape.to_vec(), Device::Meta)
        );
        let grads = autograd::grad(
            &[&y],
            &[&x, &w],
            GradOptions {
                grad_outputs: vec![Some(homura::zeros_with(
                    out_shape,
                    (DType::F64, Device::Meta),
                ))],
                ..Default::default()
            },
        );
        assert_eq!(grads[0].as_ref().unwrap().shape().to_vec(), x_shape);
        assert_eq!(grads[1].as_ref().unwrap().shape(), [3, 4]);
    }
    let xt = homura::zeros_with([2, 4, 5], Device::Meta).transpose(1, 2);
    assert_eq!(
        linear(&xt, &homura::zeros_with([3, 4], Device::Meta), None).shape(),
        [2, 5, 3]
    );

    // The bias layouts of each branch (R56) on meta: the shapes of the output and the gradients.
    let strided = meta(&[3, 2]).detach().select(1, 0);
    strided.requires_grad_(true);
    for (x_shape, b) in [
        (&[2, 5, 4][..], meta(&[5, 3])),
        (&[2, 5, 4], meta(&[2, 5, 3])),
        (&[2, 5, 4], meta(&[])),
        (&[2, 3, 2, 4], meta(&[1, 3])),
        (&[2, 5, 4], strided.clone()),
        (&[4], strided),
    ] {
        let x = meta(x_shape);
        let y = linear(&x, &w, Some(&b));
        let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
        out_shape.push(3);
        assert_eq!(
            (y.shape().to_vec(), y.device()),
            (out_shape.clone(), Device::Meta)
        );
        let grads = autograd::grad(
            &[&y],
            &[&x, &w, &b],
            GradOptions {
                grad_outputs: vec![Some(homura::zeros_with(
                    &out_shape[..],
                    (DType::F64, Device::Meta),
                ))],
                ..Default::default()
            },
        );
        let shapes: Vec<_> = grads.iter().map(|g| g.as_ref().unwrap().shape()).collect();
        assert_eq!(shapes, [x.shape(), w.shape(), b.shape()], "{x_shape:?}");
    }

    // Empty batches give empty outputs, zero weight gradients and zero bias gradients.
    for (x_shape, b_shape) in [
        (&[0, 4][..], &[3][..]),
        (&[2, 0, 4], &[3]),
        (&[0, 5, 4], &[5, 3]),
    ] {
        let (x, w, b) = (var(x_shape, 0), var(&[3, 4], 1), var(b_shape, 2));
        let y = linear(&x, &w, Some(&b));
        assert_eq!(y.num_elements(), 0);
        y.sum().backward();
        assert_eq!(x.grad().unwrap().shape().to_vec(), x_shape);
        assert_eq!(w.grad().unwrap().to_vec::<f64>(), [0.0; 12]);
        let gb = b.grad().unwrap();
        assert_eq!(
            (gb.shape().to_vec(), as_f64(&gb).iter().all(|&v| v == 0.0)),
            (b_shape.to_vec(), true)
        );
    }
}

#[test]
fn linear_rejects_invalid_arguments() {
    assert_eq!(
        panic_message(|| drop(linear(&homura::zeros([]), &homura::zeros([3, 1]), None))),
        "both arguments to linear need to be at least 1D, but they are 0D and 2D"
    );
    assert_eq!(
        panic_message(|| drop(linear(
            &homura::zeros([2, 4]),
            &homura::zeros([1, 3, 4]),
            None
        ))),
        "linear expects a weight with at most 2 dimensions, but weight is 3D"
    );
    assert_eq!(
        panic_message(|| drop(linear(&homura::zeros([2, 5]), &homura::zeros([3, 4]), None))),
        "mat1 and mat2 shapes cannot be multiplied (2x5 and 4x3)"
    );
    assert_eq!(
        panic_message(|| drop(linear(
            &homura::zeros([2, 4]),
            &homura::zeros_with([3, 4], DType::F64),
            None
        ))),
        "expected m1 and m2 to have the same dtype, but got: f32 != f64"
    );
    assert_eq!(
        panic_message(|| drop(linear(
            &homura::zeros([2, 4]),
            &homura::zeros([3, 4]),
            Some(&homura::zeros_with([3], DType::F64))
        ))),
        "self and mat2 must have the same dtype, but got f64 and f32"
    );
}

#[test]
fn activation_layers_apply_their_functions() {
    let x = homura::randn([3, 4]);
    let same = |a: &Tensor, b: &Tensor| {
        assert!(!a.is_same(&x));
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    };
    same(&Relu.forward(&x), &x.relu());
    same(&Gelu::new().forward(&x), &x.gelu(GeluApproximate::None));
    same(
        &Gelu::new().approximate(GeluApproximate::Tanh).forward(&x),
        &x.gelu(GeluApproximate::Tanh),
    );
    same(&Silu.forward(&x), &x.silu());
    same(&Tanh.forward(&x), &x.tanh());
    same(&Sigmoid.forward(&x), &x.sigmoid());
    same(&Softmax::new(-1).forward(&x), &x.softmax(-1));
    same(&Softmax::new(0).forward(&x), &x.softmax(0));
    // The functional forms.
    same(&f::relu(&x), &x.relu());
    same(
        &f::gelu(&x, GeluApproximate::Tanh),
        &x.gelu(GeluApproximate::Tanh),
    );
    same(&f::silu(&x), &x.silu());
    same(&f::tanh(&x), &x.tanh());
    same(&f::sigmoid(&x), &x.sigmoid());
    same(&f::softmax(&x, 1), &x.softmax(1));
    same(&f::log_softmax(&x, 1), &x.log_softmax(1));
    // Through `Forward`, as in a Sequential.
    let layers: [&dyn Forward; 6] = [
        &Relu,
        &Gelu::new(),
        &Silu,
        &Tanh,
        &Sigmoid,
        &Softmax::new(1),
    ];
    for layer in layers {
        assert!(layer.named_parameters().is_empty() && layer.state_dict().is_empty());
        assert_eq!(layer.forward(&x).shape(), [3, 4]);
    }
    assert_eq!(Gelu::default().approximate, GeluApproximate::None);
    assert_eq!(format!("{:?}", Gelu::new()), "Gelu { approximate: None }");
    assert_eq!(format!("{:?}", Softmax::new(1)), "Softmax { dim: 1 }");
    assert_eq!(format!("{Relu:?}"), "Relu");

    // The hyperparameters are plain values of the modules that hold them.
    #[derive(Module)]
    struct Block {
        act: Gelu,
        approximate: GeluApproximate,
        out: Softmax,
    }
    let block = Block {
        act: Gelu::new(),
        approximate: GeluApproximate::Tanh,
        out: Softmax::new(-1),
    };
    let names: Vec<String> = block.named_modules().into_iter().map(|(k, _)| k).collect();
    assert_eq!(names, ["", "act", "out"]);
}

/// The model of the golden case `layers.mlp_*`, see `tools/golden/gen_m2_layers.py`.
fn mlp() -> Sequential {
    Sequential::new()
        .append(Linear::new(4, 6))
        .append(Relu)
        .append(Linear::new(6, 5))
        .append(Tanh)
        .append(Linear::new(5, 5).bias(false))
        .append(Gelu::new().approximate(GeluApproximate::Tanh))
        .append(Linear::new(5, 4))
        .append(Silu)
        .append(Linear::new(4, 4))
        .append(Sigmoid)
        .append(Gelu::new())
        .append(Linear::new(4, 3))
        .append(Softmax::new(-1))
}

#[test]
fn layers_match_pytorch() {
    let golden = Golden::load("m2_layers.safetensors");
    for (tag, dtype) in [("f64", DType::F64), ("f32", DType::F32)] {
        let name = format!("layers.mlp_{tag}");
        let c = Case::new(&golden, &name);
        let model = mlp();
        model.to(dtype);
        let state: StateDict = golden
            .names(&name)
            .iter()
            .filter_map(|n| {
                n.strip_prefix("param.")
                    .map(|k| (k.to_string(), c.input(n)))
            })
            .collect();
        assert_eq!(state.len(), 11);
        model.load_state_dict(&state, true).unwrap();
        let x = c.input("x");
        x.requires_grad_(true);
        let out = model.forward(&x);
        let (rtol, atol) = golden_tolerance(dtype);
        let check = |key: &str, t: &Tensor| {
            c.check_with(key, t, |a, e| {
                assert_close_tol(a, e, rtol, atol, &format!("{name}/{key}"))
            })
        };
        check("out", &out);
        let params = model.named_parameters();
        let mut inputs = vec![&x];
        inputs.extend(params.iter().map(|(_, p)| &**p));
        let grads = autograd::grad(
            &[&out],
            &inputs,
            GradOptions {
                grad_outputs: vec![Some(c.input("g"))],
                ..Default::default()
            },
        );
        check("grad_x", grads[0].as_ref().unwrap());
        for ((key, _), g) in params.iter().zip(&grads[1..]) {
            check(&format!("grad.{key}"), g.as_ref().unwrap());
        }
        c.assert_all_used();
    }
}

#[test]
fn scalar_fills_check_overflow_like_pytorch() {
    use homura::Scalar;
    // (dtype, value, the elements written, or None where PyTorch reports an overflow), from
    // tools/golden/check_m2_layers.py.
    let cases: Vec<(DType, Scalar, Option<f64>)> = vec![
        (DType::U8, Scalar::Int(300), None),
        (DType::U8, Scalar::Float(300.0), None),
        (DType::U8, Scalar::Int(-1), Some(255.0)),
        (DType::U8, Scalar::Int(-255), Some(1.0)),
        (DType::U8, Scalar::Int(-256), None),
        (DType::U8, Scalar::Float(-0.5), None),
        (DType::U8, Scalar::Float(-1.0), None),
        (DType::U8, Scalar::Float(255.9), Some(255.0)),
        (DType::U8, Scalar::Float(256.0), None),
        (DType::U8, Scalar::Bool(true), Some(1.0)),
        (DType::I8, Scalar::Float(128.0), None),
        (DType::I8, Scalar::Float(127.9), Some(127.0)),
        (DType::I8, Scalar::Float(-128.9), None),
        (DType::I8, Scalar::Float(-0.9), Some(0.0)),
        (DType::I8, Scalar::Int(-128), Some(-128.0)),
        (DType::I8, Scalar::Int(-129), None),
        (DType::I32, Scalar::Float(2147483647.5), Some(2147483647.0)),
        (DType::I32, Scalar::Float(-2147483648.5), None),
        (DType::I64, Scalar::Float(2f64.powi(63)), None),
        (DType::I64, Scalar::Float(9.3e18), None),
        (DType::I64, Scalar::Float(f64::INFINITY), None),
        (DType::I64, Scalar::Float(f64::NAN), None),
        (DType::F16, Scalar::Float(65504.0), Some(65504.0)),
        (DType::F16, Scalar::Float(65519.0), None),
        (DType::F16, Scalar::Int(65505), None),
        (DType::F16, Scalar::Float(1e6), None),
        (
            DType::F16,
            Scalar::Float(f64::NEG_INFINITY),
            Some(f64::NEG_INFINITY),
        ),
        (DType::F16, Scalar::Float(f64::NAN), Some(f64::NAN)),
        (DType::BF16, Scalar::Float(65504.0), Some(65536.0)),
        (DType::BF16, Scalar::Float(3.4e38), None),
        (DType::F32, Scalar::Float(3.4e38), Some(3.4e38f32 as f64)),
        (DType::F32, Scalar::Float(3.5e38), None),
        (DType::F32, Scalar::Float(1e40), None),
        (DType::F32, Scalar::Float(1e-50), Some(0.0)),
        (DType::F64, Scalar::Float(1e300), Some(1e300)),
        (DType::Bool, Scalar::Float(f64::NAN), Some(1.0)),
        (DType::Bool, Scalar::Int(-256), Some(1.0)),
        (DType::Bool, Scalar::Float(0.0), Some(0.0)),
    ];
    let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
    for (dtype, value, expected) in cases {
        let what = format!("{dtype} {value:?}");
        let message = format!("value cannot be converted to type {dtype} without overflow");
        type Fill = Box<dyn Fn(&[usize]) -> Tensor>;
        let fills: [(&str, Fill); 4] = [
            (
                "fill_",
                Box::new(move |s| homura::zeros_with(s, dtype).fill_(value).clone()),
            ),
            (
                "full_with",
                Box::new(move |s| homura::full_with(s, value, dtype)),
            ),
            (
                "constant_",
                Box::new(move |s| init::constant_(&homura::zeros_with(s, dtype), value).clone()),
            ),
            (
                "masked_fill",
                Box::new(move |s| {
                    homura::zeros_with(s, dtype)
                        .masked_fill(&homura::ones_with(s, DType::Bool), value)
                }),
            ),
        ];
        for (name, fill) in &fills {
            // Tensors without elements are checked too.
            for shape in [&[2][..], &[0]] {
                match expected {
                    Some(v) if shape == [2] => {
                        let t = fill(shape);
                        assert_eq!(t.dtype(), dtype, "{name} {what}");
                        assert!(
                            as_f64(&t).iter().all(|&x| same(x, v)),
                            "{name} {what}: {t:?}"
                        );
                    }
                    Some(_) => assert_eq!(fill(shape).shape().to_vec(), shape),
                    None => {
                        assert_eq!(
                            panic_message(|| drop(fill(shape))),
                            message,
                            "{name} {what}"
                        )
                    }
                }
            }
        }
        // Meta tensors take any value, like PyTorch's meta kernels.
        let meta = homura::zeros_with([2], (dtype, Device::Meta));
        meta.fill_(value);
        assert_eq!(
            homura::full_with([2], value, (dtype, Device::Meta)).device(),
            Device::Meta
        );
        let meta_mask = homura::ones_with([2], (DType::Bool, Device::Meta));
        assert_eq!(meta.masked_fill(&meta_mask, value).device(), Device::Meta);
    }

    // `full` without a dtype converts floats to the default dtype.
    assert_eq!(
        panic_message(|| drop(homura::full([2], 1e40))),
        "value cannot be converted to type f32 without overflow"
    );

    // A rejected fill changes nothing: data, version (a graph that saved the tensor still runs),
    // history and an existing `.grad`; and `constant_` restores grad mode.
    let leaf = homura::tensor([1.0f32, 2.0]);
    leaf.requires_grad_(true);
    leaf.set_grad(Some(homura::tensor([5.0f32, 6.0])));
    let grad = leaf.grad().unwrap();
    let saved = &leaf * &leaf;
    assert_eq!(
        panic_message(|| {
            let _ = init::constant_(&leaf, 1e40);
        }),
        "value cannot be converted to type f32 without overflow"
    );
    assert!(homura::is_grad_enabled());
    // In grad mode, the in-place check on the leaf comes first, as in the reference.
    assert!(
        panic_message(|| {
            let _ = leaf.fill_(1e40);
        })
        .starts_with("a leaf Variable that requires grad is being used in an in-place operation")
    );
    assert_eq!(leaf.to_vec::<f32>(), [1.0, 2.0]);
    assert!(leaf.grad().unwrap().is_same(&grad) && leaf.is_leaf());
    let y = &leaf * 1.0;
    let node = y.grad_fn().unwrap();
    assert_eq!(
        panic_message(|| {
            let _ = y.fill_(1e40);
        }),
        "value cannot be converted to type f32 without overflow"
    );
    assert!(y.grad_fn().unwrap().is_same(&node));
    saved.sum().backward();
    assert_eq!(leaf.grad().unwrap().to_vec::<f32>(), [7.0, 10.0]);
}

#[test]
fn masked_fill_checks_tensor_values_like_scalars() {
    // (R58) A zero-dimensional tensor value is checked like a scalar value: (dtype of self,
    // value), all overflowing.
    let value = |v: f64, dtype: DType| homura::tensor_with(v, dtype).reshape(Shape::scalar());
    let overflowing = [
        (DType::U8, Tensor::from_vec(vec![300i64], ())),
        (DType::U8, value(-1.0, DType::F32)),
        (DType::F16, value(1e6, DType::F64)),
        (DType::F32, value(1e40, DType::F64)),
    ];
    for (dtype, v) in &overflowing {
        let message = format!("value cannot be converted to type {dtype} without overflow");
        for shape in [&[2][..], &[0]] {
            for fill in [true, false] {
                let mask = homura::full_with(shape, fill, DType::Bool);
                let x = homura::zeros_with(shape, *dtype);
                let what = format!("{dtype} {v:?} {shape:?} {fill}");
                assert_eq!(
                    panic_message(|| drop(x.masked_fill(&mask, v))),
                    message,
                    "{what}"
                );
                assert_eq!(
                    panic_message(|| {
                        let _ = x.masked_fill_(&mask, v);
                    }),
                    message,
                    "{what}"
                );
                assert!(as_f64(&x).iter().all(|&e| e == 0.0));
            }
        }
    }
    // Values in range, of the same or other dtypes, as PyTorch converts them.
    let mask = homura::tensor([true, false]);
    let x = homura::zeros_with([2], DType::U8);
    assert_eq!(
        x.masked_fill(&mask, Tensor::from_vec(vec![-1i64], ()))
            .to_vec::<u8>(),
        [255, 0]
    );
    assert_eq!(
        x.masked_fill(&mask, value(255.9, DType::F64))
            .to_vec::<u8>(),
        [255, 0]
    );
    let x = homura::zeros_with([2], DType::I64);
    let max = Tensor::from_vec(vec![i64::MAX], ());
    assert_eq!(x.masked_fill(&mask, &max).to_vec::<i64>(), [i64::MAX, 0]);
    let x = homura::zeros_with([2], DType::F16);
    let inf = value(f64::INFINITY, DType::F64);
    assert_eq!(as_f64(&x.masked_fill(&mask, &inf)), [f64::INFINITY, 0.0]);

    // A value that requires grad gets the gradient of the positions it fills.
    let x = homura::tensor([1.0f32, 2.0, 3.0]);
    x.requires_grad_(true);
    let v = value(2.7, DType::F64);
    v.requires_grad_(true);
    let mask3 = homura::tensor([true, false, true]);
    let y = x.masked_fill(&mask3, &v);
    assert_eq!(y.to_vec::<f32>(), [2.7, 2.0, 2.7]);
    (&y * &homura::tensor([1.0f32, 2.0, 3.0])).sum().backward();
    assert_eq!(v.grad().unwrap().to_vec::<f64>(), [4.0]);
    assert_eq!(x.grad().unwrap().to_vec::<f32>(), [0.0, 2.0, 0.0]);

    // A rejected in-place fill of a recorded non-leaf changes nothing: data, history and a graph
    // that saved it.
    let leaf = homura::tensor([1.0f32, 2.0]);
    leaf.requires_grad_(true);
    let y = &leaf * 1.0;
    let saved = &y * &y;
    let node = y.grad_fn().unwrap();
    let big = value(1e40, DType::F64);
    big.requires_grad_(true);
    assert_eq!(
        panic_message(|| {
            let _ = y.masked_fill_(&homura::tensor([true, true]), &big);
        }),
        "value cannot be converted to type f32 without overflow"
    );
    assert!(y.grad_fn().unwrap().is_same(&node));
    assert_eq!(y.to_vec::<f32>(), [1.0, 2.0]);
    saved.sum().backward();
    assert_eq!(leaf.grad().unwrap().to_vec::<f32>(), [2.0, 4.0]);
    assert!(big.grad().is_none());

    // Meta tensors take any value.
    let meta = homura::zeros_with([2], (DType::U8, Device::Meta));
    let meta_mask = homura::ones_with([2], (DType::Bool, Device::Meta));
    let meta_value = homura::full_with([], 300, (DType::I64, Device::Meta));
    assert_eq!(
        meta.masked_fill(&meta_mask, &meta_value).device(),
        Device::Meta
    );
    meta.masked_fill_(&meta_mask, &meta_value);
}
