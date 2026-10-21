//! M2 step 9a: the layers `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AdaptiveAvgPool2d` and
//! `Flatten`. Sequences of `BatchNorm2d` calls and the CNN of step 9 are compared with
//! `tests/golden/m2_cnn_layers.safetensors` from `tools/golden/gen_m2_cnn_layers.py`; the
//! reference behavior encoded in the other tests is asserted by
//! `tools/golden/check_m2_cnn_layers.py`, in the same order.
//!
//! The layers are thin wrappers of functions that the step 8 tests compare with PyTorch in every
//! dtype: [`layers_compute_their_functions`] checks that each layer gives the same values as its
//! function with zero tolerance, in each dtype. The golden comparisons here are about what the
//! layers pass to the functions (which statistics, momentum and mode), so they use the default
//! tolerances.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden, assert_close_tol};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::nn::functional::{
    self, BatchNormOptions, Conv2dOptions, Conv2dPadding, MaxPool2dOptions,
};
use homura::nn::{
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Flatten, Forward, Linear, MaxPool2d, Module, Relu,
    Sequential, StateDict,
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

fn assert_panics(f: impl FnOnce(), message: &str) {
    assert_eq!(panic_message(f), message);
}

/// Distinct values of both signs, between 0.3 and 1.3 in magnitude, in F64.
fn constant(shape: &[usize], seed: usize) -> Tensor {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let magnitude = 0.3 + ((i * 7 + seed * 3) % 11) as f64 / 10.0 + i as f64 * 1e-3;
            if (i + seed).is_multiple_of(3) {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect();
    Tensor::from_vec(values, shape)
}

/// [`constant`] requiring grad.
fn var(shape: &[usize], seed: usize) -> Tensor {
    let x = constant(shape, seed);
    x.requires_grad_(true);
    x
}

fn grads(out: &Tensor, inputs: &[&Tensor], g: &Tensor) -> Vec<Option<Tensor>> {
    autograd::grad(
        &[out],
        inputs,
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            allow_unused: true,
            ..Default::default()
        },
    )
}

fn keys(m: &impl Module) -> Vec<String> {
    m.state_dict().keys().map(String::from).collect()
}

fn as_f64(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

fn count(bn: &BatchNorm2d) -> i64 {
    bn.num_batches_tracked.as_ref().unwrap().item::<i64>()
}

/// Asserts that `a` and `b` have the same shape, dtype and values with zero tolerance (`0.0`
/// equals `-0.0`, NaN matches NaN).
fn assert_equal(a: &Tensor, b: &Tensor, what: &str) {
    assert_close_tol(a, b, 0.0, 0.0, what);
}

#[test]
fn conv2d_layers_initialize_like_pytorch() {
    let conv = Conv2d::new(3, 8, 3).padding(1);
    assert_eq!(keys(&conv), ["weight", "bias"]);
    assert_eq!(conv.weight.shape(), [8, 3, 3, 3]);
    assert_eq!(conv.bias.as_ref().unwrap().shape(), [8]);
    assert!(conv.weight.requires_grad() && conv.weight.is_leaf());
    assert_eq!(conv.kernel_size, [3, 3]);
    assert_eq!(
        format!("{conv:?}"),
        "Conv2d { in_channels: 3, out_channels: 8, kernel_size: [3, 3], stride: [1, 1], \
         padding: Explicit([1, 1]), .. }"
    );
    // kaiming_uniform_(a = √5) and the bias both sample from U(-1/√fan_in, 1/√fan_in).
    let bound = 1.0 / 27f64.sqrt();
    for t in [&*conv.weight, conv.bias.as_ref().unwrap()] {
        let v = as_f64(t);
        assert!(v.iter().all(|x| x.abs() <= bound) && v.iter().any(|&x| x != 0.0));
    }
    let conv = Conv2d::new(4, 6, [3, 2])
        .stride([2, 1])
        .dilation(2)
        .groups(2)
        .bias(false);
    assert_eq!(keys(&conv), ["weight"]);
    assert_eq!(conv.weight.shape(), [6, 2, 3, 2]);
    assert!(
        as_f64(&conv.weight)
            .iter()
            .all(|x| x.abs() <= 1.0 / 12f64.sqrt())
    );
    assert_eq!(
        format!("{conv:?}"),
        "Conv2d { in_channels: 4, out_channels: 6, kernel_size: [3, 2], stride: [2, 1], \
         dilation: [2, 2], groups: 2, bias: false, .. }"
    );
    let conv = Conv2d::new(4, 6, 3).padding(Conv2dPadding::Valid);
    assert_eq!(
        format!("{conv:?}"),
        "Conv2d { in_channels: 4, out_channels: 6, kernel_size: [3, 3], stride: [1, 1], \
         padding: Valid, .. }"
    );
    assert_eq!(
        format!("{:?}", Conv2d::new(4, 6, 3).padding(0)),
        "Conv2d { in_channels: 4, out_channels: 6, kernel_size: [3, 3], stride: [1, 1], .. }"
    );

    // Sizes of 0 make empty weights; a fan_in of 0 gives a zero bias.
    for (c, o, k) in [(0, 6, 3), (3, 0, 3), (3, 4, 0)] {
        let conv = Conv2d::new(c, o, k);
        assert_eq!(conv.weight.num_elements(), 0);
        if c == 0 || k == 0 {
            assert!(
                as_f64(conv.bias.as_ref().unwrap())
                    .iter()
                    .all(|&x| x == 0.0)
            );
        }
    }

    // `reset_parameters` draws new values in place.
    let conv = Conv2d::new(2, 2, 2);
    let before = conv.weight.deep_clone();
    conv.reset_parameters();
    assert_ne!(as_f64(&conv.weight), as_f64(&before));

    // Layers follow the default device.
    let conv = {
        let _meta = homura::device_scope(Device::Meta);
        Conv2d::new(3, 2, 3)
    };
    assert_eq!(conv.weight.device(), Device::Meta);
    let y = conv.forward(&homura::zeros_with([5, 3, 4, 4], Device::Meta));
    assert_eq!((y.shape(), y.device()), ([5, 2, 2, 2].into(), Device::Meta));
}

#[test]
fn conv2d_layers_check_their_options() {
    // PyTorch's checks and order: groups, then the padding.
    assert_panics(
        || drop(Conv2d::new(4, 6, 3).groups(0)),
        "groups must be a positive integer",
    );
    assert_panics(
        || drop(Conv2d::new(4, 6, 3).groups(3)),
        "in_channels must be divisible by groups",
    );
    assert_panics(
        || drop(Conv2d::new(4, 6, 3).groups(4)),
        "out_channels must be divisible by groups",
    );
    let same = "padding='same' is not supported for strided convolutions";
    assert_panics(
        || drop(Conv2d::new(4, 6, 3).padding(Conv2dPadding::Same).stride(2)),
        same,
    );
    assert_panics(
        || {
            drop(
                Conv2d::new(4, 6, 3)
                    .stride([1, 2])
                    .padding(Conv2dPadding::Same),
            )
        },
        same,
    );
    let conv = Conv2d::new(4, 6, 3)
        .padding(Conv2dPadding::Same)
        .dilation(2);
    assert_eq!(
        conv.forward(&homura::ones([1, 4, 5, 5])).shape(),
        [1, 6, 5, 5]
    );
    // The other options are checked by the forward computation, as in PyTorch.
    let conv = Conv2d::new(4, 6, 3).stride(0);
    assert_panics(
        || drop(conv.forward(&homura::ones([1, 4, 5, 5]))),
        "non-positive stride is not supported",
    );
}

#[test]
fn conv2d_setters_keep_the_parameters_options() {
    for options in [DType::F64.into(), homura::TensorOptions::from(Device::Meta)] {
        let conv = Conv2d::new(4, 6, 3).bias(false);
        conv.to(options);
        let (dtype, device) = (conv.weight.dtype(), conv.weight.device());
        // A new bias follows the weights.
        let conv = conv.bias(true);
        let bias = conv.bias.as_ref().unwrap();
        assert_eq!((bias.dtype(), bias.device()), (dtype, device));
        // The same number of groups keeps the parameters; another one replaces them.
        let (weight, bias) = (conv.weight.clone(), bias.clone());
        let conv = conv.groups(1);
        assert!(conv.weight.is_same(&weight));
        assert!(conv.bias.as_ref().unwrap().is_same(&bias));
        let conv = conv.groups(2);
        assert!(!conv.weight.is_same(&weight));
        assert_eq!(conv.weight.shape(), [6, 2, 3, 3]);
        let new_bias = conv.bias.as_ref().unwrap();
        assert!(!new_bias.is_same(&bias));
        for t in [&*conv.weight, new_bias] {
            assert_eq!((t.dtype(), t.device()), (dtype, device));
            assert!(t.requires_grad() && t.is_leaf());
        }
    }
    // The new parameters are initialized for the new fan_in: 2 * 3 * 3.
    let conv = Conv2d::new(4, 6, 3).groups(2);
    let bound = 1.0 / 18f64.sqrt();
    for t in [&*conv.weight, conv.bias.as_ref().unwrap()] {
        assert!(as_f64(t).iter().all(|x| x.abs() <= bound));
    }
    // Some of the 108 weights exceed the bound of twice the fan_in (all below has odds 0.707^108).
    assert!(
        as_f64(&conv.weight)
            .iter()
            .any(|x| x.abs() > 1.0 / 36f64.sqrt())
    );
    // A layer without a bias stays without one.
    assert!(Conv2d::new(4, 6, 3).bias(false).groups(2).bias.is_none());
}

const BN_KEYS: [&str; 5] = [
    "weight",
    "bias",
    "running_mean",
    "running_var",
    "num_batches_tracked",
];

#[test]
fn batch_norm_layers_initialize_like_pytorch() {
    let bn = BatchNorm2d::new(3);
    assert_eq!(keys(&bn), BN_KEYS);
    let sd = bn.state_dict();
    assert_eq!(as_f64(sd.get("weight").unwrap()), [1.0; 3]);
    assert_eq!(as_f64(sd.get("bias").unwrap()), [0.0; 3]);
    assert_eq!(as_f64(sd.get("running_mean").unwrap()), [0.0; 3]);
    assert_eq!(as_f64(sd.get("running_var").unwrap()), [1.0; 3]);
    let nbt = sd.get("num_batches_tracked").unwrap();
    assert_eq!((nbt.shape(), nbt.dtype()), ([].into(), DType::I64));
    assert_eq!(nbt.item::<i64>(), 0);
    let names: Vec<String> = bn.named_parameters().into_iter().map(|(k, _)| k).collect();
    assert_eq!(names, ["weight", "bias"]);
    let names: Vec<String> = bn.named_buffers().into_iter().map(|(k, _)| k).collect();
    assert_eq!(names, BN_KEYS[2..]);
    assert_eq!(
        format!("{bn:?}"),
        "BatchNorm2d { num_features: 3, eps: 1e-5, momentum: Some(0.1), affine: true, \
         bias: true, track_running_stats: true, .. }"
    );
    let bn = BatchNorm2d::new(3)
        .eps(1e-3)
        .momentum(None)
        .affine(false)
        .track_running_stats(false);
    assert!(keys(&bn).is_empty() && bn.named_buffers().is_empty());
    assert_eq!(
        format!("{bn:?}"),
        "BatchNorm2d { num_features: 3, eps: 0.001, momentum: None, affine: false, \
         bias: false, track_running_stats: false, .. }"
    );
    assert_eq!(
        keys(&BatchNorm2d::new(3).bias(false)),
        [
            "weight",
            "running_mean",
            "running_var",
            "num_batches_tracked"
        ]
    );
    assert_eq!(
        keys(&BatchNorm2d::new(3).track_running_stats(false)),
        ["weight", "bias"]
    );
    // Converting the layer keeps the count an I64.
    let bn = BatchNorm2d::new(3);
    bn.to(DType::F64);
    let dtypes: Vec<DType> = bn.state_dict().values().map(Tensor::dtype).collect();
    assert_eq!(
        dtypes,
        [DType::F64, DType::F64, DType::F64, DType::F64, DType::I64]
    );
}

#[test]
fn batch_norm_setters_keep_the_members_options() {
    let options_of = |t: &Tensor| (t.dtype(), t.device());
    for options in [DType::F64.into(), homura::TensorOptions::from(Device::Meta)] {
        let bn = BatchNorm2d::new(3);
        bn.to(options);
        let like = options_of(bn.weight.as_ref().unwrap());
        let members = |bn: &BatchNorm2d| -> Vec<Tensor> {
            [
                bn.weight.as_deref(),
                bn.bias.as_deref(),
                bn.running_mean.as_ref(),
                bn.running_var.as_ref(),
                bn.num_batches_tracked.as_ref(),
            ]
            .into_iter()
            .flatten()
            .cloned()
            .collect()
        };
        // Options that are on keep their members.
        let old = members(&bn);
        let bn = bn.affine(true).bias(true).track_running_stats(true);
        let new = members(&bn);
        assert_eq!(new.len(), 5);
        assert!(old.iter().zip(&new).all(|(a, b)| a.is_same(b)));
        // A bias added again follows the weight, also without statistics.
        let bn = bn.bias(false).track_running_stats(false).bias(true);
        let bias = bn.bias.as_ref().unwrap();
        assert_eq!(options_of(bias), like);
        assert!(bias.requires_grad() && bias.is_leaf());
        // Statistics added again follow the parameters.
        let bn = bn.track_running_stats(true);
        assert_eq!(options_of(bn.running_mean.as_ref().unwrap()), like);
        assert_eq!(options_of(bn.running_var.as_ref().unwrap()), like);
        let nbt = bn.num_batches_tracked.as_ref().unwrap();
        assert_eq!(options_of(nbt), (DType::I64, like.1));
        assert_eq!(nbt.shape(), []);
        // Parameters added again follow the statistics.
        let bn = bn.affine(false).affine(true);
        for t in [bn.weight.as_deref().unwrap(), bn.bias.as_deref().unwrap()] {
            assert_eq!(options_of(t), like);
            assert!(t.requires_grad() && t.is_leaf());
        }
        if like.1 == Device::Cpu {
            assert_eq!(as_f64(bn.weight.as_ref().unwrap()), [1.0; 3]);
            assert_eq!(as_f64(bn.bias.as_ref().unwrap()), [0.0; 3]);
            assert_eq!(as_f64(bn.running_mean.as_ref().unwrap()), [0.0; 3]);
            assert_eq!(as_f64(bn.running_var.as_ref().unwrap()), [1.0; 3]);
            assert_eq!(count(&bn), 0);
        }
    }
    // Without affine, `bias(true)` adds no bias.
    assert!(BatchNorm2d::new(3).affine(false).bias(true).bias.is_none());
    // Without any floating point member, new members get the default options.
    let bn = BatchNorm2d::new(3).affine(false).track_running_stats(false);
    let bn = {
        let _meta = homura::device_scope(Device::Meta);
        bn.track_running_stats(true).affine(true)
    };
    let all: Vec<(DType, Device)> = bn.state_dict().values().map(options_of).collect();
    let f32_meta = (DType::F32, Device::Meta);
    assert_eq!(
        all,
        [
            f32_meta,
            f32_meta,
            f32_meta,
            f32_meta,
            (DType::I64, Device::Meta)
        ]
    );
}

/// The calls of the golden `bn.*` cases: (training, batch size).
const SEQUENCE: [(bool, usize); 5] = [(true, 4), (true, 5), (false, 3), (true, 0), (false, 2)];

fn golden_batch_norm(variant: &str) -> BatchNorm2d {
    let bn = BatchNorm2d::new(3);
    match variant {
        "momentum" => bn,
        "cumulative" => bn.momentum(None),
        "options" => bn.eps(1e-3).momentum(0.3),
        "untracked" => bn.track_running_stats(false),
        "no_affine" => bn.affine(false),
        "no_bias" => bn.bias(false),
        _ => panic!("unknown variant {variant}"),
    }
}

/// The entries of `c` named `<prefix>.<key>`, as a state dict.
fn golden_state(c: &Case, prefix: &str) -> StateDict {
    c.golden
        .names(c.name)
        .into_iter()
        .filter_map(|n| {
            let key = n.strip_prefix(prefix)?.strip_prefix('.')?.to_string();
            Some((key, c.input(&n)))
        })
        .collect()
}

#[test]
fn batch_norm_layers_match_pytorch() {
    let golden = Golden::load("m2_cnn_layers.safetensors");
    let mut seen = 0;
    for name in golden.cases().iter().filter(|n| n.starts_with("bn.")) {
        let c = Case::new(&golden, name);
        let (variant, tag) = name["bn.".len()..].rsplit_once('_').unwrap();
        let dtype = if tag == "f64" { DType::F64 } else { DType::F32 };
        let bn = golden_batch_norm(variant);
        bn.to(dtype);
        bn.load_state_dict(&golden_state(&c, "init"), true).unwrap();
        let params: Vec<(String, homura::nn::Parameter)> = bn.named_parameters();
        for (i, (training, n)) in SEQUENCE.into_iter().enumerate() {
            if training {
                bn.train();
            } else {
                bn.eval();
            }
            let x = c.input(&format!("x{i}"));
            assert_eq!(x.shape()[0], n);
            x.requires_grad_(true);
            let out = bn.forward(&x);
            c.check(&format!("out{i}"), &out);
            if n > 0 {
                let mut inputs = vec![&x];
                inputs.extend(params.iter().map(|(_, p)| &**p));
                let g = grads(&out, &inputs, &c.input(&format!("g{i}")));
                c.check(&format!("grad_x{i}"), g[0].as_ref().unwrap());
                for ((key, _), gk) in params.iter().zip(&g[1..]) {
                    c.check(&format!("grad.{key}{i}"), gk.as_ref().unwrap());
                }
            }
            let mut want: Vec<String> = golden_state(&c, &format!("state{i}"))
                .keys()
                .map(String::from)
                .collect();
            let mut have = keys(&bn);
            want.sort();
            have.sort();
            assert_eq!(have, want);
            for (key, t) in bn.state_dict().iter() {
                c.check(&format!("state{i}.{key}"), t);
            }
        }
        c.assert_all_used();
        seen += 1;
    }
    assert_eq!(seen, 12);
}

#[test]
fn batch_norm_layers_count_like_pytorch() {
    let x = constant(&[3, 2, 2, 2], 0);
    let bn = BatchNorm2d::new(2);
    bn.to(DType::F64);
    // The rank is checked before counting.
    assert_panics(
        || drop(bn.forward(&constant(&[2, 2, 2], 0))),
        "expected 4D input (got 3D input)",
    );
    assert_eq!(count(&bn), 0);
    // Later errors come after counting, and the statistics stay.
    assert_panics(
        || drop(bn.forward(&constant(&[1, 2, 1, 1], 0))),
        "Expected more than 1 value per channel when training, got input size torch.Size([1, 2, 1, 1])",
    );
    assert_panics(
        || drop(bn.forward(&constant(&[2, 3, 2, 2], 0))),
        "running_mean should contain 3 elements not 2",
    );
    assert_eq!(count(&bn), 2);
    assert_eq!(as_f64(bn.running_mean.as_ref().unwrap()), [0.0; 2]);
    assert_eq!(as_f64(bn.running_var.as_ref().unwrap()), [1.0; 2]);
    // Training passes count without grad mode too; evaluation passes don't, with grad mode
    // either.
    {
        let _no_grad = homura::no_grad();
        bn.forward(&x);
    }
    assert_eq!(count(&bn), 3);
    bn.eval();
    {
        let _grad = homura::enable_grad();
        bn.forward(&x);
    }
    assert_eq!(count(&bn), 3);

    // Without a count, a training pass doesn't count, and a momentum of None keeps a factor of
    // 0: the statistics stay.
    let mut bn = BatchNorm2d::new(2).momentum(None);
    bn.to(DType::F64);
    bn.num_batches_tracked = None;
    bn.forward(&x);
    assert_eq!(as_f64(bn.running_mean.as_ref().unwrap()), [0.0; 2]);
    assert_eq!(as_f64(bn.running_var.as_ref().unwrap()), [1.0; 2]);
    // A layer that doesn't track statistics passes none to a training pass, even if it has
    // them, and normalizes with them in evaluation mode.
    let mut bn = BatchNorm2d::new(2);
    bn.to(DType::F64);
    bn.track_running_stats = false;
    let y = bn.forward(&x);
    assert_eq!(count(&bn), 0);
    assert_eq!(as_f64(bn.running_mean.as_ref().unwrap()), [0.0; 2]);
    let plain = |training| BatchNormOptions {
        training,
        ..Default::default()
    };
    let w = bn.weight.as_deref();
    let b = bn.bias.as_deref();
    assert_equal(
        &y,
        &functional::batch_norm(&x, None, None, w, b, plain(true)),
        "untracked",
    );
    bn.eval();
    let (rm, rv) = (bn.running_mean.as_ref(), bn.running_var.as_ref());
    let want = functional::batch_norm(&x, rm, rv, w, b, plain(false));
    assert_equal(&bn.forward(&x), &want, "untracked flag, statistics");
    // In evaluation mode, a layer without statistics normalizes with the batch's; one with only
    // one of them fails in the function.
    bn.running_mean = None;
    assert_panics(
        || drop(bn.forward(&x)),
        "running_mean must be defined in evaluation mode",
    );
    bn.running_var = None;
    let want = functional::batch_norm(&x, None, None, w, b, plain(true));
    assert_equal(&bn.forward(&x), &want, "no statistics");
    let mut bn = BatchNorm2d::new(2);
    bn.eval();
    bn.running_var = None;
    assert_panics(
        || drop(bn.forward(&x.to(DType::F32))),
        "running_var must be defined in evaluation mode",
    );

    // The resets set values in place (a graph that saved them can't run backward any more) and
    // keep the mode.
    let bn = BatchNorm2d::new(2);
    bn.to(DType::F64);
    let xv = var(&[3, 2, 2, 2], 1);
    bn.forward(&xv);
    bn.forward(&xv);
    assert_eq!(count(&bn), 2);
    bn.eval();
    let out = bn.forward(&xv);
    bn.reset_running_stats();
    assert_eq!(count(&bn), 0);
    assert_eq!(as_f64(bn.running_mean.as_ref().unwrap()), [0.0; 2]);
    assert_eq!(as_f64(bn.running_var.as_ref().unwrap()), [1.0; 2]);
    assert!(!bn.training.get());
    let message = panic_message(|| drop(grads(&out, &[&xv], &constant(&[3, 2, 2, 2], 2))));
    assert!(
        message.contains("modified by an inplace operation"),
        "{message}"
    );
    bn.weight.as_ref().unwrap().detach().fill_(2.0);
    bn.train();
    let out = bn.forward(&xv);
    bn.reset_parameters();
    assert_eq!(as_f64(bn.weight.as_ref().unwrap()), [1.0; 2]);
    assert_eq!(count(&bn), 0);
    assert!(bn.training.get());
    let message = panic_message(|| drop(grads(&out, &[&xv], &constant(&[3, 2, 2, 2], 2))));
    assert!(
        message.contains("modified by an inplace operation"),
        "{message}"
    );
    // A layer that doesn't track statistics has none to reset.
    BatchNorm2d::new(2)
        .track_running_stats(false)
        .reset_running_stats();
}

/// A `BatchNorm2d` with `momentum` whose state dict, loaded strictly, has the count `count`.
fn with_count(dtype: DType, count: i64, momentum: Option<f64>) -> BatchNorm2d {
    let bn = BatchNorm2d::new(2).momentum(momentum);
    bn.to(dtype);
    let mut state = bn.state_dict();
    state.insert("num_batches_tracked", homura::tensor(count));
    bn.load_state_dict(&state, true).unwrap();
    bn
}

#[test]
fn batch_norm_layers_reject_a_zero_cumulative_count() {
    // A count of -1 loads like any I64 value. The cumulative average of the next training pass
    // divides by the new count of 0, which PyTorch's Python division rejects before
    // normalizing: the count is 0 afterwards, and the running statistics keep their values and
    // versions (a graph that saved them still runs backward).
    for dtype in [DType::F64, DType::F32] {
        let x = constant(&[3, 2, 2, 2], 0).to(dtype);
        for stats in [true, false] {
            for grad in [true, false] {
                let mut bn = with_count(dtype, -1, None);
                let saved = if stats {
                    bn.eval();
                    let xv = x.deep_clone();
                    xv.requires_grad_(true);
                    let out = bn.forward(&xv);
                    bn.train();
                    Some((xv, out))
                } else {
                    bn.running_mean = None;
                    bn.running_var = None;
                    None
                };
                {
                    let _mode = homura::set_grad_enabled(grad);
                    assert_panics(|| drop(bn.forward(&x)), "division by zero");
                }
                assert_eq!(count(&bn), 0, "{dtype} {stats} {grad}");
                if let Some((xv, out)) = saved {
                    assert_eq!(as_f64(bn.running_mean.as_ref().unwrap()), [0.0; 2]);
                    assert_eq!(as_f64(bn.running_var.as_ref().unwrap()), [1.0; 2]);
                    assert!(grads(&out, &[&xv], &homura::ones_like(&out))[0].is_some());
                }
            }
        }
        // The other counts and modes are not affected: from -2 the factor is 1 / -1.
        for (start, factor) in [(-2, -1.0), (0, 1.0), (3, 0.25)] {
            let bn = with_count(dtype, start, None);
            let (rm, rv) = (
                bn.running_mean.as_ref().unwrap().deep_clone(),
                bn.running_var.as_ref().unwrap().deep_clone(),
            );
            let options = BatchNormOptions {
                training: true,
                momentum: factor,
                ..Default::default()
            };
            let (w, b) = (bn.weight.as_deref(), bn.bias.as_deref());
            let want = functional::batch_norm(&x, Some(&rm), Some(&rv), w, b, options);
            assert_equal(&bn.forward(&x), &want, "cumulative");
            assert_equal(bn.running_mean.as_ref().unwrap(), &rm, "running_mean");
            assert_equal(bn.running_var.as_ref().unwrap(), &rv, "running_var");
            assert_eq!(count(&bn), start + 1);
        }
        let bn = with_count(dtype, -1, Some(0.1));
        bn.forward(&x);
        assert_eq!(count(&bn), 0);
        let bn = with_count(dtype, -1, None);
        bn.eval();
        bn.forward(&x);
        assert_eq!(count(&bn), -1);
        let mut bn = with_count(dtype, -1, None);
        bn.track_running_stats = false;
        bn.forward(&x);
        assert_eq!(count(&bn), -1);
        let mut bn = with_count(dtype, -1, None);
        bn.num_batches_tracked = None;
        bn.forward(&x);
    }
}

#[test]
fn batch_norm_layers_on_meta() {
    let x = homura::zeros_with([2, 2, 3, 3], Device::Meta);
    let bn = BatchNorm2d::new(2);
    bn.to(Device::Meta);
    let y = bn.forward(&x);
    assert_eq!((y.shape(), y.device()), (x.shape(), Device::Meta));
    // A momentum of None reads the count, which has no value on the meta device.
    let bn = BatchNorm2d::new(2).momentum(None);
    bn.to(Device::Meta);
    assert_panics(
        || drop(bn.forward(&x)),
        "Tensor.item() cannot be called on meta tensors",
    );
    bn.eval();
    assert_eq!(bn.forward(&x).shape(), x.shape());
}

#[test]
fn batch_norm_layers_load_state_dicts() {
    let bn = BatchNorm2d::new(2);
    bn.num_batches_tracked.as_ref().unwrap().fill_(7);
    let sd = bn.state_dict();
    let other = BatchNorm2d::new(2);
    other.load_state_dict(&sd, true).unwrap();
    assert_eq!(count(&other), 7);
    // A state dict without the count fails a strict load, and a non-strict load lists it, like
    // PyTorch for a state dict of the current version. (PyTorch takes one without version
    // metadata for one of an old version and keeps the count; homura's have no versions.)
    let mut old = bn.state_dict();
    old.remove("num_batches_tracked");
    let e = other.load_state_dict(&old, true).unwrap_err();
    assert_eq!(e.missing_keys, ["num_batches_tracked"]);
    let report = other.load_state_dict(&old, false).unwrap();
    assert_eq!(report.missing_keys, ["num_batches_tracked"]);
    assert_eq!(count(&other), 7);
}

#[test]
fn pooling_layers_apply_their_functions() {
    let x = var(&[2, 3, 5, 6], 0);
    let pool = MaxPool2d::new(2);
    assert!(keys(&pool).is_empty() && pool.named_buffers().is_empty());
    assert_eq!(
        format!("{pool:?}"),
        "MaxPool2d { kernel_size: [2, 2], stride: [2, 2], padding: [0, 0], dilation: [1, 1], \
         ceil_mode: false }"
    );
    let want = functional::max_pool2d(&x, MaxPool2dOptions::new([2, 2]));
    assert_equal(&pool.forward(&x), &want, "max_pool2d");
    let pool = MaxPool2d::new([3, 2])
        .stride(1)
        .padding([1, 0])
        .dilation([1, 2])
        .ceil_mode(true);
    let options = MaxPool2dOptions {
        stride: Some([1, 1]),
        padding: [1, 0],
        dilation: [1, 2],
        ceil_mode: true,
        ..MaxPool2dOptions::new([3, 2])
    };
    assert_eq!(pool.options, options);
    assert_eq!(
        format!("{pool:?}"),
        "MaxPool2d { kernel_size: [3, 2], stride: [1, 1], padding: [1, 0], dilation: [1, 2], \
         ceil_mode: true }"
    );
    let (values, indices) = pool.forward_with_indices(&x);
    let (want, want_indices) = functional::max_pool2d_with_indices(&x, options);
    assert_equal(&values, &want, "values");
    assert_equal(&indices, &want_indices, "indices");
    assert_eq!(
        values.grad_fn().unwrap().name(),
        "MaxPool2DWithIndicesBackward0"
    );
    // The options are checked by the forward computation.
    assert_panics(
        || drop(MaxPool2d::new(2).stride(0).forward(&x)),
        "stride should not be zero",
    );

    let pool = AdaptiveAvgPool2d::new(1);
    assert_eq!(pool.output_size, [1, 1]);
    assert!(keys(&pool).is_empty());
    assert_eq!(
        format!("{pool:?}"),
        "AdaptiveAvgPool2d { output_size: [1, 1] }"
    );
    let y = pool.forward(&x);
    assert_equal(
        &y,
        &functional::adaptive_avg_pool2d(&x, [1, 1]),
        "adaptive [1, 1]",
    );
    assert_eq!(y.grad_fn().unwrap().name(), "MeanBackward1");
    let pool = AdaptiveAvgPool2d::new([2, 4]);
    let y = pool.forward(&x);
    assert_equal(
        &y,
        &functional::adaptive_avg_pool2d(&x, [2, 4]),
        "adaptive [2, 4]",
    );
    assert_eq!(y.grad_fn().unwrap().name(), "AdaptiveAvgPool2DBackward0");
}

#[test]
fn flatten_layers_merge_dimensions() {
    let flatten = Flatten::default();
    assert_eq!((flatten.start_dim, flatten.end_dim), (1, -1));
    assert!(keys(&flatten).is_empty());
    assert_eq!(
        format!("{flatten:?}"),
        "Flatten { start_dim: 1, end_dim: -1 }"
    );
    // A contiguous input gives a view, which shares its storage.
    let x = var(&[2, 3, 4], 0);
    let y = flatten.forward(&x);
    assert_eq!(y.shape(), [2, 12]);
    assert_eq!(y.grad_fn().unwrap().name(), "ViewBackward0");
    {
        let _no_grad = homura::no_grad();
        x.select(0, 0).select(0, 0).fill_(5.0);
    }
    assert_eq!(as_f64(&y)[..4], [5.0; 4]);
    // Dimensions that can't be merged without copying are copied.
    let x = var(&[2, 3, 4], 1);
    let t = x.transpose(1, 2);
    let z = flatten.forward(&t);
    assert_equal(&z, &t.contiguous().reshape([2, 12]), "transposed");
    assert_eq!(z.grad_fn().unwrap().name(), "UnsafeViewBackward0");
    {
        let _no_grad = homura::no_grad();
        x.fill_(5.0);
    }
    assert!(as_f64(&z).iter().all(|&v| v != 5.0));
    let g = constant(&[2, 12], 2);
    let gx = grads(&z, &[&x], &g)[0].clone().unwrap();
    assert_equal(&gx, &g.reshape([2, 4, 3]).transpose(1, 2), "gradient");
    // Negative dimensions, empty inputs.
    assert_eq!(Flatten::new(-2, -1).forward(&x).shape(), [2, 12]);
    assert_eq!(Flatten::new(0, 1).forward(&x).shape(), [6, 4]);
    assert_eq!(flatten.forward(&homura::zeros([0, 3, 4])).shape(), [0, 12]);
    // PyTorch's errors.
    for x in [homura::tensor(1.0f32), homura::ones([3])] {
        assert_panics(
            || drop(flatten.forward(&x)),
            "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
        );
    }
    assert_panics(
        || drop(Flatten::new(-4, -1).forward(&homura::ones([2, 3, 4]))),
        "Dimension out of range (expected to be in range of [-3, 2], but got -4)",
    );
    assert_panics(
        || drop(Flatten::new(2, 1).forward(&homura::ones([2, 3, 4]))),
        "flatten() has invalid args: start_dim cannot come after end_dim",
    );
}

/// The output of `f` for `x` and the gradients of `x` and `params` for the upstream gradient
/// of ones.
fn output_and_grads(f: impl Fn(&Tensor) -> Tensor, x: &Tensor, params: &[&Tensor]) -> Vec<Tensor> {
    let out = f(x);
    let mut inputs = vec![x];
    inputs.extend(params);
    let g = homura::ones_with(out.shape(), out.dtype());
    let mut all = vec![out.clone()];
    all.extend(grads(&out, &inputs, &g).into_iter().map(Option::unwrap));
    all
}

#[test]
fn layers_compute_their_functions() {
    for dtype in [DType::F64, DType::F32, DType::F16, DType::BF16] {
        let input = |shape: &[usize], seed| {
            let x = constant(shape, seed).to(dtype);
            x.requires_grad_(true);
            x
        };
        let same = |a: Vec<Tensor>, b: Vec<Tensor>, what: &str| {
            assert_eq!(a.len(), b.len());
            for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                assert_equal(a, b, &format!("{what} {dtype} {i}"));
            }
        };
        let x = input(&[2, 4, 5, 5], 0);

        let conv = Conv2d::new(4, 6, 3)
            .stride(2)
            .padding(1)
            .dilation(1)
            .groups(2);
        conv.to(dtype);
        let (w, b) = (&*conv.weight, conv.bias.as_deref().unwrap());
        let options = Conv2dOptions {
            stride: [2, 2],
            padding: 1.into(),
            groups: 2,
            ..Default::default()
        };
        same(
            output_and_grads(|x| conv.forward(x), &x, &[w, b]),
            output_and_grads(|x| functional::conv2d(x, w, Some(b), options), &x, &[w, b]),
            "conv2d",
        );

        // Batch normalization with parameters and statistics of the dtype, and F32 ones for
        // half precision inputs.
        let mut param_dtypes = vec![dtype];
        if matches!(dtype, DType::F16 | DType::BF16) {
            param_dtypes.push(DType::F32);
        }
        for p in param_dtypes {
            for training in [true, false] {
                let bn = BatchNorm2d::new(4);
                bn.to(p);
                bn.weight
                    .as_ref()
                    .unwrap()
                    .detach()
                    .copy_(&constant(&[4], 1));
                bn.bias.as_ref().unwrap().detach().copy_(&constant(&[4], 2));
                bn.training.set(training);
                let (w, b) = (bn.weight.as_deref().unwrap(), bn.bias.as_deref().unwrap());
                let (rm, rv) = (
                    bn.running_mean.as_ref().unwrap().deep_clone(),
                    bn.running_var.as_ref().unwrap().deep_clone(),
                );
                let options = BatchNormOptions {
                    training,
                    ..Default::default()
                };
                let want = output_and_grads(
                    |x| functional::batch_norm(x, Some(&rm), Some(&rv), Some(w), Some(b), options),
                    &x,
                    &[w, b],
                );
                let got = output_and_grads(|x| bn.forward(x), &x, &[w, b]);
                same(got, want, &format!("batch_norm {p} {training}"));
                assert_equal(bn.running_mean.as_ref().unwrap(), &rm, "running_mean");
                assert_equal(bn.running_var.as_ref().unwrap(), &rv, "running_var");
            }
        }

        let pool = MaxPool2d::new(3).stride(2).padding(1);
        let options = MaxPool2dOptions {
            stride: Some([2, 2]),
            padding: [1, 1],
            ..MaxPool2dOptions::new([3, 3])
        };
        same(
            output_and_grads(|x| pool.forward(x), &x, &[]),
            output_and_grads(|x| functional::max_pool2d(x, options), &x, &[]),
            "max_pool2d",
        );
        for size in [[1, 1], [2, 3]] {
            let pool = AdaptiveAvgPool2d::new(size);
            same(
                output_and_grads(|x| pool.forward(x), &x, &[]),
                output_and_grads(|x| functional::adaptive_avg_pool2d(x, size), &x, &[]),
                "adaptive_avg_pool2d",
            );
        }
        same(
            output_and_grads(|x| Flatten::default().forward(x), &x, &[]),
            output_and_grads(|x| x.flatten(1, -1), &x, &[]),
            "flatten",
        );
    }
}

/// The CNN of M2 step 9, as `tools/golden/gen_m2_cnn_layers.py` builds it.
fn cnn() -> Sequential {
    Sequential::new()
        .append(Conv2d::new(3, 8, 3).padding(1))
        .append(BatchNorm2d::new(8))
        .append(Relu)
        .append(MaxPool2d::new(2))
        .append(Conv2d::new(8, 16, 3).padding(1))
        .append(AdaptiveAvgPool2d::new(1))
        .append(Flatten::default())
        .append(Linear::new(16, 4))
}

#[test]
fn cnn_matches_pytorch() {
    let golden = Golden::load("m2_cnn_layers.safetensors");
    let c = Case::new(&golden, "cnn_f64");
    let model = cnn();
    let bn_keys = BN_KEYS.map(|k| format!("1.{k}"));
    let mut want = vec!["0.weight", "0.bias"];
    want.extend(bn_keys.iter().map(String::as_str));
    want.extend(["4.weight", "4.bias", "7.weight", "7.bias"]);
    assert_eq!(keys(&model), want);
    model.to(DType::F64);
    model
        .load_state_dict(&golden_state(&c, "param"), true)
        .unwrap();
    let x = c.input("x");
    x.requires_grad_(true);
    let out = model.forward(&x);
    c.check("out", &out);
    let params = model.named_parameters();
    let mut inputs = vec![&x];
    inputs.extend(params.iter().map(|(_, p)| &**p));
    let g = grads(&out, &inputs, &c.input("g"));
    c.check("grad_x", g[0].as_ref().unwrap());
    for ((key, _), gk) in params.iter().zip(&g[1..]) {
        c.check(&format!("grad.{key}"), gk.as_ref().unwrap());
    }
    for (key, t) in model.state_dict().iter() {
        c.check(&format!("state.{key}"), t);
    }
    model.eval();
    c.check("out_eval", &model.forward(&c.input("x_eval")));
    c.assert_all_used();
}

#[test]
fn layers_pass_gradcheck() {
    let opts = GradcheckOptions::default;
    let x = var(&[2, 4, 4, 4], 0);
    for groups in [1, 2] {
        let conv = Conv2d::new(4, 2, [2, 3])
            .padding([1, 0])
            .stride([1, 2])
            .groups(groups);
        conv.to(DType::F64);
        let (w, b) = (&*conv.weight, conv.bias.as_deref().unwrap());
        let f = |t: &[Tensor]| vec![conv.forward(&t[0])];
        gradcheck(f, &[&x, w, b], opts()).unwrap();
        gradgradcheck(f, &[&x, w, b], opts()).unwrap();
    }
    for training in [true, false] {
        let bn = BatchNorm2d::new(4);
        bn.to(DType::F64);
        bn.training.set(training);
        bn.weight
            .as_ref()
            .unwrap()
            .detach()
            .copy_(&constant(&[4], 1));
        let (w, b) = (bn.weight.as_deref().unwrap(), bn.bias.as_deref().unwrap());
        // The running statistics are updated by every training pass; their values don't change
        // the gradients of a training pass.
        let f = |t: &[Tensor]| vec![bn.forward(&t[0])];
        gradcheck(f, &[&x, w, b], opts()).unwrap();
        gradgradcheck(f, &[&x, w, b], opts()).unwrap();
    }
    // Pooling: the elements of `constant` are distinct, so no window has a tie near the
    // perturbation.
    let pool = MaxPool2d::new(2).stride(1);
    let f = |t: &[Tensor]| vec![pool.forward(&t[0])];
    gradcheck(f, &[&x], opts()).unwrap();
    gradgradcheck(f, &[&x], opts()).unwrap();
    for size in [[1, 1], [3, 2]] {
        let pool = AdaptiveAvgPool2d::new(size);
        let f = |t: &[Tensor]| vec![pool.forward(&t[0])];
        gradcheck(f, &[&x], opts()).unwrap();
        gradgradcheck(f, &[&x], opts()).unwrap();
    }
    let f = |t: &[Tensor]| vec![Flatten::new(1, 2).forward(&t[0].transpose(1, 3))];
    gradcheck(f, &[&x], opts()).unwrap();
    gradgradcheck(f, &[&x], opts()).unwrap();
}

#[test]
fn layers_are_modules() {
    let model = cnn();
    // `to` converts the floating point members only.
    model.to(DType::F64);
    let dtypes: Vec<(String, DType)> = model
        .state_dict()
        .iter()
        .map(|(k, t)| (k.to_string(), t.dtype()))
        .collect();
    for (key, dtype) in &dtypes {
        let want = if key.ends_with("num_batches_tracked") {
            DType::I64
        } else {
            DType::F64
        };
        assert_eq!(*dtype, want, "{key}");
    }
    // `train` and `eval` reach the batch normalization through the `Sequential`.
    let x = constant(&[2, 3, 4, 4], 0);
    model.forward(&x);
    model.eval();
    model.forward(&x);
    let sd = model.state_dict();
    assert_eq!(sd.get("1.num_batches_tracked").unwrap().item::<i64>(), 1);
    model.train();
    model.forward(&x);
    assert_eq!(sd.get("1.num_batches_tracked").unwrap().item::<i64>(), 2);
    // State dicts go back and forth.
    let other = cnn();
    other.to(DType::F64);
    other.load_state_dict(&model.state_dict(), true).unwrap();
    other.eval();
    model.eval();
    assert_equal(&other.forward(&x), &model.forward(&x), "loaded");
    // Layers are `Forward`.
    let layers: Vec<Box<dyn Forward>> = vec![
        Box::new(Conv2d::new(3, 3, 1)),
        Box::new(BatchNorm2d::new(3)),
        Box::new(MaxPool2d::new(1)),
        Box::new(AdaptiveAvgPool2d::new(2)),
        Box::new(Flatten::default()),
    ];
    let x = homura::ones([2, 3, 4, 4]);
    let shapes: Vec<Vec<usize>> = layers
        .iter()
        .map(|l| l.forward(&x).shape().to_vec())
        .collect();
    assert_eq!(
        shapes,
        [
            vec![2, 3, 4, 4],
            vec![2, 3, 4, 4],
            vec![2, 3, 4, 4],
            vec![2, 3, 2, 2],
            vec![2, 48]
        ]
    );
}
