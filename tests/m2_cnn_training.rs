//! M2 step 9b: ten steps of training the small CNN of step 9 with `Sgd` and `AdamW`, compared
//! step by step with PyTorch: the batch norm output (the ReLU input) and the max pooling
//! indices, the loss, every gradient, every parameter after the update, the optimizer state and
//! the batch norm buffers, then an evaluation pass. The data, the initial weights and PyTorch's
//! values are in `tests/golden/m2_cnn_training.safetensors` from
//! `tools/golden/gen_m2_cnn_training.py`; `tools/golden/check_m2_cnn_training.py` checks the
//! file against a fresh run. `tools/golden/measure_m2_tolerances.py` measures how much of the
//! tolerances the errors use (with `HOMURA_TOLERANCE_LOG` set, this test writes its
//! measurements there).

mod common;

use std::io::Write;

use common::{Case, Golden, assert_close_tol};
use homura::nn::functional::{
    CrossEntropyOptions, MaxPool2dOptions, cross_entropy, max_pool2d_with_indices,
};
use homura::nn::{
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, Module, Parameter, Relu,
    Sequential, StateDict,
};
use homura::optim::{AdamOptions, AdamW, Optimizer, ParamGroup, Sgd, TypedOptimizer};
use homura::{DType, Tensor};

const STEPS: usize = 10;
const KEYS: [&str; 8] = [
    "0.weight", "0.bias", "1.weight", "1.bias", "4.weight", "4.bias", "7.weight", "7.bias",
];
const BUFFERS: [&str; 3] = ["1.running_mean", "1.running_var", "1.num_batches_tracked"];
/// The parameters of AdamW's first group, with weight decay; the others are in the second.
const WEIGHTS: [&str; 3] = ["0.weight", "4.weight", "7.weight"];
/// The bias of the first convolution doesn't require grad: the batch norm after it subtracts the
/// mean of each channel, so its gradient is zero and its computed value rounding noise (see
/// `gen_m2_cnn_training.py`). The optimizers hold it and skip it.
const FROZEN: &str = "0.bias";

/// The model of the golden file, PyTorch's `cnn` of `gen_m2_cnn_training.py`.
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

/// The output of `model` for `x`, layer by layer like `Sequential::forward`, with the batch
/// norm output and the indices of the max pooling of the ReLU output.
fn forward(model: &Sequential, x: &Tensor) -> (Tensor, Tensor, Tensor) {
    let (mut h, mut pre, mut indices) = (x.clone(), None, None);
    for (i, layer) in model.iter().enumerate() {
        h = layer.forward(&h);
        match i {
            1 => pre = Some(h.clone()),
            2 => indices = Some(max_pool2d_with_indices(&h, MaxPool2dOptions::new([2, 2])).1),
            _ => {}
        }
    }
    (h, pre.unwrap(), indices.unwrap())
}

/// The optimizer of a case, driven through `dyn Optimizer`, with its state read through the
/// concrete type.
trait CaseOptimizer: Optimizer {
    /// The state tensors of `p`, by PyTorch's names.
    fn state(&self, p: &Parameter) -> Vec<(&'static str, Tensor)>;
}

impl CaseOptimizer for Sgd {
    fn state(&self, p: &Parameter) -> Vec<(&'static str, Tensor)> {
        Sgd::state(self, p)
            .map(|s| vec![("momentum_buffer", s.momentum_buffer().clone())])
            .unwrap_or_default()
    }
}

impl CaseOptimizer for AdamW {
    fn state(&self, p: &Parameter) -> Vec<(&'static str, Tensor)> {
        AdamW::state(self, p)
            .map(|s| {
                assert!(s.max_exp_avg_sq().is_none());
                vec![
                    ("step", s.step().clone()),
                    ("exp_avg", s.exp_avg().clone()),
                    ("exp_avg_sq", s.exp_avg_sq().clone()),
                ]
            })
            .unwrap_or_default()
    }
}

/// The optimizer of a case, with its options from the file.
fn optimizer(kind: &str, case: &Case, params: &[(String, Parameter)]) -> Box<dyn CaseOptimizer> {
    let opt = |name: &str| case.input(&format!("opt.{name}")).item::<f64>();
    let all = || params.iter().map(|(_, p)| p.clone());
    match kind {
        "sgd" => Box::new(
            Sgd::new(all(), opt("lr"))
                .momentum(opt("momentum"))
                .weight_decay(opt("weight_decay")),
        ),
        "adamw" => {
            let (weights, others): (Vec<_>, Vec<_>) = params
                .iter()
                .partition(|(k, _)| WEIGHTS.contains(&k.as_str()));
            let parameters = |group: Vec<&(String, Parameter)>| {
                group
                    .into_iter()
                    .map(|(_, p)| p.clone())
                    .collect::<Vec<_>>()
            };
            let mut o =
                AdamW::new(parameters(weights), opt("lr")).weight_decay(opt("weight_decay"));
            let options = AdamOptions {
                weight_decay: opt("other_weight_decay"),
                ..*o.defaults()
            };
            o.add_param_group(ParamGroup::new(parameters(others), options));
            Box::new(o)
        }
        _ => unreachable!("{kind}"),
    }
}

/// `(rtol, atol)` of the comparisons: an element may differ by `rtol·|expected| + atol·m`, where
/// `m` is the largest magnitude of the expected tensor (gradients and optimizer states have
/// elements near zero whose errors come from the larger terms they are computed from). F64 is
/// the bound of the MLP's test; homura's errors are below 5e-15·m. In F32 they are below 3.1e-6·m
/// over the ten steps (the gradient of the second convolution's bias), as large as the distance
/// between PyTorch's own F32 and F64 runs (3.4e-6·m); `measure_m2_tolerances.py` reports both.
/// The bound leaves a factor of three.
fn tolerance(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::F64 => (1e-10, 1e-12),
        DType::F32 => (1e-6, 1e-5),
        _ => unreachable!(),
    }
}

/// The largest magnitude of `t`, at least the smallest normal number.
fn scale(t: &Tensor) -> f64 {
    t.to(DType::F64)
        .to_vec::<f64>()
        .iter()
        .fold(f64::MIN_POSITIVE, |m, v| m.max(v.abs()))
}

/// Appends the largest errors of `actual` against `expected` (absolute, relative to nonzero
/// expected values, relative to the largest expected magnitude, and the fraction of the
/// allowance used) as a line of JSON to the file that `HOMURA_TOLERANCE_LOG` names, if it is
/// set.
fn log_measurement(case: &str, tensor: &str, actual: &Tensor, expected: &Tensor, tol: (f64, f64)) {
    let Ok(path) = std::env::var("HOMURA_TOLERANCE_LOG") else {
        return;
    };
    let a = actual.to(DType::F64).to_vec::<f64>();
    let e = expected.to(DType::F64).to_vec::<f64>();
    let m = scale(expected);
    let (mut abs, mut rel, mut used) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(&e) {
        let error = (x - y).abs();
        abs = abs.max(error);
        if *y != 0.0 {
            rel = rel.max(error / y.abs());
        }
        used = used.max(error / (tol.0 * y.abs() + tol.1 * m));
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("can't open the tolerance log");
    writeln!(
        file,
        r#"{{"case": "{case}", "tensor": "{tensor}", "abs": {abs:e}, "rel": {rel:e}, "norm": {:e}, "used": {used:e}}}"#,
        abs / m
    )
    .unwrap();
}

/// The entries of the case `data` named `init.<key>`, rounded to `dtype` (the count stays I64).
fn initial_state(data: &Case, dtype: DType) -> StateDict {
    KEYS.iter()
        .chain(&BUFFERS)
        .map(|k| {
            let t = data.input(&format!("init.{k}"));
            let t = if t.is_floating_point() {
                t.to(dtype)
            } else {
                t
            };
            (k.to_string(), t)
        })
        .collect()
}

#[test]
fn cnn_training_matches_pytorch() {
    let golden = Golden::load("m2_cnn_training.safetensors");
    let data = Case::new(&golden, "data");
    for kind in ["sgd", "adamw"] {
        for (tag, dtype) in [("f64", DType::F64), ("f32", DType::F32)] {
            let name = format!("{kind}_{tag}");
            let c = Case::new(&golden, &name);
            let (rtol, atol) = tolerance(dtype);
            let check = |key: &str, t: &Tensor| {
                c.check_with(key, t, |a, e| {
                    assert_close_tol(a, e, rtol, atol * scale(e), &format!("{name}/{key}"));
                    log_measurement(&name, key, a, e, (rtol, atol));
                })
            };
            // The ReLU and the max pooling select the same elements as PyTorch's.
            let check_selection = |suffix: &str, pre: &Tensor, indices: &Tensor| {
                check(&format!("pre{suffix}"), pre);
                let positive = |t: &Tensor| t.greater(0.0).to_vec::<bool>();
                let expected = c.input(&format!("pre{suffix}"));
                assert_eq!(positive(pre), positive(&expected), "{name}: ReLU {suffix}");
                c.check(&format!("indices{suffix}"), indices);
            };

            let model = cnn();
            model.to(dtype);
            model
                .load_state_dict(&initial_state(&data, dtype), true)
                .unwrap();
            let params = model.named_parameters();
            params[1].1.requires_grad_(false);
            let keys: Vec<&str> = params.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(keys, KEYS, "PyTorch's parameter order");
            let buffers: Vec<String> = model.named_buffers().into_iter().map(|(k, _)| k).collect();
            assert_eq!(buffers, BUFFERS);

            let mut opt = optimizer(kind, &c, &params);
            for t in 0..STEPS {
                opt.zero_grad();
                assert!(params.iter().all(|(_, p)| p.grad().is_none()));
                let x = data.input(&format!("x{t}")).to(dtype);
                let target = data.input(&format!("target{t}"));
                let (logits, pre, indices) = forward(&model, &x);
                check_selection(&t.to_string(), &pre, &indices);
                let loss = cross_entropy(&logits, &target, CrossEntropyOptions::default());
                check(&format!("loss{t}"), &loss);
                loss.backward();
                for (k, p) in &params {
                    match p.grad() {
                        Some(g) => check(&format!("grad{t}.{k}"), &g),
                        None => assert_eq!(k, FROZEN),
                    }
                }

                opt.step();
                for (k, p) in &params {
                    assert!(p.is_leaf() && p.grad_fn().is_none());
                    check(&format!("after{t}.{k}"), p);
                    let state = opt.state(p);
                    let mut names: Vec<String> = state
                        .iter()
                        .map(|(s, _)| format!("after{t}.{k}.{s}"))
                        .collect();
                    let mut expected: Vec<String> = golden
                        .names(&name)
                        .into_iter()
                        .filter(|n| n.starts_with(&format!("after{t}.{k}.")))
                        .collect();
                    names.sort();
                    expected.sort();
                    assert_eq!(names, expected, "{name}: the state of {k}");
                    for (s, v) in &state {
                        assert!(!v.requires_grad() && v.grad_fn().is_none());
                        check(&format!("after{t}.{k}.{s}"), v);
                    }
                }
                for (k, b) in model.named_buffers() {
                    assert!(!b.requires_grad());
                    check(&format!("after{t}.{k}"), &b);
                }
            }

            model.eval();
            let x = data.input("x_eval").to(dtype);
            let (out, pre, indices) = {
                let _no_grad = homura::no_grad();
                forward(&model, &x)
            };
            check_selection("_eval", &pre, &indices);
            check("out_eval", &out);
            // Sequential's forward computes the same, and evaluation passes don't count.
            let count = |m: &Sequential| m.state_dict().get(BUFFERS[2]).unwrap().item::<i64>();
            assert_eq!(count(&model), STEPS as i64);
            assert_close_tol(&model.forward(&x), &out, 0.0, 0.0, "Sequential::forward");
            assert_eq!(count(&model), STEPS as i64);
            c.assert_all_used();
        }
    }
    data.assert_all_used();
}
