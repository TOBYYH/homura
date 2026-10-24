//! M2 step 4b: ten steps of training an MLP with `Sgd` and `AdamW`, compared step by step with
//! PyTorch: the loss, every gradient, every parameter after the update and the optimizer state.
//! The data, the initial weights and PyTorch's values are in
//! `tests/golden/m2_training.safetensors` from `tools/golden/gen_m2_training.py`;
//! `tools/golden/check_m2_training.py` checks the file against a fresh run.
//! `tools/golden/measure_m2_tolerances.py` measures how much of the tolerances the errors use
//! (with `HOMURA_TOLERANCE_LOG` set, this test writes its measurements there).

mod common;

use std::io::Write;

use common::{Case, Golden, assert_close_tol};
use homura::nn::functional::{CrossEntropyOptions, cross_entropy};
use homura::nn::{Linear, Module, Parameter, Relu, StateDict};
use homura::optim::{AdamOptions, AdamW, Optimizer, ParamGroup, Sgd, TypedOptimizer};
use homura::{DType, Tensor};

const STEPS: usize = 10;
const KEYS: [&str; 4] = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"];

/// The model of the golden file, PyTorch's `Mlp` of `gen_m2_training.py`.
#[derive(Module)]
struct Mlp {
    fc1: Linear,
    act: Relu,
    fc2: Linear,
}

impl Mlp {
    fn new() -> Mlp {
        Mlp {
            fc1: Linear::new(8, 32),
            act: Relu,
            fc2: Linear::new(32, 4),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc2.forward(&self.act.forward(&self.fc1.forward(x)))
    }
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
fn optimizer(kind: &str, case: &Case, model: &Mlp) -> Box<dyn CaseOptimizer> {
    let opt = |name: &str| case.input(&format!("opt.{name}")).item::<f64>();
    match kind {
        "sgd" => Box::new(
            Sgd::new(model.parameters(), opt("lr"))
                .momentum(opt("momentum"))
                .weight_decay(opt("weight_decay")),
        ),
        "adamw" => {
            let weights = vec![model.fc1.weight.clone(), model.fc2.weight.clone()];
            let bias = |l: &Linear| l.bias.clone().unwrap();
            let biases = vec![bias(&model.fc1), bias(&model.fc2)];
            let mut o = AdamW::new(weights, opt("lr")).weight_decay(opt("weight_decay"));
            let options = AdamOptions {
                weight_decay: opt("bias_weight_decay"),
                ..*o.defaults()
            };
            o.add_param_group(ParamGroup::new(biases, options));
            Box::new(o)
        }
        _ => unreachable!("{kind}"),
    }
}

/// `(rtol, atol)` of the comparisons. F64 is the bound of the plan; homura's values are within
/// 2.3e-16 of PyTorch's. In F32 they are within 1.2e-7 (one rounding step of the loss, whose
/// magnitude is about 1.5), the size of the distance between PyTorch's own F32 and F64 runs that
/// `check_m2_training.py` prints: the sums of the matrix products and reductions are taken in
/// another order. The bounds leave a factor of two or more.
fn tolerance(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::F64 => (1e-10, 1e-12),
        DType::F32 => (2e-7, 2e-7),
        _ => unreachable!(),
    }
}

/// Appends the largest errors of `actual` against `expected` (absolute, relative to nonzero
/// expected values, relative to the largest expected magnitude, and the fraction of the
/// allowance `atol + rtol·|expected|` used) as a line of JSON to the file that
/// `HOMURA_TOLERANCE_LOG` names, if it is set.
fn log_measurement(case: &str, tensor: &str, actual: &Tensor, expected: &Tensor, tol: (f64, f64)) {
    let Ok(path) = std::env::var("HOMURA_TOLERANCE_LOG") else {
        return;
    };
    let a = actual.to(DType::F64).to_vec::<f64>();
    let e = expected.to(DType::F64).to_vec::<f64>();
    let scale = e.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
    let (mut abs, mut rel, mut used) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(&e) {
        let error = (x - y).abs();
        abs = abs.max(error);
        if *y != 0.0 {
            rel = rel.max(error / y.abs());
        }
        used = used.max(error / (tol.1 + tol.0 * y.abs()));
    }
    let norm = if scale > 0.0 { abs / scale } else { 0.0 };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("can't open the tolerance log");
    writeln!(
        file,
        r#"{{"case": "{case}", "tensor": "{tensor}", "abs": {abs:e}, "rel": {rel:e}, "norm": {norm:e}, "used": {used:e}}}"#
    )
    .unwrap();
}

#[test]
fn mlp_training_matches_pytorch() {
    let golden = Golden::load("m2_training.safetensors");
    let data = Case::new(&golden, "data");
    for kind in ["sgd", "adamw"] {
        for (tag, dtype) in [("f64", DType::F64), ("f32", DType::F32)] {
            let name = format!("{kind}_{tag}");
            let c = Case::new(&golden, &name);
            let (rtol, atol) = tolerance(dtype);
            let check = |key: &str, t: &Tensor| {
                c.check_with(key, t, |a, e| {
                    assert_close_tol(a, e, rtol, atol, &format!("{name}/{key}"));
                    log_measurement(&name, key, a, e, (rtol, atol));
                })
            };

            let model = Mlp::new();
            model.to(dtype);
            let init: StateDict = KEYS
                .iter()
                .map(|k| (k.to_string(), data.input(&format!("init.{k}")).to(dtype)))
                .collect();
            model.load_state_dict(&init, true).unwrap();
            let params = model.named_parameters();
            let keys: Vec<&str> = params.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(keys, KEYS, "PyTorch's parameter order");

            let mut opt = optimizer(kind, &c, &model);
            for t in 0..STEPS {
                opt.zero_grad();
                assert!(params.iter().all(|(_, p)| p.grad().is_none()));
                let x = data.input(&format!("x{t}")).to(dtype);
                let target = data.input(&format!("target{t}"));
                let loss =
                    cross_entropy(&model.forward(&x), &target, CrossEntropyOptions::default());
                check(&format!("loss{t}"), &loss);
                loss.backward();
                for (k, p) in &params {
                    check(&format!("grad{t}.{k}"), &p.grad().unwrap());
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
            }
            c.assert_all_used();
        }
    }
    data.assert_all_used();
}
