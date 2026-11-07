//! M4 step 3: the optimizers' steps on a ROCm GPU. The multi-step and edge-value cases of M2
//! step 4a (`tests/golden/m2_optim*.safetensors`, which the CPU matches bit for bit) run with the
//! parameters, gradients and state on the GPU and must give the same bits; the step counts of
//! Adam and AdamW stay on the CPU, as in PyTorch without `capturable`.
#![cfg(feature = "rocm")]

mod common;
#[path = "common/rocm.rs"]
mod rocm;

use common::{Case, Golden};
use homura::nn::Parameter;
use homura::optim::{
    Adam, AdamOptions, AdamState, AdamW, Optimizer, ParamGroup, Sgd, SgdOptions, SgdState,
    TypedOptimizer,
};
use homura::{DType, Device, Tensor};
use rocm::GPU;

const STEPS: usize = 5;
const PARAMS: [&str; 3] = ["a", "b", "c"];

fn values(t: &Tensor) -> Vec<f64> {
    t.to((DType::F64, Device::Cpu)).to_vec::<f64>()
}

/// Asserts that `actual` has the dtype, shape and bits of `expected` (`-0.0` isn't `0.0`, NaNs
/// must be in the same places): the optimizers compute what PyTorch's vectorized CPU kernels do.
fn assert_same(actual: &Tensor, expected: &Tensor, what: &str) {
    assert_eq!(actual.dtype(), expected.dtype(), "{what}: dtype");
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    let (a, e) = (values(actual), values(expected));
    for (i, (x, y)) in a.iter().zip(&e).enumerate() {
        let same = x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
        assert!(same, "{what}: element {i} is {x:e}, PyTorch's is {y:e}");
    }
}

/// The optimizers of the golden file, built from a case's options and driven through
/// `dyn Optimizer`; their state is read through the concrete type.
trait CaseOptimizer: Optimizer {
    /// The state tensors of `p`, by PyTorch's names.
    fn state(&self, p: &Parameter) -> Vec<(&'static str, Tensor)>;
}

/// The state tensors of an Adam or AdamW parameter.
fn adam_tensors(s: Option<&AdamState>) -> Vec<(&'static str, Tensor)> {
    s.map(|s| {
        let mut out = vec![
            ("step", s.step().clone()),
            ("exp_avg", s.exp_avg().clone()),
            ("exp_avg_sq", s.exp_avg_sq().clone()),
        ];
        if let Some(m) = s.max_exp_avg_sq() {
            out.push(("max_exp_avg_sq", m.clone()));
        }
        out
    })
    .unwrap_or_default()
}

impl CaseOptimizer for Sgd {
    fn state(&self, p: &Parameter) -> Vec<(&'static str, Tensor)> {
        Sgd::state(self, p)
            .map(|s: &SgdState| vec![("momentum_buffer", s.momentum_buffer().clone())])
            .unwrap_or_default()
    }
}

impl CaseOptimizer for Adam {
    fn state(&self, p: &Parameter) -> Vec<(&'static str, Tensor)> {
        adam_tensors(Adam::state(self, p))
    }
}

impl CaseOptimizer for AdamW {
    fn state(&self, p: &Parameter) -> Vec<(&'static str, Tensor)> {
        adam_tensors(AdamW::state(self, p))
    }
}

/// The optimizer of a case, with the group `first` and, if given, a group `second` with twice
/// the learning rate.
fn build(
    kind: &str,
    case: &Case,
    first: Vec<Parameter>,
    second: Option<Parameter>,
) -> Box<dyn CaseOptimizer> {
    let opt = |name: &str| case.input(&format!("opt.{name}")).item::<f64>();
    let flag = |name: &str| opt(name) != 0.0;
    match kind {
        "sgd" => {
            let mut o = Sgd::new(first, opt("lr"))
                .momentum(opt("momentum"))
                .dampening(opt("dampening"))
                .weight_decay(opt("weight_decay"))
                .nesterov(flag("nesterov"))
                .maximize(flag("maximize"));
            let options = SgdOptions {
                lr: 2.0 * opt("lr"),
                ..*o.defaults()
            };
            if second.is_some() {
                o.add_param_group(ParamGroup::new(second, options));
            }
            Box::new(o)
        }
        "adam" | "adamw" => {
            let betas = (opt("beta1"), opt("beta2"));
            let options = |defaults: &AdamOptions| AdamOptions {
                lr: 2.0 * opt("lr"),
                ..*defaults
            };
            macro_rules! adam {
                ($t:ident) => {{
                    let mut o = $t::new(first, opt("lr"))
                        .betas(betas)
                        .eps(opt("eps"))
                        .weight_decay(opt("weight_decay"))
                        .amsgrad(flag("amsgrad"))
                        .maximize(flag("maximize"));
                    if second.is_some() {
                        let group = ParamGroup::new(second.clone(), options(o.defaults()));
                        o.add_param_group(group);
                    }
                    o
                }};
            }
            if kind == "adam" {
                Box::new(adam!(Adam))
            } else {
                Box::new(adam!(AdamW))
            }
        }
        _ => unreachable!("{kind}"),
    }
}

/// Asserts where a state tensor lives: the step count on the CPU, the others with the parameter.
fn check_device(name: &str, s: &str, value: &Tensor) {
    let expected = if s == "step" { Device::Cpu } else { GPU };
    assert_eq!(value.device(), expected, "{name}: device of {s}");
}

#[test]
fn multi_step_golden_on_the_gpu() {
    let golden = Golden::load("m2_optim.safetensors");
    let cases = golden.cases();
    assert_eq!(cases.len(), 4 * 10);
    for name in &cases {
        let case = Case::new(&golden, name);
        let (prefix, _) = name.split_once('.').unwrap();
        let (kind, _) = prefix.split_once('_').unwrap();
        let params: Vec<Parameter> = PARAMS
            .iter()
            .map(|p| Parameter::new(case.input(&format!("init.{p}")).to(GPU)))
            .collect();
        let first = vec![params[0].clone(), params[2].clone()];
        let mut opt = build(kind, &case, first, Some(params[1].clone()));
        for t in 0..STEPS {
            if t == 2 {
                opt.set_lr(0, opt.lr(0) * 0.5);
            }
            for (p, n) in params.iter().zip(PARAMS) {
                let key = format!("grad{t}.{n}");
                let has = golden.names(name).contains(&key);
                p.set_grad(has.then(|| case.input(&key).to(GPU)));
            }
            opt.step();
            for (p, n) in params.iter().zip(PARAMS) {
                assert_eq!(p.device(), GPU);
                let compare = |what: String, actual: &Tensor| {
                    case.check_with(&what, actual, |a, e| {
                        assert_same(a, e, &format!("{name}/{what}"));
                    });
                };
                compare(format!("after{t}.{n}"), p);
                for (s, value) in opt.state(p) {
                    check_device(name, s, &value);
                    compare(format!("after{t}.{n}.{s}"), &value);
                }
            }
        }
        case.assert_all_used();
    }
}

#[test]
fn edge_values_golden_on_the_gpu() {
    let golden = Golden::load("m2_optim_edges.safetensors");
    let cases = golden.cases();
    assert_eq!(cases.len(), 18);
    for name in &cases {
        let case = Case::new(&golden, name);
        let (prefix, _) = name.split_once('.').unwrap();
        let (kind, _) = prefix.split_once('_').unwrap();
        let p = Parameter::new(case.input("init.p").to(GPU));
        let mut opt = build(kind, &case, vec![p.clone()], None);
        let names = golden.names(name);
        for t in 0.. {
            let key = format!("grad{t}.p");
            if !names.contains(&key) {
                break;
            }
            p.set_grad(Some(case.input(&key).to(GPU)));
            opt.step();
            case.check_with(&format!("after{t}.p"), &p, |a, e| {
                assert_same(a, e, &format!("{name}/after{t}.p"))
            });
            for (s, value) in opt.state(&p) {
                check_device(name, s, &value);
                case.check_with(&format!("after{t}.p.{s}"), &value, |a, e| {
                    assert_same(a, e, &format!("{name}/after{t}.p.{s}"))
                });
            }
        }
        case.assert_all_used();
    }
}

fn panic_message(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("a panic");
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// R66 of review/M2.md on the GPU (the device migration table's optimizer row, M4.3): a step
/// rejects the coefficients that overflow their type with the CPU's message, before updating
/// anything, for GPU parameters too.
#[test]
fn coefficients_that_overflow_are_rejected_on_the_gpu() {
    let overflow =
        |dtype: DType| format!("value cannot be converted to type {dtype} without overflow");
    for device in [Device::Cpu, GPU] {
        let ones = |dtype: DType| Parameter::new(homura::ones_with([32], (dtype, device)));
        let with_grad =
            |p: &Parameter| p.set_grad(Some(homura::ones_with([32], (p.dtype(), device))));
        for (dtype, big) in [(DType::F16, 1e5), (DType::BF16, 1e40), (DType::F32, 1e40)] {
            let p = ones(dtype);
            with_grad(&p);
            let mut opt = Sgd::new([p.clone()], big).momentum(0.9);
            assert_eq!(
                panic_message(|| opt.step()),
                overflow(dtype),
                "{device} {dtype}"
            );
            assert!(values(&p).iter().all(|&v| v == 1.0));
            assert!(opt.state(&p).is_none());
        }
        for dtype in [DType::F32, DType::F16] {
            let p = ones(dtype);
            with_grad(&p);
            let mut adam = Adam::new([p.clone()], 1e38);
            assert_eq!(
                panic_message(|| adam.step()),
                overflow(DType::F32),
                "{device} {dtype}"
            );
            assert!(values(&p).iter().all(|&v| v == 1.0));
            let mut adamw = AdamW::new([p.clone()], 3e38).weight_decay(0.0);
            assert_eq!(
                panic_message(|| adamw.step()),
                overflow(DType::F32),
                "{device} {dtype}"
            );
            assert!(values(&p).iter().all(|&v| v == 1.0));
        }
    }
}
