//! M2 step 4a: the optimizers of `homura::optim`. Several steps of each optimizer are compared
//! bit for bit with `tests/golden/m2_optim.safetensors` and `m2_optim_edges.safetensors` from
//! `tools/golden/gen_m2_optim.py`; the reference behavior encoded in the other tests is asserted
//! by `tools/golden/check_m2_optim.py`, in the same order.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden};
use homura::nn::{Module, Parameter};
use homura::optim::{
    Adam, AdamOptions, AdamState, AdamW, Optimizer, ParamGroup, Sgd, SgdOptions, SgdState,
};
use homura::{DType, Device, Tensor};

const STEPS: usize = 5;
const PARAMS: [&str; 3] = ["a", "b", "c"];

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn expect_panic(f: impl FnOnce(), message: &str) {
    assert_eq!(panic_message(f), message);
}

fn param(values: &[f64]) -> Parameter {
    Parameter::new(homura::tensor_with(values.to_vec(), DType::F64))
}

fn grad(p: &Parameter, values: &[f64]) {
    p.set_grad(Some(homura::tensor_with(values.to_vec(), p.dtype())));
}

fn values(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
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

/// The optimizers of the golden file, built from a case's options.
enum Any {
    Sgd(Sgd),
    Adam(Adam),
    AdamW(AdamW),
}

impl Any {
    fn step(&mut self) {
        match self {
            Any::Sgd(o) => o.step(),
            Any::Adam(o) => o.step(),
            Any::AdamW(o) => o.step(),
        }
    }

    fn halve_first_lr(&mut self) {
        match self {
            Any::Sgd(o) => o.options_mut(0).lr *= 0.5,
            Any::Adam(o) => o.options_mut(0).lr *= 0.5,
            Any::AdamW(o) => o.options_mut(0).lr *= 0.5,
        }
    }

    /// The state tensors of `p`, by PyTorch's names.
    fn state(&self, p: &Parameter) -> Vec<(&'static str, Tensor)> {
        let adam = |s: Option<&AdamState>| {
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
        };
        match self {
            Any::Sgd(o) => o
                .state(p)
                .map(|s: &SgdState| vec![("momentum_buffer", s.momentum_buffer().clone())])
                .unwrap_or_default(),
            Any::Adam(o) => adam(o.state(p)),
            Any::AdamW(o) => adam(o.state(p)),
        }
    }
}

/// The optimizer of a case, with the group `first` and, if given, a group `second` with twice
/// the learning rate.
fn build(kind: &str, case: &Case, first: Vec<Parameter>, second: Option<Parameter>) -> Any {
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
            Any::Sgd(o)
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
                Any::Adam(adam!(Adam))
            } else {
                Any::AdamW(adam!(AdamW))
            }
        }
        _ => unreachable!("{kind}"),
    }
}

#[test]
fn multi_step_golden() {
    let golden = Golden::load("m2_optim.safetensors");
    let cases = golden.cases();
    assert_eq!(cases.len(), 4 * 10);
    for name in &cases {
        let case = Case::new(&golden, name);
        let (prefix, _) = name.split_once('.').unwrap();
        let (kind, _) = prefix.split_once('_').unwrap();
        let params: Vec<Parameter> = PARAMS
            .iter()
            .map(|p| Parameter::new(case.input(&format!("init.{p}"))))
            .collect();
        let first = vec![params[0].clone(), params[2].clone()];
        let mut opt = build(kind, &case, first, Some(params[1].clone()));
        for t in 0..STEPS {
            if t == 2 {
                opt.halve_first_lr();
            }
            for (p, n) in params.iter().zip(PARAMS) {
                let key = format!("grad{t}.{n}");
                let has = golden.names(name).contains(&key);
                p.set_grad(has.then(|| case.input(&key)));
            }
            opt.step();
            for (p, n) in params.iter().zip(PARAMS) {
                let compare = |what: String, actual: &Tensor| {
                    case.check_with(&what, actual, |a, e| {
                        assert_same(a, e, &format!("{name}/{what}"));
                    });
                };
                compare(format!("after{t}.{n}"), p);
                let state = opt.state(p);
                let stored: Vec<String> = golden
                    .names(name)
                    .into_iter()
                    .filter(|k| k.starts_with(&format!("after{t}.{n}.")))
                    .collect();
                assert_eq!(
                    state.len(),
                    stored.len(),
                    "{name}: state of {n} after step {t}"
                );
                for (s, value) in state {
                    assert!(!value.requires_grad() && value.grad_fn().is_none());
                    compare(format!("after{t}.{n}.{s}"), &value);
                }
            }
        }
        case.assert_all_used();
    }
}

/// R65 of review/M2.md: a step computes each of PyTorch's compound operations in one go, so
/// intermediate products neither overflow nor vanish, also for tiny, subnormal and cancelling
/// values, with existing state, and for lengths that end in PyTorch's scalar loop (F32 / F64).
#[test]
fn edge_values_golden() {
    let golden = Golden::load("m2_optim_edges.safetensors");
    let cases = golden.cases();
    assert_eq!(cases.len(), 18);
    for name in &cases {
        let case = Case::new(&golden, name);
        let (prefix, _) = name.split_once('.').unwrap();
        let (kind, _) = prefix.split_once('_').unwrap();
        let p = Parameter::new(case.input("init.p"));
        let mut opt = build(kind, &case, vec![p.clone()], None);
        let names = golden.names(name);
        for t in 0.. {
            let key = format!("grad{t}.p");
            if !names.contains(&key) {
                break;
            }
            p.set_grad(Some(case.input(&key)));
            opt.step();
            case.check_with(&format!("after{t}.p"), &p, |a, e| {
                assert_same(a, e, &format!("{name}/after{t}.p"))
            });
            let state = opt.state(&p);
            let stored = names
                .iter()
                .filter(|k| k.starts_with(&format!("after{t}.p.")))
                .count();
            assert_eq!(state.len(), stored, "{name}: state after step {t}");
            for (s, value) in state {
                case.check_with(&format!("after{t}.p.{s}"), &value, |a, e| {
                    assert_same(a, e, &format!("{name}/after{t}.p.{s}"))
                });
            }
        }
        case.assert_all_used();
    }

    // The reviewer's examples, spelled out: one step from a single value.
    let step = |kind: &str, dtype: DType, p: f64, g: f64, lr: f64, eps: f64| {
        let param = Parameter::new(homura::full_with([32], p, dtype));
        param.set_grad(Some(homura::full_with([32], g, dtype)));
        let ps = [param.clone()];
        match kind {
            "sgd" => Sgd::new(ps, lr).step(),
            "adam" => Adam::new(ps, lr).eps(eps).step(),
            _ => AdamW::new(ps, lr).eps(eps).step(),
        }
        values(&param)
    };
    // -lr / bc1 * exp_avg is 1e-7, below F16's smallest normal, before the division.
    for kind in ["adam", "adamw"] {
        let p = step(kind, DType::F16, 0.0, 1e-5, 1e-3, 1e-4);
        assert!(
            p.iter().all(|&v| v == -0.00010132789611816406),
            "{kind}: {p:?}"
        );
    }
    // lr * grad overflows, p - lr * grad doesn't.
    let p = step("sgd", DType::F16, 60000.0, 2.0, 60000.0, 0.0);
    assert!(p.iter().all(|&v| v == -60000.0), "{p:?}");
    for dtype in [DType::BF16, DType::F32] {
        let p = step("sgd", dtype, 3e38, 2.0, 3e38, 0.0);
        assert!(
            p.iter().all(|&v| v.is_finite() && v < -2.9e38),
            "{dtype}: {p:?}"
        );
    }
    let p = step("sgd", DType::F64, 1e308, 2.0, 1e308, 0.0);
    assert!(p.iter().all(|&v| v == -1e308), "{p:?}");
}

/// R66 of review/M2.md: like PyTorch's CPU kernels, a step rejects a finite `alpha` of a scaled
/// addition that doesn't fit the parameter's dtype, and a finite `value` of the update's addcdiv
/// that doesn't fit its computation type, here before it updates anything. Infinities, NaN and
/// the scalars of other operations pass, and the meta device checks nothing.
#[test]
fn coefficients_that_overflow_their_type_are_rejected() {
    let ones = |dtype: DType| Parameter::new(homura::ones_with([32], dtype));
    let overflow =
        |dtype: DType| format!("value cannot be converted to type {dtype} without overflow");
    // A step that panics with `message` and leaves the parameters `ps` as they were.
    fn rejected(opt: &mut dyn FnMut(), ps: &[&Parameter], message: &str) {
        let before: Vec<Vec<f64>> = ps.iter().map(|p| values(p)).collect();
        assert_eq!(panic_message(AssertUnwindSafe(opt)), message);
        for (p, b) in ps.iter().zip(before) {
            assert_eq!(values(p), b);
        }
    }
    let with_grad = |p: &Parameter, g: f64| p.set_grad(Some(homura::full_with([32], g, p.dtype())));

    // SGD: -lr and weight_decay in the parameter's dtype, also with a zero gradient.
    for (dtype, big) in [(DType::F16, 1e5), (DType::BF16, 1e40), (DType::F32, 1e40)] {
        for (lr, weight_decay, g) in [(big, 0.0, 1.0), (0.1, big, 1.0), (big, 0.0, 0.0)] {
            let p = ones(dtype);
            with_grad(&p, g);
            let mut opt = Sgd::new([p.clone()], lr)
                .weight_decay(weight_decay)
                .momentum(0.9);
            rejected(&mut || opt.step(), &[&p], &overflow(dtype));
            assert!(opt.state(&p).is_none());
        }
    }
    // Nesterov's momentum is an alpha; the dampening's only once there is a buffer.
    let p = ones(DType::F16);
    with_grad(&p, 1.0);
    let mut opt = Sgd::new([p.clone()], 0.1).momentum(1e5).nesterov(true);
    rejected(&mut || opt.step(), &[&p], &overflow(DType::F16));
    let mut opt = Sgd::new([p.clone()], 0.1).momentum(0.9).dampening(-1e5);
    opt.step();
    let buf = values(opt.state(&p).unwrap().momentum_buffer());
    rejected(&mut || opt.step(), &[&p], &overflow(DType::F16));
    assert_eq!(values(opt.state(&p).unwrap().momentum_buffer()), buf);
    // A valid first group isn't updated when a scalar of the second overflows: its lr, weight
    // decay, Nesterov momentum or, at the second step, dampening.
    let second: [fn(&mut SgdOptions); 4] = [
        |o| o.lr = 1e5,
        |o| o.weight_decay = 1e5,
        |o| (o.momentum, o.nesterov) = (1e5, true),
        |o| o.dampening = -1e5,
    ];
    for (i, change) in second.into_iter().enumerate() {
        let (a, b) = (ones(DType::F16), ones(DType::F16));
        with_grad(&a, 1.0);
        with_grad(&b, 1.0);
        let mut opt = Sgd::new([a.clone()], 0.1).momentum(0.9);
        let mut options = *opt.defaults();
        change(&mut options);
        opt.add_param_group(ParamGroup::new([b.clone()], options));
        if i == 3 {
            opt.step();
        }
        let buf = opt.state(&a).map(|s| values(s.momentum_buffer()));
        rejected(&mut || opt.step(), &[&a, &b], &overflow(DType::F16));
        assert_eq!(opt.state(&a).map(|s| values(s.momentum_buffer())), buf);
    }

    // Adam / AdamW: lr fits F32, lr / (1 - beta1) doesn't; F16 computes the update in F32.
    for dtype in [DType::F32, DType::F16] {
        let p = ones(dtype);
        with_grad(&p, 1.0);
        let mut adam = Adam::new([p.clone()], 1e38);
        rejected(&mut || adam.step(), &[&p], &overflow(DType::F32));
        assert!(adam.state(&p).is_none());
        let mut adamw = AdamW::new([p.clone()], 3e38).weight_decay(0.0);
        rejected(&mut || adamw.step(), &[&p], &overflow(DType::F32));
    }
    // Later steps check the step size of that step: at step 2 with beta1 = 0.5, it is
    // lr / 0.75, which fits F32 for lr = 2.4e38 (lr / 0.5 wouldn't) and not for lr = 2.7e38.
    for (lr, fits) in [(2.4e38, true), (2.7e38, false)] {
        let p = ones(DType::F32);
        with_grad(&p, 1.0);
        let mut opt = Adam::new([p.clone()], 0.1);
        opt.step();
        let options = opt.options_mut(0);
        options.lr = lr;
        options.betas.0 = 0.5;
        if fits {
            opt.step();
            assert_eq!(opt.state(&p).unwrap().step().item::<f32>(), 2.0);
        } else {
            rejected(&mut || opt.step(), &[&p], &overflow(DType::F32));
            assert_eq!(opt.state(&p).unwrap().step().item::<f32>(), 1.0);
        }
    }
    // Adam's coupled weight decay is an alpha.
    let p = ones(DType::F16);
    with_grad(&p, 1.0);
    let mut opt = Adam::new([p.clone()], 0.1).weight_decay(1e5);
    rejected(&mut || opt.step(), &[&p], &overflow(DType::F16));
    assert!(opt.state(&p).is_none());

    // Controls. The largest finite values, infinities and NaN are converted.
    let p = ones(DType::F16);
    with_grad(&p, 1.0);
    Sgd::new([p.clone()], 65504.0).step();
    assert!(values(&p).iter().all(|&v| v == -65504.0));
    for (lr, check) in [
        (f64::INFINITY, f64::is_infinite as fn(f64) -> bool),
        (f64::NAN, f64::is_nan),
    ] {
        let p = ones(DType::F16);
        with_grad(&p, 1.0);
        Sgd::new([p.clone()], lr).step();
        assert!(values(&p).into_iter().all(check));
    }
    // `mul_(momentum)` (also without a buffer to update yet), AdamW's `mul_(1 - lr * wd)` and
    // `add_(eps)` aren't checked; an update that overflows by itself is Inf.
    let p = ones(DType::F16);
    with_grad(&p, 1.0);
    Sgd::new([p.clone()], 0.1).momentum(1e5).step();
    assert!(values(&p).iter().all(|&v| v == 0.89990234375));
    let p = ones(DType::F32);
    with_grad(&p, 1.0);
    AdamW::new([p.clone()], 1.0).weight_decay(1e40).step();
    assert!(values(&p).iter().all(|&v| v == f64::NEG_INFINITY));
    let p = ones(DType::F32);
    with_grad(&p, 1.0);
    Adam::new([p.clone()], 0.1).eps(1e40).step();
    assert!(values(&p).iter().all(|&v| v == 1.0));
    let p = ones(DType::F16);
    with_grad(&p, 1.0);
    Adam::new([p.clone()], 1e5).step();
    assert!(values(&p).iter().all(|&v| v == f64::NEG_INFINITY));
    // Parameters without a gradient are skipped, Meta parameters aren't checked.
    let p = ones(DType::F16);
    Sgd::new([p.clone()], 1e5).step();
    assert!(values(&p).iter().all(|&v| v == 1.0));
    let p = Parameter::new(homura::ones_with([32], (DType::F16, Device::Meta)));
    p.set_grad(Some(homura::ones_with([32], (DType::F16, Device::Meta))));
    Sgd::new([p.clone()], 1e5).step();
    let mut adam = Adam::new([p.clone()], 1e38);
    adam.step();
    assert_eq!(adam.state(&p).unwrap().step().item::<f32>(), 1.0);
}

/// The update doesn't depend on the layout of the parameter, its gradient or the state.
#[test]
fn non_contiguous_parameters_and_gradients() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let data = homura::randn_with([5, 7], DType::F64).to(dtype);
        let grads: Vec<Tensor> = (0..3)
            .map(|_| homura::randn_with([7, 5], DType::F64).to(dtype))
            .collect();
        let run = |transposed: bool| {
            // The same values, in a transposed layout or a contiguous one.
            let p = if transposed {
                Parameter::new(data.deep_clone().transpose(0, 1))
            } else {
                Parameter::new(data.transpose(0, 1).deep_clone())
            };
            assert_eq!(p.is_contiguous(), !transposed);
            let mut sgd = Sgd::new([p.clone()], 0.1)
                .momentum(0.9)
                .dampening(0.2)
                .weight_decay(0.1)
                .maximize(true);
            let q = Parameter::new(p.deep_clone());
            let mut adam = AdamW::new([q.clone()], 0.1).amsgrad(true).weight_decay(0.2);
            for g in &grads {
                let g = if transposed {
                    g.transpose(0, 1).deep_clone().transpose(0, 1)
                } else {
                    g.deep_clone()
                };
                assert_eq!(g.is_contiguous(), !transposed);
                p.set_grad(Some(g.clone()));
                q.set_grad(Some(g));
                sgd.step();
                adam.step();
            }
            let s = adam.state(&q).unwrap();
            [
                values(&p),
                values(sgd.state(&p).unwrap().momentum_buffer()),
                values(&q),
                values(s.exp_avg()),
                values(s.max_exp_avg_sq().unwrap()),
            ]
        };
        let (a, b) = (run(true), run(false));
        for (x, y) in a.iter().zip(&b) {
            let same = x.iter().zip(y).all(|(u, v)| u.to_bits() == v.to_bits());
            assert!(same, "{dtype}");
        }
    }
}

/// R64 of review/M2.md: only the options of a group can be changed (the groups themselves can't,
/// see `src/optim/errors.rs`), so the state stays with its parameter.
#[test]
fn options_mut_keeps_the_state_with_its_parameter() {
    let (a, b) = (param(&[1.0]), param(&[2.0]));
    let mut opt = Sgd::new([a.clone()], 0.1).momentum(0.9);
    opt.add_param_group(ParamGroup::new([b.clone()], *opt.defaults()));
    grad(&a, &[1.0]);
    grad(&b, &[10.0]);
    opt.step();
    opt.options_mut(1).lr = 0.01;
    opt.options_mut(0).momentum = 0.5;
    opt.step();
    // a: buffer 0.5 * 1 + 1, update 0.1 * 1.5; b: buffer 0.9 * 10 + 10, update 0.01 * 19.
    assert_eq!(values(opt.state(&a).unwrap().momentum_buffer()), [1.5]);
    assert_eq!(values(opt.state(&b).unwrap().momentum_buffer()), [19.0]);
    assert!((values(&a)[0] - (1.0 - 0.1 - 0.15)).abs() < 1e-15);
    assert!((values(&b)[0] - (2.0 - 1.0 - 0.19)).abs() < 1e-15);
    assert_eq!(opt.param_groups()[1].options.lr, 0.01);
    assert!(opt.param_groups()[1].params()[0].is_same(&b));
    expect_panic(
        || {
            opt.options_mut(2);
        },
        "parameter group index 2 is out of range for 2 groups",
    );
}

#[test]
fn defaults_are_pytorchs() {
    let p = || [param(&[1.0])];
    let sgd = Sgd::new(p(), 0.1);
    let expected = SgdOptions {
        lr: 0.1,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        maximize: false,
    };
    assert_eq!(*sgd.defaults(), expected);
    assert_eq!(sgd.param_groups()[0].options, expected);
    let adam = AdamOptions {
        lr: 0.1,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.0,
        amsgrad: false,
        maximize: false,
    };
    assert_eq!(*Adam::new(p(), 0.1).defaults(), adam);
    let adamw = AdamOptions {
        weight_decay: 0.01,
        ..adam
    };
    assert_eq!(*AdamW::new(p(), 0.1).defaults(), adamw);
    assert_eq!(AdamW::new(p(), 0.1).param_groups()[0].options, adamw);
}

#[test]
fn options_are_checked_with_pytorchs_messages() {
    let p = || [param(&[1.0])];
    expect_panic(|| drop(Sgd::new(p(), -0.5)), "Invalid learning rate: -0.5");
    expect_panic(
        || drop(Sgd::new(p(), 0.1).momentum(-0.5)),
        "Invalid momentum value: -0.5",
    );
    expect_panic(
        || drop(Sgd::new(p(), 0.1).weight_decay(-0.5)),
        "Invalid weight_decay value: -0.5",
    );
    expect_panic(|| drop(Adam::new(p(), -0.5)), "Invalid learning rate: -0.5");
    expect_panic(
        || drop(Adam::new(p(), 0.1).eps(-0.5)),
        "Invalid epsilon value: -0.5",
    );
    expect_panic(
        || drop(Adam::new(p(), 0.1).betas((1.0, 0.9))),
        "Invalid beta parameter at index 0: 1",
    );
    expect_panic(
        || drop(Adam::new(p(), 0.1).betas((0.9, -0.1))),
        "Invalid beta parameter at index 1: -0.1",
    );
    expect_panic(
        || drop(Adam::new(p(), 0.1).weight_decay(-0.5)),
        "Invalid weight_decay value: -0.5",
    );
    expect_panic(
        || drop(AdamW::new(p(), 0.1).weight_decay(-0.5)),
        "Invalid weight_decay value: -0.5",
    );
    // Adam's checks are `not 0 <= value`, which NaN fails.
    expect_panic(
        || drop(Adam::new(p(), f64::NAN)),
        "Invalid learning rate: NaN",
    );
    expect_panic(
        || drop(AdamW::new(p(), 0.1).betas((f64::NAN, 0.9))),
        "Invalid beta parameter at index 0: NaN",
    );
    expect_panic(
        || drop(Adam::new(p(), 0.1).eps(f64::NAN)),
        "Invalid epsilon value: NaN",
    );
    // SGD's checks are `value < 0`, which NaN passes; dampening and the size of the momentum
    // aren't checked.
    drop(Sgd::new(p(), f64::NAN).momentum(5.0).dampening(-3.0));

    // Nesterov needs a momentum and zero dampening; the chained methods can come in any order,
    // the combination is checked by `step` and `add_param_group`.
    let nesterov = "Nesterov momentum requires a momentum and zero dampening";
    let mut opt = Sgd::new(p(), 0.1).nesterov(true);
    assert_eq!(panic_message(AssertUnwindSafe(|| opt.step())), nesterov);
    let mut opt = Sgd::new(p(), 0.1)
        .nesterov(true)
        .momentum(0.9)
        .dampening(0.1);
    assert_eq!(panic_message(AssertUnwindSafe(|| opt.step())), nesterov);
    let mut opt = Sgd::new(p(), 0.1).nesterov(true).momentum(0.9);
    opt.step();
    let options = SgdOptions {
        dampening: 0.5,
        ..*opt.defaults()
    };
    let group = ParamGroup::new(p(), options);
    assert_eq!(
        panic_message(AssertUnwindSafe(|| opt.add_param_group(group))),
        nesterov
    );
    let group = ParamGroup::new(
        p(),
        SgdOptions {
            lr: -1.0,
            ..options
        },
    );
    assert_eq!(
        panic_message(AssertUnwindSafe(|| opt.add_param_group(group))),
        "Invalid learning rate: -1"
    );
    assert_eq!(opt.param_groups().len(), 1);
}

#[test]
fn step_checks_every_group_before_updating() {
    // PyTorch doesn't check options set through `param_groups` again (with lr = -1 it ascends);
    // homura's `step` panics before it updates any parameter of any group.
    let (a, b) = (param(&[1.0]), param(&[1.0]));
    let mut opt = Adam::new([a.clone()], 0.1);
    opt.add_param_group(ParamGroup::new([b.clone()], *opt.defaults()));
    grad(&a, &[1.0]);
    grad(&b, &[1.0]);
    opt.options_mut(1).betas.1 = 1.0;
    assert_eq!(
        panic_message(AssertUnwindSafe(|| opt.step())),
        "Invalid beta parameter at index 1: 1"
    );
    assert!(values(&a) == [1.0] && opt.state(&a).is_none());
    opt.options_mut(1).betas.1 = 0.999;
    opt.step();
    assert!(values(&a)[0] < 1.0 && values(&b)[0] < 1.0);

    let mut opt = Sgd::new([a.clone()], 0.1);
    opt.options_mut(0).lr = -1.0;
    let before = values(&a);
    assert_eq!(
        panic_message(AssertUnwindSafe(|| opt.step())),
        "Invalid learning rate: -1"
    );
    assert_eq!(values(&a), before);
}

#[test]
fn parameters_are_checked() {
    assert_eq!(
        panic_message(|| drop(Sgd::new(Vec::<Parameter>::new(), 0.1))),
        "optimizer got an empty parameter list"
    );
    // An empty group can be added.
    let (a, b) = (param(&[1.0]), param(&[2.0]));
    let mut opt = Sgd::new([a.clone()], 0.1);
    opt.add_param_group(ParamGroup::new([], *opt.defaults()));
    assert_eq!(opt.param_groups().len(), 2);

    let cross = "some parameters appear in more than one parameter group";
    let group = ParamGroup::new([b.clone(), a.clone()], *opt.defaults());
    assert_eq!(
        panic_message(AssertUnwindSafe(|| opt.add_param_group(group))),
        cross
    );
    // PyTorch warns about a duplicate in one group and updates it twice.
    let duplicate = "optimizer contains a parameter group with duplicate parameters";
    assert_eq!(
        panic_message(|| drop(Adam::new([a.clone(), a.clone()], 0.1))),
        duplicate
    );
    let group = ParamGroup::new([b.clone(), b.clone()], *opt.defaults());
    assert_eq!(
        panic_message(AssertUnwindSafe(|| opt.add_param_group(group))),
        duplicate
    );
    assert_eq!(opt.param_groups().len(), 2);
    opt.add_param_group(ParamGroup::new([b.clone()], *opt.defaults()));
    assert!(opt.param_groups()[2].params()[0].is_same(&b));

    // A parameter becomes a non-leaf when an in-place operation in grad mode writes a value
    // that requires grad into it while it doesn't require grad.
    let non_leaf = "can't optimize a non-leaf Tensor";
    let make_non_leaf = |p: &Parameter| {
        p.requires_grad_(false);
        let x = homura::tensor_with([1.0], DType::F64);
        x.requires_grad_(true);
        p.add_(&x);
        assert!(!p.is_leaf());
    };
    let c = param(&[1.0]);
    make_non_leaf(&c);
    assert_eq!(panic_message(|| drop(Sgd::new([c], 0.1))), non_leaf);
    // `step` checks it again for the parameters that have a gradient, before any update.
    let (d, e) = (param(&[1.0]), param(&[1.0]));
    let mut opt = Sgd::new([d.clone(), e.clone()], 0.1);
    grad(&d, &[1.0]);
    grad(&e, &[1.0]);
    make_non_leaf(&e);
    assert_eq!(panic_message(AssertUnwindSafe(|| opt.step())), non_leaf);
    assert_eq!(values(&d), [1.0]);
    e.set_grad(None);
    opt.step();
    assert_eq!(values(&d), [0.9]);

    // A parameter that doesn't require grad may be an integer tensor; without a gradient it
    // is skipped.
    let int = Parameter::with_requires_grad(homura::tensor([1, 2]), false);
    let mut opt = Adam::new([int.clone()], 0.1);
    opt.step();
    assert!(opt.state(&int).is_none() && int.to_vec::<i64>() == [1, 2]);
    assert!(opt.state(&param(&[1.0])).is_none());
}

#[test]
fn sgd_momentum_buffer() {
    // The first buffer is the gradient after weight decay, without dampening, and a copy.
    let p = param(&[1.0]);
    let mut opt = Sgd::new([p.clone()], 0.1)
        .momentum(0.9)
        .dampening(0.5)
        .weight_decay(0.1);
    grad(&p, &[2.0]);
    opt.step();
    let buf = opt.state(&p).unwrap().momentum_buffer().clone();
    assert_eq!(values(&buf), [2.0 + 0.1]);
    assert!((values(&p)[0] - (1.0 - 0.1 * 2.1)).abs() < 1e-15);
    p.grad().unwrap().fill_(7.0);
    assert_eq!(values(&buf), [2.1]);
    // Then `momentum * buf + (1 - dampening) * g`.
    let g = 7.0 + 0.1 * values(&p)[0];
    opt.step();
    assert!((values(&buf)[0] - (0.9 * 2.1 + 0.5 * g)).abs() < 1e-15);

    // maximize negates the gradient before the weight decay.
    let p = param(&[1.0]);
    let mut opt = Sgd::new([p.clone()], 0.1)
        .momentum(0.9)
        .weight_decay(0.1)
        .maximize(true);
    grad(&p, &[2.0]);
    opt.step();
    assert_eq!(
        values(opt.state(&p).unwrap().momentum_buffer()),
        [-2.0 + 0.1]
    );

    // Nesterov: the buffers are 2 and 3.8, the updates use 2 + 0.9 * 2 and 2 + 0.9 * 3.8.
    let p = param(&[1.0]);
    let mut opt = Sgd::new([p.clone()], 0.1).momentum(0.9).nesterov(true);
    grad(&p, &[2.0]);
    opt.step();
    opt.step();
    assert!((values(&p)[0] - (1.0 - 0.38 - 0.542)).abs() < 1e-15);

    // Without momentum there is no state; a momentum turned on later starts the buffer, one
    // turned off keeps it unused.
    let p = param(&[1.0]);
    let mut opt = Sgd::new([p.clone()], 0.1);
    grad(&p, &[2.0]);
    opt.step();
    assert!(opt.state(&p).is_none());
    opt.options_mut(0).momentum = 0.9;
    opt.step();
    assert_eq!(values(opt.state(&p).unwrap().momentum_buffer()), [2.0]);
    opt.options_mut(0).momentum = 0.0;
    let before = values(&p)[0];
    opt.step();
    assert_eq!(values(opt.state(&p).unwrap().momentum_buffer()), [2.0]);
    assert!((values(&p)[0] - (before - 0.2)).abs() < 1e-15);

    // A gradient with history leaves no history in the buffer or the parameter.
    let p = param(&[1.0]);
    let mut opt = Sgd::new([p.clone()], 0.1).momentum(0.9);
    let q = homura::tensor_with([3.0], DType::F64);
    q.requires_grad_(true);
    p.set_grad(Some(&q * 1.0));
    opt.step();
    let buf = opt.state(&p).unwrap().momentum_buffer();
    assert!(buf.grad_fn().is_none() && !buf.requires_grad());
    assert!(p.is_leaf() && p.grad_fn().is_none());
}

#[test]
fn none_and_zero_gradients_and_frozen_parameters() {
    // A zero gradient is a gradient: weight decay applies and Adam counts the step. A `None`
    // gradient skips the parameter.
    let (p, r) = (param(&[1.0]), param(&[1.0]));
    let mut opt = Sgd::new([p.clone(), r.clone()], 0.1).weight_decay(0.5);
    grad(&p, &[0.0]);
    opt.step();
    assert!(values(&p) == [1.0 - 0.1 * 0.5] && values(&r) == [1.0]);
    // Freezing keeps the gradient, which still updates the parameter, as in PyTorch.
    p.requires_grad_(false);
    opt.step();
    assert!(values(&p)[0] < 0.95 - 0.04);
    opt.zero_grad();
    assert!(p.grad().is_none() && r.grad().is_none());
    let frozen = values(&p);
    opt.step();
    assert_eq!(values(&p), frozen);

    let (p, r) = (param(&[1.0]), param(&[1.0]));
    let mut opt = AdamW::new([p.clone(), r.clone()], 0.1).weight_decay(0.5);
    grad(&p, &[0.0]);
    opt.step();
    assert!(values(&p) == [1.0 - 0.1 * 0.5] && values(&r) == [1.0]);
    assert!(opt.state(&r).is_none());
    assert_eq!(opt.state(&p).unwrap().step().item::<f32>(), 1.0);
    // Adam adds the decay to the gradient: g = 0.5, and the first step moves by about lr.
    let p = param(&[1.0]);
    let mut opt = Adam::new([p.clone()], 0.1).weight_decay(0.5);
    grad(&p, &[0.0]);
    opt.step();
    assert!((values(&p)[0] - 0.9).abs() < 1e-7);
}

#[test]
fn adam_state() {
    // The state has the parameter's dtype and device; `step` is a 0-D F32 CPU tensor with the
    // default dtype F32 (F64 with the default dtype F64, tests/optim_default_dtype.rs).
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let p = Parameter::new(homura::ones_with([2], dtype));
        let mut opt = Adam::new([p.clone()], 0.1);
        p.set_grad(Some(homura::ones_with([2], dtype)));
        opt.step();
        let s = opt.state(&p).unwrap();
        assert_eq!(s.step().dtype(), DType::F32);
        assert!(s.step().device() == Device::Cpu && s.step().shape().is_empty());
        assert!(s.exp_avg().dtype() == dtype && s.exp_avg_sq().dtype() == dtype);
        assert!(s.max_exp_avg_sq().is_none());
    }
    let p = param(&[1.0]);
    let mut opt = Adam::new([p.clone()], 0.1).amsgrad(true);
    grad(&p, &[1.0]);
    opt.step();
    assert!(opt.state(&p).unwrap().max_exp_avg_sq().is_some());

    // Turning amsgrad on after the state was created: PyTorch raises KeyError when it reaches
    // that group, after updating the earlier groups; homura panics before updating anything.
    let (p, r) = (param(&[1.0]), param(&[1.0]));
    let mut opt = Adam::new([p.clone()], 0.1);
    grad(&p, &[1.0]);
    opt.step();
    opt.add_param_group(ParamGroup::new(
        [r.clone()],
        AdamOptions {
            amsgrad: true,
            ..*opt.defaults()
        },
    ));
    grad(&r, &[1.0]);
    opt.step();
    let (p_before, r_before) = (values(&p), values(&r));
    opt.options_mut(0).amsgrad = true;
    let message = panic_message(AssertUnwindSafe(|| opt.step()));
    assert!(message.contains("has no max_exp_avg_sq"), "{message}");
    assert!(values(&p) == p_before && values(&r) == r_before);
    assert_eq!(opt.state(&r).unwrap().step().item::<f32>(), 1.0);
    // Turned off again, the step works without the maximum.
    opt.options_mut(0).amsgrad = false;
    opt.step();
    assert_eq!(opt.state(&p).unwrap().step().item::<f32>(), 3.0);
}

#[test]
fn meta_parameters() {
    let p = Parameter::new(homura::ones_with([3], Device::Meta));
    p.set_grad(Some(homura::ones_with([3], Device::Meta)));
    let mut opt = Adam::new([p.clone()], 0.1).amsgrad(true).weight_decay(0.1);
    opt.step();
    let s = opt.state(&p).unwrap();
    assert!(s.step().device() == Device::Cpu && s.step().item::<f32>() == 1.0);
    assert!(s.exp_avg().device() == Device::Meta && s.exp_avg_sq().device() == Device::Meta);
    let mut opt = Sgd::new([p.clone()], 0.1).momentum(0.9).nesterov(true);
    opt.step();
    opt.step();
    let buf = opt.state(&p).unwrap().momentum_buffer();
    assert!(buf.device() == Device::Meta && buf.shape() == [3]);
    assert!(p.device() == Device::Meta && p.is_leaf());
}

#[test]
fn step_records_nothing_and_bumps_versions() {
    let p = param(&[1.0, 2.0]);
    let mut opt = AdamW::new([p.clone()], 0.1).amsgrad(true);
    let saved = &*p * &*p;
    (&*p * 2.0).sum().backward();
    {
        let _grad = homura::enable_grad();
        opt.step();
    }
    let s = opt.state(&p).unwrap();
    for t in [s.exp_avg(), s.exp_avg_sq(), s.max_exp_avg_sq().unwrap()] {
        assert!(!t.requires_grad() && t.grad_fn().is_none());
    }
    assert!(p.is_leaf() && p.requires_grad());
    // The update is an in-place write: a graph that saved the parameter can't run backward.
    let message = panic_message(|| saved.sum().backward());
    assert!(
        message.contains("has been modified by an inplace operation"),
        "{message}"
    );
}

#[derive(Module)]
struct Tied {
    head: Parameter,
    tail: Parameter,
}

#[test]
fn shared_parameters_are_updated_once() {
    let w = param(&[1.0, 2.0]);
    let model = Tied {
        head: w.clone(),
        tail: w.clone(),
    };
    let params = model.parameters();
    assert_eq!(params.len(), 1);
    let mut opt = Sgd::new(params, 0.1);
    let x = homura::tensor_with([1.0, 1.0], DType::F64);
    ((&*model.head * &x).sum() + (&*model.tail * &x).sum()).backward();
    assert_eq!(values(&w.grad().unwrap()), [2.0, 2.0]);
    opt.step();
    assert_eq!(values(&model.tail), [0.8, 1.8]);
    model.zero_grad();
    assert!(w.grad().is_none());
}

#[test]
fn moving_the_model_after_the_state_was_created() {
    let model = homura::nn::Linear::new(2, 2);
    // Without state, the parameters move and the state is created with the new dtype.
    let mut opt = Adam::new(model.parameters(), 0.1);
    model.to(DType::F64);
    let x = homura::ones_with([1, 2], DType::F64);
    model.forward(&x).sum().backward();
    opt.step();
    assert_eq!(
        opt.state(&model.weight).unwrap().exp_avg().dtype(),
        DType::F64
    );

    // With state, a step panics before it updates any parameter.
    model.to(DType::F32);
    let x = homura::ones([1, 2]);
    model.forward(&x).sum().backward();
    let (w, b) = (values(&model.weight), values(model.bias.as_ref().unwrap()));
    let message = panic_message(AssertUnwindSafe(|| opt.step()));
    assert!(
        message.contains("the optimizer state exp_avg of a parameter has shape [2, 2], dtype f64")
            && message.contains("but the parameter has shape [2, 2], dtype f32"),
        "{message}"
    );
    assert!(values(&model.weight) == w && values(model.bias.as_ref().unwrap()) == b);
    assert_eq!(opt.state(&model.weight).unwrap().step().item::<f32>(), 1.0);

    // SGD's momentum buffer is checked too, also when only a later parameter was moved.
    let (p, r) = (param(&[1.0]), param(&[1.0]));
    let mut opt = Sgd::new([p.clone(), r.clone()], 0.1).momentum(0.9);
    grad(&p, &[1.0]);
    grad(&r, &[1.0]);
    opt.step();
    [r.clone()].to(DType::F32);
    grad(&r, &[1.0]);
    let before = values(&p);
    let message = panic_message(AssertUnwindSafe(|| opt.step()));
    assert!(message.contains("momentum_buffer"), "{message}");
    assert_eq!(values(&p), before);
}
