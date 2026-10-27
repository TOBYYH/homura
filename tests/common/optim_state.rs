//! The cases of `tests/golden/m3_optim_state.pt` from `tools/golden/gen_m3_optim_state.py`:
//! PyTorch's optimizers trained for six steps, with their state dicts. [`run`] trains homura's
//! optimizer from PyTorch's state dicts and compares every step bit for bit, and writes what
//! homura saves for `tools/golden/check_m3_optim_state.py`, which resumes from it in PyTorch.

#![allow(dead_code)]

use std::path::Path;

use homura::Tensor;
use homura::io::{Dict, Key, Value};
use homura::nn::Parameter;
use homura::optim::{Adam, AdamW, Optimizer, OptimizerStateDict, ParamGroup, Sgd, TypedOptimizer};

pub const STEPS: usize = 6;
pub const PARAMS: [&str; 3] = ["a", "b", "c"];

/// The cases by name, in the file's order.
pub fn cases() -> Vec<(String, Dict)> {
    let path = format!(
        "{}/tests/golden/m3_optim_state.pt",
        env!("CARGO_MANIFEST_DIR")
    );
    let Value::Dict(cases) = homura::io::load_value(&path).unwrap() else {
        panic!("{path} doesn't hold a dict");
    };
    cases
        .iter()
        .map(|(k, v)| {
            let Key::Str(name) = k else {
                panic!("case {k:?}")
            };
            (name.clone(), as_dict(v).clone())
        })
        .collect()
}

pub fn get<'a>(d: &'a Dict, key: &str) -> &'a Value {
    d.get(key).unwrap_or_else(|| panic!("no {key:?} in {d:?}"))
}

pub fn as_dict(v: &Value) -> &Dict {
    match v {
        Value::Dict(d) => d,
        v => panic!("expected a dict, found {v:?}"),
    }
}

pub fn as_list(v: &Value) -> &[Value] {
    match v {
        Value::List(l) => l,
        v => panic!("expected a list, found {v:?}"),
    }
}

pub fn as_tensor(v: &Value) -> &Tensor {
    match v {
        Value::Tensor(t) => t,
        v => panic!("expected a tensor, found {v:?}"),
    }
}

/// Asserts that `actual` has the dtype, shape and bits of `expected`.
pub fn assert_same(actual: &Tensor, expected: &Tensor, what: &str) {
    assert_eq!(actual.dtype(), expected.dtype(), "{what}: dtype");
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    let bits = |t: &Tensor| -> Vec<u64> {
        let t = t.to(homura::DType::F64).to_vec::<f64>();
        t.iter().map(|x| x.to_bits()).collect()
    };
    assert_eq!(bits(actual), bits(expected), "{what}: values");
}

/// Asserts that a number, flag, pair or `None` option is the same as PyTorch's, which keeps the
/// `int`s it was given where homura saves `float`s.
pub fn assert_same_option(actual: &Value, expected: &Value, what: &str) {
    let number = |v: &Value| match v {
        Value::Int(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    };
    match (actual, expected) {
        (Value::Tuple(a), Value::Tuple(e)) if a.len() == e.len() => {
            for (x, y) in a.iter().zip(e) {
                assert_same_option(x, y, what);
            }
        }
        (Value::Bool(a), Value::Bool(e)) => assert_eq!(a, e, "{what}"),
        (Value::None, Value::None) => {}
        (a, e) => match (number(a), number(e)) {
            (Some(x), Some(y)) => assert_eq!(x.to_bits(), y.to_bits(), "{what}"),
            _ => panic!("{what}: {a:?} is not {e:?}"),
        },
    }
}

/// Asserts that the state of `actual` is PyTorch's `state` (a dict by parameter number), bit for
/// bit, with the names in the same order.
pub fn assert_same_state(actual: &OptimizerStateDict, expected: &Dict, what: &str) {
    let numbers: Vec<Key> = actual.state.keys().map(|&n| Key::Int(n as i64)).collect();
    let expected_numbers: Vec<Key> = expected.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(numbers, expected_numbers, "{what}: parameters with state");
    for (n, values) in &actual.state {
        let e = as_dict(expected.get(*n as i64).unwrap());
        let names: Vec<&str> = values.iter().map(|(k, _)| k.as_str()).collect();
        let expected_names: Vec<&str> = e
            .iter()
            .map(|(k, _)| match k {
                Key::Str(k) => k.as_str(),
                k => panic!("{what}: state name {k:?}"),
            })
            .collect();
        assert_eq!(names, expected_names, "{what}: state of {n}");
        for (name, v) in values {
            let at = format!("{what}: state[{n}][{name:?}]");
            assert_same(as_tensor(v), as_tensor(e.get(name.as_str()).unwrap()), &at);
        }
    }
}

/// Asserts that `actual` is PyTorch's state dict `expected`: the same groups and options, in the
/// same order, and the same state.
pub fn assert_same_state_dict(
    actual: &OptimizerStateDict,
    expected: &OptimizerStateDict,
    what: &str,
) {
    assert_eq!(
        actual.param_groups.len(),
        expected.param_groups.len(),
        "{what}: groups"
    );
    for (g, (a, e)) in actual
        .param_groups
        .iter()
        .zip(&expected.param_groups)
        .enumerate()
    {
        assert_eq!(a.params, e.params, "{what}: params of group {g}");
        let names = |o: &[(String, Value)]| o.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>();
        assert_eq!(
            names(&a.options),
            names(&e.options),
            "{what}: options of group {g}"
        );
        for ((name, x), (_, y)) in a.options.iter().zip(&e.options) {
            assert_same_option(x, y, &format!("{what}: group {g} {name}"));
        }
    }
    let Value::Dict(expected) = Value::from(expected) else {
        unreachable!()
    };
    assert_same_state(actual, as_dict(get(&expected, "state")), what);
}

/// The parameters with the values of `values` (copies) and the optimizer `kind` of them, in
/// the groups of the cases: [a, c] and [b]. The options are what a state dict loads.
pub fn build(kind: &str, values: &Dict) -> (Vec<Parameter>, Box<dyn Optimizer>) {
    let params: Vec<Parameter> = PARAMS
        .iter()
        .map(|k| Parameter::new(as_tensor(get(values, k)).deep_clone()))
        .collect();
    let first = [params[0].clone(), params[2].clone()];
    let second = [params[1].clone()];
    let opt: Box<dyn Optimizer> = match kind {
        "sgd" => {
            let mut o = Sgd::new(first, 0.0);
            o.add_param_group(ParamGroup::new(second, *o.defaults()));
            Box::new(o)
        }
        "adam" => {
            let mut o = Adam::new(first, 0.0);
            o.add_param_group(ParamGroup::new(second, *o.defaults()));
            Box::new(o)
        }
        "adamw" => {
            let mut o = AdamW::new(first, 0.0);
            o.add_param_group(ParamGroup::new(second, *o.defaults()));
            Box::new(o)
        }
        k => panic!("optimizer {k:?}"),
    };
    (params, opt)
}

/// Sets the options `options` in every group, through the state dict.
pub fn set_options(opt: &mut dyn Optimizer, options: &Dict) {
    if options.is_empty() {
        return;
    }
    let mut sd = opt.state_dict();
    for group in &mut sd.param_groups {
        for (key, value) in options.iter() {
            let Key::Str(key) = key else {
                panic!("option {key:?}")
            };
            let slot = group.options.iter_mut().find(|(k, _)| k == key).unwrap();
            slot.1 = value.clone();
        }
    }
    opt.load_state_dict(&sd).unwrap();
}

/// Step `t` of a case: its options, its gradients, and the step.
pub fn step(case: &Dict, opt: &mut dyn Optimizer, params: &[Parameter], t: usize) {
    set_options(opt, as_dict(&as_list(get(case, "options"))[t]));
    let grads = as_dict(&as_list(get(case, "grads"))[t]);
    for (k, p) in PARAMS.iter().zip(params) {
        p.set_grad(grads.get(*k).map(|g| as_tensor(g).deep_clone()));
    }
    opt.step();
}

fn snapshot(params: &[Parameter]) -> Value {
    let d: Dict = PARAMS
        .iter()
        .zip(params)
        .map(|(k, p)| (*k, p.detach().deep_clone()))
        .collect();
    Value::Dict(d)
}

/// A copy of a state dict, so that it keeps its values while training goes on.
fn deep_copy(sd: &OptimizerStateDict) -> OptimizerStateDict {
    let mut sd = sd.clone();
    for values in sd.state.values_mut() {
        for (_, v) in values {
            if let Value::Tensor(t) = v {
                *v = Value::Tensor(t.deep_clone());
            }
        }
    }
    sd
}

fn assert_same_params(params: &[Parameter], expected: &Dict, what: &str) {
    for (k, p) in PARAMS.iter().zip(params) {
        assert_same(p, as_tensor(get(expected, k)), &format!("{what}: {k}"));
    }
}

/// Trains homura's optimizer of the case `name` from the start with the options of PyTorch's
/// first state dict, and again from PyTorch's saved state dict, comparing each step with
/// PyTorch's. Writes `<out>/<name>.pt` with homura's own saved parameters and state dict and the
/// steps after it, which `check_m3_optim_state.py` resumes from in PyTorch.
pub fn run(name: &str, case: &Dict, out: &Path) {
    let Value::Str(kind) = get(case, "kind") else {
        panic!("kind")
    };
    let Value::Int(saved_at) = get(case, "saved_at") else {
        panic!("saved_at")
    };
    let saved_at = *saved_at as usize;
    let after = as_list(get(case, "after"));
    let state_dict = |key: &str| OptimizerStateDict::try_from(get(case, key)).unwrap();

    // From the start, with the options of PyTorch's state dict before the first step.
    let (params, mut opt) = build(kind, as_dict(get(case, "init")));
    let first = state_dict("optimizer0");
    assert!(first.state.is_empty());
    opt.load_state_dict(&first).unwrap();
    assert_same_state_dict(&opt.state_dict(), &first, &format!("{name}: optimizer0"));
    let mut saved = None;
    let mut steps = vec![];
    for t in 0..STEPS {
        if t == saved_at {
            let what = format!("{name}: saved");
            assert_same_params(&params, as_dict(get(case, "params")), &what);
            let sd = deep_copy(&opt.state_dict());
            assert_same_state_dict(&sd, &state_dict("optimizer"), &what);
            saved = Some((snapshot(&params), sd));
        }
        step(case, &mut *opt, &params, t);
        if t >= saved_at {
            let what = format!("{name}: step {t} from the start");
            let expected = as_dict(&after[t - saved_at]);
            assert_same_params(&params, as_dict(get(expected, "params")), &what);
            assert_same_state(&opt.state_dict(), as_dict(get(expected, "state")), &what);
            let state = Value::from(&deep_copy(&opt.state_dict()));
            let state = get(as_dict(&state), "state").clone();
            steps.push(Value::Dict(
                [("params", snapshot(&params)), ("state", state)]
                    .into_iter()
                    .collect(),
            ));
        }
    }

    // Resumed from PyTorch's saved state dict.
    let (params, mut opt) = build(kind, as_dict(get(case, "params")));
    opt.load_state_dict(&state_dict("optimizer")).unwrap();
    for t in saved_at..STEPS {
        step(case, &mut *opt, &params, t);
        let what = format!("{name}: step {t} resumed");
        let expected = as_dict(&after[t - saved_at]);
        assert_same_params(&params, as_dict(get(expected, "params")), &what);
        assert_same_state(&opt.state_dict(), as_dict(get(expected, "state")), &what);
    }

    let (saved_params, saved_sd) = saved.unwrap();
    let file: Dict = [
        ("kind", Value::from(kind.as_str())),
        ("params", saved_params),
        ("optimizer", Value::from(&saved_sd)),
        ("after", Value::List(steps)),
    ]
    .into_iter()
    .collect();
    std::fs::create_dir_all(out).unwrap();
    homura::io::save_value(&Value::Dict(file), out.join(format!("{name}.pt"))).unwrap();
}
