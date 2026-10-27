//! M3 step 6: optimizer state dicts. Training resumes from PyTorch's state dicts bit for bit
//! (`tests/golden/m3_optim_state.pt` from `tools/golden/gen_m3_optim_state.py`), and the files
//! homura writes here are resumed from in PyTorch by `tools/golden/check_m3_optim_state.py`. The
//! case with the default dtype F64 is in `tests/optim_default_dtype.rs`, which has its own binary.

#[path = "common/optim_state.rs"]
mod optim_state;

use std::path::Path;

use homura::io::{self, Value};
use homura::nn::Parameter;
use homura::optim::{
    Adam, AdamW, LoadError, Optimizer, OptimizerStateDict, ParamGroup, ParamGroupRecord, Sgd,
    TypedOptimizer,
};
use homura::{DType, Device, Tensor};
use optim_state::{assert_same, assert_same_state_dict};

#[test]
fn training_resumes_from_pytorchs_state_dicts_bit_for_bit() {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("m3_optim_state");
    let cases = optim_state::cases();
    assert_eq!(cases.len(), 15);
    for (name, case) in cases.iter().filter(|(n, _)| n != "adam_default_f64_f64") {
        optim_state::run(name, case, &out);
    }
}

fn param(shape: &[usize], seed: f64) -> Parameter {
    let n: usize = shape.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.37 + seed as f32).sin())
        .collect();
    Parameter::new(homura::tensor(values).reshape(shape))
}

fn grads(params: &[&Parameter], scale: f64) {
    for p in params {
        p.set_grad(Some(p.detach().cos().mul(scale)));
    }
}

/// Parameters `p` ([2, 3]) and `q` ([3]) in group 0 and `r` ([1]) in group 1 of an AMSGrad Adam
/// after two steps in which `q` had no gradient: `p` and `r` have state, `q` has none.
fn adam() -> ([Parameter; 3], Adam) {
    let (p, q, r) = (param(&[2, 3], 0.0), param(&[3], 1.0), param(&[1], 2.0));
    let mut opt = Adam::new([p.clone(), q.clone()], 0.01).amsgrad(true);
    opt.add_param_group(ParamGroup::new([r.clone()], *opt.defaults()));
    for scale in [1.0, 2.0] {
        grads(&[&p, &r], scale);
        opt.step();
    }
    ([p, q, r], opt)
}

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

fn options(sd: &mut OptimizerStateDict, group: usize) -> &mut Vec<(String, Value)> {
    &mut sd.param_groups[group].options
}

fn set(values: &mut Vec<(String, Value)>, name: &str, value: impl Into<Value>) {
    let value = value.into();
    match values.iter_mut().find(|(k, _)| k == name) {
        Some(slot) => slot.1 = value,
        None => values.push((name.to_string(), value)),
    }
}

fn remove(values: &mut Vec<(String, Value)>, name: &str) {
    let len = values.len();
    values.retain(|(k, _)| k != name);
    assert_eq!(values.len(), len - 1, "{name}");
}

fn names(values: &[(String, Value)]) -> Vec<&str> {
    values.iter().map(|(k, _)| k.as_str()).collect()
}

#[test]
fn state_dicts_have_pytorchs_layout() {
    let ([p, _, r], opt) = adam();
    let sd = opt.state_dict();
    assert_eq!(sd.param_groups.len(), 2);
    assert_eq!(sd.param_groups[0].params, [0, 1]);
    assert_eq!(sd.param_groups[1].params, [2]);
    assert_eq!(
        names(&sd.param_groups[0].options),
        [
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay"
        ]
    );
    let o = &sd.param_groups[0].options;
    assert!(matches!(o[0].1, Value::Float(0.01)));
    assert!(
        matches!(&o[1].1, Value::Tuple(b) if matches!(b[..], [Value::Float(0.9), Value::Float(0.999)]))
    );
    assert!(matches!(o[4].1, Value::Bool(true)));
    assert!(matches!(o[6].1, Value::None));
    assert!(matches!(o[7].1, Value::Bool(false)));
    assert!(matches!(o[9].1, Value::None));
    assert!(matches!(o[10].1, Value::Bool(false)));
    // The parameter without a gradient has no state.
    assert_eq!(sd.state.keys().copied().collect::<Vec<_>>(), [0, 2]);
    assert_eq!(
        names(&sd.state[&0]),
        ["step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"]
    );
    // The state is the optimizer's, not a copy.
    let Value::Tensor(exp_avg) = &sd.state[&0][1].1 else {
        panic!()
    };
    assert!(exp_avg.is_same(opt.state(&p).unwrap().exp_avg()));
    let Value::Tensor(step) = &sd.state[&2][0].1 else {
        panic!()
    };
    assert_eq!(step.dtype(), DType::F32);
    assert!(step.shape().is_empty());
    assert_eq!(step.item::<f32>(), 2.0);
    assert!(opt.state(&r).is_some());

    let w = Parameter::new(homura::ones([2]));
    let adamw = AdamW::new([w.clone()], 0.1).state_dict();
    let o = &adamw.param_groups[0].options;
    assert!(matches!(o[3].1, Value::Float(0.01)));
    assert!(matches!(o[10].1, Value::Bool(true)));
    assert!(adamw.state.is_empty());

    let mut sgd = Sgd::new([w.clone()], 0.1).momentum(0.9).nesterov(true);
    assert_eq!(
        names(&sgd.state_dict().param_groups[0].options),
        [
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "maximize",
            "foreach",
            "differentiable",
            "fused"
        ]
    );
    w.set_grad(Some(homura::ones([2])));
    sgd.step();
    let sd = sgd.state_dict();
    assert_eq!(names(&sd.state[&0]), ["momentum_buffer"]);
    assert!(matches!(sd.param_groups[0].options[4].1, Value::Bool(true)));
    // Without momentum there is no state.
    let mut plain = Sgd::new([w.clone()], 0.1);
    plain.step();
    assert!(plain.state_dict().state.is_empty());

    // As a Python object: {"state": {0: {...}, 2: {...}}, "param_groups": [{..., "params": [0, 1]}, ...]}.
    let Value::Dict(top) = Value::from(&adam().1.state_dict()) else {
        panic!()
    };
    assert!(!top.is_ordered());
    assert_eq!(
        format!("{:?}", top.iter().map(|(k, _)| k).collect::<Vec<_>>()),
        r#"["state", "param_groups"]"#
    );
    let Some(Value::Dict(state)) = top.get("state") else {
        panic!()
    };
    assert_eq!(
        format!("{:?}", state.iter().map(|(k, _)| k).collect::<Vec<_>>()),
        "[0, 2]"
    );
    let Some(Value::List(groups)) = top.get("param_groups") else {
        panic!()
    };
    let Value::Dict(group) = &groups[1] else {
        panic!()
    };
    let (last, params) = group.iter().last().unwrap();
    assert_eq!(format!("{last:?} {params:?}"), r#""params" List([Int(2)])"#);
}

#[test]
fn loading_copies_the_state() {
    let ([p, q, r], opt) = adam();
    let sd = opt.state_dict();
    let (p2, q2, r2) = (
        Parameter::new(p.deep_clone()),
        Parameter::new(q.deep_clone()),
        Parameter::new(r.deep_clone()),
    );
    let mut other = Adam::new([p2.clone(), q2.clone()], 0.5);
    other.add_param_group(ParamGroup::new([r2.clone()], *other.defaults()));
    other.load_state_dict(&sd).unwrap();
    assert_eq!(other.lr(0), 0.01);
    assert!(other.param_groups()[1].options.amsgrad);
    // The defaults stay.
    assert_eq!(other.defaults().lr, 0.5);
    assert!(other.state(&q2).is_none());
    let loaded = other.state(&p2).unwrap();
    let original = opt.state(&p).unwrap();
    assert_same(loaded.exp_avg(), original.exp_avg(), "exp_avg");
    assert_same(loaded.step(), original.step(), "step");
    // Writing the source changes nothing: the state is a copy.
    original.exp_avg().fill_(7.0);
    original.step().fill_(9.0);
    assert!(loaded.exp_avg().to_vec::<f32>().iter().all(|&x| x != 7.0));
    assert_eq!(loaded.step().item::<f32>(), 2.0);

    // Converted to the parameters' dtype; `step` stays F32.
    let p64 = Parameter::new(p.to(DType::F64));
    let q64 = Parameter::new(q.to(DType::F64));
    let r64 = Parameter::new(r.to(DType::F64));
    let mut f64_opt = Adam::new([p64.clone(), q64], 0.5);
    f64_opt.add_param_group(ParamGroup::new([r64], *f64_opt.defaults()));
    f64_opt.load_state_dict(&sd).unwrap();
    let s = f64_opt.state(&p64).unwrap();
    assert_eq!(s.exp_avg().dtype(), DType::F64);
    assert_eq!(s.max_exp_avg_sq().unwrap().dtype(), DType::F64);
    assert_eq!(s.step().dtype(), DType::F32);
    grads(&[&p64], 1.0);
    f64_opt.step();

    // A state tensor that requires grad or is a view is copied into a plain tensor.
    let mut sd = sd.clone();
    let base = homura::zeros([3, 2]);
    base.requires_grad_(true);
    let view = base.transpose(0, 1);
    sd.state.get_mut(&0).unwrap()[1].1 = Value::Tensor(view.clone());
    let mut third = Adam::new([p.clone(), q.clone()], 0.5);
    third.add_param_group(ParamGroup::new([r.clone()], *third.defaults()));
    third.load_state_dict(&sd).unwrap();
    let exp_avg = third.state(&p).unwrap().exp_avg();
    assert!(!exp_avg.requires_grad() && exp_avg.is_contiguous());
    assert_same(exp_avg, &view.detach(), "copied view");
}

/// Loads `sd` into a copy of [`adam`]'s optimizer and checks that it fails with an error that
/// contains `message`, and that nothing changed.
fn rejected(edit: impl FnOnce(&mut OptimizerStateDict), message: &str) {
    let ([p, q, r], mut opt) = adam();
    let before = deep_copy(&opt.state_dict());
    let handles = (
        opt.state(&p).unwrap().exp_avg().clone(),
        opt.state(&r).unwrap().step().clone(),
    );
    let mut sd = deep_copy(&opt.state_dict());
    // Different values that load: the moments, not the step counts, which must stay whole.
    for (name, v) in sd.state.values_mut().flatten() {
        if let Value::Tensor(t) = v
            && name != "step"
        {
            t.fill_(0.5);
        }
    }
    set(options(&mut sd, 0), "lr", 0.5);
    set(options(&mut sd, 1), "amsgrad", false);
    // Unedited, it loads: the errors come from the edit.
    adam().1.load_state_dict(&sd).unwrap();
    edit(&mut sd);
    let err = opt.load_state_dict(&sd).expect_err(message);
    let text = err.to_string();
    assert!(
        text.contains(message),
        "{text:?} doesn't contain {message:?}"
    );
    assert!(text.starts_with("Error(s) in loading optimizer state_dict:\n\t"));
    assert_same_state_dict(&opt.state_dict(), &before, message);
    assert!(opt.state(&p).unwrap().exp_avg().is_same(&handles.0));
    assert!(opt.state(&r).unwrap().step().is_same(&handles.1));
    assert!(opt.state(&q).is_none());
    assert!(opt.param_groups()[1].options.amsgrad);
}

fn state(sd: &mut OptimizerStateDict, n: usize) -> &mut Vec<(String, Value)> {
    sd.state.get_mut(&n).unwrap()
}

#[test]
fn loads_that_fail_change_nothing() {
    let step = |value: Value| move |sd: &mut OptimizerStateDict| set(state(sd, 0), "step", value);
    let scalar = |x: f64, dtype: DType| {
        Value::Tensor(homura::tensor_with(vec![x], dtype).reshape([0i64; 0]))
    };
    let range = "it must be a whole number from 0 to 2^53";
    rejected(
        step(scalar(-1.0, DType::F32)),
        &format!("the state of parameter 0: step is -1; {range}"),
    );
    rejected(
        step(scalar(f64::NAN, DType::F32)),
        &format!("step is NaN; {range}"),
    );
    rejected(
        step(scalar(0.5, DType::F64)),
        &format!("step is 0.5; {range}"),
    );
    rejected(step(scalar(9_007_199_254_740_994.0, DType::F64)), range);
    rejected(
        step(Value::Float(f64::INFINITY)),
        &format!("step is inf; {range}"),
    );
    rejected(step(Value::Int(-3)), &format!("step is -3; {range}"));
    rejected(
        step(Value::Int(9_007_199_254_740_993)),
        &format!("step is 9007199254740993; {range}"),
    );
    rejected(step(Value::Int(i64::MAX)), range);
    let shape = "it must be a 0-D tensor of dtype F32 or F64 on the CPU";
    rejected(
        step(Value::Tensor(homura::tensor([1.0f32]))),
        &format!("step has shape [1], dtype f32 and device cpu; {shape}"),
    );
    rejected(
        step(scalar(1.0, DType::I64)),
        "step has shape [], dtype i64 and device cpu",
    );
    rejected(step(scalar(1.0, DType::F16)), "dtype f16");
    rejected(
        step(Value::Tensor(scalar_meta())),
        "step has shape [], dtype f32 and device meta",
    );
    rejected(step(Value::from("1")), "step is a str, not a tensor");
    rejected(
        |sd| remove(state(sd, 0), "exp_avg"),
        "the state of parameter 0: exp_avg is missing",
    );
    rejected(
        |sd| state(sd, 2).retain(|(k, _)| k == "step"),
        "the state of parameter 2: exp_avg is missing\n\tthe state of parameter 2: exp_avg_sq is missing",
    );
    rejected(
        |sd| set(state(sd, 0), "momentum_buffer", homura::zeros([2, 3])),
        "momentum_buffer is not known",
    );
    rejected(
        |sd| set(state(sd, 0), "exp_avg", homura::zeros([3, 2])),
        "exp_avg has shape [3, 2], but the parameter has shape [2, 3]",
    );
    rejected(
        |sd| {
            set(
                state(sd, 0),
                "exp_avg_sq",
                homura::zeros_with([2, 3], DType::I32),
            )
        },
        "exp_avg_sq has dtype i32; it must be floating point",
    );
    rejected(
        |sd| {
            set(
                state(sd, 0),
                "exp_avg",
                homura::zeros([2, 3]).to(Device::Meta),
            )
        },
        "exp_avg is a meta tensor, which has no data to copy to a parameter on cpu",
    );
    rejected(
        |sd| set(state(sd, 0), "exp_avg", Value::None),
        "exp_avg is a None, not a tensor",
    );
    // Group 0 has amsgrad on.
    rejected(
        |sd| remove(state(sd, 0), "max_exp_avg_sq"),
        "the state of parameter 0: max_exp_avg_sq is missing",
    );
    rejected(
        |sd| state(sd, 0).push(("step".into(), Value::Int(1))),
        "step appears twice",
    );

    rejected(
        |sd| set(options(sd, 0), "momentum", 0.9),
        "parameter group 0: momentum is not known",
    );
    rejected(
        |sd| remove(options(sd, 1), "eps"),
        "parameter group 1: eps is missing",
    );
    rejected(
        |sd| set(options(sd, 0), "lr", -1.0),
        "parameter group 0: Invalid learning rate: -1",
    );
    rejected(
        |sd| set(options(sd, 1), "betas", Value::tuple([0.9, 1.0])),
        "parameter group 1: Invalid beta parameter at index 1: 1",
    );
    rejected(
        |sd| set(options(sd, 0), "lr", "0.1"),
        "lr is a str, not a number",
    );
    rejected(
        |sd| set(options(sd, 0), "lr", true),
        "lr is a bool, not a number",
    );
    rejected(
        |sd| {
            set(
                options(sd, 0),
                "lr",
                homura::tensor([0.1f32]).reshape([0i64; 0]),
            )
        },
        "lr is a Tensor, not a number",
    );
    rejected(
        |sd| set(options(sd, 0), "betas", Value::tuple([0.9])),
        "betas is a tuple, not a pair of numbers",
    );
    rejected(
        |sd| {
            set(
                options(sd, 0),
                "betas",
                Value::tuple([Value::from(0.9), Value::None]),
            )
        },
        "betas is a None, not a number",
    );
    rejected(
        |sd| set(options(sd, 0), "amsgrad", 1),
        "amsgrad is a int, not a bool",
    );
    rejected(
        |sd| set(options(sd, 0), "fused", true),
        "fused is Bool(true); homura supports only None or False",
    );
    rejected(
        |sd| set(options(sd, 0), "capturable", true),
        "capturable is Bool(true); homura supports only False",
    );
    rejected(
        |sd| set(options(sd, 0), "differentiable", true),
        "differentiable is Bool(true); homura supports only False",
    );
    rejected(
        |sd| set(options(sd, 0), "foreach", "yes"),
        "foreach is Str(\"yes\"); homura supports only None or a bool",
    );
    rejected(
        |sd| set(options(sd, 0), "decoupled_weight_decay", true),
        "decoupled_weight_decay is true: the state dict is AdamW's, this is Adam",
    );
    rejected(
        |sd| set(options(sd, 0), "param_names", Value::list(["p"])),
        "param_names is not a list of 2 names, one per parameter",
    );
    rejected(
        |sd| options(sd, 0).push(("lr".into(), Value::Float(0.1))),
        "parameter group 0: lr appears twice",
    );
    // An option problem stops the load before the state is read.
    rejected(
        |sd| {
            set(options(sd, 0), "eps", -1.0);
            remove(state(sd, 0), "exp_avg");
        },
        "Error(s) in loading optimizer state_dict:\n\tparameter group 0: Invalid epsilon value: -1",
    );

    rejected(
        |sd| sd.param_groups.push(ParamGroupRecord::default()),
        "loaded state dict has a different number of parameter groups (3, the optimizer has 2)",
    );
    rejected(
        |sd| sd.param_groups[1].params.push(3),
        "loaded state dict contains a parameter group that doesn't match the size of optimizer's \
         group: group 1 has 2 parameters, the optimizer's 1",
    );
    rejected(
        |sd| sd.param_groups[0].params = vec![0, 0],
        "parameter group 0 lists the parameter number 0, which was listed before",
    );
    rejected(
        |sd| sd.param_groups[1].params = vec![3],
        "parameter group 1 lists the parameter number 3, but there are 3 parameters",
    );
    rejected(
        |sd| {
            let s = sd.state[&0].clone();
            sd.state.insert(3, s);
        },
        "the state has an entry for the parameter number 3, but there are 3 parameters",
    );
}

fn scalar_meta() -> Tensor {
    homura::zeros([]).to(Device::Meta)
}

#[test]
fn loads_what_pytorch_loads() {
    // A state dict of another type of optimizer.
    let ([p, q, r], saved) = adam();
    let mut sgd = Sgd::new([p.clone(), q.clone()], 0.1);
    sgd.add_param_group(ParamGroup::new([r.clone()], *sgd.defaults()));
    let err = sgd.load_state_dict(&saved.state_dict()).unwrap_err();
    assert!(
        err.errors
            .contains(&"parameter group 0: momentum is missing".to_string())
    );
    assert!(
        err.errors
            .contains(&"parameter group 0: betas is not known".to_string())
    );
    let w = Parameter::new(homura::ones([2]));
    let err = Adam::new([w.clone()], 0.1)
        .load_state_dict(&AdamW::new([w.clone()], 0.1).state_dict())
        .unwrap_err();
    assert_eq!(
        err,
        LoadError {
            errors: vec![
                "parameter group 0: decoupled_weight_decay is true: the state dict is AdamW's, \
                 this is Adam"
                    .into()
            ]
        }
    );

    // Only the options of `Adam()`, as ints and a list, as older versions of PyTorch saved
    // them; the names of named parameters; a step count saved as a Python int; amsgrad off with a
    // `max_exp_avg_sq`, which is kept.
    let ([p, q, r], mut opt) = adam();
    let mut sd = opt.state_dict();
    for g in 0..2 {
        let o = options(&mut sd, g);
        o.retain(|(k, _)| ["lr", "betas", "eps", "weight_decay"].contains(&k.as_str()));
        set(o, "weight_decay", 1);
        set(o, "betas", Value::list([0.5, 0.75]));
    }
    set(options(&mut sd, 0), "param_names", Value::list(["p", "q"]));
    set(options(&mut sd, 1), "foreach", true);
    set(options(&mut sd, 1), "fused", false);
    set(state(&mut sd, 0), "step", 5);
    set(state(&mut sd, 2), "step", 0.0);
    opt.load_state_dict(&sd).unwrap();
    let o = opt.param_groups()[0].options;
    assert_eq!(
        (o.betas, o.weight_decay, o.amsgrad, o.maximize),
        ((0.5, 0.75), 1.0, false, false)
    );
    let s = opt.state(&p).unwrap();
    assert_eq!(
        (s.step().dtype(), s.step().item::<f32>()),
        (DType::F32, 5.0)
    );
    assert!(s.max_exp_avg_sq().is_some());
    assert_eq!(opt.state(&r).unwrap().step().item::<f32>(), 0.0);
    let mut boundary = opt.state_dict();
    set(state(&mut boundary, 0), "step", 9_007_199_254_740_992i64);
    opt.load_state_dict(&boundary).unwrap();
    assert_eq!(
        opt.state(&p).unwrap().step().item::<f32>(),
        9_007_199_254_740_992.0
    );
    opt.load_state_dict(&sd).unwrap();
    // Saved again without the names, which homura doesn't keep.
    assert!(!names(&opt.state_dict().param_groups[0].options).contains(&"param_names"));
    grads(&[&p, &q, &r], 1.0);
    opt.step();
    assert_eq!(opt.state(&p).unwrap().step().item::<f32>(), 6.0);
    assert_eq!(opt.state(&q).unwrap().step().item::<f32>(), 1.0);

    // Empty state, as PyTorch has before the first step, and an empty entry.
    let mut sd = opt.state_dict();
    sd.state.clear();
    sd.state.insert(1, vec![]);
    opt.load_state_dict(&sd).unwrap();
    assert!(opt.state(&p).is_none() && opt.state(&q).is_none() && opt.state(&r).is_none());

    // SGD: a `None` buffer (PyTorch before 2.0 with momentum 0) is no state; a buffer with
    // momentum 0 is kept.
    let w = Parameter::new(homura::ones([2]));
    let mut sgd = Sgd::new([w.clone()], 0.1).momentum(0.9);
    w.set_grad(Some(homura::ones([2])));
    sgd.step();
    let mut sd = sgd.state_dict();
    set(options(&mut sd, 0), "momentum", 0);
    sgd.load_state_dict(&sd).unwrap();
    assert!(sgd.state(&w).is_some());
    set(state(&mut sd, 0), "momentum_buffer", Value::None);
    sgd.load_state_dict(&sd).unwrap();
    assert!(sgd.state(&w).is_none());
}

#[test]
fn state_dicts_go_through_pt_files() {
    let ([p, q, r], mut opt) = adam();
    let sd = opt.state_dict();
    let checkpoint: io::Dict = [("optimizer", Value::from(&sd)), ("epoch", Value::from(2))]
        .into_iter()
        .collect();
    let bytes = io::save_value_to_bytes(&Value::Dict(checkpoint)).unwrap();
    let Value::Dict(loaded) = io::load_value_from_bytes(&bytes).unwrap() else {
        panic!()
    };
    let loaded = OptimizerStateDict::try_from(loaded.get("optimizer").unwrap()).unwrap();
    assert_same_state_dict(&loaded, &sd, "read back");

    // Resuming from the file continues as the optimizer does.
    let copies: Vec<Parameter> = [&p, &q, &r]
        .iter()
        .map(|x| Parameter::new(x.deep_clone()))
        .collect();
    let mut resumed: Box<dyn Optimizer> = {
        let mut o = Adam::new([copies[0].clone(), copies[1].clone()], 0.1);
        o.add_param_group(ParamGroup::new([copies[2].clone()], *o.defaults()));
        Box::new(o)
    };
    resumed.load_state_dict(&loaded).unwrap();
    for scale in [0.5, 3.0] {
        grads(&[&p, &q, &r], scale);
        opt.step();
        let refs: Vec<&Parameter> = copies.iter().collect();
        grads(&refs, scale);
        resumed.step();
    }
    for (a, b) in [&p, &q, &r].iter().zip(&copies) {
        assert_same(b, a, "resumed");
    }
    assert_same_state_dict(&resumed.state_dict(), &opt.state_dict(), "resumed");
}

#[test]
fn values_that_arent_optimizer_state_dicts() {
    let sd = Value::from(&adam().1.state_dict());
    let check = |edit: &dyn Fn(&mut io::Dict), message: &str| {
        let Value::Dict(mut top) = sd.clone() else {
            panic!()
        };
        edit(&mut top);
        let err = OptimizerStateDict::try_from(&Value::Dict(top)).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("format error: expected an optimizer state dict: {message}")
        );
    };
    let err = OptimizerStateDict::try_from(&Value::list([1])).unwrap_err();
    assert_eq!(
        err.to_string(),
        "format error: expected an optimizer state dict: it is a list, not a dict"
    );
    check(
        &|d| {
            d.insert("epoch", 3);
        },
        r#"it has the key "epoch"; it can have only "state" and "param_groups""#,
    );
    check(
        &|d| {
            d.insert(0, 3);
        },
        r#"it has the key 0; it can have only "state" and "param_groups""#,
    );
    check(
        &|d| *d = [("state", io::Dict::new())].into_iter().collect(),
        r#"it has no "param_groups""#,
    );
    check(
        &|d| {
            d.insert("state", Value::list([1]));
        },
        r#"["state"] is a list, not a dict"#,
    );
    let with_state = |key: Value, entry: Value| {
        move |d: &mut io::Dict| {
            let mut state = io::Dict::new();
            match key.clone() {
                Value::Int(i) => state.insert(i, entry.clone()),
                Value::Str(s) => state.insert(s, entry.clone()),
                _ => unreachable!(),
            };
            d.insert("state", state);
        }
    };
    check(
        &with_state(Value::Int(-1), Value::Dict(io::Dict::new())),
        r#"["state"][-1]: the parameter number -1 is negative"#,
    );
    check(
        &with_state(Value::from("0"), Value::Dict(io::Dict::new())),
        r#"["state"]["0"]: the key is a str, not a parameter number"#,
    );
    check(
        &with_state(Value::Int(0), Value::None),
        r#"["state"][0] is a None, not a dict"#,
    );
    let entry: io::Dict = [(1, 2)].into_iter().collect();
    check(
        &with_state(Value::Int(0), Value::Dict(entry)),
        r#"["state"][0] has the key 1, which is not a str"#,
    );
    check(
        &|d| {
            d.insert("param_groups", io::Dict::new());
        },
        r#"["param_groups"] is a dict, not a list"#,
    );
    check(
        &|d| {
            d.insert("param_groups", Value::list([1]));
        },
        r#"["param_groups"][0] is a int, not a dict"#,
    );
    let group = |params: Value| {
        move |d: &mut io::Dict| {
            let g: io::Dict = [("lr", Value::from(0.1)), ("params", params.clone())]
                .into_iter()
                .collect();
            d.insert("param_groups", Value::list([g]));
        }
    };
    check(
        &|d| {
            d.insert("param_groups", Value::list([io::Dict::new()]));
        },
        r#"["param_groups"][0] has no "params""#,
    );
    check(
        &group(Value::tuple([0])),
        r#"["param_groups"][0]["params"] is a tuple, not a list"#,
    );
    check(
        &group(Value::list([Value::from(0), Value::from(0.5)])),
        r#"["param_groups"][0]["params"][1] is a float, not an int"#,
    );
    check(
        &group(Value::list([-2])),
        r#"["param_groups"][0]["params"][0]: the parameter number -2 is negative"#,
    );

    // The structure is all that is checked here.
    let Value::Dict(mut top) = sd.clone() else {
        panic!()
    };
    let g: io::Dict = [("anything", Value::None), ("params", Value::list([7]))]
        .into_iter()
        .collect();
    top.insert("param_groups", Value::list([g]));
    let parsed = OptimizerStateDict::try_from(&Value::Dict(top)).unwrap();
    assert_eq!(parsed.param_groups[0].params, [7]);
    assert_eq!(names(&parsed.param_groups[0].options), ["anything"]);
}

#[test]
fn optimizers_load_through_references() {
    fn round_trip<O: Optimizer>(mut opt: O) {
        let sd = opt.state_dict();
        opt.load_state_dict(&sd).unwrap();
    }
    let (_, mut opt) = adam();
    round_trip(&mut opt);
    round_trip(Box::new(opt) as Box<dyn Optimizer>);
    let err = LoadError {
        errors: vec!["a".into(), "b".into()],
    };
    assert_eq!(
        err.to_string(),
        "Error(s) in loading optimizer state_dict:\n\ta\n\tb"
    );
}
