//! The dtype of Adam's `step` follows the default dtype, which is global state, so these tests
//! have their own binary and take turns.

#[path = "common/optim_state.rs"]
mod optim_state;

use std::path::Path;
use std::sync::Mutex;

use homura::io::Value;
use homura::nn::Parameter;
use homura::optim::{Adam, Optimizer, TypedOptimizer};
use homura::{DType, Device};

/// Held by each test while it changes the default dtype.
static DEFAULT_DTYPE: Mutex<()> = Mutex::new(());

#[test]
fn adam_step_follows_the_default_dtype() {
    let _turn = DEFAULT_DTYPE.lock().unwrap_or_else(|e| e.into_inner());
    homura::set_default_dtype(DType::F64);
    // PyTorch's `_get_scalar_dtype()`: F64 with the default dtype F64, whatever the parameter.
    let p = Parameter::new(homura::ones_with([2], DType::F32));
    let mut opt = Adam::new([p.clone()], 0.1);
    p.set_grad(Some(homura::ones_with([2], DType::F32)));
    opt.step();
    let step = opt.state(&p).unwrap().step();
    assert!(step.dtype() == DType::F64 && step.device() == Device::Cpu);
    assert_eq!(step.item::<f64>(), 1.0);
    // F32 otherwise, also with a half default dtype; existing state keeps its dtype.
    homura::set_default_dtype(DType::BF16);
    let q = Parameter::new(homura::ones_with([2], DType::F32));
    opt.add_param_group(homura::optim::ParamGroup::new([q.clone()], *opt.defaults()));
    q.set_grad(Some(homura::ones_with([2], DType::F32)));
    opt.step();
    assert_eq!(opt.state(&q).unwrap().step().dtype(), DType::F32);
    assert_eq!(opt.state(&p).unwrap().step().item::<f64>(), 2.0);
    homura::set_default_dtype(DType::F32);
}

/// M3 step 6: PyTorch's F64 step counts resume bit for bit, and a step count saved as a Python
/// number becomes an F64 tensor, as new state would have.
#[test]
fn state_dicts_keep_f64_step_counts() {
    let _turn = DEFAULT_DTYPE.lock().unwrap_or_else(|e| e.into_inner());
    homura::set_default_dtype(DType::F64);
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("m3_optim_state");
    let name = "adam_default_f64_f64";
    let cases = optim_state::cases();
    let (_, case) = cases.iter().find(|(n, _)| n == name).unwrap();
    optim_state::run(name, case, &out);

    let p = Parameter::new(homura::ones_with([2], DType::F32));
    let mut opt = Adam::new([p.clone()], 0.1);
    p.set_grad(Some(homura::ones_with([2], DType::F32)));
    opt.step();
    let mut sd = opt.state_dict();
    let step = &sd.state[&0][0].1;
    assert!(matches!(step, Value::Tensor(t) if t.dtype() == DType::F64));
    sd.state.get_mut(&0).unwrap()[0].1 = Value::Int(4);
    opt.load_state_dict(&sd).unwrap();
    let step = opt.state(&p).unwrap().step();
    assert_eq!((step.dtype(), step.item::<f64>()), (DType::F64, 4.0));
    // A loaded F32 step count stays F32.
    sd.state.get_mut(&0).unwrap()[0].1 = Value::Tensor(homura::zeros_with([], DType::F32));
    opt.load_state_dict(&sd).unwrap();
    assert_eq!(opt.state(&p).unwrap().step().dtype(), DType::F32);
    homura::set_default_dtype(DType::F32);
}
