//! The dtype of Adam's `step` follows the default dtype, which is global state, so this test
//! has its own binary.

use homura::nn::Parameter;
use homura::optim::{Adam, Optimizer};
use homura::{DType, Device};

#[test]
fn adam_step_follows_the_default_dtype() {
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
