//! Process-wide defaults. Kept in a separate test binary with a single test because the
//! defaults are global state.

use homura::{DType, Device};

#[test]
fn default_dtype_and_device() {
    assert_eq!(homura::default_dtype(), DType::F32);
    homura::set_default_dtype(DType::F64);
    assert_eq!(homura::zeros([2]).dtype(), DType::F64);
    assert_eq!(homura::tensor([1.0, 2.0]).dtype(), DType::F64);
    assert_eq!((homura::tensor([1, 2]) * 0.5).dtype(), DType::F64);
    assert_eq!(homura::tensor([1, 2]).dtype(), DType::I64);
    homura::set_default_dtype(DType::F32);

    homura::set_default_device(Device::Meta);
    assert_eq!(homura::ones([2]).device(), Device::Meta);
    {
        let _cpu = homura::device_scope(Device::Cpu);
        assert_eq!(homura::ones([2]).device(), Device::Cpu);
    }
    assert_eq!(homura::ones([2]).device(), Device::Meta);
    homura::set_default_device(Device::Cpu);

    let result = std::panic::catch_unwind(|| homura::set_default_dtype(DType::I64));
    assert!(result.is_err());
}
