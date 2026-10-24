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
    device_closures();

    let result = std::panic::catch_unwind(|| homura::set_default_dtype(DType::I64));
    assert!(result.is_err());
}

/// `with_device`, also over a process-wide default other than `Cpu`.
fn device_closures() {
    let x = homura::with_device(Device::Meta, || {
        assert_eq!(homura::default_device(), Device::Meta);
        homura::zeros([2])
    });
    assert_eq!(x.device(), Device::Meta);
    assert_eq!(homura::default_device(), Device::Cpu);
    homura::set_default_device(Device::Meta);
    // Nested closures and guards; each restores what it found.
    let (inner, outer) = homura::with_device(Device::Cpu, || {
        let inner = homura::with_device(Device::Meta, || homura::ones([2]));
        {
            let _scope = homura::device_scope(Device::Meta);
            assert_eq!(homura::rand([2]).device(), Device::Meta);
        }
        (inner, homura::ones([2]))
    });
    assert_eq!(
        (inner.device(), outer.device()),
        (Device::Meta, Device::Cpu)
    );
    let _scope = homura::device_scope(Device::Cpu);
    homura::with_device(Device::Meta, || {
        assert_eq!(homura::default_device(), Device::Meta)
    });
    assert_eq!(homura::default_device(), Device::Cpu);
    drop(_scope);
    assert_eq!(homura::default_device(), Device::Meta);
    // Borrowed and moved captures; operations on existing tensors keep their device.
    let source = homura::with_device(Device::Cpu, || homura::ones([3]));
    let (sum, zeros) = homura::with_device(Device::Meta, || {
        (&source + 1.0, homura::zeros([source.size(0)]))
    });
    assert_eq!((sum.device(), zeros.device()), (Device::Cpu, Device::Meta));
    let shape = vec![2usize, 3];
    let y = homura::with_device(Device::Cpu, move || homura::zeros(shape));
    assert_eq!((y.device(), y.shape().to_vec()), (Device::Cpu, vec![2, 3]));
    // A panic restores the default device; the scope is per thread.
    let result = std::panic::catch_unwind(|| homura::with_device(Device::Cpu, || panic!("inside")));
    assert!(result.is_err());
    assert_eq!(homura::default_device(), Device::Meta);
    homura::with_device(Device::Cpu, || {
        let other = std::thread::spawn(homura::default_device).join().unwrap();
        assert_eq!(other, Device::Meta);
    });
    homura::set_default_device(Device::Cpu);
}
