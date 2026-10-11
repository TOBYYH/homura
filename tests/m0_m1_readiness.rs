//! Regression checks for the final M0 fixes before M1. Run in debug and release profiles.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::{DType, Device};

#[test]
fn integer_composites_accept_fractional_upper_boundary_coefficients() {
    let mut failures = vec![];
    for device in [Device::Cpu, Device::Meta] {
        for (dtype, alpha, expected) in [
            (DType::U8, 255.5, 255),
            (DType::I8, 127.5, 127),
            (DType::I16, 32767.5, 32767),
            (DType::I32, 2147483647.5, 2147483647),
        ] {
            let m = homura::ones_with([1, 1], (dtype, device));
            let v = homura::ones_with([1], (dtype, device));
            let b = homura::ones_with([1, 1, 1], (dtype, device));
            for op in ["addmm", "addmv", "addbmm", "baddbmm"] {
                let result = catch_unwind(AssertUnwindSafe(|| match op {
                    "addmm" => m.addmm(&m, &m, 0, alpha),
                    "addmv" => v.addmv(&m, &v, 0, alpha),
                    "addbmm" => m.addbmm(&b, &b, 0, alpha),
                    "baddbmm" => b.baddbmm(&b, &b, 0, alpha),
                    _ => unreachable!(),
                }));
                match result {
                    Ok(out) => {
                        assert_eq!(out.dtype(), dtype);
                        if device == Device::Cpu {
                            assert_eq!(out.to(DType::I64).to_vec::<i64>(), [expected]);
                        }
                    }
                    Err(_) => failures.push((op, dtype, device)),
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "valid coefficients rejected: {failures:?}"
    );
}

#[test]
fn integer_composites_reject_two_to_the_63_float_coefficient() {
    let mut accepted = vec![];
    for device in [Device::Cpu, Device::Meta] {
        let m = homura::ones_with([1, 1], (DType::I64, device));
        if catch_unwind(AssertUnwindSafe(|| {
            m.addmm(&m, &m, 0, 2f64.powi(63));
        }))
        .is_ok()
        {
            accepted.push(device);
        }
    }
    assert!(
        accepted.is_empty(),
        "2^63 coefficient accepted on {accepted:?}"
    );
}

#[test]
fn addr_promotes_all_three_input_dtypes_before_computation() {
    let mut failures = vec![];
    for device in [Device::Cpu, Device::Meta] {
        for (input_dtype, left_dtype, right_dtype, expected) in [
            (DType::F32, DType::F32, DType::F64, DType::F64),
            (DType::F64, DType::F32, DType::F32, DType::F64),
            (DType::I32, DType::I64, DType::I64, DType::I64),
            (DType::I32, DType::F32, DType::F32, DType::F32),
        ] {
            let input = homura::ones_with([1, 1], (input_dtype, device));
            let left = homura::ones_with([1], (left_dtype, device));
            let right = homura::ones_with([1], (right_dtype, device));
            match catch_unwind(AssertUnwindSafe(|| input.addr(&left, &right, 1, 1))) {
                Ok(out) => {
                    assert_eq!(out.dtype(), expected);
                    if device == Device::Cpu {
                        assert_eq!(out.to(DType::F64).to_vec::<f64>(), [2.0]);
                    }
                }
                Err(_) => failures.push((input_dtype, left_dtype, right_dtype, device)),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "valid mixed inputs rejected: {failures:?}"
    );

    // Promote before the product too: a float32 product would discard the 2^-46 term.
    let input = homura::zeros_with([1, 1], DType::F64);
    let vector = homura::tensor_with([1.0 + 2f32.powi(-23)], DType::F32);
    assert_eq!(
        input.addr(&vector, &vector, 0, 1).to_vec::<f64>(),
        [(1.0 + 2f64.powi(-23)).powi(2)]
    );
}

#[test]
fn integer_coefficient_bounds_preserve_lower_limits_and_reject_nonfinite_values() {
    for device in [Device::Cpu, Device::Meta] {
        for (dtype, lower, upper) in [
            (DType::U8, 0.0, 256.0),
            (DType::I8, -128.0, 128.0),
            (DType::I16, -32768.0, 32768.0),
            (DType::I32, -2147483648.0, 2147483648.0),
            (DType::I64, -2f64.powi(63), 2f64.powi(63)),
        ] {
            let m = homura::ones_with([1, 1], (dtype, device));
            let below = if dtype == DType::I64 {
                lower - 2048.0
            } else {
                lower - 0.5
            };
            for invalid in [below, upper, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for (beta, alpha) in [(0.0, invalid), (invalid, 0.0)] {
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| m.addmm(&m, &m, beta, alpha))).is_err(),
                        "{dtype} accepted beta={beta}, alpha={alpha} on {device}"
                    );
                }
            }
            let out = m.addmm(&m, &m, 0, lower);
            assert_eq!(out.dtype(), dtype);
            if device == Device::Cpu {
                assert_eq!(out.to(DType::I64).to_vec::<i64>(), [lower as i64]);
            }
        }
    }
    let m = homura::ones_with([1, 1], DType::I64);
    let largest = 2f64.powi(63) - 1024.0;
    assert_eq!(
        m.addmm(&m, &m, 0, largest).to_vec::<i64>(),
        [largest as i64]
    );
    assert_eq!(m.addmm(&m, &m, 0, i64::MAX).to_vec::<i64>(), [i64::MAX]);
}

#[test]
fn addr_promotes_scalar_inputs_and_preserves_validation_when_beta_is_zero() {
    for device in [Device::Cpu, Device::Meta] {
        let input = homura::full_with((), 1, (DType::F64, device));
        let vector = homura::ones_with([2], (DType::F32, device));
        let out = input.addr(&vector, &vector, 1, 1);
        assert_eq!(out.dtype(), DType::F64);
        assert_eq!(out.shape(), [2, 2]);
        if device == Device::Cpu {
            assert_eq!(out.to_vec::<f64>(), [2.0; 4]);
        }
        let ignored = homura::full_with([2, 2], f64::NAN, (DType::F64, device));
        let out = ignored.addr(&vector, &vector, 0, 1);
        assert_eq!(out.dtype(), DType::F64);
        if device == Device::Cpu {
            assert_eq!(out.to_vec::<f64>(), [1.0; 4]);
        }
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                homura::ones_with([3, 3], (DType::F64, device)).addr(&vector, &vector, 0, 1);
            }))
            .is_err()
        );
        let empty = homura::ones_with([0], (DType::F32, device));
        let out = input.addr(&empty, &vector, 1, 1);
        assert_eq!(out.shape(), [0, 2]);
        assert_eq!(out.dtype(), DType::F64);
    }
    let cpu = homura::ones_with([1, 1], DType::F32);
    let meta = homura::ones_with([1], (DType::F64, Device::Meta));
    let vector = homura::ones_with([1], DType::F32);
    assert!(catch_unwind(AssertUnwindSafe(|| cpu.addr(&vector, &meta, 0, 1))).is_err());
    // Matrix-multiply composites keep their stricter same-dtype signature.
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            homura::ones_with([1, 1], DType::F64).addmm(&cpu, &cpu, 0, 1);
        }))
        .is_err()
    );
}
