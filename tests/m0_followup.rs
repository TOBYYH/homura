//! Checks from the M0 follow-up review.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::{DType, Device, Tensor};

fn rejects(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

fn file(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn all_noncanonical_bool_bytes_are_rejected_without_typed_reads() {
    let header = r#"{"x":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}}"#;
    for byte in 2..=255 {
        assert!(homura::load_safetensors_from_bytes(&file(header, &[byte])).is_err());
    }
    for byte in 0..=1 {
        let loaded = homura::load_safetensors_from_bytes(&file(header, &[byte])).unwrap();
        assert_eq!(loaded[0].1.to_vec::<bool>(), [byte == 1]);
    }
}

#[test]
fn safetensors_rejects_huge_descriptors_before_allocation() {
    for header in [
        r#"{"x":{"dtype":"F64","shape":[18446744073709551615],"data_offsets":[0,0]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[18446744073709551614,18446744073709551615]}}"#,
    ] {
        assert!(homura::load_safetensors_from_bytes(&file(header, &[])).is_err());
    }
    let header = r#"{"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#;
    assert!(homura::load_safetensors_from_bytes(&file(header, &[])).is_ok());
}

#[test]
fn float_to_integer_follows_the_chosen_defined_policy() {
    // The policy is intentionally tested independently of a PyTorch compatibility claim.
    let x = homura::tensor_with([f64::NAN, f64::INFINITY, f64::NEG_INFINITY], DType::F64);
    assert_eq!(x.to(DType::I64).to_vec::<i64>(), [0, i64::MAX, i64::MIN]);
    assert_eq!(x.to(DType::U8).to_vec::<u8>(), [0, 255, 0]);
    assert_eq!(
        homura::tensor_with([-2147483649.0f64], DType::F64)
            .to(DType::I32)
            .to_vec::<i32>(),
        [i32::MAX]
    );
}

#[test]
fn composite_integer_coefficients_preserve_dtype_on_cpu_and_meta() {
    for device in [Device::Cpu, Device::Meta] {
        let x = homura::ones_with([1, 1], (DType::I32, device));
        let out = x.addmm(&x, &x, 2, 3);
        assert_eq!(out.dtype(), DType::I32);
        if device == Device::Cpu {
            assert_eq!(out.to_vec::<i32>(), [5]);
        }
    }
}

#[test]
fn uniform_rejects_overflowing_f64_interval_before_device_dispatch() {
    let accepted: Vec<_> = [Device::Cpu, Device::Meta]
        .into_iter()
        .filter(|&device| {
            !rejects(|| {
                let x = homura::zeros_with([8], (DType::F64, device));
                x.uniform_(-1e308, 1e308);
                if device == Device::Cpu {
                    eprintln!("wide uniform samples: {:?}", x.to_vec::<f64>());
                }
            })
        })
        .collect();
    assert!(
        accepted.is_empty(),
        "invalid interval accepted on {accepted:?}"
    );
}

#[test]
fn uniform_rejects_bounds_outside_half_range() {
    let accepted: Vec<_> = [Device::Cpu, Device::Meta]
        .into_iter()
        .filter(|&device| {
            !rejects(|| {
                let x = homura::zeros_with([8], (DType::F16, device));
                x.uniform_(65520.0, 65521.0);
                if device == Device::Cpu {
                    eprintln!("out-of-range half samples: {:?}", x.to_vec::<f32>());
                }
            })
        })
        .collect();
    assert!(
        accepted.is_empty(),
        "invalid half bounds accepted on {accepted:?}"
    );
}

#[test]
fn randint_validates_float_destination_range_and_interval() {
    let mut accepted = vec![];
    for device in [Device::Cpu, Device::Meta] {
        for (dtype, low, high) in [
            (DType::F16, 70000, 70001),
            (DType::F32, (1 << 24) + 3, (1 << 24) + 4),
        ] {
            if !rejects(|| {
                let x = homura::randint_with(low, high, [8], (dtype, device));
                if device == Device::Cpu {
                    eprintln!("{dtype} randint samples: {:?}", x.to_vec::<f32>());
                }
            }) {
                accepted.push((dtype, device));
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "invalid intervals accepted: {accepted:?}"
    );
}

#[test]
fn integer_matrix_composites_validate_and_convert_coefficients() {
    let mut wrong = vec![];
    for device in [Device::Cpu, Device::Meta] {
        let m = homura::ones_with([1, 1], (DType::I32, device));
        let v = homura::ones_with([1], (DType::I32, device));
        let b = homura::ones_with([1, 1, 1], (DType::I32, device));
        for (beta, alpha) in [(1.0, 0.5), (0.5, 1.0)] {
            let outputs: [(&str, Tensor); 4] = [
                ("addmm", m.addmm(&m, &m, beta, alpha)),
                ("addmv", v.addmv(&m, &v, beta, alpha)),
                ("baddbmm", b.baddbmm(&b, &b, beta, alpha)),
                ("addbmm", m.addbmm(&b, &b, beta, alpha)),
            ];
            for (name, out) in outputs {
                if out.dtype() != DType::I32 {
                    wrong.push(format!(
                        "{name}({beta}, {alpha}) on {device}: {}",
                        out.dtype()
                    ));
                    if device == Device::Cpu {
                        eprintln!("{name} samples: {:?}", out.to_vec::<f32>());
                    }
                } else if device == Device::Cpu {
                    assert_eq!(out.to_vec::<i32>(), [1], "{name}");
                }
            }
            // Unlike addmm/addmv/addbmm/baddbmm, PyTorch addr rejects float coefficients for
            // integral input, even when the coefficient happens to be an integer value.
            if !rejects(|| {
                m.addr(&v, &v, beta, alpha);
            }) {
                wrong.push(format!(
                    "addr({beta}, {alpha}) on {device}: float coefficients accepted"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "wrong output types: {wrong:?}");
}

#[test]
fn empty_extreme_stride_view_can_be_read_and_copied() {
    let view = homura::zeros_with([0], DType::I64).as_strided([0, 2], [0, usize::MAX], usize::MAX);
    assert!(view.to_vec::<i64>().is_empty());
    assert!(view.deep_clone().to_vec::<i64>().is_empty());
}

#[test]
fn safetensors_rejects_nonhex_unicode_escape_digits() {
    let header = r#"{"\u+001":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    assert!(homura::load_safetensors_from_bytes(&file(header, &[7])).is_err());
}

#[test]
fn composite_coefficients_are_range_checked_for_integer_inputs() {
    for device in [Device::Cpu, Device::Meta] {
        let m = homura::ones_with([1, 1], (DType::I32, device));
        let v = homura::ones_with([1], (DType::I32, device));
        assert!(rejects(|| {
            m.addmm(&m, &m, 1, 1i64 << 40);
        }));
        assert!(rejects(|| {
            m.addmm(&m, &m, f64::NAN, 1);
        }));
        assert!(rejects(|| {
            m.addr(&v, &v, 1, true);
        }));
        let h = homura::ones_with([1, 1], (DType::F16, device));
        // Only addr checks floating point coefficients against the dtype.
        assert_eq!(h.addmm(&h, &h, 1, 1e5).dtype(), DType::F16);
        assert!(rejects(|| {
            h.addr(&h.flatten(0, -1), &h.flatten(0, -1), 1, 1e5);
        }));
    }
    let m = homura::ones_with([1, 1], DType::I32);
    assert_eq!(m.addmm(&m, &m, 1, -2.7).to_vec::<i32>(), [-1]);
    let v = m.flatten(0, -1);
    assert_eq!(m.addr(&v, &v, 2, 3).to_vec::<i32>(), [5]);
    let u = homura::ones_with([1, 1], DType::U8);
    assert_eq!(u.addmm(&u, &u, 1, -1).to_vec::<u8>(), [0]);
    assert!(rejects(|| {
        u.addmm(&u, &u, 1, -1.0);
    }));
    let b = homura::tensor([true]);
    assert_eq!(
        b.unsqueeze(0).addr(&b, &b, true, 2).to_vec::<bool>(),
        [true]
    );
}

#[test]
fn random_float_bounds_accept_the_supported_range() {
    let x = homura::zeros_with([64], DType::F16);
    x.uniform_(-65504.0, 0.0);
    assert!(
        x.to_vec::<f32>()
            .iter()
            .all(|v| (-65504.0..0.0).contains(v))
    );
    assert!(rejects(|| {
        homura::zeros_with([1], DType::F16).uniform_(-60000.0, 60000.0);
    }));
    for (dtype, exact) in [
        (DType::F16, 1 << 11),
        (DType::BF16, 1 << 8),
        (DType::F32, 1 << 24),
    ] {
        let v = homura::randint_with(exact - 1, exact + 1, [64], dtype)
            .to(DType::F64)
            .to_vec::<f64>();
        assert!(
            v.iter()
                .all(|&s| s == (exact - 1) as f64 || s == exact as f64)
        );
        assert!(rejects(|| {
            homura::randint_with(exact - 1, exact + 2, [1], dtype);
        }));
        assert!(rejects(|| {
            homura::randint_with(-exact - 1, 0, [1], (dtype, Device::Meta));
        }));
    }
    let max = 1i64 << 53;
    assert_eq!(
        homura::randint_with(max, max + 1, [2], DType::F64).to_vec::<f64>(),
        [max as f64; 2]
    );
}
