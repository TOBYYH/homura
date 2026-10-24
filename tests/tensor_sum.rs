//! M3 step 0e: `Sum` and `Product` for `Tensor`, which follow Python's `sum(tensors)` and
//! `math.prod(tensors)`. The reference behavior is asserted by `tools/golden/check_m3_sum.py`, in
//! the same order.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

use homura::{DType, Device, Tensor, tensor_with};

const DTYPES: [DType; 10] = [
    DType::Bool,
    DType::U8,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

fn sum(items: &[Tensor]) -> Tensor {
    items.iter().sum()
}

fn prod(items: &[Tensor]) -> Tensor {
    items.iter().product()
}

fn values(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

fn edge_names(t: &Tensor) -> Vec<Option<&'static str>> {
    t.grad_fn()
        .unwrap()
        .next_functions()
        .iter()
        .map(|e| e.as_ref().map(|(f, _)| f.name()))
        .collect()
}

#[test]
fn dtypes_of_sums_and_products() {
    // A single tensor keeps its dtype, except that bool becomes I64; an I64 tensor as the start
    // would promote zero-dimensional integer tensors.
    for dtype in DTYPES {
        for shape in [&[][..], &[2]] {
            let x = homura::full_with(shape, 3, dtype);
            let expected = if dtype == DType::Bool {
                DType::I64
            } else {
                dtype
            };
            for y in [sum(slice::from_ref(&x)), prod(slice::from_ref(&x))] {
                assert_eq!(
                    (y.dtype(), y.shape()),
                    (expected, x.shape()),
                    "{dtype} {shape:?}"
                );
                assert_eq!(values(&y), values(&x));
            }
            // Owned items as well as borrowed ones; a bool sum counts, unlike `Bool + Bool`.
            let w = x.to(expected);
            let owned: Tensor = vec![x.clone(), x.clone()].into_iter().sum();
            assert_eq!(
                (owned.dtype(), values(&owned)),
                (expected, values(&(&w + &w)))
            );
            let owned: Tensor = vec![x.clone(), x.clone()].into_iter().product();
            assert_eq!(
                (owned.dtype(), values(&owned)),
                (expected, values(&(&w * &w)))
            );
        }
    }
    // Three tensors: [2, 3, 4] (bool: [true, true, false]).
    for dtype in DTYPES {
        if dtype == DType::Bool {
            let items = [true, true, false].map(|v| tensor_with(v, dtype));
            assert_eq!(
                (values(&sum(&items)), values(&prod(&items))),
                (vec![2.0], vec![0.0])
            );
            assert_eq!(
                (sum(&items).dtype(), prod(&items).dtype()),
                (DType::I64, DType::I64)
            );
        } else {
            let items = [2, 3, 4].map(|v| tensor_with(v, dtype));
            assert_eq!(
                (values(&sum(&items)), values(&prod(&items))),
                (vec![9.0], vec![24.0])
            );
            assert_eq!((sum(&items).dtype(), prod(&items).dtype()), (dtype, dtype));
        }
    }
    // Integer sums wrap in the inputs' dtype.
    let i8s = [tensor_with(100, DType::I8), tensor_with(100, DType::I8)];
    assert_eq!(sum(&i8s).dtype(), DType::I8);
    assert_eq!(sum(&i8s).item::<i8>(), -56);
    assert_eq!(prod(&i8s).item::<i8>(), 16);
}

#[test]
fn sums_run_from_left_to_right() {
    let big = tensor_with(65504.0, DType::F16);
    let neg = -&big;
    assert_eq!(
        values(&sum(&[big.clone(), big.clone(), neg.clone()])),
        [f64::INFINITY]
    );
    assert_eq!(values(&sum(&[neg, big.clone(), big])), [65504.0]);
}

#[test]
fn mixed_dtypes_promote_step_by_step() {
    let check = |items: &[Tensor], dtype: DType, expected: &[f64]| {
        let y = sum(items);
        assert_eq!((y.dtype(), values(&y)), (dtype, expected.to_vec()));
    };
    check(
        &[
            tensor_with([1, 2], DType::I32),
            tensor_with(0.5, DType::F32),
        ],
        DType::F32,
        &[1.5, 2.5],
    );
    check(
        &[tensor_with(1, DType::I64), tensor_with([2], DType::F16)],
        DType::F16,
        &[3.0],
    );
    check(
        &[tensor_with([1], DType::F32), tensor_with(2, DType::F64)],
        DType::F32,
        &[3.0],
    );
    check(
        &[tensor_with([1], DType::I32), tensor_with(2, DType::I64)],
        DType::I32,
        &[3.0],
    );
    check(
        &[tensor_with([true], DType::Bool), tensor_with(2, DType::I8)],
        DType::I64,
        &[3.0],
    );
    let y = prod(&[tensor_with([2], DType::I16), tensor_with(1.5, DType::F64)]);
    assert_eq!((y.dtype(), values(&y)), (DType::F64, vec![3.0]));
}

#[test]
fn meta_inputs_keep_their_device() {
    // The start is a scalar, not a tensor on the default device (the CPU here), so tensors on
    // another device add up there.
    assert_eq!(homura::default_device(), Device::Cpu);
    let m = homura::ones_with([2], Device::Meta);
    for y in [sum(&[m.clone(), m.clone()]), prod(&[m.clone(), m.clone()])] {
        assert_eq!((y.device(), y.shape()), (Device::Meta, m.shape()));
    }
    // Tensors on different devices don't add up, as elsewhere.
    let mixed = [homura::ones([2]), m];
    let caught = catch_unwind(AssertUnwindSafe(|| sum(&mixed)));
    assert!(caught.is_err());
}

#[test]
fn nodes_and_gradients() {
    let (a, b) = (homura::tensor(2.0), homura::tensor(3.0));
    a.requires_grad_(true);
    b.requires_grad_(true);
    for (y, name) in [
        (sum(slice::from_ref(&a)), "AddBackward0"),
        (prod(slice::from_ref(&a)), "MulBackward0"),
    ] {
        assert_eq!(y.grad_fn().unwrap().name(), name);
        assert_eq!(edge_names(&y), [Some("AccumulateGrad"), None]);
        assert!(!y.is_same(&a) && y.item::<f32>() == 2.0);
    }
    sum(&[a.clone(), b.clone(), a.clone()]).backward();
    assert_eq!(
        (
            a.grad().unwrap().item::<f32>(),
            b.grad().unwrap().item::<f32>()
        ),
        (2.0, 1.0)
    );
    a.set_grad(None);
    b.set_grad(None);
    prod(&[a.clone(), b.clone()]).backward();
    assert_eq!(
        (
            a.grad().unwrap().item::<f32>(),
            b.grad().unwrap().item::<f32>()
        ),
        (3.0, 2.0)
    );
}

#[test]
fn inputs_are_unchanged() {
    let x = homura::tensor([1.0, 2.0]);
    for y in [sum(slice::from_ref(&x)), prod(slice::from_ref(&x))] {
        y.add_(10);
        assert_eq!(values(&x), [1.0, 2.0]);
    }
    assert!(x.grad_fn().is_none());
}

#[test]
fn empty_iterators() {
    // Python's int 0 and 1, as zero-dimensional I64 tensors on the default device.
    for (y, value) in [(sum(&[]), 0), (prod(&[]), 1)] {
        assert_eq!(
            (y.dtype(), y.shape(), y.device()),
            (DType::I64, [].into(), Device::Cpu)
        );
        assert_eq!(y.item::<i64>(), value);
    }
    let y = homura::with_device(Device::Meta, || {
        std::iter::empty::<Tensor>().sum::<Tensor>()
    });
    assert_eq!((y.dtype(), y.device()), (DType::I64, Device::Meta));
    let y: Tensor = homura::with_device(Device::Meta, || std::iter::empty::<&Tensor>().product());
    assert_eq!((y.dtype(), y.device()), (DType::I64, Device::Meta));
}
