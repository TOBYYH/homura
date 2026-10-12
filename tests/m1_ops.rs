//! Forward behaviour of the operators M1 adds for backward formulas (`erf`, `sign`, `index_add`,
//! `gather`, `scatter_add`, `select_scatter` and `slice_scatter`) that the golden data doesn't
//! cover: argument checks with PyTorch's messages, the precision of repeated positions, empty and
//! zero-dimensional tensors, the meta device and the in-place variants.
//! `tools/golden/check_m1_formulas.py` checks the PyTorch side of these cases. Run in debug and
//! release profiles.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::{DType, Device, Tensor};

fn panic_message(f: impl FnOnce() -> Tensor) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

fn vals(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec()
}

fn ints(values: &[i64]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

/// An expected panic message and the call that should panic with it.
type Case<'a> = (&'static str, Box<dyn Fn() -> Tensor + 'a>);

// ---------------------------------------------------------------- erf and sign

#[test]
fn erf_promotes_integers_and_sign_keeps_the_dtype() {
    use DType::*;
    for device in [Device::Cpu, Device::Meta] {
        for (dtype, erf_dtype) in [
            (I64, F32),
            (I32, F32),
            (U8, F32),
            (Bool, F32),
            (F16, F16),
            (BF16, BF16),
            (F64, F64),
        ] {
            let x = homura::zeros_with([2], (dtype, device));
            assert_eq!(x.erf().dtype(), erf_dtype, "erf {dtype}");
            assert_eq!(x.sign().dtype(), dtype, "sign {dtype}");
            assert_eq!(x.sign().device(), device);
        }
    }
    let x = homura::tensor([1i64, -3, 0]);
    assert_eq!(
        vals(&x.erf()),
        vals(&homura::tensor([1.0f32, -3.0, 0.0]).erf())
    );
}

#[test]
fn sign_of_special_values() {
    // NaN and both zeros give +0.0, as in PyTorch.
    let x = homura::tensor([f32::NAN, -0.0, 0.0, -2.5, f32::INFINITY, f32::NEG_INFINITY]);
    let s = x.sign().to_vec::<f32>();
    assert_eq!(s, [0.0, 0.0, 0.0, -1.0, 1.0, -1.0]);
    assert!(s[..3].iter().all(|v| v.is_sign_positive()));
    let i8s = Tensor::from_slice(&[-128i8, 0, 127], [3]);
    assert_eq!(i8s.sign().to_vec::<i8>(), [-1, 0, 1]);
    let u8s = Tensor::from_slice(&[0u8, 255], [2]);
    assert_eq!(u8s.sign().to_vec::<u8>(), [0, 1]);
    let bools = homura::tensor([true, false]);
    assert_eq!(bools.sign().to_vec::<bool>(), [true, false]);
    let erf = homura::tensor([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0]).erf();
    let e = erf.to_vec::<f32>();
    assert!(e[0].is_nan());
    assert_eq!(&e[1..], [1.0, -1.0, -0.0]);
    assert!(e[3].is_sign_negative());
}

// ---------------------------------------------------------------- index_add

#[test]
fn index_add_checks_its_arguments_like_pytorch() {
    let x = homura::zeros([3, 4]);
    let src = homura::ones([2, 4]);
    let idx = ints(&[0, 2]);
    let cases: Vec<Case> = vec![
        (
            "index_add(): Index is supposed to be a vector",
            Box::new(|| x.index_add(0, &homura::tensor([[0i64, 2]]), &src)),
        ),
        (
            "index_add(): Expected dtype i32 or i64 for index, got f32",
            Box::new(|| x.index_add(0, &idx.to(DType::F32), &src)),
        ),
        (
            "index_add(): Expected dtype i32 or i64 for index, got i16",
            Box::new(|| x.index_add(0, &idx.to(DType::I16), &src)),
        ),
        (
            "index_add() cannot be applied to a 0-dim tensor.",
            Box::new(|| homura::tensor(1.0).index_add(0, &ints(&[0]), &homura::tensor(5.0))),
        ),
        (
            "Dimension out of range (expected to be in range of [-2, 1], but got 2)",
            Box::new(|| x.index_add(2, &idx, &src)),
        ),
        (
            "index_add(): self (f32) and source (f64) must have the same scalar type",
            Box::new(|| x.index_add(0, &idx, &src.to(DType::F64))),
        ),
        (
            "index_add(): Number of indices (3) should be equal to source.size(dim): (2), for \
             dim: 0",
            Box::new(|| x.index_add(0, &ints(&[0, 1, 2]), &src)),
        ),
        (
            "index_add(): Indexing dim 1 is out of bounds of the source tensor with dim 1",
            Box::new(|| x.index_add(1, &idx, &homura::ones([2]))),
        ),
        (
            "source tensor shape must match self tensor shape, excluding the specified \
             dimension. Got self.shape = [3, 4] source.shape = [2, 3]",
            Box::new(|| x.index_add(0, &idx, &homura::ones([2, 3]))),
        ),
        (
            "source tensor shape must match self tensor shape, excluding the specified \
             dimension. Got self.shape = [3, 4] source.shape = [2]",
            Box::new(|| x.index_add(0, &idx, &homura::ones([2]))),
        ),
        (
            "source tensor shape must match self tensor shape, excluding the specified \
             dimension. Got self.shape = [3, 4] source.shape = [2, 1]",
            Box::new(|| x.index_add(0, &idx, &homura::ones([2, 1]))),
        ),
        (
            "index 3 is out of bounds for dimension 0 with size 3",
            Box::new(|| x.index_add(0, &ints(&[3, 0]), &src)),
        ),
        (
            "index -1 is out of bounds for dimension 0 with size 3",
            Box::new(|| x.index_add(0, &ints(&[-1, 0]), &src)),
        ),
        // PyTorch skips the check when `self` has no elements and more than one dimension.
        (
            "index 0 is out of bounds for dimension 0 with size 0",
            Box::new(|| homura::zeros([0, 4]).index_add(0, &ints(&[0]), &homura::ones([1, 4]))),
        ),
        (
            "Expected all tensors to be on the same device, but found at least two devices, meta \
             and cpu!",
            Box::new(|| x.to(Device::Meta).index_add(0, &idx, &src.to(Device::Meta))),
        ),
    ];
    for (expected, op) in cases {
        assert_eq!(panic_message(op), expected);
    }
}

#[test]
fn index_add_sums_repeated_positions_and_rounds_once() {
    let x = homura::zeros([3, 2]);
    let src = homura::tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let y = x.index_add(0, &ints(&[2, 0, 2]), &src);
    assert_eq!(vals(&y), [3., 4., 0., 0., 6., 8.]);
    assert_eq!(vals(&x), [0.; 6], "self is not modified");
    let y = homura::zeros([3]).index_add(0, &homura::tensor(1i32), &homura::ones([1]));
    assert_eq!(vals(&y), [0., 1., 0.]);
    // Half precision sums are rounded once, in one or more dimensions alike. PyTorch rounds after
    // every addition for one-dimensional tensors, giving 2048 there.
    for shape in [vec![5], vec![5, 1]] {
        let src = Tensor::from_slice(&[2048f32, 1., 1., 1., 1.], shape.as_slice()).to(DType::F16);
        let mut out_shape = shape.clone();
        out_shape[0] = 1;
        let x = homura::zeros_with(out_shape.as_slice(), DType::F16);
        let y = x.index_add(0, &ints(&[0; 5]), &src);
        assert_eq!(vals(&y), [2052.], "{shape:?}");
    }
    let src = Tensor::from_slice(&[256f32, 1., 1.], [3, 1]).to(DType::BF16);
    let y = homura::zeros_with([1, 1], DType::BF16).index_add(0, &ints(&[0; 3]), &src);
    assert_eq!(vals(&y), [258.]);
    // Single precision sums are accumulated in f64, like the reductions (PyTorch gives 0 here).
    let src = homura::tensor([[1e8f32], [1.], [-1e8]]);
    let y = homura::zeros([1, 1]).index_add(0, &ints(&[0; 3]), &src);
    assert_eq!(vals(&y), [1.]);
    // Integer sums wrap around, bool sums are a logical or.
    let x = Tensor::from_slice(&[250u8], [1]);
    let y = x.index_add(0, &ints(&[0, 0]), &Tensor::from_slice(&[3u8, 4], [2]));
    assert_eq!(y.to_vec::<u8>(), [1]);
    let x = homura::tensor([false, false, true]);
    let src = homura::tensor([true, false, false]);
    let y = x.index_add(0, &ints(&[0, 0, 2]), &src);
    assert_eq!(y.to_vec::<bool>(), [true, false, true]);
}

#[test]
fn index_add_of_views_empty_tensors_and_meta() {
    // Non-contiguous `self` and `source`.
    let x = homura::arange(6.0).view([2, 3]).transpose(0, 1);
    let src = homura::arange(12.0)
        .view([3, 4])
        .index((.., (0..).step_by(2)));
    let y = x.index_add(1, &ints(&[1, 1]), &src);
    assert_eq!(vals(&y), [0., 5., 1., 14., 2., 23.]);
    assert!(y.is_contiguous());
    // No positions: a copy.
    let x = homura::ones([2, 3]);
    let y = x.index_add(0, &ints(&[]), &homura::zeros([0, 3]));
    assert!(!y.is_same(&x));
    assert_eq!(vals(&y), [1.; 6]);
    // The meta device checks shapes and dtypes but has no positions to check.
    let m = homura::zeros_with([3, 4], Device::Meta);
    let y = m.index_add(
        0,
        &ints(&[7, 9]).to(Device::Meta),
        &homura::ones_with([2, 4], Device::Meta),
    );
    assert_eq!(
        (y.shape().to_vec(), y.dtype(), y.device()),
        (vec![3, 4], DType::F32, Device::Meta)
    );
}

// ---------------------------------------------------------------- gather and scatter_add

#[test]
fn gather_and_scatter_add_check_their_arguments_like_pytorch() {
    let x = homura::arange(12.0).view([3, 4]);
    let z = homura::zeros([3, 4]);
    let cases: Vec<Case> = vec![
        (
            "gather(): Expected dtype i32 or i64 for index, got f32",
            Box::new(|| x.gather(1, &homura::tensor([[0.0f32]]))),
        ),
        (
            "Index tensor must have the same number of dimensions as input tensor",
            Box::new(|| x.gather(1, &ints(&[0, 1]))),
        ),
        (
            "Size does not match at dimension 1 expected index [1, 5] to be no larger than self \
             [3, 4] apart from dimension 0",
            Box::new(|| x.gather(0, &homura::tensor([[0i64; 5]]))),
        ),
        (
            "index 4 is out of bounds for dimension 1 with size 4",
            Box::new(|| x.gather(1, &homura::tensor([[4i64]]))),
        ),
        (
            "index -1 is out of bounds for dimension 1 with size 4",
            Box::new(|| x.gather(1, &homura::tensor([[-1i64]]))),
        ),
        (
            "index 0 is out of bounds for dimension 1 with size 0",
            Box::new(|| homura::zeros([3, 0]).gather(1, &homura::zeros_with([3, 2], DType::I64))),
        ),
        (
            "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
            Box::new(|| homura::tensor(1.0).gather(1, &homura::tensor(0i64))),
        ),
        (
            "scatter_add(): Expected dtype i32 or i64 for index, got bool",
            Box::new(|| z.scatter_add(1, &homura::tensor([[true]]), &homura::ones([1, 1]))),
        ),
        (
            "scatter_add(): Expected self.dtype to be equal to src.dtype",
            Box::new(|| {
                z.scatter_add(
                    1,
                    &homura::tensor([[0i64]]),
                    &homura::ones_with([1, 1], DType::F64),
                )
            }),
        ),
        (
            "Index tensor must have the same number of dimensions as self tensor",
            Box::new(|| z.scatter_add(1, &ints(&[0]), &homura::ones([1, 1]))),
        ),
        (
            "Index tensor must have the same number of dimensions as src tensor",
            Box::new(|| z.scatter_add(1, &homura::tensor([[0i64]]), &homura::ones([1]))),
        ),
        (
            "Expected index [1, 2] to be no larger than self [3, 4] apart from dimension 1 and to \
             be no larger size than src [1, 1]",
            Box::new(|| z.scatter_add(1, &homura::tensor([[0i64, 1]]), &homura::ones([1, 1]))),
        ),
        (
            "Expected index [4, 1] to be no larger than self [3, 4] apart from dimension 1 and to \
             be no larger size than src [4, 1]",
            Box::new(|| {
                z.scatter_add(
                    1,
                    &homura::zeros_with([4, 1], DType::I64),
                    &homura::ones([4, 1]),
                )
            }),
        ),
        (
            "index 4 is out of bounds for dimension 1 with size 4",
            Box::new(|| z.scatter_add(1, &homura::tensor([[4i64]]), &homura::ones([1, 1]))),
        ),
        (
            "index -1 is out of bounds for dimension 1 with size 4",
            Box::new(|| z.scatter_add(1, &homura::tensor([[-1i64]]), &homura::ones([1, 1]))),
        ),
    ];
    for (expected, op) in cases {
        assert_eq!(panic_message(op), expected);
    }
}

#[test]
fn gather_and_scatter_add_values() {
    let x = homura::arange(12.0).view([3, 4]);
    // An index smaller than `self`, larger along `dim`, and with repeated positions.
    let y = x.gather(1, &homura::tensor([[0i64, 1, 2, 3, 0, 1]]));
    assert_eq!(vals(&y), [0., 1., 2., 3., 0., 1.]);
    let y = x.gather(0, &homura::tensor([[2i32], [0]]));
    assert_eq!(vals(&y), [8., 0.]);
    // Only the part of `src` that the index covers is added.
    let y = homura::zeros([3, 4]).scatter_add(1, &homura::tensor([[0i64], [1], [2]]), &x);
    assert_eq!(vals(&y), [0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 8., 0.]);
    // A position may receive several elements, and larger along `dim` than `self` is fine.
    let y = homura::zeros([3, 4]).scatter_add(
        1,
        &homura::zeros_with([1, 6], DType::I64),
        &homura::ones([1, 6]),
    );
    assert_eq!(vals(&y)[..4], [6., 0., 0., 0.]);
    // Half precision sums are rounded once, as PyTorch does here.
    let src = Tensor::from_slice(&[2048f32, 1., 1., 1., 1.], [5]).to(DType::F16);
    let y = homura::zeros_with([1], DType::F16).scatter_add(0, &ints(&[0; 5]), &src);
    assert_eq!(vals(&y), [2052.]);
    // Integers and bools.
    let y = homura::tensor([1i64, 2]).scatter_add(0, &ints(&[1, 1]), &homura::tensor([10i64, 20]));
    assert_eq!(y.to_vec::<i64>(), [1, 32]);
    let y = homura::tensor([false, false]).scatter_add(
        0,
        &ints(&[1, 1]),
        &homura::tensor([true, true]),
    );
    assert_eq!(y.to_vec::<bool>(), [false, true]);
    // Non-contiguous index, `self` and `src`.
    let index = homura::tensor([[0i64, 2], [1, 3], [1, 0]]).transpose(0, 1);
    let y = x.transpose(0, 1).gather(0, &index);
    assert_eq!(vals(&y), [0., 5., 9., 2., 7., 8.]);
}

#[test]
fn gather_and_scatter_add_treat_scalars_as_vectors_of_one_element() {
    let y = homura::tensor(5.0).gather(0, &ints(&[0, 0]));
    assert_eq!((y.shape().to_vec(), vals(&y)), (vec![2], vec![5., 5.]));
    let y = homura::tensor([5.0, 6.0]).gather(0, &homura::tensor(1i64));
    assert_eq!((y.num_dims(), vals(&y)), (0, vec![6.]));
    let y = homura::tensor(1.0).gather(-1, &homura::tensor(0i64));
    assert_eq!(vals(&y), [1.]);
    let y = homura::tensor(1.0).scatter_add(0, &ints(&[0, 0]), &homura::tensor([5.0, 6.0]));
    assert_eq!((y.num_dims(), vals(&y)), (0, vec![12.]));
    let y = homura::zeros([3]).scatter_add(0, &homura::tensor(1i64), &homura::tensor(5.0));
    assert_eq!(vals(&y), [0., 5., 0.]);
    // Empty indices.
    let y = homura::ones([3, 4]).gather(1, &homura::zeros_with([3, 0], DType::I64));
    assert_eq!(y.shape(), [3, 0]);
    let z = homura::ones([3, 4]);
    let y = z.scatter_add(
        1,
        &homura::zeros_with([3, 0], DType::I64),
        &homura::ones([3, 2]),
    );
    assert!(!y.is_same(&z));
    assert_eq!(vals(&y), [1.; 12]);
    // Meta.
    let m = homura::zeros_with([3, 4], Device::Meta);
    let index = homura::zeros_with([2, 2], (DType::I64, Device::Meta));
    assert_eq!(m.gather(1, &index).shape(), [2, 2]);
    let src = homura::zeros_with([2, 3], Device::Meta);
    let y = m.scatter_add(0, &index, &src);
    assert_eq!((y.shape().to_vec(), y.device()), (vec![3, 4], Device::Meta));
}

// ---------------------------------------------------------------- select_scatter and slice_scatter

#[test]
fn select_scatter_and_slice_scatter_check_their_arguments_like_pytorch() {
    let x = homura::zeros([3, 4]);
    let cases: Vec<Case> = vec![
        (
            "select() cannot be applied to a 0-dim tensor.",
            Box::new(|| homura::tensor(1.0).select_scatter(&homura::tensor(5.0), 0, 0)),
        ),
        (
            "select(): index 3 out of range for tensor of size [3, 4] at dimension 0",
            Box::new(|| x.select_scatter(&homura::ones([4]), 0, 3)),
        ),
        (
            "select(): index -4 out of range for tensor of size [3, 4] at dimension 0",
            Box::new(|| x.select_scatter(&homura::ones([4]), 0, -4)),
        ),
        (
            "Dimension out of range (expected to be in range of [-2, 1], but got 2)",
            Box::new(|| x.select_scatter(&homura::ones([4]), 2, 1)),
        ),
        (
            "expected src to have a size equal to the slice of self. src size = [3], slice size \
             = [4]",
            Box::new(|| x.select_scatter(&homura::ones([3]), 0, 1)),
        ),
        (
            "expected src to have a size equal to the slice of self. src size = [1], slice size \
             = [4]",
            Box::new(|| x.select_scatter(&homura::ones([1]), 0, 1)),
        ),
        (
            "slice() cannot be applied to a 0-dim tensor.",
            Box::new(|| homura::tensor(1.0).slice_scatter(&homura::tensor(5.0), 0, None, None, 1)),
        ),
        (
            "slice step must be positive",
            Box::new(|| x.slice_scatter(&homura::ones([0, 4]), 0, Some(0), Some(0), 0)),
        ),
        (
            "slice step must be positive",
            Box::new(|| x.slice_scatter(&homura::ones([1, 4]), 0, Some(0), Some(1), -1)),
        ),
        (
            "expected src to have a size equal to the slice of self. src size = [3, 4], slice \
             size = [2, 4]",
            Box::new(|| x.slice_scatter(&homura::ones([3, 4]), 0, Some(1), Some(3), 1)),
        ),
        (
            "Expected all tensors to be on the same device, but found at least two devices, meta \
             and cpu!",
            Box::new(|| x.to(Device::Meta).select_scatter(&homura::ones([4]), 0, 1)),
        ),
    ];
    for (expected, op) in cases {
        assert_eq!(panic_message(op), expected);
    }
}

#[test]
fn select_scatter_and_slice_scatter_values() {
    let x = homura::arange(12.0).view([3, 4]);
    let y = x.select_scatter(&homura::tensor([-1.0, -2.0, -3.0]), -1, -1);
    assert_eq!(
        vals(&y),
        [0., 1., 2., -1., 4., 5., 6., -2., 8., 9., 10., -3.]
    );
    // The result is a new contiguous tensor, also for a non-contiguous `self`.
    let t = x.transpose(0, 1);
    let y = t.select_scatter(&homura::zeros([3]), 0, 0);
    assert!(y.is_contiguous() && !y.is_same(&t));
    assert_eq!(vals(&y), [0., 0., 0., 1., 5., 9., 2., 6., 10., 3., 7., 11.]);
    y.fill_(7.0);
    assert_eq!(vals(&x)[..2], [0., 1.], "self is not modified");
    // `src` is converted to the dtype of `self`, truncating toward zero for integers.
    let y = homura::zeros_with([3, 4], DType::I64).select_scatter(&homura::full([4], 2.7), 0, 1);
    assert_eq!(y.to_vec::<i64>()[4..8], [2, 2, 2, 2]);
    let y = homura::zeros_with([2], DType::Bool).select_scatter(&homura::tensor(3.0), 0, 1);
    assert_eq!(y.to_vec::<bool>(), [false, true]);
    // Python-style bounds: negative, clamped, empty and with a huge step.
    let row = homura::zeros([5]);
    let y = row.slice_scatter(&homura::ones([2]), 0, Some(-2), None, 1);
    assert_eq!(vals(&y), [0., 0., 0., 1., 1.]);
    let y = row.slice_scatter(&homura::ones([2]), 0, Some(3), Some(100), 1);
    assert_eq!(vals(&y), [0., 0., 0., 1., 1.]);
    let y = row.slice_scatter(&homura::ones([5]), 0, Some(-100), Some(100), 1);
    assert_eq!(vals(&y), [1.; 5]);
    let y = row.slice_scatter(&homura::ones([0]), 0, Some(4), Some(2), 1);
    assert_eq!(vals(&y), [0.; 5]);
    let y = row.slice_scatter(&homura::ones([1]), 0, Some(1), None, 1 << 62);
    assert_eq!(vals(&y), [0., 1., 0., 0., 0.]);
    let y = x.slice_scatter(&homura::zeros([3, 2]), -1, None, None, 2);
    assert_eq!(vals(&y), [0., 1., 0., 3., 0., 5., 0., 7., 0., 9., 0., 11.]);
    // Empty `self`.
    let e = homura::zeros([3, 0]);
    assert_eq!(e.select_scatter(&homura::ones([0]), 0, 1).shape(), [3, 0]);
    // Meta.
    let m = homura::zeros_with([3, 4], Device::Meta);
    let y = m.slice_scatter(
        &homura::ones_with([2, 4], Device::Meta),
        0,
        Some(1),
        Some(3),
        1,
    );
    assert_eq!((y.shape().to_vec(), y.device()), (vec![3, 4], Device::Meta));
}

// ---------------------------------------------------------------- in-place variants

#[test]
fn in_place_index_add_and_scatter_add() {
    let x = homura::zeros([3, 2]);
    let alias = x.clone();
    x.index_add_(0, &ints(&[2, 2]), &homura::ones([2, 2]));
    assert_eq!(vals(&alias), [0., 0., 0., 0., 2., 2.]);
    x.scatter_add_(
        1,
        &homura::tensor([[1i64], [1], [0]]),
        &homura::ones([3, 1]),
    );
    assert_eq!(vals(&alias), [0., 1., 0., 1., 3., 2.]);
    // Through a view, into the base.
    let base = homura::zeros([2, 3]);
    base.narrow(1, 1, 2)
        .index_add_(1, &ints(&[0]), &homura::ones([2, 1]));
    assert_eq!(vals(&base), [0., 1., 0., 0., 1., 0.]);
    // The in-place rules of autograd apply: rejected before anything is written.
    let leaf = homura::zeros([3]);
    leaf.requires_grad_(true);
    let message = panic_message(|| {
        leaf.index_add_(0, &ints(&[0]), &homura::ones([1]));
        homura::zeros([1])
    });
    assert_eq!(
        message,
        "a leaf Variable that requires grad is being used in an in-place operation."
    );
    let source = homura::ones([1]);
    source.requires_grad_(true);
    let message = panic_message(|| {
        homura::zeros([3]).scatter_add_(0, &ints(&[0]), &source);
        homura::zeros([1])
    });
    assert!(message.starts_with("in-place operations that need to be recorded"));
    // In no-grad mode, as optimizers update parameters.
    {
        let _no_grad = homura::no_grad();
        leaf.index_add_(0, &ints(&[1]), &homura::ones([1]));
    }
    assert_eq!(vals(&leaf), [0., 1., 0.]);
}
