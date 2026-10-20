//! M2 step 8a: `nn::functional::pad` (constant mode) and `Tensor::unfold`. Outputs and gradients
//! are compared with `tests/golden/m2_pad_unfold.safetensors` from
//! `tools/golden/gen_m2_pad_unfold.py`; the reference behavior encoded in the other tests is
//! asserted by `tools/golden/check_m2_pad_unfold.py`, in the same order.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::nn::functional::pad;
use homura::{DType, Device, Tensor};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn assert_panics(f: impl FnOnce(), message: &str) {
    assert_eq!(panic_message(f), message);
}

/// The node names of `t`'s graph, depth first, as `(depth, name)`.
fn chain(t: &Tensor) -> Vec<(usize, &'static str)> {
    fn walk(f: &autograd::GradFn, depth: usize, out: &mut Vec<(usize, &'static str)>) {
        out.push((depth, f.name()));
        for (next, _) in f.next_functions().into_iter().flatten() {
            walk(&next, depth + 1, out);
        }
    }
    let mut out = vec![];
    if let Some(f) = t.grad_fn() {
        walk(&f, 0, &mut out);
    }
    out
}

const ACC: &str = "AccumulateGrad";
const PAD_NODE: [(usize, &str); 2] = [(0, "ConstantPadNdBackward0"), (1, ACC)];

/// `1, 2, 3, ...` in `shape`, as an `F64` leaf that requires grad.
fn leaf(shape: &[usize]) -> Tensor {
    let x = counting(shape);
    x.requires_grad_(true);
    x
}

/// `1, 2, 3, ...` in `shape`, `F64`.
fn counting(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec((1..=n).map(|v| v as f64).collect(), shape)
}

fn f64s(values: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape)
}

fn vals(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

/// The values of a floating point tensor as their `f64` bit patterns, so that `-0.0` differs
/// from `0.0`; every NaN is the same.
fn bits(t: &Tensor) -> Vec<u64> {
    vals(t)
        .into_iter()
        .map(|v| if v.is_nan() { u64::MAX } else { v.to_bits() })
        .collect()
}

fn same_bits(actual: &Tensor, expected: &[f64]) {
    let expected = f64s(expected, &[expected.len()]);
    assert_eq!(bits(actual), bits(&expected), "{actual:?}");
}

/// Asserts that `actual` has the shape, dtype and exact values (signs of zeros included) of
/// `expected`.
fn assert_identical(actual: &Tensor, expected: &Tensor, what: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{what}: dtype");
    assert_eq!(
        bits(actual),
        bits(expected),
        "{what}\n  actual: {actual:?}\n  expected: {expected:?}"
    );
}

fn grads(out: &Tensor, inputs: &[&Tensor], g: &Tensor, create_graph: bool) -> Vec<Option<Tensor>> {
    autograd::grad(
        &[out],
        inputs,
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            create_graph,
            allow_unused: true,
            ..Default::default()
        },
    )
}

fn grad(out: &Tensor, x: &Tensor, g: &Tensor) -> Tensor {
    grads(out, &[x], g, false).remove(0).expect("a gradient")
}

/// Whether `a` and `b` share memory: writing through `a` changes `b`.
fn shares_memory(a: &Tensor, b: &Tensor) -> bool {
    let _no_grad = homura::no_grad();
    let before = vals(b);
    a.add_(1.0);
    let changed = vals(b) != before;
    a.sub_(1.0);
    changed
}

/// The output of the golden case `c` of `function` for the input `x`.
fn evaluate(function: &str, c: &Case, x: &Tensor) -> Tensor {
    let transposed = c.golden.names(c.name).iter().any(|n| n == "opt.transpose");
    let input = if transposed {
        c.input("opt.transpose");
        x.transpose(-2, -1)
    } else {
        x.clone()
    };
    match function {
        "pad" => pad(
            &input,
            &c.input("opt.pad").to_vec::<i64>(),
            c.input("opt.value").item::<f64>(),
        ),
        "unfold" => c
            .input("opt.unfold")
            .to_vec::<i64>()
            .chunks(3)
            .fold(input, |t, o| t.unfold(o[0], o[1] as usize, o[2] as usize)),
        _ => panic!("unknown function {function}"),
    }
}

#[test]
fn pad_and_unfold_match_pytorch() {
    let golden = Golden::load("m2_pad_unfold.safetensors");
    let cases = golden.cases();
    // 11 pad and 15 unfold cases in f64 and f32, 9 of them in f16 and bf16.
    assert_eq!(cases.len(), 2 * 26 + 2 * 9);
    let mut second_order = 0;
    for name in &cases {
        let c = Case::new(&golden, name);
        let function = name.split('_').next().unwrap();
        let x = c.input("x");
        x.requires_grad_(true);
        let out = evaluate(function, &c, &x);
        // Copies and window sums in the reference's order: exact in every dtype.
        c.check_with("out", &out, |a, e| {
            assert_identical(a, e, &format!("{name}/out"))
        });
        let second = golden.names(name).iter().any(|n| n == "grad2_g");
        let g = c.input("g");
        g.requires_grad_(second);
        let grad_x = grads(&out, &[&x], &g, second).remove(0).unwrap();
        c.check_with("grad_x", &grad_x, |a, e| {
            assert_identical(a, e, &format!("{name}/grad_x"))
        });
        if second {
            second_order += 1;
            let gg = c.input("gg_x");
            let total = grad_x.mul(&gg).sum();
            let second_grads = autograd::grad(
                &[&total],
                &[&x, &g],
                GradOptions {
                    allow_unused: true,
                    ..Default::default()
                },
            );
            assert!(
                second_grads[0].is_none(),
                "{name}: x has no second derivative"
            );
            c.check_with("grad2_g", second_grads[1].as_ref().unwrap(), |a, e| {
                assert_identical(a, e, &format!("{name}/grad2_g"))
            });
        }
        c.assert_all_used();
    }
    assert_eq!(second_order, 9);
}

#[test]
fn pad_records_constant_pad_nd() {
    let x = leaf(&[2, 3]);
    let pads: [&[i64]; 6] = [
        &[1, 2],
        &[-1, 0],
        &[0, 0],
        &[-1, 2, 1, -1],
        &[],
        &[0, 0, 0, 0],
    ];
    for p in pads {
        assert_eq!(chain(&pad(&x, p, 0)), PAD_NODE, "{p:?}");
    }
    assert_eq!(chain(&pad(&x, &[1, 1], 2.5)), PAD_NODE);
    let g = homura::ones_with([2, 6], DType::F64);
    g.requires_grad_(true);
    let gx = grads(&pad(&x, &[1, 2], 0), &[&x], &g, true)
        .remove(0)
        .unwrap();
    assert_eq!(chain(&gx), PAD_NODE);
    assert_eq!(vals(&gx), [1.0; 6]);
    {
        let _no_grad = homura::no_grad();
        assert!(pad(&x, &[1, 1], 0).grad_fn().is_none());
    }
    let ints = homura::ones_with([2], DType::I64);
    assert!(pad(&ints, &[1, 1], 0).grad_fn().is_none());
}

#[test]
fn pad_values_layouts_and_storage() {
    let x = homura::arange_with(6.0, DType::F64).reshape([2, 3]);
    let padded = pad(&x, &[1, 2], 0);
    assert_eq!(padded.shape(), [2, 6]);
    assert_eq!(
        vals(&padded),
        [0., 0., 1., 2., 0., 0., 0., 3., 4., 5., 0., 0.]
    );
    let rows = pad(&x, &[0, 0, 1, 0], -1.0);
    assert_eq!(vals(&rows), [-1., -1., -1., 0., 1., 2., 3., 4., 5.]);
    assert_eq!(vals(&pad(&x, &[-1, 0], 0)), [1., 2., 4., 5.]);
    let mixed = pad(&x, &[-1, 2, 1, -1], 7.0);
    assert_eq!(mixed.shape(), [2, 4]);
    assert_eq!(vals(&mixed), [7., 7., 7., 7., 1., 2., 7., 7.]);
    assert_eq!(pad(&x, &[-2, -1], 0).shape(), [2, 0]);
    assert_eq!(pad(&x, &[0, 0, -2, 0], 0).shape(), [0, 3]);
    assert_eq!(pad(&x, &[-3, 0, 1, 1], 0).shape(), [4, 0]);
    let s = homura::tensor_with(1.5, DType::F64);
    assert_eq!(pad(&s, &[], 0).shape(), []);
    assert_eq!(pad(&s, &[], 0).item::<f64>(), 1.5);
    // Every result is new, contiguous storage; PyTorch keeps the dense strides of the
    // non-contiguous input when it doesn't fill (a documented difference).
    let t = homura::arange_with(6.0, DType::F64)
        .reshape([2, 3])
        .transpose(0, 1);
    assert_eq!(t.stride(), [1, 3]);
    let pads: [(&[i64], [usize; 2]); 5] = [
        (&[0, 0], [3, 2]),
        (&[-1, 0], [3, 1]),
        (&[0, 0, -1, 0], [2, 2]),
        (&[1, 0], [3, 3]),
        (&[0, 0, 1, 0], [4, 2]),
    ];
    for (p, shape) in pads {
        let y = pad(&t, p, 0);
        assert_eq!(y.shape(), shape, "{p:?}");
        assert!(y.is_contiguous(), "{p:?}");
        assert!(!shares_memory(&y, &t), "{p:?}");
    }
    assert_eq!(vals(&pad(&t, &[-1, 0], 0)), [3., 4., 5.]);
    assert_eq!(vals(&pad(&t, &[0, 0, -1, 0], 0)), [1., 4., 2., 5.]);
    let y = pad(&x, &[0, 0], 0);
    assert!(!shares_memory(&y, &x));
}

#[test]
fn pad_checks_its_arguments() {
    let x = homura::ones([2, 3]);
    let odd = "Padding length must be divisible by 2";
    assert_panics(|| drop(pad(&x, &[1], 0)), odd);
    assert_panics(|| drop(pad(&x, &[1, 1, 1], 0)), odd);
    assert_panics(
        || drop(pad(&x, &[1, 1, 1, 1, 1, 1], 0)),
        "Padding length should be less than or equal to two times the input dimension but got \
         padding length 6 and input of dimension 2",
    );
    assert_panics(
        || drop(pad(&homura::tensor(1.0), &[1, 1], 0)),
        "Padding length should be less than or equal to two times the input dimension but got \
         padding length 2 and input of dimension 0",
    );
    let negative = "narrow(): length must be non-negative.";
    let pads: [&[i64]; 6] = [
        &[-2, -2],
        &[-4, 0],
        &[-4, 2],
        &[0, -4],
        &[0, 0, -1, -2],
        &[5, -9],
    ];
    for p in pads {
        assert_panics(|| drop(pad(&x, p, 0)), negative);
    }
    // The length is checked first, then the crops; extreme values don't overflow.
    assert_panics(|| drop(pad(&x, &[-9, 0, 1], 0)), odd);
    let bytes = homura::ones_with([2, 3], DType::U8);
    assert_panics(|| drop(pad(&bytes, &[-4, 0], 1e10)), negative);
    assert_panics(|| drop(pad(&x, &[i64::MIN, 0], 0)), negative);
    assert_panics(|| drop(pad(&x, &[0, i64::MIN], 0)), negative);
    let message = panic_message(|| drop(pad(&x, &[i64::MAX, i64::MAX], 0)));
    assert!(message.contains("too large"), "{message}");
}

#[test]
fn pad_converts_the_value_only_when_filling() {
    let ints = homura::ones_with([2], DType::I64);
    assert_eq!(pad(&ints, &[1, 1], 1.5).to_vec::<i64>(), [1, 1, 1, 1]);
    assert_eq!(pad(&ints, &[1, 0], -2.7).to_vec::<i64>(), [-2, 1, 1]);
    let bytes = homura::ones_with([2], DType::U8);
    let overflow = "value cannot be converted to type u8 without overflow";
    assert_panics(|| drop(pad(&bytes, &[1, 1], 300)), overflow);
    assert_panics(
        || drop(pad(&homura::ones_with([2], DType::F16), &[1, 0], 1e5)),
        "value cannot be converted to type f16 without overflow",
    );
    assert_panics(
        || drop(pad(&homura::ones_with([0, 2], DType::U8), &[1, 0], 1e10)),
        overflow,
    );
    let pads: [&[i64]; 3] = [&[0, 0], &[-1, 0], &[]];
    for p in pads {
        assert_eq!(pad(&bytes, p, 1e10).dtype(), DType::U8, "{p:?}");
    }
    let bools = homura::zeros_with([2], DType::Bool);
    assert_eq!(
        pad(&bools, &[1, 0], 2).to_vec::<bool>(),
        [true, false, false]
    );
    let bools = homura::ones_with([2], DType::Bool);
    assert_eq!(
        pad(&bools, &[1, 0], 0).to_vec::<bool>(),
        [false, true, true]
    );
    let y = pad(&homura::ones([2]), &[1, 0], f64::NAN);
    let nan: Vec<bool> = vals(&y).iter().map(|v| v.is_nan()).collect();
    assert_eq!(nan, [true, false, false]);
    same_bits(&pad(&homura::ones([1]), &[1, 0], -0.0), &[-0.0, 1.0]);
    let half = pad(&homura::ones_with([1], DType::F16), &[1, 0], f64::INFINITY);
    assert_eq!(vals(&half), [f64::INFINITY, 1.0]);
}

#[test]
fn pad_on_meta_and_empty_inputs() {
    let m = homura::ones_with([2, 3], Device::Meta);
    let y = pad(&m, &[1, 1], 0);
    assert_eq!((y.shape().to_vec(), y.device()), (vec![2, 5], Device::Meta));
    assert_eq!(pad(&m, &[-1, 0, 2, 2], 0).shape(), [6, 2]);
    // Meta tensors take any value, like PyTorch's meta kernels.
    let bytes = homura::ones_with([2], (DType::U8, Device::Meta));
    assert_eq!(pad(&bytes, &[1, 1], 1e10).shape(), [4]);
    assert_panics(
        || drop(pad(&m, &[-4, 0], 0)),
        "narrow(): length must be non-negative.",
    );
    let ml = homura::ones_with([2, 3], (DType::F32, Device::Meta));
    ml.requires_grad_(true);
    let gm = grad(
        &pad(&ml, &[1, -1], 0).sum(),
        &ml,
        &homura::ones_with([], Device::Meta),
    );
    assert_eq!(
        (gm.shape().to_vec(), gm.device()),
        (vec![2, 3], Device::Meta)
    );
    let y = pad(&homura::ones([0, 3]), &[1, 1, 1, 1], 0);
    assert_eq!(y.shape(), [2, 5]);
    assert_eq!(vals(&y), [0.0; 10]);
    let e = homura::ones_with([2, 0], DType::F64);
    e.requires_grad_(true);
    let y = pad(&e, &[1, 2], 0);
    assert_eq!(vals(&y), [0.0; 6]);
    let ge = grad(&y.sum(), &e, &homura::tensor_with(1.0, DType::F64));
    assert_eq!(ge.shape(), [2, 0]);
    // Empty inputs with any strides.
    let odd = homura::zeros([0]).as_strided([3, 0], [usize::MAX / 4, 1], 0);
    assert_eq!(pad(&odd, &[0, 0, -1, 1], 0).shape(), [3, 0]);
    assert_eq!(pad(&odd, &[1, 0, -1, 1], 0).shape(), [3, 1]);
}

#[test]
fn pad_gradients_crop_the_upstream_gradient() {
    let x = leaf(&[2, 3]);
    let inf = f64::INFINITY;
    let g = f64s(&[1.0, inf, 2.0, 3.0, f64::NAN, 4.0, 5.0, -6.0], &[2, 4]);
    same_bits(
        &grad(&pad(&x, &[-1, 2], 0), &x, &g),
        &[0.0, 1.0, inf, 0.0, f64::NAN, 4.0],
    );
    let gz = homura::full_with([2, 3], -0.0, DType::F64);
    same_bits(&grad(&pad(&x, &[0, 0], 0), &x, &gz), &[-0.0; 6]);
    let pads: [&[i64]; 5] = [&[1, 2], &[-1, 2, 1, -1], &[0, 0], &[-2, 0, 0, 3], &[]];
    for p in pads {
        let f = |xs: &[Tensor]| vec![pad(&xs[0], p, 0.5)];
        gradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
        gradgradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
    }
}

#[test]
fn unfold_is_a_view() {
    let x = leaf(&[2, 5]);
    let u = x.unfold(1, 3, 2);
    assert_eq!(u.shape(), [2, 2, 3]);
    assert_eq!(u.stride(), [5, 2, 1]);
    assert_eq!(chain(&u), [(0, "UnfoldBackward0"), (1, ACC)]);
    assert_eq!(vals(&u), [1., 2., 3., 3., 4., 5., 6., 7., 8., 8., 9., 10.]);
    assert!(shares_memory(&x, &u));
    let base = homura::arange(12.0).reshape([3, 4]);
    let t = base.transpose(0, 1).narrow(0, 1, 3);
    let v = t.unfold(1, 2, 1);
    assert_eq!(v.shape(), [3, 2, 2]);
    assert_eq!(v.stride(), [1, 4, 4]);
    assert_eq!(v.storage_offset(), 1);
    assert!(shares_memory(&base, &v));
    let u2 = leaf(&[4, 4]).unfold(0, 2, 1).unfold(1, 2, 1);
    assert_eq!(u2.shape(), [3, 3, 2, 2]);
    assert_eq!(u2.stride(), [4, 1, 4, 1]);
    assert_eq!(
        chain(&u2),
        [(0, "UnfoldBackward0"), (1, "UnfoldBackward0"), (2, ACC)]
    );
    let one = x.unfold(1, 5, 3);
    assert_eq!(
        (one.shape().to_vec(), one.stride().to_vec()),
        (vec![2, 1, 5], vec![5, 3, 1])
    );
    // A single window of a step too large for its stride keeps the stride.
    let huge = x.unfold(0, 2, usize::MAX);
    assert_eq!(
        (huge.shape().to_vec(), huge.stride().to_vec()),
        (vec![1, 5, 2], vec![5, 1, 5])
    );
    let _no_grad = homura::no_grad();
    assert!(x.unfold(1, 2, 1).grad_fn().is_none());
}

#[test]
fn unfold_checks_its_arguments() {
    let x = homura::ones([2, 5]);
    let too_big = "maximum size for tensor at dimension 1 is 5 but size is 6";
    assert_panics(|| drop(x.unfold(1, 6, 1)), too_big);
    assert_panics(|| drop(x.unfold(-1, 6, 0)), too_big);
    assert_panics(|| drop(x.unfold(1, 2, 0)), "step is 0 but must be > 0");
    assert_panics(
        || drop(x.unfold(2, 2, 1)),
        "Dimension out of range (expected to be in range of [-2, 1], but got 2)",
    );
    let s = homura::tensor(3.0);
    assert_panics(
        || drop(s.unfold(1, 1, 1)),
        "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
    );
    assert_panics(
        || drop(s.unfold(0, 2, 1)),
        "maximum size for tensor at dimension 0 is 1 but size is 2",
    );
}

#[test]
fn unfold_edge_cases() {
    let x = homura::ones([2, 3]);
    let layout = |t: &Tensor| (t.shape().to_vec(), t.stride().to_vec());
    assert_eq!(layout(&x.unfold(1, 0, 1)), (vec![2, 4, 0], vec![3, 1, 1]));
    assert_eq!(x.unfold(1, 0, 2).shape(), [2, 2, 0]);
    assert_eq!(layout(&x.unfold(-2, 1, 1)), (vec![2, 3, 1], vec![3, 1, 3]));
    let s = homura::tensor(3.0);
    for step in [1, 5] {
        assert_eq!(layout(&s.unfold(0, 1, step)), (vec![1], vec![1]));
        assert_eq!(layout(&s.unfold(-1, 1, step)), (vec![1], vec![1]));
        assert_eq!(s.unfold(0, 1, step).to_vec::<f32>(), [3.0]);
    }
    assert_eq!(s.unfold(0, 0, 1).shape(), [0]);
    let ints = homura::arange_with(5, DType::I64);
    assert_eq!(ints.unfold(0, 2, 2).to_vec::<i64>(), [0, 1, 2, 3]);
    let bools = homura::ones_with([2, 5], DType::Bool);
    assert_eq!(bools.unfold(1, 3, 1).shape(), [2, 3, 3]);
    let m = homura::ones_with([2, 5], Device::Meta).unfold(1, 3, 2);
    assert_eq!(layout(&m), (vec![2, 2, 3], vec![5, 2, 1]));
    assert_eq!(m.device(), Device::Meta);
    assert_eq!(homura::ones([2, 0]).unfold(1, 0, 1).shape(), [2, 1, 0]);
    assert_eq!(homura::ones([0, 5]).unfold(1, 3, 1).shape(), [0, 3, 3]);
    assert_eq!(homura::ones([0, 5]).unfold(0, 0, 3).shape(), [1, 5, 0]);
}

#[test]
fn unfold_gradients() {
    let x = leaf(&[2, 5]);
    let one = homura::tensor_with(1.0, DType::F64);
    assert_eq!(
        vals(&grad(&x.unfold(1, 3, 1).sum(), &x, &one)),
        [1., 2., 3., 2., 1., 1., 2., 3., 2., 1.]
    );
    assert_eq!(
        vals(&grad(&x.unfold(1, 2, 2).sum(), &x, &one)),
        [1., 1., 1., 1., 0., 1., 1., 1., 1., 0.]
    );
    assert_eq!(
        vals(&grad(&x.unfold(1, 1, 3).sum(), &x, &one)),
        [1., 0., 0., 1., 0., 1., 0., 0., 1., 0.]
    );
    let e = leaf(&[1]);
    assert_eq!(
        vals(&grad(&e.expand([4]).unfold(0, 2, 1).sum(), &e, &one)),
        [6.0]
    );
    let base = homura::arange_with(12.0, DType::F64).reshape([4, 3]);
    base.requires_grad_(true);
    let t = base.transpose(0, 1).narrow(0, 1, 2);
    assert_eq!((t.stride().to_vec(), t.storage_offset()), (vec![1, 3], 1));
    let w = counting(&[2, 3, 2]);
    assert_eq!(
        vals(&grad(&t.unfold(1, 2, 1).mul(&w).sum(), &base, &one)),
        [0., 1., 7., 0., 5., 17., 0., 9., 21., 0., 6., 12.]
    );
    // Upstream non-finite values reach only the covered positions.
    let x = leaf(&[1, 4]);
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let gin = f64s(&[inf, 1.0, nan, 2.0], &[1, 2, 2]);
    same_bits(&grad(&x.unfold(1, 2, 2), &x, &gin), &[inf, 1.0, nan, 2.0]);
    same_bits(
        &grad(&x.unfold(1, 2, 1).narrow(1, 0, 2), &x, &gin),
        &[inf, nan, 2.0, 0.0],
    );
    // Windows that don't overlap copy the gradient (keeping -0.0), overlapping ones add it to
    // zero.
    let z = homura::full_with([1, 2, 2], -0.0, DType::F64);
    same_bits(&grad(&x.unfold(1, 2, 2), &x, &z), &[-0.0; 4]);
    let z = homura::full_with([1, 3, 2], -0.0, DType::F64);
    same_bits(&grad(&x.unfold(1, 2, 1), &x, &z), &[0.0; 4]);
    // Half precision adds the windows in the dtype, one at a time in window order.
    let xh = homura::zeros_with([7], DType::F16);
    xh.requires_grad_(true);
    let mut gh = vec![0.0; 15];
    (gh[2], gh[4], gh[6]) = (2048.0, 1.0, 1.0);
    (gh[8], gh[10], gh[12]) = (1.0, 1.0, 2048.0);
    let gh = f64s(&gh, &[5, 3]);
    assert_eq!(
        vals(&grad(&xh.unfold(0, 3, 1), &xh, &gh.to(DType::F16))),
        [0., 0., 2048., 0., 2050., 0., 0.]
    );
    let in_f32 = grad(&xh.to(DType::F32).unfold(0, 3, 1), &xh, &gh.to(DType::F32));
    assert_eq!(vals(&in_f32)[2], 2050.0);
    // Zero-size windows and zero-dimensional inputs.
    let x = leaf(&[2, 3]);
    let empty = homura::ones_with([2, 4, 0], DType::F64);
    assert_eq!(vals(&grad(&x.unfold(-1, 0, 1), &x, &empty)), [0.0; 6]);
    empty.requires_grad_(true);
    let g = grads(&x.unfold(-1, 0, 1), &[&x], &empty, true)
        .remove(0)
        .unwrap();
    assert_eq!(chain(&g), [(0, "UnfoldBackwardBackward0"), (1, ACC)]);
    let m = homura::ones_with([2, 3], Device::Meta);
    m.requires_grad_(true);
    let um = m.unfold(-1, 0, 1);
    assert_eq!(
        (um.shape().to_vec(), um.stride().to_vec()),
        (vec![2, 4, 0], vec![3, 1, 1])
    );
    let gm = grad(&um.sum(), &m, &homura::ones_with([], Device::Meta));
    assert_eq!(
        (gm.shape().to_vec(), gm.device()),
        (vec![2, 3], Device::Meta)
    );
    for (size, step) in [(2, 1), (2, 2)] {
        let gm = grad(
            &m.unfold(1, size, step).sum(),
            &m,
            &homura::ones_with([], Device::Meta),
        );
        assert_eq!(
            (gm.shape().to_vec(), gm.device()),
            (vec![2, 3], Device::Meta)
        );
    }
    let s = homura::tensor_with(3.0, DType::F64);
    s.requires_grad_(true);
    let gs = grad(&s.unfold(0, 1, 4), &s, &f64s(&[5.0], &[1]));
    assert_eq!((gs.shape().to_vec(), gs.item::<f64>()), (vec![], 5.0));
    let gs = grad(&s.unfold(0, 0, 1), &s, &f64s(&[], &[0]));
    assert_eq!((gs.shape().to_vec(), gs.item::<f64>()), (vec![], 0.0));
    // Numerical checks, also of the gradient's own derivative, on non-contiguous inputs.
    let base = leaf(&[6, 4]);
    let options: [&[(i64, usize, usize)]; 6] = [
        &[(1, 3, 1)],
        &[(0, 2, 3)],
        &[(1, 2, 2)],
        &[(0, 3, 1), (1, 2, 1)],
        &[(0, 2, 2), (2, 2, 1)],
        &[(1, 0, 1)],
    ];
    for o in options {
        let f = |xs: &[Tensor]| {
            let t = xs[0].transpose(0, 1);
            vec![
                o.iter()
                    .fold(t, |t, &(d, size, step)| t.unfold(d, size, step)),
            ]
        };
        gradcheck(f, &[&base], GradcheckOptions::default()).unwrap();
        gradgradcheck(f, &[&base], GradcheckOptions::default()).unwrap();
    }
}

#[test]
fn unfold_backward_records_its_own_node() {
    let x = leaf(&[2, 5]);
    let gin = homura::ones_with([2, 3, 3], DType::F64);
    gin.requires_grad_(true);
    let g = grads(&x.unfold(1, 3, 1), &[&x], &gin, true)
        .remove(0)
        .unwrap();
    assert_eq!(chain(&g), [(0, "UnfoldBackwardBackward0"), (1, ACC)]);
    let r = homura::arange_with(10.0, DType::F64).reshape([2, 5]);
    r.requires_grad_(true);
    let gg = grads(&g, &[&gin], &r, true).remove(0).unwrap();
    assert_identical(&gg, &r.detach().unfold(1, 3, 1).contiguous(), "gg");
    assert_eq!(chain(&gg), [(0, "UnfoldBackward0"), (1, ACC)]);
    let gin = homura::ones_with([2, 2, 2], DType::F64);
    gin.requires_grad_(true);
    let g = grads(&x.unfold(1, 2, 2), &[&x], &gin, true)
        .remove(0)
        .unwrap();
    assert_eq!(chain(&g), [(0, "UnfoldBackwardBackward0"), (1, ACC)]);
}

#[test]
fn writes_into_unfold_results() {
    let overlap = "unsupported operation: more than one element of the written-to tensor may \
                   refer to a single memory location. Please deep_clone() the tensor before \
                   performing the operation.";
    {
        // PyTorch writes overlapping windows several times; homura rejects the write (a
        // documented difference) and leaves the data unchanged.
        let _no_grad = homura::no_grad();
        let v = homura::zeros([5]);
        assert_panics(
            || {
                v.unfold(0, 3, 1).add_(1.0);
            },
            overlap,
        );
        assert_panics(
            || {
                v.unfold(0, 3, 1).fill_(4.0);
            },
            overlap,
        );
        assert_eq!(v.to_vec::<f32>(), [0.0; 5]);
        v.unfold(0, 2, 2).fill_(4.0);
        assert_eq!(v.to_vec::<f32>(), [4., 4., 4., 4., 0.]);
    }
    // Outside no-grad mode too, for tensors that don't require grad.
    let v = homura::zeros([5]);
    assert_panics(
        || {
            v.unfold(0, 3, 1).add_(1.0);
        },
        overlap,
    );
    let w = homura::ones([5]);
    w.requires_grad_(true);
    assert_panics(
        || {
            w.unfold(0, 3, 1).add_(1.0);
        },
        "a view of a leaf Variable that requires grad is being used in an in-place operation.",
    );
    // PyTorch records views of non-leaves written in place (CopySlices); homura doesn't
    // support that yet (D7).
    let z = w.mul(1.0);
    let message = panic_message(|| {
        z.unfold(0, 2, 2).mul_(2.0);
    });
    assert!(
        message.starts_with("in-place operations on a view are not supported yet"),
        "{message}"
    );
}
