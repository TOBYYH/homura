//! M2 step 8b: `nn::functional::{max_pool2d, max_pool2d_with_indices, adaptive_avg_pool2d}`.
//! Outputs and gradients are compared with `tests/golden/m2_conv_pool.safetensors` from
//! `tools/golden/gen_m2_conv_pool.py`; the reference behavior encoded in the other tests is
//! asserted by `tools/golden/check_m2_conv_pool.py`, in the same order.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden, assert_close_tol};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::nn::functional::{
    MaxPool2dOptions, adaptive_avg_pool2d, max_pool2d, max_pool2d_with_indices,
};
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
const INF: f64 = f64::INFINITY;
const NAN: f64 = f64::NAN;

/// `1, 2, 3, ...` in `shape`, `F64`.
fn counting(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec((1..=n).map(|v| v as f64).collect(), shape)
}

fn leaf(shape: &[usize]) -> Tensor {
    let x = counting(shape);
    x.requires_grad_(true);
    x
}

fn f64s(values: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape)
}

fn vals(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

fn ints(t: &Tensor) -> Vec<i64> {
    t.to(DType::I64).to_vec::<i64>()
}

/// The values as `f64` bit patterns, so that `-0.0` differs from `0.0`; every NaN is the same.
fn bits(t: &Tensor) -> Vec<u64> {
    vals(t)
        .into_iter()
        .map(|v| if v.is_nan() { u64::MAX } else { v.to_bits() })
        .collect()
}

/// Asserts that `actual` has the shape, dtype and exact values of `expected`.
fn assert_identical(actual: &Tensor, expected: &Tensor, what: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{what}: dtype");
    if expected.is_floating_point() {
        assert_eq!(
            bits(actual),
            bits(expected),
            "{what}\n  actual: {actual:?}\n  expected: {expected:?}"
        );
    } else {
        assert_eq!(ints(actual), ints(expected), "{what}");
    }
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

fn sum_grad(out: &Tensor, x: &Tensor) -> Tensor {
    grad(
        &out.sum(),
        x,
        &homura::ones_with([], (out.dtype(), out.device())),
    )
}

fn pair(c: &Case, name: &str) -> [usize; 2] {
    let v = c.input(name).to_vec::<i64>();
    [v[0] as usize, v[1] as usize]
}

fn has(c: &Case, name: &str) -> bool {
    c.golden.names(c.name).iter().any(|n| n == name)
}

/// The golden case's input, transposed with `opt.transpose`.
fn pooled_input(c: &Case, x: &Tensor) -> Tensor {
    if has(c, "opt.transpose") {
        c.input("opt.transpose");
        x.transpose(-2, -1)
    } else {
        x.clone()
    }
}

fn pool_options(c: &Case) -> MaxPool2dOptions {
    MaxPool2dOptions {
        kernel_size: pair(c, "opt.kernel"),
        stride: has(c, "opt.stride").then(|| pair(c, "opt.stride")),
        padding: pair(c, "opt.padding"),
        dilation: pair(c, "opt.dilation"),
        ceil_mode: c.input("opt.ceil").item::<i64>() != 0,
    }
}

/// Checks the first and, where stored, second derivatives of the golden case `c` for the output
/// `out` of the input `x`, with `compare` for each pair of tensors.
fn check_gradients(c: &Case, x: &Tensor, out: &Tensor, compare: &dyn Fn(&Tensor, &Tensor, &str)) {
    let second = has(c, "grad2_g");
    let g = c.input("g");
    g.requires_grad_(second);
    let grad_x = grads(out, &[x], &g, second).remove(0).unwrap();
    c.check_with("grad_x", &grad_x, |a, e| {
        compare(a, e, &format!("{}/grad_x", c.name))
    });
    if second {
        let total = grad_x.mul(c.input("gg_x")).sum();
        let second_grads = autograd::grad(
            &[&total],
            &[x, &g],
            GradOptions {
                allow_unused: true,
                ..Default::default()
            },
        );
        // A missing entry is a None second derivative (the mean path's for x).
        for (k, t) in ["grad2_x", "grad2_g"].into_iter().zip(&second_grads) {
            if has(c, k) {
                c.check_with(k, t.as_ref().unwrap(), |a, e| {
                    compare(a, e, &format!("{}/{k}", c.name))
                });
            } else {
                assert!(t.is_none(), "{}/{k}", c.name);
            }
        }
    }
}

#[test]
fn max_pool2d_matches_pytorch() {
    let golden = Golden::load("m2_conv_pool.safetensors");
    let cases: Vec<String> = golden
        .cases()
        .into_iter()
        .filter(|c| c.starts_with("maxpool_"))
        .collect();
    // 12 cases in f64 and f32, 7 in f16 and bf16, 1 in i64 and u8.
    assert_eq!(cases.len(), 2 * 12 + 2 * 7 + 2);
    for name in &cases {
        let c = Case::new(&golden, name);
        let x = c.input("x");
        let floating = x.is_floating_point();
        x.requires_grad_(floating);
        let (out, indices) = max_pool2d_with_indices(&pooled_input(&c, &x), pool_options(&c));
        // The same selections and sums in the same order: exact in every dtype.
        c.check_with("out", &out, |a, e| {
            assert_identical(a, e, &format!("{name}/out"))
        });
        c.check_with("indices", &indices, |a, e| {
            assert_identical(a, e, &format!("{name}/indices"))
        });
        if floating {
            check_gradients(&c, &x, &out, &assert_identical);
        }
        c.assert_all_used();
    }
}

#[test]
fn adaptive_avg_pool2d_matches_pytorch() {
    let golden = Golden::load("m2_conv_pool.safetensors");
    let cases: Vec<String> = golden
        .cases()
        .into_iter()
        .filter(|c| c.starts_with("adaptive_"))
        .collect();
    // 11 cases in f64 and f32, 5 in f16 and bf16.
    assert_eq!(cases.len(), 2 * 11 + 2 * 5);
    for name in &cases {
        let c = Case::new(&golden, name);
        let x = c.input("x");
        x.requires_grad_(true);
        let output = pair(&c, "opt.output");
        let out = adaptive_avg_pool2d(&pooled_input(&c, &x), output);
        let mean = output == [1, 1];
        // The kernel sums and divides like PyTorch's: exact. The [1, 1] mean, and the mean of
        // its second derivative, sum in f64 and round once; PyTorch's sum in the dtype.
        let compare = |a: &Tensor, e: &Tensor, what: &str| {
            if mean {
                let (rtol, atol) = common::tolerance(e.dtype());
                assert_close_tol(a, e, 2.0 * rtol, atol, what);
            } else {
                assert_identical(a, e, what);
            }
        };
        c.check_with("out", &out, |a, e| compare(a, e, &format!("{name}/out")));
        check_gradients(&c, &x, &out, &compare);
        c.assert_all_used();
    }
}

#[test]
fn max_pool2d_selection_rules() {
    // Several NaN: the last; ordinary ties: the first; padding never.
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let x = f64s(&[NAN, 2.0, NAN, 1.0], &[1, 1, 2, 2]).to(dtype);
        x.requires_grad_(true);
        let (y, i) = max_pool2d_with_indices(&x, MaxPool2dOptions::new([2, 2]));
        assert_eq!(ints(&i), [2], "{dtype}");
        assert!(vals(&y)[0].is_nan());
        assert_eq!(vals(&sum_grad(&y, &x)), [0.0, 0.0, 1.0, 0.0]);
    }
    let x = f64s(&[3.0, 3.0, 1.0, 3.0], &[1, 1, 2, 2]);
    assert_eq!(
        ints(&max_pool2d_with_indices(&x, MaxPool2dOptions::new([2, 2])).1),
        [0]
    );
    let padded = MaxPool2dOptions {
        stride: Some([1, 1]),
        padding: [1, 1],
        ..MaxPool2dOptions::new([2, 2])
    };
    let x = f64s(&[-INF], &[1, 1, 1, 1]);
    x.requires_grad_(true);
    let (y, i) = max_pool2d_with_indices(&x, padded);
    assert_eq!((ints(&i), vals(&y)), (vec![0; 4], vec![-INF; 4]));
    assert_eq!(vals(&sum_grad(&y, &x)), [4.0]);
    let (y, i) = max_pool2d_with_indices(&homura::full_with([1, 2, 2], -128, DType::I8), padded);
    assert_eq!(ints(&i), [0, 0, 1, 0, 0, 1, 2, 2, 3]);
    assert_eq!(ints(&y), [-128; 9]);
    let zeros = homura::zeros_with([1, 2, 2], DType::U8);
    assert_eq!(
        ints(&max_pool2d_with_indices(&zeros, MaxPool2dOptions::new([2, 2])).1),
        [0]
    );
    // Upstream infinities and NaN reach only the selected positions.
    let x = f64s(&[1.0, 5.0, 2.0, 0.0], &[1, 1, 1, 4]);
    x.requires_grad_(true);
    let y = max_pool2d(&x, MaxPool2dOptions::new([1, 2]));
    let g = grad(&y, &x, &f64s(&[INF, NAN], &[1, 1, 1, 2]));
    assert_eq!(bits(&g), bits(&f64s(&[0.0, INF, NAN, 0.0], &[4])));
    // ceil_mode: the last window must start inside the input or the left padding.
    let ceil = |k: usize, stride: usize, padding: usize| MaxPool2dOptions {
        stride: Some([stride; 2]),
        padding: [padding; 2],
        ceil_mode: true,
        ..MaxPool2dOptions::new([k, k])
    };
    let (_, i) = max_pool2d_with_indices(&counting(&[1, 1, 5, 5]), ceil(2, 2, 0));
    assert_eq!(ints(&i), [6, 8, 9, 16, 18, 19, 21, 23, 24]);
    assert_eq!(
        max_pool2d(&counting(&[1, 1, 4, 4]), ceil(2, 2, 1)).shape(),
        [1, 1, 3, 3]
    );
    assert_eq!(
        max_pool2d(&counting(&[1, 1, 5, 5]), ceil(3, 2, 1)).shape(),
        [1, 1, 3, 3]
    );
    assert_eq!(
        max_pool2d(&counting(&[1, 1, 6, 6]), ceil(3, 2, 0)).shape(),
        [1, 1, 3, 3]
    );
    let floor = MaxPool2dOptions {
        stride: Some([2, 2]),
        ..MaxPool2dOptions::new([3, 3])
    };
    assert_eq!(
        max_pool2d(&counting(&[1, 1, 6, 6]), floor).shape(),
        [1, 1, 2, 2]
    );
    // A third window would start in the right padding: two windows, as without ceil_mode.
    let (y, i) = max_pool2d_with_indices(&counting(&[1, 1, 5, 5]), ceil(2, 3, 1));
    assert_eq!(
        (y.shape().to_vec(), ints(&i)),
        (vec![1, 1, 2, 2], vec![0, 3, 15, 18])
    );
    // Dilation, a rectangular kernel, the default stride and plane indices.
    let dilated = MaxPool2dOptions {
        stride: Some([1, 1]),
        dilation: [2, 2],
        ..MaxPool2dOptions::new([2, 2])
    };
    let (_, i) = max_pool2d_with_indices(&counting(&[1, 1, 5, 5]), dilated);
    assert_eq!(ints(&i), [12, 13, 14, 17, 18, 19, 22, 23, 24]);
    let (y, i) = max_pool2d_with_indices(&counting(&[2, 2, 4, 6]), MaxPool2dOptions::new([2, 3]));
    assert_eq!(y.shape(), [2, 2, 2, 2]);
    assert_eq!(ints(&i)[12..], [8, 11, 20, 23]);
    let x = counting(&[1, 3, 3]);
    let explicit = MaxPool2dOptions {
        stride: Some([2, 2]),
        ..MaxPool2dOptions::new([2, 2])
    };
    assert_eq!(
        vals(&max_pool2d(&x, MaxPool2dOptions::new([2, 2]))),
        vals(&max_pool2d(&x, explicit))
    );
    // Overlapping windows add their gradients where they select the same position, also with
    // dilation when the stride equals the kernel (review round 65, R81).
    let x = leaf(&[1, 1, 3, 3]);
    let overlapping = MaxPool2dOptions {
        stride: Some([1, 1]),
        ..MaxPool2dOptions::new([2, 2])
    };
    assert_eq!(
        vals(&sum_grad(&max_pool2d(&x, overlapping), &x)),
        [0., 0., 0., 0., 1., 1., 0., 1., 1.]
    );
    let x = f64s(&[0.0, 0.0, 10.0, 0.0, 0.0], &[1, 1, 1, 5]);
    x.requires_grad_(true);
    let repeating = MaxPool2dOptions {
        stride: Some([1, 2]),
        dilation: [1, 2],
        ..MaxPool2dOptions::new([1, 2])
    };
    let (y, i) = max_pool2d_with_indices(&x, repeating);
    assert_eq!(ints(&i), [2, 2]);
    assert_eq!(vals(&sum_grad(&y, &x)), [0., 0., 2., 0., 0.]);
    // Half precision adds the selected gradients in the dtype, in output order.
    let x = f64s(&[0.0, 0.0, 10.0, 0.0, 0.0], &[1, 1, 1, 5]).to(DType::F16);
    x.requires_grad_(true);
    let sliding = MaxPool2dOptions {
        stride: Some([1, 1]),
        ..MaxPool2dOptions::new([1, 3])
    };
    for (upstream, expected) in [([2048.0, 1.0, 1.0], 2048.0), ([1.0, 1.0, 2048.0], 2050.0)] {
        let y = max_pool2d(&x, sliding);
        let g = grad(&y, &x, &f64s(&upstream, &[1, 1, 1, 3]).to(DType::F16));
        assert_eq!(vals(&g), [0.0, 0.0, expected, 0.0, 0.0]);
    }
    // Numerical checks of the first and second derivatives (values without ties).
    let x = f64s(
        &[
            0.3, -1.2, 0.8, 2.1, -0.4, 1.7, 0.1, -2.3, 1.1, 0.6, -0.9, 1.4,
        ],
        &[1, 1, 3, 4],
    );
    x.requires_grad_(true);
    for options in [
        overlapping,
        MaxPool2dOptions {
            padding: [1, 1],
            ..MaxPool2dOptions::new([2, 3])
        },
        repeating,
    ] {
        let f = |xs: &[Tensor]| vec![max_pool2d(&xs[0].transpose(-2, -1), options)];
        gradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
        gradgradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
    }
}

#[test]
fn max_pool2d_checks_its_arguments() {
    let m = homura::ones([1, 1, 5, 5]);
    let big = (i32::MAX as usize) + 1;
    let k = MaxPool2dOptions::new([2, 2]);
    let overflow = "value cannot be converted to type int without overflow";
    let too_big = [
        MaxPool2dOptions::new([big, big]),
        MaxPool2dOptions {
            stride: Some([big, 1]),
            ..k
        },
        MaxPool2dOptions {
            padding: [0, big],
            ..k
        },
        MaxPool2dOptions {
            dilation: [big, 1],
            ..k
        },
    ];
    for options in too_big {
        assert_panics(
            || drop(max_pool2d(&m.select(0, 0).select(0, 0), options)),
            overflow,
        );
    }
    let rank = "non-empty 3D or 4D (batch mode) tensor expected for input";
    let flat = m.select(0, 0).select(0, 0);
    assert_panics(
        || drop(max_pool2d(&flat, MaxPool2dOptions::new([0, 0]))),
        rank,
    );
    assert_panics(|| drop(max_pool2d(&m.unsqueeze(0), k)), rank);
    let zero_stride = "stride should not be zero";
    assert_panics(
        || drop(max_pool2d(&m, MaxPool2dOptions::new([0, 0]))),
        zero_stride,
    );
    for stride in [[0, 0], [1, 0]] {
        let options = MaxPool2dOptions {
            stride: Some(stride),
            ..MaxPool2dOptions::new([2, 2])
        };
        assert_panics(|| drop(max_pool2d(&m, options)), zero_stride);
    }
    let half_effective = |pad: usize, k: usize, d: usize| {
        format!(
            "pad should be at most half of effective kernel size, but got pad={pad}, \
             kernel_size={k} and dilation={d}"
        )
    };
    let padded = MaxPool2dOptions {
        padding: [2, 2],
        ..k
    };
    assert_panics(|| drop(max_pool2d(&m, padded)), &half_effective(2, 2, 1));
    let rect = MaxPool2dOptions {
        padding: [2, 2],
        dilation: [1, 2],
        ..MaxPool2dOptions::new([2, 3])
    };
    assert_panics(|| drop(max_pool2d(&m, rect)), &half_effective(2, 2, 1));
    let no_dilation = MaxPool2dOptions {
        padding: [1, 1],
        dilation: [0, 0],
        ..k
    };
    assert_panics(
        || drop(max_pool2d(&m, no_dilation)),
        &half_effective(1, 2, 0),
    );
    assert_panics(
        || drop(max_pool2d(&homura::ones([1, 0, 4, 4]), padded)),
        &half_effective(2, 2, 1),
    );
    let kernel_zero = MaxPool2dOptions {
        stride: Some([1, 1]),
        ..MaxPool2dOptions::new([0, 0])
    };
    assert_panics(
        || drop(max_pool2d(&m, kernel_zero)),
        "kernel size should be greater than zero, but got kH: 0 kW: 0",
    );
    let dilation_zero = MaxPool2dOptions {
        dilation: [0, 0],
        ..k
    };
    assert_panics(
        || drop(max_pool2d(&m, dilation_zero)),
        "dilation should be greater than zero, but got dilationH: 0 dilationW: 0",
    );
    for shape in [&[1, 0, 4, 4][..], &[0, 4, 4], &[1, 1, 0, 4], &[1, 0, 1, 1]] {
        assert_panics(
            || drop(max_pool2d(&homura::ones(shape), k)),
            &format!(
                "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, \
                 but got:{shape:?}"
            ),
        );
    }
    // The pad must be at most half of the effective kernel (checked first, height then width)
    // and of the kernel itself (review round 64, R78).
    let nine = homura::ones([1, 1, 9, 9]);
    let dilated = |padding: [usize; 2]| MaxPool2dOptions {
        padding,
        dilation: [2, 2],
        ..MaxPool2dOptions::new([3, 3])
    };
    assert_panics(
        || drop(max_pool2d(&nine, dilated([2, 2]))),
        "pad should be smaller than or equal to half of kernel size, but got padW = 2, padH = 2, \
         kW = 3, kH = 3",
    );
    assert_panics(
        || drop(max_pool2d(&nine, dilated([0, 2]))),
        "pad should be smaller than or equal to half of kernel size, but got padW = 2, padH = 0, \
         kW = 3, kH = 3",
    );
    assert_eq!(max_pool2d(&nine, dilated([1, 1])).shape(), [1, 1, 3, 3]);
    assert_panics(
        || drop(max_pool2d(&homura::ones([1, 1, 1]), k)),
        "Given input size: (1x1x1). Calculated output size: (1x0x0). Output size is too small",
    );
    let wide = MaxPool2dOptions {
        stride: Some([1, 1]),
        padding: [1, 1],
        dilation: [2, 2],
        ..MaxPool2dOptions::new([3, 3])
    };
    assert_panics(
        || drop(max_pool2d(&homura::ones([1, 1, 2, 2]), wide)),
        "Given input size: (1x2x2). Calculated output size: (1x0x0). Output size is too small",
    );
    // PyTorch's meta kernel accepts bool inputs; homura rejects them everywhere.
    let bools = "max_pool2d is not implemented for bool tensors";
    assert_panics(
        || drop(max_pool2d(&homura::ones_with([0, 1, 4, 4], DType::Bool), k)),
        bools,
    );
    let meta_bools = homura::ones_with([1, 4, 4], (DType::Bool, Device::Meta));
    assert_panics(|| drop(max_pool2d(&meta_bools, k)), bools);
}

#[test]
fn max_pool2d_rejects_windows_without_input() {
    // PyTorch outputs -inf with an index outside the plane there, and no gradient (M2.8 #10).
    let gap = MaxPool2dOptions {
        stride: Some([1, 1]),
        padding: [1, 1],
        dilation: [2, 2],
        ..MaxPool2dOptions::new([2, 2])
    };
    let message = "max_pool2d(): the window at output position 0 of the height covers only \
                   padding (input size 1, kernel size 2, stride 1, padding 1, dilation 2); \
                   PyTorch returns -inf there with an index outside the input";
    for device in [Device::Cpu, Device::Meta] {
        let x = homura::ones_with([1, 1, 1, 1], device);
        assert_panics(|| drop(max_pool2d(&x, gap)), message);
    }
    // Controls: every window has an input, and the first column has none, only in width.
    let ceil = MaxPool2dOptions {
        stride: Some([2, 2]),
        padding: [1, 1],
        dilation: [3, 3],
        ceil_mode: true,
        ..MaxPool2dOptions::new([2, 2])
    };
    let (_, i) = max_pool2d_with_indices(&counting(&[1, 1, 3, 3]), ceil);
    assert_eq!(ints(&i), [8, 7, 5, 4]);
    let columns = MaxPool2dOptions {
        stride: Some([1, 1]),
        padding: [0, 1],
        dilation: [1, 2],
        ..MaxPool2dOptions::new([2, 2])
    };
    for device in [Device::Cpu, Device::Meta] {
        let x = homura::ones_with([1, 1, 3, 1], device);
        assert_panics(
            || drop(max_pool2d(&x, columns)),
            "max_pool2d(): the window at output position 0 of the width covers only padding \
             (input size 1, kernel size 2, stride 1, padding 1, dilation 2); PyTorch returns \
             -inf there with an index outside the input",
        );
    }
}

/// The output size of a max pooling dimension of `n` elements and its first window without an
/// input position, found by listing every window's positions; `None` if PyTorch rejects the
/// geometry.
fn pooling_oracle(
    n: i64,
    k: i64,
    s: i64,
    p: i64,
    d: i64,
    ceil_mode: bool,
) -> Option<(usize, Option<usize>)> {
    if p > ((k - 1) * d + 1) / 2 || p > k / 2 {
        return None;
    }
    let numerator = n + 2 * p - d * (k - 1) - 1 + if ceil_mode { s - 1 } else { 0 };
    let mut size = numerator.div_euclid(s) + 1;
    if ceil_mode && (size - 1) * s >= n + p {
        size -= 1;
    }
    if size < 1 {
        return None;
    }
    let empty = (0..size).find(|&o| !(0..k).any(|j| (0..n).contains(&(o * s - p + j * d))));
    Some((size as usize, empty.map(|o| o as usize)))
}

/// Checks `max_pool2d` on the CPU and the meta device for a pooling of the height and then of the
/// width of `n` elements against [`pooling_oracle`], and returns whether a window has no input
/// (`None` if the geometry is rejected for another reason).
fn check_window_geometry(
    n: usize,
    k: usize,
    s: usize,
    p: usize,
    d: usize,
    ceil: bool,
) -> Option<bool> {
    let expected = pooling_oracle(n as i64, k as i64, s as i64, p as i64, d as i64, ceil);
    for (dim, name) in [(0, "height"), (1, "width")] {
        let at = |v: usize, other: usize| if dim == 0 { [v, other] } else { [other, v] };
        let options = MaxPool2dOptions {
            kernel_size: at(k, 1),
            stride: Some(at(s, 1)),
            padding: at(p, 0),
            dilation: at(d, 1),
            ceil_mode: ceil,
        };
        let [h, w] = at(n, 2);
        for device in [Device::Cpu, Device::Meta] {
            let x = homura::ones_with([1, 1, h, w], device);
            let pool = || drop(max_pool2d(&x, options));
            match expected {
                // PyTorch rejects these too, for the reasons max_pool2d_checks_its_arguments tests.
                None => assert!(catch_unwind(AssertUnwindSafe(pool)).is_err()),
                Some((_, Some(o))) => assert_panics(
                    pool,
                    &format!(
                        "max_pool2d(): the window at output position {o} of the {name} covers \
                         only padding (input size {n}, kernel size {k}, stride {s}, padding {p}, \
                         dilation {d}); PyTorch returns -inf there with an index outside the input"
                    ),
                ),
                Some((size, None)) => {
                    let [oh, ow] = at(size, 2);
                    assert_eq!(max_pool2d(&x, options).shape(), [1, 1, oh, ow]);
                }
            }
        }
    }
    expected.map(|(_, empty)| empty.is_some())
}

#[test]
fn max_pool2d_window_check_matches_the_windows() {
    // The check for windows without input looks only at the windows that can have none; it finds
    // the same first window as listing them all, on the CPU and the meta device, for every small
    // geometry (review round 66, R83).
    let (mut valid, mut rejected) = (0, 0);
    for n in 1..=6 {
        for k in 1..=4 {
            for s in 1..=3 {
                for p in 0..=2 {
                    for d in 1..=4 {
                        for ceil in [false, true] {
                            match check_window_geometry(n, k, s, p, d, ceil) {
                                Some(true) => rejected += 1,
                                Some(false) => valid += 1,
                                None => {}
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!((valid, rejected), (616, 24));
    // The check takes the same time for any output size: these inputs have 2^50 rows or columns
    // on the meta device.
    let huge = 1usize << 50;
    let x = homura::ones_with([1, 1, huge, 1], Device::Meta);
    assert_eq!(
        max_pool2d(&x, MaxPool2dOptions::new([1, 1])).shape(),
        [1, 1, huge, 1]
    );
    let options = MaxPool2dOptions {
        stride: Some([1, 1]),
        padding: [1, 1],
        dilation: [2, 1],
        ceil_mode: true,
        ..MaxPool2dOptions::new([3, 3])
    };
    let x = homura::ones_with([1, 1, 3, huge], Device::Meta);
    let (y, i) = max_pool2d_with_indices(&x, options);
    assert_eq!(
        (y.shape(), i.shape()),
        ([1, 1, 1, huge].into(), [1, 1, 1, huge].into())
    );
}

#[test]
fn max_pool2d_nodes_and_saved_tensors() {
    let x = leaf(&[1, 1, 4, 4]);
    let k = MaxPool2dOptions::new([2, 2]);
    let node = [(0, "MaxPool2DWithIndicesBackward0"), (1, ACC)];
    assert_eq!(chain(&max_pool2d(&x, k)), node);
    let (y, i) = max_pool2d_with_indices(&x, k);
    assert_eq!(chain(&y), node);
    assert_eq!(i.dtype(), DType::I64);
    assert!(!i.requires_grad() && i.grad_fn().is_none());
    let z = x.mul(1.0);
    let y = max_pool2d(&z, k);
    let gin = homura::ones_with([1, 1, 2, 2], DType::F64);
    gin.requires_grad_(true);
    let gx = grads(&y, &[&z], &gin, true).remove(0).unwrap();
    assert_eq!(
        chain(&gx),
        [
            (0, "MaxPool2DWithIndicesBackwardBackward0"),
            (1, ACC),
            (1, "MulBackward0"),
            (2, ACC)
        ]
    );
    let total = gx.mul(counting(&[1, 1, 4, 4])).sum();
    let second = autograd::grad(
        &[&total],
        &[&z, &gin],
        GradOptions {
            retain_graph: Some(true),
            ..Default::default()
        },
    );
    assert_eq!(vals(second[0].as_ref().unwrap()), [0.0; 16]);
    assert_eq!(vals(second[1].as_ref().unwrap()), [6.0, 8.0, 14.0, 16.0]);
    let r = counting(&[1, 1, 4, 4]);
    r.requires_grad_(true);
    let gg = grads(&gx, &[&gin], &r, true).remove(0).unwrap();
    assert_eq!(vals(&gg), [6.0, 8.0, 14.0, 16.0]);
    // Modifying the input or the indices after the forward fails the backward pass.
    let version = "one of the variables needed for gradient computation has been modified by an \
                   inplace operation";
    let z = x.mul(1.0);
    let (y, _) = max_pool2d_with_indices(&z, k);
    {
        let _no_grad = homura::no_grad();
        z.add_(1.0);
    }
    assert!(panic_message(|| drop(sum_grad(&y, &x))).starts_with(version));
    let z = x.mul(1.0);
    let (y, i) = max_pool2d_with_indices(&z, k);
    i.zero_();
    assert!(panic_message(|| drop(sum_grad(&y, &x))).starts_with(version));
    // The second derivatives need the indices whichever are requested, so modifying them after
    // the first backward pass fails the second one; these nodes don't save the input (review
    // round 66, R82).
    for modified in ["none", "indices", "input"] {
        for wrt in [&[0][..], &[1], &[0, 1]] {
            let z = x.mul(1.0);
            let (y, i) = max_pool2d_with_indices(&z, k);
            let gin = homura::ones_with([1, 1, 2, 2], DType::F64);
            gin.requires_grad_(true);
            let gx = grads(&y, &[&z], &gin, true).remove(0).unwrap();
            match modified {
                "indices" => {
                    i.zero_();
                }
                "input" => {
                    let _no_grad = homura::no_grad();
                    z.add_(1.0);
                }
                _ => {}
            }
            let inputs: Vec<&Tensor> = wrt.iter().map(|&w| [&z, &gin][w]).collect();
            let second = || {
                autograd::grad(
                    &[&gx],
                    &inputs,
                    GradOptions {
                        grad_outputs: vec![Some(counting(&[1, 1, 4, 4]))],
                        ..Default::default()
                    },
                )
            };
            if modified == "indices" {
                let message = panic_message(|| drop(second()));
                assert!(message.starts_with(version), "{wrt:?}: {message}");
                continue;
            }
            for (&w, t) in wrt.iter().zip(second()) {
                let expected = [vec![0.0; 16], vec![6.0, 8.0, 14.0, 16.0]];
                assert_eq!(vals(&t.unwrap()), expected[w], "{modified} {wrt:?}");
            }
        }
    }
}

#[test]
fn max_pool2d_shapes_dtypes_and_meta() {
    let k = MaxPool2dOptions::new([2, 2]);
    assert_eq!(
        max_pool2d(&homura::ones([0, 1, 4, 4]), k).shape(),
        [0, 1, 2, 2]
    );
    assert_eq!(max_pool2d(&counting(&[2, 4, 4]), k).shape(), [2, 2, 2]);
    let values = homura::arange_with(16, DType::I64).reshape([1, 4, 4]);
    for dtype in [
        DType::U8,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
    ] {
        let y = max_pool2d(&values.to(dtype), k);
        assert_eq!(y.dtype(), dtype);
        assert_eq!(ints(&y), [5, 7, 13, 15], "{dtype}");
    }
    let options = MaxPool2dOptions {
        stride: Some([2, 2]),
        padding: [1, 1],
        ..MaxPool2dOptions::new([3, 3])
    };
    let (y, i) = max_pool2d_with_indices(&homura::ones_with([2, 3, 7, 7], Device::Meta), options);
    assert_eq!(
        (y.shape(), i.shape()),
        ([2, 3, 4, 4].into(), [2, 3, 4, 4].into())
    );
    assert_eq!((i.dtype(), i.device()), (DType::I64, Device::Meta));
    let xm = homura::ones_with([2, 3, 7, 7], Device::Meta);
    xm.requires_grad_(true);
    let gm = sum_grad(&max_pool2d(&xm, MaxPool2dOptions::new([3, 3])), &xm);
    assert_eq!(
        (gm.shape().to_vec(), gm.device()),
        (vec![2, 3, 7, 7], Device::Meta)
    );
    // Empty batches and inputs of any strides.
    let empty = homura::ones_with([0, 2, 4, 4], DType::F64);
    empty.requires_grad_(true);
    assert_eq!(
        sum_grad(&max_pool2d(&empty, k), &empty).shape(),
        [0, 2, 4, 4]
    );
    let odd = homura::zeros([0]).as_strided([0, 1, 4, 4], [usize::MAX / 8, 1, 4, 1], 0);
    assert_eq!(max_pool2d(&odd, k).shape(), [0, 1, 2, 2]);
}

#[test]
fn adaptive_avg_pool2d_windows_and_mean_path() {
    let x = homura::arange_with(5.0, DType::F64).reshape([1, 1, 1, 5]);
    assert_eq!(vals(&adaptive_avg_pool2d(&x, [1, 3])), [0.5, 2.0, 3.5]);
    let x = f64s(&[0.0, 1.0], &[1, 1, 1, 2]);
    assert_eq!(vals(&adaptive_avg_pool2d(&x, [1, 3])), [0.0, 0.5, 1.0]);
    let x = f64s(&[1e8, -1e8, 1.0, 1.0, 0.0, 0.0], &[1, 1, 3, 2]).to(DType::F32);
    assert_eq!(vals(&adaptive_avg_pool2d(&x, [2, 1])), [0.5, 0.5]);
    let half = f64s(&[2048.0, 1.0, 1.0, 1.0], &[1, 1, 1, 4]).to(DType::F16);
    assert_eq!(vals(&adaptive_avg_pool2d(&half, [1, 2])), [1024.0, 1.0]);
    // [1, 1] is a mean over the last two dimensions, recorded as MeanBackward1, for any rank of
    // at least 3.
    let x = leaf(&[2, 3, 4, 5]);
    assert_eq!(
        chain(&adaptive_avg_pool2d(&x, [1, 1])),
        [(0, "MeanBackward1"), (1, ACC)]
    );
    assert_eq!(
        chain(&adaptive_avg_pool2d(&x, [1, 2])),
        [(0, "AdaptiveAvgPool2DBackward0"), (1, ACC)]
    );
    assert_eq!(
        adaptive_avg_pool2d(&homura::ones([1, 1, 1, 4, 4]), [1, 1]).shape(),
        [1, 1, 1, 1, 1]
    );
    assert!(vals(&adaptive_avg_pool2d(&homura::ones([1, 0, 3]), [1, 1]))[0].is_nan());
    assert_panics(
        || {
            drop(adaptive_avg_pool2d(
                &homura::ones_with([1, 1, 4, 4], DType::I64),
                [1, 1],
            ))
        },
        "mean(): could not infer output dtype. Input dtype must be either a floating point or \
         complex dtype. Got: i64",
    );
    // A zero output size gives an empty output whose gradient is zeros; PyTorch's backward pass
    // rejects it.
    let x = leaf(&[1, 1, 4, 4]);
    let y = adaptive_avg_pool2d(&x, [0, 2]);
    assert_eq!(y.shape(), [1, 1, 0, 2]);
    assert_eq!(vals(&sum_grad(&y, &x)), [0.0; 16]);
    let g = homura::zeros_with([1, 1, 0, 2], DType::F64);
    g.requires_grad_(true);
    let y = adaptive_avg_pool2d(&x, [0, 2]);
    let gx = grads(&y, &[&x], &g, true).remove(0).unwrap();
    assert_eq!(chain(&gx)[0], (0, "AdaptiveAvgPool2DBackwardBackward0"));
    assert_eq!(
        adaptive_avg_pool2d(&homura::ones([0, 1, 4, 4]), [2, 2]).shape(),
        [0, 1, 2, 2]
    );
    assert_eq!(
        adaptive_avg_pool2d(&homura::ones([1, 0, 4, 4]), [2, 2]).shape(),
        [1, 0, 2, 2]
    );
    let m = adaptive_avg_pool2d(&homura::ones_with([2, 4, 4], Device::Meta), [3, 5]);
    assert_eq!(
        (m.shape().to_vec(), m.device()),
        (vec![2, 3, 5], Device::Meta)
    );
    let xm = homura::ones_with([2, 4, 4], Device::Meta);
    xm.requires_grad_(true);
    let gm = sum_grad(&adaptive_avg_pool2d(&xm, [3, 5]), &xm);
    assert_eq!(
        (gm.shape().to_vec(), gm.device()),
        (vec![2, 4, 4], Device::Meta)
    );
    // Numerical checks of the first and second derivatives, on non-contiguous inputs.
    let x = leaf(&[1, 2, 5, 4]);
    for size in [[3, 2], [2, 5], [1, 3], [1, 1]] {
        let f = |xs: &[Tensor]| vec![adaptive_avg_pool2d(&xs[0].transpose(-2, -1), size)];
        gradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
        gradgradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
    }
}

#[test]
fn adaptive_avg_pool2d_checks_its_arguments() {
    for shape in [&[][..], &[4], &[4, 4]] {
        for size in [[1, 1], [2, 2]] {
            assert_panics(
                || drop(adaptive_avg_pool2d(&homura::ones(shape), size)),
                "Input dimension should be at least 3",
            );
        }
    }
    assert_panics(
        || drop(adaptive_avg_pool2d(&homura::ones([1, 1, 1, 4, 4]), [2, 2])),
        "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got [1, 1, 1, 4, 4]",
    );
    assert_panics(
        || {
            drop(adaptive_avg_pool2d(
                &homura::ones_with([1, 0, 4], DType::I64),
                [2, 2],
            ))
        },
        "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, \
         but input has sizes [1, 0, 4] with dimension 1 being empty",
    );
    assert_panics(
        || drop(adaptive_avg_pool2d(&homura::ones([1, 1, 3, 0]), [2, 2])),
        "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, \
         but input has sizes [1, 1, 3, 0] with dimension 3 being empty",
    );
    for dtype in [DType::I64, DType::Bool] {
        assert_panics(
            || {
                drop(adaptive_avg_pool2d(
                    &homura::ones_with([1, 4, 4], dtype),
                    [2, 2],
                ))
            },
            &format!("adaptive_avg_pool2d is not implemented for {dtype} tensors"),
        );
    }
    // PyTorch checks the dtype only where it computes elements; homura everywhere.
    let ints = homura::ones_with([1, 1, 4, 4], DType::I64);
    assert_panics(
        || drop(adaptive_avg_pool2d(&ints, [0, 2])),
        "adaptive_avg_pool2d is not implemented for i64 tensors",
    );
    let meta = homura::ones_with([1, 1, 4, 4], (DType::I64, Device::Meta));
    assert_panics(
        || drop(adaptive_avg_pool2d(&meta, [2, 2])),
        "adaptive_avg_pool2d is not implemented for i64 tensors",
    );
}

#[test]
fn adaptive_avg_pool2d_nodes_and_saved_tensors() {
    let a = leaf(&[1, 2, 5, 4]);
    let z = a.mul(1.0);
    let y = adaptive_avg_pool2d(&z, [2, 3]);
    let gin = homura::ones_with([1, 2, 2, 3], DType::F64);
    gin.requires_grad_(true);
    let gx = grads(&y, &[&z], &gin, true).remove(0).unwrap();
    assert_eq!(
        chain(&gx),
        [
            (0, "AdaptiveAvgPool2DBackwardBackward0"),
            (1, ACC),
            (1, "MulBackward0"),
            (2, ACC)
        ]
    );
    let total = gx.mul(counting(&[1, 2, 5, 4])).sum();
    let second = autograd::grad(
        &[&total],
        &[&z, &gin],
        GradOptions {
            retain_graph: Some(true),
            ..Default::default()
        },
    );
    assert_eq!(vals(second[0].as_ref().unwrap()), [0.0; 40]);
    assert_eq!(
        bits(second[1].as_ref().unwrap()),
        bits(&adaptive_avg_pool2d(&counting(&[1, 2, 5, 4]), [2, 3]))
    );
    let r = counting(&[1, 2, 5, 4]);
    r.requires_grad_(true);
    let gg = grads(&gx, &[&gin], &r, true).remove(0).unwrap();
    assert_eq!(chain(&gg), [(0, "AdaptiveAvgPool2DBackward0"), (1, ACC)]);
    {
        let _no_grad = homura::no_grad();
        z.add_(1.0);
    }
    let message = panic_message(|| drop(sum_grad(&y, &a)));
    assert!(
        message.starts_with(
            "one of the variables needed for gradient computation has been modified by an \
             inplace operation"
        ),
        "{message}"
    );
}
