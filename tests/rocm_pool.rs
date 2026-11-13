//! Pooling and unfold on a ROCm GPU against the CPU (M4 step 4b): max_pool2d (values and
//! indices, every float and integer dtype, NaNs and ties, M2 step 8's kernel / stride / padding /
//! dilation / ceil_mode combinations), adaptive_avg_pool2d (windows smaller and larger than the
//! input), and the gradients of both and of unfold, which add overlapping contributions one at a
//! time in the dtype: every result with the CPU's bits. The gradient of max_pool2d gathers, for
//! each input position, the windows that cover it (step 4c): every geometry, the CPU's order of
//! the additions, second-order gradients.
#![cfg(feature = "rocm")]

use std::panic::{AssertUnwindSafe, catch_unwind};

#[path = "common/rocm.rs"]
mod rocm;

use homura::autograd::{self, GradOptions};
use homura::nn::functional::{self as F, MaxPool2dOptions};
use homura::{DType, Device, Tensor};
use rocm::{GPU, Words, assert_bitwise, assert_same, values};

const FLOATS: [DType; 4] = [DType::F16, DType::BF16, DType::F32, DType::F64];
const INTS: [DType; 5] = [DType::U8, DType::I8, DType::I16, DType::I32, DType::I64];

/// `x` of `shape` on the CPU in three layouts (contiguous, the spatial dimensions transposed in
/// memory, narrowed with an offset), with the same views on the GPU.
fn inputs(x: &Tensor, shape: [usize; 4]) -> Vec<(&'static str, Tensor, Tensor)> {
    let [n, c, h, w] = shape.map(|s| s as i64);
    let contiguous = |t: &Tensor| t.reshape([n, c, h, w]);
    let transposed = |t: &Tensor| t.reshape([n, c, w, h]).transpose(2, 3);
    let narrowed = |t: &Tensor| t.reshape([n, c, h, w + 3]).narrow(3, 2, w as usize);
    let n_all = (n * c * h * w) as usize;
    let wide = x
        .reshape([-1])
        .repeat([2])
        .narrow(0, 0, (n * c * h * (w + 3)) as usize);
    let base = x.reshape([-1]).narrow(0, 0, n_all);
    vec![
        ("contiguous", contiguous(&base), contiguous(&base.to(GPU))),
        ("transposed", transposed(&base), transposed(&base.to(GPU))),
        ("narrowed", narrowed(&wide), narrowed(&wide.to(GPU))),
    ]
}

/// Values with many ties (a few distinct values) and, for floats, NaNs and infinities.
fn tied(dtype: DType, n: usize, seed: u64) -> Tensor {
    let mut w = Words::new(seed);
    let v: Vec<f64> = (0..n)
        .map(|_| match w.next() % 9 {
            0 if dtype.is_floating_point() => f64::NAN,
            1 if dtype.is_floating_point() => f64::NEG_INFINITY,
            k => (k % 3) as f64,
        })
        .collect();
    Tensor::from_slice(&v, [n]).to(dtype)
}

fn pool_options() -> Vec<MaxPool2dOptions> {
    let mut out = vec![];
    for kernel in [[2, 2], [3, 2], [3, 3]] {
        for stride in [None, Some([1, 2]), Some([2, 1])] {
            for padding in [[0, 0], [1, 0], [1, 1]] {
                for dilation in [[1, 1], [2, 1]] {
                    for ceil_mode in [false, true] {
                        let mut o = MaxPool2dOptions::new(kernel);
                        o.stride = stride;
                        o.padding = padding;
                        o.dilation = dilation;
                        o.ceil_mode = ceil_mode;
                        out.push(o);
                    }
                }
            }
        }
    }
    out
}

#[test]
fn max_pool2d_matches_the_cpu() {
    let shape = [2, 3, 9, 11];
    let n: usize = shape.iter().product();
    for (s, &dtype) in FLOATS.iter().chain(&INTS).enumerate() {
        for (vname, x) in [
            ("random", values(dtype, 2 * n + 6, 20 + s as u64)),
            ("tied", tied(dtype, 2 * n + 6, 30 + s as u64)),
        ] {
            for (lname, cpu, gpu) in inputs(&x, shape) {
                for o in pool_options() {
                    let what = format!("{dtype} {vname} {lname} {o:?}");
                    let (c, ci) = F::max_pool2d_with_indices(&cpu, o);
                    let (g, gi) = F::max_pool2d_with_indices(&gpu, o);
                    assert_same(&g, &c, &what);
                    assert_same(&gi, &ci, &format!("{what}: indices"));
                }
            }
        }
    }
}

/// `d<f(x), v>/dx` on `device` for a seeded `v`, as a CPU tensor.
fn gradient(f: &dyn Fn(&Tensor) -> Tensor, x: &Tensor, device: Device, seed: u64) -> Tensor {
    let x = x.to(device).detach();
    x.requires_grad_(true);
    let y = f(&x);
    let v = values(y.dtype(), y.num_elements(), seed)
        .reshape(y.shape())
        .to(device);
    autograd::grad(
        &[&y],
        &[&x],
        GradOptions {
            grad_outputs: vec![Some(v)],
            ..Default::default()
        },
    )
    .remove(0)
    .expect("a gradient")
    .to(Device::Cpu)
}

#[track_caller]
fn compare_gradients(what: &str, f: &dyn Fn(&Tensor) -> Tensor, x: &Tensor, seed: u64) {
    let c = gradient(f, x, Device::Cpu, seed);
    let g = gradient(f, x, GPU, seed).to(GPU);
    assert_bitwise(&g, &c, what);
}

/// Options with a stride, padding and dilation for both dimensions and no ceil mode.
fn options(
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> MaxPool2dOptions {
    let mut o = MaxPool2dOptions::new(kernel);
    o.stride = Some(stride);
    o.padding = padding;
    o.dilation = dilation;
    o
}

/// Geometries beyond `pool_options`, with the input shape for each: strides above the kernel
/// size (inputs no window covers, whose gradient is 0), large dilations (a position covered by
/// up to 9 windows of a 3 × 3 kernel with stride 2, review round 133, R157), ceil mode's last
/// window cut by the input, outputs of 1 × 1, and dilations far larger than the kernel, also
/// different for the rows and the columns, where the gather visits only the windows that cover a
/// position, not their whole span (review round 135, R159).
fn more_options() -> Vec<([usize; 4], MaxPool2dOptions)> {
    let mut ceil = options([3, 3], [2, 2], [1, 1], [1, 1]);
    ceil.ceil_mode = true;
    let mut ceil_dilated = options([2, 3], [3, 2], [1, 0], [2, 3]);
    ceil_dilated.ceil_mode = true;
    vec![
        ([2, 3, 9, 11], options([2, 2], [3, 3], [0, 0], [1, 1])),
        ([2, 3, 9, 11], options([1, 1], [2, 3], [0, 0], [1, 1])),
        ([2, 3, 9, 11], options([2, 3], [4, 5], [1, 1], [1, 1])),
        ([2, 3, 9, 11], options([3, 3], [2, 2], [0, 0], [2, 2])),
        ([2, 3, 13, 13], options([3, 3], [2, 2], [1, 1], [3, 2])),
        ([1, 1, 9, 9], options([3, 3], [2, 2], [0, 0], [2, 2])),
        ([2, 3, 10, 12], ceil),
        ([2, 3, 9, 11], ceil_dilated),
        ([2, 3, 3, 3], options([3, 3], [1, 1], [0, 0], [1, 1])),
        ([2, 3, 5, 5], options([3, 3], [2, 2], [1, 1], [2, 2])),
        ([3, 1, 1, 1], options([1, 1], [1, 1], [0, 0], [1, 1])),
        (
            [1, 1, 512, 512],
            options([2, 2], [1, 1], [0, 0], [256, 256]),
        ),
        ([2, 3, 64, 64], options([3, 3], [1, 1], [0, 0], [16, 16])),
        ([2, 3, 9, 17], options([2, 3], [1, 2], [1, 1], [1, 3])),
        ([2, 3, 20, 30], options([3, 2], [2, 1], [1, 0], [7, 1])),
    ]
}

#[test]
fn max_pool2d_gradients_add_like_the_cpu() {
    for (s, &dtype) in FLOATS.iter().enumerate() {
        // Overlapping windows over many ties: several outputs pick the same input, whose
        // gradient adds them in output order, rounding each addition in the dtype.
        let cases = pool_options()
            .into_iter()
            .map(|o| ([2, 3, 9, 11], o))
            .chain(more_options());
        for (k, (shape, o)) in cases.enumerate() {
            let n: usize = shape.iter().product();
            let x = tied(dtype, n, 40 + s as u64 + k as u64).reshape(shape);
            let what = format!("{dtype} {shape:?} {o:?}");
            compare_gradients(&what, &|x| F::max_pool2d(x, o), &x, 50 + s as u64);
        }
        // A strided gradient: the output transposed (its gradient is a transposed view).
        for o in pool_options().into_iter().step_by(7) {
            let x = tied(dtype, 2 * 3 * 9 * 11, 45 + s as u64).reshape([2, 3, 9, 11]);
            let what = format!("{dtype} {o:?}, transposed gradient");
            compare_gradients(
                &what,
                &|x| F::max_pool2d(x, o).transpose(2, 3),
                &x,
                55 + s as u64,
            );
        }
    }
}

/// `d<max_pool2d(x), v>/dx` on `device` for this `v`, as a CPU tensor.
fn gradient_for(x: &Tensor, o: MaxPool2dOptions, v: &Tensor, device: Device) -> Tensor {
    let x = x.to(device).detach();
    x.requires_grad_(true);
    let y = F::max_pool2d(&x, o);
    autograd::grad(
        &[&y],
        &[&x],
        GradOptions {
            grad_outputs: vec![Some(v.reshape(y.shape()).to(device))],
            ..Default::default()
        },
    )
    .remove(0)
    .expect("a gradient")
    .to(Device::Cpu)
}

/// Sums whose value depends on the order of the additions: an input picked by several windows
/// whose gradients in output order are 2048 and then ones, in `F16`, where 2048 + 1 rounds back
/// to 2048. The CPU's order gives 2048, the reverse 2048 + the number of ones.
#[test]
fn max_pool2d_gradients_add_in_the_cpus_order() {
    // A row of 5 with the maximum in the middle, picked by 3 windows of width 3; and the case of
    // R157, a 9 × 9 plane whose centre all 9 windows (kernel 3, stride 2, dilation 2) pick.
    let mut row = vec![0.0; 5];
    row[2] = 1.0;
    let mut plane = vec![0.0; 81];
    plane[40] = 1.0;
    let cases = [
        (
            Tensor::from_slice(&row, [1, 1, 1, 5]),
            options([1, 3], [1, 1], [0, 0], [1, 1]),
            3,
            2,
            2050.0,
        ),
        (
            Tensor::from_slice(&plane, [1, 1, 9, 9]),
            options([3, 3], [2, 2], [0, 0], [2, 2]),
            9,
            40,
            2056.0,
        ),
    ];
    for (x, o, windows, at, reversed) in cases {
        let x = x.to(DType::F16);
        let (_, indices) = F::max_pool2d_with_indices(&x, o);
        assert_eq!(indices.to_vec::<i64>(), vec![at as i64; windows], "{o:?}");
        let mut v = vec![1.0; windows];
        v[0] = 2048.0;
        let v = Tensor::from_slice(&v, [windows]).to(DType::F16);
        let c = gradient_for(&x, o, &v, Device::Cpu);
        let g = gradient_for(&x, o, &v, GPU);
        assert_eq!(c.to(DType::F64).to_vec::<f64>()[at], 2048.0, "{o:?}");
        assert_bitwise(&g.to(GPU), &c, &format!("{o:?}"));
        // The reverse order would give another value, so the test sees the order.
        let backwards: f64 = (0..windows).rev().fold(0.0, |acc, k| {
            let term = if k == 0 { 2048.0 } else { 1.0 };
            Tensor::from_slice(&[acc + term], [1])
                .to(DType::F16)
                .to(DType::F64)
                .to_vec::<f64>()[0]
        });
        assert_eq!(backwards, reversed, "{o:?}");
    }
}

/// The gradient of the gradient: with respect to the upstream gradient, it gathers at the
/// indices; with respect to the input, zeros.
#[test]
fn max_pool2d_second_order_gradients_match_the_cpu() {
    for (s, &dtype) in FLOATS.iter().enumerate() {
        for (shape, o) in more_options().into_iter().step_by(2) {
            let n: usize = shape.iter().product();
            let x = tied(dtype, n, 100 + s as u64).reshape(shape);
            let run = |device: Device| {
                let x = x.to(device).detach();
                x.requires_grad_(true);
                let y = F::max_pool2d(&x, o);
                let v = values(dtype, y.num_elements(), 110 + s as u64)
                    .reshape(y.shape())
                    .to(device);
                v.requires_grad_(true);
                let gx = autograd::grad(
                    &[&y],
                    &[&x],
                    GradOptions {
                        grad_outputs: vec![Some(v.clone())],
                        create_graph: true,
                        ..Default::default()
                    },
                )
                .remove(0)
                .expect("a gradient");
                let w = values(dtype, n, 120 + s as u64).reshape(shape).to(device);
                autograd::grad(
                    &[&gx],
                    &[&v, &x],
                    GradOptions {
                        grad_outputs: vec![Some(w)],
                        allow_unused: true,
                        ..Default::default()
                    },
                )
                .into_iter()
                .map(|g| g.map(|g| g.to(Device::Cpu)))
                .collect::<Vec<_>>()
            };
            let (c, g) = (run(Device::Cpu), run(GPU));
            let what = format!("{dtype} {shape:?} {o:?}");
            assert_eq!(c.len(), g.len());
            for (k, (c, g)) in c.iter().zip(&g).enumerate() {
                match (c, g) {
                    (Some(c), Some(g)) => assert_bitwise(&g.to(GPU), c, &format!("{what} {k}")),
                    (None, None) => {}
                    _ => panic!("{what} {k}: one gradient is missing"),
                }
            }
        }
    }
}

/// A window that covers only padding is rejected before any kernel runs, on the GPU as on the
/// CPU (`tests/nn_pool.rs`), so the backward pass never sees one (review round 133, R158).
#[test]
fn max_pool2d_rejects_windows_of_padding_on_the_gpu() {
    let columns = MaxPool2dOptions {
        stride: Some([1, 1]),
        padding: [0, 1],
        dilation: [1, 2],
        ..MaxPool2dOptions::new([2, 2])
    };
    let message = |device: Device| {
        let x = homura::ones_with([1, 1, 3, 1], device);
        let payload = catch_unwind(AssertUnwindSafe(|| drop(F::max_pool2d(&x, columns))))
            .expect_err("expected a panic");
        payload
            .downcast_ref::<String>()
            .cloned()
            .expect("a message")
    };
    assert_eq!(message(GPU), message(Device::Cpu));
}

#[test]
fn adaptive_avg_pool2d_matches_the_cpu() {
    let shape = [2, 3, 9, 11];
    let n: usize = shape.iter().product();
    for (s, &dtype) in FLOATS.iter().enumerate() {
        let x = values(dtype, 2 * n + 6, 60 + s as u64);
        for (lname, cpu, gpu) in inputs(&x, shape) {
            // Not [1, 1]: that output is `mean_dim`, a reduction (step 4a, in the tolerance class).
            for out in [[3, 4], [9, 11], [13, 17], [5, 2], [2, 23], [1, 3], [0, 3]] {
                let what = format!("{dtype} {lname} {out:?}");
                let c = F::adaptive_avg_pool2d(&cpu, out);
                let g = F::adaptive_avg_pool2d(&gpu, out);
                // Sums: a NaN only has to be a NaN (its sign and payload come from the hardware).
                assert_bitwise(&g, &c, &what);
            }
        }
        // Gradients from ordinary values (the windows overlap when the output is larger).
        let x = Tensor::from_slice(
            &(0..n)
                .map(|i| (i % 17) as f64 * 0.37 - 2.0)
                .collect::<Vec<_>>(),
            shape,
        )
        .to(dtype);
        // An output size of 0: an empty gradient, so the input's is zero.
        for out in [[1, 1], [3, 4], [9, 11], [13, 17], [2, 23], [0, 3]] {
            let what = format!("gradient {dtype} {out:?}");
            compare_gradients(
                &what,
                &|x| F::adaptive_avg_pool2d(x, out),
                &x,
                70 + s as u64,
            );
        }
    }
}

#[test]
fn unfold_gradients_match_the_cpu() {
    for (s, &dtype) in FLOATS.iter().enumerate() {
        let x = values(dtype, 4 * 10 * 3, 80 + s as u64).reshape([4, 10, 3]);
        // Overlapping windows, adjacent ones, gaps, windows of one element and none.
        for (size, step) in [(3, 1), (4, 2), (3, 3), (3, 4), (1, 2), (0, 1)] {
            for dim in 0..3 {
                if size > x.shape()[dim] {
                    continue;
                }
                let what = format!("{dtype} dim {dim} size {size} step {step}");
                compare_gradients(
                    &what,
                    &|x| x.unfold(dim as i64, size, step),
                    &x,
                    90 + s as u64,
                );
            }
        }
        // A strided input, and a zero-dimensional one.
        let t = x.transpose(0, 2);
        compare_gradients(&format!("{dtype} strided"), &|x| x.unfold(1, 3, 2), &t, 91);
        let scalar = x.reshape([-1]).select(0, 5);
        compare_gradients(
            &format!("{dtype} 0-dim"),
            &|x| x.unfold(0, 1, 1),
            &scalar,
            92,
        );
    }
}
