//! Pooling and unfold on a ROCm GPU against the CPU (M4 step 4b): max_pool2d (values and
//! indices, every float and integer dtype, NaNs and ties, M2 step 8's kernel / stride / padding /
//! dilation / ceil_mode combinations), adaptive_avg_pool2d (windows smaller and larger than the
//! input), and the gradients of both and of unfold, which add overlapping contributions one at a
//! time in the dtype: every result with the CPU's bits.
#![cfg(feature = "rocm")]

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

#[test]
fn max_pool2d_gradients_add_like_the_cpu() {
    let shape = [2, 3, 9, 11];
    let n: usize = shape.iter().product();
    for (s, &dtype) in FLOATS.iter().enumerate() {
        // Overlapping windows over many ties: several outputs pick the same input, whose
        // gradient adds them in output order, rounding each addition in the dtype.
        let x = tied(dtype, n, 40 + s as u64).reshape([2, 3, 9, 11]);
        for o in pool_options().into_iter().step_by(5) {
            let what = format!("{dtype} {o:?}");
            compare_gradients(&what, &|x| F::max_pool2d(x, o), &x, 50 + s as u64);
        }
    }
}

#[test]
fn adaptive_avg_pool2d_matches_the_cpu() {
    let shape = [2, 3, 9, 11];
    let n: usize = shape.iter().product();
    for (s, &dtype) in FLOATS.iter().enumerate() {
        let x = values(dtype, 2 * n + 6, 60 + s as u64);
        for (lname, cpu, gpu) in inputs(&x, shape) {
            // Not [1, 1]: that output is `mean_dim`, a reduction (step 4a, in the tolerance class).
            for out in [[3, 4], [9, 11], [13, 17], [5, 2], [2, 23], [1, 3]] {
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
        for out in [[1, 1], [3, 4], [9, 11], [13, 17], [2, 23]] {
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
