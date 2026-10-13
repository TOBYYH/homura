//! Regression checks for the issues found while reviewing M1. `tools/golden/check_m1_formulas.py`
//! checks the PyTorch side of the gradient and half precision cases. Run in debug and release
//! profiles.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::FRAC_2_SQRT_PI;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};

use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::{DType, Device, GeluApproximate, Tensor};

const DEVICES: [Device; 2] = [Device::Cpu, Device::Meta];

/// Element values with NaN compared by bit pattern, so that NaN equals NaN.
fn bits(t: &Tensor) -> Vec<u64> {
    t.to(DType::F64)
        .to_vec::<f64>()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The system allocator, recording the largest allocation made by a thread that is
/// [`tracking`](largest_allocation) its allocations.
struct Counting;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: forwards to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.try_with(Cell::get).unwrap_or(false) {
            LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: the caller's guarantees.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's guarantees.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The size of the largest allocation `f` makes on this thread. Only one test uses it.
fn largest_allocation(f: impl FnOnce()) -> usize {
    LARGEST.store(0, Ordering::Relaxed);
    TRACKING.set(true);
    f();
    TRACKING.set(false);
    LARGEST.load(Ordering::Relaxed)
}

fn panic_message(f: impl FnOnce()) -> String {
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

/// Asserts that the values of `t` are within `rtol` of `expected`, relative to each expected
/// value, so that zeros and infinities must match exactly, and NaN only NaN.
fn assert_values(t: &Tensor, expected: &[f64], rtol: f64, what: &str) {
    let actual = vals(t);
    let close = |a: f64, e: f64| {
        a == e || (a.is_nan() && e.is_nan()) || (e.is_finite() && (a - e).abs() <= rtol * e.abs())
    };
    let ok =
        actual.len() == expected.len() && actual.iter().zip(expected).all(|(&a, &e)| close(a, e));
    assert!(ok, "{what}: {actual:?} vs {expected:?}");
}

/// The gradients of `out` with respect to `inputs` for the upstream gradient `g`.
fn grads(out: &Tensor, inputs: &[&Tensor], g: &Tensor, create_graph: bool) -> Vec<Tensor> {
    let options = GradOptions {
        grad_outputs: vec![Some(g.clone())],
        create_graph,
        ..Default::default()
    };
    autograd::grad(&[out], inputs, options)
        .into_iter()
        .map(|g| g.expect("a gradient"))
        .collect()
}

/// An empty `[4, 0]` tensor whose first stride is so large that walking the first dimension with
/// it overflows. Empty layouts touch no memory, so `as_strided` accepts any strides for them.
fn empty_with_huge_stride(device: Device) -> Tensor {
    homura::zeros_with([0], device).as_strided([4, 0], [isize::MAX as usize, 0], 0)
}

#[test]
fn matmul_with_an_empty_result_allocates_nothing_per_column() {
    // The output and both operands have no elements, but a row of `n` accumulators would not
    // fit in memory.
    let n = isize::MAX as usize / 8 + 1;
    for device in DEVICES {
        let a = homura::zeros_with([0, 0], (DType::I64, device));
        let b = homura::zeros_with([0, n], (DType::I64, device));
        assert_eq!(a.matmul(&b).shape(), [0, n]);
        let a = homura::zeros_with([0, 2, 0], device);
        let b = homura::zeros_with([1, 0, n], device);
        assert_eq!(a.matmul(&b).shape(), [0, 2, n]);
        let input = homura::zeros_with([0, n], device);
        let m = homura::zeros_with([0, 0], device);
        assert_eq!(input.addmm(&m, &b.select(0, 0), 1, 1).shape(), [0, n]);
    }
}

#[test]
fn matmul_with_an_empty_inner_dimension_ignores_operand_strides() {
    for device in DEVICES {
        let a =
            homura::zeros_with([0], device).as_strided([4, 2, 0], [isize::MAX as usize, 0, 0], 0);
        let b = homura::zeros_with([4, 0, 3], device);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), [4, 2, 3]);
        if device == Device::Cpu {
            assert_eq!(c.to_vec::<f32>(), [0.0; 24]);
        }
    }
}

#[test]
fn inclusive_ranges_up_to_i64_max_reach_the_end() {
    for device in DEVICES {
        let x = homura::arange_with(3, device);
        assert_eq!(x.index(..=i64::MAX).shape(), [3]);
        assert_eq!(x.index(0..=i64::MAX).shape(), [3]);
        assert_eq!(x.index(1..=i64::MAX).shape(), [2]);
        assert_eq!(x.index(i64::MAX..=i64::MAX).shape(), [0]);
        assert_eq!(x.index((0..=i64::MAX).step_by(2)).shape(), [2]);
        assert_eq!(x.index(..=-1).shape(), [3]);
        assert_eq!(x.index(..=-2).shape(), [2]);
        assert_eq!(x.index(-3..=-2).shape(), [2]);
    }
    let x = homura::arange(3);
    assert_eq!(x.index(..=i64::MAX).to_vec::<i64>(), [0, 1, 2]);
    assert_eq!(x.index(1..=i64::MAX).to_vec::<i64>(), [1, 2]);
    assert_eq!(x.index((0..=i64::MAX).step_by(2)).to_vec::<i64>(), [0, 2]);
}

#[test]
fn empty_splits_and_chunks_are_distinct_views() {
    for device in DEVICES {
        let x = homura::zeros_with([0, 3], device);
        let pieces = x.chunk(4, 0);
        assert_eq!(pieces.len(), 4);
        for (i, p) in pieces.iter().enumerate() {
            assert_eq!(p.shape(), [0, 3]);
            assert!(!p.is_same(&x));
            for q in &pieces[i + 1..] {
                assert!(!p.is_same(q));
            }
        }
        let split = x.split(0, 0);
        assert_eq!(split.len(), 1);
        assert_eq!(split[0].shape(), [0, 3]);
        assert!(!split[0].is_same(&x));
    }
}

type Op = fn(&Tensor) -> Tensor;

#[test]
fn reductions_and_softmax_of_empty_layouts_ignore_their_strides() {
    for device in DEVICES {
        let x = empty_with_huge_stride(device);
        let plain = homura::zeros_with([4, 0], device);
        let ops: [(&str, Op); 10] = [
            ("sum_dim", |t| t.sum_dim(1, false)),
            ("prod_dim", |t| t.prod_dim(1, false)),
            ("mean_dim", |t| t.mean_dim(1, true)),
            ("var_dim", |t| t.var_dim(1, 1, false)),
            ("std_dim", |t| t.std_dim(1, 0, false)),
            ("all_dim", |t| t.all_dim(1, false)),
            ("softmax", |t| t.softmax(1)),
            ("log_softmax", |t| t.log_softmax(1)),
            ("triu", |t| t.unsqueeze(1).triu(0)),
            ("max_dim over the nonempty dimension", |t| {
                t.max_dim(0, false).0
            }),
        ];
        for (name, op) in ops {
            let (got, expected) = (op(&x), op(&plain));
            assert_eq!(got.shape(), expected.shape(), "{name} on {device}");
            assert_eq!(got.dtype(), expected.dtype(), "{name} on {device}");
            if device == Device::Cpu {
                assert_eq!(bits(&got), bits(&expected), "{name}");
            }
        }
    }
    let x = empty_with_huge_stride(Device::Cpu);
    assert_eq!(x.sum_dim(1, false).to_vec::<f32>(), [0.0; 4]);
    assert_eq!(x.prod_dim(1, false).to_vec::<f32>(), [1.0; 4]);
}

#[test]
fn indexing_empty_layouts_checks_indices_without_computing_slice_offsets() {
    let x = empty_with_huge_stride(Device::Cpu);
    let index = |v: &[i64]| homura::tensor(v.to_vec());
    assert_eq!(x.index_select(0, &index(&[3])).shape(), [1, 0]);
    assert_eq!(x.index_select(0, &index(&[3, 0, 2])).shape(), [3, 0]);
    assert_eq!(x.index_select(0, &index(&[])).shape(), [0, 0]);
    assert_eq!(x.index((&index(&[3]), ..)).shape(), [1, 0]);
    assert_eq!(x.index((&index(&[-1]), ..)).shape(), [1, 0]);
    let grid = homura::tensor([[3i64, -4], [0, 1]]);
    assert_eq!(x.index((&grid, ..)).shape(), [2, 2, 0]);
    assert_eq!(x.index((&grid, ..)).to_vec::<f32>(), [0.0; 0]);
    // Invalid indices are still rejected, even though nothing is copied.
    for bad in [
        || {
            empty_with_huge_stride(Device::Cpu).index_select(0, &homura::tensor([4i64]));
        },
        || {
            empty_with_huge_stride(Device::Cpu).index_select(0, &homura::tensor([-1i64]));
        },
        || {
            empty_with_huge_stride(Device::Cpu).index((&homura::tensor([-5i64]), ..));
        },
    ] {
        assert!(std::panic::catch_unwind(bad).is_err());
    }
    let x = empty_with_huge_stride(Device::Meta);
    assert_eq!(
        x.index_select(0, &index(&[3]).to(Device::Meta)).shape(),
        [1, 0]
    );
    assert_eq!(x.index((&grid.to(Device::Meta), ..)).shape(), [2, 2, 0]);
}

#[test]
fn zero_dimensional_scatter_add_sources_get_zero_dimensional_gradients() {
    // As in PyTorch, whose engine sums the gradient of `src`, shaped like the index, to the
    // zero-dimensional `src`. `hits` is how many times `src` is added.
    let indices: [(&[usize], f32); 3] = [(&[], 1.0), (&[1], 1.0), (&[0], 0.0)];
    for device in DEVICES {
        for self_shape in [&[][..], &[1], &[3]] {
            for (index_shape, hits) in indices {
                for (dim, both) in [(0, false), (0, true), (-1, false), (-1, true)] {
                    let what = format!(
                        "{device:?}, self {self_shape:?}, index {index_shape:?}, dim {dim}, \
                         both {both}"
                    );
                    let x = homura::ones_with(self_shape, device);
                    x.requires_grad_(both);
                    let src = homura::full_with((), 3.0, device);
                    src.requires_grad_(true);
                    let index = homura::zeros_with(index_shape, (DType::I64, device));
                    let y = x.scatter_add(dim, &index, &src);
                    // A differentiable upstream gradient, which the gradient of `src` picks from.
                    let w = homura::full_with(y.shape(), 2.0, device);
                    w.requires_grad_(true);
                    let inputs = if both { vec![&src, &x] } else { vec![&src] };
                    let g = grads(&y, &inputs, &w, true);
                    assert_eq!(g[0].shape().to_vec(), [0usize; 0], "{what}");
                    let options = GradOptions::default();
                    let gw = autograd::grad(&[&g[0]], &[&w], options).remove(0);
                    let gw = gw.expect("a second-order gradient");
                    assert_eq!(gw.shape(), y.shape(), "{what}");
                    if both {
                        assert_eq!(g[1].shape().to_vec(), self_shape, "{what}");
                    }
                    if device == Device::Cpu {
                        assert_eq!(g[0].item::<f32>(), 2.0 * hits, "{what}");
                        let mut picked = vec![0.0; y.num_elements()];
                        picked[0] = hits;
                        assert_eq!(gw.to_vec::<f32>(), picked, "{what}");
                        if both {
                            assert_eq!(g[1].to_vec::<f32>(), vec![2.0; x.num_elements()]);
                        }
                    }
                }
            }
        }
        // A zero-dimensional index with a `src` of one element still fails, as in PyTorch.
        let src = homura::ones_with([1], device);
        src.requires_grad_(true);
        let index = homura::zeros_with((), (DType::I64, device));
        let y = homura::ones_with([3], device).scatter_add(0, &index, &src);
        assert_eq!(
            panic_message(|| y.sum().backward()),
            "Function ScatterAddBackward0 returned an invalid gradient at index 1 - got [] but \
             expected shape compatible with [1]"
        );
    }
}

#[test]
fn meta_as_strided_gradients_enumerate_no_positions() {
    // The host allocations of the backward pass don't grow with the number of elements, and so
    // shapes near the size limit work. A host array of the output's storage positions would
    // take 8 MiB for the first `n`.
    // The overlapping output has `2 * n` elements, whose `I64` positions must fit in memory.
    for n in [1 << 20, isize::MAX as usize / 32] {
        let x = homura::zeros_with([n], Device::Meta);
        x.requires_grad_(true);
        let every_other = x.index((0..).step_by(2));
        let outputs = [
            x.as_strided([n], [1], 0),
            // Output elements that overlap, an offset, and a non-contiguous input.
            x.as_strided([2, n - 1], [1, 1], 0),
            x.as_strided([n - 1], [1], 1),
            every_other.as_strided([n / 4], [4], 2),
        ];
        for (i, y) in outputs.iter().enumerate() {
            let u = homura::zeros_with(y.shape(), Device::Meta);
            u.requires_grad_(true);
            let mut g = None;
            let largest = largest_allocation(|| g = Some(grads(y, &[&x], &u, true).remove(0)));
            assert!(largest < 1 << 16, "n = {n}, output {i}: {largest} bytes");
            let g = g.unwrap();
            assert_eq!(
                (g.shape().to_vec(), g.dtype(), g.device()),
                (vec![n], DType::F32, Device::Meta),
                "output {i}"
            );
            // The gradient is connected to the upstream one.
            let gu = autograd::grad(&[&g.sum()], &[&u], GradOptions::default()).remove(0);
            assert_eq!(gu.expect("a gradient").shape(), y.shape(), "output {i}");
        }
    }
    // An input whose elements overlap is still rejected.
    let row = homura::zeros_with([2], Device::Meta);
    row.requires_grad_(true);
    let y = row.expand([3, 2]).as_strided([2], [1], 0);
    assert_eq!(
        panic_message(|| y.sum().backward()),
        "computing the gradient of as_strided for an input whose elements overlap in memory is \
         not supported yet; deep_clone() the input first"
    );
}

#[test]
fn half_products_and_quotients_keep_a_scalar_in_f32() {
    // As in PyTorch's CPU kernels: a scalar outside the range or precision of `f16` or `bf16` is
    // kept in `f32` for `*` and `/`, while `+`, `-`, `pow` and comparisons round it first.
    let c = FRAC_2_SQRT_PI;
    let inf = f64::INFINITY;
    type Op = (&'static str, fn(&Tensor) -> Tensor);
    let ops: [Op; 14] = [
        ("x * 1e5", |x| x * 1e5),
        ("1e5 * x", |x| 1e5 * x),
        ("x * 1e-8", |x| x * 1e-8),
        ("x * c", |x| x * FRAC_2_SQRT_PI),
        ("x * 100000", |x| x * 100_000i64),
        ("x / c", |x| x / FRAC_2_SQRT_PI),
        ("x / 1e5", |x| x / 1e5),
        ("x / 1e-8", |x| x / 1e-8),
        ("mul_", |x| x.deep_clone().mul_(1e5).clone()),
        ("div_", |x| x.deep_clone().div_(1e-8).clone()),
        ("x - 1e5", |x| x - 1e5),
        ("x + c", |x| x + FRAC_2_SQRT_PI),
        ("x ** c", |x| x.pow(FRAC_2_SQRT_PI)),
        ("x == 1.0001", |x| x.equal(1.0001)),
    ];
    let f16: [[f64; 4]; 14] = [
        [0., inf, inf, inf],
        [0., inf, inf, inf],
        [0., 0.0006551742553710938, 5.960464477539063e-08, 0.],
        [0., inf, 3.384765625, 1.1279296875],
        [0., inf, inf, inf],
        [0., 58048., 2.658203125, 0.88623046875],
        [
            0.,
            0.6552734375,
            2.9981136322021484e-05,
            1.0013580322265625e-05,
        ],
        [0., inf, inf, inf],
        [0., inf, inf, inf],
        [0., inf, inf, inf],
        [-inf, -inf, -inf, -inf],
        [1.1279296875, 65504., 4.12890625, 2.12890625],
        [0., inf, 3.453125, 1.],
        [0., 0., 0., 1.],
    ];
    let bf16: [[f64; 4]; 14] = [
        [0., 6543114240., 299008., 99840.],
        [0., 6543114240., 299008., 99840.],
        [
            0.,
            0.0006561279296875,
            3.003515303134918e-08,
            1.0011717677116394e-08,
        ],
        [0., 73728., 3.390625, 1.125],
        [0., 6543114240., 299008., 99840.],
        [0., 58112., 2.65625, 0.88671875],
        [0., 0.65625, 3.0040740966796875e-05, 1.0013580322265625e-05],
        [0., 6562710028288., 299892736., 100139008.],
        [0., 6543114240., 299008., 99840.],
        [0., 6562710028288., 299892736., 100139008.],
        [-99840., -34304., -99840., -99840.],
        [1.125, 65536., 4.125, 2.125],
        [0., 262144., 3.4375, 1.],
        [0., 0., 0., 1.],
    ];
    for (dtype, expected, quotients) in [
        (DType::F16, f16, [inf, 49984., 24992., inf]),
        (DType::BF16, bf16, [inf, 49920., 24960., 199680.]),
    ] {
        let x = Tensor::from_slice(&[0f64, 65504., 3., 1.], [4]).to(dtype);
        for ((name, op), expected) in ops.iter().zip(expected) {
            let y = op(&x);
            let out_dtype = if name.contains("==") {
                DType::Bool
            } else {
                dtype
            };
            assert_eq!(y.dtype(), out_dtype, "{dtype} {name}");
            assert_eq!(vals(&y), expected, "{dtype} {name}");
            let meta = op(&x.to(Device::Meta));
            assert_eq!((meta.dtype(), meta.shape()), (out_dtype, y.shape()));
        }
        let d = Tensor::from_slice(&[0f64, 2., 4., 0.5], [4]).to(dtype);
        assert_eq!(vals(&(1e5 / &d)), quotients, "{dtype} 1e5 / d");
    }
    // The other floating point dtypes compute in their own type, as before.
    for (dtype, products, quotients) in [
        (
            DType::F32,
            [
                3.3851375579833984,
                7.898654460906982,
                1.1283792206876573e-30,
            ],
            [2.6586806774139404, 6.203588008880615, 8.862269142205257e-31],
        ),
        (
            DType::F64,
            [3.385137501286538, 7.898654169668588, 1.1283791670955127e-30],
            [2.658680776358274, 6.203588478169306, 8.862269254527581e-31],
        ),
    ] {
        let x = Tensor::from_slice(&[3f64, 7., 1e-30], [3]).to(dtype);
        assert_eq!(vals(&(&x * c)), products, "{dtype}");
        assert_eq!(vals(&(&x / c)), quotients, "{dtype}");
    }
}

#[test]
fn half_product_gradients_keep_their_scalar_factor_in_f32() {
    let c = FRAC_2_SQRT_PI;
    let erf = [
        (DType::F16, [0.87890625, 0.70947265625, -0.04132080078125]),
        (DType::BF16, [0.87890625, 0.7109375, -0.041259765625]),
    ];
    for ((dtype, erf), product) in erf.into_iter().zip([3.384765625, 3.390625]) {
        let tensor = |values: &[f64]| {
            let t = Tensor::from_slice(values, [values.len()]).to(dtype);
            t.requires_grad_(true);
            t
        };
        for scalar_first in [false, true] {
            let what = format!("{dtype}, scalar first {scalar_first}");
            let x = tensor(&[3.]);
            let y = if scalar_first { c * &x } else { x.mul(c) };
            let u = tensor(&[3.]);
            let g = grads(&y, &[&x], &u, true).remove(0);
            assert_eq!(vals(&g), [product], "{what}");
            // The second-order gradient multiplies by the same scalar.
            let r = Tensor::from_slice(&[3f64], [1]).to(dtype);
            let gu = grads(
                &(&g * &r).sum(),
                &[&u],
                &homura::ones_with((), dtype),
                false,
            );
            assert_eq!(vals(&gu[0]), [product], "{what}");
        }
        // A factor outside the dtype's range doesn't turn a zero gradient into NaN.
        let x = tensor(&[3.]);
        let g = grads(&(&x * 1e5), &[&x], &homura::zeros_like(&x), false);
        assert_eq!(vals(&g[0]), [0.], "{dtype}");
        // `erf`'s formula multiplies by the constant 2/sqrt(pi).
        let x = tensor(&[0.5, -1.25, 2.]);
        let upstream = Tensor::from_slice(&[1f64, 3., -2.], [3]).to(dtype);
        assert_eq!(vals(&grads(&x.erf(), &[&x], &upstream, false)[0]), erf);
        // On the meta device, only the dtype and shape.
        let x = homura::zeros_with([2], (dtype, Device::Meta));
        x.requires_grad_(true);
        let g = grads(&(c * &x), &[&x], &homura::ones_like(&x), false).remove(0);
        assert_eq!((g.dtype(), g.device()), (dtype, Device::Meta));
    }
}

#[test]
fn gradient_checks_run_in_grad_mode_whatever_the_callers_mode() {
    let x = Tensor::from_slice(&[0.5f64], [1]);
    x.requires_grad_(true);
    let square = |xs: &[Tensor]| vec![&xs[0] * &xs[0]];
    // Right first derivative (x), wrong second one: the factor is a constant to the backward
    // pass.
    let cut = |xs: &[Tensor]| vec![&xs[0] * xs[0].detach()];
    let options = GradcheckOptions::default;
    for enabled in [true, false] {
        let _mode = homura::set_grad_enabled(enabled);
        assert!(gradcheck(square, &[&x], options()).is_ok(), "{enabled}");
        assert!(gradgradcheck(square, &[&x], options()).is_ok(), "{enabled}");
        assert!(gradcheck(cut, &[&x], options()).is_err(), "{enabled}");
        assert!(gradgradcheck(cut, &[&x], options()).is_err(), "{enabled}");
        // The caller's mode is restored after a success, a failure and a panic.
        assert_eq!(homura::is_grad_enabled(), enabled);
        let message = panic_message(|| {
            let _ = gradgradcheck(|_: &[Tensor]| panic!("failed"), &[&x], options());
        });
        assert_eq!(message, "failed");
        assert_eq!(homura::is_grad_enabled(), enabled);
    }
    // A function without differentiable outputs still passes in any mode.
    let _mode = homura::no_grad();
    let constant = |xs: &[Tensor]| vec![homura::zeros_like(&xs[0])];
    assert!(gradgradcheck(constant, &[&x], options()).is_ok());
}

#[test]
fn gradient_checks_never_count_a_finite_derivative_close_to_an_infinite_one() {
    // The outputs, ±1e308, are finite, but their difference overflows: the numerical derivative
    // is infinite and the analytical one, without a graph, zero.
    let x = Tensor::from_slice(&[0.0f64], [1]);
    x.requires_grad_(true);
    let options = GradcheckOptions {
        eps: 1.0,
        ..Default::default()
    };
    let message = gradcheck(|xs| vec![xs[0].detach() * 1e308], &[&x], options)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("is 0 analytically but inf numerically"),
        "{message}"
    );
}

#[test]
fn relu_gradients_have_zero_derivatives_with_respect_to_the_result() {
    // The gradient of relu selects the upstream gradient where the result is positive or NaN.
    // As a function of the result it is constant, so its derivatives with respect to the result
    // and to `x` are zeros, also for infinite or NaN gradients in the second pass; with respect to
    // the upstream gradient it selects again. PyTorch gives the same values in every dtype.
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let tensor = |values: &[f64]| {
            let t = Tensor::from_slice(values, [values.len()]).to(dtype);
            t.requires_grad_(true);
            t
        };
        let x = tensor(&[-1., 0., 1., nan, inf, -inf]);
        let y = x.relu();
        let u = tensor(&[1., 2., 3., 4., 5., 6.]);
        let dx = grads(&y, &[&x], &u, true).remove(0);
        assert!(dx.requires_grad(), "{dtype}");
        assert_values(&dx, &[0., 0., 3., 4., 5., 0.], 0., &format!("{dtype} dx"));
        let gg = Tensor::from_slice(&[inf, nan, 2., inf, nan, -inf], [6]).to(dtype);
        let [gx, gy, gu] = <[Tensor; 3]>::try_from(grads(&dx, &[&x, &y, &u], &gg, false)).unwrap();
        assert_values(&gx, &[0.; 6], 0., &format!("{dtype} d(dx)/dx"));
        assert_values(&gy, &[0.; 6], 0., &format!("{dtype} d(dx)/dy"));
        let selected = [0., 0., 2., inf, nan, 0.];
        assert_values(&gu, &selected, 0., &format!("{dtype} d(dx)/du"));
    }
    // The review's example, whose derivative with respect to the result was [1, 1, 0].
    let x = Tensor::from_slice(&[-1f32, 0., 1.], [3]);
    x.requires_grad_(true);
    let y = x.relu();
    let dx = grads(&y, &[&x], &homura::ones_like(&y), true).remove(0);
    let one = homura::ones_with((), DType::F32);
    assert_values(
        &grads(&dx.sum(), &[&y], &one, false)[0],
        &[0.; 3],
        0.,
        "d(dx)/dy",
    );
}

#[test]
fn half_gelu_gradients_are_computed_in_f32() {
    // PyTorch computes half precision gelu gradients in f32 and rounds them once. Computed step by
    // step in f16, x² overflowed for |x| > 256, where the tanh is saturated, and the gradient of
    // the tanh approximation became 0 * inf = NaN for finite inputs. Infinite and NaN inputs keep
    // their NaN gradients. The reference values come from PyTorch's native kernels; half precision
    // values must be within one rounding step, f32 and f64 ones within 1e-6.
    use GeluApproximate::{None as Exact, Tanh};
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    // Large finite inputs on both sides, zero and non-finite inputs give the same gradients in
    // every dtype.
    let special = [-300., 0., 10., 300., inf, -inf, nan];
    let special_grads = [0., 0.5, 1., 1., nan, nan, nan];
    // Ordinary inputs: f16 and bf16 gradients within one rounding step of PyTorch's, f32 and f64
    // ones within 1e-6 of its f64 values.
    let ordinary = [-10., -1.25, 0.5, 2.];
    let f16_tanh = [0., -0.12249755859375, 0.8671875, 1.0859375];
    let f16_exact = [-0., -0.1226806640625, 0.86767578125, 1.0849609375];
    let bf16_tanh = [0., -0.12255859375, 0.8671875, 1.0859375];
    let bf16_exact = [-7.7093e-22, -0.12255859375, 0.8671875, 1.0859375];
    let f64_tanh = [0., -0.12249263, 0.8673699, 1.0860993];
    let f64_exact = [-7.6945986e-22, -0.12266158, 0.8674951, 1.0852318];
    let cases = [
        (DType::F16, Tanh, f16_tanh),
        (DType::F16, Exact, f16_exact),
        (DType::BF16, Tanh, bf16_tanh),
        (DType::BF16, Exact, bf16_exact),
        (DType::F32, Tanh, f64_tanh),
        (DType::F32, Exact, f64_exact),
        (DType::F64, Tanh, f64_tanh),
        (DType::F64, Exact, f64_exact),
    ];
    let rtol = |dtype| match dtype {
        DType::F16 => 2f64.powi(-10),
        DType::BF16 => 2f64.powi(-7),
        _ => 1e-6,
    };
    let tensor = |values: &[f64], dtype| {
        let t = Tensor::from_slice(values, [values.len()]).to(dtype);
        t.requires_grad_(true);
        t
    };
    for (dtype, approximate, ordinary_grads) in cases {
        for (x, expected) in [
            (&special[..], &special_grads[..]),
            (&ordinary, &ordinary_grads),
        ] {
            let what = format!("{dtype} {approximate:?} at {x:?}");
            let x = tensor(x, dtype);
            let y = x.gelu(approximate);
            let g = grads(&y, &[&x], &homura::ones_like(&y), false).remove(0);
            assert_eq!(g.dtype(), dtype, "{what}");
            assert_values(&g, expected, rtol(dtype), &what);
        }
    }
    // The second derivatives differentiate the f32 computation, so they stay finite at ±300, where
    // PyTorch's own half precision second derivative, computed step by step, is NaN. They are
    // within one rounding step of PyTorch's f32 values, and the derivative with respect to the
    // upstream gradient is the first derivative.
    let second = [
        (Tanh, [0., 0.081181064, 0.61550695, -0.10787512, 0.]),
        (Exact, [0., 0.079908967, 0.61611432, -0.10798194, 0.]),
    ];
    for dtype in [DType::F16, DType::BF16] {
        for (approximate, expected) in second {
            let what = format!("{dtype} {approximate:?}");
            let x = tensor(&[-300., -1.25, 0.5, 2., 300.], dtype);
            let u = tensor(&[1.; 5], dtype);
            let d1 = grads(&x.gelu(approximate), &[&x], &u, true).remove(0);
            assert!(d1.requires_grad(), "{what}");
            let ones = homura::ones_with([5], dtype);
            let [d2x, d2u] = <[Tensor; 2]>::try_from(grads(&d1, &[&x, &u], &ones, false)).unwrap();
            assert_eq!((d2x.dtype(), d2u.dtype()), (dtype, dtype), "{what}");
            assert_values(&d2x, &expected, rtol(dtype), &what);
            assert_eq!(vals(&d2u), vals(&d1), "{what}");
        }
    }
    // The saved input is checked and released like any other, before the conversion to f32.
    let x = tensor(&[0.5, -1.25], DType::F16);
    let y = x.gelu(Tanh).sum();
    {
        let _no_grad = homura::no_grad();
        x.mul_(1);
    }
    assert_eq!(
        panic_message(|| y.backward()),
        "one of the variables needed for gradient computation has been modified by an inplace \
         operation: [f16 [2]] is at version 1; expected version 0 instead."
    );
    let x = tensor(&[0.5, -1.25], DType::BF16);
    let y = x.gelu(Exact).sum();
    y.backward();
    assert!(panic_message(|| y.backward()).starts_with("Trying to backward through the graph"));
    // On the meta device, only the dtype and shape.
    for approximate in [Tanh, Exact] {
        let x = homura::zeros_with([2, 3], (DType::F16, Device::Meta));
        x.requires_grad_(true);
        let y = x.gelu(approximate);
        let g = grads(&y, &[&x], &homura::ones_like(&y), true).remove(0);
        assert_eq!(
            (g.shape().to_vec(), g.dtype(), g.device()),
            (vec![2, 3], DType::F16, Device::Meta)
        );
    }
}

#[test]
fn zero_dimensional_tensors_reject_repeated_dimensions() {
    // Review round 20 (R37): `0` and `-1` both name the one dimension a zero-dimensional tensor
    // accepts, so every list that names it twice is rejected, on every device.
    type Op = (&'static str, fn(&Tensor, &[i64]) -> Tensor);
    let ops: [Op; 8] = [
        ("squeeze_dim", |s, d| s.squeeze_dim(d)),
        ("sum_dim", |s, d| s.sum_dim(d, false)),
        ("mean_dim", |s, d| s.mean_dim(d, true)),
        ("var_dim", |s, d| s.var_dim(d, 0, false)),
        ("std_dim", |s, d| s.std_dim(d, 0, false)),
        ("all_dim", |s, d| s.to(DType::Bool).all_dim(d, false)),
        ("any_dim", |s, d| s.to(DType::Bool).any_dim(d, false)),
        ("flip", |s, d| s.flip(d)),
    ];
    for device in DEVICES {
        let s = Tensor::from_slice(&[2.0f64], []).to(device);
        for (name, op) in ops {
            for dims in [[0, 0], [0, -1], [-1, -1]] {
                assert_eq!(
                    panic_message(|| {
                        op(&s, &dims);
                    }),
                    "dim 0 appears multiple times in the list of dims",
                    "{name} {dims:?} on {device}"
                );
            }
            // Naming it once, or not at all, is fine; any other dimension is out of range.
            for dims in [&[0][..], &[-1], &[]] {
                assert_eq!(op(&s, dims).shape(), [], "{name} {dims:?} on {device}");
            }
            assert_eq!(
                panic_message(|| {
                    op(&s, &[1]);
                }),
                "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
                "{name} on {device}"
            );
        }
    }
    // Tensors with dimensions keep their checks, and a scalar squeeze its gradient.
    let x = Tensor::from_slice(&[1.0f64, 2.0], [2, 1]);
    assert_eq!(
        panic_message(|| {
            x.squeeze_dim([1, -1]);
        }),
        "dim 1 appears multiple times in the list of dims"
    );
    let s = Tensor::from_slice(&[2.0f64], []);
    s.requires_grad_(true);
    let options = GradOptions {
        grad_outputs: vec![Some(Tensor::from_slice(&[3.0f64], []))],
        ..Default::default()
    };
    let g = autograd::grad(&[&s.squeeze_dim([-1])], &[&s], options).remove(0);
    assert_eq!(g.unwrap().to_vec::<f64>(), [3.0]);
}
