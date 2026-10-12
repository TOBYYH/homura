//! Regression checks for the issues found while reviewing M1. `tools/golden/check_m1_formulas.py`
//! checks the PyTorch side of the gradient and half precision cases. Run in debug and release
//! profiles.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::FRAC_2_SQRT_PI;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};

use homura::autograd::{self, GradOptions};
use homura::{DType, Device, Tensor};

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
