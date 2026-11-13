//! The writeback of the scaled products (`addmm`, `addmv`, `baddbmm`, `addbmm`), computed with
//! the product by the CPU kernel since M4 step 5a, and `addr`, computed by one kernel since then,
//! against the composites of separate operations they replaced (`src/ops/composite.rs` before
//! step 5a): the same bits for every dtype, coefficient and broadcast input. Run in debug and
//! release profiles.

use homura::{DType, Scalar, Tensor};

/// Element values with NaN compared by bit pattern.
fn bits(t: &Tensor) -> Vec<u64> {
    t.to(DType::F64)
        .to_vec::<f64>()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Mm,
    Mv,
    Bmm,
    SumBmm,
}

/// `t * s`, skipped when `s` is one.
fn scale(t: Tensor, s: Scalar) -> Tensor {
    if s.to_f64() == 1.0 { t } else { t.mul(s) }
}

/// The composite: the product, then `alpha * p`, then (unless `beta` is zero) `beta * c + ...`,
/// each a separate operation; half precision in f32, rounded at the end.
fn composite(
    op: Op,
    input: &Tensor,
    x: &Tensor,
    y: &Tensor,
    beta: Scalar,
    alpha: Scalar,
) -> Tensor {
    let product = |a: &Tensor, b: &Tensor| match op {
        Op::SumBmm => a.matmul(b).sum_dim(0, false).to(a.dtype()),
        _ => a.matmul(b),
    };
    let scaled_sum = |c: &Tensor, p: Tensor| {
        let p = scale(p, alpha);
        if beta.to_f64() == 0.0 {
            return p;
        }
        scale(c.clone(), beta).add(&p)
    };
    let dtype = x.dtype();
    if matches!(dtype, DType::F16 | DType::BF16) {
        let p = product(&x.to(DType::F32), &y.to(DType::F32));
        scaled_sum(&input.to(DType::F32), p).to(dtype)
    } else {
        scaled_sum(input, product(x, y))
    }
}

fn fused(op: Op, input: &Tensor, x: &Tensor, y: &Tensor, beta: Scalar, alpha: Scalar) -> Tensor {
    match op {
        Op::Mm => input.addmm(x, y, beta, alpha),
        Op::Mv => input.addmv(x, y, beta, alpha),
        Op::Bmm => input.baddbmm(x, y, beta, alpha),
        Op::SumBmm => input.addbmm(x, y, beta, alpha),
    }
}

/// Factors of `op` in `dtype` (`x` transposed when `transposed`, a strided view) and the shapes
/// of the inputs that broadcast to its result.
fn case(op: Op, dtype: DType, transposed: bool) -> (Tensor, Tensor, Vec<Vec<usize>>) {
    let (b, m, k, n) = (3, 4, 5, 6);
    let values = |shape: &[usize]| {
        let t = homura::randn(shape.to_vec()).mul(4.0);
        if dtype.is_floating_point() {
            t.to(dtype)
        } else {
            t.to(DType::I64).to(dtype)
        }
    };
    let left = |shape: &[usize]| {
        if transposed {
            let mut s = shape.to_vec();
            let r = s.len();
            s.swap(r - 1, r - 2);
            values(&s).transpose(-1, -2)
        } else {
            values(shape)
        }
    };
    match op {
        Op::Mm => (
            left(&[m, k]),
            values(&[k, n]),
            vec![vec![m, n], vec![n], vec![1], vec![m, 1], vec![]],
        ),
        Op::Mv => (left(&[m, k]), values(&[k]), vec![vec![m], vec![1], vec![]]),
        Op::Bmm => (
            left(&[b, m, k]),
            values(&[b, k, n]),
            vec![vec![b, m, n], vec![m, n], vec![1, 1, n], vec![b, 1, 1]],
        ),
        Op::SumBmm => (
            left(&[b, m, k]),
            values(&[b, k, n]),
            vec![vec![m, n], vec![n], vec![m, 1], vec![]],
        ),
    }
}

#[test]
fn the_writeback_gives_the_composites_bits() {
    homura::manual_seed(5);
    let floats = [0.0, 1.0, -1.0, 0.5, 2.0, -1.5, 3.0];
    let ints = [0i64, 1, -1, 2, 3];
    let mut checked = 0;
    for dtype in [
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::U8,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
    ] {
        let coefficients: Vec<Scalar> = if dtype.is_floating_point() {
            let mut c: Vec<Scalar> = floats.iter().map(|&v| Scalar::Float(v)).collect();
            if matches!(dtype, DType::F32 | DType::F64) {
                // Rounded to f32 (`F64`: kept) the same way by both.
                c.push(Scalar::Float(0.1));
            }
            c
        } else {
            ints.iter().map(|&v| Scalar::Int(v)).collect()
        };
        for op in [Op::Mm, Op::Mv, Op::Bmm, Op::SumBmm] {
            for transposed in [false, true] {
                let (x, y, inputs) = case(op, dtype, transposed);
                for shape in &inputs {
                    let c = {
                        let t = homura::randn(shape.clone()).mul(4.0);
                        if dtype.is_floating_point() {
                            t.to(dtype)
                        } else {
                            t.to(DType::I64).to(dtype)
                        }
                    };
                    for &alpha in &coefficients {
                        // F32 and F64 with a zero alpha don't compute the product (unchanged).
                        if matches!(dtype, DType::F32 | DType::F64) && alpha.to_f64() == 0.0 {
                            continue;
                        }
                        for &beta in &coefficients {
                            let want = composite(op, &c, &x, &y, beta, alpha);
                            let got = fused(op, &c, &x, &y, beta, alpha);
                            let what =
                                format!("{op:?} {dtype} {transposed} {shape:?} {alpha} {beta}");
                            assert_eq!(got.shape(), want.shape(), "{what}");
                            assert_eq!(got.dtype(), dtype, "{what}");
                            assert_eq!(bits(&got), bits(&want), "{what}");
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 5000, "{checked}");
}

#[test]
fn the_writeback_keeps_the_special_cases() {
    let f32s = |v: &[f32], shape: &[usize]| Tensor::from_slice(v, shape.to_vec());
    // A beta of zero ignores NaN in the input, also in half precision.
    for dtype in [DType::F32, DType::F16] {
        let input = f32s(&[f32::NAN], &[1, 1]).to(dtype);
        let m = f32s(&[2.0], &[1, 1]).to(dtype);
        assert_eq!(
            input.addmm(&m, &m, 0, 1).to(DType::F32).to_vec::<f32>(),
            [4.0]
        );
    }
    // Half precision computes the product with a zero alpha: 0 * Inf is NaN.
    let inf = f32s(&[f32::INFINITY], &[1, 1]).to(DType::F16);
    let one = f32s(&[1.0], &[1, 1]).to(DType::F16);
    assert!(
        one.addmm(&inf, &one, 1, 0)
            .to(DType::F32)
            .item::<f32>()
            .is_nan()
    );
    // An empty inner dimension: the product is an exact zero, and Inf * 0 is NaN.
    let three = f32s(&[3.0], &[1, 1]);
    let empty = |shape: &[usize]| homura::zeros(shape.to_vec());
    let out = three.addmm(&empty(&[1, 0]), &empty(&[0, 1]), 1, f64::INFINITY);
    assert!(out.item::<f32>().is_nan());
    // addbmm adds the batches' rounded products in f64: [2^24, 1, -2^24] gives 1.
    let two24 = (1u32 << 24) as f32;
    let b1 = f32s(&[two24, 1.0, -two24], &[3, 1, 1]);
    let b2 = f32s(&[1.0, 1.0, 1.0], &[3, 1, 1]);
    let zero = f32s(&[0.0], &[1, 1]);
    assert_eq!(zero.addbmm(&b1, &b2, 1, 1).to_vec::<f32>(), [1.0]);
    // No batches: zeros, scaled.
    let none = homura::zeros([0, 1, 1]);
    assert_eq!(three.addbmm(&none, &none, 1, 1).to_vec::<f32>(), [3.0]);
    // Integers wrap in i64: [2^53, 1, -2^53] gives 1 and [i64::MAX, 1] wraps to i64::MIN.
    let i64s = |v: &[i64]| Tensor::from_slice(v, [v.len(), 1, 1]);
    let zero = Tensor::from_slice(&[0i64], [1, 1]);
    let ones = |n: usize| i64s(&vec![1; n]);
    let p53 = 1i64 << 53;
    assert_eq!(
        zero.addbmm(&i64s(&[p53, 1, -p53]), &ones(3), 1, 1)
            .to_vec::<i64>(),
        [1]
    );
    assert_eq!(
        zero.addbmm(&i64s(&[i64::MAX, 1]), &ones(2), 1, 1)
            .to_vec::<i64>(),
        [i64::MIN]
    );
}

/// Element bits of any dtype: floats through f64 (exact), with every NaN as one pattern (where
/// two NaNs meet, which one an operation returns isn't specified: the optimizer may swap the
/// operands of a product), the others as i64.
fn any_bits(t: &Tensor) -> Vec<u64> {
    if t.dtype().is_floating_point() {
        bits(t)
            .into_iter()
            .map(|b| {
                if f64::from_bits(b).is_nan() {
                    f64::NAN.to_bits()
                } else {
                    b
                }
            })
            .collect()
    } else {
        t.to(DType::I64)
            .to_vec::<i64>()
            .iter()
            .map(|&v| v as u64)
            .collect()
    }
}

/// `addr`'s composite before step 5a: `alpha` scales `vec1` (floats) or the outer product
/// (integers, bools), then `beta * input` is added unless `beta` is zero. The coefficients are
/// values of the dtype.
fn addr_composite(input: &Tensor, v1: &Tensor, v2: &Tensor, beta: Scalar, alpha: Scalar) -> Tensor {
    let outer = |a: &Tensor, b: &Tensor| a.unsqueeze(1).mul(b.unsqueeze(0));
    let product = if v1.dtype().is_floating_point() {
        outer(&scale(v1.clone(), alpha), v2)
    } else {
        scale(outer(v1, v2), alpha)
    };
    if beta.to_f64() == 0.0 {
        return product;
    }
    scale(input.clone(), beta).add(&product)
}

/// `n` values of `dtype`: special values first (NaN, infinities, signed zeros, the extremes),
/// then random ones.
fn addr_values(dtype: DType, n: usize) -> Tensor {
    if dtype == DType::Bool {
        return homura::randn([n]).greater(0.0);
    }
    if !dtype.is_floating_point() {
        let special = Tensor::from_slice(&[0i64, 1, -1, i64::MAX, i64::MIN, 255, -128], [7]);
        let random = homura::randn([n]).mul(1e6).to(DType::I64);
        return homura::concat(&[special, random], 0)
            .narrow(0, 0, n)
            .to(dtype);
    }
    let special = Tensor::from_slice(
        &[
            f64::NAN,
            -f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1e-40,
            -5.96e-8,
            65504.0,
            3.0e38,
            -1.0e300,
        ],
        [11],
    );
    let random = homura::randn([n]).mul(8.0).to(DType::F64);
    homura::concat(&[special, random], 0)
        .narrow(0, 0, n)
        .to(dtype)
}

#[test]
fn addr_gives_the_composites_bits() {
    homura::manual_seed(7);
    let (n, m) = (13, 12);
    let mut checked = 0;
    for dtype in [
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::U8,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::Bool,
    ] {
        // Coefficients of the dtype (rounded to it first, as the operation does).
        let coefficients: Vec<Scalar> = if dtype == DType::Bool {
            vec![Scalar::Bool(false), Scalar::Bool(true)]
        } else if dtype.is_floating_point() {
            [
                0.0,
                1.0,
                -1.0,
                0.5,
                2.0,
                -1.5,
                0.1,
                1e-30,
                f64::INFINITY,
                f64::NAN,
            ]
            .iter()
            .map(|&v| {
                let r = Tensor::from_slice(&[v], [1]).to(dtype).to(DType::F64);
                Scalar::Float(r.to_vec::<f64>()[0])
            })
            .collect()
        } else {
            [0i64, 1, -1, 2, 3, 127]
                .iter()
                .map(|&v| Scalar::Int(v))
                .collect()
        };
        // Strided factors: every other element, and a reversed-then-contiguous copy.
        let v1 = addr_values(dtype, 2 * n).reshape([n, 2]).select(1, 1);
        let v2 = addr_values(dtype, m).flip([0]);
        let full = addr_values(dtype, n * m);
        let inputs = [
            full.reshape([n, m]),
            full.reshape([m, n]).transpose(0, 1),
            full.narrow(0, 0, m),
            full.narrow(0, 0, n).reshape([n, 1]),
            full.narrow(0, 0, 1),
            full.narrow(0, 0, 1).reshape(()),
        ];
        for input in &inputs {
            for &alpha in &coefficients {
                for &beta in &coefficients {
                    let want = addr_composite(input, &v1, &v2, beta, alpha);
                    let got = input.addr(&v1, &v2, beta, alpha);
                    let what = format!("{dtype} {:?} {alpha} {beta}", input.shape());
                    assert_eq!(got.shape(), [n, m], "{what}");
                    assert_eq!(got.dtype(), dtype, "{what}");
                    assert_eq!(any_bits(&got), any_bits(&want), "{what}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 2000, "{checked}");
}
