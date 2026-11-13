//! Matrix products on a ROCm GPU against the CPU (M4 step 5a): `matmul` and the scaled products
//! (`addmm`, `addmv`, `baddbmm`, `addbmm`), with the factors in every layout (row-major,
//! transposed, strided, broadcast batches) and shapes that cross the tiles of the paths. Integer
//! results have the CPU's bits; float results are within M4.12's classes (GEMM for products,
//! scaled products for the writeback); products that are exact on both have the same bits.
//! `addr`, an elementwise kernel, has the CPU's bits in every dtype (a NaN only has to be a NaN).
#![cfg(feature = "rocm")]

use std::panic::{AssertUnwindSafe, catch_unwind};

#[path = "common/rocm.rs"]
mod rocm;

use homura::{DType, Device, Scalar, Tensor};
use rocm::{GPU, assert_bitwise, assert_same, floats, ulp, unit_roundoff, values};

const INTS: [DType; 5] = [DType::U8, DType::I8, DType::I16, DType::I32, DType::I64];

/// (M, K, N) of the products: 1, odd sizes, sizes around the generic path's 64 x 64 tiles and
/// its 16 steps of k, and an empty dimension.
const SHAPES: [[usize; 3]; 10] = [
    [1, 1, 1],
    [1, 7, 1],
    [3, 5, 7],
    [16, 16, 16],
    [63, 17, 65],
    [64, 64, 64],
    [65, 33, 129],
    [130, 1, 3],
    [2, 0, 3],
    [0, 4, 5],
];

/// The factors `[B, M, K] @ [B, K, N]` on the CPU in a layout, with the same views on the GPU.
/// `values` makes the elements.
fn factors(
    shape: [usize; 3],
    batch: usize,
    layout: &str,
    values: &dyn Fn(usize, u64) -> Tensor,
) -> [(Tensor, Tensor); 2] {
    let [m, k, n] = shape;
    let make = |rows: usize, cols: usize, seed: u64, transposed: bool| {
        let base = values(batch * rows * cols * 2 + 3, seed);
        let view = |t: &Tensor| match layout {
            // Stored transposed: a unit stride along the rows.
            _ if transposed => t
                .narrow(0, 0, batch * rows * cols)
                .reshape([batch, cols, rows])
                .transpose(1, 2),
            // Every other element, from an offset: no unit stride.
            "strided" => t
                .narrow(0, 3, batch * rows * cols * 2)
                .reshape([batch, rows, cols, 2])
                .select(3, 1),
            _ => t
                .narrow(0, 1, batch * rows * cols)
                .reshape([batch, rows, cols]),
        };
        (view(&base), view(&base.to(GPU)))
    };
    let (ta, tb) = match layout {
        "NT" => (false, true),
        "TN" => (true, false),
        "TT" => (true, true),
        _ => (false, false),
    };
    [make(m, k, 11, ta), make(k, n, 12, tb)]
}

const LAYOUTS: [&str; 5] = ["NN", "NT", "TN", "TT", "strided"];

/// The entries, on 3-D factors: the result for (input, x, y, beta, alpha). `addmm` and `addmv`
/// take the first batch.
type Entry = (
    &'static str,
    fn(&Tensor, &Tensor, &Tensor, Scalar, Scalar) -> Tensor,
);

const ENTRIES: [Entry; 5] = [
    ("matmul", |_, x, y, _, _| x.matmul(y)),
    ("addmm", |c, x, y, b, a| {
        c.addmm(&x.select(0, 0), &y.select(0, 0), b, a)
    }),
    ("addmv", |c, x, y, b, a| {
        c.addmv(&x.select(0, 0), &y.select(0, 0).select(1, 0), b, a)
    }),
    ("baddbmm", |c, x, y, b, a| c.baddbmm(x, y, b, a)),
    ("addbmm", |c, x, y, b, a| c.addbmm(x, y, b, a)),
];

/// The input of an entry, broadcast from a smaller shape where the result allows it.
fn input_shape(entry: &str, batch: usize, m: usize, n: usize) -> Vec<usize> {
    match entry {
        "addmv" => vec![m],
        "baddbmm" => vec![batch, 1, n],
        _ => vec![n],
    }
}

#[test]
fn integer_products_have_the_cpus_bits() {
    let coefficients = [(1, 1), (0, 1), (1, 0), (-1, 3), (2, -1)];
    for dtype in INTS {
        let v = |n: usize, seed: u64| values(dtype, n, seed);
        for shape in SHAPES {
            let [m, _, n] = shape;
            for batch in [1, 3] {
                for layout in LAYOUTS {
                    let [(xc, xg), (yc, yg)] = factors(shape, batch, layout, &v);
                    for (entry, f) in ENTRIES {
                        if entry == "addmv" && n == 0 {
                            continue;
                        }
                        let input = values(dtype, n.max(m) * batch + 1, 13);
                        let shape_c = input_shape(entry, batch, m, n);
                        let count: usize = shape_c.iter().product();
                        let cc = input.narrow(0, 0, count).reshape(shape_c.clone());
                        let cg = cc.to(GPU);
                        for (beta, alpha) in coefficients {
                            let what = format!(
                                "{entry} {dtype} {shape:?} batch {batch} {layout} beta {beta} alpha {alpha}"
                            );
                            let (b, a) = (Scalar::Int(beta), Scalar::Int(alpha));
                            assert_same(&f(&cg, &xg, &yg, b, a), &f(&cc, &xc, &yc, b, a), &what);
                        }
                    }
                }
            }
        }
    }
}

/// Moderate finite values of a float dtype (normal, of both signs, no overflow in a product).
fn moderate(dtype: DType) -> impl Fn(usize, u64) -> Tensor {
    move |n, seed| {
        homura::manual_seed(seed);
        homura::randn([n]).mul(3.0).to(dtype)
    }
}

/// `|gpu - cpu| <= c * K * eps * sum_k |a_ik * b_kj| + u_out(max(|gpu|, |cpu|))` (M4.12, GEMM),
/// with `abs` the CPU's `|a| @ |b|` in f64.
#[track_caller]
fn check_gemm(gpu: &Tensor, cpu: &Tensor, abs: &Tensor, k: usize, what: &str) {
    assert_eq!(gpu.device(), GPU, "{what}");
    assert_eq!(gpu.shape(), cpu.shape(), "{what}");
    assert_eq!(gpu.dtype(), cpu.dtype(), "{what}");
    let dtype = cpu.dtype();
    let eps = unit_roundoff(dtype);
    let (g, c, a) = (floats(gpu), floats(cpu), floats(abs));
    for i in 0..g.len() {
        let bound = 2.0 * k as f64 * eps * a[i] + ulp(dtype, g[i].abs().max(c[i].abs()));
        assert!(
            (g[i] - c[i]).abs() <= bound,
            "{what}: element {i}: gpu {} cpu {} bound {bound}",
            g[i],
            c[i]
        );
    }
}

#[test]
fn float_products_are_within_the_gemm_class() {
    for dtype in [DType::F32, DType::F64] {
        let v = moderate(dtype);
        for shape in SHAPES {
            for batch in [1, 3] {
                for layout in LAYOUTS {
                    let [(xc, xg), (yc, yg)] = factors(shape, batch, layout, &v);
                    let what = format!("matmul {dtype} {shape:?} batch {batch} {layout}");
                    let abs = xc.abs().to(DType::F64).matmul(&yc.abs().to(DType::F64));
                    check_gemm(&xg.matmul(&yg), &xc.matmul(&yc), &abs, shape[1], &what);
                }
            }
        }
    }
}

/// The scaled products (M4.12, scaled products): the product within the GEMM class `E_P`, then
/// one rounding of `alpha * P` and one of the sum in the computation type, and one of the output.
/// `p_abs` is `sum_k |a_ik * b_kj|` (summed over the batches for `addbmm`), `p` the CPU's product
/// (in f64), `batches` the number of batches summed (1 otherwise).
#[allow(clippy::too_many_arguments)]
#[track_caller]
fn check_scaled(
    gpu: &Tensor,
    cpu: &Tensor,
    p_abs: &Tensor,
    p: &Tensor,
    k: usize,
    batches: usize,
    (beta, alpha): (f64, f64),
    what: &str,
) {
    assert_eq!(gpu.shape(), cpu.shape(), "{what}");
    let dtype = cpu.dtype();
    let eps = unit_roundoff(dtype);
    let (g, c, pa, pv) = (floats(gpu), floats(cpu), floats(p_abs), floats(p));
    for i in 0..g.len() {
        let mut e_p = 2.0 * k as f64 * eps * pa[i] + batches as f64 * ulp(dtype, pv[i]);
        if batches > 1 {
            e_p += ulp(dtype, pv[i]) + batches as f64 * 2f64.powi(-53) * pa[i];
        }
        let mut bound = alpha.abs() * e_p + ulp(dtype, g[i].abs().max(c[i].abs()));
        if alpha != 1.0 {
            bound += ulp(dtype, (alpha * pv[i]).abs());
        }
        if beta != 0.0 {
            bound += ulp(dtype, c[i].abs().max(g[i].abs()));
        }
        assert!(
            (g[i] - c[i]).abs() <= bound,
            "{what}: element {i}: gpu {} cpu {} bound {bound}",
            g[i],
            c[i]
        );
    }
}

#[test]
fn float_scaled_products_are_within_their_class() {
    let coefficients = [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (-0.5, 3.0), (2.0, 0.1)];
    for dtype in [DType::F32, DType::F64] {
        let v = moderate(dtype);
        for shape in SHAPES {
            let [m, k, n] = shape;
            for batch in [1, 3] {
                for layout in LAYOUTS {
                    let [(xc, xg), (yc, yg)] = factors(shape, batch, layout, &v);
                    let abs3 = xc.abs().to(DType::F64).matmul(&yc.abs().to(DType::F64));
                    let p3 = xc.matmul(&yc).to(DType::F64);
                    for (entry, f) in &ENTRIES[1..] {
                        if *entry == "addmv" && n == 0 {
                            continue;
                        }
                        let shape_c = input_shape(entry, batch, m, n);
                        let cc = v(shape_c.iter().product(), 14).reshape(shape_c.clone());
                        let cg = cc.to(GPU);
                        let (pa, p, batches) = match *entry {
                            "addmm" => (abs3.select(0, 0), p3.select(0, 0), 1),
                            "addmv" => (
                                abs3.select(0, 0).select(1, 0),
                                p3.select(0, 0).select(1, 0),
                                1,
                            ),
                            "baddbmm" => (abs3.clone(), p3.clone(), 1),
                            _ => (abs3.sum_dim(0, false), p3.sum_dim(0, false), batch),
                        };
                        for (beta, alpha) in coefficients {
                            let what = format!(
                                "{entry} {dtype} {shape:?} batch {batch} {layout} beta {beta} alpha {alpha}"
                            );
                            let (b, a) = (Scalar::Float(beta), Scalar::Float(alpha));
                            let gpu = f(&cg, &xg, &yg, b, a);
                            let cpu = f(&cc, &xc, &yc, b, a);
                            check_scaled(&gpu, &cpu, &pa, &p, k, batches, (beta, alpha), &what);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn exact_products_have_the_cpus_bits() {
    // Small integers: every product and sum is exact in f32 and f64, whatever the order.
    for dtype in [DType::F32, DType::F64] {
        let small = |n: usize, seed: u64| {
            homura::manual_seed(seed);
            homura::randint(-8, 9, [n]).to(dtype)
        };
        for shape in SHAPES {
            for layout in LAYOUTS {
                let [(xc, xg), (yc, yg)] = factors(shape, 3, layout, &small);
                let what = format!("{dtype} {shape:?} {layout}");
                assert_same(&xg.matmul(&yg), &xc.matmul(&yc), &what);
                let [m, _, n] = shape;
                let cc = small(m * n, 15).reshape([m, n]);
                let cg = cc.to(GPU);
                assert_same(
                    &cg.addbmm(&xg, &yg, 2, -3),
                    &cc.addbmm(&xc, &yc, 2, -3),
                    &what,
                );
            }
        }
    }
    let f32s = |v: &[f32], shape: &[usize]| Tensor::from_slice(v, shape.to_vec());
    let both = |t: Tensor| (t.to(GPU), t);
    // addbmm rounds each batch's product, then adds them in f64: [2^24, 1, -2^24] gives 1.
    let two24 = (1u32 << 24) as f32;
    let (b1g, b1) = both(f32s(&[two24, 1.0, -two24], &[3, 1, 1]));
    let (b2g, b2) = both(f32s(&[1.0; 3], &[3, 1, 1]));
    let (zg, z) = both(f32s(&[0.0], &[1, 1]));
    let out = zg.addbmm(&b1g, &b2g, 1, 1);
    assert_same(&out, &z.addbmm(&b1, &b2, 1, 1), "addbmm cancellation");
    assert_eq!(out.to(Device::Cpu).to_vec::<f32>(), [1.0]);
    // A beta of zero ignores NaN in the input; a zero alpha (F32) doesn't read the factors.
    let (ng, n) = both(f32s(&[f32::NAN], &[1, 1]));
    let (ig, i) = both(f32s(&[f32::INFINITY], &[1, 1]));
    let (tg, t) = both(f32s(&[3.0], &[1, 1]));
    assert_same(&ng.addmm(&tg, &tg, 0, 1), &n.addmm(&t, &t, 0, 1), "beta 0");
    assert_same(&tg.addmm(&ig, &ig, 1, 0), &t.addmm(&i, &i, 1, 0), "alpha 0");
    // An empty inner dimension: the product is an exact zero, and Inf * 0 is NaN.
    let (eg, e) = both(homura::zeros([1, 0]));
    let (fg, f) = both(homura::zeros([0, 1]));
    let out = tg.addmm(&eg, &fg, 1, f64::INFINITY);
    assert_same(&out, &t.addmm(&e, &f, 1, f64::INFINITY), "k = 0");
    // No batches to add.
    let (ng, n) = both(homura::zeros([0, 1, 1]));
    assert_same(
        &tg.addbmm(&ng, &ng, 1, 1),
        &t.addbmm(&n, &n, 1, 1),
        "no batches",
    );
    // Integers wrap in i64: [2^53, 1, -2^53] gives 1, [i64::MAX, 1] wraps to i64::MIN.
    let i64s = |v: &[i64]| Tensor::from_slice(v, [v.len(), 1, 1]);
    let zero = Tensor::from_slice(&[0i64], [1, 1]);
    let p53 = 1i64 << 53;
    for v in [vec![p53, 1, -p53], vec![i64::MAX, 1]] {
        let (xg, x) = both(i64s(&v));
        let (og, o) = both(i64s(&vec![1; v.len()]));
        let (zg, z) = both(zero.clone());
        assert_same(
            &zg.addbmm(&xg, &og, 1, 1),
            &z.addbmm(&x, &o, 1, 1),
            "i64 addbmm",
        );
    }
}

#[test]
fn launches_are_split_at_the_grid_limits() {
    // More than 65535 row tiles of 64 (4,194,240 rows), and more than 65535 batches.
    let v = |n: usize, seed: u64| values(DType::I32, n, seed);
    let rows = 65536 * 64 + 5;
    let x = v(rows, 21).reshape([rows, 1]);
    let y = v(3, 22).reshape([1, 3]);
    assert_same(&x.to(GPU).matmul(&y.to(GPU)), &x.matmul(&y), "rows");
    let batches = 65536 + 7;
    let x = v(batches * 2, 23).reshape([batches, 1, 2]);
    let y = v(batches * 2, 24).reshape([batches, 2, 1]);
    assert_same(&x.to(GPU).matmul(&y.to(GPU)), &x.matmul(&y), "batches");
    let c = v(batches, 25).reshape([batches, 1, 1]);
    assert_same(
        &c.to(GPU).baddbmm(&x.to(GPU), &y.to(GPU), 2, 3),
        &c.baddbmm(&x, &y, 2, 3),
        "scaled batches",
    );
}

#[test]
fn half_and_bool_products_are_rejected() {
    let message = |f: &dyn Fn()| {
        let e = catch_unwind(AssertUnwindSafe(f)).expect_err("panics");
        e.downcast_ref::<String>().cloned().unwrap_or_default()
    };
    for dtype in [DType::F16, DType::BF16] {
        let x = homura::ones([2, 2]).to((dtype, GPU));
        assert_eq!(
            message(&|| drop(x.matmul(&x))),
            "gemm is not implemented on ROCm yet (rocm:0)"
        );
    }
    let x = Tensor::from_slice(&[true, false], [1, 2]);
    let y = x.reshape([2, 1]);
    let cpu = message(&|| drop(x.matmul(&y)));
    assert_eq!(message(&|| drop(x.to(GPU).matmul(&y.to(GPU)))), cpu);
}

/// (batch dimensions, M, K, N) of products whose right factor is shared by the batches.
const SHARED: [(&[usize], usize, usize, usize); 6] = [
    // Folded: four rows of a long k (a narrow path), and of a short one (a tile path).
    (&[4], 1, 256, 300),
    (&[4], 1, 40, 300),
    (&[3], 5, 40, 70),
    (&[8], 16, 33, 65),
    (&[2, 3], 7, 19, 130),
    (&[2], 130, 17, 9),
];

/// Products whose batches share the right factor (a weight broadcast over the batches of the
/// input, as in `linear` without a bias on 3-D input) are folded by the library into one batch
/// of more rows: the CPU's results (integers bit for bit, floats within the GEMM class), and the
/// same where the left factor's rows don't continue from batch to batch (stored with the batch
/// inside, or broadcast) and nothing folds. The scaled products with a shared `batch2` fold with
/// a whole input or a bias row, not with one `[m, n]` matrix for every batch.
#[test]
fn products_with_a_shared_factor_match_the_cpu() {
    for dtype in [DType::F32, DType::F64, DType::I32, DType::U8] {
        let float = dtype.is_floating_point();
        let v = |n: usize, seed: u64| {
            if float {
                moderate(dtype)(n, seed)
            } else {
                values(dtype, n, seed)
            }
        };
        for (batch, m, k, n) in SHARED {
            let count: usize = batch.iter().product();
            let x_shape: Vec<usize> = batch.iter().copied().chain([m, k]).collect();
            let xc = v(count * m * k, 21).reshape(x_shape.clone());
            let x2 = v(m * k, 22).reshape([m, k]);
            let yc = v(k * n, 23).reshape([k, n]);
            let nd = batch.len() as i64;
            let front: Vec<i64> = [nd].into_iter().chain(0..nd).chain([nd + 1]).collect();
            let back: Vec<i64> = (1..=nd).chain([0, nd + 1]).collect();
            // The left factor as it is, with the batch inside each row ([m, batch..., k] permuted
            // back), and one [m, k] broadcast to every batch; made on each device, since a copy
            // to the GPU is contiguous.
            let views = |x: &Tensor, x2: &Tensor| {
                [
                    ("contiguous", x.clone()),
                    (
                        "batch inside",
                        x.permute(front.clone()).contiguous().permute(back.clone()),
                    ),
                    ("broadcast", x2.expand(x_shape.clone())),
                ]
            };
            let on_gpu = views(&xc.to(GPU), &x2.to(GPU));
            for ((name, x), (_, xg)) in views(&xc, &x2).into_iter().zip(on_gpu) {
                let what = format!("{dtype} {batch:?} x {m}x{k} @ {k}x{n}, x {name}");
                let (gpu, cpu) = (xg.matmul(&yc.to(GPU)), x.matmul(&yc));
                if float {
                    let abs = x.abs().to(DType::F64).matmul(&yc.abs().to(DType::F64));
                    check_gemm(&gpu, &cpu, &abs, k, &what);
                } else {
                    assert_same(&gpu, &cpu, &what);
                }
            }
            if batch.len() > 1 {
                continue;
            }
            // baddbmm with batch2 broadcast: the input whole, a bias row, one matrix for all.
            let b = batch[0];
            let y3 = yc.unsqueeze(0).expand([b, k, n]);
            let y3g = yc.to(GPU).unsqueeze(0).expand([b, k, n]);
            let inputs = [vec![b, m, n], vec![n], vec![m, n]];
            for shape_c in inputs {
                let cc = v(shape_c.iter().product(), 24).reshape(shape_c.clone());
                for (beta, alpha) in [(1.0, 1.0), (0.0, 2.0), (-0.5, 3.0)] {
                    let what = format!(
                        "baddbmm {dtype} [{b}, {m}, {k}] @ shared [{k}, {n}], input {shape_c:?}, beta {beta} alpha {alpha}"
                    );
                    let (bs, al) = if float {
                        (Scalar::Float(beta), Scalar::Float(alpha))
                    } else {
                        (Scalar::Int(beta as i64), Scalar::Int(alpha as i64))
                    };
                    let gpu = cc.to(GPU).baddbmm(&xc.to(GPU), &y3g, bs, al);
                    let cpu = cc.baddbmm(&xc, &y3, bs, al);
                    if float {
                        let pa = xc.abs().to(DType::F64).matmul(&yc.abs().to(DType::F64));
                        let p = xc.matmul(&yc).to(DType::F64);
                        check_scaled(&gpu, &cpu, &pa, &p, k, 1, (beta, alpha), &what);
                    } else {
                        assert_same(&gpu, &cpu, &what);
                    }
                }
            }
        }
    }
}

/// `linear` without a bias on 3-D input (`input.matmul(weight.t())`, folded) and its gradients:
/// within the GEMM class of the CPU's.
#[test]
fn gradients_of_a_linear_without_bias_are_within_the_gemm_class() {
    let v = moderate(DType::F32);
    let abs = |a: &Tensor, b: &Tensor| a.abs().to(DType::F64).matmul(&b.abs().to(DType::F64));
    for (b, t, d, e) in [(4, 1, 300, 70), (3, 9, 40, 130)] {
        let (x, w) = (
            v(b * t * d, 51).reshape([b, t, d]),
            v(e * d, 52).reshape([e, d]),
        );
        let g = v(b * t * e, 53).reshape([b, t, e]);
        let f = |a: &[Tensor]| homura::nn::functional::linear(&a[0], &a[1], None);
        let (yc, gc) = gradients(&f, &[x.clone(), w.clone()], &g, Device::Cpu);
        let (yg, gg) = gradients(&f, &[x.clone(), w.clone()], &g, GPU);
        let what = format!("linear [{b}, {t}, {d}] -> {e}");
        check_gemm(&yg.to(GPU), &yc, &abs(&x, &w.transpose(0, 1)), d, &what);
        check_gemm(
            &gg[0].to(GPU),
            &gc[0],
            &abs(&g, &w),
            e,
            &format!("{what} dx"),
        );
        let dw_abs = abs(&g.flatten(0, 1).transpose(0, 1), &x.flatten(0, 1));
        check_gemm(
            &gg[1].to(GPU),
            &gc[1],
            &dw_abs,
            b * t,
            &format!("{what} dw"),
        );
    }
}

/// The products of the logging test: two of one description (logged once), one of another, and
/// one folded by the library.
fn logged_products() -> Vec<Tensor> {
    let v = moderate(DType::F32);
    let x = v(65 * 33, 31).reshape([65, 33]).to(GPU);
    let y = v(33 * 70, 32).reshape([33, 70]).to(GPU);
    let c = v(70, 33).to(GPU);
    // Four batches of one row that share y: logged as one batch of four rows.
    let s = v(4 * 33, 34).reshape([4, 1, 33]).to(GPU);
    vec![
        x.matmul(&y),
        x.matmul(&y),
        c.addmm(&x, &y, 0.5, 2.0),
        s.matmul(&y),
    ]
}

/// With `HOMURA_ROCM_GEMM_LOG=1` the choice is printed once per description; a second process
/// computes the same bits (the choice depends on the description only).
#[test]
fn the_choice_is_logged_once_and_the_same_in_another_process() {
    const NAME: &str = "the_choice_is_logged_once_and_the_same_in_another_process";
    if std::env::var_os("HOMURA_ROCM_GEMM_LOG").is_some() {
        for t in logged_products() {
            println!("bits {:?}", rocm::bits(&t));
        }
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", NAME, "--nocapture", "--test-threads=1"])
        .env("HOMURA_ROCM_GEMM_LOG", "1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let logged: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("homura: gemm"))
        .collect();
    assert_eq!(
        logged,
        [
            "homura: gemm f32 NN batch=1 m=65 n=70 k=33 sum_batches=0 scaled=0 -> s08a_s7a5_unal",
            "homura: gemm f32 NN batch=1 m=65 n=70 k=33 sum_batches=0 scaled=1 -> s08a_s7a5_unal",
            "homura: gemm f32 NN batch=1 m=4 n=70 k=33 sum_batches=0 scaled=0 -> s08a_s7a5_unal",
        ]
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The harness prints the test's name on the line of the first output, without a newline.
    let theirs: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.find("bits [").map(|i| &l[i..]))
        .collect();
    let ours: Vec<String> = logged_products()
        .iter()
        .map(|t| format!("bits {:?}", rocm::bits(t)))
        .collect();
    assert_eq!(theirs, ours);
}

/// The gradients of `f` at the inputs on `device`, for the output gradient `g`, on the CPU.
fn gradients(
    f: &dyn Fn(&[Tensor]) -> Tensor,
    inputs: &[Tensor],
    g: &Tensor,
    device: Device,
) -> (Tensor, Vec<Tensor>) {
    let xs: Vec<Tensor> = inputs.iter().map(|t| t.to(device).detach()).collect();
    for x in &xs {
        x.requires_grad_(true);
    }
    let y = f(&xs);
    let grads = homura::autograd::grad(
        &[&y],
        &xs.iter().collect::<Vec<_>>(),
        homura::autograd::GradOptions {
            grad_outputs: vec![Some(g.to(device))],
            ..Default::default()
        },
    );
    (
        y.detach().to(Device::Cpu),
        grads
            .into_iter()
            .map(|t| t.expect("a gradient").to(Device::Cpu))
            .collect(),
    )
}

/// `matmul` (with batch broadcasting) and `linear` on the GPU: the forward and every gradient
/// within the GEMM class of the CPU's (the bias gradient is a sum of `M` products of 1).
#[test]
fn gradients_of_products_are_within_the_gemm_class() {
    let v = moderate(DType::F32);
    let abs = |a: &Tensor, b: &Tensor| a.abs().to(DType::F64).matmul(&b.abs().to(DType::F64));
    // matmul [2, 1, 5, 7] @ [3, 7, 6]: the batch broadcasts to [2, 3].
    let (x, y) = (
        v(70, 41).reshape([2, 1, 5, 7]),
        v(126, 42).reshape([3, 7, 6]),
    );
    let g = v(180, 43).reshape([2, 3, 5, 6]);
    let f = |t: &[Tensor]| t[0].matmul(&t[1]);
    let (yc, gc) = gradients(&f, &[x.clone(), y.clone()], &g, Device::Cpu);
    let (yg, gg) = gradients(&f, &[x.clone(), y.clone()], &g, GPU);
    check_gemm(&yg.to(GPU), &yc, &abs(&x, &y), 7, "matmul");
    // dx = sum over the broadcast batch of g @ y^T; dy = sum of x^T @ g.
    let dx_abs = abs(&g, &y.transpose(-1, -2)).sum_dim(1, true);
    let dy_abs = abs(&x.transpose(-1, -2), &g).sum_dim(0, false);
    check_gemm(&gg[0].to(GPU), &gc[0], &dx_abs, 6 * 3, "matmul dx");
    check_gemm(&gg[1].to(GPU), &gc[1], &dy_abs, 5 * 2, "matmul dy");
    // linear: x [33, 65] (rows of 65 features), weight [70, 65], bias [70].
    let (x, w, b) = (
        v(33 * 65, 44).reshape([33, 65]),
        v(70 * 65, 45).reshape([70, 65]),
        v(70, 46),
    );
    let g = v(33 * 70, 47).reshape([33, 70]);
    let f = |t: &[Tensor]| homura::nn::functional::linear(&t[0], &t[1], Some(&t[2]));
    let (yc, gc) = gradients(&f, &[x.clone(), w.clone(), b.clone()], &g, Device::Cpu);
    let (yg, gg) = gradients(&f, &[x.clone(), w.clone(), b.clone()], &g, GPU);
    let y_abs = abs(&x, &w.transpose(0, 1)).add(b.abs().to(DType::F64));
    check_gemm(&yg.to(GPU), &yc, &y_abs, 65 + 1, "linear");
    check_gemm(&gg[0].to(GPU), &gc[0], &abs(&g, &w), 70, "linear dx");
    check_gemm(
        &gg[1].to(GPU),
        &gc[1],
        &abs(&g.transpose(0, 1), &x),
        33,
        "linear dw",
    );
    let db_abs = g.abs().to(DType::F64).sum_dim(0, false);
    check_gemm(&gg[2].to(GPU), &gc[2], &db_abs, 33, "linear db");
}

unsafe extern "C" {
    /// `hmr_gemm_select` on the calling thread chooses `path` (-1: its own rules), or fails if
    /// that path doesn't take the description.
    fn hmr_test_force_gemm_path(path: i32);
}

/// Forces `path` on this thread until dropped.
struct Forced;

fn force(path: i32) -> Forced {
    // SAFETY: sets a thread-local value.
    unsafe { hmr_test_force_gemm_path(path) };
    Forced
}

impl Drop for Forced {
    fn drop(&mut self) {
        // SAFETY: as above.
        unsafe { hmr_test_force_gemm_path(-1) };
    }
}

/// The S series' paths (`HMR_GEMM_*` in csrc/homura_rocm.h), the layout they take and whether
/// they are the M <= 16 kernels.
const S_PATHS: [(i32, &str, &str, bool); 10] = [
    (1, "s08a_s7a5_unal", "NN", false),
    (2, "s07g_bk16", "NN", false),
    (3, "s08c_narrow_t256", "NN", true),
    (4, "s08c_splitk_t128", "NN", false),
    (5, "s08b_s8a1_nt", "NT", false),
    (6, "s08b_narrow_nt", "NT", true),
    (7, "s08b_splitk_nt_t128", "NT", false),
    (8, "s08b_s7g1_nt", "NT", false),
    (9, "s08b_s7g1_tn", "TN", false),
    (10, "s08b_narrow_tn", "TN", true),
];

/// (M, K, N) for the 128 x 128 tile paths: partial tiles, k not a multiple of 16 or 32 (an odd k
/// also makes S7g-1's leading dimension odd: scalar staging), several split-K slices, and one
/// with 128 tiles (a single slice for the split-K paths).
const TILED: [[usize; 3]; 6] = [
    [3, 5, 7],
    [127, 33, 129],
    [128, 64, 128],
    [130, 40, 257],
    [257, 300, 131],
    [1024, 40, 2048],
];

/// (M, K, N) for the M <= 16 paths: every row instance (1, 2, 4, 8, 16), one slice (k <= 16) and
/// several, columns past a block, a k that isn't a multiple of 4.
const NARROW: [[usize; 3]; 6] = [
    [2, 7, 3],
    [3, 40, 300],
    [5, 16, 257],
    [8, 33, 64],
    [9, 300, 513],
    [16, 1001, 31],
];

#[test]
fn every_path_matches_the_cpu() {
    let v = moderate(DType::F32);
    let small = |n: usize, seed: u64| {
        homura::manual_seed(seed);
        homura::randint(-8, 9, [n]).to(DType::F32)
    };
    for (path, name, layout, narrow) in S_PATHS {
        let shapes: &[[usize; 3]] = if narrow { &NARROW } else { &TILED };
        for &shape in shapes {
            let [m, k, n] = shape;
            // The 128-tile shape is there for the split-K paths' single slice; the CPU's
            // reference is slow, so it is checked once, without the writeback entries.
            let big = m * n >= 1 << 20;
            let split_k = name.contains("splitk");
            if big && !split_k {
                continue;
            }
            for batch in [1, 3] {
                if big && batch > 1 {
                    continue;
                }
                let what = format!("{name} {shape:?} batch {batch}");
                // Exact on both: the same bits.
                let [(xc, xg), (yc, yg)] = factors(shape, batch, layout, &small);
                let got = {
                    let _f = force(path);
                    xg.matmul(&yg)
                };
                assert_same(&got, &xc.matmul(&yc), &format!("{what} exact"));
                // Rounded: within the GEMM class, with the writeback of each entry.
                let [(xc, xg), (yc, yg)] = factors(shape, batch, layout, &v);
                let abs3 = xc.abs().to(DType::F64).matmul(&yc.abs().to(DType::F64));
                let p3 = xc.matmul(&yc).to(DType::F64);
                let got = {
                    let _f = force(path);
                    xg.matmul(&yg)
                };
                check_gemm(&got, &xc.matmul(&yc), &abs3, k, &what);
                if big {
                    continue;
                }
                for (entry, f) in [ENTRIES[1], ENTRIES[3], ENTRIES[4]] {
                    if entry == "addbmm" && batch > 1 {
                        continue;
                    }
                    let shape_c = input_shape(entry, batch, m, n);
                    let cc = v(shape_c.iter().product(), 14).reshape(shape_c.clone());
                    let cg = cc.to(GPU);
                    let (pa, p) = match entry {
                        "addmm" => (abs3.select(0, 0), p3.select(0, 0)),
                        "baddbmm" => (abs3.clone(), p3.clone()),
                        _ => (abs3.sum_dim(0, false), p3.sum_dim(0, false)),
                    };
                    for (beta, alpha) in [(1.0, 1.0), (-0.5, 3.0), (0.0, 2.0)] {
                        let (b, a) = (Scalar::Float(beta), Scalar::Float(alpha));
                        let gpu = {
                            let _f = force(path);
                            f(&cg, &xg, &yg, b, a)
                        };
                        let cpu = f(&cc, &xc, &yc, b, a);
                        let what = format!("{what} {entry} beta {beta} alpha {alpha}");
                        check_scaled(&gpu, &cpu, &pa, &p, k, 1, (beta, alpha), &what);
                    }
                }
            }
        }
    }
}

/// A forced path that doesn't take the description is an error, not a wrong result.
#[test]
fn forced_paths_refuse_what_they_dont_take() {
    let message = |f: &dyn Fn()| {
        let e = catch_unwind(AssertUnwindSafe(f)).expect_err("panics");
        e.downcast_ref::<String>().cloned().unwrap_or_default()
    };
    let x = homura::ones_with([17, 4], GPU);
    let y = homura::ones_with([4, 5], GPU);
    let m = message(&|| {
        let _f = force(3);
        drop(x.matmul(&y))
    });
    assert_eq!(
        m,
        "matrix product path 3 can't take this description: M > 16"
    );
    let m = message(&|| {
        let _f = force(5);
        drop(x.matmul(&y))
    });
    assert_eq!(
        m,
        "matrix product path 5 can't take this description: another layout"
    );
    let xd = x.to(DType::F64);
    let m = message(&|| {
        let _f = force(1);
        drop(xd.matmul(&y.to(DType::F64)))
    });
    assert_eq!(
        m,
        "matrix product path 1 can't take this description: an F32 path"
    );
}

/// The writeback rounds `beta * c` before adding `alpha * p` (`-ffp-contract=off`, no fused
/// multiply-add): with c = beta = 1 + 2^-23 and alpha * p = -(1 + 2^-22), the CPU rounds
/// beta * c to 1 + 2^-22 and gets 0, a fused `fma(c, beta, -(1 + 2^-22))` gets 2^-46 (F64: the
/// same with 2^-52 and 2^-51, 2^-104). On every path that takes the 1 x 1 product.
#[test]
fn the_writeback_rounds_each_step() {
    let e = 1.0 + f32::EPSILON;
    let alpha = -(1.0 + 2.0 * f32::EPSILON as f64);
    let c = Tensor::from_slice(&[e], [1, 1]);
    let one = Tensor::from_slice(&[1.0f32], [1, 1]);
    let cpu = c.addmm(&one, &one, e as f64, alpha);
    assert_eq!(cpu.to_vec::<f32>(), [0.0]);
    let (cg, og) = (c.to(GPU), one.to(GPU));
    assert_same(
        &cg.addmm(&og, &og, e as f64, alpha),
        &cpu,
        "rule-chosen path",
    );
    for path in [0, 1, 2, 3, 4] {
        let _f = force(path);
        assert_same(
            &cg.addmm(&og, &og, e as f64, alpha),
            &cpu,
            &format!("path {path}"),
        );
    }
    let e64 = 1.0 + f64::EPSILON;
    let alpha64 = -(1.0 + 2.0 * f64::EPSILON);
    let cd = Tensor::from_slice(&[e64], [1, 1]);
    let od = Tensor::from_slice(&[1.0f64], [1, 1]);
    let cpu = cd.addmm(&od, &od, e64, alpha64);
    assert_eq!(cpu.to_vec::<f64>(), [0.0]);
    let got = cd.to(GPU).addmm(&od.to(GPU), &od.to(GPU), e64, alpha64);
    assert_same(&got, &cpu, "f64");
}

const ALL: [DType; 10] = [
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

/// Coefficients of `addr` in `dtype` that every dtype of the kind takes (floats: special values,
/// a tiny one that rounds to zero in half precision, one that doesn't round exactly).
fn addr_coefficients(dtype: DType) -> Vec<Scalar> {
    if dtype == DType::Bool {
        vec![Scalar::Bool(false), Scalar::Bool(true)]
    } else if dtype.is_floating_point() {
        [
            0.0,
            1.0,
            -1.0,
            0.5,
            -1.5,
            0.1,
            1e-30,
            f64::INFINITY,
            f64::NAN,
        ]
        .map(Scalar::Float)
        .to_vec()
    } else {
        let mut c = [0i64, 1, -1, 2, 3, 127].map(Scalar::Int).to_vec();
        if dtype == DType::U8 {
            // Negative coefficients down to -255 wrap for U8.
            c.push(Scalar::Int(-255));
        }
        c
    }
}

/// The layout of `v2` in [`addr_case`]: contiguous from an aligned start (the vector kernel
/// when the output's rows are aligned), from one element into its storage, or every other
/// element (the scalar kernel).
#[derive(Clone, Copy, Debug)]
enum V2 {
    Aligned,
    Offset,
    Strided,
}

/// `v1` `[n]`, `v2` `[m]` and the inputs that broadcast to `[n, m]`, on the CPU and the same
/// views on the GPU: random bit patterns (NaN payloads, subnormals) after the special values,
/// `v1` every other element from an offset, `v2` as `kind` says, the inputs full (from an
/// offset), stored transposed, every other column, a row, a column, one element and
/// zero-dimensional.
fn addr_case(dtype: DType, (n, m, kind): (usize, usize, V2), seed: u64) -> Vec<[Tensor; 6]> {
    let b1 = values(dtype, 2 * n + 1, seed);
    let b2 = values(dtype, 2 * m + 1, seed + 1);
    let bc = values(dtype, 2 * n * m + n + m + 1, seed + 2);
    let v1 = |t: &Tensor| t.narrow(0, 1, 2 * n).reshape([n, 2]).select(1, 0);
    let v2 = |t: &Tensor| match kind {
        V2::Aligned => t.narrow(0, 0, m),
        V2::Offset => t.narrow(0, 1, m),
        V2::Strided => t.narrow(0, 0, 2 * m).reshape([m, 2]).select(1, 0),
    };
    let inputs: [fn(&Tensor, usize, usize) -> Tensor; 7] = [
        |t, n, m| t.narrow(0, 1, n * m).reshape([n, m]),
        |t, n, m| t.narrow(0, 0, n * m).reshape([m, n]).transpose(0, 1),
        // Every other column: aligned rows (for even `m`) without a unit column stride.
        |t, n, m| t.narrow(0, 0, 2 * n * m).reshape([n, m, 2]).select(2, 0),
        |t, _, m| t.narrow(0, 0, m),
        |t, n, _| t.narrow(0, 0, n).reshape([n, 1]),
        |t, _, _| t.narrow(0, 0, 1),
        |t, _, _| t.narrow(0, 0, 1).reshape(()),
    ];
    let (g1, g2, gc) = (b1.to(GPU), b2.to(GPU), bc.to(GPU));
    inputs
        .iter()
        .map(|input| {
            [
                input(&bc, n, m),
                v1(&b1),
                v2(&b2),
                input(&gc, n, m),
                v1(&g1),
                v2(&g2),
            ]
        })
        .collect()
}

/// `addr` on the GPU has the CPU's bits in every dtype, layout and coefficient: all pairs of
/// coefficients on two small shapes (rows of four columns take the vector kernel when aligned),
/// a few on shapes that cross the block's columns and rows (one column, one row with a scalar
/// tail, 257 columns, more rows than a block holds, rows that are multiples of four columns,
/// with `v2` aligned, misaligned or strided) and empty ones.
#[test]
fn addr_has_the_cpus_bits() {
    use V2::*;
    let shapes = [
        (3, 5, Aligned),
        (3, 8, Aligned),
        (1, 1, Aligned),
        (1, 1031, Aligned),
        (1031, 1, Offset),
        (17, 257, Strided),
        (300, 3, Aligned),
        (5, 1028, Aligned),
        (5, 1028, Offset),
        (5, 1028, Strided),
        (300, 4, Aligned),
        (0, 4, Aligned),
        (4, 0, Aligned),
    ];
    let mut checked = 0;
    for dtype in ALL {
        let coefficients = addr_coefficients(dtype);
        let at = |i: usize| coefficients[i % coefficients.len()];
        let few = [
            (at(1), at(1)),
            (at(0), at(2)),
            (at(3), at(0)),
            (at(8), at(5)),
        ];
        for (k, &(n, m, kind)) in shapes.iter().enumerate() {
            for [c, v1, v2, gc, g1, g2] in addr_case(dtype, (n, m, kind), k as u64 + 3) {
                let pairs: Vec<(Scalar, Scalar)> = if k < 2 {
                    coefficients
                        .iter()
                        .flat_map(|&a| coefficients.iter().map(move |&b| (a, b)))
                        .collect()
                } else {
                    few.to_vec()
                };
                for (alpha, beta) in pairs {
                    let what = format!(
                        "{dtype} [{n}, {m}] {kind:?} {:?} alpha {alpha} beta {beta}",
                        c.shape()
                    );
                    let want = c.addr(&v1, &v2, beta, alpha);
                    assert_bitwise(&gc.addr(&g1, &g2, beta, alpha), &want, &what);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 2500, "{checked}");
}

/// `addr` promotes its inputs, and rejects the coefficients the CPU rejects with its messages.
#[test]
fn addr_promotes_and_checks_like_the_cpu() {
    let x = Tensor::from_slice(&[1i32, -2, 3], [3]);
    let y = Tensor::from_slice(&[0.5f32, 2.0], [2]);
    let c = Tensor::from_slice(&[1u8], [1]);
    let want = c.addr(&x, &y, 2, 3);
    assert_eq!(want.dtype(), DType::F32);
    assert_bitwise(
        &c.to(GPU).addr(&x.to(GPU), &y.to(GPU), 2, 3),
        &want,
        "promoted",
    );
    let message = |f: &dyn Fn()| {
        let e = catch_unwind(AssertUnwindSafe(f)).expect_err("panics");
        e.downcast_ref::<String>().cloned().unwrap_or_default()
    };
    let rejected: [(Scalar, Scalar); 3] = [
        (Scalar::Float(0.5), Scalar::Int(1)),
        (Scalar::Int(1), Scalar::Bool(true)),
        (Scalar::Int(1), Scalar::Int(300)),
    ];
    let (xi, ci) = (x.to(DType::I8), Tensor::from_slice(&[1i8], [1]));
    for (beta, alpha) in rejected {
        let cpu = message(&|| drop(ci.addr(&xi, &xi, beta, alpha)));
        let (xg, cg) = (xi.to(GPU), ci.to(GPU));
        assert_eq!(message(&|| drop(cg.addr(&xg, &xg, beta, alpha))), cpu);
    }
}

/// The grid's limits, for the scalar kernel (a strided `v2`, or one column) and the vector
/// kernel (four columns per thread): more rows than 65535 blocks hold, and more columns (or
/// groups of four) than 65536 blocks of 256 threads.
#[test]
fn addr_crosses_the_grid_limits() {
    let blocks = 65536 * 256;
    for (n, m, strided) in [
        (65535 * 256 + 5, 1, false),
        (65535 + 3, 256, true),
        (blocks / 4 + 5, 4, false),
        (1, blocks + 5, true),
        (1, 4 * blocks + 7, false),
    ] {
        let v1 = values(DType::U8, n, 31);
        let b2 = values(DType::I8, 2 * m, 32).to(DType::U8);
        let v2 = if strided {
            b2.reshape([m, 2]).select(1, 0)
        } else {
            b2.narrow(0, 0, m)
        };
        let c = values(DType::U8, m, 33);
        let want = c.addr(&v1, &v2, 2, 3);
        let g2 = if strided {
            b2.to(GPU).reshape([m, 2]).select(1, 0)
        } else {
            b2.to(GPU).narrow(0, 0, m)
        };
        let got = c.to(GPU).addr(&v1.to(GPU), &g2, 2, 3);
        assert_same(&got, &want, &format!("[{n}, {m}] strided {strided}"));
    }
}

/// The gradients of `addr` on the GPU (through the matrix products) within the GEMM class of the
/// CPU's.
#[test]
fn gradients_of_addr_are_within_the_gemm_class() {
    let v = moderate(DType::F32);
    let (c, x, y) = (v(70, 51), v(33, 52), v(70, 53));
    let g = v(33 * 70, 54).reshape([33, 70]);
    let f = |t: &[Tensor]| t[0].addr(&t[1], &t[2], 2.0, 0.5);
    let (yc, gc) = gradients(&f, &[c.clone(), x.clone(), y.clone()], &g, Device::Cpu);
    let (yg, gg) = gradients(&f, &[c.clone(), x.clone(), y.clone()], &g, GPU);
    assert_bitwise(&yg.to(GPU), &yc, "addr");
    let ga = g.abs().to(DType::F64);
    // dc = beta * sum over the rows of g; dx = alpha * g @ y; dy = alpha * g^T @ x.
    let dc_abs = ga.sum_dim(0, false).mul(2.0);
    let dx_abs = ga.matmul(&y.abs().to(DType::F64)).mul(0.5);
    let dy_abs = ga.transpose(0, 1).matmul(&x.abs().to(DType::F64)).mul(0.5);
    check_gemm(&gg[0].to(GPU), &gc[0], &dc_abs, 33, "addr dc");
    check_gemm(&gg[1].to(GPU), &gc[1], &dx_abs, 70, "addr dx");
    check_gemm(&gg[2].to(GPU), &gc[2], &dy_abs, 33, "addr dy");
}
