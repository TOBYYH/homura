//! Reductions and softmax on a ROCm GPU against the CPU (M4 step 4a).
//!
//! - The bitwise class (M4.12): integer and bool sums and products, all, any, max, min and the
//!   arg reductions (ties and NaNs included).
//! - The tolerance classes of M4.12 for the float sums, means, products, variances and softmax,
//!   which both sides accumulate in f64 but in different orders: ordinary finite inputs within
//!   the bounds, exact cases with the same bits, special values classified alike, and wrong
//!   results that the judges must reject.
//! - Every plan of `csrc/reduce.hpp` (lanes per output, chunks, rows cut by chunks, the grid's
//!   rounds), with inputs that reveal a missing, repeated or misplaced element, and repeated
//!   runs that give the same bits.
#![cfg(feature = "rocm")]

#[path = "common/rocm.rs"]
mod rocm;

use homura::{DType, Device, Tensor};
use rocm::{DTYPES, GPU, Words, assert_bitwise, check_within, floats, pow2, ulp, values};

const FLOATS: [DType; 4] = [DType::F16, DType::BF16, DType::F32, DType::F64];

/// The unit roundoff of f64, the type both sides accumulate float reductions in.
fn eps() -> f64 {
    pow2(-53)
}

/// Shapes of rank 0 to 5, with size-1 and empty dimensions.
const SHAPES: [&[usize]; 7] = [
    &[],
    &[7],
    &[4, 5],
    &[3, 1, 4],
    &[3, 0, 2],
    &[2, 3, 2, 5],
    &[2, 1, 3, 2, 3],
];

/// A tensor of `shape` on the CPU and the same one on the GPU, in one layout.
struct Case {
    name: String,
    cpu: Tensor,
    gpu: Tensor,
}

/// `shape` in four layouts: contiguous, with reversed strides, narrowed in its last dimension
/// (an offset and gaps), and broadcast along its first; `make(n)` gives a base of `n` elements
/// on the CPU.
fn cases(shape: &[usize], make: &dyn Fn(usize) -> Tensor) -> Vec<Case> {
    let n: usize = shape.iter().product();
    let rank = shape.len();
    let mut out = vec![];
    let mut add = |name: &str, base: Tensor, view: &dyn Fn(&Tensor) -> Tensor| {
        let cpu = view(&base);
        let gpu = view(&base.to(GPU));
        assert_eq!(cpu.shape(), shape, "{name}");
        assert_eq!(gpu.stride(), cpu.stride(), "{name}");
        out.push(Case {
            name: format!("{name} {shape:?}"),
            cpu,
            gpu,
        });
    };
    add("contiguous", make(n), &|t| t.reshape(shape));
    if rank >= 2 {
        let reversed: Vec<usize> = shape.iter().rev().copied().collect();
        let perm: Vec<i64> = (0..rank as i64).rev().collect();
        add("reversed strides", make(n), &|t| {
            t.reshape(&reversed[..]).permute(&perm[..])
        });
    }
    if rank >= 1 {
        let mut wide = shape.to_vec();
        wide[rank - 1] = 2 * shape[rank - 1] + 1;
        let m: usize = wide.iter().product();
        let last = shape[rank - 1];
        add("narrowed", make(m), &|t| {
            t.reshape(&wide[..]).narrow(rank as i64 - 1, 1, last)
        });
        if shape[0] > 1 {
            let mut one = shape.to_vec();
            one[0] = 1;
            add("broadcast", make(n / shape[0]), &|t| {
                t.reshape(&one[..]).expand(shape)
            });
        }
    }
    out
}

/// Every subset of the dimensions of a rank, every other one spelled negatively.
fn dim_sets(rank: usize) -> Vec<Vec<i64>> {
    (0..1usize << rank)
        .map(|m| {
            (0..rank)
                .filter(|d| m >> d & 1 == 1)
                .map(|d| {
                    if (m + d) % 2 == 0 {
                        d as i64
                    } else {
                        d as i64 - rank as i64
                    }
                })
                .collect()
        })
        .collect()
}

/// The number of elements folded into each output of a reduction of `x` into `out`.
fn folded(x: &Tensor, out: &Tensor) -> usize {
    if out.num_elements() == 0 {
        0
    } else {
        x.num_elements() / out.num_elements()
    }
}

/// Values `±m * 2^e` with `m` in [1, 2) and `e` in `[-scale, scale]`, of mixed signs (so that
/// sums cancel), rounded to `dtype`.
fn finite(dtype: DType, n: usize, scale: u64, seed: u64) -> Tensor {
    let mut w = Words::new(seed);
    let v: Vec<f64> = (0..n)
        .map(|_| {
            let r = w.next();
            let e = (r % (2 * scale + 1)) as i32 - scale as i32;
            let m = 1.0 + (r >> 11) as f64 / (1u64 << 53) as f64;
            let sign = if r & (1 << 8) != 0 { -1.0 } else { 1.0 };
            sign * m * pow2(e)
        })
        .collect();
    Tensor::from_slice(&v, [n]).to(dtype)
}

/// Values in ±[0.5, 2): no order of their products leaves the normal range of f64 for fewer
/// than 1021 factors (M4.12: the product class excludes intermediate underflow).
fn factors(dtype: DType, n: usize, seed: u64) -> Tensor {
    let mut w = Words::new(seed);
    let v: Vec<f64> = (0..n)
        .map(|_| {
            let r = w.next();
            let m = 0.5 + 1.5 * ((r >> 11) as f64 / (1u64 << 53) as f64);
            if r & 1 == 1 { -m } else { m }
        })
        .collect();
    Tensor::from_slice(&v, [n]).to(dtype)
}

// ---- The judges of M4.12 ------------------------------------------------------------------

/// The sum class: `|gpu - cpu| <= 2 n eps Σ|x| (/ n for a mean) + ulp_out`.
fn check_sum(
    gpu: &Tensor,
    cpu: &Tensor,
    abs_sum: &Tensor,
    n: usize,
    mean: bool,
) -> Result<(), String> {
    let div = if mean { n.max(1) as f64 } else { 1.0 };
    let extra: Vec<f64> = floats(abs_sum)
        .iter()
        .map(|&s| 2.0 * n as f64 * eps() * s / div)
        .collect();
    check_within(gpu, cpu, &extra).map(|_| ())
}

/// The product class: `|gpu - cpu| <= 2 n eps |cpu| + ulp_out`.
fn check_prod(gpu: &Tensor, cpu: &Tensor, n: usize) -> Result<(), String> {
    let extra: Vec<f64> = floats(cpu)
        .iter()
        .map(|&c| 2.0 * n as f64 * eps() * c.abs())
        .collect();
    check_within(gpu, cpu, &extra).map(|_| ())
}

/// The variance class: `Δ = 4 n eps v`, `v` the CPU's variance in f64 (Σ(x - x̄)² / (n -
/// correction)); `|gpu - cpu| <= Δ + ulp_out` for var, `min(Δ / √v, √Δ) + ulp_out` for std.
fn check_var(
    gpu: &Tensor,
    cpu: &Tensor,
    var_f64: &Tensor,
    n: usize,
    std: bool,
) -> Result<(), String> {
    let extra: Vec<f64> = floats(var_f64)
        .iter()
        .map(|&v| {
            let delta = 4.0 * n as f64 * eps() * v;
            if !std {
                delta
            } else if v > 0.0 {
                (delta / v.sqrt()).min(delta.sqrt())
            } else {
                delta.sqrt()
            }
        })
        .collect();
    check_within(gpu, cpu, &extra).map(|_| ())
}

/// The rows of `t` along `dim`, as f64.
fn rows(t: &Tensor, dim: i64) -> Vec<Vec<f64>> {
    let last = t.num_dims() as i64 - 1;
    if t.num_dims() == 0 {
        return vec![floats(t)];
    }
    let n = t.shape()[dim.rem_euclid(t.num_dims() as i64) as usize];
    let v = floats(&t.transpose(dim, last));
    if n == 0 {
        return vec![];
    }
    v.chunks(n).map(<[f64]>::to_vec).collect()
}

/// The softmax class along `dim` of `x`: for each element `(2 n eps + e_exp) |cpu| + ulp_out`,
/// `e_exp = 2 eps` (exp computed in f64), and the rows of the GPU's result sum to 1 within
/// `2 n eps + n e_exp + n ulp_out(1)`. For log-softmax: `2 n eps + 2 eps |log S| + ulp_out` with
/// `S = Σ exp(x - max)`, and `Σ exp(gpu)` is 1 within `2 n eps + n e_exp + Σ exp(gpu_i)
/// ulp_out(|gpu_i|)`.
fn check_softmax(
    gpu: &Tensor,
    cpu: &Tensor,
    x: &Tensor,
    dim: i64,
    log: bool,
) -> Result<(), String> {
    let dtype = cpu.dtype();
    let e_exp = 2.0 * eps();
    let n = if x.num_dims() == 0 {
        1
    } else {
        x.shape()[dim.rem_euclid(x.num_dims() as i64) as usize]
    } as f64;
    let acc = 2.0 * n * eps();
    // The element bounds, in the transposed (row-major by rows) order, moved back.
    let xr = rows(x, dim);
    let cr = rows(cpu, dim);
    let mut extra_rows: Vec<Vec<f64>> = vec![];
    for (xs, cs) in xr.iter().zip(&cr) {
        let max = xs
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| if v > m { v } else { m });
        let s: f64 = xs.iter().map(|&v| (v - max).exp()).sum();
        extra_rows.push(
            cs.iter()
                .map(|&c| {
                    if log {
                        acc + 2.0 * eps() * s.ln().abs()
                    } else {
                        (acc + e_exp) * c.abs()
                    }
                })
                .collect(),
        );
    }
    // Back to `cpu`'s element order: the bound of each element through the same transpose.
    let flat: Vec<f64> = extra_rows.concat();
    let extra = if x.num_dims() == 0 {
        flat
    } else {
        let last = x.num_dims() as i64 - 1;
        let shape_t = cpu.transpose(dim, last).shape().to_vec();
        floats(
            &Tensor::from_slice(&flat, &shape_t[..])
                .transpose(dim, last)
                .contiguous(),
        )
    };
    check_within(gpu, cpu, &extra)?;
    for (r, gs) in rows(gpu, dim).iter().enumerate() {
        if gs.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let (sum, slack) = if log {
            let e: Vec<f64> = gs.iter().map(|v| v.exp()).collect();
            let slack: f64 = gs
                .iter()
                .zip(&e)
                .map(|(&g, &e)| e * ulp(dtype, g.abs()))
                .sum();
            (e.iter().sum::<f64>(), slack)
        } else {
            (gs.iter().sum::<f64>(), n * ulp(dtype, 1.0))
        };
        let bound = acc + n * e_exp + slack;
        if (sum - 1.0).abs() > bound {
            return Err(format!("row {r}: sums to {sum:e}, bound {bound:e}"));
        }
    }
    Ok(())
}

// ---- The bitwise class ---------------------------------------------------------------------

#[test]
fn integer_sums_all_and_any_match_the_cpu() {
    for dtype in DTYPES {
        let floating = dtype.is_floating_point();
        for (s, shape) in SHAPES.iter().enumerate() {
            for c in cases(shape, &|n| values(dtype, n, 11 + s as u64)) {
                for dims in dim_sets(shape.len()) {
                    for keepdim in [false, true] {
                        let what = format!("{dtype} {} dims {dims:?} keepdim {keepdim}", c.name);
                        let d = &dims[..];
                        if !floating {
                            assert_bitwise(
                                &c.gpu.sum_dim(d, keepdim),
                                &c.cpu.sum_dim(d, keepdim),
                                &format!("sum {what}"),
                            );
                        }
                        assert_bitwise(
                            &c.gpu.all_dim(d, keepdim),
                            &c.cpu.all_dim(d, keepdim),
                            &format!("all {what}"),
                        );
                        assert_bitwise(
                            &c.gpu.any_dim(d, keepdim),
                            &c.cpu.any_dim(d, keepdim),
                            &format!("any {what}"),
                        );
                    }
                }
                if !floating {
                    assert_bitwise(
                        &c.gpu.prod(),
                        &c.cpu.prod(),
                        &format!("prod {dtype} {}", c.name),
                    );
                    for d in 0..shape.len() as i64 {
                        assert_bitwise(
                            &c.gpu.prod_dim(d, false),
                            &c.cpu.prod_dim(d, false),
                            &format!("prod_dim {d} {dtype} {}", c.name),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn max_min_and_arg_reductions_match_the_cpu() {
    for dtype in DTYPES {
        for (s, shape) in SHAPES.iter().enumerate() {
            for c in cases(shape, &|n| values(dtype, n, 31 + s as u64)) {
                let what = format!("{dtype} {}", c.name);
                if c.cpu.num_elements() > 0 {
                    assert_bitwise(&c.gpu.max(), &c.cpu.max(), &format!("max {what}"));
                    assert_bitwise(&c.gpu.min(), &c.cpu.min(), &format!("min {what}"));
                    assert_bitwise(&c.gpu.argmax(), &c.cpu.argmax(), &format!("argmax {what}"));
                    assert_bitwise(&c.gpu.argmin(), &c.cpu.argmin(), &format!("argmin {what}"));
                }
                let rank = shape.len() as i64;
                for d in -rank.max(1)..rank.max(1) {
                    if rank > 0 && shape[d.rem_euclid(rank) as usize] == 0 {
                        continue;
                    }
                    for keepdim in [false, true] {
                        let w = format!("{what} dim {d} keepdim {keepdim}");
                        let (g, p) = (c.gpu.max_dim(d, keepdim), c.cpu.max_dim(d, keepdim));
                        assert_bitwise(&g.values, &p.values, &format!("max_dim {w}"));
                        assert_bitwise(&g.indices, &p.indices, &format!("max_dim indices {w}"));
                        let (g, p) = (c.gpu.min_dim(d, keepdim), c.cpu.min_dim(d, keepdim));
                        assert_bitwise(&g.values, &p.values, &format!("min_dim {w}"));
                        assert_bitwise(&g.indices, &p.indices, &format!("min_dim indices {w}"));
                        assert_bitwise(
                            &c.gpu.argmax_dim(d, keepdim),
                            &c.cpu.argmax_dim(d, keepdim),
                            &format!("argmax_dim {w}"),
                        );
                    }
                }
            }
        }
    }
}

/// Ties pick the first position, +0 and -0 included, and the first NaN wins.
#[test]
fn ties_and_nans_pick_the_cpus_position() {
    let rows: [&[f64]; 6] = [
        &[1.0, 3.0, 3.0, 2.0, 3.0],
        &[0.0, -0.0, 0.0, -0.0, -1.0],
        &[-0.0, 0.0, -0.0, 0.0, 1.0],
        &[1.0, f64::NAN, 5.0, -f64::NAN, f64::INFINITY],
        &[f64::NEG_INFINITY; 5],
        &[2.0, 2.0, 2.0, 2.0, 2.0],
    ];
    let v: Vec<f64> = rows.concat();
    for dtype in FLOATS {
        let x = Tensor::from_slice(&v, [6, 5]).to(dtype);
        let g = x.to(GPU);
        for d in [0, 1] {
            for (gv, cv) in [
                (g.max_dim(d, false), x.max_dim(d, false)),
                (g.min_dim(d, false), x.min_dim(d, false)),
            ] {
                assert_bitwise(&gv.values, &cv.values, &format!("{dtype} dim {d}"));
                assert_bitwise(&gv.indices, &cv.indices, &format!("{dtype} dim {d}"));
            }
        }
        assert_bitwise(&g.max(), &x.max(), &format!("{dtype} max"));
        assert_bitwise(&g.argmin(), &x.argmin(), &format!("{dtype} argmin"));
    }
}

// ---- The tolerance classes -----------------------------------------------------------------

#[test]
fn float_sums_and_means_are_within_the_sum_class() {
    for dtype in FLOATS {
        for (s, shape) in SHAPES.iter().enumerate() {
            for scale in [0, 8] {
                for c in cases(shape, &|n| finite(dtype, n, scale, 51 + s as u64)) {
                    let abs = c.cpu.to(DType::F64).abs();
                    for dims in dim_sets(shape.len()) {
                        for keepdim in [false, true] {
                            let what = format!(
                                "{dtype} {} scale {scale} dims {dims:?} keepdim {keepdim}",
                                c.name
                            );
                            let d = &dims[..];
                            let a = abs.sum_dim(d, keepdim);
                            let cpu = c.cpu.sum_dim(d, keepdim);
                            let n = folded(&c.cpu, &cpu);
                            check_sum(&c.gpu.sum_dim(d, keepdim), &cpu, &a, n, false)
                                .unwrap_or_else(|e| panic!("sum {what}: {e}"));
                            check_sum(
                                &c.gpu.mean_dim(d, keepdim),
                                &c.cpu.mean_dim(d, keepdim),
                                &a,
                                n,
                                true,
                            )
                            .unwrap_or_else(|e| panic!("mean {what}: {e}"));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn float_products_are_within_the_product_class() {
    for dtype in FLOATS {
        for (s, shape) in SHAPES.iter().enumerate() {
            for c in cases(shape, &|n| factors(dtype, n, 61 + s as u64)) {
                let what = format!("{dtype} {}", c.name);
                let cpu = c.cpu.prod();
                check_prod(&c.gpu.prod(), &cpu, c.cpu.num_elements())
                    .unwrap_or_else(|e| panic!("prod {what}: {e}"));
                for d in 0..shape.len() as i64 {
                    let cpu = c.cpu.prod_dim(d, true);
                    let n = folded(&c.cpu, &cpu);
                    check_prod(&c.gpu.prod_dim(d, true), &cpu, n)
                        .unwrap_or_else(|e| panic!("prod_dim {d} {what}: {e}"));
                }
            }
        }
    }
}

#[test]
fn variances_are_within_the_variance_class() {
    for dtype in FLOATS {
        for (s, shape) in SHAPES.iter().enumerate() {
            for c in cases(shape, &|n| finite(dtype, n, 4, 71 + s as u64)) {
                let x64 = c.cpu.to(DType::F64);
                for dims in dim_sets(shape.len()) {
                    let d = &dims[..];
                    for correction in [0, 1] {
                        let what =
                            format!("{dtype} {} dims {dims:?} correction {correction}", c.name);
                        let v64 = x64.var_dim(d, correction, false);
                        let cpu = c.cpu.var_dim(d, correction, false);
                        let n = folded(&c.cpu, &cpu);
                        // Too few elements for the correction: the CPU divides by 0, and finite
                        // inputs that aren't all equal make that Inf on both sides only when the
                        // squares are exactly the same, which the exact case below covers.
                        if n as i64 <= correction {
                            continue;
                        }
                        check_var(&c.gpu.var_dim(d, correction, false), &cpu, &v64, n, false)
                            .unwrap_or_else(|e| panic!("var {what}: {e}"));
                        check_var(
                            &c.gpu.std_dim(d, correction, false),
                            &c.cpu.std_dim(d, correction, false),
                            &v64,
                            n,
                            true,
                        )
                        .unwrap_or_else(|e| panic!("std {what}: {e}"));
                    }
                }
            }
        }
    }
}

#[test]
fn softmax_is_within_the_softmax_class() {
    for dtype in FLOATS {
        for (s, shape) in SHAPES.iter().enumerate() {
            for c in cases(shape, &|n| finite(dtype, n, 3, 81 + s as u64)) {
                let rank = shape.len() as i64;
                for d in -rank.max(1)..rank.max(1) {
                    for log in [false, true] {
                        let what = format!("{dtype} {} dim {d} log {log}", c.name);
                        let (g, p) = if log {
                            (c.gpu.log_softmax(d), c.cpu.log_softmax(d))
                        } else {
                            (c.gpu.softmax(d), c.cpu.softmax(d))
                        };
                        check_softmax(&g, &p, &c.cpu.to(DType::F64), d, log)
                            .unwrap_or_else(|e| panic!("{what}: {e}"));
                    }
                }
            }
        }
    }
}

/// Sums whose every partial sum is exact give the CPU's bits, and so do means, variances and
/// products of such values; the special values propagate as on the CPU.
#[test]
fn exact_cases_and_special_values_match_the_cpu() {
    for dtype in FLOATS {
        // Small integers in rows of 8 and 32: every partial sum is exact in f64, so are the
        // means (a power of two divides them) and the partial sums of the squared deviations.
        let v: Vec<f64> = (0..256).map(|i| ((i * 37) % 17) as f64 - 8.0).collect();
        let x = Tensor::from_slice(&v, [8, 32]).to(dtype);
        let g = x.to(GPU);
        for d in [0i64, 1] {
            assert_bitwise(
                &g.sum_dim(d, false),
                &x.sum_dim(d, false),
                &format!("{dtype} sum {d}"),
            );
            assert_bitwise(
                &g.mean_dim(d, false),
                &x.mean_dim(d, false),
                &format!("{dtype} mean {d}"),
            );
            assert_bitwise(
                &g.var_dim(d, 0, false),
                &x.var_dim(d, 0, false),
                &format!("{dtype} var {d}"),
            );
            assert_bitwise(
                &g.std_dim(d, 1, false),
                &x.std_dim(d, 1, false),
                &format!("{dtype} std {d}"),
            );
        }
        // Exact sums that the dtype can't hold: the mean divides the f64 sum, not its rounding
        // (F16: 4202 rounds to 4200, so 1400 instead of 1401; F32: 50331643 rounds up, so
        // 16777215 instead of 16777214).
        let m: &[f64] = match dtype {
            DType::F16 => &[1401.0, 1401.0, 1400.0],
            DType::F32 => &[16777215.0, 16777215.0, 16777213.0],
            _ => &[1.0, 2.0, 2.0],
        };
        let m = Tensor::from_slice(m, [1, 3]).to(dtype);
        assert_bitwise(
            &m.to(GPU).mean_dim(1, false),
            &m.mean_dim(1, false),
            &format!("{dtype} mean of an unrepresentable sum"),
        );
        let p = Tensor::from_slice(&[2.0, -0.5, 4.0, 0.25, -1.0, 8.0], [6]).to(dtype);
        assert_bitwise(&p.to(GPU).prod(), &p.prod(), &format!("{dtype} prod"));
        // Too few elements for the correction: n / 0 is Inf for spread values, NaN for equal.
        let w = Tensor::from_slice(&[1.0, 3.0, 2.0, 2.0], [2, 2]).to(dtype);
        for correction in [2, 3] {
            assert_bitwise(
                &w.to(GPU).var_dim(1, correction, false),
                &w.var_dim(1, correction, false),
                &format!("{dtype} var correction {correction}"),
            );
        }
        // Empty reductions: 0, 1, NaN, true, false.
        let e = homura::zeros([3, 0]).to(dtype);
        let eg = e.to(GPU);
        assert_bitwise(&eg.sum_dim(1, false), &e.sum_dim(1, false), "empty sum");
        assert_bitwise(&eg.prod_dim(1, false), &e.prod_dim(1, false), "empty prod");
        assert_bitwise(&eg.mean_dim(1, false), &e.mean_dim(1, false), "empty mean");
        assert_bitwise(
            &eg.var_dim(1, 0, false),
            &e.var_dim(1, 0, false),
            "empty var",
        );
        assert_bitwise(&eg.all_dim(1, false), &e.all_dim(1, false), "empty all");
        assert_bitwise(&eg.any_dim(1, false), &e.any_dim(1, false), "empty any");
        // NaN, +Inf, -Inf and both infinities, among small finite values.
        let inf = f64::INFINITY;
        let special: [&[f64]; 5] = [
            &[1.0, f64::NAN, 2.0, 3.0],
            &[1.0, inf, 2.0, 3.0],
            &[1.0, -inf, 2.0, 3.0],
            &[inf, 1.0, -inf, 3.0],
            &[0.0, inf, 2.0, 3.0],
        ];
        let x = Tensor::from_slice(&special.concat(), [5, 4]).to(dtype);
        let g = x.to(GPU);
        for (name, gv, cv) in [
            ("sum", g.sum_dim(1, false), x.sum_dim(1, false)),
            ("mean", g.mean_dim(1, false), x.mean_dim(1, false)),
            ("prod", g.prod_dim(1, false), x.prod_dim(1, false)),
            ("var", g.var_dim(1, 1, false), x.var_dim(1, 1, false)),
        ] {
            assert_bitwise(&gv, &cv, &format!("{dtype} {name} of special values"));
        }
    }
}

/// Softmax rows of -Inf only, with a NaN, with +Inf, of large magnitude, and of one element.
#[test]
fn softmax_extreme_rows_match_the_cpu() {
    for dtype in FLOATS {
        let big = match dtype {
            DType::F16 => 6e4,
            DType::BF16 | DType::F32 => 3e38,
            _ => 1e300,
        };
        let inf = f64::INFINITY;
        let rows: [&[f64]; 6] = [
            &[-inf, -inf, -inf, -inf],
            &[1.0, f64::NAN, 2.0, 3.0],
            &[1.0, inf, 2.0, 3.0],
            &[big, -big, big, 0.0],
            &[-big, -big, 1.0, -inf],
            &[0.0, -0.0, 0.0, -0.0],
        ];
        let x = Tensor::from_slice(&rows.concat(), [6, 4]).to(dtype);
        let g = x.to(GPU);
        for log in [false, true] {
            let (gv, cv) = if log {
                (g.log_softmax(1), x.log_softmax(1))
            } else {
                (g.softmax(1), x.softmax(1))
            };
            check_softmax(&gv, &cv, &x.to(DType::F64), 1, log)
                .unwrap_or_else(|e| panic!("{dtype} log {log}: {e}"));
        }
        let one = Tensor::from_slice(&[3.0, -inf, f64::NAN], [3, 1]).to(dtype);
        assert_bitwise(&one.to(GPU).softmax(1), &one.softmax(1), "one-element rows");
        assert_bitwise(
            &one.to(GPU).log_softmax(1),
            &one.log_softmax(1),
            "one-element rows",
        );
    }
}

// ---- Plans ---------------------------------------------------------------------------------

/// Shapes and reduced dimensions that reach every plan of `csrc/reduce.hpp` (the unit test
/// `reduction_plans_cover_every_path` in `backend::rocm::tests` checks the plans of these
/// shapes): 256 lanes per output with and without chunks (chunks inside and across rows), 32
/// lanes with and without chunks (rows cut by chunks), one lane per output with and without
/// chunks (and with reduced rows of several dimensions), and grids of more than one round.
pub const PLAN_SHAPES: [(&[usize], &[i64]); 11] = [
    (&[300, 512], &[1]),
    (&[4, 100000], &[1]),
    (&[9, 2, 3000], &[0, 2]),
    (&[5000, 100], &[1]),
    (&[64, 3, 100], &[0, 2]),
    (&[200000, 2], &[1]),
    (&[100000, 3], &[0]),
    (&[7, 9, 11], &[0, 2]),
    (&[70000, 256], &[1]),
    (&[(1 << 24) + 1000, 2], &[1]),
    (&[5, 1], &[1]),
];

/// Every plan: integer sums of random words (any missing or repeated element changes them),
/// float sums and variances within their classes, and max_dim / argmax over values with many
/// ties (the first position must win across lanes and chunks).
#[test]
fn every_plan_folds_each_element_once() {
    for (shape, dims) in PLAN_SHAPES {
        let n: usize = shape.iter().product();
        let what = format!("{shape:?} dims {dims:?}");
        let words = values(DType::I64, n, 91).reshape(shape);
        let wg = words.to(GPU);
        assert_bitwise(
            &wg.sum_dim(dims, false),
            &words.sum_dim(dims, false),
            &format!("I64 sum {what}"),
        );
        for dtype in [DType::F32, DType::F64] {
            let x = finite(dtype, n, 6, 92).reshape(shape);
            let g = x.to(GPU);
            let abs = x.to(DType::F64).abs().sum_dim(dims, false);
            let cpu = x.sum_dim(dims, false);
            let k = folded(&x, &cpu);
            check_sum(&g.sum_dim(dims, false), &cpu, &abs, k, false)
                .unwrap_or_else(|e| panic!("{dtype} sum {what}: {e}"));
            let v64 = x.to(DType::F64).var_dim(dims, 1, false);
            if k > 1 {
                check_var(
                    &g.var_dim(dims, 1, false),
                    &x.var_dim(dims, 1, false),
                    &v64,
                    k,
                    false,
                )
                .unwrap_or_else(|e| panic!("{dtype} var {what}: {e}"));
            }
        }
        if dims.len() == 1 {
            // Small integers: the maximum is repeated many times along every row.
            let t: Vec<f32> = (0..n).map(|i| ((i * 2654435761) % 7) as f32).collect();
            let x = Tensor::from_slice(&t, [n]).reshape(shape);
            let g = x.to(GPU);
            let (gv, cv) = (g.max_dim(dims[0], false), x.max_dim(dims[0], false));
            assert_bitwise(&gv.values, &cv.values, &format!("max_dim {what}"));
            assert_bitwise(&gv.indices, &cv.indices, &format!("max_dim indices {what}"));
            assert_bitwise(
                &g.argmin_dim(dims[0], false),
                &x.argmin_dim(dims[0], false),
                &format!("argmin_dim {what}"),
            );
            check_softmax(
                &g.softmax(dims[0]),
                &x.softmax(dims[0]),
                &x.to(DType::F64),
                dims[0],
                false,
            )
            .unwrap_or_else(|e| panic!("softmax {what}: {e}"));
        }
    }
}

/// Shapes and dimensions that reach every softmax plan (the unit test
/// `softmax_plans_cover_every_path` checks them): rows in registers (16, 8, 4 elements per lane;
/// 256, 32 and 1 lanes per row), rows too long for the registers, strided rows, and few long
/// rows split across blocks, contiguous and strided.
const SOFTMAX_PLAN_SHAPES: [(&[usize], i64); 10] = [
    (&[300, 4096], 1),
    (&[300, 4097], 1),
    (&[3000, 1000], 1),
    (&[5000, 100], 1),
    (&[5000, 255], 1),
    (&[100000, 16], 1),
    (&[20, 7], 0),
    (&[4, 100000], 1),
    (&[100000, 3], 0),
    (&[1, 8192], 1),
];

/// Every softmax plan within the softmax class, for every float dtype and both forms, also with
/// a NaN, an infinity and a row of -Inf among the rows.
#[test]
fn every_softmax_plan_is_within_the_class() {
    for (shape, dim) in SOFTMAX_PLAN_SHAPES {
        let n: usize = shape.iter().product();
        let row = shape[dim as usize];
        for dtype in FLOATS {
            let x = finite(dtype, n, 3, 111).reshape(shape);
            // Special values: a NaN in the first row, +Inf in the second, a row of -Inf third
            // (rows along `dim`).
            let mut v = floats(&x.transpose(dim, -1).contiguous());
            v[0] = f64::NAN;
            if n / row > 1 {
                v[row + 1] = f64::INFINITY;
            }
            if n / row > 2 {
                for e in &mut v[2 * row..3 * row] {
                    *e = f64::NEG_INFINITY;
                }
            }
            let mut tshape = shape.to_vec();
            tshape.swap(dim as usize, shape.len() - 1);
            let special = Tensor::from_slice(&v, &tshape[..])
                .to(dtype)
                .transpose(dim, -1);
            for (name, x) in [("finite", x), ("special", special)] {
                let g = x.to(GPU);
                for log in [false, true] {
                    let (gv, cv) = if log {
                        (g.log_softmax(dim), x.log_softmax(dim))
                    } else {
                        (g.softmax(dim), x.softmax(dim))
                    };
                    check_softmax(&gv, &cv, &x.to(DType::F64), dim, log).unwrap_or_else(|e| {
                        panic!("{dtype} {shape:?} dim {dim} {name} log {log}: {e}")
                    });
                }
            }
        }
    }
}

/// A reduction gives the same bits every time it runs (the plan depends only on the shape).
#[test]
fn reductions_are_reproducible() {
    let x = finite(DType::F32, 4 * 100000, 8, 101)
        .reshape([4, 100000])
        .to(GPU);
    let y = finite(DType::F64, 100000 * 3, 8, 102)
        .reshape([100000, 3])
        .to(GPU);
    let first = (x.sum_dim(1, false), y.var_dim(0, 1, false), x.softmax(1));
    for _ in 0..3 {
        assert_bitwise(&x.sum_dim(1, false), &first.0.to(Device::Cpu), "sum");
        assert_bitwise(&y.var_dim(0, 1, false), &first.1.to(Device::Cpu), "var");
        assert_bitwise(&x.softmax(1), &first.2.to(Device::Cpu), "softmax");
    }
}

// ---- The judges reject wrong results -------------------------------------------------------

/// Wrong results that the judges must reject (M4.12: zeroed outputs, a missing element, a
/// wrong divisor), next to the right ones, which pass.
#[test]
fn the_judges_reject_wrong_results() {
    let on_gpu = |v: &[f64], dtype: DType| Tensor::from_slice(v, [v.len()]).to(dtype).to(GPU);
    // F16 sums: 64 ones; dropping one gives 63, one F16 ulp at 64 is 1/16.
    let ones = homura::ones([1, 64]).to(DType::F16);
    let cpu = ones.sum_dim(1, false);
    let abs = ones.to(DType::F64).sum_dim(1, false);
    assert!(check_sum(&on_gpu(&[64.0], DType::F16), &cpu, &abs, 64, false).is_ok());
    assert!(check_sum(&on_gpu(&[63.0], DType::F16), &cpu, &abs, 64, false).is_err());
    assert!(check_sum(&on_gpu(&[0.0], DType::F16), &cpu, &abs, 64, false).is_err());
    // F16, 1024 ones: the bound is about 1 (the example of M4.12), so 0 fails.
    let ones = homura::ones([1, 1024]).to(DType::F16);
    let (cpu, abs) = (
        ones.sum_dim(1, false),
        ones.to(DType::F64).sum_dim(1, false),
    );
    assert!(check_sum(&on_gpu(&[0.0], DType::F16), &cpu, &abs, 1024, false).is_err());
    // A NaN, or an infinity, for a finite sum; sums of two outputs in each other's places.
    assert!(check_sum(&on_gpu(&[f64::NAN], DType::F16), &cpu, &abs, 1024, false).is_err());
    assert!(
        check_sum(
            &on_gpu(&[f64::INFINITY], DType::F16),
            &cpu,
            &abs,
            1024,
            false
        )
        .is_err()
    );
    let two = Tensor::from_slice(&[1.0, 2.0, 30.0, 40.0], [2, 2]);
    let (cpu2, abs2) = (two.sum_dim(1, false), two.abs().sum_dim(1, false));
    assert!(check_sum(&on_gpu(&[3.0, 70.0], DType::F64), &cpu2, &abs2, 2, false).is_ok());
    assert!(check_sum(&on_gpu(&[70.0, 3.0], DType::F64), &cpu2, &abs2, 2, false).is_err());
    // Near the bound: F64 [1e17, 1, -1e17] sums to 0 on the CPU, the bound before rounding is
    // 6 eps (2e17 + 1), about 133: 100 passes, 400 fails.
    let c = Tensor::from_slice(&[1e17, 1.0, -1e17], [1, 3]);
    let (cpu3, abs3) = (c.sum_dim(1, false), c.abs().sum_dim(1, false));
    assert!(check_sum(&on_gpu(&[100.0], DType::F64), &cpu3, &abs3, 3, false).is_ok());
    assert!(check_sum(&on_gpu(&[400.0], DType::F64), &cpu3, &abs3, 3, false).is_err());
    // F64 sum missing its smallest element.
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 0.125], [1, 4]);
    let (cpu, abs) = (x.sum_dim(1, false), x.abs().sum_dim(1, false));
    assert!(check_sum(&on_gpu(&[6.0], DType::F64), &cpu, &abs, 4, false).is_err());
    assert!(check_sum(&on_gpu(&[6.125], DType::F64), &cpu, &abs, 4, false).is_ok());
    // Products: one factor missing.
    let cpu = x.prod_dim(1, false);
    assert!(check_prod(&on_gpu(&[6.0], DType::F64), &cpu, 4).is_err());
    assert!(check_prod(&on_gpu(&[0.75], DType::F64), &cpu, 4).is_ok());
    // Variance divided by n instead of n - 1, and a zeroed one.
    let x = Tensor::from_slice(&[1.0, 2.0, 4.0, 8.0], [1, 4]).to(DType::F32);
    let v64 = x.to(DType::F64).var_dim(1, 1, false);
    let cpu = x.var_dim(1, 1, false);
    let biased = x.var_dim(1, 0, false).to(GPU);
    assert!(check_var(&cpu.to(GPU), &cpu, &v64, 4, false).is_ok());
    assert!(check_var(&biased, &cpu, &v64, 4, false).is_err());
    assert!(check_var(&on_gpu(&[0.0], DType::F32), &cpu, &v64, 4, false).is_err());
    // Near the bound: F64 var of [1, 2, 4, 8] is about 9.58, Δ = 16 eps v is about 1.7e-14;
    // 1e-14 off passes, 1e-13 off fails (the std rule's √Δ would accept 1e-7).
    let y = Tensor::from_slice(&[1.0, 2.0, 4.0, 8.0], [1, 4]);
    let (vy, cy) = (y.var_dim(1, 1, false), y.var_dim(1, 1, false));
    let off = |d: f64| on_gpu(&[cy.item::<f64>() + d], DType::F64);
    assert!(check_var(&off(1e-14), &cy, &vy, 4, false).is_ok());
    assert!(check_var(&off(1e-13), &cy, &vy, 4, false).is_err());
    let cpu = x.std_dim(1, 1, false);
    assert!(check_var(&x.std_dim(1, 0, false).to(GPU), &cpu, &v64, 4, true).is_err());
    // Softmax: uniform probabilities, a zeroed output, a row that doesn't sum to 1.
    let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], [1, 4]).to(DType::F32);
    let x64 = x.to(DType::F64);
    let cpu = x.softmax(1);
    assert!(check_softmax(&cpu.to(GPU), &cpu, &x64, 1, false).is_ok());
    let uniform = homura::full([1, 4], 0.25).to(DType::F32).to(GPU);
    assert!(check_softmax(&uniform, &cpu, &x64, 1, false).is_err());
    assert!(check_softmax(&cpu.mul(0.0).to(GPU), &cpu, &x64, 1, false).is_err());
    let cpu = x.log_softmax(1);
    assert!(check_softmax(&cpu.to(GPU), &cpu, &x64, 1, true).is_ok());
    assert!(check_softmax(&x.sub(3.0).to(GPU), &cpu, &x64, 1, true).is_err());
}
