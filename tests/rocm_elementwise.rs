//! Elementwise operations on a ROCm GPU against the CPU (M4 step 3): the bitwise class for the
//! IEEE operations, integers, comparisons, logic and data selection, the transcendental class of
//! M4.12 for the device library's functions, with their ulp statistics printed (run with
//! `--nocapture` to see them).
#![cfg(feature = "rocm")]

#[path = "common/rocm.rs"]
mod rocm;

use homura::{DType, GeluApproximate, Tensor};
use rocm::{
    DTYPES, GPU, Pair, UlpStats, Words, assert_bitwise, check_transcendental, floats,
    large_layouts, layouts, pow2, ulp, unit_roundoff, values,
};

const FLOATS: [DType; 4] = [DType::F16, DType::BF16, DType::F32, DType::F64];

fn float_tensor(v: &[f64], dtype: DType) -> Tensor {
    Tensor::from_slice(v, [v.len()]).to(dtype)
}

/// Uniform values in `[-r, r]`.
fn uniform(n: usize, r: f64, seed: u64) -> Vec<f64> {
    let mut w = Words::new(seed);
    (0..n)
        .map(|_| ((w.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * r)
        .collect()
}

/// Values of magnitude below 2^-10, subnormals of every float dtype included.
fn near_zero(n: usize, seed: u64) -> Vec<f64> {
    let mut w = Words::new(seed);
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            let r = w.next();
            let e = -10 - (r % 1070) as i32;
            let m = 1.0 + (r >> 20) as f64 / (1u64 << 44) as f64;
            let s = if r & (1 << 12) != 0 { -1.0 } else { 1.0 };
            s * m * pow2(e.max(-1074))
        })
        .collect();
    v.extend([0.0, -0.0, 5.96e-8, 1e-40, -1e-45, pow2(-1074)]);
    v
}

/// The inputs a transcendental function is compared on: special values and random bits in
/// every layout, uniform values at three scales, values near zero, and `tails`.
fn domains(dtype: DType, tails: &[f64], seed: u64) -> Vec<(String, Tensor)> {
    let mut d: Vec<(String, Tensor)> = layouts(dtype, seed)
        .into_iter()
        .map(|p| (p.name.to_string(), p.cpu))
        .collect();
    for r in [1.0, 10.0, 100.0] {
        d.push((
            format!("uniform ±{r}"),
            float_tensor(&uniform(4096, r, seed + 1), dtype),
        ));
    }
    d.push((
        "near zero".into(),
        float_tensor(&near_zero(2048, seed + 2), dtype),
    ));
    d.push(("tails".into(), float_tensor(tails, dtype)));
    d
}

/// Inputs of a half dtype whose result, computed in f32 on the CPU, lies within one f32 ulp of
/// a rounding midpoint of the dtype: there the two sides may round to neighbouring values.
fn midpoint_inputs(dtype: DType, f: &dyn Fn(&Tensor) -> Tensor, seed: u64) -> Option<Tensor> {
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return None;
    }
    let x = float_tensor(&uniform(1 << 16, 8.0, seed), dtype);
    let y32 = floats(&f(&x.to(DType::F32)));
    let xs = floats(&x);
    let picked: Vec<f64> = xs
        .iter()
        .zip(&y32)
        .filter(|&(_, &y)| {
            if !y.is_finite() || y == 0.0 {
                return false;
            }
            let rounded = float_tensor(&[y], dtype);
            let r = floats(&rounded)[0];
            let half_ulp = ulp(dtype, y) / 2.0;
            let mid = if y > r { r + half_ulp } else { r - half_ulp };
            (y - mid).abs() <= ulp(DType::F32, y)
        })
        .map(|(&x, _)| x)
        .collect();
    (!picked.is_empty()).then(|| float_tensor(&picked, dtype))
}

type Op = fn(&Tensor) -> Tensor;
/// The size of the largest intermediate at input `x` with the CPU's result `y`.
type Magnitude = fn(f64, f64) -> f64;

fn result_size(_: f64, y: f64) -> f64 {
    y.abs()
}

fn erf_f64(x: f64) -> f64 {
    floats(&float_tensor(&[x], DType::F64).erf())[0]
}

/// Compares one transcendental op on every domain and float dtype; returns the statistics per
/// dtype.
fn compare_transcendental(name: &str, f: Op, m: Magnitude, k: f64, tails: &[f64]) {
    for dtype in FLOATS {
        let mut stats = UlpStats::default();
        let mut inputs = domains(dtype, tails, 11);
        if let Some(mid) = midpoint_inputs(dtype, &f, 12) {
            inputs.push(("midpoints".into(), mid));
        }
        for (what, x) in inputs {
            let cpu = f(&x);
            let gpu = f(&x.to(GPU));
            let xs = floats(&x.contiguous());
            let ys = floats(&cpu);
            let ms: Vec<f64> = xs.iter().zip(&ys).map(|(&x, &y)| m(x, y)).collect();
            match check_transcendental(&gpu, &cpu, &ms, k) {
                Ok(s) => stats.merge(s),
                Err(e) => panic!("{name} {dtype} {what}: {e}"),
            }
        }
        println!(
            "ulp {name:<12} {dtype:<5} k={k}: max {} ulp, {} of {} elements differ",
            stats.max_ulp, stats.differ, stats.elements
        );
    }
}

#[test]
fn ieee_unary_ops_are_bitwise() {
    let ops: [(&str, Op); 8] = [
        ("neg", |t| t.neg()),
        ("abs", |t| t.abs()),
        ("sign", |t| t.sign()),
        ("relu", |t| t.relu()),
        ("sqrt", |t| t.sqrt()),
        ("rsqrt", |t| t.rsqrt()),
        ("reciprocal", |t| t.reciprocal()),
        ("logical_not", |t| t.logical_not()),
    ];
    for dtype in DTYPES {
        for p in layouts(dtype, 1) {
            for (name, f) in ops {
                // Integer and bool inputs of the float ops promote, as on the CPU.
                let unsupported = dtype == DType::Bool && matches!(name, "neg" | "abs" | "relu");
                if unsupported {
                    continue;
                }
                assert_bitwise(
                    &f(&p.gpu),
                    &f(&p.cpu),
                    &format!("{name} {dtype} {}", p.name),
                );
            }
        }
    }
    for p in large_layouts(DType::F32, 2) {
        assert_bitwise(&p.gpu.sqrt(), &p.cpu.sqrt(), p.name);
        assert_bitwise(&p.gpu.neg(), &p.cpu.neg(), p.name);
    }
}

#[test]
fn transcendental_unary_ops_stay_within_the_bounds() {
    let exp_tails = [
        88.7, 88.72, 88.73, -87.3, -103.9, -104.0, 709.7, 709.8, -745.1, 11.09, -17.3,
    ];
    compare_transcendental("exp", |t| t.exp(), result_size, 2.0, &exp_tails);
    compare_transcendental(
        "log",
        |t| t.log(),
        result_size,
        2.0,
        &[1e-300, 1e300, 1.0 + 1e-7],
    );
    compare_transcendental(
        "sin",
        |t| t.sin(),
        result_size,
        2.0,
        &[std::f64::consts::PI, 1e6, 1e20, 355.0],
    );
    compare_transcendental(
        "cos",
        |t| t.cos(),
        result_size,
        2.0,
        &[std::f64::consts::FRAC_PI_2, 1e6, 1e20, 33.0],
    );
    compare_transcendental(
        "tanh",
        |t| t.tanh(),
        result_size,
        2.0,
        &[9.0, 20.0, -20.0, 1e-5],
    );
    let erf_tails = [5.9, 6.0, 6.1, -7.0, 1.49999, 1.5, 1.50001, 3.0];
    compare_transcendental("erf", |t| t.erf(), result_size, 2.0, &erf_tails);
    let sig_tails = [-30.0, -88.0, -104.0, 30.0, 40.0, -745.0];
    compare_transcendental("sigmoid", |t| t.sigmoid(), result_size, 4.0, &sig_tails);
    compare_transcendental(
        "silu",
        |t| t.silu(),
        |x, _| x.abs() / (1.0 + (-x).exp()),
        4.0,
        &sig_tails,
    );
    compare_transcendental(
        "log_sigmoid",
        |t| t.log_sigmoid(),
        |x, _| x.min(0.0).abs() + (-x.abs()).exp().ln_1p(),
        6.0,
        &[-30.0, -31.0, -1e4, 30.0, 31.0, 40.0, 1e4],
    );
    compare_transcendental(
        "gelu",
        |t| t.gelu(GeluApproximate::None),
        |x, _| 0.5 * x.abs() * (1.0 + erf_f64(x * std::f64::consts::FRAC_1_SQRT_2).abs()),
        4.0,
        &[-10.0, -10.5, -12.0, -30.0, 10.0],
    );
    compare_transcendental(
        "gelu_tanh",
        |t| t.gelu(GeluApproximate::Tanh),
        |x, _| {
            let k = (2.0 / std::f64::consts::PI).sqrt();
            0.5 * x.abs() * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh().abs())
        },
        4.0,
        &[-10.0, -10.5, -12.0, -30.0, 10.0],
    );
}

#[test]
fn the_transcendental_bound_rejects_wrong_results() {
    // M4.12's negative examples: computing in a narrower type, dropping a factor of a
    // composition, and cos for sin, must all fail.
    let x = float_tensor(&uniform(4096, 10.0, 21), DType::F64);
    let g = x.to(GPU);
    let m: Vec<f64> = floats(&x.exp()).iter().map(|v| v.abs()).collect();
    let narrow = x.to(DType::F32).exp().to(DType::F64);
    assert!(check_transcendental(&g.exp(), &narrow, &m, 2.0).is_err());
    let gelu = x.gelu(GeluApproximate::None);
    let mg: Vec<f64> = floats(&gelu).iter().map(|v| 2.0 * v.abs() + 1.0).collect();
    let doubled = gelu.mul(2.0);
    assert!(check_transcendental(&g.gelu(GeluApproximate::None), &doubled, &mg, 4.0).is_err());
    let ms: Vec<f64> = vec![1.0; 4096];
    assert!(check_transcendental(&g.sin(), &x.cos(), &ms, 2.0).is_err());
    // Near the bound: 64 ulp off in F64 fails (the bound for exp is about 2 ulp), so a bound
    // that is much too loose would be noticed.
    let off = x.exp().mul(1.0 + 64.0 * f64::EPSILON);
    assert!(check_transcendental(&g.exp(), &off, &m, 2.0).is_err());
    // A NaN against a finite value, and an infinity against a finite value, fail whatever the
    // bound: the classification must agree.
    let y = float_tensor(&[-1.0, 1e300, 2.0], DType::F64);
    let finite = float_tensor(&[0.0, 1e300, std::f64::consts::LN_2], DType::F64);
    let big = vec![1e308; 3];
    assert!(check_transcendental(&y.to(GPU).log(), &finite, &big, 2.0).is_err());
    let finite = float_tensor(&[f64::NAN, 1e300, 2f64.exp()], DType::F64);
    assert!(check_transcendental(&y.to(GPU).exp(), &finite, &big, 2.0).is_err());
    // And the right results pass.
    assert!(check_transcendental(&g.exp(), &x.exp(), &m, 2.0).is_ok());
}

/// The second operand for `p`: other values in the same layout.
fn partner(dtype: DType, seed: u64) -> Vec<Pair> {
    layouts(dtype, seed)
}

#[test]
fn binary_ops_are_bitwise() {
    type BinOp = fn(&Tensor, &Tensor) -> Tensor;
    let ops: [(&str, BinOp); 13] = [
        ("add", |a, b| a.add(b)),
        ("sub", |a, b| a.sub(b)),
        ("mul", |a, b| a.mul(b)),
        ("div", |a, b| a.div(b)),
        ("maximum", |a, b| a.maximum(b)),
        ("minimum", |a, b| a.minimum(b)),
        ("greater", |a, b| a.greater(b)),
        ("greater_equal", |a, b| a.greater_equal(b)),
        ("less", |a, b| a.less(b)),
        ("less_equal", |a, b| a.less_equal(b)),
        ("equal", |a, b| a.equal(b)),
        ("not_equal", |a, b| a.not_equal(b)),
        ("logical_and", |a, b| a.logical_and(b)),
    ];
    for dtype in DTYPES {
        let bs = partner(dtype, 2);
        for (p, q) in layouts(dtype, 1).iter().zip(&bs) {
            for (name, f) in ops {
                if dtype == DType::Bool && matches!(name, "sub" | "div") {
                    continue;
                }
                let what = format!("{name} {dtype} {}", p.name);
                assert_bitwise(&f(&p.gpu, &q.gpu), &f(&p.cpu, &q.cpu), &what);
            }
            let what = format!("logical_or {dtype} {}", p.name);
            assert_bitwise(&p.gpu.logical_or(&q.gpu), &p.cpu.logical_or(&q.cpu), &what);
            // The same tensor on both sides.
            assert_bitwise(&p.gpu.add(&p.gpu), &p.cpu.add(&p.cpu), &what);
        }
        // Broadcasting against a row and a column, and promotion with another dtype.
        let a = values(dtype, 15, 3).reshape([3, 5]);
        for b in [values(dtype, 5, 4), values(dtype, 3, 5).reshape([3, 1])] {
            for (name, f) in ops {
                if dtype == DType::Bool && matches!(name, "sub" | "div") {
                    continue;
                }
                let what = format!("{name} {dtype} broadcast {:?}", b.shape());
                assert_bitwise(&f(&a.to(GPU), &b.to(GPU)), &f(&a, &b), &what);
            }
        }
        for other in [DType::I32, DType::F16, DType::F64] {
            let b = values(other, 15, 6).reshape([3, 5]);
            let what = format!("mul {dtype} * {other}");
            assert_bitwise(&a.to(GPU).mul(b.to(GPU)), &a.mul(&b), &what);
        }
    }
    for (p, q) in large_layouts(DType::F32, 7)
        .iter()
        .zip(&large_layouts(DType::F32, 8))
    {
        assert_bitwise(&p.gpu.mul(&q.gpu), &p.cpu.mul(&q.cpu), p.name);
    }
}

#[test]
fn scalar_operands_are_bitwise() {
    // F16 and BF16 products and quotients keep the scalar in f32 (binary_scalar); the other
    // operations round it to the dtype.
    let scalars = [
        2.5,
        1.0 / 3.0,
        1e-5,
        70000.0,
        -0.0,
        f64::INFINITY,
        f64::NAN,
        1e300,
    ];
    for dtype in DTYPES {
        for p in layouts(dtype, 9) {
            for s in scalars {
                if dtype.is_floating_point() || s.is_finite() && s.fract() == 0.0 {
                    let what = format!("{dtype} {} with {s}", p.name);
                    assert_bitwise(&p.gpu.mul(s), &p.cpu.mul(s), &format!("mul {what}"));
                    assert_bitwise(&p.gpu.div(s), &p.cpu.div(s), &format!("div {what}"));
                    assert_bitwise(&p.gpu.add(s), &p.cpu.add(s), &format!("add {what}"));
                    assert_bitwise(&p.gpu.maximum(s), &p.cpu.maximum(s), &format!("max {what}"));
                    assert_bitwise(&p.gpu.less(s), &p.cpu.less(s), &format!("less {what}"));
                }
            }
        }
    }
}

#[test]
fn integer_powers_are_bitwise_and_float_powers_within_the_bound() {
    for dtype in [DType::U8, DType::I8, DType::I16, DType::I32, DType::I64] {
        let base = values(dtype, 200, 13);
        // Exponents from -3 to 70, so that the loop and the negative rules both run.
        let e: Vec<i64> = (0..200).map(|i| i % 74 - 3).collect();
        let e = Tensor::from_slice(&e, [200]).to(dtype);
        let what = format!("pow {dtype}");
        assert_bitwise(&base.to(GPU).pow(e.to(GPU)), &base.pow(&e), &what);
        assert_bitwise(&base.to(GPU).pow(3), &base.pow(3), &what);
    }
    for dtype in FLOATS {
        let mut stats = UlpStats::default();
        let bases = [
            values(dtype, 1031, 14),
            float_tensor(&uniform(1031, 10.0, 15), dtype).abs(),
            float_tensor(&uniform(1031, 10.0, 16), dtype),
        ];
        let exps = [
            values(dtype, 1031, 17),
            float_tensor(&uniform(1031, 4.0, 18), dtype),
            float_tensor(&uniform(1031, 4.0, 19), dtype).round_to_integers(),
        ];
        for (a, b) in bases.iter().zip(&exps) {
            let cpu = a.pow(b);
            let m: Vec<f64> = floats(&cpu).iter().map(|v| v.abs()).collect();
            match check_transcendental(&a.to(GPU).pow(b.to(GPU)), &cpu, &m, 2.0) {
                Ok(s) => stats.merge(s),
                Err(e) => panic!("pow {dtype}: {e}"),
            }
        }
        println!(
            "ulp {:<12} {dtype:<5} k=2: max {} ulp, {} of {} elements differ",
            "pow", stats.max_ulp, stats.differ, stats.elements
        );
    }
}

trait RoundToIntegers {
    fn round_to_integers(&self) -> Tensor;
}

impl RoundToIntegers for Tensor {
    /// Rounds toward zero through I64 (integer exponents).
    fn round_to_integers(&self) -> Tensor {
        self.to(DType::I64).to(self.dtype())
    }
}

#[test]
fn where_and_masked_fill_are_bitwise() {
    for dtype in DTYPES {
        let conds = layouts(DType::Bool, 31);
        let ys = layouts(dtype, 32);
        for ((p, c), y) in layouts(dtype, 30).iter().zip(&conds).zip(&ys) {
            let what = format!("where {dtype} {}", p.name);
            let gpu = homura::where_(&c.gpu, &p.gpu, &y.gpu);
            assert_bitwise(&gpu, &homura::where_(&c.cpu, &p.cpu, &y.cpu), &what);
            let what = format!("masked_fill {dtype} {}", p.name);
            let gpu = p.gpu.masked_fill(&c.gpu, 1);
            assert_bitwise(&gpu, &p.cpu.masked_fill(&c.cpu, 1), &what);
            let v = Tensor::from_slice(&[7i64], [1]).reshape(());
            let gpu = p.gpu.masked_fill(&c.gpu, v.to(GPU));
            assert_bitwise(&gpu, &p.cpu.masked_fill(&c.cpu, &v), &what);
        }
    }
}

/// The panic message of `f`.
fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).expect_err("expected a panic");
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn masked_fill_values_are_checked_on_the_gpu_as_on_the_cpu() {
    let x = Tensor::from_slice(&[1i8, 2, 3], [3]);
    let mask = Tensor::from_slice(&[true, false, true], [3]);
    let value = Tensor::from_slice(&[300i64], [1]).reshape(());
    for (cpu, gpu) in [
        (
            panic_message(|| drop(x.masked_fill(&mask, 300))),
            panic_message(|| drop(x.to(GPU).masked_fill(&mask.to(GPU), 300))),
        ),
        (
            panic_message(|| drop(x.masked_fill(&mask, &value))),
            panic_message(|| drop(x.to(GPU).masked_fill(&mask.to(GPU), value.to(GPU)))),
        ),
    ] {
        assert!(!cpu.is_empty());
        assert_eq!(gpu, cpu);
    }
}

#[test]
fn triangles_are_bitwise() {
    for dtype in DTYPES {
        for p in layouts(dtype, 40) {
            if p.cpu.num_dims() < 2 {
                continue;
            }
            for diagonal in [-3, -1, 0, 1, 2, 300] {
                let what = format!("{dtype} {} diagonal {diagonal}", p.name);
                assert_bitwise(
                    &p.gpu.triu(diagonal),
                    &p.cpu.triu(diagonal),
                    &format!("triu {what}"),
                );
                assert_bitwise(
                    &p.gpu.tril(diagonal),
                    &p.cpu.tril(diagonal),
                    &format!("tril {what}"),
                );
            }
        }
    }
}

#[test]
fn flips_are_bitwise() {
    for dtype in DTYPES {
        for p in layouts(dtype, 50) {
            let n = p.cpu.num_dims() as i64;
            let mut choices: Vec<Vec<i64>> = (0..n).map(|d| vec![d]).collect();
            choices.push((0..n).collect());
            if n >= 2 {
                choices.push(vec![0, n - 1]);
                choices.push(vec![n - 2, n - 1]);
            }
            for dims in choices {
                let what = format!("flip {dtype} {} {dims:?}", p.name);
                assert_bitwise(&p.gpu.flip(&dims[..]), &p.cpu.flip(&dims[..]), &what);
            }
        }
    }
    for p in large_layouts(DType::I16, 51) {
        assert_bitwise(&p.gpu.flip([0]), &p.cpu.flip([0]), p.name);
    }
}

#[test]
fn in_place_ops_are_bitwise() {
    for dtype in [DType::I32, DType::F16, DType::F32, DType::F64] {
        let qs = layouts(dtype, 61);
        // Each layout views a base of its own, so writing into it changes nothing else.
        for (p, q) in layouts(dtype, 60).iter().zip(&qs) {
            if p.cpu.stride().contains(&0) {
                continue; // in-place writes into a broadcast view are refused on both devices
            }
            p.cpu.add_(&q.cpu).mul_(&p.cpu).sub_(1);
            p.gpu.add_(&q.gpu).mul_(&p.gpu).sub_(1);
            assert_bitwise(&p.gpu, &p.cpu, &format!("in place {dtype} {}", p.name));
            p.cpu.clamp_(-1, 1);
            p.gpu.clamp_(-1, 1);
            assert_bitwise(&p.gpu, &p.cpu, &format!("clamp_ {dtype} {}", p.name));
        }
    }
}

#[test]
fn unit_roundoff_matches_the_computation_types() {
    assert_eq!(unit_roundoff(DType::F16), unit_roundoff(DType::F32));
    assert_eq!(unit_roundoff(DType::F64), pow2(-53));
    assert_eq!(pow2(-1074), f64::from_bits(1));
    assert_eq!(pow2(-1022), f64::MIN_POSITIVE);
    assert_eq!(pow2(1023), f64::MAX / (2.0 - f64::EPSILON));
}

/// The spacing of every float dtype on both sides of powers of two, at the smallest normal
/// value, zero, the smallest subnormal and the largest finite value (review round 126, R151).
#[test]
fn ulp_is_the_spacing_of_the_dtype() {
    // (dtype, mantissa bits, smallest normal exponent, largest exponent)
    for (dtype, man, min_exp, max_exp) in [
        (DType::F16, 10, -14, 15),
        (DType::BF16, 7, -126, 127),
        (DType::F32, 23, -126, 127),
        (DType::F64, 52, -1022, 1023),
    ] {
        // The value of `dtype` just below `x`, a power of two above the smallest normal one.
        let below = |x: f64| x - pow2(x.log2().round() as i32 - 1 - man);
        for e in [min_exp + 1, -1, 0, 1, 7, max_exp] {
            let x = pow2(e);
            assert_eq!(ulp(dtype, x), pow2(e - man), "{dtype} 2^{e}");
            assert_eq!(ulp(dtype, -x), pow2(e - man), "{dtype} -2^{e}");
            let b = below(x);
            assert_eq!(
                floats(&float_tensor(&[b], dtype))[0],
                b,
                "{dtype} below 2^{e}"
            );
            assert_eq!(ulp(dtype, b), pow2(e - 1 - man), "{dtype} below 2^{e}");
        }
        let smallest = pow2(min_exp - man);
        for v in [pow2(min_exp), 0.0, -0.0, smallest, pow2(min_exp) - smallest] {
            assert_eq!(ulp(dtype, v), smallest, "{dtype} {v:e}");
        }
        let largest = pow2(max_exp) * (2.0 - pow2(-man));
        assert_eq!(ulp(dtype, largest), pow2(max_exp - man), "{dtype} largest");
    }
    // The reviewer's probes: 4 ulp below 128- in F64 exceeds the bound (k = 2, m = v) ...
    let v = f64::from_bits(0x405f_ffff_ffff_ffff);
    let cpu = float_tensor(&[v], DType::F64);
    let off = float_tensor(&[f64::from_bits(v.to_bits() - 4)], DType::F64).to(GPU);
    assert!(check_transcendental(&off, &cpu, &[v], 2.0).is_err());
    // ... and one output ulp above zero is within it, also for subnormals (m = 0).
    let zero = float_tensor(&[0.0], DType::F64);
    let tiny = float_tensor(&[pow2(-1074)], DType::F64).to(GPU);
    assert!(check_transcendental(&tiny, &zero, &[0.0], 2.0).is_ok());
    let two = float_tensor(&[pow2(-1073)], DType::F64).to(GPU);
    assert!(check_transcendental(&two, &zero, &[0.0], 2.0).is_err());
    // The values near zero keep their F64 subnormals in every build: none of the generated
    // ones (all at least 2^-1074 in magnitude) is 0.
    let v = near_zero(2048, 1);
    assert!(v[..2048].iter().all(|&x| x != 0.0));
    let subnormals = v
        .iter()
        .filter(|&&x| x != 0.0 && x.abs() < f64::MIN_POSITIVE);
    assert!(subnormals.count() > 50);
}

#[test]
fn contiguous_paths_handle_every_offset_and_tail() {
    // Contiguous operands whose first elements are aligned to four elements run four elements
    // per thread, the others one: every offset from 0 to 4 and every length up to 9 (each tail
    // length of the last group), with an output of another dtype too.
    for dtype in [DType::U8, DType::F16, DType::F32, DType::F64] {
        let a = values(dtype, 16, 70);
        let b = values(dtype, 16, 71);
        let (ag, bg) = (a.to(GPU), b.to(GPU));
        for offset in 0..=4 {
            for len in 0..=9 {
                let what = format!("{dtype} offset {offset} length {len}");
                let (x, y) = (a.narrow(0, offset, len), b.narrow(0, 4, len));
                let (xg, yg) = (ag.narrow(0, offset, len), bg.narrow(0, 4, len));
                assert_bitwise(&xg.add(&yg), &x.add(&y), &format!("add {what}"));
                assert_bitwise(&xg.neg(), &x.neg(), &format!("neg {what}"));
                assert_bitwise(&xg.less(&yg), &x.less(&y), &format!("less {what}"));
                let c = homura::where_(&x.less(&y), &x, &y);
                let cg = homura::where_(&xg.less(&yg), &xg, &yg);
                assert_bitwise(&cg, &c, &format!("where {what}"));
            }
        }
    }
}

#[test]
fn concurrent_submissions_give_the_serial_results() {
    // M4.12's robustness row: threads queueing kernels on one device at the same time get what
    // one thread gets.
    let a = values(DType::F32, 1 << 16, 80).to(GPU);
    let b = values(DType::F32, 1 << 16, 81).to(GPU);
    let work = |a: &Tensor, b: &Tensor| a.mul(b).add(a).sigmoid().maximum(b).flip([0]);
    let serial = work(&a, &b);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| s.spawn(|| (0..20).map(|_| work(&a, &b)).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            for r in h.join().unwrap() {
                assert_bitwise(&r, &serial, "concurrent");
            }
        }
    });
}
