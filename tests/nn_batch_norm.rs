//! M2 step 8c: `nn::functional::batch_norm`. Outputs, gradients and running statistics are
//! compared with `tests/golden/m2_batch_norm.safetensors` from
//! `tools/golden/gen_m2_batch_norm.py`; the reference behavior encoded in the other tests is
//! asserted by `tools/golden/check_m2_batch_norm.py`, in the same order.
//!
//! The floating point tolerances scale with the magnitude of what is computed: an element is
//! compared within `ulps` units in the last place of its dtype plus `k` units of the statistics'
//! dtype times a magnitude bound, the same formula evaluated in F64 with `|x| + |mean|` for the
//! centered input and absolute values for everything else (see [`bounds`]): the mean is rounded
//! (and PyTorch's sums run in other orders), which moves the centered values by about a unit of
//! `|mean|`. `tools/golden/measure_m2_tolerances.py` measures how much of these the errors use
//! (with `HOMURA_TOLERANCE_LOG` set, this test writes its measurements there).

mod common;

use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden, assert_close_tol};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::nn::functional::{BatchNormOptions, batch_norm};
use homura::nn::{LayerNorm, Module};
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

/// A variable of `shape` with deterministic values of both signs, between 0.3 and 1.3 in
/// magnitude.
fn var(shape: &[usize], seed: usize) -> Tensor {
    let x = constant(shape, seed);
    x.requires_grad_(true);
    x
}

/// The values of [`var`] without requiring grad.
fn constant(shape: &[usize], seed: usize) -> Tensor {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let magnitude = 0.3 + ((i * 7 + seed * 3) % 11) as f64 / 10.0;
            if (i + seed).is_multiple_of(3) {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect();
    Tensor::from_vec(values, shape)
}

/// Running statistics of `c` channels: means about ±0.5, variances in [0.6, 1.4].
fn stats(c: usize) -> (Tensor, Tensor) {
    let mean = constant(&[c], 5).mul(0.5);
    let var = constant(&[c], 6).abs().add(0.3);
    (mean, var)
}

fn vals(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
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

fn training(momentum: f64) -> BatchNormOptions {
    BatchNormOptions {
        training: true,
        momentum,
        ..Default::default()
    }
}

fn has(c: &Case, key: &str) -> bool {
    c.golden.names(c.name).iter().any(|n| n == key)
}

/// How a dtype's results are compared: `ulps` units in the last place of the expected value plus
/// `k` units of the statistics' dtype times the magnitude bound.
fn allowance(dtype: DType) -> (f64, f64) {
    match dtype {
        DType::F64 => (0.0, 16.0),
        DType::F32 => (1.0, 2.0),
        DType::F16 | DType::BF16 => (1.0, 2.0),
        _ => (0.0, 0.0),
    }
}

/// The spacing of `dtype` relative to a value: an upper bound of one unit in the last place.
fn unit(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 2f64.powi(-10),
        DType::BF16 => 2f64.powi(-7),
        DType::F32 => 2f64.powi(-23),
        _ => 2f64.powi(-52),
    }
}

/// The largest errors of a comparison: absolute, relative to the expected value (for nonzero
/// ones), relative to the magnitude bound, and the fraction of the allowance used.
#[derive(Default)]
struct Errors {
    abs: f64,
    rel: f64,
    norm: f64,
    used: f64,
}

/// Asserts `|actual - expected| <= ulps * unit * |expected| + k * unit(stats) * bound`
/// elementwise (NaN and infinities must match) and returns the largest errors.
fn assert_within(
    actual: &Tensor,
    expected: &Tensor,
    bound: &Tensor,
    stats: DType,
    what: &str,
) -> Errors {
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{what}: dtype");
    let dtype = expected.dtype();
    let (ulps, k) = allowance(dtype);
    let (a, e, m) = (vals(actual), vals(expected), vals(bound));
    let mut errors = Errors::default();
    for i in 0..a.len() {
        let (x, y) = (a[i], e[i]);
        if y.is_nan() || x.is_nan() || y.is_infinite() || x.is_infinite() {
            assert!(
                (x.is_nan() && y.is_nan()) || x == y,
                "{what}: element {i}: got {x:e}, expected {y:e}"
            );
            continue;
        }
        let allowed = ulps * unit(dtype) * y.abs() + k * unit(stats) * m[i] + f64::MIN_POSITIVE;
        let error = (x - y).abs();
        assert!(
            error <= allowed,
            "{what}: element {i}: got {x:e}, expected {y:e} (error {error:e}, allowed {allowed:e})"
        );
        errors.abs = errors.abs.max(error);
        if y != 0.0 {
            errors.rel = errors.rel.max(error / y.abs());
        }
        if m[i] > 0.0 {
            errors.norm = errors.norm.max(error / m[i]);
        }
        errors.used = errors.used.max(error / allowed);
    }
    errors
}

/// Appends the errors of `tensor` of `case` as a line of JSON to the file that
/// `HOMURA_TOLERANCE_LOG` names, if it is set.
fn log_measurement(case: &str, tensor: &str, errors: &Errors) {
    let Ok(path) = std::env::var("HOMURA_TOLERANCE_LOG") else {
        return;
    };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("can't open the tolerance log");
    let Errors {
        abs,
        rel,
        norm,
        used,
    } = errors;
    writeln!(
        file,
        r#"{{"case": "{case}", "tensor": "{tensor}", "abs": {abs:e}, "rel": {rel:e}, "norm": {norm:e}, "used": {used:e}}}"#
    )
    .unwrap();
}

/// The per-channel `[1, C, 1, ...]` form of `t` for an input with `dims` dimensions, in F64.
fn per_channel(t: &Tensor, dims: usize) -> Tensor {
    let mut shape = vec![1i64; dims];
    shape[1] = -1;
    t.to(DType::F64).reshape(shape)
}

/// The magnitude bounds of the output and of the gradients of `x`, `weight` and `bias` (those
/// that are given) of the golden case `c`, in F64: with `X = (|x| + |mean|)·invstd`, the output
/// gets `X·|w| + |b|`; the input's gradient `(|g| + Σ|g|/n + X·Σ(|g|·X)/n)·invstd·|w|` in
/// training and `|g|·invstd·|w|` otherwise, the weight's `Σ |g|·X` and the bias's `Σ |g|`.
/// The statistics are those of the batch (the biased variance) or the running ones.
fn bounds(c: &Case, g: &Tensor) -> Vec<Tensor> {
    let x = c.input("x").to(DType::F64);
    let dims = x.num_dims();
    let axes: Vec<i64> = std::iter::once(0).chain(2..dims as i64).collect();
    let eps = c.input("opt.eps").item::<f64>();
    let train = c.input("opt.training").item::<i64>() == 1;
    let n = x.shape()[0] * x.shape()[2..].iter().product::<usize>();
    let (mean, var) = if train {
        (
            x.mean_dim(axes.clone(), true),
            x.var_dim(axes.clone(), 0, true),
        )
    } else {
        (
            per_channel(&c.input("running_mean"), dims),
            per_channel(&c.input("running_var"), dims),
        )
    };
    let invstd = var.add(eps).sqrt().reciprocal();
    let w = has(c, "weight").then(|| per_channel(&c.input("weight"), dims).abs());
    let scale = |t: Tensor| match &w {
        Some(w) => t.mul(w),
        None => t,
    };
    let big_x = x.abs().add(mean.abs()).mul(&invstd);
    let mut out = scale(big_x.clone());
    if has(c, "bias") {
        out = out.add(per_channel(&c.input("bias"), dims).abs());
    }
    let g = g.detach().to(DType::F64).abs();
    let sum = g.sum_dim(axes.clone(), true);
    let dotp = g.mul(&big_x).sum_dim(axes.clone(), true);
    let grad_x = if train {
        g.add(sum.div(n as f64))
            .add(big_x.mul(dotp.div(n as f64)))
            .mul(&invstd)
    } else {
        g.mul(&invstd)
    };
    let mut result = vec![out, scale(grad_x)];
    if has(c, "weight") {
        result.push(dotp.reshape([-1]));
    }
    if has(c, "bias") {
        result.push(sum.reshape([-1]));
    }
    result
}

#[test]
fn batch_norm_matches_pytorch() {
    let golden = Golden::load("m2_batch_norm.safetensors");
    let cases = golden.cases();
    // 24 cases in f64 and f32, 6 of them and 3 mixed ones in f16 and bf16.
    assert_eq!(cases.len(), 2 * 24 + 2 * 6 + 2 * 3);
    let mut second_order = 0;
    for name in &cases {
        let c = Case::new(&golden, name);
        let keys: Vec<&str> = ["x", "weight", "bias"]
            .into_iter()
            .filter(|k| has(&c, k))
            .collect();
        let args: Vec<Tensor> = keys.iter().map(|&k| c.input(k)).collect();
        for t in &args {
            t.requires_grad_(true);
        }
        let running: Vec<Option<Tensor>> = ["running_mean", "running_var"]
            .iter()
            .map(|k| has(&c, k).then(|| c.input(k)))
            .collect();
        let options = BatchNormOptions {
            training: c.input("opt.training").item::<i64>() == 1,
            momentum: c.input("opt.momentum").item::<f64>(),
            eps: c.input("opt.eps").item::<f64>(),
        };
        let param = |k: &str| keys.iter().position(|n| *n == k).map(|i| &args[i]);
        let mut out = None;
        for _ in 0..c.input("opt.steps").item::<i64>() {
            out = Some(batch_norm(
                &args[0],
                running[0].as_ref(),
                running[1].as_ref(),
                param("weight"),
                param("bias"),
                options,
            ));
        }
        let out = out.unwrap();
        // The dtype of the parameters, which PyTorch rounds the batch statistics to.
        let stats = [param("weight"), param("bias"), running[0].as_ref()]
            .into_iter()
            .flatten()
            .next()
            .unwrap_or(&args[0])
            .dtype();
        let second = has(&c, "gg_x");
        let g = c.input("g");
        g.requires_grad_(second);
        let bound = bounds(&c, &g);
        let check = |key: &str, t: &Tensor, bound: &Tensor| {
            c.check_with(key, t, |a, e| {
                let errors = assert_within(a, e, bound, stats, &format!("{name}/{key}"));
                log_measurement(name, key, &errors);
            });
        };
        check("out", &out, &bound[0]);
        if options.training && running[0].is_some() {
            for (k, t) in ["running_mean", "running_var"].iter().zip(&running) {
                let key = format!("{k}_out");
                // The update `(1 - momentum)·running + momentum·batch` can cancel: the bound is
                // the magnitude of the old, the batch's and the new values. PyTorch updates
                // BF16 statistics in BF16, rounding each operation.
                let x = c.input("x").to(DType::F64);
                let axes: Vec<i64> = std::iter::once(0).chain(2..x.num_dims() as i64).collect();
                let batch = if x.num_elements() == 0 {
                    // Not updated.
                    homura::zeros_with([x.shape()[1]], DType::F64)
                } else if *k == "running_mean" {
                    x.mean_dim(axes, false)
                } else {
                    x.var_dim(axes, 1, false)
                };
                let old = c.input(k).to(DType::F64).reshape([-1]);
                c.check_with(&key, t.as_ref().unwrap(), |a, e| {
                    let bound = old.abs().add(batch.abs()).add(e.to(DType::F64).abs());
                    let what = format!("{name}/{key}");
                    let errors = assert_within(a, e, &bound, e.dtype(), &what);
                    log_measurement(name, &key, &errors);
                });
            }
        }
        let inputs: Vec<&Tensor> = args.iter().collect();
        let first = grads(&out, &inputs, &g, second);
        for (i, k) in keys.iter().enumerate() {
            check(
                &format!("grad_{k}"),
                first[i].as_ref().unwrap(),
                &bound[1 + i],
            );
        }
        if second {
            second_order += 1;
            let mut total = homura::zeros_with([], DType::F64);
            for (i, k) in keys.iter().enumerate() {
                let gg = c.input(&format!("gg_{k}"));
                total = total.add(first[i].as_ref().unwrap().mul(&gg).sum());
            }
            let mut variables = inputs.clone();
            variables.push(&g);
            let second_grads = autograd::grad(
                &[&total],
                &variables,
                GradOptions {
                    allow_unused: true,
                    ..Default::default()
                },
            );
            for (k, t) in keys.iter().chain(&["g"]).zip(&second_grads) {
                let key = format!("grad2_{k}");
                if has(&c, &key) {
                    c.check_with(&key, t.as_ref().unwrap(), |a, e| {
                        assert_close_tol(a, e, 1e-10, 1e-10, &format!("{name}/{key}"))
                    });
                } else {
                    // PyTorch's None (the bias, or the input in evaluation mode): homura's None
                    // or zeros.
                    assert!(
                        t.as_ref().is_none_or(|t| vals(t).iter().all(|&v| v == 0.0)),
                        "{name}/{key}"
                    );
                }
            }
        }
        c.assert_all_used();
    }
    assert_eq!(second_order, 10);
}

#[test]
fn batch_norm_records_the_reference_node() {
    let (x, w, b) = (var(&[4, 3, 2], 0), var(&[3], 1), var(&[3], 2));
    let (rm, rv) = stats(3);
    let node = [
        (0, "NativeBatchNormBackward0"),
        (1, ACC),
        (1, ACC),
        (1, ACC),
    ];
    for options in [training(0.1), BatchNormOptions::default()] {
        let out = batch_norm(&x, Some(&rm), Some(&rv), Some(&w), Some(&b), options);
        assert_eq!(chain(&out), node);
    }
    let out = batch_norm(&x, None, None, Some(&w), Some(&b), training(0.1));
    assert_eq!(chain(&out), node);
    // Without a weight the node has no edge for it.
    let out = batch_norm(&x, None, None, None, Some(&b), training(0.1));
    let next = out.grad_fn().unwrap().next_functions();
    assert_eq!(next.len(), 3);
    assert!(next[0].is_some() && next[1].is_none() && next[2].is_some());
    let out = batch_norm(&x, None, None, None, None, training(0.1));
    assert_eq!(chain(&out), [(0, "NativeBatchNormBackward0"), (1, ACC)]);
    // The gradients record the operations of the formulas, not PyTorch's
    // NativeBatchNormBackwardBackward0.
    for options in [training(0.1), BatchNormOptions::default()] {
        let out = batch_norm(&x, Some(&rm), Some(&rv), Some(&w), Some(&b), options);
        let first = grads(&out, &[&x, &w, &b], &var(&[4, 3, 2], 3), true);
        let names: Vec<&str> = first
            .iter()
            .map(|t| t.as_ref().unwrap().grad_fn().unwrap().name())
            .collect();
        assert_eq!(names, ["MulBackward0", "ViewBackward0", "ViewBackward0"]);
    }
    let _no_grad = homura::no_grad();
    let out = batch_norm(&x, Some(&rm), Some(&rv), Some(&w), Some(&b), training(0.1));
    assert!(out.grad_fn().is_none());
}

#[test]
fn batch_norm_checks_its_arguments() {
    let x = constant(&[4, 3], 0);
    let (rm, rv) = stats(3);
    let (w, b) = (constant(&[3], 1), constant(&[3], 2));
    let eval = BatchNormOptions::default();
    let train = training(0.1);
    let bn = |x: &Tensor,
              rm: Option<&Tensor>,
              rv: Option<&Tensor>,
              w: Option<&Tensor>,
              b: Option<&Tensor>,
              options| {
        let x = x.clone();
        let (rm, rv, w, b) = (rm.cloned(), rv.cloned(), w.cloned(), b.cloned());
        move || {
            batch_norm(
                &x,
                rm.as_ref(),
                rv.as_ref(),
                w.as_ref(),
                b.as_ref(),
                options,
            );
        }
    };
    // PyTorch's Python checks: one value per channel in training, then eps.
    let one = "Expected more than 1 value per channel when training, got input size";
    assert_panics(
        bn(&constant(&[1, 3], 0), None, None, None, None, train),
        &format!("{one} torch.Size([1, 3])"),
    );
    assert_panics(
        bn(&constant(&[1], 0), None, None, None, None, train),
        &format!("{one} torch.Size([1])"),
    );
    let eps0 = BatchNormOptions { eps: 0.0, ..train };
    assert_panics(
        bn(&constant(&[1, 3], 0), None, None, None, None, eps0),
        &format!("{one} torch.Size([1, 3])"),
    );
    assert_panics(
        bn(&x, None, None, None, None, eps0),
        "batch_norm eps must be positive during training, but got 0.0",
    );
    let negative = BatchNormOptions { eps: -1.0, ..eval };
    assert_panics(
        bn(&x, Some(&rm), Some(&rv), None, None, negative),
        "batch_norm eps must be non-negative, but got -1.0",
    );
    // A NaN eps passes; a channel of [1, 3, 2] has 2 values, and evaluation mode doesn't count.
    let nan = BatchNormOptions {
        eps: f64::NAN,
        ..train
    };
    bn(&x, None, None, None, None, nan)();
    bn(&constant(&[1, 3, 2], 0), None, None, None, None, train)();
    bn(
        &constant(&[1, 3], 0),
        Some(&rm),
        Some(&rv),
        None,
        None,
        eval,
    )();
    // Fewer than 2 dimensions (PyTorch: an IndexError, or a message about running_mean in
    // evaluation mode).
    let rank = "Dimension out of range (expected to be in range of [-1, 0], but got 1)";
    assert_panics(
        bn(&constant(&[3], 0), Some(&rm), Some(&rv), None, None, eval),
        rank,
    );
    assert_panics(bn(&constant(&[3], 0), None, None, None, None, train), rank);
    assert_panics(bn(&constant(&[], 0), None, None, None, None, train), rank);
    // Running statistics, then the parameters' element counts.
    assert_panics(
        bn(&x, Some(&rm), None, None, None, eval),
        "running_var must be defined in evaluation mode",
    );
    for (rm, rv) in [(None, Some(&rv)), (None, None)] {
        assert_panics(
            bn(&x, rm, rv, None, None, eval),
            "running_mean must be defined in evaluation mode",
        );
    }
    let both = "running_mean and running_var must either both be None or neither be None";
    assert_panics(bn(&x, Some(&rm), None, None, None, train), both);
    assert_panics(bn(&x, None, Some(&rv), None, None, train), both);
    let (four, five) = (constant(&[4], 3), constant(&[5], 4));
    assert_panics(
        bn(&x, Some(&four), None, None, None, train),
        "running_mean should contain 3 elements not 4",
    );
    assert_panics(
        bn(&x, Some(&four), Some(&rv), Some(&five), None, eval),
        "running_mean should contain 3 elements not 4",
    );
    assert_panics(
        bn(&x, Some(&rm), Some(&four), None, None, eval),
        "running_var should contain 3 elements not 4",
    );
    assert_panics(
        bn(
            &x,
            Some(&rm),
            Some(&rv),
            Some(&five),
            Some(&constant(&[6], 5)),
            eval,
        ),
        "weight should contain 3 elements not 5",
    );
    assert_panics(
        bn(&x, Some(&rm), Some(&rv), Some(&five), None, eval).clone(),
        "weight should contain 3 elements not 5",
    );
    assert_panics(
        bn(&x, Some(&rm), Some(&rv), Some(&w), Some(&four), eval),
        "bias should contain 3 elements not 4",
    );
    // Running statistics that require grad, while grad mode is enabled.
    let grad = |name| {
        format!(
            "The function 'native_batch_norm' is not differentiable with respect to argument \
             '{name}'. This input cannot have requires_grad True."
        )
    };
    let (rm_grad, rv_grad) = (rm.detach(), rv.detach());
    rm_grad.requires_grad_(true);
    rv_grad.requires_grad_(true);
    assert_panics(
        bn(&x, Some(&rm_grad), Some(&rv), None, None, eval),
        &grad("running_mean"),
    );
    assert_panics(
        bn(&x, Some(&rm), Some(&rv_grad), None, None, eval),
        &grad("running_var"),
    );
    assert_panics(
        bn(&x, Some(&rm_grad), None, None, None, train),
        &grad("running_mean"),
    );
    let four_grad = four.detach();
    four_grad.requires_grad_(true);
    assert_panics(
        bn(&x, Some(&four_grad), Some(&rv), None, None, eval),
        "running_mean should contain 3 elements not 4",
    );
    assert_panics(
        bn(
            &x,
            Some(&rm_grad.to(DType::F32)),
            Some(&rv.to(DType::F32)),
            None,
            None,
            eval,
        ),
        &grad("running_mean"),
    );
    {
        let _no_grad = homura::no_grad();
        bn(&x, Some(&rm_grad), Some(&rv), None, None, eval)();
    }
    // The input's dtype, then PyTorch's mixed dtype rules.
    let f32 = |t: &Tensor| t.to(DType::F32);
    let f16 = |t: &Tensor| t.to(DType::F16);
    assert_panics(
        bn(
            &x.to(DType::I64),
            Some(&rm.to(DType::I64)),
            Some(&rv.to(DType::I64)),
            None,
            None,
            eval,
        ),
        "batch_norm is not implemented for i64 tensors",
    );
    assert_panics(
        bn(&x.to(DType::I64), Some(&rm), Some(&rv), None, None, eval),
        "batch_norm is not implemented for i64 tensors",
    );
    assert_panics(
        bn(&x.greater(0.0), None, None, None, None, train),
        "batch_norm is not implemented for bool tensors",
    );
    let all_same = "mixed dtype (CPU): all inputs must share same datatype.";
    let float = "mixed dtype (CPU): expect parameter to have scalar type of Float";
    let cases: [(Tensor, [Option<Tensor>; 4], &str); 11] = [
        (
            x.clone(),
            [None, None, Some(f32(&rm)), Some(f32(&rv))],
            all_same,
        ),
        (
            x.clone(),
            [Some(f32(&w)), None, Some(rm.clone()), Some(rv.clone())],
            float,
        ),
        (
            f32(&x),
            [Some(w.clone()), None, Some(f32(&rm)), Some(f32(&rv))],
            float,
        ),
        (
            f32(&x),
            [Some(f32(&w)), Some(b.clone()), None, None],
            "expected scalar type f32 but found f64",
        ),
        (
            f32(&x),
            [Some(f32(&w)), None, Some(rm.clone()), Some(rv.clone())],
            "expected scalar type f32 but found f64",
        ),
        (f32(&x), [Some(f16(&w)), None, None, None], float),
        (
            f16(&x),
            [Some(f32(&w)), None, Some(f16(&rm)), Some(f16(&rv))],
            float,
        ),
        (
            f16(&x),
            [Some(f16(&w)), None, Some(f32(&rm)), Some(f32(&rv))],
            "expected scalar type f16 but found f32",
        ),
        (f16(&x), [Some(w.to(DType::BF16)), None, None, None], float),
        (
            x.to(DType::BF16),
            [Some(f32(&w)), Some(f16(&b)), None, None],
            float,
        ),
        (
            f16(&x),
            [Some(f16(&w)), Some(f32(&b)), None, None],
            "expected scalar type f16 but found f32",
        ),
    ];
    for (input, [w, b, rm, rv], message) in cases {
        let options = if rm.is_some() { eval } else { train };
        assert_panics(
            bn(
                &input,
                rm.as_ref(),
                rv.as_ref(),
                w.as_ref(),
                b.as_ref(),
                options,
            ),
            message,
        );
    }
    let mixed = f16(&x);
    bn(
        &mixed,
        Some(&f32(&rm)),
        Some(&f32(&rv)),
        Some(&f32(&w)),
        Some(&f32(&b)),
        eval,
    )();
    // Tensors on other devices (PyTorch: "Tensor on device cpu is not on the expected device
    // meta!").
    assert_panics(
        bn(&x.to(Device::Meta), Some(&rm), Some(&rv), None, None, eval),
        "Expected all tensors to be on the same device, but found at least two devices, meta and \
         cpu!",
    );
    assert_panics(
        bn(
            &x,
            Some(&rm),
            Some(&rv),
            Some(&w.to(Device::Meta)),
            None,
            eval,
        ),
        "Expected all tensors to be on the same device, but found at least two devices, cpu and \
         meta!",
    );
}

#[test]
fn batch_norm_updates_the_running_statistics() {
    let x = constant(&[4, 3, 2], 0);
    let (rm0, rv0) = stats(3);
    let (rm, rv) = (rm0.deep_clone(), rv0.deep_clone());
    batch_norm(&x, Some(&rm), Some(&rv), None, None, training(0.3));
    // The batch mean and the unbiased variance, with momentum 0.3.
    let mean = x.mean_dim([0, 2], false);
    let var = x.var_dim([0, 2], 1, false);
    let expected_mean = rm0.mul(0.7).add(mean.mul(0.3));
    let expected_var = rv0.mul(0.7).add(var.mul(0.3));
    assert_close_tol(&rm, &expected_mean, 1e-15, 1e-15, "running_mean");
    assert_close_tol(&rv, &expected_var, 1e-15, 1e-15, "running_var");
    // Evaluation mode and training without them leave them alone.
    batch_norm(
        &x,
        Some(&rm),
        Some(&rv),
        None,
        None,
        BatchNormOptions::default(),
    );
    batch_norm(&x, None, None, None, None, training(0.3));
    assert_eq!(vals(&rm), vals(&expected_mean));
    assert_eq!(vals(&rv), vals(&expected_var));
    // Momentum 0 keeps them, 1 replaces them, 2 extrapolates, NaN makes them NaN.
    for (momentum, f) in [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)] {
        let (rm, rv) = (rm0.deep_clone(), rv0.deep_clone());
        batch_norm(&x, Some(&rm), Some(&rv), None, None, training(momentum));
        let m = rm0.mul(1.0 - f).add(mean.mul(f));
        let v = rv0.mul(1.0 - f).add(var.mul(f));
        assert_close_tol(&rm, &m, 1e-15, 1e-15, "momentum");
        assert_close_tol(&rv, &v, 1e-15, 1e-15, "momentum");
    }
    let (rm, rv) = (rm0.deep_clone(), rv0.deep_clone());
    batch_norm(&x, Some(&rm), Some(&rv), None, None, training(f64::NAN));
    assert!(vals(&rm).iter().chain(&vals(&rv)).all(|v| v.is_nan()));
    // Views and running statistics of other shapes with C elements are written in place
    // (PyTorch accepts views, but rejects statistics with more than one dimension).
    let storage = homura::zeros_with([3, 2], DType::F64);
    let (rm, rv) = (storage.select(1, 0), rv0.reshape([1, 3]).contiguous());
    batch_norm(&x, Some(&rm), Some(&rv), None, None, training(0.3));
    assert_close_tol(&storage.select(1, 0), &mean.mul(0.3), 1e-15, 1e-15, "view");
    assert_eq!(vals(&storage.select(1, 1)), [0.0; 3]);
    assert_eq!(rv.shape(), [1, 3]);
    assert_close_tol(&rv.reshape([3]), &expected_var, 1e-15, 1e-15, "[1, 3]");
    // A failed check writes nothing and leaves grad mode enabled.
    let (rm, rv) = (rm0.to(DType::F32), rv0.to(DType::F32));
    let w = constant(&[3], 1).to(DType::F32);
    let b = constant(&[3], 2);
    let message = panic_message(|| {
        batch_norm(
            &x.to(DType::F32),
            Some(&rm),
            Some(&rv),
            Some(&w),
            Some(&b),
            training(0.3),
        );
    });
    assert_eq!(message, "expected scalar type f32 but found f64");
    assert_eq!(vals(&rm), vals(&rm0.to(DType::F32)));
    assert_eq!(vals(&rv), vals(&rv0.to(DType::F32)));
    assert!(autograd::is_grad_enabled());
}

#[test]
fn batch_norm_checks_the_running_statistics_before_writing() {
    // Both statistics are checked before either is written (PyTorch writes into expanded
    // statistics, the last write winning, and reads statistics that alias the input after
    // updating them).
    let unwritable = "unsupported operation: more than one element of the written-to tensor may \
                      refer to a single memory location. Please deep_clone() the tensor before \
                      performing the operation.";
    let shared = "unsupported operation: some elements of the input tensor and the written-to \
                  tensor refer to a single memory location. Please deep_clone() the tensor \
                  before performing the operation.";
    let x = Tensor::from_vec(vec![1.0, 10.0, 3.0, 20.0], [2, 2]);
    x.requires_grad_(true);
    let g = constant(&[2, 2], 3);
    for device in [Device::Cpu, Device::Meta] {
        let x = x.detach().to(device);
        x.requires_grad_(true);
        let g = g.to(device);
        let expanded = homura::ones_with([1], (DType::F64, device)).expand([2]);
        let plain = || homura::zeros_with([2], (DType::F64, device));
        for unwritable_first in [false, true] {
            let (rm, rv) = if unwritable_first {
                (expanded.clone(), plain())
            } else {
                (plain(), expanded.clone())
            };
            // Expanded statistics can be read; that graph saves both.
            let eval = batch_norm(
                &x,
                Some(&rm),
                Some(&rv),
                None,
                None,
                BatchNormOptions::default(),
            );
            assert_panics(
                || {
                    batch_norm(&x, Some(&rm), Some(&rv), None, None, training(0.5));
                },
                unwritable,
            );
            // Neither was written: the old graph's saved statistics are unchanged.
            assert!(grads(&eval, &[&x], &g, false)[0].is_some());
            if device == Device::Cpu {
                assert_eq!(
                    vals(&rm)
                        .iter()
                        .chain(&vals(&rv))
                        .filter(|&&v| v == 0.0)
                        .count(),
                    2
                );
            }
        }
    }
    // Statistics that share memory with the input, the parameters or each other.
    let x = Tensor::from_vec(vec![1.0, 3.0, 5.0], [3, 1]);
    x.requires_grad_(true);
    let square = x.mul(&x);
    let rm = x.detach().select(0, 0);
    let rv = homura::ones([1]).to(DType::F64);
    assert_panics(
        || {
            batch_norm(&x, Some(&rm), Some(&rv), None, None, training(0.5));
        },
        shared,
    );
    assert_eq!(vals(&x), [1.0, 3.0, 5.0]);
    // The input wasn't written: a graph that saved it still runs.
    assert!(grads(&square, &[&x], &constant(&[3, 1], 0), false)[0].is_some());
    // Reading them is fine.
    batch_norm(
        &x,
        Some(&rm),
        Some(&rv),
        None,
        None,
        BatchNormOptions::default(),
    );
    let x = constant(&[4, 2], 0);
    let (w, b) = (
        homura::ones([2]).to(DType::F64),
        homura::zeros_with([2], DType::F64),
    );
    let other = || homura::ones([2]).to(DType::F64);
    for (rm, rv, weight, bias) in [
        (w.clone(), other(), Some(&w), None),
        (other(), b.clone(), None, Some(&b)),
        (w.clone(), w.clone(), None, None),
    ] {
        let before = (vals(&rm), vals(&rv));
        assert_panics(
            || {
                batch_norm(&x, Some(&rm), Some(&rv), weight, bias, training(0.5));
            },
            shared,
        );
        assert_eq!((vals(&rm), vals(&rv)), before);
    }
    // Rows of one buffer don't overlap; interleaved views count as sharing.
    let buffer = homura::ones([2, 2]).to(DType::F64);
    batch_norm(
        &x,
        Some(&buffer.select(0, 0)),
        Some(&buffer.select(0, 1)),
        None,
        None,
        training(0.5),
    );
    assert_ne!(vals(&buffer), [1.0; 4]);
    let columns = homura::ones([2, 2]).to(DType::F64);
    assert_panics(
        || {
            batch_norm(
                &x,
                Some(&columns.select(1, 0)),
                Some(&columns.select(1, 1)),
                None,
                None,
                training(0.5),
            );
        },
        shared,
    );
    // Without elements nothing is written, so nothing is checked.
    let empty = constant(&[0, 2], 0);
    batch_norm(
        &empty,
        Some(&expanded_f64()),
        Some(&expanded_f64()),
        None,
        None,
        training(0.5),
    );
}

fn expanded_f64() -> Tensor {
    homura::ones_with([1], DType::F64).expand([2])
}

#[test]
fn batch_norm_rejects_stale_views_before_writing() {
    // A view whose base was modified in place in grad mode can't be recorded; batch_norm fails
    // before it updates the running statistics (PyTorch replays such views).
    let stale = "a view whose base was modified by an in-place operation in grad mode after the \
                 view was created can't be used for gradient computation yet; create the view \
                 again after the in-place operation";
    for device in [Device::Cpu, Device::Meta] {
        let on = |t: Tensor| t.to(device);
        let stale_view = |values: Vec<f64>, shape: &[usize]| {
            let leaf = on(Tensor::from_vec(values, shape));
            leaf.requires_grad_(true);
            let base = leaf.mul(1.0);
            let view = base.view(shape.iter().map(|&d| d as i64).collect::<Vec<_>>());
            base.mul_(2.0);
            view
        };
        let x = on(Tensor::from_vec(vec![1.0, 10.0, 3.0, 20.0], [2, 2]));
        let fresh = || on(homura::ones([2]).to(DType::F64));
        for which in 0..3 {
            let input = if which == 0 {
                stale_view(vec![1.0, 10.0, 3.0, 20.0], &[2, 2])
            } else {
                x.clone()
            };
            let w = (which == 1).then(|| stale_view(vec![1.0, 1.0], &[2]));
            let b = (which == 2).then(|| stale_view(vec![0.0, 0.0], &[2]));
            let (rm, rv) = (on(homura::zeros_with([2], DType::F64)), fresh());
            // A graph that saves both statistics: it still runs after the failure.
            let xe = x.detach();
            xe.requires_grad_(true);
            let eval = batch_norm(
                &xe,
                Some(&rm),
                Some(&rv),
                None,
                None,
                BatchNormOptions::default(),
            );
            assert_panics(
                || {
                    batch_norm(
                        &input,
                        Some(&rm),
                        Some(&rv),
                        w.as_ref(),
                        b.as_ref(),
                        training(0.5),
                    );
                },
                stale,
            );
            assert!(autograd::is_grad_enabled());
            assert!(grads(&eval, &[&xe], &eval.detach(), false)[0].is_some());
            if device == Device::Cpu {
                assert_eq!(vals(&rm), [0.0, 0.0], "{which}");
                assert_eq!(vals(&rv), [1.0, 1.0], "{which}");
            }
            // Without grad mode nothing is recorded, and the statistics are updated.
            let _no_grad = homura::no_grad();
            batch_norm(
                &input,
                Some(&rm),
                Some(&rv),
                w.as_ref(),
                b.as_ref(),
                training(0.5),
            );
            if device == Device::Cpu {
                assert_ne!(vals(&rm), [0.0, 0.0], "{which}");
            }
        }
    }
    // A view of a parameter whose data Module::to replaced.
    let layer = LayerNorm::new([2]);
    let weight = layer.named_parameters()[0].1.view([2]);
    layer.to(DType::F64);
    let x = constant(&[4, 2], 0).to(DType::F32);
    let (rm, rv) = (homura::zeros([2]), homura::ones([2]));
    assert_panics(
        || {
            batch_norm(&x, Some(&rm), Some(&rv), Some(&weight), None, training(0.5));
        },
        "a view whose base had its data replaced (by Module::to or by loading with assign) after \
         the view was created can't be used for gradient computation: it still refers to the \
         old data; create the view again",
    );
    assert_eq!(vals(&rm), [0.0, 0.0]);
    assert_eq!(vals(&rv), [1.0, 1.0]);
}

#[test]
fn batch_norm_saves_the_running_statistics_only_in_evaluation_mode() {
    let x = var(&[4, 3, 2], 0);
    let (rm, rv) = stats(3);
    let g = constant(&[4, 3, 2], 3);
    // Two training passes with the same running statistics, then one backward pass: the graphs
    // don't hold them.
    let first = batch_norm(&x, Some(&rm), Some(&rv), None, None, training(0.1));
    let second = batch_norm(&x, Some(&rm), Some(&rv), None, None, training(0.1));
    let total = first.mul(&g).sum().add(second.mul(&g).sum());
    let both = autograd::grad(&[&total], &[&x], GradOptions::default());
    let (rm1, rv1) = stats(3);
    let one = batch_norm(&x, Some(&rm1), Some(&rv1), None, None, training(0.1));
    let single = grads(&one, &[&x], &g, false);
    assert_close_tol(
        both[0].as_ref().unwrap(),
        &single[0].as_ref().unwrap().mul(2.0),
        1e-14,
        1e-14,
        "two passes",
    );
    // An evaluation graph saves them: a training pass that updates them afterwards makes its
    // backward pass an error (PyTorch doesn't count the update as a modification and uses the
    // new values).
    let eval = batch_norm(
        &x,
        Some(&rm),
        Some(&rv),
        None,
        None,
        BatchNormOptions::default(),
    );
    batch_norm(&x, Some(&rm), Some(&rv), None, None, training(0.1));
    let message = panic_message(|| {
        grads(&eval, &[&x], &g, false);
    });
    assert!(
        message.contains("modified by an inplace operation"),
        "{message}"
    );
    // A training pass without elements doesn't update them.
    let eval = batch_norm(
        &x,
        Some(&rm),
        Some(&rv),
        None,
        None,
        BatchNormOptions::default(),
    );
    let empty = constant(&[0, 3, 2], 0);
    batch_norm(&empty, Some(&rm), Some(&rv), None, None, training(0.1));
    assert!(grads(&eval, &[&x], &g, false)[0].is_some());
}

#[test]
fn batch_norm_takes_parameters_by_element_count() {
    // Per-channel tensors only need C elements; their gradients have their shapes (PyTorch's
    // backward pass rejects the gradient of a [C, 1] weight).
    let x = var(&[4, 3, 2], 0);
    let (w, b) = (var(&[3, 1], 1), var(&[1, 3], 2));
    let (wf, bf) = (var(&[3], 1), var(&[3], 2));
    let g = constant(&[4, 3, 2], 3);
    for options in [training(0.1), BatchNormOptions::default()] {
        let (rm, rv) = stats(3);
        let (rm, rv) = (rm.reshape([3, 1]), rv.reshape([1, 1, 3]));
        let out = batch_norm(&x, Some(&rm), Some(&rv), Some(&w), Some(&b), options);
        let (rmf, rvf) = stats(3);
        let flat = batch_norm(&x, Some(&rmf), Some(&rvf), Some(&wf), Some(&bf), options);
        assert_eq!(vals(&out), vals(&flat));
        let shaped = grads(&out, &[&x, &w, &b], &g, false);
        let expected = grads(&flat, &[&x, &wf, &bf], &g, false);
        for (t, e) in shaped.iter().zip(&expected) {
            let (t, e) = (t.as_ref().unwrap(), e.as_ref().unwrap());
            assert_eq!(vals(t), vals(e));
        }
        assert_eq!(shaped[1].as_ref().unwrap().shape(), [3, 1]);
        assert_eq!(shaped[2].as_ref().unwrap().shape(), [1, 3]);
        assert_eq!(vals(&rm), vals(&rmf));
        assert_eq!(vals(&rv), vals(&rvf));
    }
}

#[test]
fn batch_norm_bias_without_weight_has_second_derivatives() {
    // PyTorch's batchnorm_double_backward fails with a bias but no weight. The bias's gradient
    // is the sum of g over all but the channels; the input's gradient is that of the plain
    // normalization.
    let (x, b) = (var(&[4, 3, 2], 0), var(&[3], 1));
    let g = var(&[4, 3, 2], 2);
    let out = batch_norm(&x, None, None, None, Some(&b), training(0.1));
    let first = grads(&out, &[&x, &b], &g, true);
    let (ggx, ggb) = (constant(&[4, 3, 2], 3), constant(&[3], 4));
    let total = first[0]
        .as_ref()
        .unwrap()
        .mul(&ggx)
        .sum()
        .add(first[1].as_ref().unwrap().mul(&ggb).sum());
    let second = autograd::grad(
        &[&total],
        &[&x, &b, &g],
        GradOptions {
            allow_unused: true,
            ..Default::default()
        },
    );
    assert!(second[1].is_none());
    let plain = batch_norm(&x, None, None, None, None, training(0.1));
    let plain_first = grads(&plain, &[&x], &g, true);
    let plain_second = autograd::grad(
        &[&plain_first[0].as_ref().unwrap().mul(&ggx).sum()],
        &[&x, &g],
        GradOptions::default(),
    );
    assert_close_tol(
        second[0].as_ref().unwrap(),
        plain_second[0].as_ref().unwrap(),
        1e-14,
        1e-14,
        "x",
    );
    let expected_g = plain_second[1]
        .as_ref()
        .unwrap()
        .add(ggb.reshape([1, 3, 1]));
    assert_close_tol(second[2].as_ref().unwrap(), &expected_g, 1e-14, 1e-14, "g");
}

#[test]
fn batch_norm_edge_cases() {
    // Without a bias, PyTorch subtracts the shift from zero: -0 inputs give 0.
    let x = Tensor::from_vec(vec![-0.0, 1.0, 0.0, -1.0], [2, 2]);
    let (rm, rv) = (
        homura::zeros_with([2], DType::F64),
        homura::ones([2]).to(DType::F64),
    );
    let eval = BatchNormOptions {
        eps: 0.0,
        ..Default::default()
    };
    let out = vals(&batch_norm(&x, Some(&rm), Some(&rv), None, None, eval));
    assert_eq!(out, [0.0, 1.0, 0.0, -1.0]);
    assert!(out[0].is_sign_positive() && out[2].is_sign_positive());
    // Without grad mode, running statistics that require grad are updated like others.
    let (rm, rv) = stats(3);
    let (rm0, rv0) = (rm.deep_clone(), rv.deep_clone());
    rm.requires_grad_(true);
    rv.requires_grad_(true);
    {
        let _no_grad = homura::no_grad();
        batch_norm(
            &constant(&[4, 3], 0),
            Some(&rm),
            Some(&rv),
            None,
            None,
            training(0.1),
        );
    }
    assert_ne!(vals(&rm), vals(&rm0));
    assert_ne!(vals(&rv), vals(&rv0));
    assert!(rm.requires_grad() && rm.grad_fn().is_none());
    // Half precision statistics are rounded to the parameters' dtype: the F16 mean of 1000,
    // 1000 and 1000.5 is 1000, while F32 parameters keep 1000.1667.
    let x = Tensor::from_vec(vec![1000.0, 1000.0, 1000.5], [3, 1]).to(DType::F16);
    let half = batch_norm(&x, None, None, None, None, training(0.1));
    assert_eq!(vals(&half), [0.0, 0.0, 2.12109375]);
    let w = homura::ones([1]);
    let mixed = batch_norm(&x, None, None, Some(&w), None, training(0.1));
    assert_eq!(vals(&mixed), [-0.70751953125, -0.70751953125, 1.4140625]);
}

#[test]
fn batch_norm_without_elements() {
    // PyTorch computes `input·weight[0] + bias[0]` for inputs without elements (nodes
    // AddBackward0 / MulBackward0 / CloneBackward0), skipping the checks after eps; homura
    // records its node and checks everything.
    let (w, b) = (var(&[3], 1), var(&[3], 2));
    for shape in [[0, 3, 2], [2, 3, 0]] {
        let x = var(&shape, 0);
        let (rm, rv) = stats(3);
        let out = batch_norm(&x, Some(&rm), Some(&rv), Some(&w), Some(&b), training(0.1));
        assert_eq!(out.shape(), shape);
        assert_eq!(out.grad_fn().unwrap().name(), "NativeBatchNormBackward0");
        assert_eq!(vals(&rm), vals(&stats(3).0));
        assert_eq!(vals(&rv), vals(&stats(3).1));
        let g = constant(&shape, 3);
        let first = grads(&out, &[&x, &w, &b], &g, false);
        assert_eq!(first[0].as_ref().unwrap().shape(), shape);
        assert_eq!(vals(first[1].as_ref().unwrap()), [0.0; 3]);
        assert_eq!(vals(first[2].as_ref().unwrap()), [0.0; 3]);
    }
    let x = var(&[2, 0, 3], 0);
    let none = homura::zeros_with([0], DType::F64);
    let out = batch_norm(&x, Some(&none), Some(&none), None, None, training(0.1));
    assert_eq!(out.shape(), [2, 0, 3]);
    let empty = constant(&[0, 3], 0);
    assert_panics(
        || {
            batch_norm(&empty, None, None, None, None, BatchNormOptions::default());
        },
        "running_mean must be defined in evaluation mode",
    );
    assert_panics(
        || {
            batch_norm(
                &empty,
                None,
                None,
                Some(&constant(&[5], 0)),
                None,
                training(0.1),
            );
        },
        "weight should contain 3 elements not 5",
    );
}

#[test]
fn batch_norm_on_meta() {
    let meta = |shape: &[usize], dtype| homura::zeros_with(shape, (dtype, Device::Meta));
    let x = meta(&[4, 3, 2], DType::F64);
    x.requires_grad_(true);
    let w = meta(&[3], DType::F64);
    w.requires_grad_(true);
    let (rm, rv) = (meta(&[3], DType::F64), meta(&[3], DType::F64));
    for options in [training(0.1), BatchNormOptions::default()] {
        let out = batch_norm(&x, Some(&rm), Some(&rv), Some(&w), None, options);
        assert_eq!(
            (out.shape().to_vec(), out.dtype(), out.device()),
            (vec![4, 3, 2], DType::F64, Device::Meta)
        );
        let g = meta(&[4, 3, 2], DType::F64);
        let first = grads(&out, &[&x, &w], &g, false);
        assert_eq!(first[0].as_ref().unwrap().shape(), [4, 3, 2]);
        assert_eq!(first[1].as_ref().unwrap().shape(), [3]);
        assert_eq!(first[1].as_ref().unwrap().device(), Device::Meta);
    }
    let half = batch_norm(
        &meta(&[4, 3], DType::F16),
        Some(&meta(&[3], DType::F32)),
        Some(&meta(&[3], DType::F32)),
        None,
        None,
        training(0.1),
    );
    assert_eq!(half.dtype(), DType::F16);
    // The dtype rules apply on Meta too (PyTorch's Meta kernel accepts mixed dtypes and errors
    // only in its mean for integers).
    assert_panics(
        || {
            batch_norm(
                &meta(&[4, 3], DType::I64),
                None,
                None,
                None,
                None,
                training(0.1),
            );
        },
        "batch_norm is not implemented for i64 tensors",
    );
    assert_panics(
        || {
            let s = meta(&[3], DType::F32);
            batch_norm(
                &x,
                Some(&s),
                Some(&s),
                None,
                None,
                BatchNormOptions::default(),
            );
        },
        "mixed dtype (CPU): all inputs must share same datatype.",
    );
}

#[test]
fn batch_norm_gradients_pass_gradcheck() {
    for shape in [&[5, 3][..], &[3, 2, 2, 2]] {
        let c = shape[1];
        let (x, w, b) = (var(shape, 0), var(&[c], 1), var(&[c], 2));
        for train in [true, false] {
            let f = |t: &[Tensor]| {
                let (rm, rv) = stats(c);
                let options = BatchNormOptions {
                    training: train,
                    ..Default::default()
                };
                vec![batch_norm(
                    &t[0],
                    Some(&rm),
                    Some(&rv),
                    Some(&t[1]),
                    Some(&t[2]),
                    options,
                )]
            };
            gradcheck(f, &[&x, &w, &b], GradcheckOptions::default()).unwrap();
            gradgradcheck(f, &[&x, &w, &b], GradcheckOptions::default()).unwrap();
            let plain =
                |t: &[Tensor]| vec![batch_norm(&t[0], None, None, None, None, training(0.1))];
            gradcheck(plain, &[&x], GradcheckOptions::default()).unwrap();
            gradgradcheck(plain, &[&x], GradcheckOptions::default()).unwrap();
        }
    }
    // A non-contiguous input.
    let x = var(&[3, 4, 2], 0).transpose(0, 1);
    let w = var(&[3], 1);
    let f = |t: &[Tensor]| {
        vec![batch_norm(
            &t[0],
            None,
            None,
            Some(&t[1]),
            None,
            training(0.1),
        )]
    };
    gradcheck(f, &[&x, &w], GradcheckOptions::default()).unwrap();
    gradgradcheck(f, &[&x, &w], GradcheckOptions::default()).unwrap();
}

#[test]
fn batch_norm_gradients_of_each_subset() {
    let g = constant(&[4, 3, 2], 3);
    let make = || {
        (
            constant(&[4, 3, 2], 0),
            constant(&[3], 1),
            constant(&[3], 2),
        )
    };
    for options in [training(0.1), BatchNormOptions::default()] {
        let run = |x: &Tensor, w: &Tensor, b: &Tensor| {
            let (rm, rv) = stats(3);
            batch_norm(x, Some(&rm), Some(&rv), Some(w), Some(b), options)
        };
        let (x, w, b) = make();
        for t in [&x, &w, &b] {
            t.requires_grad_(true);
        }
        let full: Vec<Vec<f64>> = grads(&run(&x, &w, &b), &[&x, &w, &b], &g, false)
            .iter()
            .map(|t| vals(t.as_ref().unwrap()))
            .collect();
        for mask in 1..8usize {
            let (x, w, b) = make();
            let inputs = [&x, &w, &b];
            let chosen: Vec<usize> = (0..3).filter(|i| mask & (1 << i) != 0).collect();
            for &i in &chosen {
                inputs[i].requires_grad_(true);
            }
            let out = run(&x, &w, &b);
            let variables: Vec<&Tensor> = chosen.iter().map(|&i| inputs[i]).collect();
            for (&i, grad) in chosen.iter().zip(grads(&out, &variables, &g, false)) {
                assert_eq!(vals(&grad.unwrap()), full[i], "mask {mask:03b}, input {i}");
            }
        }
    }
}
