//! M2 step 5: `nn::functional::{embedding, layer_norm, rms_norm}` and the layers `Embedding`,
//! `LayerNorm` and `RmsNorm`. Outputs and gradients are compared with
//! `tests/golden/m2_embed_norm.safetensors` from `tools/golden/gen_m2_embed_norm.py`; the
//! reference behavior encoded in the other tests is asserted by
//! `tools/golden/check_m2_embed_norm.py`, in the same order.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden, assert_close_tol, tolerance};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::nn::functional::{embedding, layer_norm, rms_norm};
use homura::nn::{Embedding, Forward, LayerNorm, Module, RmsNorm, Sequential};
use homura::{DType, Device, Tensor};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
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

fn as_f64(t: &Tensor) -> Vec<f64> {
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

/// The output of the golden case `c` of `function`, from its stored inputs.
fn evaluate(function: &str, c: &Case, lookup: &dyn Fn(&str) -> Option<Tensor>) -> Tensor {
    if function == "emb" {
        let padding_idx = lookup("opt.padding_idx").map(|p| p.item::<i64>());
        return embedding(
            &lookup("indices").unwrap(),
            &lookup("weight").unwrap(),
            padding_idx,
        );
    }
    let x = lookup("x").unwrap();
    let k = c.input("opt.normalized_ndim").item::<i64>() as usize;
    let shape = x.shape()[x.num_dims() - k..].to_vec();
    let eps = lookup("opt.eps").map(|e| e.item::<f64>());
    let (weight, bias) = (lookup("weight"), lookup("bias"));
    match function {
        "ln" => layer_norm(&x, shape, weight.as_ref(), bias.as_ref(), eps.unwrap()),
        "rms" => rms_norm(&x, shape, weight.as_ref(), eps),
        _ => panic!("unknown function {function}"),
    }
}

/// Tolerances `(rtol, atol)` for comparing with the golden tensor `expected`. homura's sums
/// accumulate in f64 and round once, PyTorch's in the dtype; half precision allows two rounding
/// steps. A half precision embedding gradient (`summands`: the incoming gradient) is the sum of
/// up to three rows of it, which PyTorch rounds after each addition: `atol` allows a rounding
/// step of the largest partial sum per addition, which can be much more than the result.
fn golden_tolerance(expected: &Tensor, summands: Option<&Tensor>) -> (f64, f64) {
    let dtype = expected.dtype();
    let (rtol, atol) = tolerance(dtype);
    match dtype {
        DType::F16 | DType::BF16 => {
            let largest = summands.map_or(0.0, |g| {
                as_f64(g)
                    .into_iter()
                    .filter(|v| v.is_finite())
                    .fold(0.0, |m: f64, v| m.max(v.abs()))
            });
            (2.0 * rtol, atol + 2.0 * 3.0 * largest * rtol)
        }
        _ => (rtol, atol),
    }
}

/// Tolerances `(rtol, atol)` for the weight or bias gradient (`key`) of the half precision
/// layer norm case `c`. PyTorch adds the rows of these gradients in a buffer of the input's
/// dtype, also for `F32` parameters, rounding after each row: `atol` allows a rounding step of
/// the sum of the magnitudes of the rows per row, which can be much more than the result.
fn layer_norm_row_sum_tolerance(c: &Case, key: &str) -> (f64, f64) {
    let x = c.input("x");
    let (rtol, atol) = tolerance(x.dtype());
    let x = x.to(DType::F64);
    let g = c.input("g").to(DType::F64);
    let k = c.input("opt.normalized_ndim").item::<i64>() as usize;
    let axis = x.num_dims() - k;
    let summands = if key == "grad_weight" {
        let eps = c.input("opt.eps").item::<f64>();
        g.mul(layer_norm(&x, x.shape()[axis..].to_vec(), None, None, eps))
    } else {
        g
    };
    let batch: Vec<i64> = (0..axis as i64).collect();
    let rows: usize = x.shape()[..axis].iter().product();
    let bound = as_f64(&summands.abs().sum_dim(batch, false))
        .into_iter()
        .fold(0.0, f64::max);
    (2.0 * rtol, atol + rows as f64 * rtol * bound)
}

/// An `atol` for the `F32` tensor `expected` of the layer norm case `c` that allows for the
/// rounding of the statistics: the normalized values `(x - mean) / std` lose about
/// `κ = max |x| / std` rounding steps (PyTorch's mean and variance are `F32`, and homura's are
/// rounded to `F32`), so four steps of `κ` times the largest expected value. It is about the
/// default tolerance for inputs of zero mean, and allows `1e-4` for the case `offset`, whose
/// rows are about 1000 with a standard deviation of about 1; both libraries are that far from an
/// `F64` evaluation there.
fn layer_norm_cancellation_atol(c: &Case, expected: &Tensor) -> f64 {
    let x = c.input("x");
    let k = c.input("opt.normalized_ndim").item::<i64>() as usize;
    let n: usize = x.shape()[x.num_dims() - k..].iter().product();
    let values = as_f64(&x);
    let kappa = values
        .chunks(n.max(1))
        .map(|row| {
            let mean = row.iter().sum::<f64>() / n as f64;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            row.iter().fold(0.0, |m: f64, v| m.max(v.abs())) / var.sqrt()
        })
        .filter(|v| v.is_finite())
        .fold(0.0, f64::max);
    let scale = as_f64(expected)
        .into_iter()
        .filter(|v| v.is_finite())
        .fold(0.0, |m: f64, v| m.max(v.abs()));
    4.0 * kappa * f32::EPSILON as f64 * scale
}

#[test]
fn embedding_and_norms_match_pytorch() {
    let golden = Golden::load("m2_embed_norm.safetensors");
    let cases: Vec<String> = golden
        .cases()
        .into_iter()
        .filter(|c| !c.starts_with("dropout"))
        .collect();
    // 29 cases in f64 and f32, rms.weight_f64 only in f32, 10 cases in f16 and bf16.
    assert_eq!(cases.len(), 2 * 29 + 1 + 2 * 10);
    let mut second_order = 0;
    for name in &cases {
        let c = Case::new(&golden, name);
        let function = name.split('_').next().unwrap();
        let stored = golden.names(name);
        let keys: Vec<&str> = ["x", "weight", "bias", "indices"]
            .into_iter()
            .filter(|k| stored.iter().any(|n| n == k))
            .collect();
        let inputs: Vec<(&str, Tensor)> = keys.iter().map(|&k| (k, c.input(k))).collect();
        let differentiable: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|k| stored.contains(&format!("grad_{k}")))
            .collect();
        for (k, t) in &inputs {
            t.requires_grad_(differentiable.contains(k));
        }
        let lookup = |k: &str| {
            if k.starts_with("opt.") {
                return stored.iter().any(|n| n == k).then(|| c.input(k));
            }
            inputs.iter().find(|(n, _)| *n == k).map(|(_, t)| t.clone())
        };
        let out = evaluate(function, &c, &lookup);
        let g = c.input("g");
        let half = name.contains("_f16.") || name.contains("_bf16.");
        let check = |key: &str, t: &Tensor| {
            let summands = (function == "emb" && key == "grad_weight").then_some(&g);
            c.check_with(key, t, |a, e| {
                let (mut rtol, mut atol) = golden_tolerance(e, summands);
                if half && function == "ln" && matches!(key, "grad_weight" | "grad_bias") {
                    (rtol, atol) = layer_norm_row_sum_tolerance(&c, key);
                }
                if function == "ln" && e.dtype() == DType::F32 {
                    atol += layer_norm_cancellation_atol(&c, e);
                }
                assert_close_tol(a, e, rtol, atol, &format!("{name}/{key}"))
            })
        };
        check("out", &out);
        let second = stored.iter().any(|n| n.starts_with("gg_"));
        g.requires_grad_(second);
        let variables: Vec<Tensor> = differentiable.iter().map(|k| lookup(k).unwrap()).collect();
        let variable_refs: Vec<&Tensor> = variables.iter().collect();
        let first: Vec<Tensor> = grads(&out, &variable_refs, &g, second)
            .into_iter()
            .map(Option::unwrap)
            .collect();
        for (k, grad) in differentiable.iter().zip(&first) {
            check(&format!("grad_{k}"), grad);
        }
        if second {
            second_order += 1;
            let total = differentiable
                .iter()
                .zip(&first)
                .map(|(k, grad)| (grad * &c.input(&format!("gg_{k}"))).sum())
                .reduce(|a, b| a + b)
                .unwrap();
            let mut all = variable_refs.clone();
            all.push(&g);
            let second_grads = autograd::grad(
                &[&total],
                &all,
                GradOptions {
                    allow_unused: true,
                    ..Default::default()
                },
            );
            for (k, grad) in differentiable.iter().chain(&["g"]).zip(&second_grads) {
                let key = format!("grad2_{k}");
                match grad {
                    Some(grad) if stored.contains(&key) => check(&key, grad),
                    // The reference's None (the embedding's gradient doesn't depend on the
                    // weights): homura's None or zeros.
                    Some(grad) => assert!(
                        as_f64(grad).iter().all(|&v| v == 0.0),
                        "{name}/{key} {grad:?}"
                    ),
                    None => assert!(!stored.contains(&key), "{name}/{key}"),
                }
            }
        }
        c.assert_all_used();
    }
    assert_eq!(second_order, 10);
}

#[test]
fn embed_norm_record_the_reference_nodes() {
    let acc = "AccumulateGrad";
    let w = var(&[5, 3], 0);
    let idx = homura::tensor([[1i64, 4], [0, 1]]);
    assert_eq!(
        chain(&embedding(&idx, &w, Some(1))),
        [(0, "EmbeddingBackward0"), (1, acc)]
    );
    assert_eq!(
        chain(&embedding(&homura::tensor(2i64), &w, None)),
        [(0, "EmbeddingBackward0"), (1, acc)]
    );
    let (x, lw, lb) = (var(&[2, 3, 4], 1), var(&[4], 2), var(&[4], 3));
    assert_eq!(
        chain(&layer_norm(&x, [4], Some(&lw), Some(&lb), 1e-5)),
        [
            (0, "NativeLayerNormBackward0"),
            (1, acc),
            (1, acc),
            (1, acc)
        ]
    );
    assert_eq!(
        chain(&layer_norm(&x, [3, 4], None, None, 1e-5)),
        [(0, "NativeLayerNormBackward0"), (1, acc)]
    );
    // PyTorch's CPU rms_norm: x * rsqrt(mean(x²) + eps) * weight. homura names the addition of a
    // scalar AddBackward0, where PyTorch has AddBackward1.
    let rw = var(&[4], 4);
    let composite = |x: &Tensor, w: &Tensor| chain(&rms_norm(x, [4], Some(w), None));
    let body = [
        (1, "MulBackward0"),
        (2, acc),
        (2, "RsqrtBackward0"),
        (3, "AddBackward0"),
        (4, "MeanBackward1"),
        (5, "PowBackward0"),
        (6, acc),
        (1, acc),
    ];
    let mut expected = vec![(0, "MulBackward0")];
    expected.extend(body);
    assert_eq!(composite(&x, &rw), expected);
    // Half precision: computed in F32 between two conversions.
    let xh = constant(&[2, 4], 5).to(DType::F16);
    xh.requires_grad_(true);
    let wh = constant(&[4], 6).to(DType::F16);
    wh.requires_grad_(true);
    let names: Vec<&str> = composite(&xh, &wh).into_iter().map(|(_, n)| n).collect();
    assert_eq!(names.first(), Some(&"ToCopyBackward0"));
    // The result's conversion and the input's (reached through x² and x * rstd); the F16 weight
    // is promoted by the multiplication, as in PyTorch.
    assert_eq!(names.iter().filter(|&&n| n == "ToCopyBackward0").count(), 3);
    // Nothing is recorded in no-grad mode.
    let _guard = homura::no_grad();
    assert!(embedding(&idx, &w, None).grad_fn().is_none());
    assert!(
        layer_norm(&x, [4], Some(&lw), Some(&lb), 1e-5)
            .grad_fn()
            .is_none()
    );
    assert!(rms_norm(&x, [4], Some(&rw), None).grad_fn().is_none());
}

#[test]
fn embedding_checks_its_arguments() {
    let w = var(&[5, 3], 0);
    let idx = homura::tensor([1i64, 0]);
    for p in [5, -6] {
        assert_eq!(
            panic_message(|| {
                embedding(&idx, &w, Some(p));
            }),
            "Padding_idx must be within num_embeddings"
        );
    }
    for shape in [&[5][..], &[5, 3, 2]] {
        assert_eq!(
            panic_message(|| {
                embedding(&idx, &var(shape, 1), None);
            }),
            "'weight' must be 2-D"
        );
    }
    for dtype in [DType::I16, DType::U8, DType::Bool, DType::F32] {
        assert_eq!(
            panic_message(|| {
                embedding(&idx.to(dtype), &w, None);
            }),
            format!(
                "Expected tensor for argument #1 'indices' to have one of the following scalar \
                 types: i64, i32; but got {dtype} instead (while checking arguments for embedding)"
            )
        );
    }
    for bad in [5i64, -1] {
        assert_eq!(
            panic_message(|| {
                embedding(&homura::tensor([0i64, bad]), &w, None);
            }),
            "index out of range in self"
        );
    }
    // Tables without columns or rows: PyTorch's index_select reports these with other messages
    // ("INDICES element is out of DATA bounds, id=3 axis_dim=3", "index_select(): self indexing
    // axis dim should be positive"); homura with the same one.
    for (shape, index) in [([3, 0], 3i64), ([0, 3], 0)] {
        assert_eq!(
            panic_message(|| {
                embedding(&homura::tensor([index]), &var(&shape, 2), None);
            }),
            "index out of range in self"
        );
    }
    // -5 is the first row as a padding index.
    let out = embedding(&homura::tensor([0i64, 4]), &w, Some(-5));
    let g = homura::ones_with([2, 3], DType::F64);
    let grad = grads(&out, &[&w], &g, false)[0].clone().unwrap();
    assert_eq!(as_f64(&grad.select(0, 0)), [0.0; 3]);
    assert_eq!(as_f64(&grad.select(0, 4)), [1.0; 3]);
    // Meta indices have no values to check; the shapes follow.
    let meta_w = homura::zeros_with([5, 3], (DType::F32, Device::Meta));
    meta_w.requires_grad_(true);
    let meta_idx = homura::zeros_with([2, 4], (DType::I64, Device::Meta));
    let out = embedding(&meta_idx, &meta_w, Some(0));
    assert_eq!(out.shape().to_vec(), [2, 4, 3]);
    let grad = autograd::grad(&[&out.sum()], &[&meta_w], GradOptions::default());
    let grad = grad[0].clone().unwrap();
    assert_eq!(
        (grad.device(), grad.shape().to_vec()),
        (Device::Meta, vec![5, 3])
    );
}

#[test]
fn embedding_padding_rows_and_saved_indices() {
    // The padding row is looked up with its current value, here not zeros, and gets no
    // gradient, even an infinite or NaN one.
    let w = Tensor::from_vec(vec![1.0, 2.0, 3.0, f64::INFINITY, 5.0, 6.0], [3, 2]);
    w.requires_grad_(true);
    let idx = homura::tensor([1i64, 0, 1]);
    let out = embedding(&idx, &w, Some(1));
    assert_eq!(
        as_f64(&out),
        [3.0, f64::INFINITY, 1.0, 2.0, 3.0, f64::INFINITY]
    );
    let g = Tensor::from_vec(vec![f64::NAN, 1.0, 2.0, 3.0, f64::INFINITY, 4.0], [3, 2]);
    let grad = grads(&out, &[&w], &g, false)[0].clone().unwrap();
    assert_eq!(as_f64(&grad), [2.0, 3.0, 0.0, 0.0, 0.0, 0.0]);
    // Repeated indices add up; the gradient doesn't depend on the weights.
    let g = homura::ones_with([3, 2], DType::F64);
    g.requires_grad_(true);
    let out = embedding(&idx, &w, None);
    let grad = grads(&out, &[&w], &g, true)[0].clone().unwrap();
    assert_eq!(as_f64(&grad), [1.0, 1.0, 2.0, 2.0, 0.0, 0.0]);
    let second = autograd::grad(
        &[&grad.sum()],
        &[&w, &g],
        GradOptions {
            allow_unused: true,
            ..Default::default()
        },
    );
    assert!(second[0].is_none());
    assert_eq!(as_f64(second[1].as_ref().unwrap()), [1.0; 6]);
    // The indices are saved: changing them in place before the backward pass is an error.
    let idx = homura::tensor([0i64, 2]);
    let out = embedding(&idx, &w, None);
    idx.add_(0);
    let message = panic_message(|| {
        out.sum().backward();
    });
    assert!(
        message.starts_with("one of the variables needed for gradient computation"),
        "{message}"
    );
    // Integer and bool tables, and tables that aren't contiguous.
    let table = Tensor::from_vec((0i64..6).collect(), [3, 2]);
    assert_eq!(
        embedding(&homura::tensor([2i64, 0]), &table, None).to_vec::<i64>(),
        [4, 5, 0, 1]
    );
    let flags = homura::ones_with([3, 2], DType::Bool);
    assert_eq!(embedding(&idx, &flags, None).dtype(), DType::Bool);
    let wt = var(&[2, 3], 2).transpose(0, 1);
    let rows = embedding(&homura::tensor([[2i64, 1]]), &wt, None);
    assert!(rows.is_contiguous());
    assert_eq!(
        as_f64(&rows),
        [as_f64(&wt.select(0, 2)), as_f64(&wt.select(0, 1))].concat()
    );
    // I32 indices and empty lookups.
    let out = embedding(&homura::tensor([2i32, 2]), &w, None);
    assert_eq!(as_f64(&out), [5.0, 6.0, 5.0, 6.0]);
    let empty = embedding(&homura::zeros_with([0, 3], DType::I64), &w, Some(0));
    assert_eq!(empty.shape().to_vec(), [0, 3, 2]);
    let g = homura::zeros_with([0, 3, 2], DType::F64);
    assert_eq!(
        as_f64(&grads(&empty, &[&w], &g, false)[0].clone().unwrap()),
        [0.0; 6]
    );
}

#[test]
fn layer_norm_checks_its_arguments() {
    let x = var(&[2, 3, 4], 0);
    let message = |f: &dyn Fn()| panic_message(f);
    assert_eq!(
        message(&|| {
            layer_norm(&x, [0usize; 0], None, None, 1e-5);
        }),
        "Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one \
         element, but got normalized_shape = []"
    );
    assert_eq!(
        message(&|| {
            layer_norm(&x, [4], Some(&var(&[3], 1)), None, 1e-5);
        }),
        "Expected weight to be of same shape as normalized_shape, but got weight of shape [3] and \
         normalized_shape = [4]"
    );
    assert_eq!(
        message(&|| {
            layer_norm(&x, [4], None, Some(&var(&[2, 2], 1)), 1e-5);
        }),
        "Expected bias to be of same shape as normalized_shape, but got bias of shape [2, 2] and \
         normalized_shape = [4]"
    );
    assert_eq!(
        message(&|| {
            layer_norm(&x, [2, 4], None, None, 1e-5);
        }),
        "Given normalized_shape=[2, 4], expected input with shape [*, 2, 4], but got input of \
         size[2, 3, 4]"
    );
    assert_eq!(
        message(&|| {
            layer_norm(&var(&[4], 1), [2, 4], None, None, 1e-5);
        }),
        "Given normalized_shape=[2, 4], expected input with shape [*, 2, 4], but got input of \
         size[4]"
    );
    assert_eq!(
        message(&|| {
            layer_norm(
                &homura::ones_with([2, 4], DType::I64),
                [4],
                None,
                None,
                1e-5,
            );
        }),
        "layer_norm is not implemented for i64 tensors"
    );
    // PyTorch's mixed dtype rules: a half precision input may have F32 parameters.
    let p = |dtype: DType| homura::ones_with([4], dtype);
    let input = |dtype: DType| homura::ones_with([2, 4], dtype);
    let mixed = [
        (DType::F32, Some(DType::F64), None),
        (DType::F32, Some(DType::F16), None),
        (DType::F16, Some(DType::BF16), None),
        (DType::F16, Some(DType::F32), Some(DType::F16)),
        (DType::F32, None, Some(DType::I64)),
    ];
    for (x, w, b) in mixed {
        assert_eq!(
            message(&|| {
                layer_norm(&input(x), [4], w.map(p).as_ref(), b.map(p).as_ref(), 1e-5);
            }),
            "mixed dtype (CPU): expect parameter to have scalar type of Float",
            "{x} {w:?} {b:?}"
        );
    }
    assert_eq!(
        message(&|| {
            layer_norm(&input(DType::F64), [4], Some(&p(DType::F32)), None, 1e-5);
        }),
        "mixed dtype (CPU): all inputs must share same datatype."
    );
    // Not mixed (the first parameter has the input's dtype), but the second differs.
    for (x, b) in [(DType::F32, DType::F64), (DType::F16, DType::F32)] {
        assert_eq!(
            message(&|| {
                layer_norm(&input(x), [4], Some(&p(x)), Some(&p(b)), 1e-5);
            }),
            format!("expected scalar type {x} but found {b}")
        );
    }
    for (x, w, b) in [
        (DType::F16, Some(DType::F32), Some(DType::F32)),
        (DType::BF16, Some(DType::F32), None),
        (DType::F16, None, Some(DType::F32)),
    ] {
        let out = layer_norm(&input(x), [4], w.map(p).as_ref(), b.map(p).as_ref(), 1e-5);
        assert_eq!(out.dtype(), x);
    }
}

#[test]
fn layer_norm_edge_cases() {
    // Without a bias, PyTorch adds zero: -0 becomes 0.
    let x = homura::tensor([[1.0f32, 3.0]]);
    let w = homura::tensor([0.0f32, 1.0]);
    let y = layer_norm(&x, [2], Some(&w), None, 0.0);
    assert_eq!(y.to_vec::<f32>(), [0.0, 1.0]);
    assert!(y.to_vec::<f32>()[0].is_sign_positive());
    // A negative or NaN eps gives NaN, not an error.
    let x = homura::tensor([[1.0f32, 2.0]]);
    assert!(
        layer_norm(&x, [2], None, None, -1.0)
            .to_vec::<f32>()
            .iter()
            .all(|v| v.is_nan())
    );
    assert!(
        layer_norm(&x, [2], None, None, f64::NAN)
            .to_vec::<f32>()
            .iter()
            .all(|v| v.is_nan())
    );
    // Without normalized elements, the result and the gradients are empty.
    let x = var(&[2, 0], 0);
    let (w, b) = (var(&[0], 1), var(&[0], 2));
    let out = layer_norm(&x, [0], Some(&w), Some(&b), 1e-5);
    assert_eq!(out.shape().to_vec(), [2, 0]);
    let g = homura::zeros_with([2, 0], DType::F64);
    let shapes: Vec<Vec<usize>> = grads(&out, &[&x, &w, &b], &g, false)
        .into_iter()
        .map(|t| t.unwrap().shape().to_vec())
        .collect();
    assert_eq!(shapes, [vec![2, 0], vec![0], vec![0]]);
    // Meta: shapes of the result and the gradients.
    let meta = |shape: &[usize], dtype: DType| {
        let t = homura::zeros_with(shape, (dtype, Device::Meta));
        t.requires_grad_(true);
        t
    };
    let (x, w, b) = (
        meta(&[2, 3, 4], DType::F16),
        meta(&[3, 4], DType::F32),
        meta(&[3, 4], DType::F32),
    );
    let out = layer_norm(&x, [3, 4], Some(&w), Some(&b), 1e-5);
    assert_eq!(
        (out.dtype(), out.shape().to_vec()),
        (DType::F16, vec![2, 3, 4])
    );
    let g = homura::zeros_with([2, 3, 4], (DType::F16, Device::Meta));
    let found: Vec<(DType, Vec<usize>)> = grads(&out, &[&x, &w, &b], &g, false)
        .into_iter()
        .map(|t| {
            let t = t.unwrap();
            assert_eq!(t.device(), Device::Meta);
            (t.dtype(), t.shape().to_vec())
        })
        .collect();
    assert_eq!(
        found,
        [
            (DType::F16, vec![2, 3, 4]),
            (DType::F32, vec![3, 4]),
            (DType::F32, vec![3, 4])
        ]
    );
    // Only the bias's gradient (the sum of g over the rows); the input is still saved and
    // checked.
    let (x, w, b) = (constant(&[3, 4], 0), constant(&[4], 1), var(&[4], 2));
    let g = constant(&[3, 4], 3);
    let out = layer_norm(&x, [4], Some(&w), Some(&b), 1e-5);
    let grad = grads(&out, &[&b], &g, false)[0].clone().unwrap();
    assert_eq!(as_f64(&grad), as_f64(&g.sum_dim(0, false)));
    let out = layer_norm(&x, [4], Some(&w), Some(&b), 1e-5);
    x.add_(1.0);
    let message = panic_message(|| {
        grads(&out, &[&b], &g, false);
    });
    assert!(
        message.starts_with("one of the variables needed for gradient computation"),
        "{message}"
    );
    // The weight and the bias are saved: changing them before the backward pass is an error.
    for changed in [1, 2] {
        let (x, w, b) = (var(&[2, 4], 0), var(&[4], 1), var(&[4], 2));
        let out = layer_norm(&x, [4], Some(&w), Some(&b), 1e-5);
        {
            let _guard = homura::no_grad();
            [&w, &b][changed - 1].add_(1.0);
        }
        let message = panic_message(|| {
            out.sum().backward();
        });
        assert!(
            message.starts_with("one of the variables needed for gradient computation"),
            "{message}"
        );
    }
}

#[test]
fn layer_norm_bias_without_weight_has_second_derivatives() {
    // PyTorch's layer_norm_double_backward ignores the gradient of the bias's gradient without
    // a weight (its result is None, or an error together with the input's). The bias's gradient
    // is the sum of g over the rows, so its derivative with respect to g is the broadcast of gg.
    let (x, b) = (var(&[3, 4], 0), var(&[4], 1));
    let g = var(&[3, 4], 2);
    let out = layer_norm(&x, [4], None, Some(&b), 1e-5);
    let first = grads(&out, &[&x, &b], &g, true);
    let (gx, gb) = (first[0].clone().unwrap(), first[1].clone().unwrap());
    let gg = constant(&[4], 3);
    let second = autograd::grad(
        &[&gb.mul(&gg).sum()],
        &[&x, &b, &g],
        GradOptions {
            allow_unused: true,
            retain_graph: Some(true),
            ..Default::default()
        },
    );
    assert!(
        second[0]
            .as_ref()
            .is_none_or(|t| as_f64(t).iter().all(|&v| v == 0.0))
    );
    assert!(second[1].is_none());
    assert_eq!(
        as_f64(second[2].as_ref().unwrap()),
        as_f64(&gg.expand([3, 4]).contiguous())
    );
    // Together with the input's gradient (which errors in PyTorch).
    let total = gx.mul(constant(&[3, 4], 4)).sum().add(gb.mul(&gg).sum());
    let both = autograd::grad(&[&total], &[&x, &g], GradOptions::default());
    assert!(both.iter().all(Option::is_some));
}

#[test]
fn rms_norm_checks_its_arguments() {
    let x = var(&[2, 4], 0);
    let message = |f: &dyn Fn()| panic_message(f);
    assert_eq!(
        message(&|| {
            rms_norm(&x, [0usize; 0], None, None);
        }),
        "Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one \
         element, but got normalized_shape = []"
    );
    assert_eq!(
        message(&|| {
            rms_norm(&x, [4], Some(&var(&[3], 1)), None);
        }),
        "Expected weight to be of same shape as normalized_shape, but got weight of shape [3] and \
         normalized_shape = [4]"
    );
    assert_eq!(
        message(&|| {
            rms_norm(&var(&[2, 3, 4], 1), [2, 4], None, None);
        }),
        "Given normalized_shape=[2, 4], expected input with shape [*2, 4], but got input of \
         size[2, 3, 4]"
    );
    assert_eq!(
        message(&|| {
            rms_norm(&var(&[4], 1), [2, 4], None, None);
        }),
        "Input tensor must have at least 2 dimensions, but got 1"
    );
    assert_eq!(
        message(&|| {
            rms_norm(&homura::ones_with([2, 4], DType::I64), [4], None, None);
        }),
        "rms_norm is not implemented for i64 tensors"
    );
    // The default eps is the machine epsilon of the computation, F32 for half precision.
    let small = |dtype: DType, v: f64| homura::full_with([1, 2], v, dtype);
    let expect = |v: f64, eps: f64| v / (v * v + eps).sqrt();
    let f32_eps = f32::EPSILON as f64;
    let y = rms_norm(&small(DType::F64, 1e-9), [2], None, None);
    assert!((as_f64(&y)[0] - expect(1e-9, f64::EPSILON)).abs() < 1e-12);
    for dtype in [DType::F32, DType::F16] {
        let y = rms_norm(&small(dtype, 1e-4), [2], None, None);
        let v = small(dtype, 1e-4).to(DType::F64).to_vec::<f64>()[0];
        assert!((as_f64(&y)[0] - expect(v, f32_eps)).abs() < 1e-3, "{dtype}");
    }
    assert_eq!(
        as_f64(&rms_norm(&small(DType::F32, 0.0), [2], None, None)),
        [0.0, 0.0]
    );
    // A weight of another dtype: the result has the input's dtype.
    let w = homura::ones_with([4], DType::F64);
    assert_eq!(
        rms_norm(&x.to(DType::F32), [4], Some(&w), None).dtype(),
        DType::F32
    );
    // Meta.
    let xm = homura::zeros_with([3, 4], (DType::BF16, Device::Meta));
    xm.requires_grad_(true);
    let out = rms_norm(&xm, [4], None, Some(1e-6));
    assert_eq!(
        (out.dtype(), out.shape().to_vec()),
        (DType::BF16, vec![3, 4])
    );
    let g = autograd::grad(&[&out.sum()], &[&xm], GradOptions::default());
    assert_eq!(g[0].as_ref().unwrap().shape().to_vec(), [3, 4]);
}

#[test]
fn embed_norm_gradients_pass_gradcheck() {
    let options = GradcheckOptions::default;
    let check = |what: &str, f: &dyn Fn(&[Tensor]) -> Vec<Tensor>, inputs: &[&Tensor]| {
        gradcheck(f, inputs, options()).unwrap_or_else(|e| panic!("{what}: {e}"));
        gradgradcheck(f, inputs, options()).unwrap_or_else(|e| panic!("{what}: {e}"));
    };
    let idx = homura::tensor([[1i64, 3, 1], [0, 3, 2]]);
    let w = var(&[4, 2], 0);
    check("embedding", &|t| vec![embedding(&idx, &t[0], None)], &[&w]);
    // The padding row's zero gradient is not its derivative, which gradcheck would measure;
    // the gradient's own derivatives are checked.
    let padded = |t: &[Tensor]| vec![embedding(&idx, &t[0], Some(3))];
    gradgradcheck(padded, &[&w], options()).unwrap_or_else(|e| panic!("embedding padding: {e}"));
    let (x, w, b) = (var(&[2, 3, 4], 0), var(&[3, 4], 1), var(&[3, 4], 2));
    check(
        "layer_norm",
        &|t| vec![layer_norm(&t[0], [3, 4], Some(&t[1]), Some(&t[2]), 1e-5)],
        &[&x, &w, &b],
    );
    check(
        "layer_norm last dim",
        &|t| vec![layer_norm(&t[0], [4], Some(&t[1]), None, 1e-5)],
        &[&x, &var(&[4], 3)],
    );
    check(
        "layer_norm bias only",
        &|t| vec![layer_norm(&t[0], [4], None, Some(&t[1]), 1e-5)],
        &[&x, &var(&[4], 4)],
    );
    check(
        "layer_norm plain",
        &|t| vec![layer_norm(&t[0], [5], None, None, 1e-5)],
        &[&var(&[2, 5], 5)],
    );
    check(
        "rms_norm",
        &|t| vec![rms_norm(&t[0], [3, 4], Some(&t[1]), Some(1e-6))],
        &[&x, &w],
    );
    check(
        "rms_norm plain",
        &|t| vec![rms_norm(&t[0], [4], None, None)],
        &[&x],
    );
}

#[test]
fn embedding_layer() {
    homura::manual_seed(0);
    let table = Embedding::new(1000, 16);
    assert_eq!(table.weight.shape().to_vec(), [1000, 16]);
    assert_eq!(table.weight.dtype(), DType::F32);
    assert!(table.weight.requires_grad() && table.padding_idx.is_none());
    // N(0, 1).
    let values = as_f64(&table.weight);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    assert!(
        mean.abs() < 0.05 && (var - 1.0).abs() < 0.05,
        "{mean} {var}"
    );
    let table = Embedding::new(6, 3).padding_idx(-2);
    assert_eq!(table.padding_idx, Some(4));
    assert_eq!(as_f64(&table.weight.select(0, 4)), [0.0; 3]);
    assert!(table.weight.is_leaf());
    assert_eq!(
        format!("{table:?}"),
        "Embedding { num_embeddings: 6, embedding_dim: 3, padding_idx: Some(4), .. }"
    );
    assert_eq!(
        panic_message(|| {
            let _ = Embedding::new(6, 3).padding_idx(6);
        }),
        "Padding_idx must be within num_embeddings"
    );
    assert_eq!(table.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
    let idx = homura::tensor([[4i64, 0]]);
    let out = table.forward(&idx);
    out.sum().backward();
    let grad = table.weight.grad().unwrap();
    assert_eq!(as_f64(&grad.select(0, 4)), [0.0; 3]);
    assert_eq!(as_f64(&grad.select(0, 0)), [1.0; 3]);
    table.reset_parameters();
    assert_eq!(as_f64(&table.weight.select(0, 4)), [0.0; 3]);
    // In a Sequential, and converted with the module.
    let model = Sequential::new()
        .append(Embedding::new(5, 4))
        .append(LayerNorm::new([4]));
    model.to(DType::F64);
    assert_eq!(
        model.state_dict().keys().collect::<Vec<_>>(),
        ["0.weight", "1.weight", "1.bias"]
    );
    let y = model.forward(&homura::tensor([1i64, 2, 3]));
    assert_eq!((y.dtype(), y.shape().to_vec()), (DType::F64, vec![3, 4]));
}

#[test]
fn norm_layers() {
    let norm = LayerNorm::new([2, 3]);
    assert_eq!(norm.normalized_shape, [2, 3]);
    assert_eq!(norm.eps, 1e-5);
    assert_eq!(as_f64(norm.weight.as_ref().unwrap()), [1.0; 6]);
    assert_eq!(as_f64(norm.bias.as_ref().unwrap()), [0.0; 6]);
    assert_eq!(
        norm.state_dict().keys().collect::<Vec<_>>(),
        ["weight", "bias"]
    );
    assert_eq!(
        format!("{:?}", LayerNorm::new([4]).bias(false)),
        "LayerNorm { normalized_shape: [4], eps: 1e-5, elementwise_affine: true, bias: false, .. }"
    );
    let plain = LayerNorm::new([4]).elementwise_affine(false);
    assert!(plain.weight.is_none() && plain.bias.is_none());
    assert!(plain.state_dict().is_empty());
    // A bias needs elementwise affine, as in PyTorch.
    assert!(plain.bias(true).bias.is_none());
    let restored = LayerNorm::new([4]).bias(false).elementwise_affine(true);
    assert!(restored.weight.is_some() && restored.bias.is_some());
    // The forward computation is functional::layer_norm with the layer's options.
    let x = var(&[2, 3], 0).to(DType::F32);
    let norm = LayerNorm::new([3]).eps(1e-3);
    {
        let _guard = homura::no_grad();
        norm.weight.as_ref().unwrap().mul_(2.0);
    }
    let expected = layer_norm(
        &x,
        [3],
        Some(&homura::full([3], 2.0)),
        Some(&homura::zeros([3])),
        1e-3,
    );
    assert_eq!(norm.forward(&x).to_vec::<f32>(), expected.to_vec::<f32>());
    norm.reset_parameters();
    assert_eq!(as_f64(norm.weight.as_ref().unwrap()), [1.0; 3]);
    // An empty shape is accepted by the constructor and rejected by forward, like PyTorch.
    let empty = LayerNorm::new([0usize; 0]);
    assert!(
        panic_message(|| {
            empty.forward(&x);
        })
        .starts_with("Expected normalized_shape to be at least 1-dimensional")
    );

    let rms = RmsNorm::new([3]);
    assert_eq!(rms.eps, None);
    assert_eq!(rms.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
    assert_eq!(
        format!("{:?}", rms.eps(1e-6)),
        "RmsNorm { normalized_shape: [3], eps: Some(1e-6), elementwise_affine: true, .. }"
    );
    let rms = RmsNorm::new([3])
        .eps(1e-6)
        .eps(None)
        .elementwise_affine(false);
    assert!(rms.weight.is_none() && rms.eps.is_none());
    assert_eq!(
        rms.forward(&x).to_vec::<f32>(),
        rms_norm(&x, [3], None, None).to_vec::<f32>()
    );
    let rms = rms.elementwise_affine(true);
    assert_eq!(as_f64(rms.weight.as_ref().unwrap()), [1.0; 3]);
    // Both layers are Forward.
    let layers: [&dyn Forward; 2] = [&LayerNorm::new([3]), &RmsNorm::new([3])];
    for layer in layers {
        assert_eq!(layer.forward(&x).shape().to_vec(), [2, 3]);
    }
}

/// Review round 57 (R67): parameters that the builders add take the dtype and device of the
/// parameter the layer has, and the parameters it has stay.
#[test]
fn layer_norm_new_parameters_follow_the_existing_ones() {
    let check_new = |p: &homura::nn::Parameter, value: f64, dtype: DType, device: Device| {
        assert_eq!(
            (p.dtype(), p.device(), p.shape().to_vec()),
            (dtype, device, vec![4])
        );
        assert!(p.is_leaf() && p.requires_grad() && p.grad_fn().is_none());
        if device == Device::Cpu {
            assert_eq!(as_f64(p), [value; 4]);
        }
    };
    let x = constant(&[2, 4], 0);
    let g = constant(&[2, 4], 1);
    for device in [Device::Cpu, Device::Meta] {
        let target = |norm: &LayerNorm| {
            norm.to(DType::F64);
            norm.to(device);
        };
        // bias(true) after the conversion.
        let norm = LayerNorm::new([4]).bias(false);
        target(&norm);
        let weight = norm.weight.clone().unwrap();
        let norm = norm.bias(true);
        assert!(norm.weight.as_ref().unwrap().is_same(&weight));
        check_new(norm.bias.as_ref().unwrap(), 0.0, DType::F64, device);
        assert_eq!(
            norm.state_dict().keys().collect::<Vec<_>>(),
            ["weight", "bias"]
        );
        // elementwise_affine(true) adds the bias like the weight.
        let norm = LayerNorm::new([4]).bias(false);
        target(&norm);
        let norm = norm.elementwise_affine(true);
        check_new(norm.bias.as_ref().unwrap(), 0.0, DType::F64, device);
        // A layer with only a bias gets a weight like the bias.
        let mut norm = LayerNorm::new([4]);
        target(&norm);
        norm.weight = None;
        let bias = norm.bias.clone().unwrap();
        let norm = norm.elementwise_affine(true);
        assert!(norm.bias.as_ref().unwrap().is_same(&bias));
        check_new(norm.weight.as_ref().unwrap(), 1.0, DType::F64, device);
        // The layer computes, and both parameters get gradients of their dtype and device.
        let (x, g) = (x.to(device), g.to(device));
        x.requires_grad_(true);
        let out = norm.forward(&x);
        assert_eq!((out.dtype(), out.device()), (DType::F64, device));
        let weight = norm.weight.as_deref().unwrap();
        let bias = norm.bias.as_deref().unwrap();
        for grad in grads(&out, &[&x, weight, bias], &g, false) {
            let grad = grad.unwrap();
            assert_eq!((grad.dtype(), grad.device()), (DType::F64, device));
        }
        if device == Device::Cpu {
            let expected = layer_norm(&x, [4], Some(weight), Some(bias), 1e-5);
            assert_eq!(as_f64(&out), as_f64(&expected));
        }
    }
}

/// Each subset of the inputs that require grad gets the gradients of the full set: the backward
/// pass computes only what is requested.
#[test]
fn layer_norm_gradients_of_each_subset() {
    let g = constant(&[3, 4], 3);
    let make = || (constant(&[3, 4], 0), constant(&[4], 1), constant(&[4], 2));
    let (x, w, b) = make();
    for t in [&x, &w, &b] {
        t.requires_grad_(true);
    }
    let out = layer_norm(&x, [4], Some(&w), Some(&b), 1e-5);
    let full: Vec<Vec<f64>> = grads(&out, &[&x, &w, &b], &g, false)
        .iter()
        .map(|t| as_f64(t.as_ref().unwrap()))
        .collect();
    for mask in 1..8usize {
        let (x, w, b) = make();
        let inputs = [&x, &w, &b];
        let chosen: Vec<usize> = (0..3).filter(|i| mask & (1 << i) != 0).collect();
        for &i in &chosen {
            inputs[i].requires_grad_(true);
        }
        let out = layer_norm(&x, [4], Some(&w), Some(&b), 1e-5);
        let variables: Vec<&Tensor> = chosen.iter().map(|&i| inputs[i]).collect();
        for (&i, grad) in chosen.iter().zip(grads(&out, &variables, &g, false)) {
            assert_eq!(
                as_f64(&grad.unwrap()),
                full[i],
                "mask {mask:03b}, input {i}"
            );
        }
    }
}
