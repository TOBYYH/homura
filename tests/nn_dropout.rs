//! M2 step 5: `nn::functional::dropout` and the `Dropout` layer. The masks come from homura's
//! random number generator, which differs from PyTorch's; the values of the kept elements and
//! the other results are compared with the `dropout_*` cases of
//! `tests/golden/m2_embed_norm.safetensors` from `tools/golden/gen_m2_embed_norm.py`, and the
//! reference behavior encoded in the other tests is asserted by
//! `tools/golden/check_m2_embed_norm.py`.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard};

use common::{Case, Golden};
use homura::autograd::{self, GradOptions};
use homura::nn::functional::dropout;
use homura::nn::{Dropout, Module, Training};
use homura::{DType, Device, Tensor};

/// The tests of this file share the global random number generator: each takes this lock, so
/// that one test's draws don't change another's.
static RNG: Mutex<()> = Mutex::new(());

fn rng() -> MutexGuard<'static, ()> {
    RNG.lock().unwrap_or_else(|e| e.into_inner())
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn as_f64(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

/// Asserts that `actual` has the dtype of `expected` and the same values, bit for bit (NaN
/// matches NaN, and the sign of zeros counts).
fn assert_same_values(actual: &Tensor, expected: &Tensor, what: &str) {
    assert_eq!(actual.dtype(), expected.dtype(), "{what}: dtype");
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    let (a, e) = (as_f64(actual), as_f64(expected));
    for (i, (x, y)) in a.iter().zip(&e).enumerate() {
        let same = (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
        assert!(same, "{what}: element {i}: got {x:e}, expected {y:e}");
    }
}

fn grad_of(out: &Tensor, x: &Tensor, g: &Tensor, create_graph: bool) -> Tensor {
    autograd::grad(
        &[out],
        &[x],
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            create_graph,
            ..Default::default()
        },
    )[0]
    .clone()
    .unwrap()
}

#[test]
fn dropout_matches_pytorch() {
    let _rng = rng();
    homura::manual_seed(0);
    let golden = Golden::load("m2_embed_norm.safetensors");
    let cases: Vec<String> = golden
        .cases()
        .into_iter()
        .filter(|c| c.starts_with("dropout"))
        .collect();
    assert_eq!(cases.len(), 3 * 4);
    for name in &cases {
        let c = Case::new(&golden, name);
        match name.split('.').nth(1).unwrap() {
            "scale" => {
                let ps = c.input("p").to_vec::<f64>();
                let noise = c.input("noise");
                let dtype = noise.dtype();
                for (i, &p) in ps.iter().enumerate() {
                    let out = dropout(&homura::ones_with([20000], dtype), p, true);
                    let kept = out.not_equal(0.0);
                    let count = kept.sum().item::<i64>();
                    assert!(count > 0, "{name} p={p}");
                    let expected = noise.select(0, i as i64).expand([count]);
                    let values = homura::where_(&kept, &out, f64::NAN);
                    let values: Vec<f64> = as_f64(&values)
                        .into_iter()
                        .filter(|v| !v.is_nan())
                        .collect();
                    assert_eq!(values, as_f64(&expected), "{name} p={p}");
                }
            }
            "values" => {
                let (x, g) = (c.input("x"), c.input("g"));
                let p = c.input("p").item::<f64>();
                let (x_noise, g_noise) = (c.input("x_noise"), c.input("g_noise"));
                assert!(x.not_equal(0.0).all().item::<bool>());
                x.requires_grad_(true);
                let out = dropout(&x, p, true);
                let kept = out.not_equal(0.0);
                let dropped = x.detach().mul(0.0);
                assert_same_values(&out, &homura::where_(&kept, &x_noise, &dropped), name);
                let grad = grad_of(&out, &x, &g, false);
                let dropped = g.mul(0.0);
                assert_same_values(&grad, &homura::where_(&kept, &g_noise, &dropped), name);
            }
            "p1" => {
                let (x, g) = (c.input("x"), c.input("g"));
                x.requires_grad_(true);
                let out = dropout(&x, 1.0, true);
                c.check_with("out", &out, |a, e| assert_same_values(a, e, name));
                let grad = grad_of(&out, &x, &g, false);
                c.check_with("grad_x", &grad, |a, e| assert_same_values(a, e, name));
            }
            other => panic!("unknown case {other}"),
        }
        c.assert_all_used();
    }
}

#[test]
fn dropout_returns_the_input_without_drawing() {
    let _rng = rng();
    let x = homura::randn([3, 4]);
    let empty = homura::zeros([0, 4]);
    // Whether the call draws random numbers: the next draw after it differs.
    let draws = |f: &dyn Fn() -> Tensor| {
        homura::manual_seed(7);
        let reference = homura::rand([4]).to_vec::<f32>();
        homura::manual_seed(7);
        let out = f();
        (out, homura::rand([4]).to_vec::<f32>() != reference)
    };
    for (what, input, p, training) in [
        ("eval", &x, 0.5, false),
        ("p = 0", &x, 0.0, true),
        ("empty", &empty, 0.5, true),
        ("eval, p = 1", &x, 1.0, false),
    ] {
        let (out, drew) = draws(&|| dropout(input, p, training));
        assert!(out.is_same(input), "{what}");
        assert!(!drew, "{what}");
    }
    let (out, drew) = draws(&|| dropout(&x, 1.0, true));
    assert!(!out.is_same(&x) && !drew);
    assert_eq!(as_f64(&out), [0.0; 12]);
    let (out, drew) = draws(&|| dropout(&x, 0.5, true));
    assert!(!out.is_same(&x) && drew);
    // The same seed gives the same mask.
    let (again, _) = draws(&|| dropout(&x, 0.5, true));
    assert_eq!(as_f64(&out), as_f64(&again));
}

#[test]
fn dropout_checks_the_probability_first() {
    let _rng = rng();
    let x = homura::ones([3]);
    let empty = homura::zeros([0]);
    for (p, shown) in [(1.5, "1.5"), (-0.1, "-0.1"), (f64::NAN, "NaN")] {
        for (input, training) in [(&x, true), (&x, false), (&empty, true)] {
            assert_eq!(
                panic_message(|| {
                    dropout(input, p, training);
                }),
                format!("dropout probability has to be between 0 and 1, but got {shown}")
            );
        }
    }
    assert_eq!(
        panic_message(|| {
            Dropout::new(1.5);
        }),
        "dropout probability has to be between 0 and 1, but got 1.5"
    );
    // Like PyTorch's nn.Dropout, NaN is accepted by the constructor and rejected by forward,
    // also in evaluation mode.
    let layer = Dropout::new(f64::NAN);
    layer.eval();
    assert_eq!(
        panic_message(|| {
            layer.forward(&x);
        }),
        "dropout probability has to be between 0 and 1, but got NaN"
    );
    // Integers and bools: dropped with 0 < p < 1 only for floating point.
    let ints = Tensor::from_vec((0i64..6).collect(), [6]);
    assert_eq!(
        panic_message(|| {
            dropout(&ints, 0.5, true);
        }),
        "result type f32 can't be cast to the desired output type i64"
    );
    assert!(dropout(&ints, 0.0, true).is_same(&ints));
    let zeros = dropout(&ints, 1.0, true);
    assert_eq!(
        (zeros.dtype(), zeros.to_vec::<i64>()),
        (DType::I64, vec![0; 6])
    );
    let flags = homura::ones_with([2], DType::Bool);
    assert_eq!(
        panic_message(|| {
            dropout(&flags, 0.5, true);
        }),
        "result type f32 can't be cast to the desired output type bool"
    );
    assert_eq!(dropout(&flags, 1.0, true).to_vec::<bool>(), [false, false]);
}

#[test]
fn dropout_gradients() {
    let _rng = rng();
    homura::manual_seed(3);
    let x = homura::randn_with([4, 5], DType::F64);
    x.requires_grad_(true);
    let y = x.mul(1.0);
    let out = dropout(&y, 0.4, true);
    let names: Vec<&str> = out.grad_fn().map(|f| f.name()).into_iter().collect();
    assert_eq!(names, ["MulBackward0"]);
    // Only the noise is saved: changing the input in place before the backward pass is fine.
    {
        let _guard = homura::no_grad();
        y.add_(1.0);
    }
    let g = homura::ones_with([4, 5], DType::F64);
    g.requires_grad_(true);
    let grad = grad_of(&out, &x, &g, true);
    let kept = out.detach().not_equal(0.0);
    let expected = kept.to(DType::F64).div(1.0 - 0.4);
    assert_eq!(as_f64(&grad), as_f64(&expected));
    // The gradient doesn't depend on the input; it is linear in g.
    let second = autograd::grad(
        &[&grad.sum()],
        &[&x, &g],
        GradOptions {
            allow_unused: true,
            ..Default::default()
        },
    );
    assert!(second[0].is_none());
    assert_eq!(as_f64(second[1].as_ref().unwrap()), as_f64(&expected));
    // Meta: shapes only.
    let m = homura::zeros_with([2, 3], (DType::F16, Device::Meta));
    m.requires_grad_(true);
    let out = dropout(&m, 0.5, true);
    assert_eq!(
        (out.device(), out.dtype(), out.shape().to_vec()),
        (Device::Meta, DType::F16, vec![2, 3])
    );
    let g = autograd::grad(&[&out.sum()], &[&m], GradOptions::default());
    assert_eq!(g[0].as_ref().unwrap().shape().to_vec(), [2, 3]);
}

#[test]
fn dropout_keeps_one_minus_p() {
    let _rng = rng();
    homura::manual_seed(11);
    let n = 100_000usize;
    for (dtype, p) in [
        (DType::F32, 0.3),
        (DType::F16, 0.3),
        (DType::F64, 0.9),
        (DType::BF16, 0.05),
    ] {
        let out = dropout(&homura::ones_with([n], dtype), p, true);
        let kept = out.not_equal(0.0).sum().item::<i64>() as f64 / n as f64;
        // About seven standard deviations: never fails by chance.
        let sd = (p * (1.0 - p) / n as f64).sqrt();
        assert!((kept - (1.0 - p)).abs() < 7.0 * sd, "{dtype} {p}: {kept}");
    }
}

#[derive(Module)]
struct Block {
    drop: Dropout,
    training: Training,
}

#[test]
fn dropout_layer() {
    let _rng = rng();
    let layer = Dropout::default();
    assert_eq!(layer.p, 0.5);
    assert!(layer.training.get());
    assert!(layer.state_dict().is_empty() && layer.named_parameters().is_empty());
    let x = homura::ones([200]);
    let out = layer.forward(&x);
    assert!(!out.is_same(&x));
    assert!(
        out.equal(0.0)
            .logical_or(out.equal(2.0))
            .all()
            .item::<bool>()
    );
    // The mode of an enclosing module reaches the layer.
    let block = Block {
        drop: Dropout::new(0.25),
        training: Training::default(),
    };
    block.eval();
    assert!(!block.drop.training.get() && !block.training.get());
    assert!(block.drop.forward(&x).is_same(&x));
    block.train();
    assert!(!block.drop.forward(&x).is_same(&x));
    let copy = block.drop.clone();
    assert_eq!(copy.p, 0.25);
}
