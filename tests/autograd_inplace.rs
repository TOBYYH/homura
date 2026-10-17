//! In-place operations recorded for autograd (M1 step 7): the target becomes the output of the
//! operation's node, which replaces its history; the value before the write is saved where the
//! formula needs it; copies and fills give the overwritten value a zero gradient; leaves and
//! views are rejected; failed calls change nothing; views of a rewritten history become stale. tools/golden/check_m1_inplace.py checks the reference
//! behavior that these tests encode.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::autograd::{self, GradOptions};
use homura::{DType, Device, Tensor};

fn t(values: &[f32]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn leaf(values: &[f32]) -> Tensor {
    let x = t(values);
    x.requires_grad_(true);
    x
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to(DType::F32).to_vec()
}

fn ints(values: &[i64]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

/// The gradients of `out.sum()` with respect to `inputs`, `None` where there is none.
fn grads(out: &Tensor, inputs: &[&Tensor]) -> Vec<Option<Tensor>> {
    let options = GradOptions {
        allow_unused: true,
        ..Default::default()
    };
    autograd::grad(&[&out.sum()], inputs, options)
}

fn grad_vals(out: &Tensor, inputs: &[&Tensor]) -> Vec<Vec<f32>> {
    grads(out, inputs)
        .into_iter()
        .map(|g| vals(&g.expect("a gradient")))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1e-6 * e.abs().max(1.0),
            "{actual:?} vs {expected:?}"
        );
    }
}

/// The name of the node of `t` and of the nodes its edges lead to.
fn chain(t: &Tensor) -> (String, Vec<Option<String>>) {
    let grad_fn = t.grad_fn().expect("a history");
    let next = grad_fn
        .next_functions()
        .into_iter()
        .map(|e| e.map(|(f, _)| f.name().to_string()))
        .collect();
    (grad_fn.name().to_string(), next)
}

fn names(node: &str, next: &[Option<&str>]) -> (String, Vec<Option<String>>) {
    let next = next.iter().map(|n| n.map(String::from)).collect();
    (node.to_string(), next)
}

const ACC: Option<&str> = Some("AccumulateGrad");
const MUL: Option<&str> = Some("MulBackward0");

const LEAF: &str = "a leaf Variable that requires grad is being used in an in-place operation.";
const VIEW_OF_LEAF: &str =
    "a view of a leaf Variable that requires grad is being used in an in-place operation.";
const VIEW: &str = "in-place operations on a view are not supported yet when the view or one of \
                    the inputs requires grad; use the out-of-place operation, or deep_clone() the \
                    view first";
const STALE: &str = "a view whose base was modified by an in-place operation in grad mode after \
                     the view was created can't be used for gradient computation yet; create the \
                     view again after the in-place operation";

fn version_error(shape: &str, output: &str, current: u64, expected: u64) -> String {
    format!(
        "one of the variables needed for gradient computation has been modified by an inplace \
         operation: [f32 {shape}]{output} is at version {current}; expected version {expected} \
         instead."
    )
}

/// Counts the writes into the storage of a tensor (its version) through a graph that saved an
/// alias of it: the backward pass reports the versions when they differ.
struct VersionWatch {
    probe: Tensor,
    check: Tensor,
}

impl VersionWatch {
    fn new(t: &Tensor) -> VersionWatch {
        let probe = leaf(&[1.]);
        // Saves the alias, which shares the version counter, for the gradient of `probe`.
        let _grad_mode = homura::enable_grad();
        let check = (&probe * &t.detach()).sum();
        VersionWatch { probe, check }
    }

    /// How many times the version changed since the watch started.
    fn bumps(&self) -> u64 {
        let options = GradOptions {
            retain_graph: Some(true),
            ..Default::default()
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            autograd::grad(&[&self.check], &[&self.probe], options);
        }));
        let Err(payload) = result else {
            return 0;
        };
        let message = payload.downcast_ref::<String>().expect("a message");
        let number = |after: &str| -> u64 {
            let rest = &message[message.find(after).expect(message) + after.len()..];
            rest[..rest.find(|c: char| !c.is_ascii_digit()).unwrap()]
                .parse()
                .unwrap()
        };
        number("is at version ") - number("expected version ")
    }
}

/// Checks that `op` panics with `message` and changes neither the data, the version, the
/// history of `target`, nor whether views of it are stale.
fn assert_unchanged(target: &Tensor, message: &str, op: impl FnOnce()) {
    let data = target.to_vec::<f32>();
    let watch = VersionWatch::new(target);
    let grad_fn = target.grad_fn();
    // Recorded against the current history: a rewritten history would make it stale.
    let view = target.view(target.shape());
    assert_eq!(panic_message(op), message);
    assert_eq!(target.to_vec::<f32>(), data);
    assert_eq!(watch.bumps(), 0);
    match (grad_fn, target.grad_fn()) {
        (None, None) => {}
        (Some(a), Some(b)) => assert!(a.is_same(&b)),
        _ => panic!("the history changed"),
    }
    // Panics if stale.
    let _ = view.grad_fn();
}

// ---------------------------------------------------------------- nodes, edges and gradients

/// An in-place operation on `x * 1` with the values expected after it.
struct Recorded<'a> {
    name: &'static str,
    op: Box<dyn Fn(&Tensor) + 'a>,
    node: &'static str,
    /// The edges after the one of self.
    edges: Vec<Option<&'static str>>,
    /// The tensor operands that require grad.
    operands: Vec<&'a Tensor>,
    value: Vec<f32>,
    /// The gradients of x and of the operands.
    grads: Vec<Vec<f32>>,
}

#[test]
fn in_place_operations_record_the_node_of_the_out_of_place_operation() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    let w = leaf(&[0.25, 0.75]);
    let lo = leaf(&[0.5, 2.5]);
    let hi = leaf(&[1.5, 3.]);
    let s = leaf(&[4.]);
    let cases = [
        Recorded {
            name: "add_",
            op: Box::new(|y| _ = y.add_(&z)),
            node: "AddBackward0",
            edges: vec![ACC],
            operands: vec![&z],
            value: vec![4., 7.],
            grads: vec![vec![1., 1.], vec![1., 1.]],
        },
        Recorded {
            name: "add_ scalar",
            op: Box::new(|y| _ = y.add_(2.0)),
            node: "AddBackward0",
            edges: vec![None],
            operands: vec![],
            value: vec![3., 4.],
            grads: vec![vec![1., 1.]],
        },
        Recorded {
            name: "sub_",
            op: Box::new(|y| _ = y.sub_(&z)),
            node: "SubBackward0",
            edges: vec![ACC],
            operands: vec![&z],
            value: vec![-2., -3.],
            grads: vec![vec![1., 1.], vec![-1., -1.]],
        },
        Recorded {
            name: "mul_",
            op: Box::new(|y| _ = y.mul_(&z)),
            node: "MulBackward0",
            edges: vec![ACC],
            operands: vec![&z],
            value: vec![3., 10.],
            grads: vec![vec![3., 5.], vec![1., 2.]],
        },
        Recorded {
            name: "mul_ scalar",
            op: Box::new(|y| _ = y.mul_(2.0)),
            node: "MulBackward0",
            edges: vec![None],
            operands: vec![],
            value: vec![2., 4.],
            grads: vec![vec![2., 2.]],
        },
        Recorded {
            name: "div_",
            op: Box::new(|y| _ = y.div_(&z)),
            node: "DivBackward0",
            edges: vec![ACC],
            operands: vec![&z],
            value: vec![1. / 3., 0.4],
            grads: vec![vec![1. / 3., 0.2], vec![-1. / 9., -2. / 25.]],
        },
        Recorded {
            name: "pow_ scalar",
            op: Box::new(|y| _ = y.pow_(2.0)),
            node: "PowBackward0",
            edges: vec![],
            operands: vec![],
            value: vec![1., 4.],
            grads: vec![vec![2., 4.]],
        },
        Recorded {
            name: "pow_",
            op: Box::new(|y| _ = y.pow_(&z)),
            node: "PowBackward1",
            edges: vec![ACC],
            operands: vec![&z],
            value: vec![1., 32.],
            grads: vec![vec![3., 80.], vec![0., 32. * 2f32.ln()]],
        },
        Recorded {
            name: "clamp_ scalar",
            op: Box::new(|y| _ = y.clamp_(0.0, 1.5)),
            node: "ClampBackward1",
            edges: vec![],
            operands: vec![],
            value: vec![1., 1.5],
            grads: vec![vec![1., 0.]],
        },
        Recorded {
            name: "clamp_",
            op: Box::new(|y| _ = y.clamp_(&lo, &hi)),
            node: "ClampBackward0",
            edges: vec![ACC, ACC],
            operands: vec![&lo, &hi],
            value: vec![1., 2.5],
            grads: vec![vec![1., 0.], vec![0., 1.], vec![0., 0.]],
        },
        Recorded {
            name: "addcmul_",
            op: Box::new(|y| _ = y.addcmul_(&z, &w, 2.0)),
            node: "AddcmulBackward0",
            edges: vec![ACC, ACC],
            operands: vec![&z, &w],
            value: vec![2.5, 9.5],
            grads: vec![vec![1., 1.], vec![0.5, 1.5], vec![6., 10.]],
        },
        Recorded {
            name: "addcdiv_",
            op: Box::new(|y| _ = y.addcdiv_(&z, &w, 2.0)),
            node: "AddcdivBackward0",
            edges: vec![ACC, ACC],
            operands: vec![&z, &w],
            value: vec![25., 2. + 40. / 3.],
            grads: vec![vec![1., 1.], vec![8., 8. / 3.], vec![-96., -10. / 0.5625]],
        },
        Recorded {
            name: "lerp_ scalar",
            op: Box::new(|y| _ = y.lerp_(&z, 0.25)),
            node: "LerpBackward0",
            edges: vec![ACC],
            operands: vec![&z],
            value: vec![1.5, 2.75],
            grads: vec![vec![0.75, 0.75], vec![0.25, 0.25]],
        },
        Recorded {
            name: "lerp_",
            op: Box::new(|y| _ = y.lerp_(&z, &w)),
            node: "LerpBackward1",
            edges: vec![ACC, ACC],
            operands: vec![&z, &w],
            value: vec![1.5, 4.25],
            grads: vec![vec![0.75, 0.25], vec![0.25, 0.75], vec![2., 3.]],
        },
        Recorded {
            name: "index_add_",
            op: Box::new(|y| _ = y.index_add_(-1, &ints(&[1]), &s)),
            node: "IndexAddBackward0",
            edges: vec![ACC],
            operands: vec![&s],
            value: vec![1., 6.],
            grads: vec![vec![1., 1.], vec![1.]],
        },
        Recorded {
            name: "scatter_add_",
            op: Box::new(|y| _ = y.scatter_add_(-1, &ints(&[0]), &s)),
            node: "ScatterAddBackward0",
            edges: vec![ACC],
            operands: vec![&s],
            value: vec![5., 2.],
            grads: vec![vec![1., 1.], vec![1.]],
        },
    ];
    for case in &cases {
        let name = case.name;
        let y = &x * 1.0;
        let old = y.grad_fn().unwrap();
        let watch = VersionWatch::new(&y);
        (case.op)(&y);
        assert_eq!(watch.bumps(), 1, "{name}");
        let mut next = vec![MUL];
        next.extend(&case.edges);
        assert_eq!(chain(&y), names(case.node, &next), "{name}");
        // The edge of self leads to the old history itself.
        let (first, _) = y.grad_fn().unwrap().next_functions()[0].clone().unwrap();
        assert!(first.is_same(&old), "{name}");
        assert!(y.requires_grad() && !y.is_leaf(), "{name}");
        assert_close(&vals(&y), &case.value);
        let mut inputs = vec![&x];
        inputs.extend(&case.operands);
        let grads = grad_vals(&y, &inputs);
        assert_eq!(grads.len(), case.grads.len(), "{name}");
        for (g, e) in grads.iter().zip(&case.grads) {
            assert_close(g, e);
        }
    }
}

#[test]
fn masked_fill_is_recorded_as_where() {
    let x = leaf(&[1., 2.]);
    let mask = homura::tensor([true, false]);
    let y = &x * 1.0;
    y.masked_fill_(&mask, 7.0);
    assert_eq!(vals(&y), [7., 2.]);
    // PyTorch records MaskedFillBackward0; homura's masked_fill is where_ (accepted in 12.3.5).
    assert_eq!(chain(&y), names("WhereBackward0", &[None, MUL]));
    assert_eq!(grad_vals(&y, &[&x]), [[0., 1.]]);
    // A tensor value is converted to the dtype of self by a recorded `to`.
    let v = Tensor::from_slice(&[5.0f64], []);
    v.requires_grad_(true);
    let y = &x * 1.0;
    y.masked_fill_(&mask, &v);
    assert_eq!(vals(&y), [5., 2.]);
    assert_eq!(
        chain(&y),
        names("WhereBackward0", &[Some("ToCopyBackward0"), MUL])
    );
    let g = grads(&y, &[&x, &v]);
    assert_eq!(vals(g[0].as_ref().unwrap()), [0., 1.]);
    let gv = g[1].as_ref().unwrap();
    assert_eq!((gv.dtype(), gv.to_vec::<f64>()), (DType::F64, vec![1.0]));
}

#[test]
fn tensors_that_do_not_require_grad_join_the_graph() {
    let z = leaf(&[3., 5.]);
    let b = homura::zeros([2]);
    b.add_(&z);
    assert!(b.requires_grad() && !b.is_leaf());
    assert_eq!(chain(&b), names("AddBackward0", &[None, ACC]));
    assert_eq!(grad_vals(&b, &[&z]), [[1., 1.]]);
    // The value before the write is saved for the gradient of the other factor.
    let b = t(&[2., 4.]);
    b.mul_(&z);
    assert_eq!(vals(&b), [6., 20.]);
    assert_eq!(chain(&b), names("MulBackward0", &[None, ACC]));
    assert_eq!(grad_vals(&b, &[&z]), [[2., 4.]]);
    // A non-contiguous tensor that isn't a view: `detach` of a transposed view.
    let d = Tensor::from_slice(&[1.0f32, 2., 3., 4.], [2, 2])
        .transpose(0, 1)
        .detach();
    assert!(!d.is_contiguous());
    let o = homura::ones([2, 2]);
    o.requires_grad_(true);
    d.mul_(&o);
    assert_eq!(chain(&d), names("MulBackward0", &[None, ACC]));
    assert_eq!(vals(&d), [1., 3., 2., 4.]);
    assert_eq!(grad_vals(&d, &[&o]), [[1., 3., 2., 4.]]);
    // With an offset too: a narrowed transposed tensor, detached, that shares its storage.
    let base = Tensor::from_slice(&[1.0f32, 2., 3., 4., 5., 6.], [2, 3]);
    let n = base.transpose(0, 1).narrow(0, 1, 2).detach();
    let w = Tensor::from_slice(&[1.0f32, 2., 3., 4.], [2, 2]);
    w.requires_grad_(true);
    n.add_(&w);
    n.mul_(&w);
    assert_eq!(
        chain(&n),
        names("MulBackward0", &[Some("AddBackward0"), ACC])
    );
    assert_eq!(vals(&n), [3., 14., 18., 40.]);
    assert_eq!(vals(&base), [1., 3., 18., 4., 14., 40.]);
    // d/dw of (n + w) * w = n + 2w, with n before the writes.
    assert_eq!(grad_vals(&n, &[&w]), [[4., 9., 9., 14.]]);
    // A fresh source: the backward pass above freed what the old history saved.
    let v = Tensor::from_slice(&[1.0f32, 2., 3., 4.], [2, 2]);
    v.requires_grad_(true);
    n.copy_(&v.sum_dim(0, true));
    assert_eq!(vals(&base), [1., 4., 4., 4., 6., 6.]);
    assert_eq!(grad_vals(&n, &[&v]), [[2., 2., 2., 2.]]);
}

#[test]
fn mixed_dtypes_keep_the_dtype_of_self_and_return_gradients_in_their_own() {
    let x = leaf(&[1., 2.]);
    let z = Tensor::from_slice(&[3.0f64, 4.], [2]);
    z.requires_grad_(true);
    for (op, gx, gz) in [("mul_", [3., 4.], [1., 2.]), ("add_", [1., 1.], [1., 1.])] {
        let y = &x * 1.0;
        if op == "mul_" {
            y.mul_(&z);
        } else {
            y.add_(&z);
        }
        assert_eq!(y.dtype(), DType::F32);
        assert!(
            chain(&y)
                .0
                .starts_with(if op == "mul_" { "Mul" } else { "Add" })
        );
        let g = grads(&y, &[&x, &z]);
        let (gx_, gz_) = (g[0].as_ref().unwrap(), g[1].as_ref().unwrap());
        assert_eq!((gx_.dtype(), vals(gx_)), (DType::F32, gx.to_vec()));
        assert_eq!(
            (gz_.dtype(), gz_.to_vec::<f64>()),
            (DType::F64, gz.to_vec())
        );
    }
    // Half precision self, f32 operand: computed in f32, written as f16. The value of self
    // before the write is f16, and so is the upstream gradient.
    let h = Tensor::from_slice(&[1.5f32, 2.25], [2]).to(DType::F16);
    h.requires_grad_(true);
    let z = leaf(&[3.1, 4.7]);
    let y = &h * 1.0;
    y.mul_(&z);
    assert_eq!(y.dtype(), DType::F16);
    let g = Tensor::from_slice(&[0.3f32, 0.7], [2]).to(DType::F16);
    let options = GradOptions {
        grad_outputs: vec![Some(g)],
        ..Default::default()
    };
    let gs = autograd::grad(&[&y], &[&h, &z], options);
    let (gh, gz) = (gs[0].as_ref().unwrap(), gs[1].as_ref().unwrap());
    assert_eq!((gh.dtype(), gz.dtype()), (DType::F16, DType::F32));
    // The reference's values (check_m1_inplace.py).
    assert_eq!(vals(&y), [4.648_437_5, 10.578_125]);
    assert_eq!(vals(gh), [0.930_175_8, 3.291_015_6]);
    assert_eq!(vals(gz), [0.450_195_3, 1.575_195_3]);
}

#[test]
fn successive_in_place_operations_chain_their_nodes() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    let y = &x * 1.0;
    y.add_(&z);
    y.mul_(&z);
    assert_eq!(
        chain(&y),
        names("MulBackward0", &[Some("AddBackward0"), ACC])
    );
    assert_eq!(grad_vals(&y, &[&x, &z]), [vec![3., 5.], vec![7., 12.]]);
    // Through `+=` and friends.
    let mut y = &x * 1.0;
    y += &z;
    y *= 2.0;
    assert_eq!(
        chain(&y),
        names("MulBackward0", &[Some("AddBackward0"), None])
    );
    assert_eq!(grad_vals(&y, &[&x, &z]), [vec![2., 2.], vec![2., 2.]]);
}

#[test]
fn the_value_before_the_write_is_saved_with_its_history() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    // d/dz of the gradient of x, and d/dx of the gradient of z (through the saved old value).
    let y = &x * 1.0;
    y.mul_(&z);
    let options = GradOptions {
        create_graph: true,
        ..Default::default()
    };
    let gx = autograd::grad(&[&y.sum()], &[&x], options.clone())
        .remove(0)
        .unwrap();
    assert_eq!(grad_vals(&gx, &[&z]), [[1., 1.]]);
    let y = &x * 1.0;
    y.mul_(&z);
    let gz = autograd::grad(&[&y.sum()], &[&z], options.clone())
        .remove(0)
        .unwrap();
    assert_eq!(vals(&gz), [1., 2.]);
    assert_eq!(grad_vals(&gz, &[&x]), [[1., 1.]]);
    // Powers save the base.
    let y = &x * 1.0;
    y.pow_(3.0);
    let g = autograd::grad(&[&y.sum()], &[&x], options)
        .remove(0)
        .unwrap();
    assert_eq!(vals(&g), [3., 12.]);
    assert_eq!(grad_vals(&g, &[&x]), [[6., 12.]]);
    // Only the other factor requires grad: the old value of a tensor that doesn't is saved.
    let b = t(&[2., 4.]);
    b.div_(&z);
    assert_close(&grad_vals(&b, &[&z])[0], &[-2. / 9., -4. / 25.]);
}

// ---------------------------------------------------------------- saved tensors and versions

#[test]
fn self_as_an_operand_is_saved_as_it_was() {
    let x = leaf(&[1., 2.]);
    // Not saved: fine.
    let y = &x * 1.0;
    y.add_(&y);
    assert_eq!(vals(&y), [2., 4.]);
    assert_eq!(chain(&y), names("AddBackward0", &[MUL, MUL]));
    assert_eq!(grad_vals(&y, &[&x]), [[2., 2.]]);
    // Saved as the other operand, before the write: the backward pass detects the write and
    // names the operation that wrote it, like PyTorch.
    for (op, name) in [("mul_", "Mul"), ("div_", "Div"), ("pow_", "Pow1")] {
        let y = &x * 1.0;
        match op {
            "mul_" => y.mul_(&y),
            "div_" => y.div_(&y),
            _ => y.pow_(&y),
        };
        let output = format!(", which is output 0 of {name},");
        assert_eq!(
            panic_message(|| y.sum().backward()),
            version_error("[2]", &output, 1, 0),
            "{op}"
        );
    }
    // Through an alias without history, which shares the version counter.
    let y = &x * 1.0;
    y.mul_(y.detach());
    assert_eq!(chain(&y), names("MulBackward0", &[MUL, None]));
    assert_eq!(
        panic_message(|| y.sum().backward()),
        version_error("[2]", "", 1, 0)
    );
}

#[test]
fn saved_results_are_the_target_itself() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    // PowBackward1 saves its result, which is now self after the write.
    let y = &x * 1.0;
    y.pow_(&z);
    assert_eq!(grad_vals(&y, &[&z]).len(), 1);
    let y = &x * 1.0;
    y.pow_(&z);
    {
        let _no_grad = homura::no_grad();
        y.add_(1.0);
    }
    assert_eq!(
        panic_message(|| {
            grads(&y, &[&z]);
        }),
        version_error("[2]", ", which is output 0 of Pow1,", 2, 1)
    );
}

#[test]
fn writing_into_a_tensor_that_an_earlier_node_saved_is_detected() {
    let x = leaf(&[0., 1.]);
    let expected = version_error("[2]", ", which is output 0 of Exp,", 1, 0);
    // Recorded.
    let y = x.exp();
    y.mul_(2.0);
    assert_eq!(
        chain(&y),
        names("MulBackward0", &[Some("ExpBackward0"), None])
    );
    assert_eq!(panic_message(|| y.sum().backward()), expected);
    // Through an alias without history, and in no-grad mode: not recorded, still detected.
    let y = x.exp();
    y.detach().mul_(2.0);
    assert_eq!(panic_message(|| y.sum().backward()), expected);
    let y = x.exp();
    {
        let _no_grad = homura::no_grad();
        y.mul_(2.0);
    }
    assert_eq!(y.grad_fn().unwrap().name(), "ExpBackward0");
    assert_eq!(panic_message(|| y.sum().backward()), expected);
    // An earlier node that saved the target as an input names the target's current history.
    let a = leaf(&[1., 2.]);
    let y = &a * 1.0;
    let w = &y * &x;
    y.add_(1.0);
    assert_eq!(
        panic_message(|| w.sum().backward()),
        version_error("[2]", ", which is output 0 of Add,", 1, 0)
    );
}

#[test]
fn optimizer_updates_run_in_no_grad_mode() {
    let w = leaf(&[1., 2.]);
    for step in 0..3 {
        let loss = (&w * &w).sum();
        loss.backward();
        {
            let _no_grad = homura::no_grad();
            w.sub_(&w.grad().unwrap() * 0.25);
            w.grad().unwrap().zero_();
        }
        assert!(
            w.is_leaf() && w.requires_grad() && w.grad_fn().is_none(),
            "{step}"
        );
    }
    assert_eq!(vals(&w), [0.125, 0.25]);
}

// ---------------------------------------------------------------- rejected writes

#[test]
fn leaves_and_views_are_rejected_before_anything_is_written() {
    let x = leaf(&[1., 2.]);
    assert_unchanged(&x, LEAF, || {
        x.mul_(2.0);
    });
    // Views of a leaf whose own flag is set: created in grad mode, before the leaf required
    // grad, or in no-grad mode (PyTorch has a message of its own for the last one).
    let v = x.view([2]);
    assert_unchanged(&v, VIEW_OF_LEAF, || {
        v.add_(1.0);
    });
    let b = homura::zeros([2]);
    let early = b.view([2]);
    b.requires_grad_(true);
    assert_unchanged(&early, VIEW_OF_LEAF, || {
        early.add_(1.0);
    });
    let quiet = {
        let _no_grad = homura::no_grad();
        x.narrow(0, 0, 1)
    };
    assert_unchanged(&quiet, VIEW_OF_LEAF, || {
        quiet.add_(1.0);
    });
    // Other views that need recording: of a non-leaf, in grad or no-grad mode; with their own
    // flag set, in grad or no-grad mode; of a tensor that doesn't require grad, written from
    // one that does (PyTorch records some of these, and has other messages for the rest).
    let y = &x * 1.0;
    let yv = y.view([2]);
    assert_unchanged(&yv, VIEW, || {
        yv.mul_(2.0);
    });
    let quiet = {
        let _no_grad = homura::no_grad();
        y.view([2])
    };
    assert_unchanged(&quiet, VIEW, || {
        quiet.mul_(2.0);
    });
    for no_grad in [false, true] {
        let b = homura::zeros([2]);
        let own = {
            let _mode = homura::set_grad_enabled(!no_grad);
            b.view([2])
        };
        own.requires_grad_(true);
        assert_unchanged(&own, VIEW, || {
            own.add_(1.0);
        });
    }
    let plain = homura::zeros([2]);
    let pv = plain.view([2]);
    assert_unchanged(&pv, VIEW, || {
        pv.add_(&x);
    });
    // Nothing requires grad: written.
    pv.add_(1.0);
    assert_eq!(vals(&plain), [1., 1.]);
    // In no-grad mode, everything is written.
    {
        let _no_grad = homura::no_grad();
        v.add_(1.0);
        yv.add_(1.0);
    }
    assert_eq!(vals(&x), [2., 3.]);
}

#[test]
fn failed_writes_change_nothing() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    let shape = "output with shape [2] doesn't match the broadcast shape [2, 2]";
    let overlap = "unsupported operation: more than one element of the written-to tensor may \
                   refer to a single memory location. Please deep_clone() the tensor before \
                   performing the operation.";
    let index = "index 2 is out of bounds for dimension 0 with size 2";
    // Recorded, in grad mode but not recorded, and in no-grad mode.
    for (state, operand) in [("recorded", &z), ("not recorded", &t(&[3., 5.]))] {
        let y = if state == "recorded" {
            &x * 1.0
        } else {
            t(&[1., 2.])
        };
        assert_unchanged(&y, shape, || {
            y.add_(homura::ones([2, 2]));
        });
        assert_unchanged(&y, shape, || {
            y.lerp_(operand, homura::ones([2, 2]));
        });
        let s = homura::ones([1]);
        s.requires_grad_(state == "recorded");
        let message = panic_message(|| {
            y.index_add_(0, &ints(&[2]), &s);
        });
        assert!(message.contains(index), "{message}");
        assert_unchanged(&y, &message, || {
            y.index_add_(0, &ints(&[2]), &s);
        });
        // An operand that is a stale view fails where its edge is looked up, before the write.
        let base = &x * 1.0;
        let stale = base.view([2]);
        base.add_(1.0);
        if state == "recorded" {
            assert_unchanged(&y, STALE, || {
                y.add_(&stale);
            });
        }
        // An expanded target that isn't a view.
        let e = (&x * 1.0).expand([2, 2]).detach();
        assert_unchanged(&e, overlap, || {
            e.add_(homura::ones([2]).mul(if state == "recorded" { &z } else { operand }));
        });
        // Copies and fills check what their writes need, with their own rules (copy_
        // broadcasts and converts any dtype), and the arguments of the distribution.
        let src = t(&[1., 2., 3.]);
        src.requires_grad_(state == "recorded");
        let broadcast = panic_message(|| {
            y.detach().copy_(&src);
        });
        assert!(broadcast.contains("must match"), "{broadcast}");
        assert_unchanged(&y, &broadcast, || {
            y.copy_(&src);
        });
        assert_unchanged(&e, overlap, || {
            e.copy_(operand);
        });
        // A target that may overlap can't require grad without being a view: never recorded.
        assert_unchanged(&e, overlap, || {
            e.fill_(1.0);
        });
        let range = "uniform_ expects to return a [from, to) range, but found from=2 > to=1";
        assert_unchanged(&y, range, || {
            y.uniform_(2.0, 1.0);
        });
    }
    let y = &x * 1.0;
    {
        let _no_grad = homura::no_grad();
        assert_unchanged(&y, shape, || {
            y.mul_(homura::ones([2, 2]));
        });
        // Review round 30: operator arguments and overlapping targets in no-grad mode too.
        let message = panic_message(|| {
            y.index_add_(0, &ints(&[2]), &homura::ones([1]));
        });
        assert!(message.contains(index), "{message}");
        assert_unchanged(&y, &message, || {
            y.index_add_(0, &ints(&[2]), &homura::ones([1]));
        });
        let e = (&x * 1.0).expand([2, 2]).detach();
        for op in [
            Box::new(|e: &Tensor| _ = e.add_(1.0)) as Box<dyn Fn(&Tensor)>,
            Box::new(|e: &Tensor| _ = e.copy_(&z)),
            Box::new(|e: &Tensor| _ = e.fill_(1.0)),
        ] {
            assert_unchanged(&e, overlap, || op(&e));
        }
        assert_unchanged(&y, "Cannot copy out of meta tensor; no data!", || {
            y.copy_(&homura::ones_with([2], Device::Meta));
        });
    }
    // The checks of a recorded write come first: a leaf with invalid arguments is rejected as a
    // leaf, before the generator runs.
    assert_unchanged(&x, LEAF, || {
        x.uniform_(2.0, 1.0);
    });
    // Integer targets are never recorded; a float result doesn't fit.
    let n = homura::zeros_with([2], DType::I64);
    let message = "result type f32 can't be cast to the desired output type i64";
    assert_eq!(
        panic_message(|| {
            n.add_(&x);
        }),
        message
    );
    let watch = VersionWatch::new(&n);
    assert_eq!(
        panic_message(|| {
            n.mul_(&x);
        }),
        message
    );
    assert_eq!(watch.bumps(), 0);
    n.add_(3);
    assert!(!n.requires_grad());
}

#[test]
fn promoted_and_broadcast_operands_get_gradients_summed_where_they_were_computed() {
    // Review round 30: the gradient of a wider operand that was broadcast is computed in the
    // dtype of self and of the upstream gradient (F32), summed there, then converted, as in
    // PyTorch: 1 + 2^-30 is rounded to 1. homura sums in f64 and rounds once, PyTorch sums in
    // f32, so other sums may differ by an F32 rounding step, which F64 tolerances can't absorb.
    // copy_ converts the gradient first and sums it in F64, as PyTorch does.
    let e = 2f32.powi(-30);
    let x = leaf(&[1., e]);
    type Op = fn(&Tensor, &Tensor);
    let cases: [(&str, Op, [f32; 2], f64); 4] = [
        ("add_", |y, o| _ = y.add_(o), [1., e], 1.0),
        ("sub_", |y, o| _ = y.sub_(o), [1., e], -1.0),
        ("mul_", |y, o| _ = y.mul_(o), [1., 1.], 1.0),
        (
            "copy_",
            |y, o| _ = y.copy_(o),
            [1., e],
            1.0 + 2f64.powi(-30),
        ),
    ];
    for (name, op, g, expected) in cases {
        let o = Tensor::from_slice(&[1.0f64], [1]);
        o.requires_grad_(true);
        let y = &x * 1.0;
        op(&y, &o);
        assert_eq!(y.dtype(), DType::F32);
        let options = GradOptions {
            grad_outputs: vec![Some(t(&g))],
            ..Default::default()
        };
        let go = autograd::grad(&[&y], &[&o], options).remove(0).unwrap();
        assert_eq!(
            (go.dtype(), go.to_vec::<f64>()),
            (DType::F64, vec![expected]),
            "{name}"
        );
    }
}

// ---------------------------------------------------------------- copies and fills

/// The writes that replace the value of `x * 1` (with the operand `z`), with the values expected
/// after them: the node, the edges after the one of self, the value (`None` for random values),
/// the gradients of x and, for `copy_`, of z, and the operation that version errors name.
struct Overwrite {
    name: &'static str,
    op: fn(&Tensor, &Tensor),
    node: &'static str,
    edges: Vec<Option<&'static str>>,
    value: Option<Vec<f32>>,
    grads: Vec<Vec<f32>>,
    operation: &'static str,
}

fn overwrites() -> Vec<Overwrite> {
    vec![
        Overwrite {
            name: "copy_",
            op: |y, z| _ = y.copy_(z),
            node: "CopyBackwards",
            edges: vec![ACC],
            value: Some(vec![3., 5.]),
            grads: vec![vec![0., 0.], vec![1., 1.]],
            operation: "CopyBackwards",
        },
        Overwrite {
            name: "copy_ self",
            op: |y, _| _ = y.copy_(y),
            node: "CopyBackwards",
            edges: vec![MUL],
            value: Some(vec![1., 2.]),
            grads: vec![vec![1., 1.]],
            operation: "CopyBackwards",
        },
        Overwrite {
            name: "fill_",
            op: |y, _| _ = y.fill_(7.0),
            node: "FillBackward2",
            edges: vec![],
            value: Some(vec![7., 7.]),
            grads: vec![vec![0., 0.]],
            operation: "Fill2",
        },
        Overwrite {
            name: "zero_",
            op: |y, _| _ = y.zero_(),
            node: "ZeroBackward0",
            edges: vec![],
            value: Some(vec![0., 0.]),
            grads: vec![vec![0., 0.]],
            operation: "Zero",
        },
        Overwrite {
            name: "uniform_",
            op: |y, _| _ = y.uniform_(0.0, 1.0),
            node: "UniformBackward0",
            edges: vec![],
            value: None,
            grads: vec![vec![0., 0.]],
            operation: "Uniform",
        },
        Overwrite {
            name: "normal_",
            op: |y, _| _ = y.normal_(0.0, 1.0),
            node: "NormalBackward0",
            edges: vec![],
            value: None,
            grads: vec![vec![0., 0.]],
            operation: "Normal",
        },
    ]
}

#[test]
fn copies_and_fills_give_the_overwritten_value_a_zero_gradient() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    for case in overwrites() {
        let name = case.name;
        let y = &x * 1.0;
        let old = y.grad_fn().unwrap();
        let watch = VersionWatch::new(&y);
        (case.op)(&y, &z);
        assert_eq!(watch.bumps(), 1, "{name}");
        let mut next = vec![MUL];
        next.extend(&case.edges);
        assert_eq!(chain(&y), names(case.node, &next), "{name}");
        let (first, _) = y.grad_fn().unwrap().next_functions()[0].clone().unwrap();
        assert!(first.is_same(&old), "{name}");
        assert!(y.requires_grad() && !y.is_leaf(), "{name}");
        if let Some(value) = &case.value {
            assert_eq!(&vals(&y), value, "{name}");
        }
        // Zeros, not the upstream gradient times zero.
        if case.name != "copy_ self" {
            let options = GradOptions {
                grad_outputs: vec![Some(t(&[f32::NAN, f32::INFINITY]))],
                retain_graph: Some(true),
                ..Default::default()
            };
            let g = autograd::grad(&[&y], &[&x], options).remove(0).unwrap();
            assert_eq!(vals(&g), [0., 0.], "{name}");
        }
        let inputs: Vec<&Tensor> = [&x, &z][..case.grads.len()].to_vec();
        assert_eq!(grad_vals(&y, &inputs), case.grads, "{name}");
        // Version errors name the operation like PyTorch.
        let w = &y * &y;
        {
            let _no_grad = homura::no_grad();
            y.add_(1.0);
        }
        let output = format!(", which is output 0 of {},", case.operation);
        assert_eq!(
            panic_message(|| w.sum().backward()),
            version_error("[2]", &output, 2, 1),
            "{name}"
        );
    }
    // A tensor that doesn't require grad is recorded only when written from one that does.
    for case in overwrites() {
        let b = homura::zeros([2]);
        (case.op)(&b, &t(&[3., 5.]));
        assert!(b.is_leaf() && !b.requires_grad(), "{}", case.name);
    }
}

#[test]
fn copy_converts_the_gradient_of_src_and_sums_it_to_its_shape() {
    let x = leaf(&[1., 2.]);
    // Broadcast from [1]: the gradient is the sum of the upstream gradient, 3 = 1 + 2.
    let s = leaf(&[0.5]);
    let y = &x * 1.0;
    y.copy_(&s);
    assert_eq!(vals(&y), [0.5, 0.5]);
    let options = GradOptions {
        grad_outputs: vec![Some(t(&[1., 2.]))],
        ..Default::default()
    };
    assert_eq!(
        vals(&autograd::grad(&[&y], &[&s], options)[0].clone().unwrap()),
        [3.]
    );
    // Into a tensor that doesn't require grad, from F64: the tensor joins the graph, and the
    // gradient of src is F64.
    let b = homura::zeros([3, 2]);
    let d = Tensor::from_slice(&[1.0f64, 2.], [2]);
    d.requires_grad_(true);
    b.copy_(&d);
    assert!(b.requires_grad() && !b.is_leaf());
    assert_eq!(chain(&b), names("CopyBackwards", &[None, ACC]));
    let gd = grads(&b, &[&d]).remove(0).unwrap();
    assert_eq!((gd.dtype(), gd.to_vec::<f64>()), (DType::F64, vec![3., 3.]));
    // An integer src has no edge. An integer target records nothing, and takes a float src.
    let y = &x * 1.0;
    y.copy_(&ints(&[7, 8]));
    assert_eq!(chain(&y), names("CopyBackwards", &[MUL, None]));
    assert_eq!(vals(&y), [7., 8.]);
    assert_eq!(grad_vals(&y, &[&x]), [[0., 0.]]);
    let n = homura::zeros_with([2], DType::I64);
    n.copy_(&x);
    assert_eq!(n.to_vec::<i64>(), [1, 2]);
    assert!(n.grad_fn().is_none() && !n.requires_grad());
    // On the meta device: from a CPU src, the gradient can't be brought back; from a meta src,
    // shapes only. A meta src can't be copied into a CPU tensor.
    let m = homura::zeros_with([2], Device::Meta);
    m.copy_(&x);
    assert_eq!(chain(&m), names("CopyBackwards", &[None, ACC]));
    assert_eq!(
        panic_message(|| drop(grads(&m, &[&x]))),
        "Cannot copy out of meta tensor; no data!"
    );
    let mo = homura::ones_with([2], Device::Meta);
    mo.requires_grad_(true);
    let m = homura::zeros_with([3, 2], Device::Meta);
    m.copy_(&mo);
    let g = grads(&m, &[&mo]).remove(0).unwrap();
    assert_eq!((g.device(), g.shape().to_vec()), (Device::Meta, vec![2]));
    // Only the gradients that are needed are computed: through the old value of a meta target,
    // the gradient for a CPU src isn't converted back.
    let mm = homura::ones_with([2], Device::Meta);
    mm.requires_grad_(true);
    let m = &mm * 1.0;
    m.copy_(&x);
    let g = grads(&m, &[&mm]).remove(0).unwrap();
    assert_eq!((g.device(), g.shape().to_vec()), (Device::Meta, vec![2]));
    let y = &x * 1.0;
    assert_unchanged(&y, "Cannot copy out of meta tensor; no data!", || {
        y.copy_(&mo);
    });
    // A narrower src too: the gradient is rounded to its dtype before the sum (review round
    // 31), so 70000 and -70000 become infinities of opposite signs in F16 and give NaN, as in
    // PyTorch, while the old value gets zeros.
    let h = Tensor::from_slice(&[0.5f32], [1]).to(DType::F16);
    h.requires_grad_(true);
    let y = &x * 1.0;
    y.copy_(&h);
    let options = GradOptions {
        grad_outputs: vec![Some(t(&[70000., -70000.]))],
        ..Default::default()
    };
    let g = autograd::grad(&[&y], &[&x, &h], options);
    let (gx, gh) = (g[0].as_ref().unwrap(), g[1].as_ref().unwrap());
    assert_eq!(vals(gx), [0., 0.]);
    assert_eq!(gh.dtype(), DType::F16);
    assert!(vals(gh)[0].is_nan());
    // Zero-dimensional and empty.
    let s = Tensor::from_slice(&[2.0f32], []);
    s.requires_grad_(true);
    let y = &s * 1.0;
    y.copy_(&x.sum());
    assert_eq!(grad_vals(&y, &[&s, &x]), [vec![0.], vec![1., 1.]]);
    let e = homura::zeros([0, 2]);
    e.requires_grad_(true);
    let y = &e * 1.0;
    y.fill_(1.0);
    y.copy_(&x);
    assert_eq!(grads(&y, &[&e])[0].as_ref().unwrap().shape()[..], [0, 2]);
}

#[test]
fn gradients_through_copies_and_fills_can_be_differentiated_again() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    let create_graph = || GradOptions {
        create_graph: true,
        ..Default::default()
    };
    // The gradients of `g.sum()`, keeping the graph for the next ones.
    let again = |g: &Tensor, inputs: &[&Tensor]| -> Vec<Vec<f32>> {
        let options = GradOptions {
            retain_graph: Some(true),
            ..Default::default()
        };
        let g = autograd::grad(&[&g.sum()], inputs, options);
        g.iter().map(|g| vals(g.as_ref().unwrap())).collect()
    };
    let y = &x * &x;
    y.copy_(&(&z * &z));
    let g = autograd::grad(&[&y.sum()], &[&x, &z], create_graph());
    let (gx, gz) = (g[0].as_ref().unwrap(), g[1].as_ref().unwrap());
    assert_eq!((vals(gx), vals(gz)), (vec![0., 0.], vec![6., 10.]));
    assert_eq!(again(gz, &[&z]), [[2., 2.]]);
    assert_eq!(again(gx, &[&x]), [[0., 0.]]);
    let y = &x * &x;
    y.zero_();
    let gx = autograd::grad(&[&y.sum()], &[&x], create_graph()).remove(0);
    assert_eq!(vals(&gx.unwrap()), [0., 0.]);
    // Chained with a recorded write.
    let y = &x * 1.0;
    y.copy_(&z);
    y.mul_(&z);
    assert_eq!(
        chain(&y),
        names("MulBackward0", &[Some("CopyBackwards"), ACC])
    );
    assert_eq!(grad_vals(&y, &[&x, &z]), [[0., 0.], [6., 10.]]);
    // Second derivatives with respect to the target after the write, an intermediate result
    // (review round 30): out = sum(y^2) with y = z * x, so d(out)/dz = 2 y x, whose derivatives
    // are 2x for y, 2y + 2xz for x and 2x^2 for z.
    let y = &x * 2.0;
    y.copy_(&(&z * &x));
    let out = (&y * &y).sum();
    let g = autograd::grad(&[&out], &[&y, &z], create_graph());
    let (gy, gz) = (g[0].as_ref().unwrap(), g[1].as_ref().unwrap());
    assert_eq!((vals(gy), vals(gz)), (vec![6., 20.], vec![6., 40.]));
    assert_eq!(
        again(gz, &[&y, &x, &z]),
        [vec![2., 4.], vec![12., 40.], vec![2., 8.]]
    );
    assert_eq!(again(gy, &[&x, &z]), [[6., 10.], [2., 4.]]);
}

#[test]
fn copies_and_fills_into_leaves_and_views_are_rejected() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    for case in overwrites() {
        let op = |t: &Tensor| (case.op)(t, &z);
        assert_unchanged(&x, LEAF, || op(&x));
        // Views of a leaf: created in grad mode, before the leaf required grad, in no-grad mode.
        let v = x.view([2]);
        assert_unchanged(&v, VIEW_OF_LEAF, || op(&v));
        let b = homura::zeros([2]);
        let early = b.view([2]);
        b.requires_grad_(true);
        assert_unchanged(&early, VIEW_OF_LEAF, || op(&early));
        let quiet = {
            let _no_grad = homura::no_grad();
            x.view([2])
        };
        assert_unchanged(&quiet, VIEW_OF_LEAF, || op(&quiet));
        // Views of a non-leaf, in grad and no-grad mode, and with their own flag set.
        let y = &x * 1.0;
        let yv = y.view([2]);
        assert_unchanged(&yv, VIEW, || op(&yv));
        let quiet = {
            let _no_grad = homura::no_grad();
            y.view([2])
        };
        assert_unchanged(&quiet, VIEW, || op(&quiet));
        for no_grad in [false, true] {
            let b = homura::zeros([2]);
            let own = {
                let _mode = homura::set_grad_enabled(!no_grad);
                b.view([2])
            };
            own.requires_grad_(true);
            assert_unchanged(&own, VIEW, || op(&own));
        }
        // In no-grad mode, written.
        {
            let _no_grad = homura::no_grad();
            op(&v);
            op(&yv);
        }
        assert_eq!(chain(&y), names("MulBackward0", &[ACC, None]));
    }
    // A view of a tensor that doesn't require grad, written from one that does.
    let pv = homura::zeros([2]).view([2]);
    assert_unchanged(&pv, VIEW, || {
        pv.copy_(&x);
    });
    pv.copy_(&t(&[1., 2.]));
    assert_eq!(vals(&pv), [1., 2.]);
}

#[test]
fn views_of_a_copied_or_filled_tensor_are_stale() {
    let x = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    for case in overwrites() {
        let name = case.name;
        let y = &x * 1.0;
        let v = y.view([2]);
        (case.op)(&y, &z);
        assert_eq!(panic_message(|| drop(&v * 2.0)), STALE, "{name}");
        assert_eq!(panic_message(|| drop(v.grad_fn())), STALE, "{name}");
        assert_eq!(panic_message(|| drop(grads(&y, &[&v]))), STALE, "{name}");
        let fresh = y.view([2]);
        assert_eq!(grad_vals(&fresh, &[&x]), case.grads[..1], "{name}");
    }
}

// ---------------------------------------------------------------- edge cases

#[test]
fn zero_dimensional_empty_and_meta_targets() {
    let s = Tensor::from_slice(&[2.0f32], []);
    s.requires_grad_(true);
    let y = &s * 1.0;
    y.mul_(&s);
    y.mul_(3.0);
    assert_eq!(vals(&y), [12.]);
    assert_eq!(grad_vals(&y, &[&s]), [[12.]]);
    let e = homura::zeros([0, 3]);
    e.requires_grad_(true);
    let y = &e * 1.0;
    y.mul_(e.sum_dim(0, true));
    assert_eq!(chain(&y).0, "MulBackward0");
    let g = grads(&y, &[&e]).remove(0).unwrap();
    assert_eq!(g.shape()[..], [0, 3]);
    // On the meta device: recorded with shapes, no data.
    let m = homura::zeros_with([2, 3], Device::Meta);
    m.requires_grad_(true);
    let o = homura::ones_with([3], Device::Meta);
    o.requires_grad_(true);
    let y = &m * 1.0;
    y.mul_(&o);
    y.clamp_(0.0, 1.0);
    assert_eq!(chain(&y), names("ClampBackward1", &[MUL]));
    let g = grads(&y, &[&m, &o]);
    for (g, shape) in g.iter().zip([vec![2, 3], vec![3]]) {
        let g = g.as_ref().unwrap();
        assert_eq!((g.device(), g.shape().to_vec()), (Device::Meta, shape));
    }
}

#[test]
fn in_place_operations_inside_functions_pass_gradcheck() {
    use homura::autograd::{GradcheckOptions, gradcheck, gradgradcheck};
    let input = |values: &[f64], shape: &[usize]| {
        let x = Tensor::from_slice(values, shape);
        x.requires_grad_(true);
        x
    };
    let x = input(&[0.3, -0.7, 1.2, 0.5, -1.1, 0.8], &[2, 3]);
    let z = input(&[1.4, 0.6, 1.9], &[3]);
    let w = input(&[0.35, 0.8, 0.15, 0.6, 0.45, 0.9], &[2, 3]);
    let s = input(&[0.25, -0.5, 1.5], &[1, 3]);
    let f = |t: &[Tensor]| {
        let (x, z, w, s) = (&t[0], &t[1], &t[2], &t[3]);
        let y = x.deep_clone();
        y.mul_(z);
        y.add_(z.sin());
        y.div_(z.exp());
        y.pow_(2.0);
        y.addcmul_(w, z, 0.5);
        y.addcdiv_(x, &(w + 1.0), -0.25);
        y.lerp_(&x.cos(), w);
        y.lerp_(z, 0.3);
        y.clamp_(&(x - 5.0), &(z + 5.0));
        // Negative dimensions that aren't the first one.
        y.index_add_(-1, &Tensor::from_slice(&[2i64, 0, 1], [3]), w);
        y.scatter_add_(-1, &Tensor::from_slice(&[2i64, 0, 1], [1, 3]), s);
        y.masked_fill_(
            &homura::tensor([[true, false, false], [false, false, true]]),
            s.sum(),
        );
        let p = y.abs() + 0.5;
        p.pow_(w);
        vec![p, y]
    };
    let inputs = [&x, &z, &w, &s];
    gradcheck(f, &inputs, GradcheckOptions::default()).unwrap();
    gradgradcheck(f, &inputs, GradcheckOptions::default()).unwrap();
    // sub_ after a node that saves no output (review round 30), copies (broadcast, into a
    // tensor that doesn't require grad, of self), fills and random fills.
    let f = |t: &[Tensor]| {
        let (x, z, w) = (&t[0], &t[1], &t[2]);
        let y = x * x;
        y.sub_(z);
        y.sub_(&(w * 2.0));
        let c = homura::zeros_with([2, 3], DType::F64);
        c.copy_(z);
        c.mul_(&y);
        c.copy_(&c);
        let d = x * w;
        d.zero_();
        d.add_(&c);
        let e = y.sin();
        e.fill_(1.5);
        e.mul_(z);
        // cos saves its input, not its result, which the writes replace.
        let r = x.cos();
        r.uniform_(0.0, 1.0);
        r.copy_(&(&y * w).sum_dim(0, true));
        let n = x.neg();
        n.normal_(0.0, 1.0);
        n.zero_();
        n.add_(&(z * w));
        vec![d, e, r, n]
    };
    let inputs = [&x, &z, &w];
    gradcheck(f, &inputs, GradcheckOptions::default()).unwrap();
    gradgradcheck(f, &inputs, GradcheckOptions::default()).unwrap();
}

// ---------------------------------------------------------------- clamp_ and promoted bounds

#[test]
fn clamp_computes_its_gradient_in_the_dtypes_of_the_computation() {
    // Review round 28 (R41): F32 self, an F64 tensor bound and a scalar one. The scalar bound
    // 1 + 2^-30 is computed in F64, where x = 1 is between the bounds; in F32, the dtype of
    // self, it would round to 1 and make x a tie.
    let one = |dtype: DType| {
        let x = Tensor::from_slice(&[1.0f32], []).to(dtype);
        x.requires_grad_(true);
        x
    };
    let bound = |v: f64, dtype: DType| {
        let t = Tensor::from_slice(&[v], []).to(dtype);
        t.requires_grad_(true);
        t
    };
    let x = one(DType::F32);
    let lo = bound(0.0, DType::F64);
    let y = &x * 1.0;
    y.clamp_(&lo, 1.0 + 2f64.powi(-30));
    assert_eq!(y.dtype(), DType::F32);
    let g = grads(&y, &[&x, &lo]);
    let (gx, glo) = (g[0].as_ref().unwrap(), g[1].as_ref().unwrap());
    assert_eq!((gx.dtype(), gx.to_vec::<f32>()), (DType::F32, vec![1.0]));
    assert_eq!((glo.dtype(), glo.to_vec::<f64>()), (DType::F64, vec![0.0]));
    // In-place equals out-of-place: forward value, gradients and their dtypes, for half
    // precision targets with wider tensor bounds, both scalar/tensor orders, and x between the
    // bounds, at a bound (either side's value when rounded to self's dtype) and with min > max.
    let eps = [
        (DType::F32, 2f64.powi(-30)),
        (DType::F16, 2f64.powi(-20)),
        (DType::BF16, 2f64.powi(-10)),
    ];
    for (dtype, e) in eps {
        let wide = if dtype == DType::F32 {
            DType::F64
        } else {
            DType::F32
        };
        // (case, min, max, whether the scalar bound is max); the other bound is a tensor.
        let cases = [
            ("between", -1.0, 1.0 + e, true),
            ("between", 1.0 - e, 2.0, false),
            ("tie", 1.0, 2.0, false),
            ("tie", 0.0, 1.0, true),
            ("reversed", 2.0, 0.5, true),
            ("reversed", 1.5, 0.5, false),
        ];
        for (what, min, max, scalar_max) in cases {
            let (scalar, tensor) = if scalar_max { (max, min) } else { (min, max) };
            let run = |y: &Tensor, b: &Tensor| {
                if scalar_max {
                    y.clamp(b, scalar)
                } else {
                    y.clamp(scalar, b)
                }
            };
            let (x1, b1) = (one(dtype), bound(tensor, wide));
            let out = run(&x1, &b1);
            let expected = grads(&out, &[&x1, &b1]);
            let (x2, b2) = (one(dtype), bound(tensor, wide));
            let y = &x2 * 1.0;
            if scalar_max {
                y.clamp_(&b2, scalar);
            } else {
                y.clamp_(scalar, &b2);
            }
            let label = format!("{dtype} {what} {min} {max}");
            assert_eq!(y.dtype(), dtype, "{label}");
            assert_eq!(vals(&y), vals(&out.to(dtype)), "{label}");
            for (a, e) in grads(&y, &[&x2, &b2]).iter().zip(&expected) {
                let (a, e) = (a.as_ref().unwrap(), e.as_ref().unwrap());
                assert_eq!(a.dtype(), e.dtype(), "{label}");
                assert_eq!(
                    a.to(DType::F64).to_vec::<f64>(),
                    e.to(DType::F64).to_vec::<f64>(),
                    "{label}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------- stale views

#[test]
fn views_of_a_rewritten_history_are_stale() {
    let x = leaf(&[1., 2.]);
    let y = &x * 3.0;
    let v = y.view([2]);
    let quiet_child = || {
        let _no_grad = homura::no_grad();
        v.view([1, 2])
    };
    let before = quiet_child();
    y.mul_(4.0);
    assert_eq!(vals(&v), [12., 24.]);
    // Every use of the stale view's history panics: as an input, as a root, as a requested
    // input, as the parent of a new view, and its public grad_fn.
    let ones = homura::ones([2]);
    type Use<'a> = (&'a str, Box<dyn Fn() + 'a>);
    let uses: Vec<Use> = vec![
        ("input", Box::new(|| drop(&v * 2.0))),
        (
            "root",
            Box::new(|| autograd::backward(&[&v], &[Some(&ones)], Default::default())),
        ),
        ("requested", Box::new(|| drop(grads(&y, &[&v])))),
        ("parent", Box::new(|| drop(v.view([1, 2])))),
        ("grad_fn", Box::new(|| drop(v.grad_fn()))),
    ];
    for (name, f) in &uses {
        assert_eq!(panic_message(f), STALE, "{name}");
    }
    // Views created from it, before or after the write, are stale too, also when created in
    // no-grad mode; requires_grad and is_leaf don't change.
    for child in [before, quiet_child()] {
        assert_eq!(panic_message(|| drop(child.grad_fn())), STALE);
        assert_eq!(panic_message(|| drop(&child * 1.0)), STALE);
    }
    assert!(v.requires_grad() && !v.is_leaf());
    // Data operations in no-grad mode use it normally.
    {
        let _no_grad = homura::no_grad();
        assert_eq!(vals(&(&v * 2.0)), [24., 48.]);
        assert_eq!(vals(&v.view([1, 2])), [12., 24.]);
    }
    // A view created after the write follows the new history.
    let fresh = y.view([2]);
    assert_eq!(fresh.grad_fn().unwrap().name(), "ViewBackward0");
    assert_eq!(grad_vals(&fresh, &[&x]), [[12., 12.]]);
}

#[test]
fn a_leaf_that_joined_the_graph_rejects_old_backward_passes() {
    // Review round 3 (R16): a leaf that stops requiring grad and is then written from a tensor
    // that does joins the graph; graphs recorded while it was a leaf can't accumulate into it.
    let w = leaf(&[1., 2.]);
    let z = leaf(&[3., 5.]);
    let out = (&w * 2.0).sum();
    w.requires_grad_(false);
    w.add_(&z);
    assert!(!w.is_leaf() && w.requires_grad());
    assert_eq!(
        panic_message(|| out.backward()),
        "leaf variable has been moved into the graph interior"
    );
    assert_eq!(grad_vals(&w, &[&z]), [[1., 1.]]);
}
