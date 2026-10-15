//! Tests of the in-place recording machinery that the public API can't observe: the recording
//! context, history generations, and the reference cycles that a tensor joining the graph can
//! form (documented in the `autograd` module).

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use super::meta;
use super::record::{check_in_place, record_in_place};
use crate::Tensor;
use crate::autograd::formulas;
use crate::ops::{BinaryOp, Operand};

fn t(values: &[f32]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn leaf(values: &[f32]) -> Tensor {
    let x = t(values);
    x.requires_grad_(true);
    x
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

/// Records `target = target + other` in place with a write that counts its runs.
fn add_in_place(target: &Tensor, other: &Tensor, runs: &Rc<Cell<u32>>, recorder: impl FnOnce()) {
    let result = {
        let _no_grad = crate::no_grad();
        target + other
    };
    let (dst, runs) = (target.clone(), runs.clone());
    let write = Box::new(move || {
        runs.set(runs.get() + 1);
        dst.write_copy(&result);
    });
    record_in_place(target, write, recorder);
}

fn add_formula(target: &Tensor, other: &Tensor) {
    let lhs = Operand::Tensor(target.clone());
    formulas::binary(BinaryOp::Add, &lhs, &Operand::Tensor(other.clone()), target);
}

#[test]
fn each_recorded_call_writes_once_and_rewrites_the_history_once() {
    let x = leaf(&[1., 2.]);
    let y = &x * 1.0;
    let (version, generation) = (y.version(), meta::history_gen(&y));
    for (i, op) in ["add_", "mul_", "clamp_", "lerp_"].into_iter().enumerate() {
        match op {
            "add_" => y.add_(&x),
            "mul_" => y.mul_(&x),
            "clamp_" => y.clamp_(0.0, 10.0),
            _ => y.lerp_(&x, 0.5),
        };
        let calls = i as u64 + 1;
        assert_eq!(y.version(), version + calls, "{op}");
        assert_eq!(meta::history_gen(&y), generation + calls, "{op}");
    }
    // Not recorded: the version changes, the history doesn't.
    {
        let _no_grad = crate::no_grad();
        y.add_(1.0);
    }
    assert_eq!(meta::history_gen(&y), generation + 4);
    let b = t(&[1., 2.]);
    b.add_(1.0);
    assert_eq!((b.version(), meta::history_gen(&b)), (1, 0));
    // A failed call changes neither.
    let _ = panic_message(|| {
        y.add_(t(&[1., 2., 3.]));
    });
    assert_eq!(y.version(), version + 5);
    assert_eq!(meta::history_gen(&y), generation + 4);
}

#[test]
fn the_recording_context_writes_exactly_once_and_restores_on_exit() {
    let x = leaf(&[1., 2.]);
    let runs = Rc::new(Cell::new(0));
    let y = &x * 1.0;
    assert!(check_in_place(&y, &[Some(&x)]));
    add_in_place(&y, &x, &runs, || add_formula(&y, &x));
    assert_eq!(runs.get(), 1);
    assert_eq!(y.to_vec::<f32>(), [2., 4.]);
    assert_eq!(y.grad_fn().unwrap().name(), "AddBackward0");
    // Nested: a recording for another target inside the recorder, and an ordinary recorded
    // operation, leave the outer context in place.
    let (y, inner) = (&x * 1.0, &x * 1.0);
    let inner_runs = Rc::new(Cell::new(0));
    add_in_place(&y, &x, &runs, || {
        add_in_place(&inner, &x, &inner_runs, || add_formula(&inner, &x));
        let ordinary = &x * 3.0;
        assert_eq!(ordinary.grad_fn().unwrap().name(), "MulBackward0");
        add_formula(&y, &x);
    });
    assert_eq!((runs.get(), inner_runs.get()), (2, 1));
    assert_eq!(meta::history_gen(&inner), 1);
    assert_eq!(meta::history_gen(&y), 1);
    // A recorder that panics before the write: nothing is written, and the context is gone,
    // so ordinary recording and the next in-place operation work.
    let y = &x * 1.0;
    let message = panic_message(|| {
        add_in_place(&y, &x, &runs, || panic!("recorder failed"));
    });
    assert_eq!(message, "recorder failed");
    assert_eq!((runs.get(), y.to_vec::<f32>()), (2, vec![1., 2.]));
    assert_eq!(meta::history_gen(&y), 0);
    let z = &y * 2.0;
    assert_eq!(meta::history_gen(&z), 0);
    assert_eq!(z.grad_fn().unwrap().name(), "MulBackward0");
    y.add_(&x);
    assert_eq!(y.grad_fn().unwrap().name(), "AddBackward0");
    // Nothing is left behind that would keep the target alive.
    let weak = y.downgrade();
    drop((y, z));
    assert!(weak.upgrade().is_none());
    // A recorder that records nothing for the target is an internal error, before the write.
    let y = &x * 1.0;
    let message = panic_message(|| {
        add_in_place(&y, &x, &runs, || drop(&x * 2.0));
    });
    assert_eq!(
        message,
        "internal error: an in-place operation didn't record its target"
    );
    assert_eq!((runs.get(), y.to_vec::<f32>()), (2, vec![1., 2.]));
    // Recording the target twice is an internal error too.
    let message = panic_message(|| {
        add_in_place(&y, &x, &runs, || {
            add_formula(&y, &x);
            add_formula(&y, &x);
        });
    });
    assert_eq!(
        message,
        "internal error: an in-place operation recorded its target twice"
    );
    // Only the first recording ran, and wrote.
    assert_eq!(runs.get(), 3);
}

/// Example A of review round 28: `b.addcmul_(&b, &z, 1)` saves the leaf `b` in the node that
/// becomes its history. Example B: `y = &b * &z; b.add_(&y)` saves it in an earlier node that the
/// new history reaches.
fn cycle(example: char) -> (Tensor, Tensor) {
    let z = leaf(&[3., 5.]);
    let b = t(&[1., 2.]);
    if example == 'A' {
        b.addcmul_(&b, &z, 1.0);
    } else {
        let y = &b * &z;
        b.add_(&y);
    }
    (b, z)
}

#[test]
fn a_tensor_that_joins_the_graph_and_is_saved_in_it_lives_until_detached() {
    for example in ['A', 'B'] {
        // Not run: the tensor and its node keep each other alive after the last handle.
        let (b, _z) = cycle(example);
        let (tensor, node) = (b.downgrade(), meta::grad_fn(&b).unwrap().0.downgrade());
        drop(b);
        assert!(
            tensor.upgrade().is_some() && node.upgrade().is_some(),
            "{example}"
        );
        // Run: the saved leaf was written after it was saved, so the pass fails, and the cycle
        // stays.
        let (b, _z) = cycle(example);
        let message = panic_message(|| b.sum().backward());
        assert!(
            message.starts_with("one of the variables needed"),
            "{message}"
        );
        let (tensor, node) = (b.downgrade(), meta::grad_fn(&b).unwrap().0.downgrade());
        drop(b);
        assert!(
            tensor.upgrade().is_some() && node.upgrade().is_some(),
            "{example}"
        );
        // Detached in place before the last handle goes: both are freed.
        let (b, _z) = cycle(example);
        let (tensor, node) = (b.downgrade(), meta::grad_fn(&b).unwrap().0.downgrade());
        b.detach_();
        drop(b);
        assert!(
            tensor.upgrade().is_none() && node.upgrade().is_none(),
            "{example}"
        );
    }
    // Without the cycle, dropping the handles frees everything.
    let z = leaf(&[3., 5.]);
    let b = t(&[1., 2.]);
    b.add_(&z);
    let (tensor, node) = (b.downgrade(), meta::grad_fn(&b).unwrap().0.downgrade());
    drop(b);
    assert!(tensor.upgrade().is_none() && node.upgrade().is_none());
}
