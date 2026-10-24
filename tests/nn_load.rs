//! Loading state dicts by assignment (`LoadOptions::assign`) and models built on the meta
//! device. tools/golden/check_m3_load.py asserts the reference behavior of each test with a
//! PyTorch counterpart, in the same order; the tests after those cover what homura does
//! differently: parameters keep their identity, and graphs and views of the old data are refused,
//! whichever way they reach it.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::autograd::{self, GradOptions};
use homura::nn::{
    BatchNorm2d, Conv2d, Embedding, Forward, LayerNorm, Linear, LoadOptions, Module, Parameter,
    RmsNorm, StateDict,
};
use homura::optim::{Adam, Optimizer, Sgd};
use homura::{DType, Device, Tensor};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

fn assign() -> LoadOptions {
    LoadOptions::default().assign(true)
}

fn sd(entries: Vec<(&str, Tensor)>) -> StateDict {
    entries
        .into_iter()
        .map(|(k, t)| (k.to_string(), t))
        .collect()
}

const SAVED: &str = "the data of a leaf saved for the backward pass was replaced after it was saved \
                     (it was f32 on cpu, it is f32 on cpu); run the forward pass again";
const ACCUMULATED: &str =
    "the data of a leaf was replaced after the graph was recorded; run the forward pass again";
const STALE_VIEW: &str = "a view whose base had its data replaced (by Module::to or by loading \
                          with assign) after the view was created can't be used for gradient \
                          computation: it still refers to the old data; create the view again";

#[test]
fn assign_takes_the_data_of_the_entries() {
    let m = Linear::new(3, 2);
    m.weight.set_grad(Some(homura::ones([2, 3])));
    m.bias.as_ref().unwrap().requires_grad_(false);
    let (weight_handle, bias_handle) = (m.weight.clone(), m.bias.clone().unwrap());
    // A non-contiguous F64 entry: the parameter takes its dtype, layout and storage.
    let weight = homura::tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        .to(DType::F64)
        .transpose(0, 1);
    let bias = homura::zeros_with([2], DType::F64);
    let report = m
        .load_state_dict_with(
            &sd(vec![("weight", weight.clone()), ("bias", bias)]),
            assign(),
        )
        .unwrap();
    assert!(report.missing_keys.is_empty() && report.unexpected_keys.is_empty());
    let w = &m.weight;
    assert_eq!((w.dtype(), &w.shape()[..]), (DType::F64, &[2, 3][..]));
    assert_eq!((&w.stride()[..], w.storage_offset()), (&[1, 2][..], 0));
    assert!(w.is_leaf());
    assert_eq!(w.to_vec::<f64>(), [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
    // The parameters stay the same: other handles see the new data.
    assert!(weight_handle.is_same(&m.weight) && bias_handle.is_same(m.bias.as_ref().unwrap()));
    assert_eq!(weight_handle.dtype(), DType::F64);
    // requires_grad is the module's, the gradient is gone.
    assert!(w.requires_grad() && w.grad().is_none());
    assert!(!m.bias.as_ref().unwrap().requires_grad());
    // The parameter and the entry share the storage: writing one changes the other.
    homura::with_no_grad(|| w.add_(1.0));
    assert_eq!(weight.to_vec::<f64>()[0], 1.0);
    weight.mul_(2.0);
    assert_eq!(w.to_vec::<f64>()[0], 2.0);
    // A buffer takes the entry's data too.
    let bn = BatchNorm2d::new(2);
    let mean = homura::full([2], 3.0);
    let mut entries = bn.state_dict();
    entries.insert("running_mean", mean.clone());
    bn.load_state_dict_with(&entries, assign()).unwrap();
    let running_mean = bn.running_mean.as_ref().unwrap();
    running_mean.add_(1.0);
    assert_eq!(mean.to_vec::<f32>(), [4.0, 4.0]);
}

#[test]
fn assign_combines_with_strict() {
    let entries = sd(vec![
        ("weight", homura::ones([2, 3])),
        ("extra", homura::ones([1])),
    ]);
    let m = Linear::new(3, 2);
    let before = m.weight.to_vec::<f32>();
    let error = m.load_state_dict_with(&entries, assign()).unwrap_err();
    assert_eq!(
        (error.missing_keys, error.unexpected_keys),
        (vec!["bias".to_string()], vec!["extra".to_string()])
    );
    // Nothing was written (PyTorch assigns the weight anyway).
    assert_eq!(m.weight.to_vec::<f32>(), before);
    let m = Linear::new(3, 2);
    let bias = m.bias.as_ref().unwrap().to_vec::<f32>();
    let report = m
        .load_state_dict_with(&entries, assign().strict(false))
        .unwrap();
    assert_eq!(
        (report.missing_keys, report.unexpected_keys),
        (vec!["bias".to_string()], vec!["extra".to_string()])
    );
    assert_eq!(m.weight.to_vec::<f32>(), [1.0; 6]);
    assert_eq!(m.bias.as_ref().unwrap().to_vec::<f32>(), bias);
    // The options are independent.
    let options = LoadOptions::default().strict(false).assign(true);
    assert!(!options.is_strict() && options.is_assign());
    assert!(!LoadOptions::default().is_assign());
    assert!(options.assign(false).strict(true) == LoadOptions::default());
}

fn copy_error(key: &str, shape: &str, e: &str) -> String {
    format!(
        "While copying the parameter named \"{key}\", whose dimensions in the model are {shape} \
         and whose dimensions in the checkpoint are {shape}, an exception occurred: {e}"
    )
}

#[test]
fn assign_checks_the_entries() {
    let m = Linear::new(3, 2);
    let before = m.weight.to_vec::<f32>();
    let ints = sd(vec![
        ("weight", homura::zeros_with([2, 3], DType::I64)),
        ("bias", homura::zeros([2])),
    ]);
    let error = m.load_state_dict_with(&ints, assign()).unwrap_err();
    assert_eq!(
        error.errors,
        [copy_error(
            "weight",
            "[2, 3]",
            "Only Tensors of floating point and complex dtype can require gradients"
        )]
    );
    assert_eq!(m.weight.to_vec::<f32>(), before);
    assert_eq!(m.bias.as_ref().unwrap().dtype(), DType::F32);
    // A parameter that doesn't require grad takes an integer entry.
    let m = Linear::new(3, 2);
    m.weight.requires_grad_(false);
    m.load_state_dict_with(&ints, assign()).unwrap();
    assert_eq!(m.weight.dtype(), DType::I64);
    let wrong = sd(vec![
        ("weight", homura::zeros([3, 3])),
        ("bias", homura::zeros([2])),
    ]);
    let error = Linear::new(3, 2)
        .load_state_dict_with(&wrong, assign())
        .unwrap_err();
    assert_eq!(
        error.errors,
        [
            "size mismatch for weight: copying a param with shape [3, 3] from checkpoint, the shape \
          in current model is [2, 3]."
        ]
    );
}

/// Builds a layer, and an input for it.
type Case = (fn() -> Box<dyn Forward>, fn() -> Tensor);

/// Built-in layers, with an input.
fn layers() -> Vec<Case> {
    vec![
        (|| Box::new(Linear::new(4, 3)), || homura::randn([2, 4])),
        (
            || Box::new(Conv2d::new(2, 3, 3).padding(1)),
            || homura::randn([1, 2, 4, 4]),
        ),
        (
            || Box::new(BatchNorm2d::new(2)),
            || homura::randn([2, 2, 3, 3]),
        ),
        (|| Box::new(LayerNorm::new([4])), || homura::randn([2, 4])),
        (|| Box::new(RmsNorm::new([4])), || homura::randn([2, 4])),
        (
            || Box::new(Embedding::new(5, 4)),
            || homura::tensor([0i64, 4, 2]),
        ),
    ]
}

#[test]
fn models_built_on_meta_load_by_assignment() {
    for (make, input) in layers() {
        let empty = homura::with_device(Device::Meta, make);
        let on_meta = |m: &dyn Module, device: Device| {
            m.parameters().iter().all(|p| p.device() == device)
                && m.buffers().iter().all(|b| b.device() == device)
        };
        assert!(on_meta(&*empty, Device::Meta), "{}", empty.type_name());
        let (trained, copied) = (make(), make());
        copied.load_state_dict(&trained.state_dict()).unwrap();
        empty
            .load_state_dict_with(&trained.state_dict(), assign())
            .unwrap();
        assert!(on_meta(&*empty, Device::Cpu));
        for m in [&trained, &copied, &empty] {
            m.eval();
        }
        let x = input();
        assert_eq!(
            empty.forward(&x).to_vec::<f32>(),
            copied.forward(&x).to_vec::<f32>(),
            "{}",
            empty.type_name()
        );
    }
    // Without strict, what isn't loaded stays on the meta device.
    let m = homura::with_device(Device::Meta, || Linear::new(3, 2));
    let report = m
        .load_state_dict_with(
            &sd(vec![("weight", homura::ones([2, 3]))]),
            assign().strict(false),
        )
        .unwrap();
    assert_eq!(report.missing_keys, ["bias"]);
    assert_eq!(
        (m.weight.device(), m.bias.as_ref().unwrap().device()),
        (Device::Cpu, Device::Meta)
    );
}

#[test]
fn entries_on_the_meta_device_are_assigned() {
    let m = Linear::new(3, 2);
    let meta = |shape: &[usize]| homura::zeros_with(shape, Device::Meta);
    let entries = sd(vec![("weight", meta(&[2, 3])), ("bias", meta(&[2]))]);
    m.load_state_dict_with(&entries, assign()).unwrap();
    assert_eq!(m.weight.device(), Device::Meta);
    assert_eq!(m.bias.as_ref().unwrap().device(), Device::Meta);
    // Copying from them fails.
    let error = Linear::new(3, 2).load_state_dict(&entries).unwrap_err();
    assert_eq!(
        error.errors[0],
        copy_error(
            "weight",
            "[2, 3]",
            "Cannot copy out of meta tensor; no data!"
        )
    );
}

/// The same `Linear` under two names.
#[derive(Module)]
struct Tied {
    a: std::sync::Arc<Linear>,
    b: std::sync::Arc<Linear>,
}

#[test]
fn shared_parameters_take_the_last_entry() {
    let shared = std::sync::Arc::new(Linear::new(2, 2));
    let t = Tied {
        a: shared.clone(),
        b: shared,
    };
    let keys: Vec<String> = t.state_dict().keys().map(String::from).collect();
    assert_eq!(keys, ["a.weight", "a.bias", "b.weight", "b.bias"]);
    let entries = sd(vec![
        ("a.weight", homura::ones([2, 2])),
        ("a.bias", homura::zeros([2])),
        ("b.weight", homura::full([2, 2], 2.0)),
        ("b.bias", homura::zeros([2])),
    ]);
    t.load_state_dict_with(&entries, assign()).unwrap();
    assert!(std::sync::Arc::ptr_eq(&t.a, &t.b));
    assert_eq!(t.a.weight.to_vec::<f32>(), [2.0; 4]);
}

/// A module with one parameter.
#[derive(Module)]
struct One {
    w: Parameter,
}

fn one(values: &[f32]) -> One {
    One {
        w: Parameter::new(homura::tensor(values)),
    }
}

#[test]
fn graphs_of_the_entries_still_run() {
    // An entry that requires grad, saved by its own graph: assigning it leaves that graph alone
    // (its storage keeps its version).
    let source = homura::tensor([1.0f32, 2.0]);
    source.requires_grad_(true);
    let y = (&source * &source).sum();
    let m = one(&[0.0, 0.0]);
    m.load_state_dict_with(&sd(vec![("w", source.clone())]), assign())
        .unwrap();
    y.backward();
    assert_eq!(source.grad().unwrap().to_vec::<f32>(), [2.0, 4.0]);
    assert!(m.w.is_leaf() && m.w.grad().is_none());
    // The parameter is the entry's data, without its history or gradient.
    homura::with_no_grad(|| m.w.add_(1.0));
    assert_eq!(source.to_vec::<f32>(), [2.0, 3.0]);
}

#[test]
fn entries_are_read_before_any_is_assigned() {
    // The entry of `b` is the parameter `a` itself (not a detached copy): `b` gets what `a` held
    // before the load.
    let m = Two {
        a: Parameter::new(homura::tensor([1.0f32, 2.0])),
        b: Parameter::new(homura::zeros([2])),
    };
    let entries = sd(vec![
        ("a", homura::tensor([5.0f32, 6.0])),
        ("b", (*m.a).clone()),
    ]);
    m.load_state_dict_with(&entries, assign()).unwrap();
    assert_eq!(m.a.to_vec::<f32>(), [5.0, 6.0]);
    assert_eq!(m.b.to_vec::<f32>(), [1.0, 2.0]);
}

// What follows has no counterpart in PyTorch, which replaces the parameters: graphs and views
// recorded before keep the old parameter there.

#[test]
fn old_graphs_are_refused_whichever_way_they_reach_the_leaf() {
    let m = one(&[1.0, 2.0]);
    let w = m.w.clone();
    let accumulated = (&*w + 1.0).sum();
    let saved = (&*w * &*w).sum();
    let inner = &*w * 3.0;
    let through_inner = (&inner * &inner).sum();
    // The same dtype, device and layout, and a fresh storage of the same version as the old one.
    let entries = sd(vec![("w", homura::tensor([5.0f32, 6.0]))]);
    m.load_state_dict_with(&entries, assign()).unwrap();
    // (a) `backward` reaching the old accumulator, or unpacking the saved leaf.
    assert_eq!(panic_message(|| accumulated.backward()), ACCUMULATED);
    assert_eq!(panic_message(|| saved.backward()), SAVED);
    assert!(w.grad().is_none());
    // (b) `grad` for the parameter: the old graph doesn't reach its new accumulator.
    let options = |allow_unused| GradOptions {
        allow_unused,
        retain_graph: Some(true),
        ..Default::default()
    };
    let unused = panic_message(|| drop(autograd::grad(&[&accumulated], &[&w], options(false))));
    assert!(
        unused.starts_with("The differentiated Tensor at index 0 appears to not have been used"),
        "{unused}"
    );
    assert!(autograd::grad(&[&accumulated], &[&w], options(true))[0].is_none());
    // (c) `grad` for an old intermediate: fine unless the way to it unpacks the replaced leaf.
    let [g] = autograd::grad_array(&[&through_inner], [&inner], options(false));
    assert_eq!(g.unwrap().to_vec::<f32>(), [6.0, 12.0]);
    let [_] = autograd::grad_array(&[&saved], [&inner], options(true));
    // New graphs run.
    (&*w * &*w).sum().backward();
    assert_eq!(w.grad().unwrap().to_vec::<f32>(), [10.0, 12.0]);
}

#[test]
fn old_views_are_stale_and_detached_aliases_keep_the_old_data() {
    // A new storage, the same storage with another layout, and the same storage and layout
    // (assigning the module's own state dict).
    let square = || One {
        w: Parameter::new(homura::tensor([[1.0f32, 2.0], [3.0, 4.0]])),
    };
    type Entries = fn(&One) -> StateDict;
    let cases: Vec<(&str, Entries)> = vec![
        ("new storage", |_| sd(vec![("w", homura::zeros([2, 2]))])),
        ("transposed", |m| {
            sd(vec![("w", m.w.detach().transpose(0, 1))])
        }),
        ("itself", |m| m.state_dict()),
    ];
    for (name, entries) in cases {
        let m = square();
        let view = m.w.view([4]);
        let view_of_view = view.view([2, 2]);
        let alias = m.w.detach();
        let entries = entries(&m);
        m.load_state_dict_with(&entries, assign()).unwrap();
        assert_eq!(panic_message(|| drop(&view * 2.0)), STALE_VIEW, "{name}");
        assert_eq!(
            panic_message(|| drop(&view_of_view * 2.0)),
            STALE_VIEW,
            "{name}"
        );
        // A view created now works.
        assert_eq!(m.w.select(0, 0).mul(1.0).shape(), [2]);
        // `detach` aliases aren't views: they keep the data they had.
        assert_eq!(alias.to_vec::<f32>(), [1.0, 2.0, 3.0, 4.0], "{name}");
    }
    let m = square();
    m.load_state_dict_with(&sd(vec![("w", m.w.detach().transpose(0, 1))]), assign())
        .unwrap();
    assert_eq!(m.w.to_vec::<f32>(), [1.0, 3.0, 2.0, 4.0]);
    assert_eq!(m.w.stride(), [1, 2]);
}

/// Two parameters.
#[derive(Module)]
struct Two {
    a: Parameter,
    b: Parameter,
}

#[test]
fn targets_sharing_one_source_storage_are_replaced_independently() {
    let m = Two {
        a: Parameter::new(homura::zeros([2])),
        b: Parameter::new(homura::zeros([2])),
    };
    let old_a = (&*m.a * &*m.a).sum();
    let base = homura::tensor([[1.0f32, 2.0], [3.0, 4.0]]);
    let entries = sd(vec![("a", base.select(0, 0)), ("b", base.select(0, 1))]);
    let base_graph_input = homura::tensor([1.0f32]);
    base_graph_input.requires_grad_(true);
    m.load_state_dict_with(&entries, assign()).unwrap();
    assert_eq!(m.a.to_vec::<f32>(), [1.0, 2.0]);
    assert_eq!(
        (m.b.to_vec::<f32>(), m.b.storage_offset()),
        (vec![3.0, 4.0], 2)
    );
    assert_eq!(panic_message(|| old_a.backward()), SAVED);
    // Writing one target is a real modification of the shared storage.
    homura::with_no_grad(|| m.a.add_(1.0));
    assert_eq!(base.to_vec::<f32>(), [2.0, 3.0, 3.0, 4.0]);
    let y = (&*m.b * &*m.b).sum();
    homura::with_no_grad(|| m.a.add_(1.0));
    assert!(
        panic_message(|| y.backward())
            .starts_with("one of the variables needed for gradient computation has been modified")
    );
}

/// A module whose buffer is a view, or has a history.
#[derive(Module)]
struct Buffered {
    buf: Tensor,
}

#[test]
fn targets_that_are_views_or_have_a_history_are_errors() {
    let base = homura::zeros([4]);
    let leaf = homura::ones([2]);
    leaf.requires_grad_(true);
    let cases = [
        ("a view", base.view([2, 2]).select(0, 0)),
        ("a history", leaf.mul(2.0)),
    ];
    for (name, buf) in cases {
        let m = Buffered { buf: buf.clone() };
        let y = buf.sum();
        let entries = sd(vec![("buf", homura::full([2], 7.0))]);
        let error = m.load_state_dict_with(&entries, assign()).unwrap_err();
        assert_eq!(
            error.errors,
            [copy_error(
                "buf",
                "[2]",
                "it is a view or has a history, so its data can't be replaced; load it without \
                 assign"
            )],
            "{name}"
        );
        assert!(m.buf.is_same(&buf));
        assert_ne!(m.buf.to_vec::<f32>(), [7.0, 7.0]);
        // Copying works.
        homura::with_no_grad(|| m.load_state_dict(&entries).unwrap());
        assert_eq!(m.buf.to_vec::<f32>(), [7.0, 7.0], "{name}");
        drop(y);
    }
}

#[test]
fn failed_loads_change_nothing() {
    let m = Two {
        a: Parameter::new(homura::tensor([1.0f32, 2.0])),
        b: Parameter::new(homura::tensor([3.0f32, 4.0])),
    };
    m.a.set_grad(Some(homura::ones([2])));
    let view = m.a.view([2]);
    let old = (&*m.a * &*m.b).sum();
    // `a` could be assigned, `b` can't: nothing is.
    let entries = sd(vec![
        ("a", homura::zeros_with([2], DType::F64)),
        ("b", homura::zeros_with([2], DType::I32)),
    ]);
    assert!(m.load_state_dict_with(&entries, assign()).is_err());
    assert_eq!(
        (m.a.dtype(), m.a.to_vec::<f32>()),
        (DType::F32, vec![1.0, 2.0])
    );
    assert_eq!(m.a.grad().unwrap().to_vec::<f32>(), [1.0, 1.0]);
    // The old graph and view still work: the data wasn't replaced.
    assert_eq!(view.mul(2.0).to_vec::<f32>(), [2.0, 4.0]);
    old.backward();
    assert_eq!(m.b.grad().unwrap().to_vec::<f32>(), [1.0, 2.0]);
}

#[test]
fn optimizers_keep_the_parameters_but_not_always_their_state() {
    // Without state, or with state of the same dtype and device: the step continues.
    let m = one(&[1.0, 2.0]);
    let mut sgd = Sgd::new(m.parameters(), 0.5);
    m.load_state_dict_with(&sd(vec![("w", homura::tensor([4.0f32, 6.0]))]), assign())
        .unwrap();
    (&*m.w * 1.0).sum().backward();
    sgd.step();
    assert_eq!(m.w.to_vec::<f32>(), [3.5, 5.5]);
    let mut adam = Adam::new(m.parameters(), 0.1);
    (&*m.w * 1.0).sum().backward();
    adam.step();
    // A change of dtype: the state no longer matches, and the step refuses it.
    m.load_state_dict_with(
        &sd(vec![("w", homura::ones_with([2], DType::F64))]),
        assign(),
    )
    .unwrap();
    (&*m.w * 1.0).sum().backward();
    assert_eq!(
        panic_message(|| adam.step()),
        "the optimizer state exp_avg of a parameter has shape [2], dtype f32 and device cpu, but \
         the parameter has shape [2], dtype f64 and device cpu; move the model before creating the \
         optimizer"
    );
}
