//! Loading state dicts by assignment (`LoadOptions::assign`), models built on the meta device,
//! the versions of modules, the prefixes of keys and old state dicts without a buffer added later
//! (`added_in`). tools/golden/check_m3_load.py asserts the reference behavior of each test with a
//! PyTorch counterpart, in the same order; the tests after those cover what homura does
//! differently: parameters keep their identity, and graphs and views of the old data are refused,
//! whichever way they reach it; and what PyTorch has no counterpart for.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use homura::autograd::{self, GradOptions};
use homura::nn::{
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout, Embedding, Flatten, Forward, Gelu, LayerNorm,
    Linear, LoadOptions, MaxPool2d, Member, Module, ModuleVersions, Parameter, Relu, RmsNorm,
    Sequential, Sigmoid, Silu, Softmax, StateDict, Tanh, Visitor,
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

#[derive(Module)]
struct Net {
    conv: Conv2d,
    bn: Arc<BatchNorm2d>,
    blocks: Vec<BatchNorm2d>,
    empty: Vec<Linear>,
    heads: BTreeMap<String, Linear>,
    skip: Option<Linear>,
    /// The same module under a second name.
    tied: Arc<BatchNorm2d>,
}

impl Net {
    fn new() -> Net {
        let bn = Arc::new(BatchNorm2d::new(2));
        Net {
            conv: Conv2d::new(1, 2, 1),
            bn: bn.clone(),
            blocks: vec![BatchNorm2d::new(2), BatchNorm2d::new(2)],
            empty: vec![],
            heads: BTreeMap::from([("a".to_string(), Linear::new(2, 2))]),
            skip: None,
            tied: bn,
        }
    }
}

const NET_VERSIONS: [(&str, u32); 10] = [
    ("", 1),
    ("conv", 1),
    ("bn", 2),
    ("blocks", 1),
    ("blocks.0", 2),
    ("blocks.1", 2),
    ("empty", 1),
    ("heads", 1),
    ("heads.a", 1),
    ("tied", 2),
];

fn versions(sd: &StateDict) -> Vec<(&str, u32)> {
    sd.versions().expect("versions").iter().collect()
}

fn keys(sd: &StateDict) -> Vec<&str> {
    sd.keys().collect()
}

#[test]
fn state_dicts_record_module_versions() {
    // Every module by its path, before its submodules; a shared module under each name, a `None`
    // one not at all.
    assert_eq!(versions(&Net::new().state_dict()), NET_VERSIONS);
    let model = Sequential::new()
        .append(Linear::new(2, 2))
        .append(BatchNorm2d::new(2));
    assert_eq!(versions(&model.state_dict()), [("", 1), ("0", 1), ("1", 2)]);
    assert_eq!(versions(&BatchNorm2d::new(2).state_dict()), [("", 2)]);
    // Only the batch normalization layers have another version than 1.
    let others: Vec<Box<dyn Module>> = vec![
        Box::new(Linear::new(2, 2)),
        Box::new(Conv2d::new(1, 1, 1)),
        Box::new(LayerNorm::new([2])),
        Box::new(RmsNorm::new([2])),
        Box::new(Embedding::new(2, 2)),
        Box::new(Dropout::new(0.1)),
        Box::new(Relu),
        Box::new(Gelu::new()),
        Box::new(Silu),
        Box::new(Tanh),
        Box::new(Sigmoid),
        Box::new(Softmax::new(-1)),
        Box::new(MaxPool2d::new(2)),
        Box::new(AdaptiveAvgPool2d::new(1)),
        Box::new(Flatten::default()),
        Box::new(Sequential::new()),
        Box::new(Vec::<Linear>::new()),
        Box::new(BTreeMap::<String, Linear>::new()),
    ];
    for m in others {
        assert_eq!(m.version(), 1, "{}", m.type_name());
    }
    assert_eq!(BatchNorm2d::new(2).version(), 2);
}

/// A state dict with `keys` and the versions `paths`, after `consume_prefix(prefix)`.
fn consumed(keys: &[&str], paths: &[(&str, u32)], prefix: &str) -> StateDict {
    let mut sd: StateDict = keys.iter().map(|&k| (k, homura::zeros([1]))).collect();
    *sd.versions_mut() = Some(paths.iter().copied().collect());
    sd.consume_prefix(prefix);
    sd
}

#[test]
fn prefixes_move_keys_and_versions() {
    // Renamed entries go last; the module named by the prefix becomes the root.
    let sd = consumed(
        &["module.a", "b", "module.c.d"],
        &[("", 1), ("module", 3), ("module.c", 2), ("x", 1)],
        "module.",
    );
    assert_eq!(keys(&sd), ["b", "a", "c.d"]);
    assert_eq!(versions(&sd), [("", 3), ("x", 1), ("c", 2)]);
    // A new key that exists replaces that entry in its position.
    let mut sd = consumed(&["a", "module.a"], &[("", 7), ("module", 3)], "module.");
    assert_eq!(keys(&sd), ["a"]);
    assert_eq!(versions(&sd), [("", 3)]);
    // The prefix is removed once.
    let sd2 = consumed(&["module.module.a"], &[("module.module", 3)], "module.");
    assert_eq!(
        (keys(&sd2), versions(&sd2)),
        (vec!["module.a"], vec![("module", 3)])
    );
    // With a "." inside the prefix, homura renames the version of "a.b" (PyTorch that of "ab").
    let sd2 = consumed(
        &["a.b.w", "a.bw"],
        &[("", 1), ("a", 1), ("a.b", 2), ("ab", 4), ("a.b.c", 3)],
        "a.b.",
    );
    assert_eq!(keys(&sd2), ["a.bw", "w"]);
    assert_eq!(versions(&sd2), [("", 2), ("a", 1), ("ab", 4), ("c", 3)]);
    // Without versions, only the keys change.
    *sd.versions_mut() = None;
    sd.consume_prefix("a");
    assert_eq!((keys(&sd), sd.versions()), (vec![""], None));

    // Adding a prefix, as `state_dict(prefix=...)`.
    let net = Net::new();
    let mut sd = net.state_dict();
    sd.add_prefix("model.");
    assert_eq!(keys(&sd)[..2], ["model.conv.weight", "model.conv.bias"]);
    let mut expected = vec![("model".to_string(), 1)];
    expected.extend(
        NET_VERSIONS[1..]
            .iter()
            .map(|&(p, v)| (format!("model.{p}"), v)),
    );
    let expected: Vec<(&str, u32)> = expected.iter().map(|(p, v)| (p.as_str(), *v)).collect();
    assert_eq!(versions(&sd), expected);
    // `consume_prefix` undoes it.
    sd.consume_prefix("model.");
    assert_eq!(versions(&sd), NET_VERSIONS);
    assert_eq!(keys(&sd), keys(&net.state_dict()));

    // The state dicts of submodules, as their own `state_dict()` in PyTorch.
    let sd = net.state_dict();
    let child = sd.sub("blocks.1.");
    assert_eq!(
        keys(&child),
        [
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked"
        ]
    );
    assert_eq!(versions(&child), [("", 2)]);
    assert!(child["weight"].is_same(&sd["blocks.1.weight"]));
    net.blocks[1].load_state_dict(&child).unwrap();
    let heads = sd.sub("heads.");
    assert_eq!(keys(&heads), ["a.weight", "a.bias"]);
    assert_eq!(versions(&heads), [("", 1), ("a", 1)]);
    net.heads.load_state_dict(&heads).unwrap();
    assert!(sd.sub("nothing.").is_empty());
    assert_eq!(versions(&sd.sub("nothing.")), []);
}

/// What `versions` a state dict without the count of a `BatchNorm2d` has.
#[derive(Clone, Copy)]
enum Old {
    /// No versions.
    Absent,
    /// Versions, without one for the layer.
    Elsewhere,
    Version(u32),
}

/// The missing keys, then the device, dtype and value of the count, and whether its data stayed.
type Count = (Vec<String>, Device, DType, Option<i64>, bool);

/// Loads a state dict without the count into a `BatchNorm2d` whose count is 5 (on the CPU).
fn load_old(old: Old, device: Device, assign: bool, strict: bool) -> Result<Count, String> {
    let bn = homura::with_device(device, || BatchNorm2d::new(2));
    let count = bn.num_batches_tracked.clone().unwrap();
    if device == Device::Cpu {
        count.fill_(5);
    }
    let alias = count.detach();
    let mut sd = BatchNorm2d::new(2).state_dict();
    sd.remove("num_batches_tracked");
    *sd.versions_mut() = match old {
        Old::Absent => None,
        Old::Elsewhere => Some(ModuleVersions::from_iter([("x", 1)])),
        Old::Version(v) => Some(ModuleVersions::from_iter([("", v)])),
    };
    let options = LoadOptions::default().strict(strict).assign(assign);
    let report = bn
        .load_state_dict_with(&sd, options)
        .map_err(|e| e.to_string())?;
    let n = bn.num_batches_tracked.as_ref().unwrap();
    let (value, kept) = match n.device() {
        Device::Meta => (None, alias.device() == Device::Meta),
        _ => {
            let value = n.item::<i64>();
            n.fill_(9);
            let kept = alias.device() == n.device() && alias.item::<i64>() == 9;
            (Some(value), kept)
        }
    };
    Ok((report.missing_keys, n.device(), n.dtype(), value, kept))
}

#[test]
fn old_batch_norm_state_dicts_keep_the_count() {
    let cpu = Device::Cpu;
    // Without versions, without one for the module, or of version 1: the count stays; on the
    // meta device, copying leaves it there and assigning gives it a 0 on the CPU.
    for old in [Old::Absent, Old::Elsewhere, Old::Version(1)] {
        for assign in [false, true] {
            let count = load_old(old, cpu, assign, true).unwrap();
            assert_eq!(count, (vec![], cpu, DType::I64, Some(5), true));
        }
        let count = load_old(old, Device::Meta, false, true).unwrap();
        assert_eq!(count, (vec![], Device::Meta, DType::I64, None, true));
        let count = load_old(old, Device::Meta, true, true).unwrap();
        assert_eq!(count, (vec![], cpu, DType::I64, Some(0), false));
    }
    // Version 2 has the count: it is missing.
    let missing = "Missing key(s) in state_dict: \"num_batches_tracked\".";
    for device in [cpu, Device::Meta] {
        for assign in [false, true] {
            let e = load_old(Old::Version(2), device, assign, true).unwrap_err();
            assert!(e.contains(missing), "{e}");
        }
    }
    let count = load_old(Old::Version(2), cpu, false, false).unwrap();
    let missing_keys = vec!["num_batches_tracked".to_string()];
    assert_eq!(count, (missing_keys, cpu, DType::I64, Some(5), true));
    // An old state dict that has the count loads it.
    let mut sd = BatchNorm2d::new(2).state_dict();
    sd["num_batches_tracked"].fill_(3);
    sd.set_module_version("", 1);
    for options in [LoadOptions::default(), assign()] {
        let bn = BatchNorm2d::new(2);
        bn.load_state_dict_with(&sd, options).unwrap();
        assert_eq!(bn.num_batches_tracked.as_ref().unwrap().item::<i64>(), 3);
    }
    // The version of the module's own path counts, in a whole model.
    let mut sd = Net::new().state_dict();
    sd.remove("blocks.1.num_batches_tracked");
    let e = Net::new().load_state_dict(&sd).unwrap_err();
    assert_eq!(e.missing_keys, ["blocks.1.num_batches_tracked"]);
    sd.set_module_version("blocks.1", 1);
    assert!(
        Net::new()
            .load_state_dict(&sd)
            .unwrap()
            .missing_keys
            .is_empty()
    );
    sd.versions_mut().as_mut().unwrap().remove("blocks.1");
    assert!(
        Net::new()
            .load_state_dict(&sd)
            .unwrap()
            .missing_keys
            .is_empty()
    );
    // Without running statistics there is no count to keep.
    let plain = BatchNorm2d::new(2).track_running_stats(false);
    assert!(plain.num_batches_tracked.is_none());
    let entries = sd_of(&[("weight", homura::ones([2])), ("bias", homura::zeros([2]))]);
    assert!(
        plain
            .load_state_dict(&entries)
            .unwrap()
            .missing_keys
            .is_empty()
    );
}

fn sd_of(entries: &[(&str, Tensor)]) -> StateDict {
    entries.iter().map(|(k, t)| (*k, t.clone())).collect()
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

// What follows has no counterpart in PyTorch: `added_in` on other modules than batch
// normalization, and the API of versions and keys.

#[derive(Module)]
#[module(version = 3)]
struct Counted {
    w: Parameter,
    #[module(added_in = 2)]
    count: Tensor,
    #[module(added_in = 3, rename = "steps")]
    maybe: Option<Tensor>,
}

impl Counted {
    fn new(count: Tensor, maybe: Option<Tensor>) -> Counted {
        Counted {
            w: Parameter::new(homura::zeros([2])),
            count,
            maybe,
        }
    }
}

#[test]
fn added_in_buffers_follow_their_versions() {
    let m = Counted::new(homura::zeros_with([], DType::I32), None);
    assert_eq!(
        (m.version(), m.added_in("count"), m.added_in("steps")),
        (3, Some(2), Some(3))
    );
    assert_eq!((m.added_in("maybe"), m.added_in("w")), (None, None));
    m.count.fill_(4);
    let w = sd_of(&[("w", homura::ones([2]))]);
    // A `None` buffer has no key; the count is kept below version 2.
    for old in [None, Some(1)] {
        let mut sd = w.clone();
        if let Some(v) = old {
            sd.set_module_version("", v);
        }
        assert!(m.load_state_dict(&sd).unwrap().missing_keys.is_empty());
        assert_eq!(m.count.item::<i32>(), 4);
    }
    let mut sd = w.clone();
    sd.set_module_version("", 2);
    assert_eq!(m.load_state_dict(&sd).unwrap_err().missing_keys, ["count"]);
    // A renamed field, under its key.
    let m = Counted::new(homura::tensor(1i64), Some(homura::tensor(2i64)));
    sd.insert("count", homura::tensor(7i64));
    assert!(m.load_state_dict(&sd).unwrap().missing_keys.is_empty());
    assert_eq!(
        (
            m.count.item::<i64>(),
            m.maybe.as_ref().unwrap().item::<i64>()
        ),
        (7, 2)
    );
    sd.set_module_version("", 3);
    assert_eq!(m.load_state_dict(&sd).unwrap_err().missing_keys, ["steps"]);
    // On the meta device, assignment gives a 0 of the buffer's own dtype (PyTorch's BatchNorm2d
    // always an I64 one).
    let m = homura::with_device(Device::Meta, || {
        Counted::new(homura::zeros_with([], DType::I32), None)
    });
    m.load_state_dict_with(&w, assign()).unwrap();
    assert_eq!(
        (m.count.device(), m.count.dtype()),
        (Device::Cpu, DType::I32)
    );
    assert_eq!(m.count.item::<i32>(), 0);
    assert_eq!(m.w.device(), Device::Cpu);
    // A view can't get it; copying leaves it alone.
    let view = homura::with_device(Device::Meta, || homura::zeros_with([2], DType::I64)).index(0);
    let m = Counted::new(view, None);
    let e = m.load_state_dict_with(&w, assign()).unwrap_err();
    assert_eq!(
        e.errors,
        [
            "\"count\" can't be set to 0: it is a view or has a history, so its data can't be \
             replaced; load it without assign"
        ]
    );
    assert_eq!(m.w.to_vec::<f32>(), [0.0, 0.0]);
    m.load_state_dict(&w).unwrap();
    assert_eq!(m.count.device(), Device::Meta);
}

#[test]
fn added_in_needs_zero_dimensional_integer_buffers() {
    let error = |what: &str| {
        format!("\"count\" is {what}, but added_in only supports zero-dimensional integer buffers")
    };
    let cases = [
        (
            homura::zeros_with([2], DType::I64),
            "a buffer of shape [2] and dtype i64",
        ),
        (homura::zeros([]), "a buffer of shape [] and dtype f32"),
        (homura::tensor(true), "a buffer of shape [] and dtype bool"),
    ];
    for (count, what) in cases {
        let m = Counted::new(count, None);
        // Also with the entry, and without strict or with assign: nothing is written.
        let entries = sd_of(&[("w", homura::ones([2])), ("count", m.count.clone())]);
        for options in [
            LoadOptions::default(),
            LoadOptions::default().strict(false),
            assign(),
        ] {
            let e = m.load_state_dict_with(&entries, options).unwrap_err();
            assert_eq!(e.errors, [error(what)]);
            assert!(e.missing_keys.is_empty());
            assert_eq!(m.w.to_vec::<f32>(), [0.0, 0.0]);
        }
    }
    // A hand-written module that says it added a parameter, or a buffer outside the state dict.
    struct Hand {
        count: Parameter,
        cache: Tensor,
    }
    impl Module for Hand {
        fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
            self.count.visit_as("count", v);
            v.buffer("cache", &self.cache, false);
        }

        fn added_in(&self, _: &str) -> Option<u32> {
            Some(2)
        }
    }
    impl Member for Hand {
        fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
            v.module(name, self, None);
        }
    }
    let hand = Hand {
        count: Parameter::with_requires_grad(homura::zeros_with([], DType::I64), false),
        cache: homura::zeros([2]),
    };
    let e = hand.load_state_dict(&StateDict::new()).unwrap_err();
    assert_eq!(e.errors, [error("a parameter of shape [] and dtype i64")]);
}

#[test]
fn wrappers_give_the_versions_of_their_modules() {
    let bn = || BatchNorm2d::new(2);
    let key = "num_batches_tracked";
    let dynamic: Box<dyn Module> = Box::new(bn());
    let shared: Arc<dyn Module + Send + Sync> = Arc::new(bn());
    let wrapped: [&dyn Module; 5] = [
        &Box::new(bn()),
        &Arc::new(bn()),
        &Some(bn()),
        &dynamic,
        &shared,
    ];
    for m in wrapped {
        assert_eq!((m.version(), m.added_in(key)), (2, Some(2)));
        assert_eq!(versions(&m.state_dict()), [("", 2)]);
    }
    let none: Option<BatchNorm2d> = None;
    assert_eq!((none.version(), none.added_in(key)), (1, None));
    assert_eq!(versions(&none.state_dict()), []);
    // A derived module's field of a wrapper type.
    #[derive(Module)]
    struct Holder {
        norm: Box<BatchNorm2d>,
    }
    let holder = Holder {
        norm: Box::new(bn()),
    };
    assert_eq!(versions(&holder.state_dict()), [("", 1), ("norm", 2)]);
    let mut sd = holder.state_dict();
    sd.remove("norm.num_batches_tracked");
    assert_eq!(
        holder.load_state_dict(&sd).unwrap_err().missing_keys,
        ["norm.num_batches_tracked"]
    );
    sd.set_module_version("norm", 1);
    holder.load_state_dict(&sd).unwrap();
}

#[test]
fn versions_can_be_read_and_changed() {
    let mut sd = Linear::new(2, 2).state_dict();
    assert_eq!(sd.module_version(""), Some(1));
    assert_eq!(sd.module_version("x"), None);
    assert_eq!(sd.set_module_version("x", 4), None);
    assert_eq!(sd.set_module_version("", 2), Some(1));
    assert_eq!(versions(&sd), [("", 2), ("x", 4)]);
    // Clones keep them; inserting and removing entries leaves them alone.
    let copy = sd.clone();
    sd.remove("weight");
    sd.insert("other", homura::zeros([1]));
    assert_eq!(copy.versions(), sd.versions());
    // A state dict collected from entries has none until one is set.
    let mut collected: StateDict = sd.iter().map(|(k, t)| (k, t.clone())).collect();
    assert!(collected.versions().is_none() && StateDict::new().versions().is_none());
    assert_eq!(collected.module_version(""), None);
    collected.set_module_version("", 2);
    assert_eq!(versions(&collected), [("", 2)]);
    *collected.versions_mut() = None;
    assert!(collected.versions().is_none());
    let mut sub = collected.sub("");
    assert!(sub.versions().is_none());
    sub.add_prefix("p.");
    assert!(sub.versions().is_none());
    // ModuleVersions: ordered, compared in order.
    let mut v = ModuleVersions::from_iter([("a", 1), ("b", 2)]);
    assert_eq!(
        (v.len(), v.is_empty(), v.get("b"), v.get("c")),
        (2, false, Some(2), None)
    );
    assert_eq!(v.insert("a", 3), Some(1));
    assert_eq!(v.iter().collect::<Vec<_>>(), [("a", 3), ("b", 2)]);
    assert_ne!(v, ModuleVersions::from_iter([("b", 2), ("a", 3)]));
    assert_eq!(format!("{v:?}"), r#"{"a": 3, "b": 2}"#);
    assert_eq!(v.remove("a"), Some(3));
    assert_eq!(v.remove("a"), None);
    v.extend([("c", 1)]);
    let owned: Vec<(String, u32)> = v.into_iter().collect();
    assert_eq!(owned, [("b".to_string(), 2), ("c".to_string(), 1)]);
    assert!(ModuleVersions::new().is_empty());
}

#[test]
fn map_keys_renames_keys_and_drops_the_versions() {
    let sd = Net::new().state_dict();
    let renamed = sd.clone().map_keys(|k| {
        k.strip_prefix("blocks.")
            .map(|rest| format!("layers.{rest}"))
    });
    assert_eq!(keys(&renamed)[..2], ["layers.0.weight", "layers.0.bias"]);
    assert_eq!(renamed.len(), 10);
    assert!(renamed.versions().is_none());
    assert!(renamed["layers.1.weight"].is_same(&sd["blocks.1.weight"]));
    // Two entries with the same new key: every entry is renamed first.
    let mut calls = 0;
    let message = panic_message(|| {
        let _ = sd.clone().map_keys(|k| {
            calls += 1;
            Some(k.rsplit('.').next().unwrap().to_string())
        });
    });
    assert_eq!(
        message,
        r#"map_keys gave the key "weight" to both "conv.weight" and "bn.weight""#
    );
    assert_eq!(calls, sd.len());
}
