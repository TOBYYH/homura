//! `nn::Module` traversals, state dicts, loading and in-place conversion, with modules that
//! implement `Module` by hand. tools/golden/check_m2_module.py asserts the reference behavior of
//! each test, in the same order, on the same module trees.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};

use homura::autograd::{self, GradOptions};
use homura::nn::{
    LoadOptions, LoadReport, Member, Module, Parameter, StateDict, Training, Visitor,
};
use homura::{DType, Device, Tensor, TensorOptions};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

fn keys<T>(pairs: Vec<(String, T)>) -> Vec<String> {
    pairs.into_iter().map(|(k, _)| k).collect()
}

fn state_keys(m: &dyn Module) -> Vec<String> {
    m.state_dict().keys().map(String::from).collect()
}

/// Implements `Member` for a module type: it is a submodule.
macro_rules! member {
    ($t:ident $(<$p:ident: $bound:ident>)?) => {
        impl$(<$p: $bound>)? Member for $t$(<$p>)? {
            fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
                v.module(name, self, None);
            }
        }
    };
}

/// Implements `Module` and `Member` for a struct whose fields are all members, named after the
/// fields.
macro_rules! module {
    ($t:ident { $($field:ident),* }) => {
        impl Module for $t {
            fn visit<'a>(&'a self, _v: &mut dyn Visitor<'a>) {
                $(self.$field.visit_as(stringify!($field), _v);)*
            }
        }

        member!($t);
    };
}

struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
}

impl Linear {
    fn new(inputs: usize, outputs: usize, bias: bool) -> Linear {
        Linear {
            weight: Parameter::new(homura::zeros([outputs, inputs])),
            bias: bias.then(|| Parameter::new(homura::zeros([outputs]))),
        }
    }
}

module!(Linear { weight, bias });

struct Relu;

module!(Relu {});

struct Stats {
    running_mean: Tensor,
    num_batches_tracked: Tensor,
    training: Training,
}

impl Stats {
    fn new() -> Stats {
        Stats {
            running_mean: homura::zeros([3]),
            num_batches_tracked: homura::zeros_with([], DType::I64),
            training: Training::default(),
        }
    }
}

module!(Stats {
    running_mean,
    num_batches_tracked,
    training
});

struct Block {
    lin: Linear,
    stats: Stats,
}

module!(Block { lin, stats });

/// The members in an order unlike the traversal's: buffers after submodules, a non-persistent
/// buffer, hyperparameters, a container of hyperparameters, an empty container and a `BTreeMap`
/// filled out of key order.
struct Net {
    embed: Parameter,
    blocks: Vec<Block>,
    relu: Relu,
    head: Linear,
    scale: Tensor,
    cache: Tensor,
    table: BTreeMap<String, Linear>,
    empty: Vec<Linear>,
    training: Training,
    sizes: Vec<usize>,
    name: String,
}

impl Net {
    fn new() -> Net {
        let mut table = BTreeMap::new();
        table.insert("b".to_string(), Linear::new(3, 1, true));
        table.insert("a".to_string(), Linear::new(3, 1, true));
        Net {
            embed: Parameter::new(homura::zeros([4, 3])),
            blocks: vec![
                Block {
                    lin: Linear::new(3, 3, true),
                    stats: Stats::new(),
                },
                Block {
                    lin: Linear::new(3, 3, true),
                    stats: Stats::new(),
                },
            ],
            relu: Relu,
            head: Linear::new(3, 2, false),
            scale: homura::ones([1]),
            cache: homura::zeros([2]),
            table,
            empty: vec![],
            training: Training::default(),
            sizes: vec![1, 2],
            name: "net".into(),
        }
    }
}

impl Module for Net {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        self.embed.visit_as("embed", v);
        self.blocks.visit_as("blocks", v);
        self.relu.visit_as("relu", v);
        self.head.visit_as("head", v);
        self.scale.visit_as("scale", v);
        v.buffer("cache", &self.cache, false);
        self.table.visit_as("table", v);
        self.empty.visit_as("empty", v);
        self.training.visit_as("training", v);
        self.sizes.visit_as("sizes", v);
        self.name.visit_as("name", v);
    }
}

member!(Net);

const NET_STATE: [&str; 15] = [
    "embed",
    "scale",
    "blocks.0.lin.weight",
    "blocks.0.lin.bias",
    "blocks.0.stats.running_mean",
    "blocks.0.stats.num_batches_tracked",
    "blocks.1.lin.weight",
    "blocks.1.lin.bias",
    "blocks.1.stats.running_mean",
    "blocks.1.stats.num_batches_tracked",
    "head.weight",
    "table.a.weight",
    "table.a.bias",
    "table.b.weight",
    "table.b.bias",
];

const NET_MODULES: [&str; 14] = [
    "",
    "blocks",
    "blocks.0",
    "blocks.0.lin",
    "blocks.0.stats",
    "blocks.1",
    "blocks.1.lin",
    "blocks.1.stats",
    "relu",
    "head",
    "table",
    "table.a",
    "table.b",
    "empty",
];

#[test]
fn state_dict_and_traversals_follow_pytorch_order() {
    let net = Net::new();
    assert_eq!(state_keys(&net), NET_STATE);
    assert_eq!(
        keys(net.named_parameters()),
        [
            "embed",
            "blocks.0.lin.weight",
            "blocks.0.lin.bias",
            "blocks.1.lin.weight",
            "blocks.1.lin.bias",
            "head.weight",
            "table.a.weight",
            "table.a.bias",
            "table.b.weight",
            "table.b.bias",
        ]
    );
    assert_eq!(
        keys(net.named_buffers()),
        [
            "scale",
            "cache",
            "blocks.0.stats.running_mean",
            "blocks.0.stats.num_batches_tracked",
            "blocks.1.stats.running_mean",
            "blocks.1.stats.num_batches_tracked",
        ]
    );
    assert_eq!(keys(net.named_modules()), NET_MODULES);
    assert_eq!(
        keys(net.named_children()),
        ["blocks", "relu", "head", "table", "empty"]
    );
    assert_eq!(net.children().len(), 5);
    assert_eq!(net.parameters().len(), 10);
    assert!(net.parameters()[0].is_same(&net.embed));
    assert!(net.buffers()[1].is_same(&net.cache));

    // The same through a `Box<dyn Module>` and an `Arc`, and for a container as the root.
    let boxed: Box<dyn Module> = Box::new(Net::new());
    assert_eq!(state_keys(&boxed), NET_STATE);
    assert_eq!(keys(Arc::new(Net::new()).named_modules()), NET_MODULES);
    let list = vec![Linear::new(1, 1, false), Linear::new(1, 1, true)];
    assert_eq!(state_keys(&list), ["0.weight", "1.weight", "1.bias"]);
    assert_eq!(keys(list.named_modules()), ["", "0", "1"]);
    assert!(keys(Some(Relu).named_modules()) == [""] && None::<Relu>.named_modules().is_empty());
}

/// A shared submodule through two `Arc`s, and a parameter shared with another module.
struct Tied {
    a: Arc<Linear>,
    b: Arc<Linear>,
    c: Linear,
}

module!(Tied { a, b, c });

impl Tied {
    fn new() -> Tied {
        let a = Arc::new(Linear::new(2, 2, false));
        let c = Linear {
            weight: a.weight.clone(),
            bias: None,
        };
        Tied { b: a.clone(), a, c }
    }
}

#[test]
fn shared_modules_and_parameters_are_listed_once_but_saved_under_every_key() {
    let tied = Tied::new();
    assert_eq!(state_keys(&tied), ["a.weight", "b.weight", "c.weight"]);
    assert_eq!(keys(tied.named_parameters()), ["a.weight"]);
    assert_eq!(keys(tied.named_modules()), ["", "a", "c"]);
    assert_eq!(keys(tied.named_children()), ["a", "c"]);
    assert_eq!(tied.parameters().len(), 1);
    // Every key of the state dict shows the one tensor's data.
    let sd = tied.state_dict();
    {
        let _guard = homura::no_grad();
        tied.c.weight.fill_(2.0);
    }
    for (_, value) in &sd {
        assert_eq!(value.to_vec::<f32>(), [2.0; 4]);
    }
}

struct Inner {
    w: Parameter,
}

module!(Inner { w });

/// Lives at the address of its only field.
struct Wrapper {
    inner: Inner,
}

module!(Wrapper { inner });

struct Pair {
    r1: Relu,
    r2: Relu,
}

module!(Pair { r1, r2 });

struct SharedPair {
    r1: Arc<Relu>,
    r2: Arc<Relu>,
}

module!(SharedPair { r1, r2 });

#[test]
fn modules_are_identified_by_arc_not_by_address() {
    let wrapper = Wrapper {
        inner: Inner {
            w: Parameter::new(homura::zeros([1])),
        },
    };
    assert!(std::ptr::addr_eq(&wrapper, &wrapper.inner));
    assert_eq!(keys(wrapper.named_modules()), ["", "inner"]);
    assert_eq!(keys(wrapper.named_parameters()), ["inner.w"]);

    // Zero-sized modules: values are always distinct, one `Arc` is shared, two are not.
    assert_eq!(
        keys(Pair { r1: Relu, r2: Relu }.named_modules()),
        ["", "r1", "r2"]
    );
    let relu = Arc::new(Relu);
    let shared = SharedPair {
        r1: relu.clone(),
        r2: relu,
    };
    assert_eq!(keys(shared.named_modules()), ["", "r1"]);
    assert_eq!(keys(shared.named_children()), ["r1"]);
    let distinct = SharedPair {
        r1: Arc::new(Relu),
        r2: Arc::new(Relu),
    };
    assert_eq!(keys(distinct.named_modules()), ["", "r1", "r2"]);

    // A shared root lists the same as a plain one, including a submodule it shares with itself
    // nowhere else.
    let root = Arc::new(Tied::new());
    assert_eq!(keys(root.named_modules()), ["", "a", "c"]);
    assert_eq!(state_keys(&root), ["a.weight", "b.weight", "c.weight"]);
}

#[test]
fn listed_modules_are_borrows_of_the_tree() {
    let net = Net::new();
    let modules = net.named_modules();
    let (name, block) = &modules[2];
    assert_eq!(name, "blocks.0");
    assert_eq!(keys(block.named_parameters()), ["lin.weight", "lin.bias"]);
    let children = net.children();
    assert_eq!(state_keys(children[2]), ["weight"]);
}

/// A module that can be made to contain itself through an `Arc`.
struct Cyclic {
    w: Parameter,
    next: OnceLock<Arc<Cyclic>>,
}

impl Module for Cyclic {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        self.w.visit_as("w", v);
        if let Some(next) = self.next.get() {
            next.visit_as("next", v);
        }
    }
}

member!(Cyclic);

#[test]
fn cycles_through_shared_modules_panic() {
    let a = Arc::new(Cyclic {
        w: Parameter::new(homura::zeros([1])),
        next: OnceLock::new(),
    });
    let b = Arc::new(Cyclic {
        w: Parameter::new(homura::zeros([1])),
        next: OnceLock::new(),
    });
    a.next.set(b.clone()).ok().unwrap();
    // No cycle yet.
    assert_eq!(state_keys(&a), ["w", "next.w"]);
    b.next.set(a.clone()).ok().unwrap();
    let cycle = "the module tree has a cycle: module \"next.next\" contains itself";
    assert_eq!(panic_message(|| drop(a.state_dict())), cycle);
    assert_eq!(panic_message(|| drop(a.named_parameters())), cycle);
    // The cycle of `Arc`s is leaked.
}

/// Reports each element under the name given with it.
struct Named<T>(Vec<(&'static str, T)>);

impl<T: Member> Module for Named<T> {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        for (name, member) in &self.0 {
            member.visit_as(name, v);
        }
    }
}

member!(Named<T: Member>);

/// A member of any kind.
enum AnyMember {
    Parameter(Parameter),
    Buffer(Tensor),
    Module(Relu),
}

impl Member for AnyMember {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        match self {
            AnyMember::Parameter(p) => p.visit_as(name, v),
            AnyMember::Buffer(t) => t.visit_as(name, v),
            AnyMember::Module(m) => m.visit_as(name, v),
        }
    }
}

#[test]
fn invalid_names_panic_with_pytorch_messages() {
    let named = |entries: Vec<(&'static str, AnyMember)>| {
        let m = Named(entries);
        panic_message(|| drop(m.state_dict()))
    };
    let p = || AnyMember::Parameter(Parameter::new(homura::zeros([1])));
    let t = || AnyMember::Buffer(homura::zeros([1]));
    let relu = || AnyMember::Module(Relu);
    assert_eq!(
        named(vec![("a.b", relu())]),
        "module name can't contain \".\", got: a.b"
    );
    assert_eq!(
        named(vec![("", relu())]),
        "module name can't be empty string \"\""
    );
    assert_eq!(
        named(vec![("a.b", p())]),
        "parameter name can't contain \".\""
    );
    assert_eq!(
        named(vec![("", p())]),
        "parameter name can't be empty string \"\""
    );
    assert_eq!(named(vec![("a.b", t())]), "buffer name can't contain \".\"");
    assert_eq!(
        named(vec![("", t())]),
        "buffer name can't be empty string \"\""
    );
    assert_eq!(
        named(vec![("x", t()), ("x", p())]),
        "attribute 'x' already exists"
    );
    // `BTreeMap` keys are names too.
    let mut table = BTreeMap::new();
    table.insert("a.b".to_string(), Relu);
    assert_eq!(
        panic_message(|| drop(table.named_modules())),
        "module name can't contain \".\", got: a.b"
    );
    assert_eq!(
        panic_message(|| drop(table.named_children())),
        "module name can't contain \".\", got: a.b"
    );
}

#[test]
fn train_and_eval_reach_every_flag() {
    let net = Net::new();
    let flags = |net: &Net| {
        [
            net.training.get(),
            net.blocks[0].stats.training.get(),
            net.blocks[1].stats.training.get(),
        ]
    };
    assert_eq!(flags(&net), [true; 3]);
    net.eval();
    assert_eq!(flags(&net), [false; 3]);
    net.train();
    net.blocks[0].eval();
    assert_eq!(flags(&net), [true, false, true]);
    net.train();
    assert_eq!(flags(&net), [true; 3]);
    // A clone of a flag keeps its mode but is a separate flag.
    let flag = net.training.clone();
    net.eval();
    assert!(flag.get() && !net.training.get());
}

#[test]
fn zero_grad_and_requires_grad_cover_every_parameter() {
    let tied = Tied::new();
    tied.a.weight.set_grad(Some(homura::ones([2, 2])));
    tied.zero_grad();
    assert!(tied.c.weight.grad().is_none());
    tied.requires_grad_(false);
    assert!(tied.parameters().iter().all(|p| !p.requires_grad()));
    tied.requires_grad_(true);
    assert!(tied.c.weight.requires_grad());
}

#[test]
fn parameters_share_data_without_history() {
    let x = homura::ones([2]);
    x.requires_grad_(true);
    let y = &x * 2.0;
    let p = Parameter::new(y.clone());
    assert!(p.is_leaf() && p.requires_grad() && p.grad_fn().is_none() && !p.is_same(&y));
    p.sum().backward();
    assert!(x.grad().is_none());
    assert_eq!(p.grad().unwrap().to_vec::<f32>(), [1.0, 1.0]);

    // The data is shared, also with a view, but the parameter is not a view: an in-place
    // operation recorded on it is the one on a leaf.
    let base = homura::zeros([4]);
    let q = Parameter::new(base.narrow(0, 1, 2));
    {
        let _guard = homura::no_grad();
        q.fill_(5.0);
    }
    assert_eq!(base.to_vec::<f32>(), [0.0, 5.0, 5.0, 0.0]);
    assert_eq!(
        panic_message(|| {
            q.add_(1.0);
        }),
        "a leaf Variable that requires grad is being used in an in-place operation."
    );

    assert_eq!(
        panic_message(|| drop(Parameter::new(homura::zeros_with([2], DType::I64)))),
        "Only Tensors of floating point and complex dtype can require gradients"
    );
    let count = Parameter::with_requires_grad(homura::zeros_with([2], DType::I64), false);
    assert!(!count.requires_grad());
    assert_eq!(
        format!("{:?}", Parameter::new(homura::ones([2]))),
        "Parameter containing:\ntensor([1., 1.], requires_grad=True)"
    );
    // `to` on a parameter is the tensor's, even with `Module` in scope.
    let converted: Tensor = p.to(DType::F64);
    assert_eq!((converted.dtype(), p.dtype()), (DType::F64, DType::F32));
    assert!(p.requires_grad_(false).is_same(&p) && p.as_tensor().is_same(&p));
}

#[test]
fn state_dict_values_share_storage_without_history() {
    let lin = Linear::new(2, 2, true);
    let value = lin.state_dict()["weight"].clone();
    assert!(!value.requires_grad() && value.grad_fn().is_none() && !value.is_same(&lin.weight));
    {
        let _guard = homura::no_grad();
        lin.weight.fill_(3.0);
    }
    assert_eq!(value.to_vec::<f32>(), [3.0; 4]);
}

#[test]
fn state_dicts_keep_insertion_order() {
    let mut sd = StateDict::new();
    for key in ["b", "a", "c"] {
        sd.insert(key, homura::zeros([1]));
    }
    assert!(sd.insert("a", homura::ones([2])).is_some());
    assert!(sd.remove("b").is_some() && sd.remove("b").is_none());
    assert_eq!(sd.keys().collect::<Vec<_>>(), ["a", "c"]);
    assert_eq!(sd["a"].shape(), [2]);
    assert!(sd.contains_key("c") && sd.get("b").is_none() && sd.len() == 2);
    assert_eq!(format!("{sd:?}"), "{\"a\": [2] f32, \"c\": [1] f32}");
    let pairs: Vec<(String, Tensor)> = sd.clone().into_iter().collect();
    let again: StateDict = pairs.into_iter().collect();
    assert_eq!(again.keys().collect::<Vec<_>>(), ["a", "c"]);
}

#[test]
fn load_state_dict_converts_into_the_same_tensors() {
    let tied = Tied::new();
    let weight = tied.a.weight.clone();
    let sd: StateDict = [
        ("a.weight", homura::ones_with([2, 2], DType::F64)),
        ("b.weight", homura::full([2, 2], 2.0)),
        ("c.weight", homura::full([2, 2], 3.0)),
    ]
    .into_iter()
    .collect();
    let report = tied.load_state_dict(&sd).unwrap();
    assert_eq!(report, LoadReport::default());
    assert!(tied.c.weight.is_same(&weight) && weight.dtype() == DType::F32);
    // The last key of a shared tensor wins.
    assert_eq!(weight.to_vec::<f32>(), [3.0; 4]);

    let net = Net::new();
    let mut sd = net.state_dict();
    sd.remove("head.weight");
    sd.insert("extra", homura::zeros([1]));
    let report = net
        .load_state_dict_with(&sd, LoadOptions::default().strict(false))
        .unwrap();
    assert_eq!(report.missing_keys, ["head.weight"]);
    assert_eq!(report.unexpected_keys, ["extra"]);
    sd.insert("blocks.1.lin.bias", homura::zeros([5]));
    let mismatch = "size mismatch for blocks.1.lin.bias: copying a param with shape [5] from \
                    checkpoint, the shape in current model is [3].";
    let error = net.load_state_dict(&sd).unwrap_err();
    assert_eq!(error.missing_keys, ["head.weight"]);
    assert_eq!(error.unexpected_keys, ["extra"]);
    assert_eq!(error.errors, [mismatch]);
    assert_eq!(
        error.to_string(),
        format!(
            "Error(s) in loading state_dict:\n\tMissing key(s) in state_dict: \"head.weight\". \
             \n\tUnexpected key(s) in state_dict: \"extra\". \n\t{mismatch}"
        )
    );
    let error = net
        .load_state_dict_with(&sd, LoadOptions::default().strict(false))
        .unwrap_err();
    assert!(error.missing_keys.is_empty() && error.unexpected_keys.is_empty());
    assert_eq!(
        error.to_string(),
        format!("Error(s) in loading state_dict:\n\t{mismatch}")
    );

    // Values are converted to the target's dtype, integer buffers included.
    let stats = Stats::new();
    let sd: StateDict = [
        ("running_mean", homura::ones([3])),
        ("num_batches_tracked", homura::tensor(3.7f32)),
    ]
    .into_iter()
    .collect();
    stats.load_state_dict(&sd).unwrap();
    assert_eq!(stats.num_batches_tracked.dtype(), DType::I64);
    assert_eq!(stats.num_batches_tracked.item::<i64>(), 3);
}

#[test]
fn load_errors_leave_the_module_unchanged() {
    let net = Net::new();
    net.head.weight.set_grad(Some(homura::ones([2, 3])));
    let grad = net.head.weight.grad().unwrap();
    // A graph that saved a parameter: it still runs after a failed load, so the parameter's
    // version didn't change.
    let loss = (&*net.embed * &*net.embed).sum();
    let mut sd: StateDict = net
        .state_dict()
        .into_iter()
        .map(|(k, v)| (k, homura::full_like(&v, 7)))
        .collect();
    // The last entry comes from the meta device.
    sd.insert("table.b.bias", homura::zeros_with([1], Device::Meta));
    let error = net.load_state_dict(&sd).unwrap_err();
    assert_eq!(
        error.errors,
        [
            "While copying the parameter named \"table.b.bias\", whose dimensions in the model \
             are [1] and whose dimensions in the checkpoint are [1], an exception occurred: \
             Cannot copy out of meta tensor; no data!"
        ]
    );
    for (key, value) in &net.state_dict() {
        let expected = if key == "scale" { 1.0 } else { 0.0 };
        let values = value.to(DType::F64).to_vec::<f64>();
        assert!(values.iter().all(|&v| v == expected), "{key} changed");
    }
    assert_eq!(net.embed.dtype(), DType::F32);
    assert!(net.head.weight.grad().unwrap().is_same(&grad));
    loss.backward();

    // A target whose elements overlap can't be written.
    let stats = Stats {
        running_mean: homura::zeros([1]).expand([3]),
        ..Stats::new()
    };
    let sd = stats.state_dict();
    let error = stats.load_state_dict(&sd).unwrap_err();
    assert!(
        error.errors[0].starts_with("While copying the parameter named \"running_mean\"")
            && error.errors[0].ends_with(
                "an exception occurred: unsupported operation: more than one element of the \
                 written-to tensor may refer to a single memory location. Please deep_clone() \
                 the tensor before performing the operation."
            ),
        "{}",
        error.errors[0]
    );
}

#[test]
fn load_options_choose_strictness() {
    let options = LoadOptions::default();
    assert!(options.is_strict());
    let lax = options.strict(false);
    assert!(!lax.is_strict() && options.is_strict());
    assert_eq!(lax.strict(true), LoadOptions::default());

    let unchanged = |net: &Net| {
        net.state_dict().iter().all(|(key, value)| {
            value
                .to(DType::F64)
                .to_vec::<f64>()
                .iter()
                .all(|&v| v == f64::from(key == "scale"))
        })
    };
    let net = Net::new();
    let mut sd: StateDict = net
        .state_dict()
        .into_iter()
        .map(|(k, v)| (k, homura::full_like(&v, 7)))
        .collect();
    sd.remove("head.weight");
    sd.insert("extra", homura::zeros([1]));

    // Missing and unexpected keys alone: a strict load fails before writing anything, with the
    // default options as with `load_state_dict`; a lax one writes the other entries.
    for error in [
        net.load_state_dict(&sd).unwrap_err(),
        net.load_state_dict_with(&sd, options).unwrap_err(),
    ] {
        assert_eq!(error.missing_keys, ["head.weight"]);
        assert_eq!(error.unexpected_keys, ["extra"]);
        assert!(error.errors.is_empty());
        assert!(unchanged(&net));
    }

    // An entry that can't be loaded fails both, before writing anything.
    let mut bad = sd.clone();
    bad.insert("blocks.1.lin.bias", homura::zeros([5]));
    for lax_load in [false, true] {
        let error = if lax_load {
            net.load_state_dict_with(&bad, lax).unwrap_err()
        } else {
            net.load_state_dict(&bad).unwrap_err()
        };
        assert_eq!(error.errors.len(), 1);
        assert_eq!(error.missing_keys.is_empty(), lax_load);
        assert!(unchanged(&net));
    }

    let report = net.load_state_dict_with(&sd, lax).unwrap();
    assert_eq!(report.missing_keys, ["head.weight"]);
    assert_eq!(report.unexpected_keys, ["extra"]);
    assert_eq!(net.embed.to_vec::<f32>(), [7.0; 12]);
    assert_eq!(net.head.weight.to_vec::<f32>(), [0.0; 6]);
}

#[test]
fn to_converts_in_place_and_keeps_identity() {
    let net = Net::new();
    let embed = net.embed.clone();
    let count = net.blocks[0].stats.num_batches_tracked.clone();
    embed.set_grad(Some(homura::ones([4, 3])));
    let grad = embed.grad().unwrap();
    net.to(DType::F64);
    assert!(net.embed.is_same(&embed) && embed.dtype() == DType::F64);
    assert!(embed.is_leaf() && embed.requires_grad());
    // The gradient is replaced by a converted copy; the old gradient tensor doesn't change.
    let converted = embed.grad().unwrap();
    assert!(!converted.is_same(&grad) && grad.dtype() == DType::F32);
    assert_eq!(converted.to_vec::<f64>(), [1.0; 12]);
    assert!(!converted.requires_grad() && converted.is_leaf());
    assert_eq!(net.cache.dtype(), DType::F64);
    assert_eq!(net.scale.to_vec::<f64>(), [1.0]);
    assert!(net.blocks[0].stats.num_batches_tracked.is_same(&count));
    assert_eq!(count.dtype(), DType::I64);

    for dtype in [DType::I64, DType::Bool] {
        assert_eq!(
            panic_message(|| net.to(dtype)),
            format!("Module::to only accepts floating point dtypes, but got desired dtype={dtype}")
        );
    }
    assert_eq!(
        panic_message(|| net.to(Device::Rocm(0))),
        "Module::to can't convert parameter \"embed\": copying from cpu to rocm:0 is not \
         supported yet"
    );
    assert_eq!(embed.device(), Device::Cpu);

    // A device moves integer buffers too; the meta device has no data to move back.
    net.to(Device::Meta);
    assert_eq!(embed.device(), Device::Meta);
    assert_eq!(embed.grad().unwrap().device(), Device::Meta);
    assert_eq!((count.device(), count.dtype()), (Device::Meta, DType::I64));
    assert_eq!(
        panic_message(|| net.to(Device::Cpu)),
        "Module::to can't convert parameter \"embed\": Cannot copy out of meta tensor; no data!"
    );
    assert!(net.parameters().iter().all(|p| p.device() == Device::Meta));
}

/// Converts one parameter in place, as the only member of a module.
fn convert(p: &Parameter, options: impl Into<TensorOptions>) {
    [p.clone()].to(options);
}

fn param(values: &[f32]) -> Parameter {
    Parameter::new(homura::tensor(values.to_vec()))
}

#[test]
fn old_graphs_fail_after_to() {
    // A graph that saved the parameter.
    let w = param(&[1.0, 2.0]);
    let loss = (&*w * &*w).sum();
    convert(&w, DType::F64);
    assert_eq!(
        panic_message(|| loss.backward()),
        "the data of a leaf saved for the backward pass was replaced after it was saved (it was \
         f32 on cpu, it is f64 on cpu); run the forward pass again"
    );
    assert!(w.grad().is_none());

    // A graph that didn't, with a gradient accumulated before.
    let w = param(&[1.0, 2.0]);
    w.set_grad(Some(homura::ones([2])));
    let loss = (&*w + 1.0).sum();
    convert(&w, DType::F64);
    let grad = w.grad().unwrap();
    assert_eq!(
        panic_message(|| loss.backward()),
        "the data of a leaf was replaced after the graph was recorded (it was f32 [2] on cpu, it \
         is f64 [2] on cpu); run the forward pass again"
    );
    assert!(w.grad().unwrap().is_same(&grad));
    assert_eq!(grad.to_vec::<f64>(), [1.0, 1.0]);

    // `grad` doesn't reach the new accumulator through the old graph.
    let used = "The differentiated Tensor at index 0 appears to not have been used in the graph. \
                Set allow_unused if this is the desired behavior.";
    let options = GradOptions {
        retain_graph: Some(true),
        ..GradOptions::default()
    };
    assert_eq!(
        panic_message(|| drop(autograd::grad(&[&loss], &[&w], options.clone()))),
        used
    );
    let unused = GradOptions {
        allow_unused: true,
        ..options
    };
    assert!(autograd::grad(&[&loss], &[&w], unused)[0].is_none());

    // A new graph works and accumulates into the converted gradient.
    (&*w * 3.0).sum().backward();
    assert!(w.grad().unwrap().is_same(&grad));
    assert_eq!(grad.to_vec::<f64>(), [4.0, 4.0]);
}

#[test]
fn converting_back_or_not_at_all_keeps_some_old_graphs() {
    // Back to the recorded dtype: a graph that didn't save the parameter runs again, writing a
    // gradient that matches it; one that did still fails, as for a modification in place.
    let w = param(&[1.0, 2.0]);
    let unsaved = (&*w + 1.0).sum();
    let saved = (&*w * &*w).sum();
    convert(&w, DType::F64);
    convert(&w, DType::F32);
    unsaved.backward();
    assert_eq!(w.grad().unwrap().to_vec::<f32>(), [1.0, 1.0]);
    assert!(
        panic_message(|| saved.backward())
            .starts_with("one of the variables needed for gradient computation has been modified")
    );

    // Nothing to convert: the old graph is untouched.
    let w = param(&[1.0, 2.0]);
    let saved = (&*w * &*w).sum();
    convert(&w, (DType::F32, Device::Cpu));
    saved.backward();
    assert_eq!(w.grad().unwrap().to_vec::<f32>(), [2.0, 4.0]);
}

#[test]
fn views_and_detached_aliases_keep_the_old_data() {
    let w = param(&[1.0, 2.0]);
    let view = w.narrow(0, 0, 1);
    let no_grad_view = {
        let _guard = homura::no_grad();
        w.view([2])
    };
    let alias = w.detach();
    convert(&w, DType::F64);
    let stale = "a view whose base had its data replaced (by Module::to) after the view was \
                 created can't be used for gradient computation: it still refers to the old \
                 data; create the view again";
    assert_eq!(panic_message(|| drop(&view * 2.0)), stale);
    assert_eq!(panic_message(|| drop(&no_grad_view * 2.0)), stale);
    assert_eq!(panic_message(|| drop(view.grad_fn())), stale);
    assert_eq!(format!("{view:?}"), "tensor([1.], grad_fn=<Invalid>)");
    // Their data is the old storage, which they can still read and write without grad.
    {
        let _guard = homura::no_grad();
        alias.fill_(7.0);
        view.fill_(5.0);
    }
    assert_eq!(no_grad_view.to_vec::<f32>(), [5.0, 7.0]);
    assert_eq!(w.to_vec::<f64>(), [1.0, 2.0]);
    // A view made after the conversion is fine.
    (w.narrow(0, 0, 1) * 2.0).sum().backward();
    assert_eq!(w.grad().unwrap().to_vec::<f64>(), [2.0, 0.0]);
}

/// Two parameters and two buffers, for errors in later members.
struct Pairs {
    a: Parameter,
    b: Parameter,
    x: Tensor,
    y: Tensor,
}

module!(Pairs { a, b, x, y });

impl Pairs {
    fn new() -> Pairs {
        Pairs {
            a: param(&[1.0, 2.0, 3.0, 4.0]),
            b: param(&[1.0, 2.0, 3.0, 4.0]),
            x: homura::zeros([2]),
            y: homura::zeros([2]),
        }
    }
}

#[test]
fn to_checks_every_tensor_before_converting_any() {
    let check = |m: &Pairs, error: &str| {
        let a_grad = homura::ones([4]);
        m.a.set_grad(Some(a_grad.clone()));
        // A graph that saved `a` still runs afterwards: nothing was converted.
        let loss = (&*m.a * &*m.a).sum();
        assert_eq!(panic_message(|| m.to(DType::F64)), error);
        assert!(m.parameters().iter().all(|p| p.dtype() == DType::F32));
        assert!(m.buffers().iter().all(|t| t.dtype() == DType::F32));
        assert!(m.a.grad().unwrap().is_same(&a_grad) && a_grad.dtype() == DType::F32);
        loss.backward();
    };
    let grad_error = "Module::to can't convert parameter \"b\": its .grad has a history; clear it \
                      with set_grad(None) before converting the module";
    let m = Pairs::new();
    let x = homura::ones([4]);
    x.requires_grad_(true);
    m.b.set_grad(Some(&x * 1.0));
    check(&m, grad_error);

    let tensor_error = "Module::to can't convert buffer \"y\": only a tensor that isn't a view and \
                        has no history can be converted in place";
    let m = Pairs {
        y: homura::zeros([4]).narrow(0, 0, 2),
        ..Pairs::new()
    };
    check(&m, tensor_error);
    let m = Pairs {
        y: &x.narrow(0, 0, 2) * 2.0,
        ..Pairs::new()
    };
    check(&m, tensor_error);

    // Members that already have the target dtype are left alone, whatever their state, and
    // their old graphs stay valid.
    let m = Pairs::new();
    convert(&m.b, DType::F64);
    m.b.set_grad(Some(&x.to(DType::F64) * 1.0));
    let saved = (&*m.b * &*m.b).sum();
    m.to(DType::F64);
    assert_eq!(m.a.dtype(), DType::F64);
    assert!(m.b.grad().unwrap().grad_fn().is_some());
    m.b.set_grad(None);
    saved.backward();
}

#[test]
fn every_handle_sees_the_converted_data() {
    let w = param(&[1.0, 2.0]);
    let handle = w.clone();
    let as_tensor: Tensor = w.clone().into();
    let holder = Named(vec![("w", w)]);
    holder.to(DType::F64);
    assert_eq!(handle.dtype(), DType::F64);
    assert_eq!(as_tensor.to_vec::<f64>(), [1.0, 2.0]);
    // The parameter's accumulator was replaced: a new graph reaches another node.
    let edge = |p: &Parameter| (&**p * 1.0).grad_fn().unwrap().next_functions()[0].clone();
    let (first, _) = edge(&handle).unwrap();
    let (again, _) = edge(&handle).unwrap();
    assert!(first.is_same(&again) && first.name() == "AccumulateGrad");
    convert(&handle, DType::F32);
    let (after, _) = edge(&handle).unwrap();
    assert!(!after.is_same(&first));
}

#[test]
fn wrapped_tensors_keep_their_own_methods() {
    // `Module` is in scope: wrappers of tensors and parameters still get the tensor methods,
    // which return new tensors and leave the source alone.
    let p = param(&[1.0]);
    let boxed = Box::new(p.clone());
    let converted: Tensor = boxed.to(DType::F64);
    assert_eq!((converted.dtype(), p.dtype()), (DType::F64, DType::F32));
    let converted: Tensor = Arc::new(p.clone()).to(DType::F64);
    assert_eq!((converted.dtype(), p.dtype()), (DType::F64, DType::F32));
    let nested: Box<Arc<Parameter>> = Box::new(Arc::new(p.clone()));
    let converted: Tensor = nested.to(DType::F64);
    assert_eq!((converted.dtype(), p.dtype()), (DType::F64, DType::F32));
    let converted: Tensor = p.to(DType::F64);
    assert_eq!((converted.dtype(), p.dtype()), (DType::F64, DType::F32));

    // A tensor with a history: `to` records the copy, and the source keeps its history.
    let x = homura::ones([1]);
    x.requires_grad_(true);
    let y = &x * 2.0;
    let converted: Tensor = Arc::new(y.clone()).to(DType::F64);
    assert_eq!(converted.grad_fn().unwrap().name(), "ToCopyBackward0");
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(y.grad_fn().unwrap().name(), "MulBackward0");
    let boxed = Box::new(homura::ones([1]));
    let flagged: &Tensor = boxed.requires_grad_(true);
    assert!(flagged.requires_grad());

    // Models convert in place through every wrapper.
    let boxed = Box::new(Linear::new(1, 1, true));
    boxed.to(DType::F64);
    assert_eq!(boxed.weight.dtype(), DType::F64);
    let object: Box<dyn Module + Send + Sync> = Box::new(Linear::new(1, 1, true));
    object.to(DType::F64);
    assert_eq!(object.parameters()[0].dtype(), DType::F64);
    let shared = Arc::new(Linear::new(1, 1, true));
    shared.to(DType::F64);
    assert_eq!(shared.bias.as_ref().unwrap().dtype(), DType::F64);
}

/// Hyperparameters in every wrapper and container, and containers of modules, some empty.
// The redundant boxes are what is tested.
#[allow(clippy::vec_box, clippy::redundant_allocation, clippy::box_collection)]
struct Hyper {
    sizes: Vec<usize>,
    boxed: Vec<Box<usize>>,
    shared: Vec<Arc<usize>>,
    nested: Vec<Option<Box<Arc<usize>>>>,
    array: [Box<usize>; 2],
    map: BTreeMap<String, Arc<usize>>,
    maybe: Option<Vec<usize>>,
    outer: Box<Vec<Arc<usize>>>,
    empty: Vec<Linear>,
    objects: Vec<Box<dyn Module>>,
    synced: Vec<Arc<dyn Module + Send + Sync>>,
    layers: Vec<Arc<Linear>>,
}

module!(Hyper {
    sizes,
    boxed,
    shared,
    nested,
    array,
    map,
    maybe,
    outer,
    empty,
    objects,
    synced,
    layers
});

#[test]
fn wrapped_hyperparameters_are_not_submodules() {
    let layer = Arc::new(Linear::new(1, 1, false));
    let hyper = Hyper {
        sizes: vec![1],
        boxed: vec![Box::new(1)],
        shared: vec![Arc::new(1)],
        nested: vec![Some(Box::new(Arc::new(1))), None],
        array: [Box::new(1), Box::new(2)],
        map: [("a".to_string(), Arc::new(1))].into_iter().collect(),
        maybe: Some(vec![1]),
        outer: Box::new(vec![Arc::new(1)]),
        empty: vec![],
        objects: vec![Box::new(Relu)],
        synced: vec![],
        layers: vec![layer.clone(), layer],
    };
    assert_eq!(
        keys(hyper.named_modules()),
        [
            "",
            "empty",
            "objects",
            "objects.0",
            "synced",
            "layers",
            "layers.0"
        ]
    );
    assert_eq!(
        keys(hyper.named_children()),
        ["empty", "objects", "synced", "layers"]
    );
    assert_eq!(state_keys(&hyper), ["layers.0.weight", "layers.1.weight"]);
    assert_eq!(keys(hyper.named_parameters()), ["layers.0.weight"]);
}

#[test]
fn conversion_errors_found_late_change_nothing() {
    // The last buffer can't be converted: its F64 size exceeds `isize::MAX` bytes. The meta
    // device holds no data, so nothing large is allocated.
    let m = Pairs {
        a: Parameter::new(homura::ones_with([4], Device::Meta)),
        b: Parameter::new(homura::ones_with([4], Device::Meta)),
        x: homura::zeros_with([2], Device::Meta),
        y: homura::zeros_with([isize::MAX as usize / 2], (DType::F16, Device::Meta)),
    };
    let grad = homura::ones_with([4], Device::Meta);
    m.a.set_grad(Some(grad.clone()));
    let loss = (&*m.a * &*m.a).sum();
    assert_eq!(
        panic_message(|| m.to(DType::F64)),
        format!(
            "Module::to can't convert buffer \"y\": a tensor of shape [{}] and dtype f64 would be \
             too large",
            isize::MAX as usize / 2
        )
    );
    assert!(m.parameters().iter().all(|p| p.dtype() == DType::F32));
    assert_eq!((m.x.dtype(), m.y.dtype()), (DType::F32, DType::F16));
    assert!(m.a.grad().unwrap().is_same(&grad));
    loss.backward();
}

#[test]
fn converted_gradients_are_new_tensors() {
    // Two leaves share one gradient tensor; converting one leaves the other's alone.
    let (a, b) = (param(&[1.0, 2.0]), param(&[1.0, 2.0]));
    let g = homura::zeros([2]);
    a.set_grad(Some(g.clone()));
    b.set_grad(Some(g.clone()));
    convert(&a, DType::F64);
    assert_eq!(
        (a.grad().unwrap().dtype(), g.dtype()),
        (DType::F64, DType::F32)
    );
    assert!(b.grad().unwrap().is_same(&g));
    (&*b * 2.0).sum().backward();
    (&*a * 3.0).sum().backward();
    assert_eq!(g.to_vec::<f32>(), [2.0, 2.0]);
    assert_eq!(a.grad().unwrap().to_vec::<f64>(), [3.0, 3.0]);
    // The same with devices: `b` and its gradient stay on the CPU and can still be converted.
    let (a, b) = (param(&[1.0, 2.0]), param(&[1.0, 2.0]));
    let g = homura::zeros([2]);
    a.set_grad(Some(g.clone()));
    b.set_grad(Some(g.clone()));
    convert(&a, Device::Meta);
    assert_eq!(g.device(), Device::Cpu);
    convert(&b, DType::F64);
    assert_eq!(b.grad().unwrap().dtype(), DType::F64);

    // A gradient that is a view is converted into a contiguous copy; one that requires grad
    // gives a copy that does too.
    let w = param(&[1.0, 2.0]);
    let base = homura::tensor([[1.0f32, 5.0], [2.0, 6.0]]);
    w.set_grad(Some(base.select(1, 0)));
    convert(&w, DType::F64);
    assert_eq!(w.grad().unwrap().to_vec::<f64>(), [1.0, 2.0]);
    assert!(w.grad().unwrap().is_contiguous() && base.dtype() == DType::F32);
    let g = homura::ones_with([2], DType::F64);
    g.requires_grad_(true);
    w.set_grad(Some(g));
    convert(&w, DType::F32);
    assert!(w.grad().unwrap().requires_grad() && w.grad().unwrap().is_leaf());

    // A member whose `.grad` is another member: the gradient converted is the one before
    // anything changed.
    let m = Pairs::new();
    m.b.set_grad(Some(m.a.as_tensor().clone()));
    m.to(DType::F64);
    let grad = m.b.grad().unwrap();
    assert!(!grad.is_same(&m.a) && grad.dtype() == DType::F64);
    assert_eq!(grad.to_vec::<f64>(), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn gradients_that_no_longer_match_their_leaf_are_rejected() {
    // `p.grad` is the parameter `q` of another module, which a conversion changes.
    let p = param(&[1.0, 2.0]);
    let q = param(&[5.0, 6.0]);
    p.set_grad(Some(q.as_tensor().clone()));
    convert(&q, DType::F64);
    assert_eq!(
        panic_message(|| (&*p * 2.0).sum().backward()),
        "the .grad of a leaf no longer matches it (the gradient is f64 [2] on cpu, the leaf is \
         f32 [2] on cpu); clear it with set_grad(None)"
    );
    assert!(p.grad().unwrap().is_same(&q));
    assert_eq!(q.to_vec::<f64>(), [5.0, 6.0]);
    p.set_grad(None);
    (&*p * 2.0).sum().backward();
    assert_eq!(p.grad().unwrap().to_vec::<f32>(), [2.0, 2.0]);
}
