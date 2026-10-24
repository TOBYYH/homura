//! `#[derive(Module)]` and `nn::Sequential`, used from outside the crate.
//! tools/golden/check_m2_derive.py asserts the reference behavior of each test with a PyTorch
//! counterpart, in the same order, on the same module trees; the traversals of `Net` are those
//! of the hand-written `Net` of tests/nn_module.rs (and tools/golden/check_m2_module.py).
// The unconventional generic names below are allowed at their structs only: the derived
// implementations must not report them again.
#![deny(non_upper_case_globals, non_camel_case_types)]

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use homura::nn::{Forward, Module, Parameter, Sequential, StateDict, Training};
use homura::{DType, Tensor};

fn keys<T>(pairs: Vec<(String, T)>) -> Vec<String> {
    pairs.into_iter().map(|(k, _)| k).collect()
}

fn state_keys(m: &dyn Module) -> Vec<String> {
    m.state_dict().keys().map(String::from).collect()
}

#[derive(Module)]
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

    /// `x` times the transposed weight, plus the bias.
    fn forward(&self, x: &Tensor) -> Tensor {
        let y = x.matmul(&self.weight.transpose(0, 1));
        match &self.bias {
            Some(bias) => y + &**bias,
            None => y,
        }
    }
}

impl Forward for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        Linear::forward(self, x)
    }
}

#[derive(Module)]
struct Relu;

impl Forward for Relu {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.relu()
    }
}

#[derive(Module)]
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

#[derive(Module)]
struct Block {
    lin: Linear,
    stats: Stats,
}

/// The derived twin of the hand-written `Net` of tests/nn_module.rs, whose `visit` reports the
/// fields in this order and `cache` as a non-persistent buffer.
#[derive(Module)]
struct Net {
    embed: Parameter,
    blocks: Vec<Block>,
    relu: Relu,
    head: Linear,
    scale: Tensor,
    #[module(non_persistent)]
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
        let block = || Block {
            lin: Linear::new(3, 3, true),
            stats: Stats::new(),
        };
        Net {
            embed: Parameter::new(homura::zeros([4, 3])),
            blocks: vec![block(), block()],
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

#[test]
fn derived_modules_list_their_members_like_hand_written_ones() {
    let net = Net::new();
    assert_eq!(
        state_keys(&net),
        [
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
        ]
    );
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
    assert_eq!(
        keys(net.named_modules()),
        [
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
        ]
    );
    assert_eq!(
        keys(net.named_children()),
        ["blocks", "relu", "head", "table", "empty"]
    );
    assert!(net.parameters()[0].is_same(&net.embed));
    assert!(net.buffers()[1].is_same(&net.cache));
    // Train / eval reach the flags of the root and of every block.
    net.blocks[1].stats.training.set(false);
    net.train();
    assert!(net.training.get() && net.blocks[1].stats.training.get());
    net.blocks[0].eval();
    assert!(net.training.get() && !net.blocks[0].stats.training.get());
}

/// Attributes: a renamed parameter, non-persistent buffers, skipped fields.
#[derive(Module)]
struct Norm {
    #[module(rename = "weight")]
    scale: Parameter,
    #[module(non_persistent)]
    cache: Option<Tensor>,
    #[module(non_persistent, rename = "mask")]
    m: Option<Tensor>,
    #[module(skip)]
    log: Rc<Vec<String>>,
    #[module(skip)]
    scratch: Tensor,
    eps: f64,
}

fn norm() -> Norm {
    Norm {
        scale: Parameter::new(homura::ones([4])),
        cache: Some(homura::zeros([4])),
        m: None,
        log: Rc::default(),
        scratch: homura::zeros([4]),
        eps: 1e-5,
    }
}

#[test]
fn attributes_rename_skip_and_keep_buffers_out_of_the_state_dict() {
    let norm = norm();
    assert_eq!(state_keys(&norm), ["weight"]);
    assert_eq!(keys(norm.named_parameters()), ["weight"]);
    // `mask` is `None`.
    assert_eq!(keys(norm.named_buffers()), ["cache"]);
    assert_eq!(keys(norm.named_modules()), [""]);

    // Non-persistent buffers are converted; skipped tensors are not members, so not converted.
    norm.to(DType::F64);
    assert_eq!(norm.scale.dtype(), DType::F64);
    assert_eq!(norm.cache.as_ref().unwrap().dtype(), DType::F64);
    assert_eq!(norm.scratch.dtype(), DType::F32);
    assert!(norm.log.is_empty() && norm.eps == 1e-5);

    let mut sd = StateDict::new();
    sd.insert("weight", homura::full([4], 2.0));
    norm.load_state_dict(&sd).unwrap();
    assert_eq!(norm.scale.to_vec::<f64>(), [2.0; 4]);
    sd.insert("cache", homura::zeros([4]));
    let error = norm.load_state_dict(&sd).unwrap_err();
    assert_eq!(error.unexpected_keys, ["cache"]);

    let with_mask = Norm {
        m: Some(homura::ones([1])),
        ..crate::norm()
    };
    assert_eq!(keys(with_mask.named_buffers()), ["cache", "mask"]);
    assert_eq!(state_keys(&with_mask), ["weight"]);
}

/// Fields whose types mention a type parameter.
#[derive(Module)]
struct Wrap<M> {
    inner: M,
    extra: Option<Box<M>>,
    scale: f32,
}

#[derive(Module)]
struct Stack<const N: usize> {
    layers: [Linear; N],
}

/// A type parameter only in fields that need no bound on it: `PhantomData<T>` is a member for
/// every `T`, and skipped fields are not members.
#[derive(Module)]
struct Marked<T, U>
where
    U: Clone,
{
    weight: Parameter,
    marker: PhantomData<T>,
    #[module(skip)]
    extra: U,
}

/// A recursive generic module: the bound of `children` would need itself, so it has none.
#[derive(Module)]
struct Tree<T> {
    value: T,
    children: Vec<Tree<T>>,
}

/// A lifetime parameter: borrowed hyperparameters and boxed modules that borrow.
#[derive(Module)]
struct Borrowed<'m> {
    label: &'m str,
    inner: Box<dyn Module + 'm>,
}

#[test]
fn generic_structs_are_modules() {
    let wrap = Wrap {
        inner: Linear::new(2, 1, true),
        extra: None,
        scale: 1.0,
    };
    assert_eq!(state_keys(&wrap), ["inner.weight", "inner.bias"]);
    let wrap = Wrap {
        inner: vec![Linear::new(2, 1, false)],
        extra: Some(Box::new(vec![])),
        scale: 1.0,
    };
    assert_eq!(state_keys(&wrap), ["inner.0.weight"]);
    assert_eq!(
        keys(wrap.named_modules()),
        ["", "inner", "inner.0", "extra"]
    );
    // A plain value in place of a module is a hyperparameter.
    let plain = Wrap {
        inner: 3usize,
        extra: None,
        scale: 1.0,
    };
    assert_eq!(keys(plain.named_modules()), [""]);

    let stack = Stack {
        layers: [Linear::new(1, 1, false), Linear::new(1, 1, false)],
    };
    assert_eq!(state_keys(&stack), ["layers.0.weight", "layers.1.weight"]);

    let leaf = |v: f32| Tree {
        value: Parameter::new(homura::tensor([v])),
        children: vec![],
    };
    let tree = Tree {
        value: Parameter::new(homura::zeros([1])),
        children: vec![leaf(1.0), leaf(2.0)],
    };
    assert_eq!(
        state_keys(&tree),
        ["value", "children.0.value", "children.1.value"]
    );

    // `Rc` implements neither `Member` nor `Send`.
    let marked = Marked::<Rc<u8>, Rc<u8>> {
        weight: Parameter::new(homura::zeros([1])),
        marker: PhantomData,
        extra: Rc::new(1),
    };
    assert_eq!(state_keys(&marked), ["weight"]);
    assert_eq!(*marked.extra, 1);

    let label = String::from("borrowed");
    let borrowed = Borrowed {
        label: &label,
        inner: Box::new(Wrap {
            inner: label.as_str(),
            extra: None,
            scale: 1.0,
        }),
    };
    assert_eq!(keys(borrowed.named_modules()), ["", "inner"]);
    assert_eq!(borrowed.label, "borrowed");
}

impl Forward for Wrap<Linear> {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.inner.forward(x) * f64::from(self.scale)
    }
}

/// Generic parameters with the names of the generated method's lifetime and parameters.
#[derive(Module)]
struct Labeled<'__a, '__a1> {
    label: &'__a str,
    other: &'__a1 str,
}

#[allow(non_upper_case_globals)]
#[derive(Module)]
struct Sized<const __visitor: usize, const __name: usize> {
    layers: [Linear; __visitor],
    sizes: [usize; __name],
}

/// A submodule of another type with the same name, which is not the struct itself.
mod same_name {
    pub mod child {
        #[derive(homura::nn::Module)]
        pub struct Wrap<T> {
            pub value: T,
        }
    }

    #[derive(homura::nn::Module)]
    pub struct Wrap<T> {
        pub inner: child::Wrap<T>,
    }
}

/// A raw type parameter, written without the prefix in the field.
#[derive(Module)]
struct Raw<r#T> {
    inner: T,
}

#[allow(non_camel_case_types)]
#[derive(Module)]
struct Lower<m> {
    inner: m,
}

trait Value {
    type Item;
}

/// A projection out of the struct, which depends on its parameters.
#[derive(Module)]
struct Projected<T: Value> {
    inner: <Self as Value>::Item,
}

impl<T: Value> Value for Projected<T> {
    type Item = T;
}

/// Only for the bound of `Projected`.
impl Value for Linear {
    type Item = ();
}

macro_rules! boxed {
    ($t:ty) => {
        Box<$t>
    };
}

type Layers<T> = Vec<T>;

/// Field types through a macro and an alias, and a generic non-persistent buffer.
#[derive(Module)]
struct Indirect<T, B> {
    first: boxed!(T),
    rest: Layers<T>,
    #[module(non_persistent)]
    cache: B,
}

/// A struct generated by `macro_rules`.
macro_rules! module {
    ($name:ident { $($field:ident: $ty:ty),* }) => {
        #[derive(Module)]
        struct $name<M> {
            $($field: $ty),*
        }
    };
}

module!(Generated {
    head: M,
    scale: Parameter
});

#[test]
fn generic_names_and_types_the_expansion_must_not_confuse() {
    let label = String::from("label");
    let labeled = Labeled {
        label: &label,
        other: &label,
    };
    assert_eq!(keys(labeled.named_modules()), [""]);
    assert_eq!(labeled.label, labeled.other);

    let sized = Sized {
        layers: [Linear::new(1, 1, false), Linear::new(1, 1, false)],
        sizes: [1, 2, 3],
    };
    assert_eq!(state_keys(&sized), ["layers.0.weight", "layers.1.weight"]);
    assert_eq!(sized.sizes.len(), 3);

    let same = same_name::Wrap {
        inner: same_name::child::Wrap {
            value: Linear::new(1, 1, false),
        },
    };
    assert_eq!(state_keys(&same), ["inner.value.weight"]);
    assert_eq!(keys(same.named_modules()), ["", "inner", "inner.value"]);

    let raw = Raw {
        inner: Linear::new(1, 1, true),
    };
    assert_eq!(state_keys(&raw), ["inner.weight", "inner.bias"]);

    let lower = Lower {
        inner: Linear::new(1, 1, false),
    };
    assert_eq!(state_keys(&lower), ["inner.weight"]);

    let projected = Projected::<Linear> {
        inner: Linear::new(1, 1, false),
    };
    assert_eq!(state_keys(&projected), ["inner.weight"]);

    let indirect = Indirect {
        first: Box::new(Linear::new(1, 1, false)),
        rest: vec![Linear::new(1, 1, false)],
        cache: Some(homura::zeros([1])),
    };
    assert_eq!(state_keys(&indirect), ["first.weight", "rest.0.weight"]);
    assert_eq!(keys(indirect.named_buffers()), ["cache"]);
    let indirect = Indirect {
        first: Box::new(1u8),
        rest: vec![],
        cache: homura::zeros([1]),
    };
    assert_eq!(keys(indirect.named_buffers()), ["cache"]);

    let generated = Generated {
        head: Linear::new(1, 1, false),
        scale: Parameter::new(homura::ones([1])),
    };
    assert_eq!(state_keys(&generated), ["scale", "head.weight"]);
}

#[derive(Module)]
struct Tied {
    a: Arc<Linear>,
    b: Arc<Linear>,
}

#[derive(Module)]
struct Identity {}

#[derive(Module)]
struct Keywords {
    r#for: Parameter,
    r#in: Identity,
}

#[test]
fn shared_unit_and_raw_named_members() {
    let a = Arc::new(Linear::new(2, 2, false));
    let tied = Tied { b: a.clone(), a };
    assert_eq!(state_keys(&tied), ["a.weight", "b.weight"]);
    assert_eq!(keys(tied.named_parameters()), ["a.weight"]);
    assert_eq!(keys(tied.named_modules()), ["", "a"]);

    let keywords = Keywords {
        r#for: Parameter::new(homura::zeros([1])),
        r#in: Identity {},
    };
    assert_eq!(state_keys(&keywords), ["for"]);
    assert_eq!(keys(keywords.named_modules()), ["", "in"]);
    assert_eq!(keys(Relu.named_modules()), [""]);
}

/// The crate's path, for a renamed dependency.
mod renamed {
    use homura as hm;

    #[derive(hm::nn::Module)]
    #[module(crate = "hm")]
    pub struct Scaled<M> {
        pub inner: M,
        pub scale: hm::nn::Parameter,
    }
}

#[test]
fn the_crate_path_can_be_given() {
    let scaled = renamed::Scaled {
        inner: Linear::new(1, 1, false),
        scale: Parameter::new(homura::ones([1])),
    };
    assert_eq!(state_keys(&scaled), ["scale", "inner.weight"]);
}

fn linear(weight: &[f32], outputs: usize) -> Linear {
    Linear {
        weight: Parameter::new(homura::tensor(weight).reshape([outputs as i64, -1])),
        bias: Some(Parameter::new(homura::ones([outputs]))),
    }
}

#[test]
fn sequential_names_layers_by_position() {
    let model = Sequential::new()
        .append(Linear::new(3, 2, true))
        .append(Relu)
        .append(Sequential::new().append(Linear::new(2, 1, false)));
    assert_eq!(state_keys(&model), ["0.weight", "0.bias", "2.0.weight"]);
    assert_eq!(keys(model.named_modules()), ["", "0", "1", "2", "2.0"]);
    assert_eq!(keys(model.named_children()), ["0", "1", "2"]);
    assert_eq!(model.len(), 3);
    assert!(!model.is_empty() && model.get(3).is_none());
    assert_eq!(state_keys(model.get(2).unwrap()), ["0.weight"]);
    assert_eq!(model.iter().len(), 3);
    assert_eq!(format!("{model:?}"), "Sequential { len: 3, .. }");

    // As a field of a derived module.
    #[derive(Module)]
    struct Classifier {
        features: Sequential,
        head: Linear,
    }
    let mut features = Sequential::new();
    features.push(Linear::new(2, 2, false));
    features.push(Relu);
    let classifier = Classifier {
        features,
        head: Linear::new(2, 1, false),
    };
    assert_eq!(
        state_keys(&classifier),
        ["features.0.weight", "head.weight"]
    );
    assert_eq!(
        keys(classifier.named_modules()),
        ["", "features", "features.0", "features.1", "head"]
    );
}

#[test]
fn sequential_applies_its_layers_in_order() {
    let model = Sequential::new()
        .append(linear(&[1.0, -3.0, 3.0, 4.0], 2))
        .append(Relu)
        .append(Wrap {
            inner: linear(&[1.0, 1.0], 1),
            extra: None,
            scale: 0.5,
        });
    let x = homura::tensor([[1.0f32, 1.0], [2.0, 0.0]]);
    // [[-1, 8], [3, 7]] after the first layer, [[0, 8], [3, 7]] after the relu, then the sum
    // plus 1, times 0.5.
    let y = model.forward(&x);
    assert_eq!(y.to_vec::<f32>(), [4.5, 5.5]);
    // Through the trait, as a layer of another `Sequential`.
    let outer = Sequential::new().append(model);
    assert_eq!(Forward::forward(&outer, &x).to_vec::<f32>(), [4.5, 5.5]);

    y.sum().backward();
    let first = outer.parameters()[0].clone();
    assert_eq!(first.grad().unwrap().to_vec::<f32>(), [1.0, 0.0, 1.5, 0.5]);

    // Without layers, the input itself.
    let empty = Sequential::new();
    assert!(empty.forward(&x).is_same(&x));
    assert!(empty.is_empty() && state_keys(&empty).is_empty());
}

#[test]
fn sequential_layers_can_be_shared_and_converted() {
    let shared = Arc::new(linear(&[1.0, 0.0, 0.0, 1.0], 2));
    let model = Sequential::new()
        .append(shared.clone())
        .append(Relu)
        .append(shared.clone());
    assert_eq!(
        state_keys(&model),
        ["0.weight", "0.bias", "2.weight", "2.bias"]
    );
    assert_eq!(keys(model.named_parameters()), ["0.weight", "0.bias"]);
    assert_eq!(keys(model.named_modules()), ["", "0", "1"]);
    let x = homura::tensor([[1.0f32, -3.0]]);
    assert_eq!(model.forward(&x).to_vec::<f32>(), [3.0, 1.0]);

    // In place, so the shared layer sees it.
    model.to(DType::F64);
    assert_eq!(shared.weight.dtype(), DType::F64);
    let boxed: Box<dyn Module> = Box::new(Sequential::new().append(Linear::new(1, 1, true)));
    boxed.to(DType::F64);
    assert!(boxed.parameters().iter().all(|p| p.dtype() == DType::F64));
}
