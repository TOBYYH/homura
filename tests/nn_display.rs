//! `Module::display` (PyTorch's `print(model)`), `extra_repr`, `type_name` and
//! `num_parameters`. tools/golden/check_m3_repr.py asserts the reference text of each test with a
//! PyTorch counterpart, in the same order, on the same modules; the tests after those cover what
//! has no counterpart: wrappers, generic type names, `dyn` modules and the checks of the tree.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};

use homura::nn::functional::Conv2dPadding;
use homura::nn::{
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout, Embedding, Flatten, Forward, Gelu, LayerNorm,
    Linear, MaxPool2d, Member, Module, ModuleDisplay, Parameter, Relu, RmsNorm, Sequential,
    Sigmoid, Silu, Softmax, Tanh, Visitor,
};
use homura::{GeluApproximate, Tensor};

fn text(m: &impl Module) -> String {
    m.display().to_string()
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

#[test]
fn layers_print_their_settings() {
    let cases: Vec<(Box<dyn Module>, &str)> = vec![
        (
            Box::new(Linear::new(8, 4)),
            "Linear(in_features=8, out_features=4, bias=True)",
        ),
        (
            Box::new(Linear::new(8, 4).bias(false)),
            "Linear(in_features=8, out_features=4, bias=False)",
        ),
        (
            Box::new(Linear::new(0, 0)),
            "Linear(in_features=0, out_features=0, bias=True)",
        ),
        (
            Box::new(Conv2d::new(3, 8, 3)),
            "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))",
        ),
        (
            Box::new(Conv2d::new(3, 8, 3).stride(2).padding(1)),
            "Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))",
        ),
        (
            Box::new(
                Conv2d::new(3, 8, [3, 2])
                    .stride([2, 1])
                    .padding([1, 0])
                    .dilation([2, 1])
                    .bias(false),
            ),
            "Conv2d(3, 8, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=(2, 1), \
             bias=False)",
        ),
        (
            Box::new(Conv2d::new(4, 8, 3).groups(2).dilation(2)),
            "Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), groups=2)",
        ),
        (
            Box::new(Conv2d::new(3, 8, 3).padding(Conv2dPadding::Same)),
            "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)",
        ),
        (
            Box::new(Conv2d::new(3, 8, 3).padding(Conv2dPadding::Valid)),
            "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=valid)",
        ),
        (
            Box::new(Conv2d::new(3, 8, 3).padding([0, 0])),
            "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))",
        ),
        (
            Box::new(BatchNorm2d::new(8)),
            "BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, bias=True, \
             track_running_stats=True)",
        ),
        (
            Box::new(
                BatchNorm2d::new(8)
                    .eps(1e-3)
                    .momentum(None)
                    .affine(false)
                    .track_running_stats(false),
            ),
            "BatchNorm2d(8, eps=0.001, momentum=None, affine=False, bias=False, \
             track_running_stats=False)",
        ),
        (
            Box::new(BatchNorm2d::new(8).momentum(0.01).bias(false)),
            "BatchNorm2d(8, eps=1e-05, momentum=0.01, affine=True, bias=False, \
             track_running_stats=True)",
        ),
        (
            Box::new(MaxPool2d::new(2)),
            "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
        ),
        (
            Box::new(
                MaxPool2d::new(3)
                    .stride(2)
                    .padding(1)
                    .dilation(2)
                    .ceil_mode(true),
            ),
            "MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=True)",
        ),
        (
            Box::new(MaxPool2d::new([3, 2]).stride([2, 1]).padding([1, 0])),
            "MaxPool2d(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=1, \
             ceil_mode=False)",
        ),
        (
            Box::new(AdaptiveAvgPool2d::new(1)),
            "AdaptiveAvgPool2d(output_size=1)",
        ),
        (
            Box::new(AdaptiveAvgPool2d::new([2, 3])),
            "AdaptiveAvgPool2d(output_size=(2, 3))",
        ),
        (
            Box::new(Flatten::default()),
            "Flatten(start_dim=1, end_dim=-1)",
        ),
        (
            Box::new(Flatten::new(0, -2)),
            "Flatten(start_dim=0, end_dim=-2)",
        ),
        (
            Box::new(Dropout::default()),
            "Dropout(p=0.5, inplace=False)",
        ),
        (Box::new(Dropout::new(0.1)), "Dropout(p=0.1, inplace=False)"),
        (Box::new(Dropout::new(0.0)), "Dropout(p=0.0, inplace=False)"),
        (Box::new(Dropout::new(1.0)), "Dropout(p=1.0, inplace=False)"),
        (
            Box::new(LayerNorm::new([4])),
            "LayerNorm((4,), eps=1e-05, elementwise_affine=True, bias=True)",
        ),
        (
            Box::new(LayerNorm::new([2, 3]).eps(1e-6).elementwise_affine(false)),
            "LayerNorm((2, 3), eps=1e-06, elementwise_affine=False, bias=False)",
        ),
        (
            Box::new(LayerNorm::new([4]).bias(false)),
            "LayerNorm((4,), eps=1e-05, elementwise_affine=True, bias=False)",
        ),
        (
            Box::new(RmsNorm::new([4])),
            "RmsNorm((4,), eps=None, elementwise_affine=True)",
        ),
        (
            Box::new(RmsNorm::new([2, 3]).eps(1e-6)),
            "RmsNorm((2, 3), eps=1e-06, elementwise_affine=True)",
        ),
        (
            Box::new(RmsNorm::new([4]).elementwise_affine(false)),
            "RmsNorm((4,), eps=None, elementwise_affine=False)",
        ),
        (Box::new(Embedding::new(10, 4)), "Embedding(10, 4)"),
        (
            Box::new(Embedding::new(10, 4).padding_idx(0)),
            "Embedding(10, 4, padding_idx=0)",
        ),
        (
            Box::new(Embedding::new(10, 4).padding_idx(-1)),
            "Embedding(10, 4, padding_idx=9)",
        ),
        (Box::new(Relu), "Relu()"),
        (Box::new(Gelu::new()), "Gelu(approximate='none')"),
        (
            Box::new(Gelu::new().approximate(GeluApproximate::Tanh)),
            "Gelu(approximate='tanh')",
        ),
        (Box::new(Silu), "Silu()"),
        (Box::new(Tanh), "Tanh()"),
        (Box::new(Sigmoid), "Sigmoid()"),
        (Box::new(Softmax::new(-1)), "Softmax(dim=-1)"),
        (Box::new(Softmax::new(1)), "Softmax(dim=1)"),
        (Box::new(Sequential::new()), "Sequential()"),
        (Box::new(Vec::<Linear>::new()), "ModuleList()"),
        (Box::new(BTreeMap::<String, Linear>::new()), "ModuleDict()"),
    ];
    for (module, expected) in cases {
        assert_eq!(text(&module), expected);
    }
}

#[test]
// The ties are written exactly: rounded, they would read back as the same value.
#[allow(clippy::excessive_precision)]
fn floats_are_printed_as_python_prints_them() {
    assert_eq!(
        text(&BatchNorm2d::new(3).eps(1e-20).momentum(1e16)),
        "BatchNorm2d(3, eps=1e-20, momentum=1e+16, affine=True, bias=True, \
         track_running_stats=True)"
    );
    assert_eq!(
        text(&BatchNorm2d::new(3).eps(123456.0).momentum(1.0 / 3.0)),
        "BatchNorm2d(3, eps=123456.0, momentum=0.3333333333333333, affine=True, bias=True, \
         track_running_stats=True)"
    );
    assert_eq!(text(&Dropout::new(0.25)), "Dropout(p=0.25, inplace=False)");
    // Two shortest decimals equally close: Python's even last digit.
    let tie = 1_000_000_000_000_000.25;
    assert_eq!(
        text(&LayerNorm::new([1]).eps(tie)),
        "LayerNorm((1,), eps=1000000000000000.2, elementwise_affine=True, bias=True)"
    );
    assert_eq!(
        text(&RmsNorm::new([1]).eps(-tie)),
        "RmsNorm((1,), eps=-1000000000000000.2, elementwise_affine=True)"
    );
    assert_eq!(
        text(
            &BatchNorm2d::new(1)
                .eps(2f64.powi(-25))
                .momentum(1_000_000_000_000_000.75)
        ),
        "BatchNorm2d(1, eps=2.9802322387695312e-08, momentum=1000000000000000.8, affine=True, \
         bias=True, track_running_stats=True)"
    );
    assert_eq!(
        text(&LayerNorm::new([4]).eps(1.5e-4)),
        "LayerNorm((4,), eps=0.00015, elementwise_affine=True, bias=True)"
    );
}

/// PyTorch prints the sizes it was given (`(2, 2)` or `2`); homura keeps pairs and prints equal
/// ones as one number.
#[test]
fn pooling_sizes_given_as_equal_pairs() {
    assert_eq!(
        text(&MaxPool2d::new([2, 2])),
        "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)"
    );
    assert_eq!(
        text(&AdaptiveAvgPool2d::new([1, 1])),
        "AdaptiveAvgPool2d(output_size=1)"
    );
}

#[test]
fn containers_list_their_children() {
    let seq = Sequential::new()
        .append(Linear::new(2, 3))
        .append(Relu)
        .append(Sequential::new().append(Tanh));
    assert_eq!(
        text(&seq),
        "Sequential(\n  (0): Linear(in_features=2, out_features=3, bias=True)\n  (1): Relu()\n  \
         (2): Sequential(\n    (0): Tanh()\n  )\n)"
    );
    let mut dict: BTreeMap<String, Box<dyn Module>> = BTreeMap::new();
    dict.insert("b".into(), Box::new(Relu));
    dict.insert("a".into(), Box::new(Linear::new(1, 1)));
    assert_eq!(
        text(&dict),
        "ModuleDict(\n  (a): Linear(in_features=1, out_features=1, bias=True)\n  (b): Relu()\n)"
    );
    assert_eq!(
        text(&Sequential::new().append(Relu).append(Relu)),
        "Sequential(\n  (0): Relu()\n  (1): Relu()\n)"
    );
    let list: Vec<Box<dyn Module>> = vec![
        Box::new(Linear::new(2, 2)),
        Box::new(Linear::new(2, 2)),
        Box::new(Linear::new(2, 2)),
        Box::new(Relu),
        Box::new(Relu),
        Box::new(Linear::new(2, 2)),
    ];
    assert_eq!(
        text(&list),
        "ModuleList(\n  (0-2): 3 x Linear(in_features=2, out_features=2, bias=True)\n  (3-4): 2 x \
         Relu()\n  (5): Linear(in_features=2, out_features=2, bias=True)\n)"
    );
    // Arrays are `ModuleList`s too.
    let nested = [
        Sequential::new().append(Relu),
        Sequential::new().append(Relu),
    ];
    assert_eq!(
        text(&nested),
        "ModuleList(\n  (0-1): 2 x Sequential(\n    (0): Relu()\n  )\n)"
    );
    assert_eq!(text(&vec![Relu]), "ModuleList(\n  (0): Relu()\n)");
    let alternating = vec![Linear::new(1, 1), Linear::new(1, 2), Linear::new(1, 1)];
    assert_eq!(
        text(&alternating),
        "ModuleList(\n  (0): Linear(in_features=1, out_features=1, bias=True)\n  (1): \
         Linear(in_features=1, out_features=2, bias=True)\n  (2): Linear(in_features=1, \
         out_features=1, bias=True)\n)"
    );
}

/// A module with an `extra_repr` and a child.
#[derive(Module)]
#[module(extra_repr = Self::describe)]
struct Multi<C> {
    text: String,
    inner: C,
}

impl<C> Multi<C> {
    fn describe(&self) -> String {
        self.text.clone()
    }
}

#[derive(Module)]
#[module(extra_repr = describe_leaf)]
struct Leaf {
    text: String,
}

fn describe_leaf(leaf: &Leaf) -> String {
    leaf.text.clone()
}

/// The identity, so that `Leaf` and `Multi` can be layers of a `Sequential`.
impl Forward for Leaf {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }
}

impl<C: Member> Forward for Multi<C> {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }
}

fn leaf(text: &str) -> Leaf {
    Leaf { text: text.into() }
}

#[test]
fn extra_repr_lines_are_indented() {
    assert_eq!(text(&leaf("x")), "Leaf(x)");
    assert_eq!(text(&leaf("")), "Leaf()");
    assert_eq!(text(&leaf("x\ny")), "Leaf(\n  x\n  y\n)");
    let multi = Multi {
        text: "x".into(),
        inner: Relu,
    };
    assert_eq!(text(&multi), "Multi(\n  x\n  (inner): Relu()\n)");
    // Blank lines: at the top they keep the two spaces of the join; nested, they become empty.
    let multi = Multi {
        text: "a\n\nb\n".into(),
        inner: Sequential::new().append(Leaf {
            text: "x\ny".into(),
        }),
    };
    assert_eq!(
        text(&multi),
        "Multi(\n  a\n  \n  b\n  \n  (inner): Sequential(\n    (0): Leaf(\n      x\n      y\n    \
         )\n  )\n)"
    );
    assert_eq!(
        text(&Sequential::new().append(multi)),
        "Sequential(\n  (0): Multi(\n    a\n\n    b\n\n    (inner): Sequential(\n      (0): \
         Leaf(\n        x\n        y\n      )\n    )\n  )\n)"
    );
    assert_eq!(
        text(&vec![leaf("x\ny"), leaf("x\ny")]),
        "ModuleList(\n  (0-1): 2 x Leaf(\n    x\n    y\n  )\n)"
    );
}

/// A shared `Linear` under two names, a parameter (not printed) and nested containers.
#[derive(Module)]
struct Net {
    p: Parameter,
    a: Arc<Linear>,
    b: Arc<Linear>,
    empty: Vec<Linear>,
    blocks: Vec<Box<dyn Module + Send + Sync>>,
}

fn net() -> Net {
    let shared = Arc::new(Linear::new(2, 2));
    Net {
        p: Parameter::new(homura::zeros([2])),
        a: shared.clone(),
        b: shared,
        empty: vec![],
        blocks: vec![
            Box::new(Relu),
            Box::new(Sequential::new().append(Linear::new(2, 2))),
        ],
    }
}

const NET: &str = "\
Net(
  (a): Linear(in_features=2, out_features=2, bias=True)
  (b): Linear(in_features=2, out_features=2, bias=True)
  (empty): ModuleList()
  (blocks): ModuleList(
    (0): Relu()
    (1): Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
    )
  )
)";

#[test]
fn shared_modules_appear_under_each_name() {
    let net = net();
    assert_eq!(text(&net), NET);
    let children: Vec<String> = net.named_children().into_iter().map(|(k, _)| k).collect();
    assert_eq!(children, ["a", "empty", "blocks"]);
}

#[test]
fn parameters_are_counted_once_by_elements() {
    assert_eq!(Linear::new(2, 3).num_parameters(), 9);
    assert_eq!(Linear::new(2, 3).bias(false).num_parameters(), 6);
    // Buffers don't count.
    assert_eq!(BatchNorm2d::new(8).num_parameters(), 16);
    assert_eq!(BatchNorm2d::new(8).affine(false).num_parameters(), 0);
    // A shared module once, frozen parameters too.
    assert_eq!(net().num_parameters(), 2 + 6 + 6);
    let frozen = Linear::new(2, 3);
    frozen.weight.requires_grad_(false);
    assert_eq!(frozen.num_parameters(), 9);
    assert_eq!(Sequential::new().num_parameters(), 0);
}

/// Two parameters: distinct ones that share a storage, or the same one twice.
#[derive(Module)]
struct Pair {
    x: Parameter,
    y: Parameter,
    count: Tensor,
}

#[test]
fn parameters_are_counted_by_identity() {
    let base = homura::zeros([3]);
    let distinct = Pair {
        x: Parameter::new(base.detach()),
        y: Parameter::new(base.detach()),
        count: homura::zeros([5]),
    };
    assert_eq!(distinct.num_parameters(), 6);
    let p = Parameter::new(base);
    let same = Pair {
        x: p.clone(),
        y: p,
        count: homura::zeros([5]),
    };
    assert_eq!(same.num_parameters(), 3);
}

#[test]
fn wrapper_roots_print_the_module_they_hold() {
    let linear = "Linear(in_features=1, out_features=2, bias=True)";
    assert_eq!(text(&Box::new(Linear::new(1, 2))), linear);
    assert_eq!(text(&Arc::new(Linear::new(1, 2))), linear);
    assert_eq!(text(&Some(Linear::new(1, 2))), linear);
    assert_eq!(text(&Box::new(Arc::new(Some(Linear::new(1, 2))))), linear);
    let boxed: Box<dyn Module> = Box::new(Linear::new(1, 2));
    assert_eq!(text(&boxed), linear);
    let shared: Arc<dyn Module + Send + Sync> = Arc::new(Linear::new(1, 2));
    assert_eq!(text(&shared), linear);
    assert_eq!(text(&Some(boxed)), linear);
    assert_eq!(text(&Arc::new(net())), NET);
    // Without a module: the name `None`, no settings, no members.
    let none: Option<Linear> = None;
    assert_eq!(
        (none.type_name(), none.extra_repr()),
        ("None", String::new())
    );
    assert_eq!(text(&none), "None()");
    assert_eq!(text(&Some(none)), "None()");
    assert_eq!(text(&None::<Box<dyn Module>>), "None()");
    // The metadata through `dyn`, as `display` reads it.
    let m: &dyn Module = &Box::new(Linear::new(1, 2));
    assert_eq!(m.type_name(), "Linear");
    assert_eq!(m.extra_repr(), "in_features=1, out_features=2, bias=True");
}

#[test]
fn containers_of_plain_values_or_parameters_have_no_children() {
    assert_eq!(text(&vec![1usize, 2]), "ModuleList()");
    assert_eq!(text(&[0.5f64; 2]), "ModuleList()");
    let params = vec![Parameter::new(homura::zeros([2]))];
    assert_eq!(text(&params), "ModuleList()");
    assert_eq!(params.num_parameters(), 2);
    assert_eq!(text(&[Relu; 0]), "ModuleList()");
    assert_eq!(Sequential::new().type_name(), "Sequential");
}

mod outer {
    use homura::nn::Module;

    /// A generic module whose type name has paths and nested generic arguments.
    #[derive(Module)]
    pub struct Block<M> {
        pub inner: M,
    }
}

#[test]
fn type_names_drop_paths_and_generic_arguments() {
    let block = outer::Block {
        inner: outer::Block {
            inner: vec![Linear::new(1, 1)],
        },
    };
    assert_eq!(block.type_name(), "Block");
    let m: &dyn Module = &block;
    assert_eq!(m.type_name(), "Block");
    assert_eq!(
        text(&block),
        "Block(\n  (inner): Block(\n    (inner): ModuleList(\n      (0): Linear(in_features=1, \
         out_features=1, bias=True)\n    )\n  )\n)"
    );
    assert_eq!(
        Multi {
            text: String::new(),
            inner: Arc::new(Relu)
        }
        .type_name(),
        "Multi"
    );
}

/// A module that names itself and describes itself by hand.
struct Custom;

impl Module for Custom {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        Relu.visit_as("act", v);
    }

    fn extra_repr(&self) -> String {
        "custom=1".into()
    }

    fn type_name(&self) -> &'static str {
        "MyLayer"
    }
}

impl Member for Custom {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}

#[test]
fn hand_written_modules_choose_their_name_and_settings() {
    assert_eq!(text(&Custom), "MyLayer(\n  custom=1\n  (act): Relu()\n)");
    assert_eq!(
        text(&vec![Custom, Custom]),
        "ModuleList(\n  (0-1): 2 x MyLayer(\n    custom=1\n    (act): Relu()\n  )\n)"
    );
}

/// `display` checks the tree when it is called, and reads the settings when the text is written.
struct Counter(Cell<u32>);

impl Module for Counter {
    fn visit<'a>(&'a self, _: &mut dyn Visitor<'a>) {}

    fn extra_repr(&self) -> String {
        format!("n={}", self.0.get())
    }
}

impl Member for Counter {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}

#[test]
fn the_text_is_written_when_formatted() {
    let counter = Counter(Cell::new(1));
    let display = counter.display();
    counter.0.set(2);
    assert_eq!(display.to_string(), "Counter(n=2)");
    assert_eq!(format!("{display:?}"), "Counter(n=2)");
    assert_eq!(format!("{display}"), "Counter(n=2)");
}

#[test]
fn submodules_listed_as_dyn_modules_can_be_printed() {
    let net = net();
    let modules = net.named_modules();
    let texts: Vec<(String, String)> = modules
        .into_iter()
        .map(|(key, m)| (key, ModuleDisplay::new(m).to_string()))
        .collect();
    assert_eq!(texts[0], (String::new(), NET.to_string()));
    assert_eq!(
        texts[1],
        (
            "a".into(),
            "Linear(in_features=2, out_features=2, bias=True)".into()
        )
    );
    let blocks = &texts.iter().find(|(k, _)| k == "blocks.1").unwrap().1;
    assert_eq!(
        blocks,
        "Sequential(\n  (0): Linear(in_features=2, out_features=2, bias=True)\n)"
    );
}

#[test]
fn display_checks_the_tree_when_called() {
    let a = Arc::new(CyclicByHand {
        next: OnceLock::new(),
    });
    let b = Arc::new(CyclicByHand {
        next: OnceLock::new(),
    });
    a.next.set(b.clone()).ok().unwrap();
    assert_eq!(text(&a), "CyclicByHand(\n  (next): CyclicByHand()\n)");
    b.next.set(a.clone()).ok().unwrap();
    let cycle = "the module tree has a cycle: module \"next.next\" contains itself";
    assert_eq!(panic_message(|| drop(a.display())), cycle);
    // The same names as the traversals.
    let bad = Named(vec![("", Relu)]);
    assert_eq!(
        panic_message(|| drop(bad.display())),
        "module name can't be empty string \"\""
    );
    let bad = Named(vec![("a.b", Relu)]);
    assert_eq!(
        panic_message(|| drop(bad.display())),
        "module name can't contain \".\", got: a.b"
    );
    let bad = Named(vec![("x", Relu), ("x", Relu)]);
    assert_eq!(
        panic_message(|| drop(bad.display())),
        "attribute 'x' already exists"
    );
    // The cycle of `Arc`s is leaked.
}

/// A module that can be made to contain itself through an `Arc`.
struct CyclicByHand {
    next: OnceLock<Arc<CyclicByHand>>,
}

impl Module for CyclicByHand {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        if let Some(next) = self.next.get() {
            next.visit_as("next", v);
        }
    }
}

impl Member for CyclicByHand {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
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

impl<T: Member> Member for Named<T> {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}
