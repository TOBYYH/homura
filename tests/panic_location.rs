//! Panics for invalid arguments report the caller's line (`#[track_caller]`), through every kind
//! of entry point: tensor methods and operators, `Sum` / `Product` called directly, free
//! functions, `nn` functions, layers and module methods (directly and through `Forward` /
//! `Sequential`), optimizers (concrete and `dyn`), autograd entry points and `Function::apply`,
//! user code in the scope closures (`with_no_grad`, `with_device`), and through the internal
//! checks those entry points share (argument checks, CPU kernels that validate indices or dtypes,
//! the stale-view check of the recording).

use std::cell::RefCell;
use std::iter::{Product, Sum};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

use homura::autograd::{
    self, BackwardContext, BackwardOptions, ForwardContext, Function, GradOptions,
};
use homura::nn::functional::{
    self as F, BatchNormOptions, Conv2dOptions, Conv2dPadding, CrossEntropyOptions, Reduction,
};
use homura::nn::{self, Forward, LoadOptions, Member, Module, Parameter, StateDict, Visitor};
use homura::optim::{Adam, Optimizer, ParamGroup, Sgd, SgdOptions, TypedOptimizer};
use homura::{DType, Device, Tensor};

thread_local! {
    /// While `panic_location` runs: the location of the last panic on this thread.
    static CAUGHT: RefCell<Option<Option<(String, u32)>>> = const { RefCell::new(None) };
}

/// Installs, once for the whole process, a panic hook that records the location in the
/// panicking thread's slot while `panic_location` runs on that thread, and otherwise reports the
/// panic as usual. Tests run in parallel threads, so they must not replace the hook themselves.
fn install_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let default = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let location = info.location().map(|l| (l.file().to_string(), l.line()));
            let captured = CAUGHT.with(|c| match &mut *c.borrow_mut() {
                Some(slot) => {
                    *slot = location;
                    true
                }
                None => false,
            });
            if !captured {
                default(info);
            }
        }));
    });
}

/// Runs `f` and returns where it panicked, or `None` if it didn't.
fn panic_location(f: impl FnOnce()) -> Option<(String, u32)> {
    install_hook();
    CAUGHT.with(|c| *c.borrow_mut() = Some(None));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    let location = CAUGHT.with(|c| c.borrow_mut().take()).flatten();
    let panicked = result.is_err();
    panicked.then(|| location.expect("the panic had no location"))
}

/// Checks that evaluating the expression panics and that the panic points at this line of this
/// file; mismatches are collected and reported by `finish` at the end of the test, so that one
/// run lists them all. The expression must fit on the line of the macro call.
macro_rules! at {
    ($e:expr) => {{
        let line = line!();
        let mismatch = match panic_location(|| {
            let _ = $e;
        }) {
            None => Some("didn't panic".to_string()),
            Some((file, got)) if (file.as_str(), got) != (file!(), line) => {
                Some(format!("panicked at {file}:{got}"))
            }
            Some(_) => None,
        };
        if let Some(mismatch) = mismatch {
            let message = format!("line {line}: `{}` {mismatch}", stringify!($e));
            MISMATCHES.with(|m| m.borrow_mut().push(message));
        }
    }};
}

thread_local! {
    static MISMATCHES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Fails the test if any `at!` of this thread found a panic at another location.
fn finish() {
    let mismatches = MISMATCHES.with(|m| std::mem::take(&mut *m.borrow_mut()));
    assert!(
        mismatches.is_empty(),
        "panics at other locations:\n{}",
        mismatches.join("\n")
    );
}

fn t23() -> Tensor {
    homura::ones([2, 3])
}

#[test]
fn binary_operators_and_methods() {
    let (a, b) = (t23(), homura::ones([4]));
    let bools = homura::tensor([true, false]);
    let meta = homura::zeros_with([2, 3], Device::Meta);
    at!(&a + &b);
    at!(&a - b.clone());
    at!(a.clone() * &b);
    at!(a.clone() / b.clone());
    at!(a.add(&b));
    at!(a.mul(&b));
    at!(&bools - &bools);
    at!(1.0 - &bools);
    at!(bools.sub(1));
    at!(&a + &meta);
    at!(a.maximum(&b));
    at!(-&bools);
    let mut c = t23();
    at!(c += &b);
    at!(c -= homura::ones([5]));
    at!(a.add_(&b));
    let mask = homura::tensor([true, false, true, false]);
    at!(homura::where_(&mask, &a, 0.0));
    // `Sum` and `Product` called directly; through `Iterator::sum` the location is in `std`.
    let pair = [a.clone(), b.clone()];
    at!(<Tensor as Sum<&Tensor>>::sum(pair.iter()));
    at!(<Tensor as Product>::product(pair.clone().into_iter()));
    let (file, _) = panic_location(|| drop(pair.iter().sum::<Tensor>())).unwrap();
    assert!(file.contains("iter"), "reported at {file}");
    finish();
}

#[test]
fn views_indexing_and_reductions() {
    let a = t23();
    at!(a.view([5]));
    at!(a.reshape([7]));
    at!(a.view([-1, -1]));
    at!(a.expand([5, 5]));
    at!(a.permute([0]));
    at!(a.transpose(0, 5));
    at!(a.unsqueeze(4));
    at!(a.narrow(1, 2, 3));
    at!(a.select(0, 9));
    at!(a.index(5));
    at!(a.index((.., 9)));
    at!(a.sum_dim(3, false));
    at!(a.max_dim(4, false));
    at!(a.min_dim(4, false));
    at!(a.softmax(2));
    at!(a.flatten(1, 0));
    at!(a.matmul(&homura::ones([4, 2])));
    at!(a.item::<f32>());
    at!(a.index_put_((&homura::tensor([5i64]),), &homura::ones([3])));
    at!(a.index_put_accumulate_((&homura::tensor([5i64]),), &homura::ones([3])));
    at!(a.index_put((&homura::tensor([5i64]),), &homura::ones([3])));
    at!(a.index_put_accumulate((&homura::tensor([5i64]),), &homura::ones([3])));
    at!(a.index_select(0, &homura::tensor([1.5])));
    at!(a.index((&homura::tensor([0i64, 1]), &homura::tensor([0i64, 1, 2]))));
    at!(homura::zeros([2]).scatter_add(0, &homura::tensor([5i64]), &homura::ones([1])));
    let m = homura::ones_with([2, 2], DType::I32);
    at!(m.addmm(&m, &m, 1, i64::MAX));
    at!(homura::zeros([usize::MAX, 4]));
    finish();
}

#[test]
// `&[big, 1]` is the form under test.
#[allow(clippy::needless_borrows_for_generic_args)]
fn integer_arguments_too_large() {
    let (a, big) = (t23(), i64::MAX as usize + 1);
    at!(a.view([big]));
    at!(a.view(&[big, 1]));
    at!(a.view(&[big][..]));
    at!(a.reshape(vec![big]));
    at!(a.view((1, big)));
    at!(a.expand([big, 3]));
    at!(a.permute([big, 0]));
    at!(a.sum_dim(big, false));
    at!(a.squeeze_dim(vec![big]));
    at!(a.index(big));
    at!(a.index((.., 0..big)));
    at!(a.index(big..));
    at!(a.index(..big));
    at!(a.index(..=big));
    at!(a.index(big..=big));
    at!(a.index((big..=big).step_by(2)));
    at!(a.index((big..big + 4).step_by(2)));
    at!(a.index((0..usize::MAX).step_by(2)));
    at!(a.index((0..=usize::MAX).step_by(1)));
    at!(a.index((0..=usize::MAX).step_by(2)));
    at!(a.index((usize::MAX..=usize::MAX).step_by(1)));
    at!(a.index(usize::MAX..));
    at!(a.index((i64::MIN..=i64::MAX).step_by(usize::MAX)));
    finish();
}

#[test]
fn creation_and_random() {
    at!(homura::full_with([2], 1e40, DType::F32));
    at!(homura::zeros([2]).fill_(1e40));
    at!(homura::zeros_with([2], DType::U8).fill_(300));
    at!(homura::arange_step(0, 10, 0));
    at!(homura::randint(5, 2, [3]));
    at!(homura::concat(&[t23(), homura::ones([3, 4])], 0));
    at!(homura::stack(&[t23(), homura::ones([3, 2])], 0));
    at!(homura::stack::<Tensor>(&[], 0));
    at!(homura::tensor([1.0f32]).to(Device::Rocm(0)));
    finish();
}

/// A module with one linear layer and one non-module field, for the module tests.
#[derive(Module)]
struct Net {
    fc: nn::Linear,
}

#[test]
fn functional_and_layers() {
    let x = homura::ones([2, 5]);
    let lin = nn::Linear::new(3, 4);
    let w = homura::ones([4, 3]);
    at!(F::linear(&x, &w, None));
    at!(lin.forward(&x));
    at!(Forward::forward(&lin, &x));
    let boxed: Box<dyn Forward> = Box::new(nn::Linear::new(3, 4));
    at!(boxed.forward(&x));
    let seq = nn::Sequential::new()
        .append(nn::Linear::new(3, 4))
        .append(nn::Relu);
    at!(seq.forward(&x));
    let kernel = homura::ones([4, 3, 3, 3]);
    at!(F::conv2d(&x, &kernel, None, Conv2dOptions::default()));
    at!(nn::Conv2d::new(3, 4, 3).forward(&homura::ones([1, 2, 8, 8])));
    let (floats, ce) = (homura::tensor([1.0, 2.0]), CrossEntropyOptions::default());
    at!(F::cross_entropy(&x, &floats, ce));
    let (pool, image) = (nn::MaxPool2d::new(3).padding(5), homura::ones([1, 1, 8, 8]));
    at!(pool.forward(&image));
    at!(nn::Embedding::new(4, 2).forward(&homura::tensor([7i64])));
    let (ids, table) = (homura::tensor([0i64]), homura::ones([4, 2]));
    at!(F::embedding(&ids, &table, Some(9)));
    at!(nn::LayerNorm::new([3]).forward(&x));
    finish();
}

#[test]
fn layer_setters() {
    at!(nn::Conv2d::new(3, 4, 3).groups(2));
    at!(nn::Conv2d::new(3, 4, 3).groups(0));
    at!(nn::Dropout::new(1.5));
    at!(nn::Linear::new(3, 4).forward(&homura::ones([2, 4])));
    finish();
}

#[test]
fn module_methods() {
    let net = Net {
        fc: nn::Linear::new(3, 4),
    };
    at!(net.to(DType::I64));
    at!(net.to(Device::Rocm(0)));
    let shared = std::sync::Arc::new(Net {
        fc: nn::Linear::new(3, 4),
    });
    at!(shared.to(DType::I32));
    let sd = StateDict::new();
    at!(&sd["missing"]);
    at!(Parameter::new(homura::tensor([1i64])));
    let (one_dim, fan_in, leaky) = (
        homura::ones([3]),
        nn::init::FanMode::FanIn,
        nn::init::Nonlinearity::LeakyRelu,
    );
    at!(nn::init::kaiming_uniform_(&one_dim, 0.0, fan_in, leaky));
    finish();
}

fn params() -> Vec<Parameter> {
    vec![
        Parameter::new(homura::ones([2])),
        Parameter::new(homura::ones([3])),
    ]
}

#[test]
fn optimizers() {
    at!(Sgd::new(params(), -0.1));
    at!(Sgd::new(Vec::<Parameter>::new(), 0.1));
    at!(Sgd::new(params(), 0.1).momentum(-1.0));
    at!(Adam::new(params(), 0.1).betas((1.5, 0.9)));
    let p = params();
    let mut sgd = Sgd::new(p.clone(), 0.1);
    at!(sgd.add_param_group(ParamGroup::new(p.clone(), *sgd.defaults())));
    at!(sgd.options_mut(5));
    sgd.options_mut(0).lr = -1.0;
    for q in &p {
        q.set_grad(Some(homura::ones_like(q)));
    }
    at!(sgd.step());
    at!(sgd.params(1));
    at!(sgd.lr(1));
    at!(sgd.set_lr(1, 0.1));
    // Valid with the new learning rate but not with the momentum.
    sgd.options_mut(0).momentum = -1.0;
    at!(sgd.set_lr(0, 0.1));
    let mut sgd = Sgd::new(p.clone(), 0.1);
    sgd.options_mut(0).nesterov = true;
    let typed: &mut dyn TypedOptimizer<Options = SgdOptions> = &mut sgd;
    let invalid = SgdOptions {
        lr: -1.0,
        ..*typed.defaults()
    };
    at!(typed.add_param_group(ParamGroup::new(p.clone(), invalid)));
    at!(typed.options_mut(3));
    at!(typed.step());
    let dyn_opt: &mut dyn Optimizer = typed;
    at!(dyn_opt.step());
    at!(dyn_opt.params(2));
    at!(dyn_opt.lr(2));
    at!(dyn_opt.set_lr(2, 0.1));
    at!(dyn_opt.set_lr(0, 0.1));
    let mut boxed: Box<dyn TypedOptimizer<Options = SgdOptions>> =
        Box::new(Sgd::new(params(), 0.1));
    at!(boxed.options_mut(1));
    boxed.options_mut(0).momentum = -2.0;
    at!(boxed.step());
    let mut boxed: Box<dyn Optimizer> = Box::new(Adam::new(params(), 0.1));
    at!(boxed.set_lr(0, f64::NAN));
    at!(boxed.lr(1));
    // Generic code over the traits, through the forwarding implementations of `&mut` and `Box`;
    // the functions pass the location on.
    #[track_caller]
    fn set_invalid_lr(mut opt: impl Optimizer) {
        opt.set_lr(0, -1.0);
    }
    #[track_caller]
    fn add(mut opt: impl TypedOptimizer<Options = SgdOptions>, group: ParamGroup<SgdOptions>) {
        opt.add_param_group(group);
    }
    at!(set_invalid_lr(&mut boxed));
    at!(set_invalid_lr(boxed));
    let mut sgd = Sgd::new(params(), 0.1);
    at!(add(&mut sgd, ParamGroup::new(params(), invalid)));
    at!(add(Box::new(&mut sgd), ParamGroup::new(params(), invalid)));
    // A finite coefficient that overflows the parameter's dtype: a per-parameter check.
    let half = vec![Parameter::new(homura::ones_with([2], DType::F16))];
    half[0].set_grad(Some(homura::ones_with([2], DType::F16)));
    let mut sgd = Sgd::new(half, 1e6);
    at!(sgd.step());
    finish();
}

/// Returns its input, which custom functions don't support.
struct Passthrough;

impl Function for Passthrough {
    fn forward(&mut self, _ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].clone()]
    }

    fn backward(
        &self,
        _ctx: &BackwardContext<'_>,
        grads: &[Option<Tensor>],
    ) -> Vec<Option<Tensor>> {
        grads.to_vec()
    }
}

/// `2 x`.
struct Double;

impl Function for Double {
    fn forward(&mut self, _ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0] * 2.0]
    }

    fn backward(
        &self,
        _ctx: &BackwardContext<'_>,
        grads: &[Option<Tensor>],
    ) -> Vec<Option<Tensor>> {
        vec![grads[0].as_ref().map(|g| g * 2.0)]
    }
}

#[test]
fn autograd_entry_points() {
    let x = homura::ones([2]);
    x.requires_grad_(true);
    let y = (&x * 2.0).sum();
    let no_grad = homura::ones([2]);
    at!(no_grad.sum().backward());
    at!((&x * 2.0).backward());
    at!(autograd::grad(&[&y], &[], GradOptions::default()));
    at!(autograd::grad(&[&y], &[&no_grad], GradOptions::default()));
    let unused = homura::ones([2]);
    unused.requires_grad_(true);
    let w = (&x * 2.0).sum();
    at!(autograd::grad(&[&w], &[&unused], GradOptions::default()));
    let z = &x * 2.0;
    let (wrong, options) = (homura::ones([3]), BackwardOptions::default());
    at!(autograd::backward(&[&z], &[Some(&wrong)], options.clone()));
    at!(autograd::backward(&[&no_grad], &[], options));
    at!(Passthrough.apply(&[&x]));
    let o = GradOptions::default;
    at!(autograd::grad_array(&[&y], [], o()));
    at!(autograd::grad_array(&[&y], [&x, &no_grad], o()));
    at!(autograd::grad_array(&[&w], [&x, &unused], o()));
    let p = Parameter::new(homura::ones([2]));
    at!(autograd::grad_array(&[&w], [&p], o()));
    at!(Passthrough.apply_array::<1>(&[&x]));
    at!(Double.apply_array::<2>(&[&x]));
    at!(Double.apply_array::<0>(&[&x]));
    at!(homura::with_no_grad(|| Double.apply_array::<2>(&[&x])));
    at!(x.backward_with(&homura::ones([5])));
    at!(homura::tensor([1i64]).requires_grad_(true));
    // In-place on a leaf that requires grad, in grad mode.
    at!(x.add_(1.0));
    finish();
}

#[test]
fn shared_tensor_checks() {
    let (x, ints) = (t23(), homura::ones_with([2, 3], DType::I64));
    let meta = homura::ones_with([2], Device::Meta);
    at!(x.to_vec::<i64>());
    at!(homura::ones([1]).item::<i64>());
    at!(meta.to_vec::<f32>());
    at!(meta.item::<f32>());
    let (floats, row) = (homura::tensor([0.0f32]), homura::ones([1, 3]));
    at!(x.index_add(0, &floats, &row));
    at!(x.index_add_(0, &floats, &row));
    at!(x.gather(0, &x));
    at!(x.scatter_add(0, &x, &x));
    at!(x.select_scatter(&homura::ones([4]), 0, 0));
    at!(x.slice_scatter(&homura::ones([1, 4]), 0, None, None, 1));
    at!(x.slice_scatter(&x, 0, None, None, 0));
    at!(x.index(homura::tensor([5i64])));
    at!(x.index(homura::tensor([0.5f32])));
    at!(x.index_put_((&floats,), &homura::ones([3])));
    let expanded = homura::ones([3]).expand([2, 3]);
    at!(expanded.index_put_((&homura::tensor([0i64]),), &homura::ones([3])));
    at!(expanded.index_put_accumulate_((&homura::tensor([0i64]),), &homura::ones([3])));
    at!(expanded.copy_(&x));
    at!(x.index_put_((&homura::tensor([0i64]),), &x.select(0, 1)));
    at!(x.clamp(homura::ones([4]), 1.0));
    at!(x.copy_(&homura::ones([4])));
    at!(x.masked_fill(&homura::tensor([true, false, true]), homura::ones([2])));
    at!(x.as_strided([4, 4], [1, 1], 0));
    at!(ints.var(1));
    at!(x.var_dim([5], 1, false));
    at!(homura::ones([3]).triu(0));
    at!(x.normal_(0.0, -1.0));
    at!(x.uniform_(2.0, 1.0));
    at!(homura::tensor(vec![vec![1.0f32], vec![2.0, 3.0]]));
    at!(homura::zeros_with([2], Device::Rocm(0)));
    at!(F::pad(&x, &[1, 1, 1], 0.0));
    finish();
}

#[test]
fn checks_in_cpu_kernels() {
    let (x, bools) = (t23(), homura::tensor([[true, false], [true, true]]));
    at!(x.index_select(0, &homura::tensor([5i64])));
    at!(bools.matmul(&bools));
    finish();
}

#[test]
fn shared_nn_checks() {
    let (x, weight) = (t23(), homura::ones([4]));
    at!(F::layer_norm(&x, [], None, None, 1e-5));
    at!(F::layer_norm(&x, [3], Some(&weight), None, 1e-5));
    at!(F::rms_norm(&x, [], None, None));
    let training = BatchNormOptions {
        training: true,
        ..Default::default()
    };
    at!(F::batch_norm(&x, None, None, Some(&weight), None, training));
    at!(nn::LayerNorm::new([]).forward(&x));
    let smoothing = CrossEntropyOptions {
        label_smoothing: 2.0,
        ..Default::default()
    };
    let target = homura::tensor([0i64, 1]);
    at!(F::cross_entropy(&x, &target, smoothing));
    let weighted = CrossEntropyOptions {
        weight: Some(weight.clone()),
        ..Default::default()
    };
    at!(F::cross_entropy(&x, &x, weighted));
    weight.requires_grad_(true);
    let (w, mean) = (Some(&weight), Reduction::Mean);
    at!(F::nll_loss(&x, &target, w, -100, mean));
    at!(nn::init::xavier_uniform_(&homura::ones([0, 0]), 1.0));
    let same = nn::Conv2d::new(1, 1, 3).padding(Conv2dPadding::Same);
    at!(same.stride(2));
    finish();
}

/// A module with one parameter, registered under `name`, twice with `twice`.
struct Named {
    name: &'static str,
    twice: bool,
    p: Parameter,
}

impl Named {
    fn new(name: &'static str, twice: bool) -> Named {
        let p = Parameter::new(homura::ones([2]));
        Named { name, twice, p }
    }
}

impl Member for Named {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}

impl Module for Named {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        v.parameter(self.name, &self.p);
        if self.twice {
            v.parameter(self.name, &self.p);
        }
    }
}

/// A shared module that contains itself.
struct Cycle {
    me: std::sync::OnceLock<std::sync::Arc<Cycle>>,
}

impl Member for Cycle {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}

impl Module for Cycle {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        if let Some(me) = self.me.get() {
            me.visit_as("me", v);
        }
    }
}

/// A module whose `visit` panics, on the line it records first.
struct PanicsInVisit;

static VISIT_LINE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

impl Member for PanicsInVisit {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}

impl Module for PanicsInVisit {
    fn visit<'a>(&'a self, _v: &mut dyn Visitor<'a>) {
        VISIT_LINE.store(line!() + 1, std::sync::atomic::Ordering::Relaxed);
        panic!("user visit");
    }
}

/// Invalid names and cycles are found by the walks of the provided methods, outside the user's
/// `visit`, so they are reported at the call.
#[test]
fn module_tree_checks() {
    let sd = StateDict::new();
    let lax = LoadOptions::default().strict(false);
    for m in [
        Named::new("p", true),
        Named::new("", false),
        Named::new("a.b", false),
    ] {
        at!(m.named_parameters());
        at!(m.named_buffers());
        at!(m.named_modules());
        at!(m.named_children());
        at!(m.parameters());
        at!(m.state_dict());
        at!(m.load_state_dict(&sd));
        at!(m.load_state_dict_with(&sd, lax));
        at!(m.to(DType::F64));
        at!(m.train());
        at!(m.eval());
        // Rejected before anything is converted.
        assert_eq!(m.p.dtype(), DType::F32);
    }
    let cycle = std::sync::Arc::new(Cycle {
        me: std::sync::OnceLock::new(),
    });
    let _ = cycle.me.set(cycle.clone());
    at!(cycle.named_modules());
    at!(cycle.state_dict());
    at!(cycle.train());
    finish();
    // A panic in the user's own `visit` keeps its location.
    let (file, line) = panic_location(|| drop(PanicsInVisit.parameters())).expect("no panic");
    let expected = VISIT_LINE.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!((file.as_str(), line), (file!(), expected));
}

#[test]
fn gradcheck_arguments() {
    let f = |xs: &[Tensor]| vec![xs[0].mul(2.0)];
    let (constant, f32_leaf) = (homura::ones_with([2], DType::F64), homura::ones([2]));
    f32_leaf.requires_grad_(true);
    let (c, l, o) = (&[&constant], &[&f32_leaf], Default::default);
    at!(autograd::gradcheck(f, c, o()));
    at!(autograd::gradcheck(f, l, o()));
    at!(autograd::gradgradcheck(f, c, o()));
    finish();
}

/// Recording an operation on a view whose base was modified in place in grad mode after the view
/// was created is rejected where the operation is called: in the forward pass, in-place
/// operations, custom functions, and the autograd entry points.
#[test]
fn stale_views() {
    let leaf = homura::ones([2]);
    leaf.requires_grad_(true);
    let base = leaf.mul(2.0);
    let stale = base.view([2]);
    let stale_row = base.view([1, 2]);
    base.add_(1.0);
    let other = homura::ones([2]);
    at!(stale.add(1.0));
    at!(&stale * &other);
    at!(stale.sum());
    at!(stale.view([1, 2]));
    at!(homura::stack(&[&stale, &other], 0));
    at!(homura::concat(&[&stale, &other], 0));
    at!(stale.split(1, 0));
    at!(stale.chunk(2, 0));
    at!(other.clamp(&stale, 1.0));
    let square = homura::ones([2, 2]);
    at!(square.addr(&stale, &other, 1, 1));
    at!(F::mse_loss(&stale, &homura::ones([1]), Reduction::Mean));
    at!(F::mse_loss(&stale, &square, Reduction::Mean));
    let (row, t) = (homura::ones([1, 2]), homura::tensor([0i64]));
    at!(F::nll_loss(&row, &t, Some(&stale), -100, Reduction::Mean));
    at!(F::nll_loss(&stale_row, &t, None, -100, Reduction::Mean));
    at!(other.add_(&stale));
    at!(other.copy_(&stale));
    at!(Passthrough.apply(&[&stale]));
    at!(stale.backward_with(&homura::ones([2])));
    at!(stale.retain_grad());
    at!(stale.grad_fn());
    finish();
}

/// Errors found while the backward pass runs (a saved tensor modified in place, a graph run a
/// second time) are reported inside homura, not at the `backward` call: the backward formulas
/// unpack their saved tensors in closures, which `#[track_caller]` doesn't pass through, and
/// changing that would mean rewriting every formula. This documents the limitation, so that a
/// change is noticed.
#[test]
fn errors_during_the_backward_pass_are_reported_inside_homura() {
    let inside = |f: &dyn Fn()| {
        let (file, _) = panic_location(f).expect("the backward pass didn't panic");
        assert!(file.starts_with("src/autograd/"), "reported at {file}");
    };
    let w = homura::ones([2]);
    w.requires_grad_(true);
    let y = (&w * &w).sum();
    {
        let _no_grad = homura::no_grad();
        w.add_(1.0);
    }
    inside(&|| y.backward());
    let z = (&w * &w).sum();
    z.backward();
    inside(&|| z.backward());
}

#[test]
fn panics_in_user_closures_keep_their_own_location() {
    // Not routed through homura: a sanity check of the harness itself.
    at!(Vec::<u8>::new()[1]);
    // The scope closures: a panic of homura or of the user's code in the closure.
    let x = homura::ones([2]);
    at!(homura::with_no_grad(|| x.view([3])));
    at!(homura::with_enable_grad(|| x.view([3])));
    at!(homura::with_grad_enabled(false, || x.view([3])));
    at!(autograd::with_no_grad(|| Vec::<u8>::new()[1]));
    at!(homura::with_device(Device::Meta, || homura::zeros([2]).view([3])));
    at!(homura::with_device(Device::Meta, || x.to_vec::<f32>()[5]));
    let (m, v) = (Device::Meta, [3]);
    at!(homura::with_no_grad(|| homura::with_device(m, || x.view(v))));
    finish();
    // On separate lines: the line in the closure, not the line of the call.
    let start = line!();
    let grad_mode = panic_location(|| {
        homura::with_no_grad(|| {
            x.view([3]);
        });
    });
    let device = panic_location(|| {
        homura::with_device(Device::Meta, || {
            x.view([3]);
        });
    });
    let file = file!().to_string();
    assert_eq!(grad_mode, Some((file.clone(), start + 3)));
    assert_eq!(device, Some((file, start + 8)));
}
