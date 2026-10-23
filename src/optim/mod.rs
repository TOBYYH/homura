//! Optimizers, like PyTorch's `torch.optim`: [`Sgd`], [`Adam`] and [`AdamW`].
//!
//! An optimizer holds handles of the parameters it updates, in [`ParamGroup`]s with their own
//! options, and the state it keeps for each parameter. [`step`](Optimizer::step) updates the
//! parameters in place from their gradients, under [`no_grad`](crate::no_grad);
//! [`zero_grad`](Optimizer::zero_grad) clears the gradients, setting them to `None` like
//! PyTorch's default. Options that are not arguments of `new` are set by chained methods
//! (`.momentum(0.9)`), and later through [`options_mut`](Optimizer::options_mut). The
//! parameters of a group can't be changed once it is added.
//!
//! ```
//! use homura::nn::{Linear, Module};
//! use homura::optim::{Optimizer, Sgd};
//!
//! let model = Linear::new(4, 2);
//! let mut opt = Sgd::new(model.parameters(), 0.1).momentum(0.9);
//! let x = homura::ones([8, 4]);
//! for _ in 0..3 {
//!     opt.zero_grad();
//!     let loss = model.forward(&x).pow(2.0).mean();
//!     loss.backward();
//!     opt.step();
//! }
//! opt.options_mut(0).lr = 0.01;
//! ```
//!
//! The updates follow PyTorch's single-tensor implementations (the ones it runs on the CPU),
//! operation by operation. Each scaled addition (`add(alpha=)`), `lerp`, `addcmul` and
//! `addcdiv` of a step is computed in one go, like PyTorch's vectorized CPU kernels compute it:
//! in `f32` for `f16` and `bf16`, with the same fused multiply-adds, rounded once, so that
//! intermediate products neither overflow nor vanish. In the cases tested against the reference
//! build (PyTorch 2.14.1 on an AVX2 CPU) the parameters and the state are the same as PyTorch's,
//! bit for bit, except for `f16` and `bf16` scaled additions of the elements after the last
//! multiple of 32, which PyTorch's scalar loop rounds after each operation; homura computes them
//! like the others. Other builds of PyTorch may round differently. The other operations of a step
//! (`mul_`, `sqrt`, division by a scalar, `maximum`) are homura's tensor operations. This
//! applies to the optimizers only: [`Tensor::lerp`](crate::Tensor::lerp),
//! [`addcmul`](crate::Tensor::addcmul) and [`addcdiv`](crate::Tensor::addcdiv) round after each
//! of their operations. A parameter whose gradient is `None` is skipped: it gets no
//! update, no weight decay and no state. A zero gradient is a gradient. Freezing a parameter
//! with [`requires_grad_(false)`](crate::Tensor::requires_grad_) keeps its gradient, and the
//! optimizer still updates it from that gradient, as PyTorch's does; clear the gradient too, with
//! [`set_grad(None)`](crate::Tensor::set_grad).
//!
//! The state of a parameter is created at its first step with a gradient, with the parameter's
//! dtype and device. Create the optimizer after moving the model with
//! [`Module::to`](crate::nn::Module::to): if a parameter is moved after its state was created,
//! [`step`](Optimizer::step) panics before updating anything. Optimizers can't save or load
//! their state yet, so a training run can't be resumed exactly.
//!
//! # Differences from PyTorch
//!
//! - A parameter listed twice in one group is an error; PyTorch warns and updates it twice.
//! - The parameter groups can't be changed or reordered once added, only their options
//!   ([`options_mut`](Optimizer::options_mut)): the state of a parameter is kept by its position
//!   in the groups, where PyTorch keys it by the parameter.
//! - [`step`](Optimizer::step) checks the options of every group, that the parameters with a
//!   gradient are still leaves, and the shapes, dtypes and devices of their state before it
//!   updates anything, and panics if one is invalid. PyTorch checks the options and leaves only
//!   when it adds a group; with state of another dtype its Adam fails in the middle of a step,
//!   after updating other parameters, and its SGD goes on with the old momentum buffer. Options
//!   that depend on each other (SGD's `nesterov`) are checked when a group is added and in
//!   `step`, not by the chained methods, so they can be set in any order.
//! - Learning rates and other options are `f64`s, not tensors; there are no closures, `foreach`,
//!   `fused`, `capturable` or `differentiable` variants, and no hooks.

mod adam;
#[cfg(doctest)]
mod errors;
mod sgd;

pub use adam::{Adam, AdamOptions, AdamState, AdamW};
pub use sgd::{Sgd, SgdOptions, SgdState};

use crate::backend::cpu::{self, Fused};
use crate::nn::Parameter;
use crate::{DType, Device, Scalar, Tensor};

/// What PyTorch's `torch.optim.Optimizer` provides: updating parameters from their gradients.
///
/// The options of a group are the optimizer's typed options struct, such as [`SgdOptions`].
pub trait Optimizer {
    /// The options of a parameter group.
    type Options;

    /// Updates the parameters from their gradients (`step()`), in place and under
    /// [`no_grad`](crate::no_grad).
    ///
    /// # Panics
    ///
    /// Panics before updating anything if the options of a group are invalid, a parameter with
    /// a gradient is no longer a leaf, the state of a parameter doesn't match its shape, dtype
    /// or device, or a finite scalar of the step overflows the type PyTorch's CPU kernels convert
    /// it to ("value cannot be converted to type f16 without overflow"): the `alpha` of a scaled
    /// addition (SGD's `-lr`, weight decay, dampening and Nesterov momentum, Adam's weight decay)
    /// in the parameter's dtype, and Adam's `-lr / (1 - beta1^step)` in its computation type
    /// (F32 for F16 and BF16).
    #[track_caller]
    fn step(&mut self);

    /// Sets the gradient of every parameter to `None` (`zero_grad()`).
    #[track_caller]
    fn zero_grad(&mut self);

    /// The options that [`ParamGroup`]s are based on: the arguments of `new` and the chained
    /// methods (`defaults`). Groups added later can start from them,
    /// `SgdOptions { lr: 0.01, ..*opt.defaults() }`.
    #[track_caller]
    fn defaults(&self) -> &Self::Options;

    /// The parameter groups, in the order they were added (`param_groups`).
    #[track_caller]
    fn param_groups(&self) -> &[ParamGroup<Self::Options>];

    /// The options of the parameter group `index`, to change them
    /// (`param_groups[index][..] = ..`). The parameters of a group can't be changed: PyTorch
    /// keys the state by parameter, homura by position in the groups.
    ///
    /// # Panics
    ///
    /// Panics if there is no group `index`.
    #[track_caller]
    fn options_mut(&mut self, index: usize) -> &mut Self::Options;

    /// Adds a group of parameters with its own options (`add_param_group`).
    ///
    /// # Panics
    ///
    /// Panics if the options are invalid, a parameter is not a leaf, a parameter is already in
    /// the optimizer ("some parameters appear in more than one parameter group"), or a
    /// parameter appears twice in the group.
    #[track_caller]
    fn add_param_group(&mut self, group: ParamGroup<Self::Options>);
}

/// Parameters that an optimizer updates with the same options (PyTorch's param group).
#[derive(Clone, Debug)]
pub struct ParamGroup<O> {
    params: Vec<Parameter>,
    /// The options of this group.
    pub options: O,
}

impl<O> ParamGroup<O> {
    /// A group of `params` with `options`.
    #[track_caller]
    pub fn new(params: impl IntoIterator<Item = Parameter>, options: O) -> ParamGroup<O> {
        ParamGroup {
            params: params.into_iter().collect(),
            options,
        }
    }

    /// The parameters of this group.
    #[track_caller]
    pub fn params(&self) -> &[Parameter] {
        &self.params
    }
}

/// The checks of an optimizer's options, which [`Groups`] makes. They are trait methods called
/// statically, not callbacks: `#[track_caller]` passes the location of the user's call through
/// direct and generic calls, but not through function pointers or closures.
trait GroupOptions: Copy {
    /// The checks of single values, made by the chained option methods.
    #[track_caller]
    fn check_values(&self) {
        self.check();
    }

    /// All checks of a group's options, made when it is added and in `step`.
    #[track_caller]
    fn check(&self);
}

/// The checks of a step for one parameter with a gradient, made before anything is updated.
trait ParamCheck<O, S> {
    /// Panics if the step can't update `param`, with the options `options` of its group and its
    /// state `state`.
    #[track_caller]
    fn check_param(&self, options: &O, param: &Tensor, state: Option<&S>);
}

/// The parameter groups of an optimizer with the state `S` of each parameter, which the
/// optimizers share.
#[derive(Debug)]
struct Groups<O, S> {
    defaults: O,
    groups: Vec<ParamGroup<O>>,
    /// `state[g][i]` belongs to `groups[g].params[i]`; groups can't change their parameters.
    state: Vec<Vec<Option<S>>>,
}

/// A parameter that has a gradient in a step: its group, its index there and the gradient.
struct WithGrad {
    group: usize,
    index: usize,
    grad: Tensor,
}

impl<O: GroupOptions, S> Groups<O, S> {
    /// The groups of an optimizer made with `params` and `defaults`.
    #[track_caller]
    fn new(params: impl IntoIterator<Item = Parameter>, defaults: O) -> Self {
        let params: Vec<Parameter> = params.into_iter().collect();
        if params.is_empty() {
            panic!("optimizer got an empty parameter list");
        }
        let mut groups = Groups {
            defaults,
            groups: vec![],
            state: vec![],
        };
        groups.add(ParamGroup::new(params, defaults));
        groups
    }

    #[track_caller]
    fn add(&mut self, group: ParamGroup<O>) {
        group.options.check();
        for (i, p) in group.params.iter().enumerate() {
            check_leaf(p);
            if group.params[..i].iter().any(|q| q.is_same(p)) {
                panic!("optimizer contains a parameter group with duplicate parameters");
            }
        }
        let mut existing = self.groups.iter().flat_map(|g| &g.params);
        if existing.any(|q| group.params.iter().any(|p| q.is_same(p))) {
            panic!("some parameters appear in more than one parameter group");
        }
        self.state.push(group.params.iter().map(|_| None).collect());
        self.groups.push(group);
    }

    /// Applies a chained option method to the defaults and to every group. `f` only assigns;
    /// the check runs here.
    #[track_caller]
    fn set(&mut self, f: impl Fn(&mut O)) {
        f(&mut self.defaults);
        self.defaults.check_values();
        for g in &mut self.groups {
            f(&mut g.options);
        }
    }

    #[track_caller]
    fn options_mut(&mut self, index: usize) -> &mut O {
        let len = self.groups.len();
        match self.groups.get_mut(index) {
            Some(g) => &mut g.options,
            None => panic!("parameter group index {index} is out of range for {len} groups"),
        }
    }

    fn state(&self, param: &Parameter) -> Option<&S> {
        self.groups.iter().zip(&self.state).find_map(|(g, s)| {
            let i = g.params.iter().position(|p| p.is_same(param))?;
            s[i].as_ref()
        })
    }

    fn zero_grad(&self) {
        for p in self.groups.iter().flat_map(|g| &g.params) {
            p.set_grad(None);
        }
    }

    /// The checks of a step, before it updates anything: the options of each group, and for each
    /// parameter with a gradient that it is a leaf and `checker`'s checks of it and its state, if
    /// it has one. Returns those parameters with their gradients, in order.
    #[track_caller]
    fn prepare(&self, checker: &impl ParamCheck<O, S>) -> Vec<WithGrad> {
        let mut out = vec![];
        for (g, (group, state)) in self.groups.iter().zip(&self.state).enumerate() {
            group.options.check();
            for (i, p) in group.params.iter().enumerate() {
                let Some(grad) = p.grad() else { continue };
                check_leaf(p);
                checker.check_param(&group.options, p, state[i].as_ref());
                out.push(WithGrad {
                    group: g,
                    index: i,
                    grad,
                });
            }
        }
        out
    }
}

/// `a + alpha * b`, rounded once (see [`Fused`]).
#[track_caller]
fn scaled_add(a: &Tensor, b: &Tensor, alpha: f64) -> Tensor {
    fused(Fused::ScaledAdd(alpha), a, b, None)
}

/// `lerp(a, b, weight)`, rounded once.
#[track_caller]
fn lerp(a: &Tensor, b: &Tensor, weight: f64) -> Tensor {
    fused(Fused::Lerp(weight), a, b, None)
}

/// `a + value * b * c`, rounded once.
#[track_caller]
fn addcmul(a: &Tensor, b: &Tensor, c: &Tensor, value: f64) -> Tensor {
    fused(Fused::Addcmul(value), a, b, Some(c))
}

/// `a + value * b / c`, rounded once.
#[track_caller]
fn addcdiv(a: &Tensor, b: &Tensor, c: &Tensor, value: f64) -> Tensor {
    fused(Fused::Addcdiv(value), a, b, Some(c))
}

/// A new tensor with `op` of tensors that have the same shape, floating point dtype and device:
/// a parameter, its gradient and its state. It records no history.
///
/// # Panics
///
/// Panics on the CPU if the scalar of `op` doesn't fit the type PyTorch converts it to (see
/// [`check_coefficient`]); `step` checks the scalars before it updates anything.
#[track_caller]
fn fused(op: Fused, a: &Tensor, b: &Tensor, c: Option<&Tensor>) -> Tensor {
    let out = Tensor::empty_on(a.shape(), a.dtype(), a.device());
    if a.device() == Device::Cpu {
        let (value, dtype) = match op {
            Fused::ScaledAdd(alpha) => (alpha, a.dtype()),
            Fused::Lerp(v) | Fused::Addcmul(v) | Fused::Addcdiv(v) => (v, computation(a.dtype())),
        };
        check_coefficient(value, dtype);
        let c = c.map(Tensor::data);
        cpu::fused(op, &out.data(), &a.data(), &b.data(), c.as_ref());
    }
    out
}

/// The type the CPU kernels compute a floating point `dtype` in: F32 for F16 and BF16.
fn computation(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    }
}

/// Panics like PyTorch's CPU kernels if the finite scalar argument `value` is out of the range of
/// `dtype`, which they convert it to with an overflow check: `alpha` of `add` to the dtype of the
/// tensor, `value` of `addcmul` / `addcdiv` and the weight of `lerp` to its computation type.
/// Infinities and NaN pass, as there. Scalar operands of other operations (`mul_(momentum)`,
/// `add_(eps)`) aren't checked, like PyTorch's, and the meta device checks nothing.
#[track_caller]
fn check_coefficient(value: f64, dtype: DType) {
    if Scalar::Float(value).checked_to(dtype).is_none() {
        panic!("value cannot be converted to type {dtype} without overflow");
    }
}

/// [`check_coefficient`] for the scalars a step uses with the parameter `p`, on the CPU.
#[track_caller]
fn check_coefficients(p: &Tensor, coefficients: impl IntoIterator<Item = (f64, DType)>) {
    if p.device() == Device::Cpu {
        for (value, dtype) in coefficients {
            check_coefficient(value, dtype);
        }
    }
}

#[track_caller]
fn check_leaf(p: &Tensor) {
    if p.grad_fn().is_some() {
        panic!("can't optimize a non-leaf Tensor");
    }
}

/// Panics unless the state tensor `name` matches the parameter `p`, which changes when the
/// parameter is moved with `Module::to` after the state was created.
#[track_caller]
fn check_state_tensor(p: &Tensor, t: &Tensor, name: &str) {
    if t.shape() != p.shape() || t.dtype() != p.dtype() || t.device() != p.device() {
        panic!(
            "the optimizer state {name} of a parameter has shape {:?}, dtype {} and device {}, \
             but the parameter has shape {:?}, dtype {} and device {}; move the model before \
             creating the optimizer",
            t.shape(),
            t.dtype(),
            t.device(),
            p.shape(),
            p.dtype(),
            p.device()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f16;

    #[test]
    fn fused_operations_check_their_scalar() {
        let t = crate::ones_with([2], DType::F16);
        let r = std::panic::catch_unwind(|| scaled_add(&t, &t, 1e5));
        assert!(r.is_err());
        let r = std::panic::catch_unwind(|| addcdiv(&t, &t, &t, 1e39));
        assert!(r.is_err());
        // In F32, the computation type of F16.
        assert_eq!(addcdiv(&t, &t, &t, 1e5).to_vec::<f16>(), [f16::INFINITY; 2]);
        assert_eq!(
            scaled_add(&t, &t, f64::INFINITY).to_vec::<f16>(),
            [f16::INFINITY; 2]
        );
    }

    #[test]
    fn optimizers_are_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<Sgd>();
        check::<Adam>();
        check::<AdamW>();
        check::<ParamGroup<SgdOptions>>();
    }
}
