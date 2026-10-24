//! [`Sgd`]: stochastic gradient descent.

use super::{
    GroupOptions, Groups, Optimizer, ParamCheck, ParamGroup, TypedOptimizer, check_coefficients,
    check_state_tensor, scaled_add,
};
use crate::Tensor;
use crate::nn::Parameter;

/// The options of an [`Sgd`] parameter group, with PyTorch's names and defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgdOptions {
    /// The learning rate, `lr`.
    pub lr: f64,
    /// The momentum factor, `momentum` (0).
    pub momentum: f64,
    /// The dampening of the momentum, `dampening` (0).
    pub dampening: f64,
    /// The L2 penalty added to the gradient, `weight_decay` (0).
    pub weight_decay: f64,
    /// Whether to use Nesterov momentum, `nesterov` (false). Needs a positive momentum and zero
    /// dampening.
    pub nesterov: bool,
    /// Whether to maximize the objective instead of minimizing it, `maximize` (false).
    pub maximize: bool,
}

/// The state of a parameter in an [`Sgd`] optimizer with momentum.
#[derive(Clone, Debug)]
pub struct SgdState {
    momentum_buffer: Tensor,
}

impl SgdState {
    /// The momentum buffer (`momentum_buffer`), with the parameter's shape, dtype and device.
    #[track_caller]
    pub fn momentum_buffer(&self) -> &Tensor {
        &self.momentum_buffer
    }
}

/// Stochastic gradient descent, optionally with momentum, PyTorch's `torch.optim.SGD`.
///
/// For each parameter with a gradient `g`, a step computes, in this order: `g = -g` if
/// `maximize`; `g = g + weight_decay * p` if `weight_decay` isn't 0; with momentum, the buffer
/// `b = momentum * b + (1 - dampening) * g`, or `b = g` at the first step, and then
/// `g = g + momentum * b` if `nesterov`, `g = b` otherwise; and `p = p - lr * g`.
///
/// A parameter has state (an [`SgdState`]) once it had a step with a gradient while the
/// momentum of its group wasn't 0. The momentum buffer of a group whose momentum is set to 0
/// later is kept but not used.
///
/// ```
/// use homura::nn::Parameter;
/// use homura::optim::{Optimizer, Sgd};
///
/// let p = Parameter::new(homura::tensor([1.0, 2.0]));
/// let mut opt = Sgd::new([p.clone()], 0.5).momentum(0.9);
/// p.set_grad(Some(homura::tensor([1.0, -1.0])));
/// opt.step();
/// assert_eq!(p.to_vec::<f32>(), [0.5, 2.5]);
/// assert_eq!(opt.state(&p).unwrap().momentum_buffer().to_vec::<f32>(), [1.0, -1.0]);
/// ```
#[derive(Debug)]
pub struct Sgd {
    groups: Groups<SgdOptions, SgdState>,
}

impl Sgd {
    /// An optimizer of `params` with learning rate `lr` and the default options, as one
    /// parameter group (`SGD(params, lr)`).
    ///
    /// # Panics
    ///
    /// Panics if `params` is empty or `lr` is negative, with PyTorch's messages, or if a
    /// parameter is not a leaf or appears twice.
    #[track_caller]
    pub fn new(params: impl IntoIterator<Item = Parameter>, lr: f64) -> Sgd {
        let defaults = SgdOptions {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            maximize: false,
        };
        Sgd {
            groups: Groups::new(params, defaults),
        }
    }

    /// Sets `momentum` of the defaults and of every group.
    ///
    /// # Panics
    ///
    /// Panics if `momentum` is negative.
    #[must_use]
    #[track_caller]
    pub fn momentum(mut self, momentum: f64) -> Sgd {
        self.groups.set(|o| o.momentum = momentum);
        self
    }

    /// Sets `dampening` of the defaults and of every group.
    #[must_use]
    #[track_caller]
    pub fn dampening(mut self, dampening: f64) -> Sgd {
        self.groups.set(|o| o.dampening = dampening);
        self
    }

    /// Sets `weight_decay` of the defaults and of every group.
    ///
    /// # Panics
    ///
    /// Panics if `weight_decay` is negative.
    #[must_use]
    #[track_caller]
    pub fn weight_decay(mut self, weight_decay: f64) -> Sgd {
        self.groups.set(|o| o.weight_decay = weight_decay);
        self
    }

    /// Sets `nesterov` of the defaults and of every group. Whether the momentum and dampening
    /// allow it is checked by [`step`](Optimizer::step).
    #[must_use]
    #[track_caller]
    pub fn nesterov(mut self, nesterov: bool) -> Sgd {
        self.groups.set(|o| o.nesterov = nesterov);
        self
    }

    /// Sets `maximize` of the defaults and of every group.
    #[must_use]
    #[track_caller]
    pub fn maximize(mut self, maximize: bool) -> Sgd {
        self.groups.set(|o| o.maximize = maximize);
        self
    }

    /// The state of `param` (`state[param]`), if it is a parameter of this optimizer that has
    /// state.
    #[track_caller]
    pub fn state(&self, param: &Parameter) -> Option<&SgdState> {
        self.groups.state(param)
    }
}

impl Optimizer for Sgd {
    fn step(&mut self) {
        let params = self.groups.prepare(&SgdCheck);
        let _no_grad = crate::no_grad();
        for item in params {
            let group = &self.groups.groups[item.group];
            let (o, p) = (group.options, &group.params[item.index]);
            let mut grad = if o.maximize {
                item.grad.neg()
            } else {
                item.grad
            };
            if o.weight_decay != 0.0 {
                grad = scaled_add(&grad, p, o.weight_decay);
            }
            if o.momentum != 0.0 {
                let state = &mut self.groups.state[item.group][item.index];
                let buf = match state {
                    None => {
                        let buf = grad.detach().deep_clone();
                        *state = Some(SgdState {
                            momentum_buffer: buf.clone(),
                        });
                        buf
                    }
                    Some(s) => {
                        let buf = &s.momentum_buffer;
                        buf.mul_(o.momentum)
                            .copy_(&scaled_add(buf, &grad, 1.0 - o.dampening));
                        buf.clone()
                    }
                };
                grad = if o.nesterov {
                    scaled_add(&grad, &buf, o.momentum)
                } else {
                    buf
                };
            }
            p.copy_(&scaled_add(p, &grad, -o.lr));
        }
    }

    fn zero_grad(&mut self) {
        self.groups.zero_grad();
    }

    fn num_param_groups(&self) -> usize {
        self.groups.groups.len()
    }

    fn params(&self, group: usize) -> &[Parameter] {
        self.groups.group(group).params()
    }

    fn lr(&self, group: usize) -> f64 {
        self.groups.lr(group)
    }

    fn set_lr(&mut self, group: usize, lr: f64) {
        self.groups.set_lr(group, lr);
    }
}

impl TypedOptimizer for Sgd {
    type Options = SgdOptions;

    fn defaults(&self) -> &SgdOptions {
        &self.groups.defaults
    }

    fn param_groups(&self) -> &[ParamGroup<SgdOptions>] {
        &self.groups.groups
    }

    fn options_mut(&mut self, index: usize) -> &mut SgdOptions {
        self.groups.options_mut(index)
    }

    fn add_param_group(&mut self, group: ParamGroup<SgdOptions>) {
        self.groups.add(group);
    }
}

impl GroupOptions for SgdOptions {
    /// The checks of single values, which PyTorch makes in `SGD()`; NaN passes them, as there.
    fn check_values(&self) {
        if self.lr < 0.0 {
            panic!("Invalid learning rate: {}", self.lr);
        }
        if self.momentum < 0.0 {
            panic!("Invalid momentum value: {}", self.momentum);
        }
        if self.weight_decay < 0.0 {
            panic!("Invalid weight_decay value: {}", self.weight_decay);
        }
    }

    fn check(&self) {
        self.check_values();
        if self.nesterov && (self.momentum <= 0.0 || self.dampening != 0.0) {
            panic!("Nesterov momentum requires a momentum and zero dampening");
        }
    }

    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// The per-parameter checks of an SGD step.
struct SgdCheck;

impl ParamCheck<SgdOptions, SgdState> for SgdCheck {
    fn check_param(&self, o: &SgdOptions, p: &Tensor, s: Option<&SgdState>) {
        if let Some(s) = s {
            check_state_tensor(p, &s.momentum_buffer, "momentum_buffer");
        }
        // The `alpha`s of the step's `add`s, in the parameter's dtype.
        let mut alphas = vec![];
        if o.weight_decay != 0.0 {
            alphas.push(o.weight_decay);
        }
        if o.momentum != 0.0 {
            if s.is_some() {
                alphas.push(1.0 - o.dampening);
            }
            if o.nesterov {
                alphas.push(o.momentum);
            }
        }
        alphas.push(-o.lr);
        let dtype = p.dtype();
        check_coefficients(p, alphas.into_iter().map(|a| (a, dtype)));
    }
}
