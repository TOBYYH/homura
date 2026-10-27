//! [`Adam`] and [`AdamW`].

use super::state_dict::{Fields, Schema, load_moment, load_step};
use super::{
    GroupOptions, Groups, LoadError, Optimizer, OptimizerStateDict, ParamCheck, ParamGroup,
    TypedOptimizer, addcdiv, addcmul, check_coefficients, check_state_tensor, computation, lerp,
    scaled_add,
};
use crate::io::Value;
use crate::nn::Parameter;
use crate::{DType, Device, Tensor};

/// The options of an [`Adam`] or [`AdamW`] parameter group, with PyTorch's names. The defaults
/// are PyTorch's; `weight_decay` defaults to 0 for Adam and to 0.01 for AdamW.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamOptions {
    /// The learning rate, `lr`.
    pub lr: f64,
    /// The decay rates of the averages of the gradient and of its square, `betas`
    /// ((0.9, 0.999)).
    pub betas: (f64, f64),
    /// The term added to the denominator, `eps` (1e-8).
    pub eps: f64,
    /// The weight decay, `weight_decay`: an L2 penalty added to the gradient for Adam,
    /// decoupled from it for AdamW.
    pub weight_decay: f64,
    /// Whether to use the AMSGrad variant, `amsgrad` (false).
    pub amsgrad: bool,
    /// Whether to maximize the objective instead of minimizing it, `maximize` (false).
    pub maximize: bool,
}

/// The state of a parameter in an [`Adam`] or [`AdamW`] optimizer.
#[derive(Clone, Debug)]
pub struct AdamState {
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Option<Tensor>,
}

impl AdamState {
    fn new(p: &Tensor, amsgrad: bool) -> AdamState {
        AdamState {
            step: crate::zeros_with([], (scalar_dtype(), Device::Cpu)),
            exp_avg: crate::zeros_like(p),
            exp_avg_sq: crate::zeros_like(p),
            max_exp_avg_sq: amsgrad.then(|| crate::zeros_like(p)),
        }
    }

    /// The number of steps taken (`step`): a 0-D CPU tensor, F64 if the default dtype was F64
    /// when the state was created and F32 otherwise, as in PyTorch. The bias corrections read
    /// it.
    #[track_caller]
    pub fn step(&self) -> &Tensor {
        &self.step
    }

    /// The moving average of the gradient (`exp_avg`).
    #[track_caller]
    pub fn exp_avg(&self) -> &Tensor {
        &self.exp_avg
    }

    /// The moving average of the squared gradient (`exp_avg_sq`).
    #[track_caller]
    pub fn exp_avg_sq(&self) -> &Tensor {
        &self.exp_avg_sq
    }

    /// The maximum of `exp_avg_sq` so far (`max_exp_avg_sq`), if the state was created with
    /// `amsgrad`.
    #[track_caller]
    pub fn max_exp_avg_sq(&self) -> Option<&Tensor> {
        self.max_exp_avg_sq.as_ref()
    }
}

/// The dtype of a new `step`, PyTorch's `_get_scalar_dtype()`; it is on the CPU whatever the
/// parameter's device.
fn scalar_dtype() -> DType {
    match crate::default_dtype() {
        DType::F64 => DType::F64,
        _ => DType::F32,
    }
}

/// What Adam and AdamW share: they differ in the default weight decay and how it is applied.
#[derive(Debug)]
struct Inner {
    groups: Groups<AdamOptions, AdamState>,
    decoupled_weight_decay: bool,
}

impl Inner {
    fn new(
        params: impl IntoIterator<Item = Parameter>,
        lr: f64,
        weight_decay: f64,
        decoupled_weight_decay: bool,
    ) -> Inner {
        let defaults = AdamOptions {
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay,
            amsgrad: false,
            maximize: false,
        };
        Inner {
            groups: Groups::new(params, defaults),
            decoupled_weight_decay,
        }
    }

    fn schema(&self) -> AdamSchema {
        AdamSchema {
            decoupled: self.decoupled_weight_decay,
        }
    }

    /// PyTorch's `_single_tensor_adam`.
    #[track_caller]
    fn step(&mut self) {
        let checker = AdamCheck {
            decoupled: self.decoupled_weight_decay,
        };
        let params = self.groups.prepare(&checker);
        let _no_grad = crate::no_grad();
        for item in params {
            let group = &self.groups.groups[item.group];
            let (o, p) = (group.options, &group.params[item.index]);
            let state = self.groups.state[item.group][item.index]
                .get_or_insert_with(|| AdamState::new(p, o.amsgrad));
            let mut grad = if o.maximize {
                item.grad.neg()
            } else {
                item.grad
            };
            state.step.add_(1.0);
            if o.weight_decay != 0.0 {
                if self.decoupled_weight_decay {
                    p.mul_(1.0 - o.lr * o.weight_decay);
                } else {
                    grad = scaled_add(&grad, p, o.weight_decay);
                }
            }
            let (beta1, beta2) = o.betas;
            let (exp_avg, exp_avg_sq) = (&state.exp_avg, &state.exp_avg_sq);
            exp_avg.copy_(&lerp(exp_avg, &grad, 1.0 - beta1));
            exp_avg_sq
                .mul_(beta2)
                .copy_(&addcmul(exp_avg_sq, &grad, &grad, 1.0 - beta2));
            let step = state.step.to(DType::F64).item::<f64>();
            let bias_correction2 = 1.0 - beta2.powf(step);
            let step_size = step_size(&o, step);
            // Python's `** 0.5`, which is `pow`, not `sqrt`.
            let bias_correction2_sqrt = bias_correction2.powf(0.5);
            let second = match &state.max_exp_avg_sq {
                Some(max) if o.amsgrad => {
                    max.copy_(&max.maximum(&state.exp_avg_sq));
                    max
                }
                _ => &state.exp_avg_sq,
            };
            let denom = second.sqrt().div(bias_correction2_sqrt);
            denom.add_(o.eps);
            p.copy_(&addcdiv(p, &state.exp_avg, &denom, -step_size));
        }
    }
}

/// What Adam and AdamW save: PyTorch 2.14's options, and the state.
struct AdamSchema {
    decoupled: bool,
}

impl Schema<AdamOptions, AdamState> for AdamSchema {
    fn save_options(&self, o: &AdamOptions) -> Vec<(String, Value)> {
        [
            ("lr", Value::Float(o.lr)),
            ("betas", Value::tuple([o.betas.0, o.betas.1])),
            ("eps", Value::Float(o.eps)),
            ("weight_decay", Value::Float(o.weight_decay)),
            ("amsgrad", Value::Bool(o.amsgrad)),
            ("maximize", Value::Bool(o.maximize)),
            ("foreach", Value::None),
            ("capturable", Value::Bool(false)),
            ("differentiable", Value::Bool(false)),
            ("fused", Value::None),
            ("decoupled_weight_decay", Value::Bool(self.decoupled)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    /// The options of `Adam()` are required; the others default as in PyTorch's
    /// `Adam.__setstate__`. A `decoupled_weight_decay` must be the optimizer's: False for Adam,
    /// True for AdamW. Without it, both load the options.
    fn load_options(&self, f: &mut Fields) -> AdamOptions {
        let options = AdamOptions {
            lr: f.number("lr", None),
            betas: f.pair("betas"),
            eps: f.number("eps", None),
            weight_decay: f.number("weight_decay", None),
            amsgrad: f.flag("amsgrad", false),
            maximize: f.flag("maximize", false),
        };
        f.unsupported(
            "foreach",
            |v| matches!(v, Value::None | Value::Bool(_)),
            "None or a bool",
        );
        f.unsupported("capturable", |v| matches!(v, Value::Bool(false)), "False");
        f.unsupported(
            "differentiable",
            |v| matches!(v, Value::Bool(false)),
            "False",
        );
        f.unsupported(
            "fused",
            |v| matches!(v, Value::None | Value::Bool(false)),
            "None or False",
        );
        let name = "decoupled_weight_decay";
        if f.flag(name, self.decoupled) != self.decoupled {
            let (other, this) = match self.decoupled {
                true => ("Adam", "AdamW"),
                false => ("AdamW", "Adam"),
            };
            f.error(
                name,
                format_args!(
                    "is {}: the state dict is {other}'s, this is {this}",
                    !self.decoupled
                ),
            );
        }
        options
    }

    fn save_state(&self, s: &AdamState) -> Vec<(String, Value)> {
        let mut out = vec![
            ("step".to_string(), Value::Tensor(s.step.clone())),
            ("exp_avg".to_string(), Value::Tensor(s.exp_avg.clone())),
            (
                "exp_avg_sq".to_string(),
                Value::Tensor(s.exp_avg_sq.clone()),
            ),
        ];
        if let Some(max) = &s.max_exp_avg_sq {
            out.push(("max_exp_avg_sq".to_string(), Value::Tensor(max.clone())));
        }
        out
    }

    /// No state for an empty record. `max_exp_avg_sq` is required with `amsgrad` and kept
    /// without it, for when `amsgrad` is turned on again.
    fn load_state(&self, o: &AdamOptions, p: &Tensor, f: &mut Fields) -> Option<AdamState> {
        if f.is_empty() {
            return None;
        }
        let step = load_step(f, scalar_dtype());
        let exp_avg = load_moment(p, f, "exp_avg");
        let exp_avg_sq = load_moment(p, f, "exp_avg_sq");
        let max_exp_avg_sq = match o.amsgrad || f.get("max_exp_avg_sq").is_some() {
            true => Some(load_moment(p, f, "max_exp_avg_sq")?),
            false => None,
        };
        Some(AdamState {
            step: step?,
            exp_avg: exp_avg?,
            exp_avg_sq: exp_avg_sq?,
            max_exp_avg_sq,
        })
    }
}

/// `lr / (1 - beta1^step)`, the size of the update at `step`.
fn step_size(o: &AdamOptions, step: f64) -> f64 {
    o.lr / (1.0 - o.betas.0.powf(step))
}

/// The per-parameter checks of an Adam or AdamW step.
struct AdamCheck {
    decoupled: bool,
}

impl ParamCheck<AdamOptions, AdamState> for AdamCheck {
    fn check_param(&self, o: &AdamOptions, p: &Tensor, s: Option<&AdamState>) {
        // `alpha` of the coupled weight decay in the parameter's dtype, `value` of the update's
        // addcdiv in its computation type, at the step that comes next.
        let step = match s {
            Some(s) => s.step.add(1.0).to(DType::F64).item::<f64>(),
            None => 1.0,
        };
        let mut coefficients = vec![(-step_size(o, step), computation(p.dtype()))];
        if o.weight_decay != 0.0 && !self.decoupled {
            coefficients.push((o.weight_decay, p.dtype()));
        }
        check_coefficients(p, coefficients);
        let Some(s) = s else { return };
        check_state_tensor(p, &s.exp_avg, "exp_avg");
        check_state_tensor(p, &s.exp_avg_sq, "exp_avg_sq");
        if o.amsgrad {
            match &s.max_exp_avg_sq {
                Some(m) => check_state_tensor(p, m, "max_exp_avg_sq"),
                None => panic!(
                    "the optimizer state of a parameter has no max_exp_avg_sq: amsgrad was \
                     turned on after the state was created"
                ),
            }
        }
    }
}

impl GroupOptions for AdamOptions {
    fn validate(&self) -> Result<(), String> {
        validate(self)
    }

    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// The checks of PyTorch's `Adam()`; NaN fails them, as there.
fn validate(o: &AdamOptions) -> Result<(), String> {
    if !(0.0..).contains(&o.lr) {
        return Err(format!("Invalid learning rate: {}", o.lr));
    }
    if !(0.0..).contains(&o.eps) {
        return Err(format!("Invalid epsilon value: {}", o.eps));
    }
    let (beta1, beta2) = o.betas;
    if !(0.0..1.0).contains(&beta1) {
        return Err(format!("Invalid beta parameter at index 0: {beta1}"));
    }
    if !(0.0..1.0).contains(&beta2) {
        return Err(format!("Invalid beta parameter at index 1: {beta2}"));
    }
    if !(0.0..).contains(&o.weight_decay) {
        return Err(format!("Invalid weight_decay value: {}", o.weight_decay));
    }
    Ok(())
}

/// The chained option methods, `state` and the [`Optimizer`] and [`TypedOptimizer`]
/// implementations of Adam and AdamW.
macro_rules! adam_methods {
    ($name:ident) => {
        impl $name {
            /// Sets `betas` of the defaults and of every group.
            ///
            /// # Panics
            ///
            /// Panics unless both are in `[0, 1)`.
            #[must_use]
            #[track_caller]
            pub fn betas(mut self, betas: (f64, f64)) -> $name {
                self.0.groups.set(|o| o.betas = betas);
                self
            }

            /// Sets `eps` of the defaults and of every group.
            ///
            /// # Panics
            ///
            /// Panics if `eps` is negative or NaN.
            #[must_use]
            #[track_caller]
            pub fn eps(mut self, eps: f64) -> $name {
                self.0.groups.set(|o| o.eps = eps);
                self
            }

            /// Sets `weight_decay` of the defaults and of every group.
            ///
            /// # Panics
            ///
            /// Panics if `weight_decay` is negative or NaN.
            #[must_use]
            #[track_caller]
            pub fn weight_decay(mut self, weight_decay: f64) -> $name {
                self.0.groups.set(|o| o.weight_decay = weight_decay);
                self
            }

            /// Sets `amsgrad` of the defaults and of every group.
            #[must_use]
            #[track_caller]
            pub fn amsgrad(mut self, amsgrad: bool) -> $name {
                self.0.groups.set(|o| o.amsgrad = amsgrad);
                self
            }

            /// Sets `maximize` of the defaults and of every group.
            #[must_use]
            #[track_caller]
            pub fn maximize(mut self, maximize: bool) -> $name {
                self.0.groups.set(|o| o.maximize = maximize);
                self
            }

            /// The state of `param` (`state[param]`), if it is a parameter of this optimizer
            /// that has state.
            #[track_caller]
            pub fn state(&self, param: &Parameter) -> Option<&AdamState> {
                self.0.groups.state(param)
            }
        }

        impl Optimizer for $name {
            fn step(&mut self) {
                self.0.step();
            }

            fn zero_grad(&mut self) {
                self.0.groups.zero_grad();
            }

            fn num_param_groups(&self) -> usize {
                self.0.groups.groups.len()
            }

            fn params(&self, group: usize) -> &[Parameter] {
                self.0.groups.group(group).params()
            }

            fn lr(&self, group: usize) -> f64 {
                self.0.groups.lr(group)
            }

            fn set_lr(&mut self, group: usize, lr: f64) {
                self.0.groups.set_lr(group, lr);
            }

            fn state_dict(&self) -> OptimizerStateDict {
                self.0.groups.state_dict(&self.0.schema())
            }

            fn load_state_dict(
                &mut self,
                state_dict: &OptimizerStateDict,
            ) -> Result<(), LoadError> {
                let schema = self.0.schema();
                self.0.groups.load_state_dict(&schema, state_dict)
            }
        }

        impl TypedOptimizer for $name {
            type Options = AdamOptions;

            fn defaults(&self) -> &AdamOptions {
                &self.0.groups.defaults
            }

            fn param_groups(&self) -> &[ParamGroup<AdamOptions>] {
                &self.0.groups.groups
            }

            fn options_mut(&mut self, index: usize) -> &mut AdamOptions {
                self.0.groups.options_mut(index)
            }

            fn add_param_group(&mut self, group: ParamGroup<AdamOptions>) {
                self.0.groups.add(group);
            }
        }
    };
}

/// Adam, PyTorch's `torch.optim.Adam`, with the weight decay as an L2 penalty added to the
/// gradient.
///
/// For each parameter with a gradient `g`, a step computes, in this order: `g = -g` if
/// `maximize`; `step += 1`; `g = g + weight_decay * p` if `weight_decay` isn't 0;
/// `exp_avg = lerp(exp_avg, g, 1 - beta1)`;
/// `exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g * g`; with `amsgrad`,
/// `max_exp_avg_sq = maximum(max_exp_avg_sq, exp_avg_sq)`, which replaces `exp_avg_sq` in the
/// denominator; `denom = sqrt(exp_avg_sq) / sqrt(1 - beta2^step) + eps`; and
/// `p = p - lr / (1 - beta1^step) * exp_avg / denom`. The bias corrections are computed in
/// `f64` from the `step` of the state.
///
/// A parameter has state (an [`AdamState`]) once it had a step with a gradient.
/// `max_exp_avg_sq` is created with the state if `amsgrad` is on then; turning `amsgrad` on
/// for a parameter whose state was created without it makes [`step`](Optimizer::step) panic.
///
/// ```
/// use homura::nn::Parameter;
/// use homura::optim::{Adam, Optimizer};
///
/// let p = Parameter::new(homura::tensor([1.0, 2.0]));
/// let mut opt = Adam::new([p.clone()], 0.1).betas((0.8, 0.99));
/// p.set_grad(Some(homura::tensor([1.0, -1.0])));
/// opt.step();
/// // The first step moves each element by about `lr`.
/// assert_eq!(p.to_vec::<f32>(), [0.9, 2.1]);
/// assert_eq!(opt.state(&p).unwrap().step().item::<f32>(), 1.0);
/// ```
#[derive(Debug)]
pub struct Adam(Inner);

impl Adam {
    /// An optimizer of `params` with learning rate `lr` and the default options, as one
    /// parameter group (`Adam(params, lr)`).
    ///
    /// # Panics
    ///
    /// Panics if `params` is empty or `lr` is negative or NaN, with PyTorch's messages, or if a
    /// parameter is not a leaf or appears twice.
    #[track_caller]
    pub fn new(params: impl IntoIterator<Item = Parameter>, lr: f64) -> Adam {
        Adam(Inner::new(params, lr, 0.0, false))
    }
}

adam_methods!(Adam);

/// AdamW, PyTorch's `torch.optim.AdamW`: [`Adam`] with decoupled weight decay.
///
/// A step is Adam's, except that the weight decay is not added to the gradient: the parameter
/// is first scaled, `p = p * (1 - lr * weight_decay)`, after `step += 1`. `weight_decay`
/// defaults to 0.01.
///
/// ```
/// use homura::nn::Parameter;
/// use homura::optim::{AdamW, Optimizer};
///
/// let p = Parameter::new(homura::tensor([1.0, 2.0]));
/// let mut opt = AdamW::new([p.clone()], 0.1).weight_decay(0.5);
/// p.set_grad(Some(homura::tensor([0.0, 0.0])));
/// opt.step();
/// // A zero gradient still decays the parameter.
/// assert_eq!(p.to_vec::<f32>(), [0.95, 1.9]);
/// ```
#[derive(Debug)]
pub struct AdamW(Inner);

impl AdamW {
    /// An optimizer of `params` with learning rate `lr` and the default options, as one
    /// parameter group (`AdamW(params, lr)`).
    ///
    /// # Panics
    ///
    /// Panics if `params` is empty or `lr` is negative or NaN, with PyTorch's messages, or if a
    /// parameter is not a leaf or appears twice.
    #[track_caller]
    pub fn new(params: impl IntoIterator<Item = Parameter>, lr: f64) -> AdamW {
        AdamW(Inner::new(params, lr, 0.01, true))
    }
}

adam_methods!(AdamW);
