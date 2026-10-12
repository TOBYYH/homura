//! Elementwise unary operations.

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI, PI, SQRT_2};

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::ops::{GeluApproximate, UnaryOp};
use crate::{Tensor, where_};

/// Records the backward node of `out = op(x)`. As in PyTorch, each node saves what its formula
/// needs: the input, or the result where the derivative is a function of it.
pub(crate) fn unary(op: UnaryOp, x: &Tensor, out: &Tensor) {
    use UnaryOp::*;
    let inputs = [Some(x)];
    match op {
        Neg => record(&inputs, &[out], |_| NegBackward0),
        Sign => record(&inputs, &[out], |_| SignBackward0),
        Abs => record(&inputs, &[out], |b| AbsBackward0 { x: b.save(x) }),
        Log => record(&inputs, &[out], |b| LogBackward0 { x: b.save(x) }),
        Sin => record(&inputs, &[out], |b| SinBackward0 { x: b.save(x) }),
        Cos => record(&inputs, &[out], |b| CosBackward0 { x: b.save(x) }),
        Erf => record(&inputs, &[out], |b| ErfBackward0 { x: b.save(x) }),
        Silu => record(&inputs, &[out], |b| SiluBackward0 { x: b.save(x) }),
        Gelu | GeluTanh => record(&inputs, &[out], |b| GeluBackward0 {
            x: b.save(x),
            approximate: if op == Gelu {
                GeluApproximate::None
            } else {
                GeluApproximate::Tanh
            },
        }),
        Exp => record(&inputs, &[out], |b| ExpBackward0 {
            result: b.save_output(out, 0),
        }),
        Relu => record(&inputs, &[out], |b| ReluBackward0 {
            result: b.save_output(out, 0),
        }),
        Sqrt => record(&inputs, &[out], |b| SqrtBackward0 {
            result: b.save_output(out, 0),
        }),
        Rsqrt => record(&inputs, &[out], |b| RsqrtBackward0 {
            result: b.save_output(out, 0),
        }),
        Reciprocal => record(&inputs, &[out], |b| ReciprocalBackward0 {
            result: b.save_output(out, 0),
        }),
        Tanh => record(&inputs, &[out], |b| TanhBackward0 {
            result: b.save_output(out, 0),
        }),
        Sigmoid => record(&inputs, &[out], |b| SigmoidBackward0 {
            result: b.save_output(out, 0),
        }),
        LogicalNot => {}
    }
}

/// `-x`: the gradient is `-g`.
struct NegBackward0;

impl Node for NegBackward0 {
    fn name(&self) -> &'static str {
        "NegBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(only(grads).neg())]
    }
}

/// `sign(x)`: the gradient is zero, even where `g` is infinite or NaN, as in PyTorch.
struct SignBackward0;

impl Node for SignBackward0 {
    fn name(&self) -> &'static str {
        "SignBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(crate::zeros_like(&only(grads)))]
    }
}

/// Defines the nodes of unary operations whose formula needs one saved tensor, the input `x` or
/// the `result`: `Node(saved) => |g| formula` computes the gradient of the input from the
/// gradient `g` of the output and the unpacked saved tensor.
macro_rules! unary_nodes {
    ($($(#[$doc:meta])* $Node:ident($saved:ident) => |$g:ident| $formula:expr;)*) => {$(
        $(#[$doc])*
        struct $Node {
            $saved: SavedId,
        }

        impl Node for $Node {
            fn name(&self) -> &'static str {
                stringify!($Node)
            }

            fn apply(
                &self,
                ctx: &ApplyContext<'_>,
                grads: Vec<Option<Tensor>>,
            ) -> Vec<Option<Tensor>> {
                let $g = only(grads);
                let $saved = ctx.unpack(self.$saved);
                vec![Some($formula)]
            }
        }
    )*};
}

// The formulas follow PyTorch's `derivatives.yaml` and backward kernels, including their order
// of operations, so that they agree with PyTorch's up to rounding, and exactly at zeros,
// infinities and NaN.
unary_nodes! {
    /// `e^x`: `g * e^x`, from the saved result.
    ExpBackward0(result) => |g| g.mul(&result);

    /// `|x|`: `g * sign(x)`. A product, as in PyTorch: zero where `x` is zero, except that an
    /// infinite or NaN `g` gives NaN there.
    AbsBackward0(x) => |g| g.mul(x.sign());

    /// `relu(x)`: `g` where the result is positive or NaN, zero where it is zero, even for an
    /// infinite or NaN `g`, like PyTorch's `threshold_backward(g, result, 0)`. The zeros are
    /// taken from the result, which is exactly zero there, so that as in PyTorch the gradient
    /// depends on the result and its own derivative with respect to `x` is zero, not missing.
    ReluBackward0(result) => |g| where_(&result.less_equal(0.0), &result, &g);

    /// `ln(x)`: `g / x`, infinite at zero.
    LogBackward0(x) => |g| g.div(&x);

    /// `√x`: `g / (2 * √x)`, from the saved result; infinite at zero.
    SqrtBackward0(result) => |g| g.div(result.mul(2.0));

    /// `1 / √x`: `-0.5 * g * result³`. PyTorch's `result.pow(3)` multiplies the result by
    /// itself twice, as here.
    RsqrtBackward0(result) => |g| g.mul(-0.5).mul(result.mul(&result).mul(&result));

    /// `1 / x`: `-g * result²`.
    ReciprocalBackward0(result) => |g| g.neg().mul(result.mul(&result));

    /// `sin(x)`: `g * cos(x)`.
    SinBackward0(x) => |g| g.mul(x.cos());

    /// `cos(x)`: `g * -sin(x)`.
    CosBackward0(x) => |g| g.mul(x.sin().neg());

    /// `erf(x)`: `2/√π * e^(-x²) * g`. `x * x` equals PyTorch's `x.pow(2)` and only needs the
    /// multiplication's formula.
    ErfBackward0(x) => |g| x.mul(&x).neg().exp().mul(2.0 / PI.sqrt()).mul(&g);

    /// `tanh(x)`: `g * (1 - result²)`, PyTorch's `tanh_backward`.
    TanhBackward0(result) => |g| g.mul(1.0 - result.mul(&result));

    /// `σ(x)`: `g * (1 - result) * result`, PyTorch's `sigmoid_backward`.
    SigmoidBackward0(result) => |g| g.mul(1.0 - &result).mul(&result);

    /// `x * σ(x)`: `g * σ(x) * (1 + x * (1 - σ(x)))`, PyTorch's `silu_backward`.
    SiluBackward0(x) => |g| {
        let sigmoid = x.sigmoid();
        g.mul(&sigmoid).mul(x.mul(1.0 - &sigmoid).add(1.0))
    };
}

/// `gelu(x)`: PyTorch's `gelu_backward`, written with tensor operations so that it can be
/// differentiated again. Exact: `g * (Φ(x) + x * φ(x))`. With the tanh approximation
/// `0.5 * x * (1 + tanh(inner))`, `inner = √(2/π) * (x + 0.044715 * x³)`: `g` times the
/// derivative of the two factors, `0.5 * (1 + tanh(inner))` and
/// `0.5 * x * (1 - tanh²(inner)) * inner'`.
struct GeluBackward0 {
    x: SavedId,
    approximate: GeluApproximate,
}

impl Node for GeluBackward0 {
    fn name(&self) -> &'static str {
        "GeluBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let x = ctx.unpack(self.x);
        let derivative = match self.approximate {
            GeluApproximate::None => {
                // φ(x) = e^(-x²/2) / √(2π).
                let frac_1_sqrt_2pi = FRAC_2_SQRT_PI * FRAC_1_SQRT_2 * 0.5;
                let cdf = x.mul(FRAC_1_SQRT_2).erf().add(1.0).mul(0.5);
                let pdf = x.mul(&x).mul(-0.5).exp().mul(frac_1_sqrt_2pi);
                cdf.add(x.mul(&pdf))
            }
            GeluApproximate::Tanh => {
                // √(2/π), computed like PyTorch's constant.
                let beta = SQRT_2 * FRAC_2_SQRT_PI * 0.5;
                let kappa = 0.044715;
                let x_sq = x.mul(&x);
                let x_cube = x_sq.mul(&x);
                let tanh_inner = x.add(x_cube.mul(kappa)).mul(beta).tanh();
                let left = x.mul(0.5);
                let right = tanh_inner.add(1.0);
                let left_derivative = right.mul(0.5);
                let tanh_derivative = 1.0 - tanh_inner.mul(&tanh_inner);
                let inner_derivative = x_sq.mul(3.0 * kappa).add(1.0).mul(beta);
                let right_derivative = left.mul(&tanh_derivative).mul(&inner_derivative);
                left_derivative.add(&right_derivative)
            }
        };
        vec![Some(g.mul(derivative))]
    }
}
