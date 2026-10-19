//! The node of `nn::functional::layer_norm`, which is a single node in PyTorch.

use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::{DType, Tensor};

/// What a layer norm computes with besides its tensors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerNorm {
    /// The first normalized dimension: the input's statistics are taken over this one and the
    /// dimensions after it.
    pub axis: usize,
    pub eps: f64,
    /// The dtype the statistics are computed in: `F32` for half precision inputs.
    pub opmath: DType,
    /// The dtype of the parameters and of PyTorch's saved statistics: `F32` for a half precision
    /// input with `F32` parameters, otherwise the input's.
    pub param_dtype: DType,
}

impl LayerNorm {
    /// The normalized dimensions of a tensor with `dims` dimensions.
    fn dims(&self, dims: usize) -> Vec<i64> {
        (self.axis as i64..dims as i64).collect()
    }

    /// The mean and `rstd = 1 / √(var + eps)` (biased variance) of `x`, which has the dtype
    /// `opmath`, over the normalized dimensions, which are kept. The forward pass uses them.
    pub fn stats(&self, x: &Tensor) -> (Tensor, Tensor) {
        let dims = self.dims(x.num_dims());
        let mean = x.mean_dim(dims.clone(), true);
        let rstd = x.var_dim(dims, 0, true).add(self.eps).sqrt().reciprocal();
        (mean, rstd)
    }

    /// [`stats`](LayerNorm::stats) as PyTorch saves them for the backward pass: rounded to
    /// `param_dtype` (and converted back to `opmath`), which changes them for half precision
    /// parameters.
    fn saved_stats(&self, x: &Tensor) -> (Tensor, Tensor) {
        let round = |t: Tensor| t.to(self.param_dtype).to(self.opmath);
        let (mean, rstd) = self.stats(x);
        (round(mean), round(rstd))
    }
}

/// Records `out = layer_norm(input, weight, bias)`. `weight` and `bias` are saved, as in
/// PyTorch, so that changing them before the backward pass is an error.
pub(crate) fn layer_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    args: LayerNorm,
    out: &Tensor,
) {
    record(&[Some(input), weight, bias], &[out], |b| {
        NativeLayerNormBackward0 {
            input: b.save(input),
            weight: weight.map(|w| b.save(w)),
            bias: bias.map(|t| b.save(t)),
            args,
        }
    });
}

/// `layer_norm`: PyTorch's CPU `layer_norm_backward_frame`, in its order of operations, in
/// `opmath`. With `a = rstd`, the sums `ds = Σ g·x·w` and `db = Σ g·w` over the normalized
/// elements of a row, `b = (db·mean − ds)·a³/N` and `c = −b·mean − db·a/N`, the input gets
/// `a·g·w + b·x + c`; the weight gets the sum over the rows of `g·(a·x − a·mean)`, and the bias
/// that of `g`.
///
/// The statistics are computed again from the saved input, as the forward pass did, instead of
/// being saved: with `create_graph` the computation is recorded, so that the gradients depend on
/// the input through them too, as PyTorch's `layer_norm_double_backward` describes.
struct NativeLayerNormBackward0 {
    input: SavedId,
    weight: Option<SavedId>,
    bias: Option<SavedId>,
    args: LayerNorm,
}

impl Node for NativeLayerNormBackward0 {
    fn name(&self) -> &'static str {
        "NativeLayerNormBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = super::only(grads);
        let args = self.args;
        let input = ctx.unpack(self.input);
        let weight = self.weight.map(|w| ctx.unpack(w));
        // Unpacked only for its version check.
        let _bias = self.bias.map(|b| ctx.unpack(b));
        let (dtype, opmath) = (input.dtype(), args.opmath);
        let g = g.to(opmath);
        // The sums over the rows, for the parameters: none when the input has only the
        // normalized dimensions.
        let batch: Vec<i64> = (0..args.axis as i64).collect();
        let sum_rows = |t: Tensor| {
            if batch.is_empty() {
                t
            } else {
                t.sum_dim(batch.clone(), false)
            }
        };
        let grad_bias = (ctx.needs_input_grad(2) && self.bias.is_some())
            .then(|| sum_rows(g.clone()).to(args.param_dtype));
        let needs_weight = ctx.needs_input_grad(1) && weight.is_some();
        if !ctx.needs_input_grad(0) && !needs_weight {
            // Only the bias's gradient, which doesn't need the input or its statistics.
            return vec![None, None, grad_bias];
        }
        let x = input.to(opmath);
        let w = weight.as_ref().map(|w| w.to(opmath));
        let (mean, rstd) = args.saved_stats(&x);
        let dims = args.dims(x.num_dims());
        let n: usize = x.shape()[args.axis..].iter().product();
        let scale = 1.0 / n as f64;
        let half = matches!(dtype, DType::F16 | DType::BF16);

        let grad_input = ctx.needs_input_grad(0).then(|| {
            let (ds, db) = match &w {
                // PyTorch multiplies `g·x·w` in this order for single and double precision,
                // and as `x·(g·w)` in half precision.
                Some(w) => {
                    let gw = g.mul(w);
                    let gxw = if half { x.mul(&gw) } else { g.mul(&x).mul(w) };
                    (
                        gxw.sum_dim(dims.clone(), true),
                        gw.sum_dim(dims.clone(), true),
                    )
                }
                None => (
                    g.mul(&x).sum_dim(dims.clone(), true),
                    g.sum_dim(dims.clone(), true),
                ),
            };
            let b = db
                .mul(&mean)
                .sub(&ds)
                .mul(&rstd)
                .mul(&rstd)
                .mul(&rstd)
                .mul(scale);
            let c = b.neg().mul(&mean).sub(db.mul(&rstd).mul(scale));
            let ag = rstd.mul(&g);
            let ag = match &w {
                Some(w) => ag.mul(w),
                None => ag,
            };
            ag.add(b.mul(&x)).add(&c).to(dtype)
        });
        let grad_weight = needs_weight.then(|| {
            let shift = rstd.neg().mul(&mean);
            sum_rows(g.mul(rstd.mul(&x).add(&shift))).to(args.param_dtype)
        });
        vec![grad_input, grad_weight, grad_bias]
    }
}
