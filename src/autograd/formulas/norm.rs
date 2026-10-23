//! The nodes of `nn::functional::layer_norm` and `batch_norm`, which are single nodes in
//! PyTorch.

use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::autograd::saved::SavedId;
use crate::{DType, Shape, Tensor};

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
    #[track_caller]
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
#[track_caller]
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

/// What a batch norm computes with besides its tensors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BatchNorm {
    /// Whether the batch statistics normalize the input; otherwise the running ones do.
    pub training: bool,
    pub eps: f64,
    /// The input's dtype.
    pub dtype: DType,
    /// The dtype the output is computed in: `F32` for half precision inputs.
    pub opmath: DType,
    /// The dtype of the parameters, of the running statistics and of PyTorch's saved batch
    /// statistics: `F32` for a half precision input with `F32` parameters, otherwise the
    /// input's.
    pub param_dtype: DType,
}

impl BatchNorm {
    /// The dimensions the statistics are taken over: all but the channels, dimension 1.
    pub fn axes(num_dims: usize) -> Vec<i64> {
        std::iter::once(0).chain(2..num_dims as i64).collect()
    }

    /// The shape `[1, C, 1, ...]` with the dimensions of `x`, which a per-channel tensor of `C`
    /// elements (of any shape) is reshaped to.
    pub fn channel_shape(x: &Tensor) -> Shape {
        let mut shape = vec![1; x.num_dims()];
        shape[1] = x.shape()[1];
        Shape::new(&shape)
    }

    /// The number of elements per channel of `x`.
    pub fn count(x: &Tensor) -> usize {
        let shape = x.shape();
        shape[0] * shape[2..].iter().product::<usize>()
    }

    /// The batch mean of `x`, which has the dtype `opmath`, and the sum of the squared
    /// deviations from it, `Σ (x - mean)²`, both per channel and in `[1, C, 1, ...]`. As in
    /// PyTorch's CPU kernel the deviations are computed in `opmath` and summed with more
    /// precision.
    pub fn moments(&self, x: &Tensor) -> (Tensor, Tensor) {
        let axes = Self::axes(x.num_dims());
        let mean = x.mean_dim(axes.clone(), true);
        let deviation = x.sub(&mean);
        let var_sum = deviation.mul(&deviation).sum_dim(axes, true);
        (mean, var_sum)
    }

    /// The batch statistics as PyTorch saves them and normalizes with them: the mean and
    /// `invstd = 1 / √(var + eps)` with the biased variance `var_sum / n`, divided in `opmath`,
    /// and the root taken in PyTorch's accumulation type (`F64`, or `F32` for half precision
    /// inputs); both rounded to `param_dtype` and converted back to `opmath`.
    pub fn saved_stats(&self, mean: &Tensor, var_sum: &Tensor, n: usize) -> (Tensor, Tensor) {
        let acc = if self.opmath == self.dtype {
            DType::F64
        } else {
            DType::F32
        };
        let invstd = var_sum
            .div(n as f64)
            .to(acc)
            .add(self.eps)
            .sqrt()
            .reciprocal();
        let round = |t: Tensor| t.to(self.param_dtype).to(self.opmath);
        (round(mean.clone()), round(invstd))
    }
}

/// The running statistics a batch norm in evaluation mode normalized with; saved, so that
/// updating them before the backward pass is an error.
pub(crate) struct Running<'a> {
    pub mean: &'a Tensor,
    pub var: &'a Tensor,
}

/// Records `out = batch_norm(input, weight, bias)`. `weight` is saved, as in PyTorch, and the
/// running statistics in evaluation mode (`running`), which the gradients use; the batch
/// statistics are computed again from the saved input.
#[track_caller]
pub(crate) fn batch_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running: Option<Running<'_>>,
    args: BatchNorm,
    out: &Tensor,
) {
    record(&[Some(input), weight, bias], &[out], |b| {
        NativeBatchNormBackward0 {
            input: b.save(input),
            weight: weight.map(|w| b.save(w)),
            bias_shape: bias.map(|t| t.shape()),
            running: running.map(|r| (b.save(r.mean), b.save(r.var))),
            args,
        }
    });
}

/// `batch_norm`: PyTorch's CPU `batch_norm_backward`, in its order of operations, in `opmath`.
/// With the per-channel sums `sum = Σ g` and `dotp = Σ (x − mean)·g` over the `n` elements of a
/// channel, the input gets `(g − sum/n − (x − mean)·k)·invstd·w` with `k = dotp·invstd²/n` in
/// training, and `g·invstd·w` in evaluation mode; the weight gets `dotp·invstd`, and the bias
/// `sum`. The gradients of the parameters have the parameters' shapes, which need only have `C`
/// elements.
///
/// In training the statistics are computed again from the saved input, as the forward pass did:
/// with `create_graph` the computation is recorded, so that the gradients depend on the input
/// through them too, as PyTorch's `batchnorm_double_backward` describes. In evaluation mode
/// `invstd = 1 / √(running_var + eps)` is computed in `F64`, as PyTorch's backward kernel does.
struct NativeBatchNormBackward0 {
    input: SavedId,
    weight: Option<SavedId>,
    bias_shape: Option<Shape>,
    running: Option<(SavedId, SavedId)>,
    args: BatchNorm,
}

impl Node for NativeBatchNormBackward0 {
    fn name(&self) -> &'static str {
        "NativeBatchNormBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = super::only(grads);
        let args = self.args;
        let input = ctx.unpack(self.input);
        let weight = self.weight.map(|w| ctx.unpack(w));
        let running = self.running.map(|(m, v)| (ctx.unpack(m), ctx.unpack(v)));
        let (dtype, opmath) = (args.dtype, args.opmath);
        let axes = BatchNorm::axes(input.num_dims());
        let channels = BatchNorm::channel_shape(&input);
        let g = g.to(opmath);
        let w = weight.as_ref().map(|w| w.reshape(&channels).to(opmath));
        let sum = g.sum_dim(axes.clone(), true);
        let grad_bias = (ctx.needs_input_grad(2) && self.bias_shape.is_some()).then(|| {
            sum.reshape(self.bias_shape.as_ref().unwrap())
                .to(args.param_dtype)
        });
        let needs_weight = ctx.needs_input_grad(1) && weight.is_some();
        if !ctx.needs_input_grad(0) && !needs_weight {
            // Only the bias's gradient, which doesn't need the input or its statistics.
            return vec![None, None, grad_bias];
        }
        let x = input.to(opmath);
        let n = BatchNorm::count(&input);
        let weight_grad = |t: Tensor| {
            t.reshape(weight.as_ref().unwrap().shape())
                .to(args.param_dtype)
        };
        if n == 0 || channels[1] == 0 {
            // No elements: PyTorch's empty input path, `out = input·w + b` in effect.
            let grad_input = ctx.needs_input_grad(0).then(|| {
                match &w {
                    Some(w) => g.mul(w),
                    None => g.clone(),
                }
                .to(dtype)
            });
            let grad_weight = needs_weight.then(|| weight_grad(g.mul(&x).sum_dim(axes, true)));
            return vec![grad_input, grad_weight, grad_bias];
        }
        let (mean, invstd) = match &running {
            None => {
                let (mean, var_sum) = args.moments(&x);
                args.saved_stats(&mean, &var_sum, n)
            }
            Some((running_mean, running_var)) => {
                let invstd = running_var
                    .reshape(&channels)
                    .to(DType::F64)
                    .add(args.eps)
                    .sqrt()
                    .reciprocal();
                (
                    running_mean.reshape(&channels).to(opmath),
                    invstd.to(opmath),
                )
            }
        };
        let centered = x.sub(&mean);
        let dotp = centered.mul(&g).sum_dim(axes, true);
        let grad_input = ctx.needs_input_grad(0).then(|| {
            let scaled = if args.training {
                let k = dotp.mul(&invstd).mul(&invstd).div(n as f64);
                let grad_mean = sum.div(n as f64);
                g.sub(&grad_mean).sub(centered.mul(&k)).mul(&invstd)
            } else {
                g.mul(&invstd)
            };
            match &w {
                Some(w) => scaled.mul(w),
                None => scaled,
            }
            .to(dtype)
        });
        let grad_weight = needs_weight.then(|| weight_grad(dotp.mul(&invstd)));
        vec![grad_input, grad_weight, grad_bias]
    }
}
