use crate::autograd::formulas::norm::{self as formulas, BatchNorm, LayerNorm, Running};
use crate::ops::{check_no_overlap, check_writable};
use crate::{DType, IntoShape, Tensor, no_grad};

/// The dtype of half precision computations: `F32` for `F16` and `BF16`.
fn opmath(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        _ => dtype,
    }
}

/// Panics if `normalized_shape` is empty.
#[track_caller]
fn check_not_empty(normalized_shape: &[usize]) {
    if normalized_shape.is_empty() {
        panic!(
            "Expected normalized_shape to be at least 1-dimensional, i.e., containing at least \
             one element, but got normalized_shape = []"
        );
    }
}

/// Panics unless `input` ends with the dimensions `normalized_shape`; `star` is what PyTorch's
/// message puts after the `*` of the expected shape.
#[track_caller]
fn check_input_shape(input: &Tensor, normalized_shape: &[usize], star: &str) {
    let shape = input.shape();
    if !shape.ends_with(normalized_shape) {
        let sizes: Vec<String> = normalized_shape.iter().map(|n| n.to_string()).collect();
        panic!(
            "Given normalized_shape={normalized_shape:?}, expected input with shape \
             [*{star}{}], but got input of size{:?}",
            sizes.join(", "),
            &shape[..]
        );
    }
}

/// Panics unless the parameter `t` has the shape `normalized_shape`.
#[track_caller]
fn check_parameter(name: &str, t: Option<&Tensor>, normalized_shape: &[usize]) {
    if let Some(t) = t
        && t.shape()[..] != *normalized_shape
    {
        panic!(
            "Expected {name} to be of same shape as normalized_shape, but got {name} of shape \
             {:?} and normalized_shape = {normalized_shape:?}",
            &t.shape()[..]
        );
    }
}

/// Layer normalization, PyTorch's `F.layer_norm`: each row of `input`, the elements of its last
/// dimensions `normalized_shape`, is normalized to zero mean and unit variance,
/// `(x - mean) / √(var + eps)` with the biased variance, then multiplied by `weight` and shifted
/// by `bias`, which have the shape `normalized_shape`. PyTorch's default `eps` is `1e-5`.
///
/// It is a single node, `NativeLayerNormBackward0`, as in PyTorch, with PyTorch's CPU formulas
/// for the gradients of `input`, `weight` and `bias`; their gradients are differentiable in turn.
/// Half precision inputs are computed in `F32` and rounded once; they may have `F32` parameters.
///
/// ```
/// use homura::nn::functional::layer_norm;
///
/// let x = homura::tensor([[1.0f32, 3.0], [0.0, 4.0]]);
/// let y = layer_norm(&x, [2], None, None, 0.0);
/// assert_eq!(y.to_vec::<f32>(), [-1.0, 1.0, -1.0, 1.0]);
/// ```
///
/// # Panics
///
/// With PyTorch's messages: if the dtypes differ, other than for a half precision input with
/// `F32` parameters, if `normalized_shape` is empty, if a parameter doesn't have the shape
/// `normalized_shape` or `input` doesn't end with it, and if `input` isn't floating point.
#[track_caller]
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: impl IntoShape,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Tensor {
    let normalized_shape = normalized_shape.into_shape();
    let dtype = input.dtype();
    let params = [weight, bias];
    let first = params.iter().flatten().next().map(|p| p.dtype());
    // PyTorch's `is_mixed_type` and `check_mixed_data_type`.
    let mixed = first.is_some_and(|d| d != dtype);
    for p in params.iter().flatten() {
        if mixed && p.dtype() != DType::F32 {
            panic!("mixed dtype (CPU): expect parameter to have scalar type of Float");
        }
    }
    if mixed && !matches!(dtype, DType::F16 | DType::BF16) {
        panic!("mixed dtype (CPU): all inputs must share same datatype.");
    }
    check_not_empty(&normalized_shape);
    check_parameter("weight", weight, &normalized_shape);
    check_parameter("bias", bias, &normalized_shape);
    check_input_shape(input, &normalized_shape, ", ");
    if !dtype.is_floating_point() {
        panic!("layer_norm is not implemented for {dtype} tensors");
    }
    let param_dtype = if mixed { DType::F32 } else { dtype };
    if let Some(p) = params.iter().flatten().find(|p| p.dtype() != param_dtype) {
        panic!("expected scalar type {param_dtype} but found {}", p.dtype());
    }
    let args = LayerNorm {
        axis: input.num_dims() - normalized_shape.len(),
        eps,
        opmath: opmath(dtype),
        param_dtype,
    };
    let out = {
        let _guard = no_grad();
        let x = input.to(args.opmath);
        let (mean, rstd) = args.stats(&x);
        let normalized = if args.opmath == dtype {
            x.sub(&mean).mul(&rstd)
        } else {
            // PyTorch's half precision kernel: `x * rstd - rstd * mean`.
            x.mul(&rstd).add(rstd.neg().mul(&mean))
        };
        let scaled = match weight {
            Some(w) => normalized.mul(w.to(args.opmath)),
            None => normalized,
        };
        // Without a bias, PyTorch adds zero, which turns -0 into 0.
        let shifted = match bias {
            Some(b) => scaled.add(b.to(args.opmath)),
            None => scaled.add(0.0),
        };
        shifted.to(dtype)
    };
    formulas::layer_norm(input, weight, bias, args, &out);
    out
}

/// Root mean square normalization, PyTorch's `F.rms_norm`: each row of `input`, the elements of
/// its last dimensions `normalized_shape`, is divided by `√(mean(x²) + eps)` and multiplied by
/// `weight`, which has the shape `normalized_shape`. Without `eps` (PyTorch's default `None`),
/// it is the machine epsilon of the computation: [`f32::EPSILON`], or [`f64::EPSILON`] for `F64`
/// inputs.
///
/// It records the operations of PyTorch's CPU implementation, `x * rsqrt(mean(x²) + eps) *
/// weight`, computed in `F32` for half precision inputs; the result has the dtype of `input`,
/// also for a `weight` of another dtype (PyTorch warns about one).
///
/// ```
/// use homura::nn::functional::rms_norm;
///
/// let x = homura::tensor([[3.0f32, 4.0]]);
/// let y = rms_norm(&x, [2], None, Some(0.0));
/// let r = 1.0 / 12.5f32.sqrt();
/// assert_eq!(y.to_vec::<f32>(), [3.0 * r, 4.0 * r]);
/// ```
///
/// # Panics
///
/// With PyTorch's messages: if `normalized_shape` is empty, if `weight` doesn't have the shape
/// `normalized_shape`, if `input` doesn't end with it, if `input` isn't floating point, and like
/// the multiplication by `weight` (which must not be complex).
#[track_caller]
pub fn rms_norm(
    input: &Tensor,
    normalized_shape: impl IntoShape,
    weight: Option<&Tensor>,
    eps: Option<f64>,
) -> Tensor {
    let normalized_shape = normalized_shape.into_shape();
    check_not_empty(&normalized_shape);
    check_parameter("weight", weight, &normalized_shape);
    let (dims, k) = (input.num_dims(), normalized_shape.len());
    if dims < k {
        panic!("Input tensor must have at least {k} dimensions, but got {dims}");
    }
    check_input_shape(input, &normalized_shape, "");
    let dtype = input.dtype();
    if !dtype.is_floating_point() {
        panic!("rms_norm is not implemented for {dtype} tensors");
    }
    let opmath = opmath(dtype);
    let eps = eps.unwrap_or(if opmath == DType::F64 {
        f64::EPSILON
    } else {
        f32::EPSILON as f64
    });
    let x = input.to(opmath);
    let reduced: Vec<i64> = (0..k).map(|i| (dims - i - 1) as i64).collect();
    let rstd = x.pow(2).mean_dim(reduced, true).add(eps).rsqrt();
    let y = x.mul(&rstd);
    let y = match weight {
        Some(w) => y.mul(w),
        None => y,
    };
    y.to(dtype)
}

/// The options of [`batch_norm`], with PyTorch's defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchNormOptions {
    /// Whether to normalize with the statistics of the batch and update the running statistics
    /// (when given), as in training, or with the running statistics; `false` by default.
    pub training: bool,
    /// The weight of the batch statistics in the update of the running ones, `0.1` by default:
    /// `running = (1 - momentum)·running + momentum·batch`.
    pub momentum: f64,
    /// Added to the variance before its square root is taken, `1e-5` by default.
    pub eps: f64,
}

impl Default for BatchNormOptions {
    fn default() -> Self {
        BatchNormOptions {
            training: false,
            momentum: 0.1,
            eps: 1e-5,
        }
    }
}

/// Panics unless the per-channel tensor `t` has `channels` elements.
#[track_caller]
fn check_channels(name: &str, t: Option<&Tensor>, channels: usize) {
    if let Some(t) = t
        && t.num_elements() != channels
    {
        panic!(
            "{name} should contain {channels} elements not {}",
            t.num_elements()
        );
    }
}

/// Batch normalization, PyTorch's `F.batch_norm`: each channel of `input` (dimension 1 of an
/// input with at least 2 dimensions, `[N, C, ...]`) is normalized to zero mean and unit
/// variance, `(x - mean) / √(var + eps)`, then multiplied by `weight` and shifted by `bias`. The
/// per-channel tensors `running_mean`, `running_var`, `weight` and `bias` need `C` elements, in
/// any shape.
///
/// In training (`options.training`) the mean and the biased variance are those of the batch,
/// over all dimensions but the channels; the running statistics, if given, are then updated in
/// place with `momentum` and the unbiased variance, which increments their versions (unlike in
/// PyTorch). The update is computed in `F64` for `F64` statistics and in `F32` otherwise (PyTorch
/// computes `BF16` statistics in `BF16`). Otherwise the running statistics normalize, and are
/// saved for the backward pass: updating them before it is an error.
///
/// It is a single node, `NativeBatchNormBackward0`, as in PyTorch, with PyTorch's CPU formulas
/// for the gradients of `input`, `weight` and `bias`, which are differentiable in turn (their
/// nodes are those of the operations, not PyTorch's `NativeBatchNormBackwardBackward0`). The
/// output is computed as `x·α + β` with `α = invstd·weight` and `β = bias − mean·α` in the
/// input's dtype, or in `F32` for half precision inputs, which may have `F32` parameters and
/// running statistics. The sums accumulate in `F64`, so results can differ from PyTorch's in
/// the last bits.
///
/// ```
/// use homura::nn::functional::{BatchNormOptions, batch_norm};
///
/// let x = homura::tensor([[1.0f32, 10.0], [3.0, 20.0]]);
/// let (mean, var) = (homura::zeros([2]), homura::ones([2]));
/// let options = BatchNormOptions { training: true, momentum: 0.5, ..Default::default() };
/// let y = batch_norm(&x, Some(&mean), Some(&var), None, None, options);
/// for (y, e) in y.to_vec::<f32>().into_iter().zip([-1.0, -1.0, 1.0, 1.0]) {
///     assert!((y - e).abs() < 1e-4);
/// }
/// // Halfway from 0 to the batch mean [2, 15], from 1 to the unbiased variance [2, 50].
/// assert_eq!(mean.to_vec::<f32>(), [1.0, 7.5]);
/// assert_eq!(var.to_vec::<f32>(), [1.5, 25.5]);
///
/// let eval = BatchNormOptions { eps: 0.0, ..Default::default() };
/// let y = batch_norm(&x, Some(&mean), Some(&var), None, None, eval);
/// assert!((y.to_vec::<f32>()[1] - 2.5 / 25.5f32.sqrt()).abs() < 1e-6);
/// ```
///
/// # Panics
///
/// With PyTorch's messages and in its order: in training if a channel has only one value or
/// `eps` isn't positive, otherwise if `eps` is negative; if `input` has fewer than 2
/// dimensions; in evaluation mode without running statistics; if a per-channel tensor doesn't
/// have `C` elements; if a running statistic requires grad while grad mode is enabled; if only
/// one of the running statistics is given; if `input` isn't floating point; and if the dtypes
/// differ, other than for a half precision input with `F32` parameters and running statistics.
/// Unlike PyTorch: when the tensors are on different devices; for inputs without elements, for
/// which PyTorch skips the checks after the one of `eps`; and, before anything is written, when
/// running statistics that a training pass updates aren't writable (an expanded tensor, which
/// PyTorch writes element by element) or may share memory with `input`, `weight`, `bias` or each
/// other (which PyTorch reads after writing them). Interleaved views of one buffer, such as
/// its columns, count as sharing.
#[track_caller]
pub fn batch_norm(
    input: &Tensor,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    options: BatchNormOptions,
) -> Tensor {
    let BatchNormOptions {
        training,
        momentum,
        eps,
    } = options;
    let params = [weight, bias, running_mean, running_var];
    let mut devices = vec![input.data()];
    devices.extend(params.iter().flatten().map(|t| t.data()));
    crate::ops::common_device(&devices);
    let shape = input.shape();
    // PyTorch's Python checks.
    if training
        && let Some(&batch) = shape.first()
        && batch * shape.iter().skip(2).product::<usize>() == 1
    {
        panic!(
            "Expected more than 1 value per channel when training, got input size \
             torch.Size({:?})",
            &shape[..]
        );
    }
    if training && eps <= 0.0 {
        panic!("batch_norm eps must be positive during training, but got {eps:?}");
    } else if eps < 0.0 {
        panic!("batch_norm eps must be non-negative, but got {eps:?}");
    }
    let channels = shape[crate::shape::normalize_dim(1, shape.len())];
    for (name, t) in [("running_mean", running_mean), ("running_var", running_var)] {
        match t {
            Some(_) => check_channels(name, t, channels),
            None if !training => panic!("{name} must be defined in evaluation mode"),
            None => {}
        }
    }
    check_channels("weight", weight, channels);
    check_channels("bias", bias, channels);
    if crate::autograd::is_grad_enabled() {
        for (name, t) in [("running_mean", running_mean), ("running_var", running_var)] {
            if t.is_some_and(|t| t.requires_grad()) {
                panic!(
                    "The function 'native_batch_norm' is not differentiable with respect to \
                     argument '{name}'. This input cannot have requires_grad True."
                );
            }
        }
    }
    if running_mean.is_some() != running_var.is_some() {
        panic!("running_mean and running_var must either both be None or neither be None");
    }
    let dtype = input.dtype();
    if !dtype.is_floating_point() {
        panic!("batch_norm is not implemented for {dtype} tensors");
    }
    // PyTorch's `is_mixed_type` and `check_mixed_data_type`.
    let mixed = params
        .iter()
        .flatten()
        .next()
        .is_some_and(|p| p.dtype() != dtype);
    if mixed {
        if params.iter().flatten().any(|p| p.dtype() != DType::F32) {
            panic!("mixed dtype (CPU): expect parameter to have scalar type of Float");
        }
        if !matches!(dtype, DType::F16 | DType::BF16) {
            panic!("mixed dtype (CPU): all inputs must share same datatype.");
        }
    }
    let param_dtype = if mixed { DType::F32 } else { dtype };
    if let Some(p) = params.iter().flatten().find(|p| p.dtype() != param_dtype) {
        panic!("expected scalar type {param_dtype} but found {}", p.dtype());
    }
    let args = BatchNorm {
        training,
        eps,
        dtype,
        opmath: opmath(dtype),
        param_dtype,
    };
    let channel_shape = BatchNorm::channel_shape(input);
    let per_channel = |t: &Tensor| t.reshape(&channel_shape).to(args.opmath);
    let n = BatchNorm::count(input);
    let update = match (running_mean, running_var) {
        (Some(mean), Some(var)) if training && n > 0 && channels > 0 => Some((mean, var)),
        _ => None,
    };
    // The node's input checks, before the running statistics are written.
    crate::autograd::check_recordable(&[Some(input), weight, bias]);
    if let Some((mean, var)) = update {
        // Both targets before either write: each must be writable and share no memory with the
        // other or with what the forward and backward passes read.
        let (mean, var) = (mean.data(), var.data());
        for (target, other) in [(&mean, &var), (&var, &mean)] {
            check_writable(target);
            let read = [Some(input), weight, bias].into_iter().flatten();
            check_no_overlap(target, read.map(|t| t.data()).chain([other.clone()]));
        }
    }
    let out = {
        let _guard = no_grad();
        let x = input.to(args.opmath);
        let (mean, invstd) = match (training, running_mean, running_var) {
            (false, Some(mean), Some(var)) => (
                per_channel(mean),
                per_channel(var).add(eps).sqrt().reciprocal(),
            ),
            _ => {
                let (mean, var_sum) = args.moments(&x);
                if let Some((running_mean, running_var)) = update {
                    update_running_stats(running_mean, running_var, &mean, &var_sum, n, momentum);
                }
                args.saved_stats(&mean, &var_sum, n)
            }
        };
        let alpha = match weight {
            Some(w) => invstd.mul(per_channel(w)),
            None => invstd,
        };
        let shift = mean.mul(&alpha);
        let beta = match bias {
            Some(b) => per_channel(b).sub(&shift),
            // PyTorch subtracts from a zero bias, which turns -0 into 0.
            None => shift.neg().add(0.0),
        };
        x.mul(&alpha).add(&beta).to(dtype)
    };
    let running = (!training).then(|| Running {
        mean: running_mean.unwrap(),
        var: running_var.unwrap(),
    });
    formulas::batch_norm(input, weight, bias, running, args, &out);
    out
}

/// PyTorch's CPU update of the running statistics with the batch `mean` and `var_sum` (in
/// `opmath`) of `n` elements per channel: `running = momentum·batch + (1 − momentum)·running`
/// with the unbiased variance `var_sum / (n − 1)`, in `F64` for `F64` statistics and otherwise
/// in `F32`; for `F32` statistics `momentum·unbiased` and the sum are computed in `F64`, as in
/// PyTorch. (PyTorch updates `BF16` statistics in `BF16`, rounding each operation.)
fn update_running_stats(
    running_mean: &Tensor,
    running_var: &Tensor,
    mean: &Tensor,
    var_sum: &Tensor,
    n: usize,
    momentum: f64,
) {
    let stats = running_mean.dtype();
    let (compute, momentum, rest) = if stats == DType::F64 {
        (DType::F64, momentum, 1.0 - momentum)
    } else {
        let m = momentum as f32;
        (DType::F32, m as f64, (1.0 - m) as f64)
    };
    let wide = if stats == DType::F32 {
        DType::F64
    } else {
        compute
    };
    let flat = |t: &Tensor| t.reshape([-1]).to(compute);
    let update = |running: &Tensor, new: Tensor| {
        running.copy_(&new.to(stats).reshape(running.shape()));
    };
    update(
        running_mean,
        flat(mean).mul(momentum).add(flat(running_mean).mul(rest)),
    );
    let unbiased = flat(var_sum).div((n - 1) as f64);
    update(
        running_var,
        unbiased
            .to(wide)
            .mul(momentum)
            .add(flat(running_var).mul(rest).to(wide)),
    );
}
