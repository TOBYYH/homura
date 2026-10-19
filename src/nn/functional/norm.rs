use crate::autograd::formulas::norm::{self as formulas, LayerNorm};
use crate::{DType, IntoShape, Tensor, no_grad};

/// The dtype of half precision computations: `F32` for `F16` and `BF16`.
fn opmath(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        _ => dtype,
    }
}

/// Panics if `normalized_shape` is empty.
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
