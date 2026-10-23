use crate::autograd::formulas::loss::{self as formulas, NllLoss};
use crate::shape::broadcast_shapes;
use crate::{DType, Device, Shape, Tensor, is_grad_enabled, no_grad, promote_types, where_};

/// How a loss combines the losses of the elements or positions, PyTorch's `reduction`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Reduction {
    /// The loss of each element or position, unreduced (`'none'`).
    None,
    /// The mean (`'mean'`, the default). For class targets of [`nll_loss`] and
    /// [`cross_entropy`], the sum divided by the total weight of the targets that aren't ignored.
    /// The mean of nothing is NaN.
    #[default]
    Mean,
    /// The sum (`'sum'`).
    Sum,
}

fn reduce(loss: Tensor, reduction: Reduction) -> Tensor {
    match reduction {
        Reduction::None => loss,
        Reduction::Mean => loss.mean(),
        Reduction::Sum => loss.sum(),
    }
}

/// Rejects a tensor argument that the reference doesn't differentiate when it requires grad in
/// grad mode, as PyTorch's autograd layer does.
#[track_caller]
fn check_not_differentiable(function: &str, argument: &str, t: Option<&Tensor>) {
    if is_grad_enabled() && t.is_some_and(Tensor::requires_grad) {
        panic!(
            "The function '{function}' is not differentiable with respect to argument \
             '{argument}'. This input cannot have requires_grad True."
        );
    }
}

/// The mean squared error `(input - target)²`, PyTorch's `F.mse_loss`. `input` and `target`
/// are broadcast together (PyTorch warns when their shapes differ, homura doesn't) and promoted
/// to a common dtype, which must be floating point. Both get gradients.
///
/// ```
/// use homura::nn::functional::{Reduction, mse_loss};
///
/// let x = homura::tensor([1.0f32, 2.0, 4.0]);
/// let y = homura::tensor([1.0f32, 0.0, 1.0]);
/// assert_eq!(mse_loss(&x, &y, Reduction::Sum).item::<f32>(), 13.0);
/// assert_eq!(mse_loss(&x, &y, Reduction::None).to_vec::<f32>(), [0.0, 4.0, 9.0]);
/// ```
///
/// # Panics
///
/// Panics if the shapes don't broadcast or the common dtype is not floating point.
#[track_caller]
pub fn mse_loss(input: &Tensor, target: &Tensor, reduction: Reduction) -> Tensor {
    let shape = broadcast_shapes(&input.shape(), &target.shape());
    // PyTorch's `broadcast_tensors`, which records an expansion only where the shape changes.
    #[track_caller]
    fn expand(t: &Tensor, shape: &Shape) -> Tensor {
        if t.shape() == *shape {
            t.clone()
        } else {
            t.expand(shape)
        }
    }
    let (input, target) = (expand(input, &shape), expand(target, &shape));
    let dtype = promote_types(input.dtype(), target.dtype());
    if !dtype.is_floating_point() {
        panic!("mse_loss is not implemented for {dtype} tensors");
    }
    let out = {
        let _guard = no_grad();
        let diff = input.sub(&target);
        reduce(diff.mul(&diff), reduction)
    };
    formulas::mse_loss(&input, &target, reduction, &out);
    out
}

/// The negative log likelihood loss, PyTorch's `F.nll_loss`: `input` holds log-probabilities
/// with the classes in dimension 1 (`[N, C]`, or `[N, C, d1, d2, ...]` for a loss per position),
/// or is a single `[C]` sample, and `target` the class of each position (`[N]`,
/// `[N, d1, d2, ...]`, or `[]` / `[1]` for a single sample) as `I64` or `U8`. The loss of a
/// position is `-weight[c] * input[c]` for its class `c` (the weight is 1 without `weight`), and
/// zero where the target is `ignore_index` (PyTorch's default is `-100`). The mean divides by
/// the total weight of the targets that aren't ignored; with none, it is NaN. With zero classes,
/// every target must be ignored. Inputs of more than two dimensions take any weight with one
/// element per class, and `U8` targets, which PyTorch's spatial CPU kernel rejects.
///
/// Only `input` gets a gradient, from the `NllLossBackward0` node (`NllLoss2DBackward0` for more
/// than two dimensions, as in PyTorch): `weight` is a constant, which can't require grad in grad
/// mode, and the targets are integers.
///
/// ```
/// use homura::nn::functional::{Reduction, nll_loss};
///
/// let log_probs = homura::tensor([[-0.5f32, -1.0, -2.0], [-3.0, -0.25, -1.5]]);
/// let target = homura::tensor([0i64, 1]);
/// let loss = nll_loss(&log_probs, &target, None, -100, Reduction::Mean);
/// assert_eq!(loss.item::<f32>(), 0.375);
/// let ignored = homura::tensor([0i64, -100]);
/// assert_eq!(nll_loss(&log_probs, &ignored, None, -100, Reduction::None).to_vec::<f32>(), [0.5, 0.0]);
/// ```
///
/// # Panics
///
/// With PyTorch's messages: if `input` is 0-D or not floating point, if the shapes don't match
/// as described, if `target` is not `I64` or `U8`, if `weight` doesn't have one element per class
/// (`[C]`) or the dtype of `input`, if `weight` requires grad in grad mode, or if a target that
/// isn't `ignore_index` is not a class (checked for CPU targets).
#[track_caller]
pub fn nll_loss(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    ignore_index: i64,
    reduction: Reduction,
) -> Tensor {
    let dims = input.num_dims();
    if dims < 1 {
        panic!("Expected 1 or more dimensions (got {dims})");
    }
    let (shape, target_shape) = (input.shape(), target.shape());
    if dims != 1 {
        // PyTorch reads a 0-D target's batch size as 0.
        let batch = target_shape.first().copied().unwrap_or(0);
        if shape[0] != batch {
            panic!(
                "Expected input batch_size ({}) to match target batch_size ({batch}).",
                shape[0]
            );
        }
    }
    if (dims == 3 || dims > 4) && target_shape.get(1..) != Some(&shape[2..]) {
        let mut expected = vec![shape[0]];
        expected.extend(&shape[2..]);
        panic!("Expected target size {expected:?}, got {target_shape:?}");
    }
    // PyTorch computes inputs of more than two dimensions with `nll_loss2d`, which has its own
    // checks and messages.
    let spatial = dims > 2;
    check_not_differentiable(
        if spatial {
            "nll_loss2d_forward"
        } else {
            "nll_loss_forward"
        },
        "weight",
        weight,
    );
    let class_dim = if dims == 1 { 0 } else { 1 };
    let n_classes = shape[class_dim];
    let target_dims = target.num_dims();
    if !spatial {
        if target_dims > 1 {
            panic!("0D or 1D target tensor expected, multi-target not supported");
        }
        check_target_dtype(target);
        if dims == 1 && target_dims == 1 && target_shape[0] != 1 {
            panic!(
                "For 1D input, 1D target must have size 1, but got target size: {}",
                target_shape[0]
            );
        }
        if let Some(w) = weight
            && (w.num_dims() != 1 || w.num_elements() != n_classes)
        {
            panic!(
                "weight tensor should be defined either for all {n_classes} classes or no classes \
                 but got weight tensor of shape: {:?}",
                w.shape()
            );
        }
    } else {
        if dims == 4 && target_dims != 3 {
            panic!(
                "only batches of spatial targets supported (3D tensors) but got targets of \
                 dimension: {target_dims}"
            );
        }
        check_target_dtype(target);
        if weight.is_some_and(|w| w.num_elements() != n_classes) {
            panic!("weight tensor should be defined either for all or no classes");
        }
        if dims == 4 && target_shape[1..] != shape[2..] {
            panic!("size mismatch (got input: {shape:?} , target: {target_shape:?}");
        }
    }
    let dtype = input.dtype();
    if !dtype.is_floating_point() {
        panic!("nll_loss is not implemented for {dtype} tensors");
    }
    if let Some(w) = weight
        && w.dtype() != dtype
    {
        panic!("expected scalar type {dtype} but found {}", w.dtype());
    }
    let (out, total_weight) = {
        let _guard = no_grad();
        let target = target.to(DType::I64);
        if target.device() == Device::Cpu {
            let bad = target
                .to_vec::<i64>()
                .into_iter()
                .find(|&t| t != ignore_index && !(0..n_classes as i64).contains(&t));
            if let Some(t) = bad {
                panic!("Target {t} is out of bounds.");
            }
        }
        let valid = target.not_equal(ignore_index);
        let ignored = valid.logical_not();
        let index = where_(&valid, &target, 0i64);
        let (losses, total_weight) = if n_classes == 0 {
            // Every target is ignored (the others are out of range), and there is no class 0
            // to stand in for them: zero losses and a zero total weight.
            let zeros = crate::zeros_with(index.shape(), (dtype, input.device()));
            (zeros, valid.sum().to(dtype))
        } else {
            let picked = if class_dim == 0 {
                input.gather(0, &index.reshape([1]))
            } else {
                input.gather(1, &index.unsqueeze(1))
            }
            .reshape(index.shape());
            // `nll_loss2d` takes a weight of any shape with one element per class.
            // Not `Option::map`: a closure would lose the caller's location (`#[track_caller]`).
            #[allow(clippy::manual_map)]
            let weights = match weight {
                Some(w) => Some(
                    w.reshape([-1])
                        .index_select(0, &index.reshape([-1]))
                        .reshape(index.shape()),
                ),
                None => None,
            };
            let losses = match &weights {
                Some(w) => picked.neg().mul(w),
                None => picked.neg(),
            }
            .masked_fill(&ignored, 0.0);
            let total_weight = match &weights {
                Some(w) => w.masked_fill(&ignored, 0.0).sum(),
                None => valid.sum().to(dtype),
            };
            (losses, total_weight)
        };
        let out = match reduction {
            // A single sample has a 0-D loss also without reduction.
            Reduction::None if dims == 1 => losses.reshape(Shape::new(&[])),
            Reduction::None => losses,
            Reduction::Sum => losses.sum(),
            Reduction::Mean => losses.sum().div(&total_weight),
        };
        (out, total_weight)
    };
    let args = NllLoss {
        reduction,
        ignore_index,
        class_dim,
        spatial,
    };
    formulas::nll_loss(input, target, weight, &total_weight, args, &out);
    out
}

/// The options of [`cross_entropy`], with PyTorch's defaults.
#[derive(Clone, Debug)]
pub struct CrossEntropyOptions {
    /// The weight of each class, `[C]`, or `None` for equal weights.
    pub weight: Option<Tensor>,
    /// The class target that is ignored, `-100` by default. Must be negative for probability
    /// targets.
    pub ignore_index: i64,
    /// The reduction, [`Reduction::Mean`] by default.
    pub reduction: Reduction,
    /// The amount of smoothing, from 0 (none, the default) to 1: the targets become a mixture
    /// of themselves and the uniform distribution over the classes. As in PyTorch, negative and
    /// NaN values mean no smoothing.
    pub label_smoothing: f64,
}

impl Default for CrossEntropyOptions {
    fn default() -> Self {
        CrossEntropyOptions {
            weight: None,
            ignore_index: -100,
            reduction: Reduction::Mean,
            label_smoothing: 0.0,
        }
    }
}

#[track_caller]
fn check_label_smoothing(label_smoothing: f64) {
    if label_smoothing > 1.0 {
        panic!("label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}");
    }
}

/// The cross entropy of the softmax of `input` (logits, with the classes in dimension 1, or a
/// single `[C]` sample) and `target`, PyTorch's `F.cross_entropy`. A `target` of the shape of
/// `input` holds class probabilities, in a floating point dtype; otherwise it holds classes, as
/// for [`nll_loss`].
///
/// It is computed as in PyTorch: for class targets, [`nll_loss`] of
/// [`log_softmax`](Tensor::log_softmax) (with label smoothing, plus the smoothing term, whose
/// mean is also divided by the weight of the targets that aren't ignored); for probability
/// targets, `-Σ log_softmax(input) * target * weight` over the classes, whose mean divides by
/// the number of positions, independently of the weight. Probability targets and their weight
/// get gradients; class targets' weight can't require grad in grad mode.
///
/// ```
/// use homura::DType;
/// use homura::nn::functional::{CrossEntropyOptions, cross_entropy};
///
/// let logits = homura::tensor_with([[2.0, 0.0], [0.0, 2.0]], DType::F64);
/// let loss = cross_entropy(&logits, &homura::tensor([0i64, 1]), CrossEntropyOptions::default());
/// assert!((loss.item::<f64>() - (1.0 + (-2f64).exp()).ln()).abs() < 1e-15);
/// ```
///
/// # Panics
///
/// With PyTorch's messages: as [`nll_loss`] for class targets; if probability targets are not
/// floating point, `ignore_index` is not negative for them, or their weight isn't `[C]`; if
/// `label_smoothing` is greater than 1.
#[track_caller]
pub fn cross_entropy(input: &Tensor, target: &Tensor, options: CrossEntropyOptions) -> Tensor {
    let CrossEntropyOptions {
        weight,
        ignore_index,
        reduction,
        label_smoothing,
    } = options;
    let weight = weight.as_ref();
    let class_dim = if input.num_dims() == 1 { 0 } else { 1 };
    if input.shape() == target.shape() {
        if !target.is_floating_point() {
            panic!(
                "Expected floating point type for target with class probabilities, got {}",
                target.dtype()
            );
        }
        if ignore_index >= 0 {
            panic!("ignore_index is not supported for floating point target");
        }
        cross_entropy_probabilities(input, target, weight, reduction, label_smoothing)
    } else if label_smoothing > 0.0 {
        check_label_smoothing(label_smoothing);
        let log_probs = input.log_softmax(class_dim as i64);
        let nll = nll_loss(&log_probs, target, weight, ignore_index, reduction);
        let n_classes = log_probs.shape()[class_dim];
        let smooth = match weight {
            Some(w) => log_probs.mul(w.reshape(class_shape(input.num_dims(), class_dim, w))),
            None => log_probs,
        }
        .sum_dim(class_dim as i64, false)
        .neg();
        let ignored = target.equal(ignore_index);
        let smooth = smooth.masked_fill(&ignored, 0.0);
        let smooth = match reduction {
            Reduction::None => smooth,
            Reduction::Sum => smooth.sum(),
            Reduction::Mean => {
                let total = match weight {
                    // No class weights to gather: every target is ignored.
                    Some(w) if n_classes == 0 => w.sum(),
                    Some(w) => {
                        let classes = target.masked_fill(&ignored, 0).to(DType::I64).reshape([-1]);
                        w.gather(0, &classes)
                            .masked_fill(&ignored.reshape([-1]), 0.0)
                            .sum()
                    }
                    None => ignored.logical_not().sum(),
                };
                smooth.sum().div(&total)
            }
        };
        nll.mul(1.0 - label_smoothing)
            .add(smooth.mul(label_smoothing / n_classes as f64))
    } else {
        let log_probs = input.log_softmax(class_dim as i64);
        nll_loss(&log_probs, target, weight, ignore_index, reduction)
    }
}

/// The shape `[1, C, 1, ...]` that broadcasts a weight `w` of `C` elements along the classes of
/// a `dims`-dimensional input.
fn class_shape(dims: usize, class_dim: usize, w: &Tensor) -> Shape {
    let mut shape = vec![1; dims];
    shape[class_dim] = w.shape()[0];
    Shape::new(&shape)
}

#[track_caller]
fn cross_entropy_probabilities(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    label_smoothing: f64,
) -> Tensor {
    let dims = input.num_dims();
    let class_dim = if dims == 1 { 0 } else { 1 };
    let log_probs = input.log_softmax(class_dim as i64);
    let n_classes = input.shape()[class_dim];
    if let Some(w) = weight
        && (w.num_dims() != 1 || w.num_elements() != n_classes)
    {
        panic!(
            "cross_entropy: weight tensor should be defined either for all {n_classes} classes or \
             no classes but got weight tensor of shape: {:?}",
            w.shape()
        );
    }
    let target = if label_smoothing > 0.0 {
        check_label_smoothing(label_smoothing);
        target
            .mul(1.0 - label_smoothing)
            .add(label_smoothing / n_classes as f64)
    } else {
        target.clone()
    };
    let product = log_probs.mul(&target);
    let product = match weight {
        Some(w) => product.mul(w.reshape(class_shape(dims, class_dim, w))),
        None => product,
    };
    match reduction {
        Reduction::None => product.sum_dim(class_dim as i64, false).neg(),
        Reduction::Sum => product.sum().neg(),
        Reduction::Mean => {
            let numel = log_probs.num_elements();
            if numel == 0 {
                // PyTorch's `-(..).sum().fill_(NaN)`, a recorded fill.
                let sum = product.sum();
                sum.fill_(f64::NAN);
                sum.neg()
            } else {
                product.sum().neg().div((numel / n_classes) as i64)
            }
        }
    }
}

/// The binary cross entropy of `sigmoid(input)` and `target`, computed stably from the logits,
/// PyTorch's `F.binary_cross_entropy_with_logits`: the loss of an element is
/// `-weight * (pos_weight * target * log σ(input) + (1 - target) * log σ(-input))`.
///
/// It is computed like PyTorch's implementation, `(1 - target) * input - log_weight *
/// log_sigmoid(input)` with `log_weight = (pos_weight - 1) * target + 1`, multiplying in place,
/// so that `weight` and `pos_weight` broadcast to the shape of `input`, and the dtype follows
/// PyTorch's in-place rules. `input` and `target` get gradients from the
/// `BinaryCrossEntropyWithLogitsBackward0` node; `weight` and `pos_weight` are constants, which
/// can't require grad in grad mode.
///
/// ```
/// use homura::DType;
/// use homura::nn::functional::{Reduction, binary_cross_entropy_with_logits};
///
/// let x = homura::tensor_with([0.0, 40.0], DType::F64);
/// let y = homura::tensor_with([1.0, 1.0], DType::F64);
/// let loss = binary_cross_entropy_with_logits(&x, &y, None, Reduction::None, None);
/// assert_eq!(loss.to_vec::<f64>(), [2f64.ln(), 4.248354255291589e-18]);
/// ```
///
/// # Panics
///
/// With PyTorch's messages: if the shapes of `input` and `target` differ, `weight` or
/// `pos_weight` requires grad in grad mode, `input` isn't floating point, the weights don't
/// broadcast to the shape of `input`, or a result can't be written in place in the dtype of
/// `target` (an integer `target`).
#[track_caller]
pub fn binary_cross_entropy_with_logits(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    pos_weight: Option<&Tensor>,
) -> Tensor {
    if target.shape() != input.shape() {
        panic!(
            "Target size ({:?}) must be the same as input size ({:?})",
            target.shape(),
            input.shape()
        );
    }
    let function = "binary_cross_entropy_with_logits";
    check_not_differentiable(function, "weight", weight);
    check_not_differentiable(function, "pos_weight", pos_weight);
    let out = {
        let _guard = no_grad();
        let log_sigmoid_input = input.log_sigmoid();
        if let Some(p) = pos_weight {
            let log_weight = p.sub(1).mul(target);
            log_weight.add_(1);
            log_sigmoid_input.mul_(&log_weight);
        }
        let loss = 1i64 - target;
        loss.mul_(input).sub_(&log_sigmoid_input);
        if let Some(w) = weight {
            loss.mul_(w);
        }
        reduce(loss, reduction)
    };
    formulas::binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction, &out);
    out
}

/// Panics unless the class indices of an NLL loss are `I64` or `U8` (a function, not a closure,
/// so that the panic reports the caller's location).
#[track_caller]
fn check_target_dtype(target: &Tensor) {
    let t = target.dtype();
    if t != DType::I64 && t != DType::U8 {
        panic!("expected target dtype to be i64 or u8, but got {t}");
    }
}
