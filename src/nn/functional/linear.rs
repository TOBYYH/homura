use crate::{Shape, Tensor};

/// `input @ weightᵀ + bias`, PyTorch's `F.linear`: `input` has the input features in its last
/// dimension and `weight` the layout `[out_features, in_features]`. A 1-D `weight` is a single
/// output's weights, and that dimension is removed from the result. `bias` is broadcast to the
/// result.
///
/// It computes and records the same operations as PyTorch's CPU implementation: with a bias, a
/// 2-D `input` gives one `addmm` (`AddmmBackward0`). An `input` of other ranks is reshaped to
/// 2-D for the `addmm` and the result viewed back (`ViewBackward0`) when both `input` and the
/// bias are contiguous and the bias is 1-D (any `[n]`, `[1]` and `[0]` included) or has a single
/// dimension that isn't 1 (`[1, n]`, ...). Otherwise `input.matmul(weightᵀ)` gets the bias added in place (`AddBackward0`). The
/// paths round differently in half precision. Two nodes differ: the weight is transposed with
/// `transpose(0, 1)` (`TransposeBackward0`) where PyTorch uses `t()` (`TBackward0`), and the
/// matmul is the single node of [`Tensor::matmul`] where PyTorch records views around an `mm`.
/// The gradients have the same higher-order derivatives as those operations.
///
/// ```
/// use homura::nn::functional::linear;
///
/// let x = homura::tensor([[1.0f32, 2.0]]);
/// let w = homura::tensor([[1.0f32, 0.0], [1.0, 1.0], [0.0, -1.0]]);
/// let b = homura::tensor([0.5f32, 0.5, 0.5]);
/// assert_eq!(linear(&x, &w, Some(&b)).to_vec::<f32>(), [1.5, 3.5, -1.5]);
/// ```
///
/// # Panics
///
/// Panics if `input` or `weight` is 0-D, if `weight` has more than two dimensions, and like the
/// operations above: if `input` and `weight` have different dtypes, if the shapes don't fit, or
/// if the bias can't be added (in the `addmm` paths, its dtype must be that of `input`; added in
/// place, it must broadcast to the result and convert to its dtype).
#[track_caller]
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let (input_dims, weight_dims) = (input.num_dims(), weight.num_dims());
    if input_dims == 0 || weight_dims == 0 {
        panic!(
            "both arguments to linear need to be at least 1D, but they are {input_dims}D and \
             {weight_dims}D"
        );
    }
    if weight_dims > 2 {
        panic!("linear expects a weight with at most 2 dimensions, but weight is {weight_dims}D");
    }
    // PyTorch's `weight.t()`, which leaves a 1-D tensor as it is.
    let weight_t = if weight_dims == 2 {
        weight.transpose(0, 1)
    } else {
        weight.clone()
    };
    if let Some(bias) = bias {
        if input_dims == 2 {
            return bias.addmm(input, &weight_t, 1, 1);
        }
        // PyTorch's condition `bias->dim() == 1 || bias->squeeze().dim() == 1`, read from the
        // shape so that no squeeze is recorded.
        let single_dim = bias.shape().iter().filter(|&&n| n != 1).count() == 1;
        let fusable = bias.is_contiguous() && (bias.num_dims() == 1 || single_dim);
        if input.is_contiguous() && fusable {
            // PyTorch's `_flatten_nd_linear`.
            let shape = input.shape();
            let (batch, features) = shape.split_at(input_dims - 1);
            let rows = Shape::new(&[batch.iter().product(), features[0]]);
            let out = bias.addmm(&input.view(rows), &weight_t, 1, 1);
            let mut out_shape = batch.to_vec();
            out_shape.push(out.shape()[1]);
            return out.view(Shape::new(&out_shape));
        }
    }
    let out = input.matmul(&weight_t);
    if let Some(bias) = bias {
        out.add_(bias);
    }
    out
}
