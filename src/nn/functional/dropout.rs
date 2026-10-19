use crate::{DType, Tensor};

/// Dropout, PyTorch's `F.dropout(input, p, training)`: in training, each element is zeroed with
/// probability `p` and the others are scaled by `1 / (1 - p)`, so that the expected value stays
/// the same. PyTorch's default `p` is 0.5.
///
/// It computes what PyTorch's CPU implementation does: `input * noise`, where `noise` has the
/// dtype of `input` and is `1 / (1 - p)` (in that dtype) for the kept elements and 0 for the
/// others. The gradient is the incoming one times the same noise (`MulBackward0`). The kept
/// elements are drawn from homura's random number generator (see
/// [`manual_seed`](crate::manual_seed)), so they differ from PyTorch's for the same seed.
///
/// Without training, with `p = 0` or for an input without elements, the result is `input`
/// itself (a handle to the same tensor), and nothing is drawn. With `p = 1` the result is
/// `input` times a zero (NaN for infinite and NaN elements), and nothing is drawn.
///
/// ```
/// use homura::nn::functional::dropout;
///
/// let x = homura::ones([1000]);
/// let y = dropout(&x, 0.2, true);
/// let kept = y.not_equal(0.0).sum().item::<i64>();
/// assert!((700..900).contains(&kept));
/// assert!(y.equal(0.0).logical_or(&y.equal(1.25)).all().item::<bool>());
/// assert!(dropout(&x, 0.2, false).is_same(&x));
/// ```
///
/// # Panics
///
/// Panics if `p` isn't between 0 and 1 (also NaN), in training or not, and for `0 < p < 1` in
/// training if `input` isn't floating point (with PyTorch's messages).
pub fn dropout(input: &Tensor, p: f64, training: bool) -> Tensor {
    if !(0.0..=1.0).contains(&p) {
        panic!("dropout probability has to be between 0 and 1, but got {p}");
    }
    if p == 0.0 || !training || input.num_elements() == 0 {
        return input.clone();
    }
    let dtype = input.dtype();
    if p == 1.0 {
        return input.mul(crate::zeros_with([], (dtype, input.device())));
    }
    if !dtype.is_floating_point() {
        panic!("result type f32 can't be cast to the desired output type {dtype}");
    }
    let kept = crate::rand_with(input.shape(), (DType::F64, input.device())).less(1.0 - p);
    let noise = kept.to(dtype).div(1.0 - p);
    input.mul(&noise)
}
