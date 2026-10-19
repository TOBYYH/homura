use crate::autograd::formulas::embedding as formulas;
use crate::{DType, Device, Tensor, no_grad};

/// The rows of `weight` at `input`, PyTorch's `F.embedding(input, weight, padding_idx)`: a lookup
/// table with one row per index. `input` holds `I64` or `I32` indices of any shape, and the result
/// has the shape of `input` followed by the row size of the 2-D `weight`.
///
/// `weight` gets a gradient (`EmbeddingBackward0`, as in PyTorch): each row the sum of the
/// gradients of the positions that picked it. Positions whose index is `padding_idx` add nothing,
/// so that the padding row keeps a zero gradient, even where the incoming gradient is infinite
/// or NaN; the lookup itself returns the row's current value. A negative `padding_idx` counts
/// from the end. PyTorch's `max_norm`, `scale_grad_by_freq` and `sparse` aren't supported.
///
/// ```
/// use homura::nn::functional::embedding;
///
/// let weight = homura::tensor([[0.0f32, 0.5], [1.0, 1.5], [2.0, 2.5]]);
/// let rows = embedding(&homura::tensor([[2i64, 0], [1, 1]]), &weight, None);
/// assert_eq!(rows.shape(), [2, 2, 2]);
/// assert_eq!(rows.to_vec::<f32>(), [2.0, 2.5, 0.0, 0.5, 1.0, 1.5, 1.0, 1.5]);
/// ```
///
/// # Panics
///
/// With PyTorch's messages: if `padding_idx` is not in `-num_embeddings..num_embeddings`, if
/// `weight` isn't 2-D, if `input` isn't `I64` or `I32`, or if an index is negative or not less
/// than the number of rows (checked for CPU indices). That message is "index out of range in
/// self" also for tables without rows or columns, for which PyTorch's `index_select` reports
/// others.
pub fn embedding(input: &Tensor, weight: &Tensor, padding_idx: Option<i64>) -> Tensor {
    let rows = weight.shape().first().map_or(0, |&n| n as i64);
    // PyTorch's Python wrapper normalizes `padding_idx` first.
    let padding_idx = padding_idx.map(|p| {
        if p >= rows || p < -rows {
            panic!("Padding_idx must be within num_embeddings");
        }
        if p < 0 { p + rows } else { p }
    });
    if weight.num_dims() != 2 {
        panic!("'weight' must be 2-D");
    }
    let dtype = input.dtype();
    if !matches!(dtype, DType::I64 | DType::I32) {
        panic!(
            "Expected tensor for argument #1 'indices' to have one of the following scalar types: \
             i64, i32; but got {dtype} instead (while checking arguments for embedding)"
        );
    }
    if input.device() == Device::Cpu
        && input
            .to(DType::I64)
            .to_vec::<i64>()
            .iter()
            .any(|&i| i < 0 || i >= rows)
    {
        panic!("index out of range in self");
    }
    let out = {
        let _guard = no_grad();
        let mut shape: Vec<i64> = input.shape().iter().map(|&n| n as i64).collect();
        shape.push(weight.shape()[1] as i64);
        weight
            .index_select(0, &input.reshape([-1]))
            .unsafe_view(shape)
    };
    formulas::embedding(weight, input, padding_idx, &out);
    out
}
