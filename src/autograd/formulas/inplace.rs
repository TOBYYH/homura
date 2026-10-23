//! Writes that replace the value of their target: `copy_` and the fills. The overwritten value
//! gets a zero gradient, not `None`, as in PyTorch.

use super::only;
use crate::autograd::node::{ApplyContext, Node};
use crate::autograd::record::record;
use crate::{DType, Device, Shape, Tensor};

/// Records the backward node of `target.copy_(src)`, with `target` before the write as the first
/// input and as the output.
#[track_caller]
pub(crate) fn copy(target: &Tensor, src: &Tensor) {
    let (src_shape, src_dtype, src_device) = (src.shape(), src.dtype(), src.device());
    record(&[Some(target), Some(src)], &[target], |_| CopyBackwards {
        src_shape,
        src_dtype,
        src_device,
    });
}

/// Records the backward node `name` of an in-place write that sets every element of `target` to
/// a value that doesn't depend on any tensor: `fill_` with a scalar, `zero_`, `uniform_` and
/// `normal_`.
#[track_caller]
pub(crate) fn overwrite(target: &Tensor, name: &'static str) {
    record(&[Some(target)], &[target], |_| OverwriteBackward { name });
}

/// `self.copy_(src)`: zeros for the old value of self, and for `src` the gradient converted to
/// its dtype and device, then summed over the dimensions it was broadcast along. PyTorch
/// converts before summing too, so a gradient for a wider `src` is summed in its dtype.
struct CopyBackwards {
    src_shape: Shape,
    src_dtype: DType,
    src_device: Device,
}

impl Node for CopyBackwards {
    fn name(&self) -> &'static str {
        // PyTorch's `torch::autograd::CopyBackwards`, without the namespace like AccumulateGrad.
        "CopyBackwards"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let g = only(grads);
        let self_grad = ctx.needs_input_grad(0).then(|| crate::zeros_like(&g));
        let src_grad = ctx.needs_input_grad(1).then(|| {
            g.to((self.src_dtype, self.src_device))
                .sum_to_size(self.src_shape.clone())
        });
        vec![self_grad, src_grad]
    }
}

/// A write that overwrites every element with a constant or random value: zeros for the old
/// value, also where the upstream gradient isn't finite.
struct OverwriteBackward {
    name: &'static str,
}

impl Node for OverwriteBackward {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(crate::zeros_like(&only(grads)))]
    }
}
