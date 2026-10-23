//! [`Parameter`]: a tensor that a module learns.

use std::fmt;
use std::ops::Deref;

use crate::Tensor;

/// A tensor registered as a learnable parameter of a module, like PyTorch's `nn.Parameter`.
///
/// It derefs to [`Tensor`], so the tensor methods work on it and `&param` converts to `&Tensor`
/// where one is expected; write `&*param` where an operator or a function generic over its
/// argument needs the tensor (`&*param * 2.0`). `Clone` copies the handle,
/// like [`Tensor`]'s: putting clones of one parameter into two modules shares the parameter, and
/// traversals such as [`Module::parameters`](super::Module::parameters) list it once.
///
/// ```
/// use homura::nn::Parameter;
///
/// let w = Parameter::new(homura::zeros([2, 3]));
/// assert!(w.requires_grad() && w.is_leaf());
/// assert_eq!((&*w * 2.0).shape(), [2, 3]);
/// ```
#[derive(Clone)]
pub struct Parameter(Tensor);

impl Parameter {
    /// A parameter holding the data of `data`, which requires grad, like `nn.Parameter(data)`.
    ///
    /// The parameter is a new leaf that shares the storage of `data` (writes through either are
    /// visible in both) without its history: gradients don't flow from the parameter to `data`.
    /// It is not a view of `data`, even if `data` is a view.
    ///
    /// # Panics
    ///
    /// Panics if the dtype of `data` is not floating point, with PyTorch's message.
    #[track_caller]
    pub fn new(data: Tensor) -> Parameter {
        Parameter::with_requires_grad(data, true)
    }

    /// Like [`new`](Parameter::new), with `requires_grad` set as given
    /// (`nn.Parameter(data, requires_grad=...)`). A parameter that doesn't require grad may have
    /// an integer dtype.
    ///
    /// # Panics
    ///
    /// Panics if `requires_grad` is true and the dtype of `data` is not floating point.
    #[track_caller]
    pub fn with_requires_grad(data: Tensor, requires_grad: bool) -> Parameter {
        if requires_grad && !data.is_floating_point() {
            panic!("Only Tensors of floating point and complex dtype can require gradients");
        }
        let t = data.detach();
        t.requires_grad_(requires_grad);
        Parameter(t)
    }

    /// The parameter's tensor.
    #[track_caller]
    pub fn as_tensor(&self) -> &Tensor {
        &self.0
    }
}

impl Deref for Parameter {
    type Target = Tensor;

    fn deref(&self) -> &Tensor {
        &self.0
    }
}

impl AsRef<Tensor> for Parameter {
    fn as_ref(&self) -> &Tensor {
        &self.0
    }
}

impl From<Parameter> for Tensor {
    fn from(p: Parameter) -> Tensor {
        p.0
    }
}

/// Prints `Parameter containing:` and the tensor, like PyTorch's `repr`.
impl fmt::Debug for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parameter containing:\n{:?}", self.0)
    }
}
