use std::fmt;

use super::{Forward, IntoPair};
use crate::Tensor;
use crate::nn::functional::{self, MaxPool2dOptions};
use crate::nn::{Module, python_bool, python_size};

/// Max pooling over the last two dimensions, PyTorch's `nn.MaxPool2d`, with
/// [`functional::max_pool2d`]. The options are set one after another; the sizes take a number for
/// both dimensions or a pair ([`IntoPair`]). They are checked by the forward computation, as in
/// PyTorch.
///
/// ```
/// use homura::nn::MaxPool2d;
///
/// let pool = MaxPool2d::new(2);
/// let x = homura::arange(16.0).reshape([1, 1, 4, 4]);
/// assert_eq!(pool.forward(&x).to_vec::<f32>(), [5.0, 7.0, 13.0, 15.0]);
///
/// let (values, indices) = MaxPool2d::new(3).stride(2).padding(1).forward_with_indices(&x);
/// assert_eq!(values.shape(), [1, 1, 2, 2]);
/// assert_eq!(indices.to_vec::<i64>(), [5, 7, 13, 15]);
/// ```
#[derive(Module, Clone, Copy, PartialEq, Eq)]
#[module(extra_repr = Self::describe)]
#[non_exhaustive]
pub struct MaxPool2d {
    /// The kernel size, stride, padding, dilation and ceil mode.
    #[module(skip)]
    pub options: MaxPool2dOptions,
}

impl MaxPool2d {
    /// Max pooling with windows of `kernel_size`, a stride of the kernel size and PyTorch's other
    /// defaults (`nn.MaxPool2d(kernel_size)`).
    #[track_caller]
    pub fn new(kernel_size: impl IntoPair) -> MaxPool2d {
        MaxPool2d {
            options: MaxPool2dOptions::new(kernel_size.into_pair()),
        }
    }

    /// This layer with `stride` (`stride=...`).
    #[must_use]
    #[track_caller]
    pub fn stride(mut self, stride: impl IntoPair) -> MaxPool2d {
        self.options.stride = Some(stride.into_pair());
        self
    }

    /// This layer with `padding` (`padding=...`), an implicit `-inf` padding.
    #[must_use]
    #[track_caller]
    pub fn padding(mut self, padding: impl IntoPair) -> MaxPool2d {
        self.options.padding = padding.into_pair();
        self
    }

    /// This layer with `dilation` (`dilation=...`).
    #[must_use]
    #[track_caller]
    pub fn dilation(mut self, dilation: impl IntoPair) -> MaxPool2d {
        self.options.dilation = dilation.into_pair();
        self
    }

    /// This layer with `ceil_mode` (`ceil_mode=...`): whether a last partial window counts.
    #[must_use]
    #[track_caller]
    pub fn ceil_mode(mut self, ceil_mode: bool) -> MaxPool2d {
        self.options.ceil_mode = ceil_mode;
        self
    }

    /// The maximum of each window of `input`, `[C, H, W]` or `[N, C, H, W]`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        functional::max_pool2d(input, self.options)
    }

    /// The maximum of each window of `input` and its position in the plane, like PyTorch's
    /// layer with `return_indices=True`: see [`functional::max_pool2d_with_indices`].
    #[track_caller]
    pub fn forward_with_indices(&self, input: &Tensor) -> (Tensor, Tensor) {
        functional::max_pool2d_with_indices(input, self.options)
    }

    /// PyTorch's `extra_repr()` text.
    fn describe(&self) -> String {
        let o = &self.options;
        format!(
            "kernel_size={}, stride={}, padding={}, dilation={}, ceil_mode={}",
            python_size(o.kernel_size),
            python_size(o.stride.unwrap_or(o.kernel_size)),
            python_size(o.padding),
            python_size(o.dilation),
            python_bool(o.ceil_mode)
        )
    }
}

impl Forward for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        MaxPool2d::forward(self, input)
    }
}

/// The options, like PyTorch's `repr`, with the stride that is used:
/// `MaxPool2d { kernel_size: [2, 2], stride: [2, 2], padding: [0, 0], dilation: [1, 1], ceil_mode: false }`.
impl fmt::Debug for MaxPool2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let o = &self.options;
        f.debug_struct("MaxPool2d")
            .field("kernel_size", &o.kernel_size)
            .field("stride", &o.stride.unwrap_or(o.kernel_size))
            .field("padding", &o.padding)
            .field("dilation", &o.dilation)
            .field("ceil_mode", &o.ceil_mode)
            .finish()
    }
}

/// Average pooling into `output_size` windows over the last two dimensions, PyTorch's
/// `nn.AdaptiveAvgPool2d`, with [`functional::adaptive_avg_pool2d`].
///
/// ```
/// use homura::nn::AdaptiveAvgPool2d;
///
/// let pool = AdaptiveAvgPool2d::new(1);
/// let x = homura::arange(8.0).reshape([1, 2, 2, 2]);
/// assert_eq!(pool.forward(&x).to_vec::<f32>(), [1.5, 5.5]);
/// ```
#[derive(Module, Clone, Copy, Debug, PartialEq, Eq)]
#[module(extra_repr = Self::describe)]
#[non_exhaustive]
pub struct AdaptiveAvgPool2d {
    /// The output height and width.
    pub output_size: [usize; 2],
}

impl AdaptiveAvgPool2d {
    /// Pooling into `output_size` windows: a number for both dimensions or a pair
    /// (`nn.AdaptiveAvgPool2d(output_size)`, without `None` entries).
    #[track_caller]
    pub fn new(output_size: impl IntoPair) -> AdaptiveAvgPool2d {
        AdaptiveAvgPool2d {
            output_size: output_size.into_pair(),
        }
    }

    /// The mean of each window of `input`, `[C, H, W]` or `[N, C, H, W]`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        functional::adaptive_avg_pool2d(input, self.output_size)
    }

    /// PyTorch's `extra_repr()` text.
    fn describe(&self) -> String {
        format!("output_size={}", python_size(self.output_size))
    }
}

impl Forward for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        AdaptiveAvgPool2d::forward(self, input)
    }
}
