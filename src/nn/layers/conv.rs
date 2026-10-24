use std::fmt;

use super::{Forward, IntoPair};
use crate::Tensor;
use crate::nn::functional::{Conv2dOptions, Conv2dPadding};
use crate::nn::init::{self, FanMode, Nonlinearity};
use crate::nn::{Module, Parameter, functional, python_tuple};

/// A 2-D convolution layer, PyTorch's `nn.Conv2d`, with [`functional::conv2d`].
///
/// `weight` has the shape `[out_channels, in_channels / groups, kh, kw]` and `bias` the shape
/// `[out_channels]`, like PyTorch's, so state dicts of PyTorch's `Conv2d` load as they are. Both
/// are initialized like PyTorch does (see [`reset_parameters`](Conv2d::reset_parameters)), with
/// the default dtype and device; convert the layer with [`Module::to`] for others. The options
/// are set one after another; the sizes take a number for both dimensions or a pair
/// ([`IntoPair`]):
///
/// ```
/// use homura::nn::{Conv2d, Module};
///
/// let conv = Conv2d::new(3, 8, 3).padding(1);
/// assert_eq!(conv.weight.shape(), [8, 3, 3, 3]);
/// assert_eq!(conv.forward(&homura::ones([2, 3, 5, 5])).shape(), [2, 8, 5, 5]);
///
/// let conv = Conv2d::new(4, 4, [3, 1]).stride(2).groups(2).bias(false);
/// assert_eq!(conv.weight.shape(), [4, 2, 3, 1]);
/// assert_eq!(conv.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
/// ```
///
/// Like PyTorch, the layer checks the number of groups and that `Same` padding has a stride of
/// 1 when they are set, and the other options in the forward computation. Set the options before
/// creating an optimizer for the parameters: [`groups`](Conv2d::groups) and
/// [`bias`](Conv2d::bias) can replace them.
#[derive(Module)]
#[module(extra_repr = Self::describe)]
#[non_exhaustive]
pub struct Conv2d {
    /// The weights, `[out_channels, in_channels / groups, kh, kw]`.
    pub weight: Parameter,
    /// The bias, `[out_channels]`, if the layer has one.
    pub bias: Option<Parameter>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: [usize; 2],
    /// The stride, padding, dilation and groups.
    #[module(skip)]
    pub options: Conv2dOptions,
}

impl Conv2d {
    /// A layer from `in_channels` to `out_channels` with a kernel of `kernel_size`, a bias and
    /// PyTorch's default options (`nn.Conv2d(in_channels, out_channels, kernel_size)`). Like
    /// PyTorch, sizes of 0 are accepted here; the forward computation rejects a layer without
    /// input channels or with an empty kernel.
    #[track_caller]
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: impl IntoPair) -> Conv2d {
        let kernel_size = kernel_size.into_pair();
        let options = Conv2dOptions::default();
        let layer = Conv2d {
            weight: Parameter::new(crate::zeros(weight_shape(
                in_channels,
                out_channels,
                kernel_size,
                options.groups,
            ))),
            bias: Some(Parameter::new(crate::zeros([out_channels]))),
            in_channels,
            out_channels,
            kernel_size,
            options,
        };
        layer.reset_parameters();
        layer
    }

    /// This layer with `stride` (`stride=...`).
    ///
    /// # Panics
    ///
    /// Panics with PyTorch's message if the padding is `Same` and the stride isn't 1.
    #[must_use]
    #[track_caller]
    pub fn stride(mut self, stride: impl IntoPair) -> Conv2d {
        self.options.stride = stride.into_pair();
        self.check_same_padding();
        self
    }

    /// This layer with `padding` (`padding=...`): a number or a pair of zero paddings, or
    /// [`Conv2dPadding::Valid`] or [`Conv2dPadding::Same`].
    ///
    /// # Panics
    ///
    /// Panics with PyTorch's message if the padding is `Same` and the stride isn't 1.
    #[must_use]
    #[track_caller]
    pub fn padding(mut self, padding: impl Into<Conv2dPadding>) -> Conv2d {
        self.options.padding = padding.into();
        self.check_same_padding();
        self
    }

    /// This layer with `dilation` (`dilation=...`).
    #[must_use]
    #[track_caller]
    pub fn dilation(mut self, dilation: impl IntoPair) -> Conv2d {
        self.options.dilation = dilation.into_pair();
        self
    }

    /// This layer with `groups` (`groups=...`): the channels are split into `groups` groups,
    /// and output channel group `i` sees only input channel group `i`. Another number of groups
    /// changes the shape of the weights: the layer gets new weights and a new bias, with the
    /// dtype and device of the old weights, initialized like
    /// [`reset_parameters`](Conv2d::reset_parameters) does. (PyTorch sets `groups` only when it
    /// creates the layer.)
    ///
    /// # Panics
    ///
    /// Panics with PyTorch's messages, in its order, if `groups` is 0 or doesn't divide the
    /// input channels or the output channels.
    #[must_use]
    #[track_caller]
    pub fn groups(mut self, groups: usize) -> Conv2d {
        if groups == 0 {
            panic!("groups must be a positive integer");
        }
        if !self.in_channels.is_multiple_of(groups) {
            panic!("in_channels must be divisible by groups");
        }
        if !self.out_channels.is_multiple_of(groups) {
            panic!("out_channels must be divisible by groups");
        }
        if groups != self.options.groups {
            self.options.groups = groups;
            let shape = weight_shape(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                groups,
            );
            self.weight = Parameter::new(self.weight.new_zeros(shape));
            if let Some(b) = &self.bias {
                self.bias = Some(Parameter::new(b.new_zeros([self.out_channels])));
            }
            self.reset_parameters();
        }
        self
    }

    /// This layer with a bias if `bias` is true (`bias=True`, the default) and without one
    /// otherwise. A new bias is initialized like [`reset_parameters`](Conv2d::reset_parameters)
    /// does, with the dtype and device of the weights.
    #[must_use]
    #[track_caller]
    pub fn bias(mut self, bias: bool) -> Conv2d {
        if !bias {
            self.bias = None;
        } else if self.bias.is_none() {
            self.bias = Some(Parameter::new(self.weight.new_zeros([self.out_channels])));
            self.reset_bias();
        }
        self
    }

    /// Initializes the parameters again, like PyTorch: the weights with
    /// [`kaiming_uniform_`](init::kaiming_uniform_)`(a = √5)`, which samples from
    /// `U(-1/√fan_in, 1/√fan_in)` with `fan_in = in_channels / groups * kh * kw`, and then the
    /// bias from the same distribution (zeros for a `fan_in` of 0).
    #[track_caller]
    pub fn reset_parameters(&self) {
        init::kaiming_uniform_(
            &self.weight,
            5f64.sqrt(),
            FanMode::FanIn,
            Nonlinearity::LeakyRelu,
        );
        self.reset_bias();
    }

    fn reset_bias(&self) {
        if let Some(bias) = &self.bias {
            let (fan_in, _) = init::calculate_fan_in_and_fan_out(&self.weight);
            let bound = if fan_in > 0 {
                1.0 / (fan_in as f64).sqrt()
            } else {
                0.0
            };
            init::uniform_(bias, -bound, bound);
        }
    }

    #[track_caller]
    fn check_same_padding(&self) {
        if self.options.padding == Conv2dPadding::Same && self.options.stride != [1, 1] {
            panic!("padding='same' is not supported for strided convolutions");
        }
    }

    /// The convolution of `input`, `[N, C, H, W]` or `[C, H, W]`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        functional::conv2d(input, &self.weight, self.bias.as_deref(), self.options)
    }
}

fn weight_shape(
    in_channels: usize,
    out_channels: usize,
    [kh, kw]: [usize; 2],
    groups: usize,
) -> [usize; 4] {
    [out_channels, in_channels / groups, kh, kw]
}

impl Conv2d {
    /// PyTorch's `extra_repr()` text.
    fn describe(&self) -> String {
        let o = &self.options;
        let mut s = format!(
            "{}, {}, kernel_size={}, stride={}",
            self.in_channels,
            self.out_channels,
            python_tuple(&self.kernel_size),
            python_tuple(&o.stride)
        );
        match o.padding {
            Conv2dPadding::Explicit([0, 0]) => {}
            Conv2dPadding::Explicit(padding) => {
                s += &format!(", padding={}", python_tuple(&padding))
            }
            Conv2dPadding::Valid => s += ", padding=valid",
            Conv2dPadding::Same => s += ", padding=same",
        }
        if o.dilation != [1, 1] {
            s += &format!(", dilation={}", python_tuple(&o.dilation));
        }
        if o.groups != 1 {
            s += &format!(", groups={}", o.groups);
        }
        if self.bias.is_none() {
            s += ", bias=False";
        }
        s
    }
}

impl Forward for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        Conv2d::forward(self, input)
    }
}

/// The sizes and options, like PyTorch's `repr`, which lists the padding, dilation, groups and
/// bias only when they aren't the defaults:
/// `Conv2d { in_channels: 3, out_channels: 8, kernel_size: [3, 3], stride: [1, 1], padding: Explicit([1, 1]), .. }`.
impl fmt::Debug for Conv2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Conv2d");
        s.field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.options.stride);
        if self.options.padding != Conv2dPadding::Explicit([0, 0]) {
            s.field("padding", &self.options.padding);
        }
        if self.options.dilation != [1, 1] {
            s.field("dilation", &self.options.dilation);
        }
        if self.options.groups != 1 {
            s.field("groups", &self.options.groups);
        }
        if self.bias.is_none() {
            s.field("bias", &false);
        }
        s.finish_non_exhaustive()
    }
}
