use homura::Tensor;
use homura::nn::init::{self, FanMode, Nonlinearity};
use homura::nn::{
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Flatten, Forward, Identity, Linear, MaxPool2d, Member,
    Module, Sequential, Visitor,
};

use super::{ResNetConfig, ResNetLayerType};
use crate::activations::Activation;
use crate::pretrained::FromPretrained;

/// A convolution without bias with `kernel_size / 2` padding, initialized as transformers'
/// `_init_weights` does (`kaiming_normal_(mode="fan_out", nonlinearity="relu")`).
#[track_caller]
fn convolution(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
) -> Conv2d {
    let conv = Conv2d::new(in_channels, out_channels, kernel_size)
        .stride(stride)
        .padding(kernel_size / 2)
        .bias(false);
    init::kaiming_normal_(&conv.weight, 0.0, FanMode::FanOut, Nonlinearity::Relu);
    conv
}

/// A convolution, batch normalization and activation: transformers' `ResNetConvLayer`.
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetConvLayer {
    /// The convolution, without bias, with `kernel_size / 2` padding.
    pub convolution: Conv2d,
    /// The batch normalization of its output.
    pub normalization: BatchNorm2d,
    /// The activation, or [`Activation::Identity`] for none.
    pub activation: Activation,
}

impl ResNetConvLayer {
    /// The layer from `in_channels` to `out_channels` with a square kernel of `kernel_size` and
    /// `stride` (`ResNetConvLayer(in_channels, out_channels, kernel_size, stride,
    /// activation=...)`; `Activation::Identity` for `activation=None`).
    #[track_caller]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        activation: Activation,
    ) -> ResNetConvLayer {
        ResNetConvLayer {
            convolution: convolution(in_channels, out_channels, kernel_size, stride),
            normalization: BatchNorm2d::new(out_channels),
            activation,
        }
    }

    /// The activation of the normalized convolution of `input`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.convolution.forward(input);
        let x = self.normalization.forward(&x);
        self.activation.forward(&x)
    }
}

impl Forward for ResNetConvLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetConvLayer::forward(self, input)
    }
}

/// The stem: a 7 × 7 convolution with a stride of 2 and a 3 × 3 max pooling with a stride of 2,
/// transformers' `ResNetEmbeddings`.
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetEmbeddings {
    /// The convolution from `num_channels` to `embedding_size` channels.
    pub embedder: ResNetConvLayer,
    /// `MaxPool2d(3, 2, 1)`.
    pub pooler: MaxPool2d,
    /// The channels of the input.
    pub num_channels: usize,
}

impl ResNetEmbeddings {
    /// The stem of `config`'s models.
    ///
    /// # Panics
    ///
    /// Panics if `config` isn't valid ([`ResNetConfig::validate`]).
    #[track_caller]
    pub fn new(config: &ResNetConfig) -> ResNetEmbeddings {
        let config = config.checked();
        ResNetEmbeddings {
            embedder: ResNetConvLayer::new(
                config.num_channels,
                config.embedding_size,
                7,
                2,
                config.activation(),
            ),
            pooler: MaxPool2d::new(3).stride(2).padding(1),
            num_channels: config.num_channels,
        }
    }

    /// The pooled features of `pixel_values`, `[N, num_channels, H, W]`.
    ///
    /// # Panics
    ///
    /// Panics with transformers' message if `pixel_values` doesn't have `num_channels` channels.
    #[track_caller]
    pub fn forward(&self, pixel_values: &Tensor) -> Tensor {
        if pixel_values.size(1) != self.num_channels {
            panic!(
                "Make sure that the channel dimension of the pixel values match with the one set \
                 in the configuration."
            );
        }
        self.pooler.forward(&self.embedder.forward(pixel_values))
    }
}

impl Forward for ResNetEmbeddings {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetEmbeddings::forward(self, input)
    }
}

/// The projection of a layer's input to its output's channels and size: a 1 × 1 convolution and
/// a batch normalization, transformers' `ResNetShortCut`.
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetShortCut {
    /// The 1 × 1 convolution without bias.
    pub convolution: Conv2d,
    /// The batch normalization of its output.
    pub normalization: BatchNorm2d,
}

impl ResNetShortCut {
    /// The projection from `in_channels` to `out_channels` with `stride`.
    #[track_caller]
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> ResNetShortCut {
        ResNetShortCut {
            convolution: convolution(in_channels, out_channels, 1, stride),
            normalization: BatchNorm2d::new(out_channels),
        }
    }

    /// The projection of `input`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.normalization.forward(&self.convolution.forward(input))
    }
}

impl Forward for ResNetShortCut {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetShortCut::forward(self, input)
    }
}

/// The shortcut of a residual layer: a [`ResNetShortCut`] where the layer changes the channels
/// or the size, an [`Identity`] otherwise, as in transformers. As a module it is the one it
/// holds.
#[derive(Debug)]
#[non_exhaustive]
pub enum ResNetShortCutOrIdentity {
    /// A projection.
    ShortCut(ResNetShortCut),
    /// No projection.
    Identity(Identity),
}

impl ResNetShortCutOrIdentity {
    /// The shortcut of a layer from `in_channels` to `out_channels` with `stride`: a projection
    /// if the channels differ or the stride isn't 1.
    #[track_caller]
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> ResNetShortCutOrIdentity {
        if in_channels != out_channels || stride != 1 {
            ResNetShortCutOrIdentity::ShortCut(ResNetShortCut::new(
                in_channels,
                out_channels,
                stride,
            ))
        } else {
            ResNetShortCutOrIdentity::Identity(Identity)
        }
    }

    /// The shortcut of `input`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            ResNetShortCutOrIdentity::ShortCut(shortcut) => shortcut.forward(input),
            ResNetShortCutOrIdentity::Identity(identity) => identity.forward(input),
        }
    }

    fn module(&self) -> &dyn Module {
        match self {
            ResNetShortCutOrIdentity::ShortCut(shortcut) => shortcut,
            ResNetShortCutOrIdentity::Identity(identity) => identity,
        }
    }
}

impl Forward for ResNetShortCutOrIdentity {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetShortCutOrIdentity::forward(self, input)
    }
}

impl Module for ResNetShortCutOrIdentity {
    fn visit<'a>(&'a self, v: &mut dyn Visitor<'a>) {
        self.module().visit(v);
    }

    fn extra_repr(&self) -> String {
        self.module().extra_repr()
    }

    fn type_name(&self) -> &'static str {
        self.module().type_name()
    }

    fn version(&self) -> u32 {
        self.module().version()
    }

    fn added_in(&self, name: &str) -> Option<u32> {
        self.module().added_in(name)
    }
}

impl Member for ResNetShortCutOrIdentity {
    fn visit_as<'a>(&'a self, name: &str, v: &mut dyn Visitor<'a>) {
        v.module(name, self, None);
    }
}

/// The residual output: `activation(layer(input) + shortcut(input))`. transformers adds in
/// place (`hidden_state += residual`); homura makes a new tensor, with the same values.
#[track_caller]
fn residual(
    layer: &Sequential,
    shortcut: &ResNetShortCutOrIdentity,
    activation: &Activation,
    input: &Tensor,
) -> Tensor {
    let x = layer.forward(input);
    let residual = shortcut.forward(input);
    activation.forward(&(&x + &residual))
}

/// The residual layer of ResNet-18 and -34: two 3 × 3 convolutions, transformers'
/// `ResNetBasicLayer`.
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetBasicLayer {
    /// The shortcut.
    pub shortcut: ResNetShortCutOrIdentity,
    /// The two [`ResNetConvLayer`]s: the first with the stride and a ReLU, the second without
    /// activation.
    pub layer: Sequential,
    /// The activation of the sum.
    pub activation: Activation,
}

impl ResNetBasicLayer {
    /// The layer from `in_channels` to `out_channels` with `stride` and the output's
    /// `activation` (`ResNetBasicLayer(in_channels, out_channels, stride, activation)`).
    #[track_caller]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        activation: Activation,
    ) -> ResNetBasicLayer {
        let relu = Activation::from_name("relu").unwrap();
        let none = Activation::Identity(Identity);
        ResNetBasicLayer {
            shortcut: ResNetShortCutOrIdentity::new(in_channels, out_channels, stride),
            layer: Sequential::new()
                .append(ResNetConvLayer::new(
                    in_channels,
                    out_channels,
                    3,
                    stride,
                    relu,
                ))
                .append(ResNetConvLayer::new(out_channels, out_channels, 3, 1, none)),
            activation,
        }
    }

    /// The output of the layer for `input`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        residual(&self.layer, &self.shortcut, &self.activation, input)
    }
}

impl Forward for ResNetBasicLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetBasicLayer::forward(self, input)
    }
}

/// The residual layer of ResNet-50 and larger: a 1 × 1 convolution to a quarter of the output
/// channels, a 3 × 3 convolution and a 1 × 1 convolution to the output channels, transformers'
/// `ResNetBottleNeckLayer`.
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetBottleNeckLayer {
    /// The shortcut.
    pub shortcut: ResNetShortCutOrIdentity,
    /// The three [`ResNetConvLayer`]s, the first two with a ReLU, the last without activation.
    pub layer: Sequential,
    /// The activation of the sum.
    pub activation: Activation,
}

impl ResNetBottleNeckLayer {
    /// The layer from `in_channels` to `out_channels` with `stride` and the output's
    /// `activation`; the stride is on the 3 × 3 convolution, or on the first 1 × 1 one if
    /// `downsample_in_bottleneck` (`ResNetBottleNeckLayer(in_channels, out_channels, stride,
    /// activation, downsample_in_bottleneck=...)`).
    #[track_caller]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        activation: Activation,
        downsample_in_bottleneck: bool,
    ) -> ResNetBottleNeckLayer {
        let reduced = out_channels / 4;
        let (first_stride, middle_stride) = if downsample_in_bottleneck {
            (stride, 1)
        } else {
            (1, stride)
        };
        let relu = Activation::from_name("relu").unwrap();
        let none = Activation::Identity(Identity);
        ResNetBottleNeckLayer {
            shortcut: ResNetShortCutOrIdentity::new(in_channels, out_channels, stride),
            layer: Sequential::new()
                .append(ResNetConvLayer::new(
                    in_channels,
                    reduced,
                    1,
                    first_stride,
                    relu,
                ))
                .append(ResNetConvLayer::new(
                    reduced,
                    reduced,
                    3,
                    middle_stride,
                    relu,
                ))
                .append(ResNetConvLayer::new(reduced, out_channels, 1, 1, none)),
            activation,
        }
    }

    /// The output of the layer for `input`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        residual(&self.layer, &self.shortcut, &self.activation, input)
    }
}

impl Forward for ResNetBottleNeckLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetBottleNeckLayer::forward(self, input)
    }
}

/// The layers of one stage, transformers' `ResNetStage`.
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetStage {
    /// The [`ResNetBasicLayer`]s or [`ResNetBottleNeckLayer`]s, the first with the stride.
    pub layers: Sequential,
}

impl ResNetStage {
    /// The stage of `config`'s layer type from `in_channels` to `out_channels` whose first layer
    /// has `stride`, with `depth` layers (at least one, as in transformers).
    ///
    /// # Panics
    ///
    /// Panics if `config` isn't valid ([`ResNetConfig::validate`]).
    #[track_caller]
    pub fn new(
        config: &ResNetConfig,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        depth: usize,
    ) -> ResNetStage {
        let config = config.checked();
        let activation = config.activation();
        let mut layers = Sequential::new();
        for i in 0..depth.max(1) {
            let (in_channels, stride) = if i == 0 {
                (in_channels, stride)
            } else {
                (out_channels, 1)
            };
            match config.layer_type {
                ResNetLayerType::Basic => layers.push(ResNetBasicLayer::new(
                    in_channels,
                    out_channels,
                    stride,
                    activation,
                )),
                ResNetLayerType::Bottleneck => layers.push(ResNetBottleNeckLayer::new(
                    in_channels,
                    out_channels,
                    stride,
                    activation,
                    config.downsample_in_bottleneck,
                )),
            }
        }
        ResNetStage { layers }
    }

    /// The output of the stage for `input`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.layers.forward(input)
    }
}

impl Forward for ResNetStage {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetStage::forward(self, input)
    }
}

/// The stages, transformers' `ResNetEncoder`.
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetEncoder {
    /// One stage for each hidden size: the first from `embedding_size` channels, with a stride
    /// of 2 if `downsample_in_first_stage`, the others from the previous stage's channels with a
    /// stride of 2.
    pub stages: Vec<ResNetStage>,
}

impl ResNetEncoder {
    /// The stages of `config`.
    ///
    /// # Panics
    ///
    /// Panics if `config` isn't valid ([`ResNetConfig::validate`]).
    #[track_caller]
    pub fn new(config: &ResNetConfig) -> ResNetEncoder {
        let config = config.checked();
        let first_stride = if config.downsample_in_first_stage {
            2
        } else {
            1
        };
        let mut stages = vec![ResNetStage::new(
            config,
            config.embedding_size,
            config.hidden_sizes[0],
            first_stride,
            config.depths[0],
        )];
        for (sizes, &depth) in config.hidden_sizes.windows(2).zip(&config.depths[1..]) {
            stages.push(ResNetStage::new(config, sizes[0], sizes[1], 2, depth));
        }
        ResNetEncoder { stages }
    }

    /// The last hidden state for the stem's output `input`. transformers returns it in an output
    /// object, which can also hold every stage's output (`output_hidden_states`); homura doesn't
    /// return those.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for stage in &self.stages {
            x = stage.forward(&x);
        }
        x
    }
}

impl Forward for ResNetEncoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetEncoder::forward(self, input)
    }
}

/// The outputs of [`ResNetModel`], transformers' `BaseModelOutputWithPoolingAndNoAttention`
/// without the hidden states of each stage.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ResNetModelOutput {
    /// The last stage's output, `[N, hidden_sizes[-1], h, w]`.
    pub last_hidden_state: Tensor,
    /// Its average over the height and width, `[N, hidden_sizes[-1], 1, 1]`.
    pub pooler_output: Tensor,
}

/// The ResNet without classification head, transformers' `ResNetModel`: the features of images.
///
/// Its forward computation returns two tensors, so it doesn't implement [`Forward`]:
///
/// ```compile_fail
/// fn layer<M: homura::nn::Forward>(_: &M) {}
/// let config = homura_models::resnet::ResNetConfig::default();
/// layer(&homura_models::resnet::ResNetModel::new(&config));
/// ```
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetModel {
    /// The stem.
    pub embedder: ResNetEmbeddings,
    /// The stages.
    pub encoder: ResNetEncoder,
    /// `AdaptiveAvgPool2d((1, 1))`.
    pub pooler: AdaptiveAvgPool2d,
}

impl ResNetModel {
    /// The model of `config`, initialized as transformers does: the convolutions with
    /// `kaiming_normal_(mode="fan_out", nonlinearity="relu")`, the batch normalizations with
    /// weights 1, biases 0 and reset statistics. Build it on the meta device
    /// (`homura::with_device(Device::Meta, || ResNetModel::new(&config))`) to load weights
    /// without initializing it.
    ///
    /// # Panics
    ///
    /// Panics if `config` isn't valid ([`ResNetConfig::validate`]).
    #[track_caller]
    pub fn new(config: &ResNetConfig) -> ResNetModel {
        let config = config.checked();
        ResNetModel {
            embedder: ResNetEmbeddings::new(config),
            encoder: ResNetEncoder::new(config),
            pooler: AdaptiveAvgPool2d::new(1),
        }
    }

    /// The features of `pixel_values`, `[N, num_channels, H, W]`.
    ///
    /// # Panics
    ///
    /// Panics with transformers' message if `pixel_values` doesn't have `num_channels` channels.
    #[track_caller]
    pub fn forward(&self, pixel_values: &Tensor) -> ResNetModelOutput {
        let last_hidden_state = self.encoder.forward(&self.embedder.forward(pixel_values));
        let pooler_output = self.pooler.forward(&last_hidden_state);
        ResNetModelOutput {
            last_hidden_state,
            pooler_output,
        }
    }
}

/// The ResNet with a classification head, transformers' `ResNetForImageClassification`, as in
/// `microsoft/resnet-50`.
#[derive(Module, Debug)]
#[non_exhaustive]
pub struct ResNetForImageClassification {
    /// The features.
    pub resnet: ResNetModel,
    /// `Sequential(Flatten, Linear(hidden_sizes[-1], labels))`; with no labels the second layer
    /// is an [`Identity`], so that the logits are the pooled features, `[N, hidden_sizes[-1]]`.
    pub classifier: Sequential,
}

impl ResNetForImageClassification {
    /// The model of `config`, with one output for each of its labels, initialized as
    /// [`ResNetModel::new`] says and the linear layer as PyTorch's `nn.Linear`.
    ///
    /// # Panics
    ///
    /// Panics if `config` isn't valid ([`ResNetConfig::validate`]).
    #[track_caller]
    pub fn new(config: &ResNetConfig) -> ResNetForImageClassification {
        let config = config.checked();
        let labels = config.common.num_labels();
        let features = *config.hidden_sizes.last().unwrap();
        let mut classifier = Sequential::new().append(Flatten::default());
        if labels > 0 {
            classifier.push(Linear::new(features, labels));
        } else {
            classifier.push(Identity);
        }
        ResNetForImageClassification {
            resnet: ResNetModel::new(config),
            classifier,
        }
    }

    /// The logits of `pixel_values`, `[N, labels]` (transformers' `outputs.logits`; the loss
    /// for `labels` and the hidden states aren't computed).
    ///
    /// # Panics
    ///
    /// Panics with transformers' message if `pixel_values` doesn't have `num_channels` channels.
    #[track_caller]
    pub fn forward(&self, pixel_values: &Tensor) -> Tensor {
        self.classifier
            .forward(&self.resnet.forward(pixel_values).pooler_output)
    }
}

impl Forward for ResNetForImageClassification {
    fn forward(&self, input: &Tensor) -> Tensor {
        ResNetForImageClassification::forward(self, input)
    }
}

impl FromPretrained for ResNetModel {
    type Config = ResNetConfig;
    const BASE_MODEL_PREFIX: &'static str = "resnet";

    fn from_config(config: &ResNetConfig) -> ResNetModel {
        ResNetModel::new(config)
    }
}

impl FromPretrained for ResNetForImageClassification {
    type Config = ResNetConfig;
    const BASE_MODEL_PREFIX: &'static str = "resnet";

    fn from_config(config: &ResNetConfig) -> ResNetForImageClassification {
        ResNetForImageClassification::new(config)
    }
}
