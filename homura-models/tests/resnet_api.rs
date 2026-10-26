//! The ResNet types as callers use them: the `Forward` trait where transformers' forward returns
//! one tensor, initialization, modes, activations and the panics for invalid configurations and
//! inputs.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::nn::{Forward, Identity, Module, Sequential};
use homura::{DType, Device, Tensor};
use homura_models::activations::Activation;
use homura_models::resnet::{
    ResNetBasicLayer, ResNetBottleNeckLayer, ResNetConfig, ResNetConvLayer, ResNetEmbeddings,
    ResNetEncoder, ResNetForImageClassification, ResNetLayerType, ResNetModel, ResNetShortCut,
    ResNetShortCutOrIdentity, ResNetStage,
};

fn small() -> ResNetConfig {
    ResNetConfig::default()
        .layer_type(ResNetLayerType::Basic)
        .embedding_size(8)
        .hidden_sizes([8, 16])
        .depths([1, 2])
        .num_labels(5)
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("no panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_vec::<f32>().iter().map(|v| v.to_bits()).collect()
}

fn layer_output<M: Forward>(layer: &M, x: &Tensor) -> Tensor {
    layer.forward(x)
}

#[test]
fn the_classification_model_is_a_layer() {
    let model = ResNetForImageClassification::new(&small());
    model.eval();
    let x = homura::randn([2, 3, 32, 32]);
    let logits = model.forward(&x);
    assert_eq!(logits.shape(), [2, 5]);
    assert_eq!(bits(&layer_output(&model, &x)), bits(&logits));
    let boxed: Box<dyn Forward + Send + Sync> = Box::new(model);
    assert_eq!(bits(&boxed.forward(&x)), bits(&logits));
    // In a Sequential, as the layer before a softmax.
    let net = Sequential::new()
        .append(ResNetForImageClassification::new(&small()))
        .append(homura::nn::Softmax::new(-1));
    assert_eq!(net.forward(&x).shape(), [2, 5]);
    assert!(
        net.state_dict()
            .keys()
            .any(|k| k == "0.classifier.1.weight")
    );
}

#[test]
fn inner_modules_are_layers() {
    let config = small();
    let x = homura::randn([2, 8, 8, 8]);
    let relu = Activation::from_name("relu").unwrap();
    let layers: Vec<(Box<dyn Forward + Send + Sync>, Vec<usize>)> = vec![
        (
            Box::new(ResNetConvLayer::new(8, 4, 3, 2, relu)),
            vec![2, 4, 4, 4],
        ),
        (Box::new(ResNetShortCut::new(8, 16, 2)), vec![2, 16, 4, 4]),
        (
            Box::new(ResNetShortCutOrIdentity::new(8, 8, 1)),
            vec![2, 8, 8, 8],
        ),
        (
            Box::new(ResNetBasicLayer::new(8, 16, 2, relu)),
            vec![2, 16, 4, 4],
        ),
        (
            Box::new(ResNetBottleNeckLayer::new(8, 16, 2, relu, true)),
            vec![2, 16, 4, 4],
        ),
        (
            Box::new(ResNetStage::new(&config, 8, 16, 2, 2)),
            vec![2, 16, 4, 4],
        ),
        (Box::new(ResNetEncoder::new(&config)), vec![2, 16, 4, 4]),
    ];
    for (layer, shape) in layers {
        assert_eq!(layer.forward(&x).shape(), shape, "{}", layer.type_name());
    }
    let pixels = homura::randn([1, 3, 16, 16]);
    assert_eq!(
        layer_output(&ResNetEmbeddings::new(&config), &pixels).shape(),
        [1, 8, 4, 4]
    );
    // The identity shortcut returns its input.
    let identity = ResNetShortCutOrIdentity::new(8, 8, 1);
    assert!(matches!(identity, ResNetShortCutOrIdentity::Identity(_)));
    assert!(identity.forward(&x).is_same(&x));
    assert!(matches!(
        ResNetShortCutOrIdentity::new(8, 8, 2),
        ResNetShortCutOrIdentity::ShortCut(_)
    ));
}

#[test]
fn the_base_model_returns_its_two_outputs() {
    let model = ResNetModel::new(&small());
    model.eval();
    let x = homura::randn([2, 3, 32, 32]);
    let output = model.forward(&x);
    assert_eq!(output.last_hidden_state.shape(), [2, 16, 4, 4]);
    assert_eq!(output.pooler_output.shape(), [2, 16, 1, 1]);
    let mean = output.last_hidden_state.mean_dim([2, 3], true);
    assert!((&mean - &output.pooler_output).abs().max().item::<f32>() < 1e-6);
    // The classification model's logits are the classifier of its pooled features.
    let clf = ResNetForImageClassification::new(&small());
    clf.eval();
    let pooled = clf.resnet.forward(&x).pooler_output;
    assert_eq!(
        bits(&clf.classifier.forward(&pooled)),
        bits(&clf.forward(&x))
    );
}

#[test]
fn modes_reach_every_batch_normalization() {
    let model = ResNetForImageClassification::new(&small().downsample_in_first_stage(true));
    let norms = model
        .named_modules()
        .into_iter()
        .filter(|(_, module)| module.type_name() == "BatchNorm2d")
        .count();
    // 1 in the stem, 2 + 1 (shortcut) in stage 0, 2 + 1 + 2 in stage 1.
    assert_eq!(norms, 9);
    let x = homura::randn([2, 3, 32, 32]);
    model.eval();
    let a = model.forward(&x);
    let b = model.forward(&x);
    assert_eq!(bits(&a), bits(&b), "eval mode doesn't update statistics");
    model.train();
    let before = model.state_dict();
    let key = "resnet.encoder.stages.0.layers.0.shortcut.normalization.num_batches_tracked";
    assert_eq!(before.get(key).unwrap().item::<i64>(), 0);
    model.forward(&x);
    assert_eq!(model.state_dict().get(key).unwrap().item::<i64>(), 1);
}

#[test]
fn initialization_is_transformers() {
    let config = ResNetConfig::default()
        .layer_type(ResNetLayerType::Basic)
        .embedding_size(64)
        .hidden_sizes([64])
        .depths([1])
        .num_labels(100);
    let model = ResNetForImageClassification::new(&config);
    // kaiming_normal_(mode="fan_out", nonlinearity="relu"): std √(2 / (out × kh × kw)).
    let weight = &model.resnet.embedder.embedder.convolution.weight;
    let expected = (2.0 / (64.0 * 49.0f64)).sqrt();
    let std = weight.to(DType::F64).std(1).item::<f64>();
    assert!((std / expected - 1.0).abs() < 0.05, "{std} {expected}");
    let mean = weight.to(DType::F64).mean().item::<f64>();
    assert!(mean.abs() < 0.1 * expected, "{mean}");
    for (key, tensor) in model.state_dict().iter() {
        let values = tensor.to(DType::F64).to_vec::<f64>();
        let all = |v: f64| values.iter().all(|&x| x == v);
        if key.ends_with("normalization.weight") || key.ends_with("running_var") {
            assert!(all(1.0), "{key}");
        } else if key.ends_with("normalization.bias")
            || key.ends_with("running_mean")
            || key.ends_with("num_batches_tracked")
        {
            assert!(all(0.0), "{key}");
        }
        if key.ends_with("convolution.weight") {
            assert!(tensor.std(1).item::<f32>() > 0.0, "{key}");
        }
    }
    // The linear layer: U(-1/√64, 1/√64) for both.
    let sd = model.state_dict();
    for key in ["classifier.1.weight", "classifier.1.bias"] {
        let values = sd.get(key).unwrap().to_vec::<f32>();
        assert!(values.iter().all(|v| v.abs() <= 0.125), "{key}");
        assert!(values.iter().any(|v| v.abs() > 0.1), "{key}");
    }
    // On the meta device nothing is computed.
    let meta = homura::with_device(Device::Meta, || ResNetForImageClassification::new(&config));
    assert!(meta.parameters().iter().all(|p| p.device() == Device::Meta));
}

#[test]
fn activations_are_the_core_layers() {
    let x = homura::linspace(-4.0, 4.0, 101).to(DType::F64);
    let cases: [(&str, Tensor, &str); 7] = [
        ("relu", x.relu(), "Relu()"),
        ("sigmoid", x.sigmoid(), "Sigmoid()"),
        ("tanh", x.tanh(), "Tanh()"),
        ("silu", x.silu(), "Silu()"),
        ("swish", x.silu(), "Silu()"),
        (
            "gelu",
            x.gelu(homura::GeluApproximate::None),
            "Gelu(approximate='none')",
        ),
        (
            "gelu_pytorch_tanh",
            x.gelu(homura::GeluApproximate::Tanh),
            "Gelu(approximate='tanh')",
        ),
    ];
    assert_eq!(cases.each_ref().map(|c| c.0), Activation::NAMES);
    for (name, expected, text) in cases {
        let activation = Activation::from_name(name).unwrap();
        assert_eq!(
            activation.forward(&x).to_vec::<f64>(),
            expected.to_vec::<f64>(),
            "{name}"
        );
        assert_eq!(activation.display().to_string(), text);
        assert_eq!(activation.version(), 1);
        assert!(activation.state_dict().is_empty());
    }
    let none = Activation::Identity(Identity);
    assert!(none.forward(&x).is_same(&x));
    assert_eq!(none.display().to_string(), "Identity()");
    for name in ["gelu_new", "mish", "RELU", ""] {
        assert!(Activation::from_name(name).is_none(), "{name}");
    }
    // The model's activations follow `hidden_act`; the inner convolutions keep ReLU.
    let model = ResNetModel::new(&small().hidden_act("gelu"));
    let text = model.display().to_string();
    assert_eq!(text.matches("Gelu(approximate='none')").count(), 1 + 3);
    assert_eq!(text.matches("Relu()").count(), 3);
}

#[test]
fn invalid_configurations_and_inputs_panic() {
    let invalid = small().depths([1]);
    for build in [
        Box::new(|| drop(ResNetModel::new(&invalid))) as Box<dyn Fn()>,
        Box::new(|| drop(ResNetForImageClassification::new(&invalid))),
        Box::new(|| drop(ResNetEncoder::new(&invalid))),
        Box::new(|| drop(ResNetEmbeddings::new(&invalid))),
        Box::new(|| drop(ResNetStage::new(&invalid, 8, 8, 1, 1))),
    ] {
        let message = panic_message(build);
        assert!(
            message.starts_with("invalid ResNet configuration: `hidden_sizes` and `depths` must"),
            "{message}"
        );
    }
    let message = panic_message(|| {
        small().hidden_act("mish").activation();
    });
    assert!(
        message.contains("`hidden_act` \"mish\" isn't one of"),
        "{message}"
    );
    let model = ResNetModel::new(&small());
    let message = panic_message(|| drop(model.forward(&homura::zeros([1, 4, 32, 32]))));
    assert_eq!(
        message,
        "Make sure that the channel dimension of the pixel values match with the one set in the \
         configuration."
    );
}
