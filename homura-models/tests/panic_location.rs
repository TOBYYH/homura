//! Panics of homura-models' public API report the caller's line (`#[track_caller]`): model
//! constructors with invalid configurations, forward computations with invalid inputs (directly,
//! through `Forward` and in a `Sequential`), and the panics of the core layers below them.

use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

use homura::Device;
use homura::nn::{Forward, Sequential};
use homura_models::activations::Activation;
use homura_models::resnet::{
    ResNetBasicLayer, ResNetBottleNeckLayer, ResNetConfig, ResNetConvLayer, ResNetEmbeddings,
    ResNetEncoder, ResNetForImageClassification, ResNetLayerType, ResNetModel, ResNetShortCut,
    ResNetShortCutOrIdentity, ResNetStage,
};
thread_local! {
    /// While `panic_location` runs: the location of the last panic on this thread.
    static CAUGHT: RefCell<Option<Option<(String, u32)>>> = const { RefCell::new(None) };
}

/// Installs, once for the whole process, a panic hook that records the location in the
/// panicking thread's slot while `panic_location` runs on that thread, and otherwise reports the
/// panic as usual. Tests run in parallel threads, so they must not replace the hook themselves.
fn install_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let default = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let location = info.location().map(|l| (l.file().to_string(), l.line()));
            let captured = CAUGHT.with(|c| match &mut *c.borrow_mut() {
                Some(slot) => {
                    *slot = location;
                    true
                }
                None => false,
            });
            if !captured {
                default(info);
            }
        }));
    });
}

/// Runs `f` and returns where it panicked, or `None` if it didn't.
fn panic_location(f: impl FnOnce()) -> Option<(String, u32)> {
    install_hook();
    CAUGHT.with(|c| *c.borrow_mut() = Some(None));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    let location = CAUGHT.with(|c| c.borrow_mut().take()).flatten();
    let panicked = result.is_err();
    panicked.then(|| location.expect("the panic had no location"))
}

/// Checks that evaluating the expression panics and that the panic points at this line of this
/// file; mismatches are collected and reported by `finish` at the end of the test, so that one
/// run lists them all. The expression must fit on the line of the macro call.
macro_rules! at {
    ($e:expr) => {{
        let line = line!();
        let mismatch = match panic_location(|| {
            let _ = $e;
        }) {
            None => Some("didn't panic".to_string()),
            Some((file, got)) if (file.as_str(), got) != (file!(), line) => {
                Some(format!("panicked at {file}:{got}"))
            }
            Some(_) => None,
        };
        if let Some(mismatch) = mismatch {
            let message = format!("line {line}: `{}` {mismatch}", stringify!($e));
            MISMATCHES.with(|m| m.borrow_mut().push(message));
        }
    }};
}

thread_local! {
    static MISMATCHES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Fails the test if any `at!` of this thread found a panic at another location.
fn finish() {
    let mismatches = MISMATCHES.with(|m| std::mem::take(&mut *m.borrow_mut()));
    assert!(
        mismatches.is_empty(),
        "panics at other locations:\n{}",
        mismatches.join("\n")
    );
}

fn config() -> ResNetConfig {
    ResNetConfig::default()
        .layer_type(ResNetLayerType::Basic)
        .embedding_size(4)
        .hidden_sizes([4, 8])
        .depths([1, 1])
}

#[test]
fn invalid_configurations() {
    let bad = config().depths([1]);
    at!(ResNetModel::new(&bad));
    at!(ResNetForImageClassification::new(&bad));
    at!(ResNetEncoder::new(&bad));
    at!(ResNetEmbeddings::new(&bad));
    at!(ResNetStage::new(&bad, 4, 4, 1, 1));
    at!(config().hidden_act("mish").activation());
    finish();
}

/// Constructions whose sizes the core `Conv2d` rejects, on `Device::Meta` so that nothing is
/// allocated: the location passes through the internal convolution helper (R116).
#[test]
fn core_constructor_failures() {
    let relu = Activation::from_name("relu").unwrap();
    let meta = |f: &dyn Fn()| homura::with_device(Device::Meta, f);
    let bottleneck = ResNetBottleNeckLayer::new;
    let n = usize::MAX;
    at!(meta(&|| drop(ResNetConvLayer::new(n, 2, 3, 1, relu))));
    at!(meta(&|| drop(ResNetShortCut::new(n, 2, 1))));
    at!(meta(&|| drop(ResNetShortCutOrIdentity::new(n, 2, 1))));
    at!(meta(&|| drop(ResNetBasicLayer::new(n, 2, 1, relu))));
    at!(meta(&|| drop(bottleneck(n, 8, 1, relu, false))));
    at!(meta(&|| drop(ResNetStage::new(&config(), n, 8, 1, 1))));
    for json in [
        r#"{"num_channels":18446744073709551615,"embedding_size":2,"hidden_sizes":[4],"depths":[1]}"#,
        r#"{"embedding_size":18446744073709551615,"hidden_sizes":[4],"depths":[1]}"#,
        r#"{"hidden_sizes":[18446744073709551615],"depths":[1]}"#,
    ] {
        let bad = ResNetConfig::from_json(json).unwrap();
        at!(meta(&|| drop(ResNetModel::new(&bad))));
        at!(meta(&|| drop(ResNetForImageClassification::new(&bad))));
    }
    finish();
}

#[test]
fn invalid_inputs() {
    let wrong_channels = homura::zeros([1, 5, 8, 8]);
    let relu = Activation::from_name("relu").unwrap();
    let model = ResNetForImageClassification::new(&config());
    at!(model.forward(&wrong_channels));
    at!(Forward::forward(&model, &wrong_channels));
    at!(model.resnet.forward(&wrong_channels));
    at!(model.resnet.embedder.forward(&homura::zeros([3])));
    let boxed: Box<dyn Forward + Send + Sync> =
        Box::new(ResNetForImageClassification::new(&config()));
    at!(boxed.forward(&wrong_channels));
    let net = Sequential::new().append(ResNetForImageClassification::new(&config()));
    at!(net.forward(&wrong_channels));
    // Panics of the core layers inside.
    at!(model.resnet.encoder.forward(&wrong_channels));
    at!(ResNetStage::new(&config(), 4, 8, 2, 1).forward(&wrong_channels));
    at!(ResNetBasicLayer::new(4, 8, 2, relu).forward(&wrong_channels));
    at!(ResNetBottleNeckLayer::new(4, 8, 2, relu, false).forward(&wrong_channels));
    at!(ResNetConvLayer::new(4, 8, 3, 1, relu).forward(&wrong_channels));
    at!(ResNetShortCut::new(4, 8, 2).forward(&wrong_channels));
    at!(ResNetShortCutOrIdentity::new(4, 8, 2).forward(&wrong_channels));
    at!(relu.forward(&homura::tensor([true])));
    finish();
}
