//! transformers' ResNet against the reference: the small repositories of
//! tests/golden/repos (written by tools/golden/gen_m3_resnet.py with transformers 5.20.2), loaded
//! through homura's core API alone, give the reference's outputs, statistics and gradients.

use std::path::{Path, PathBuf};

use homura::nn::functional::cross_entropy;
use homura::nn::{LoadOptions, Module, StateDict};
use homura::{DType, Device, Tensor};
use homura_models::resnet::{ResNetConfig, ResNetForImageClassification};

const REPOS: [&str; 4] = [
    "basic",
    "basic_no_labels",
    "bottleneck",
    "bottleneck_downsample",
];

fn golden(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(path)
}

/// How a repository's weights are read.
#[derive(Clone, Copy, Debug)]
enum Weights {
    Safetensors,
    PyTorch,
    /// `model.safetensors` mapped.
    Mapped,
}

fn state_dict(repo: &str, weights: Weights) -> StateDict {
    let dir = golden(&format!("repos/{repo}"));
    match weights {
        Weights::Safetensors => homura::load(dir.join("model.safetensors")).unwrap(),
        Weights::PyTorch => homura::load(dir.join("pytorch_model.bin")).unwrap(),
        // SAFETY: the fixtures aren't modified while the tests run.
        Weights::Mapped => unsafe { homura::io::load_mmap(dir.join("model.safetensors")) }.unwrap(),
    }
}

/// The model of `repo`, built on the meta device and assigned the weights, as `from_pretrained`
/// does with the core API.
fn load(repo: &str, weights: Weights) -> ResNetForImageClassification {
    let config = ResNetConfig::from_file(golden(&format!("repos/{repo}/config.json"))).unwrap();
    let model = homura::with_device(Device::Meta, || ResNetForImageClassification::new(&config));
    model
        .load_state_dict_with(
            &state_dict(repo, weights),
            LoadOptions::default().assign(true),
        )
        .unwrap();
    model
}

struct Expected(StateDict);

impl Expected {
    fn of(repo: &str) -> Expected {
        Expected(homura::load(golden(&format!("resnet/{repo}.safetensors"))).unwrap())
    }

    fn get(&self, key: &str) -> Tensor {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("no expected tensor {key}"))
            .clone()
    }
}

fn inputs() -> (Tensor, Tensor) {
    let sd = homura::load(golden("resnet/inputs.safetensors")).unwrap();
    (
        sd.get("pixel_values").unwrap().clone(),
        sd.get("target").unwrap().clone(),
    )
}

/// The largest `|found - expected| / (1 + |expected|)`.
fn error(found: &Tensor, expected: &Tensor) -> f64 {
    assert_eq!(found.shape(), expected.shape());
    assert_eq!(found.dtype(), expected.dtype());
    let found = found.to(DType::F64);
    let expected = expected.to(DType::F64);
    ((&found - &expected).abs() / (expected.abs() + 1.0))
        .max()
        .item::<f64>()
}

fn check(what: &str, found: &Tensor, expected: &Tensor, tolerance: f64) {
    let e = error(found, expected);
    println!("{what}: {e:e}");
    assert!(e <= tolerance, "{what}: error {e:e} above {tolerance:e}");
}

/// The largest errors measured on x86-64 (`cargo test -- --nocapture` prints them) are 5.4e-7
/// in F32 and 2.1e-15 in F64.
const F32_TOLERANCE: f64 = 1e-5;
const F64_TOLERANCE: f64 = 1e-13;

#[test]
fn eval_outputs_are_transformers() {
    let (x, _) = inputs();
    for repo in REPOS {
        let expected = Expected::of(repo);
        for (dtype, prefix, tolerance) in [
            (DType::F32, "eval_f32", F32_TOLERANCE),
            (DType::F64, "eval_f64", F64_TOLERANCE),
        ] {
            let model = load(repo, Weights::Safetensors);
            model.to(dtype);
            model.eval();
            let x = x.to(dtype);
            let (logits, output) =
                homura::with_no_grad(|| (model.forward(&x), model.resnet.forward(&x)));
            check(
                &format!("{repo} {prefix} logits"),
                &logits,
                &expected.get(&format!("{prefix}.logits")),
                tolerance,
            );
            check(
                &format!("{repo} {prefix} last_hidden_state"),
                &output.last_hidden_state,
                &expected.get(&format!("{prefix}.last_hidden_state")),
                tolerance,
            );
            check(
                &format!("{repo} {prefix} pooler_output"),
                &output.pooler_output,
                &expected.get(&format!("{prefix}.pooler_output")),
                tolerance,
            );
        }
    }
}

#[test]
fn every_weight_file_gives_the_same_model() {
    let (x, _) = inputs();
    for repo in REPOS {
        let logits: Vec<Vec<u8>> = [Weights::Safetensors, Weights::PyTorch, Weights::Mapped]
            .into_iter()
            .map(|weights| {
                let model = load(repo, weights);
                model.eval();
                let logits = homura::with_no_grad(|| model.forward(&x));
                logits
                    .to_vec::<f32>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            })
            .collect();
        assert_eq!(
            logits[0], logits[1],
            "{repo}: safetensors and pytorch_model.bin"
        );
        assert_eq!(logits[0], logits[2], "{repo}: copied and mapped");
        // The two files hold the same tensors (in other orders: safetensors sorts them).
        let a = state_dict(repo, Weights::Safetensors);
        let b = state_dict(repo, Weights::PyTorch);
        let sorted = |sd: &StateDict| {
            let mut keys: Vec<String> = sd.keys().map(String::from).collect();
            keys.sort();
            keys
        };
        assert_eq!(sorted(&a), sorted(&b), "{repo}");
        // Only the .pt file records the module versions.
        assert!(a.versions().is_none() && b.versions().is_some(), "{repo}");
    }
}

#[test]
fn train_mode_outputs_and_statistics_are_transformers() {
    let (x, _) = inputs();
    let x = x.to(DType::F64);
    for repo in REPOS {
        let expected = Expected::of(repo);
        let model = load(repo, Weights::PyTorch);
        model.to(DType::F64);
        model.train();
        let logits = model.forward(&x);
        check(
            &format!("{repo} train logits"),
            &logits,
            &expected.get("train.logits"),
            F64_TOLERANCE,
        );
        let sd = model.state_dict();
        let parameters: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        let mut buffers = 0;
        for (key, value) in sd.iter() {
            if parameters.iter().any(|p| p == key) {
                continue;
            }
            buffers += 1;
            let reference = expected.get(&format!("train.{key}"));
            if value.dtype() == DType::I64 {
                assert_eq!(
                    value.to_vec::<i64>(),
                    reference.to_vec::<i64>(),
                    "{repo} {key}"
                );
            } else {
                check(
                    &format!("{repo} train {key}"),
                    value,
                    &reference,
                    F64_TOLERANCE,
                );
            }
        }
        assert!(buffers > 0);
    }
}

/// In eval mode (in train mode, the batch statistics of 2 values make the gradients
/// ill-conditioned: the reference's change by up to 1e-3 when the weights change by a few ulps).
#[test]
fn gradients_are_transformers() {
    let (x, target) = inputs();
    let x = x.to(DType::F64);
    for repo in REPOS {
        let expected = Expected::of(repo);
        let model = load(repo, Weights::Safetensors);
        model.to(DType::F64);
        model.eval();
        cross_entropy(&model.forward(&x), &target, Default::default()).backward();
        for (key, parameter) in model.named_parameters() {
            let grad = parameter
                .grad()
                .unwrap_or_else(|| panic!("{repo}: no grad for {key}"));
            check(
                &format!("{repo} grad {key}"),
                &grad,
                &expected.get(&format!("grad.{key}")),
                F64_TOLERANCE,
            );
        }
    }
}

#[test]
fn no_labels_give_the_pooled_features() {
    let (x, _) = inputs();
    let model = load("basic_no_labels", Weights::Safetensors);
    model.eval();
    let logits = homura::with_no_grad(|| model.forward(&x));
    assert_eq!(logits.shape(), [2, 12]);
    let keys: Vec<_> = model
        .state_dict()
        .keys()
        .filter(|k| k.starts_with("classifier"))
        .map(String::from)
        .collect();
    assert!(keys.is_empty(), "{keys:?}");
    assert!(model.display().to_string().ends_with(
        "  (classifier): Sequential(\n    (0): Flatten(start_dim=1, end_dim=-1)\n    (1): Identity()\n  )\n)"
    ));
}
