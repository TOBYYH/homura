//! Saving the transformers ResNet in PyTorch's format (M3 step 5): the state dict that
//! `homura::save` writes to `pytorch_model.bin`, next to the repository's `config.json`, makes a
//! repository that `from_pretrained` loads into the same model, bit for bit.
//!
//! The repositories are written to target/tmp/m3_pt_written/resnet/, where
//! tools/golden/check_m3_pt_written.py loads them into transformers 5.20.2. The one of
//! microsoft/resnet-18 needs the Hugging Face cache of tools/golden/fetch_hf_repos.py and a
//! release build (the test is ignored by default):
//!
//! ```text
//! HF_HUB_CACHE=$PWD/target/hf-cache cargo test --release -p homura-models --test resnet_pt_write -- --ignored
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use homura::Tensor;
use homura::nn::{Module, StateDict};
use homura_models::hub::{Repo, Revision, WeightFormat};
use homura_models::pretrained::{FromPretrained, PretrainedOptions};
use homura_models::resnet::ResNetForImageClassification;

fn golden(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(path)
}

/// Asserts that two tensors have the same dtype, shape and bits.
#[track_caller]
fn assert_bits(what: &str, a: &Tensor, b: &Tensor) {
    let bits = |t: &Tensor| homura::save_safetensors_to_bytes([("t", t)]).unwrap();
    assert!(bits(a) == bits(b), "{what}");
}

/// Saves the state dict of the model of `repo` as `<out>/pytorch_model.bin` with the repository's
/// `config.json`, loads that repository, and checks that it is the same model.
fn save_and_reload(repo: &Repo, out: &Path, x: &Tensor) {
    let model =
        ResNetForImageClassification::from_pretrained(repo, PretrainedOptions::default()).unwrap();
    let _ = fs::remove_dir_all(out);
    fs::create_dir_all(out).unwrap();
    fs::copy(repo.dir().join("config.json"), out.join("config.json")).unwrap();
    let sd = model.state_dict();
    homura::save(&sd, out.join("pytorch_model.bin")).unwrap();

    let saved = Repo::local(out).unwrap();
    let options = PretrainedOptions::default().format(WeightFormat::PyTorch);
    let loaded = ResNetForImageClassification::from_pretrained_with_info(&saved, options).unwrap();
    assert_eq!(
        loaded.report.weights.path.file_name().unwrap(),
        "pytorch_model.bin"
    );
    let back = loaded.model.state_dict();
    assert_eq!(
        back.keys().collect::<Vec<_>>(),
        sd.keys().collect::<Vec<_>>()
    );
    for ((key, a), (_, b)) in back.iter().zip(&sd) {
        assert_bits(key, a, b);
    }
    // The file has the versions of every module, as PyTorch's `_metadata`.
    let file: StateDict = homura::load(out.join("pytorch_model.bin")).unwrap();
    assert_eq!(file.versions(), sd.versions());
    assert!(file.versions().is_some_and(|v| v.get("") == Some(1)));
    let logits = |m: &ResNetForImageClassification| homura::with_no_grad(|| m.forward(x));
    assert_bits("logits", &logits(&loaded.model), &logits(&model));
}

#[test]
fn saved_mini_models_load_back() {
    let x = homura::load(golden("resnet/inputs.safetensors")).unwrap()["pixel_values"].clone();
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("m3_pt_written/resnet");
    for name in [
        "basic",
        "basic_no_labels",
        "bottleneck",
        "bottleneck_downsample",
    ] {
        let repo = Repo::local(golden(&format!("repos/{name}"))).unwrap();
        save_and_reload(&repo, &out.join(name), &x);
    }
}

#[test]
#[ignore = "needs microsoft/resnet-18 in the Hugging Face cache and a release build"]
fn saved_resnet_18_loads_back() {
    if cfg!(debug_assertions) {
        eprintln!("skipped: run in a release build");
        return;
    }
    let repo = Repo::cached("microsoft/resnet-18", Revision::main()).unwrap();
    let x = homura::randn([1, 3, 224, 224]);
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("m3_pt_written/resnet-18");
    save_and_reload(&repo, &out, &x);
}
