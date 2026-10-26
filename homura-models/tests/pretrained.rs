//! `FromPretrained` on the small repositories of tests/golden/repos, in local directories and in a
//! cache laid out as huggingface_hub lays it out: the models give the outputs of the same weights
//! loaded with the core API (bit for bit) and of the reference, through every reading path; the
//! base model's prefix, the dtype, `transformers_weights` and the errors follow the rules that
//! tools/golden/check_m3_hub.py checks in transformers 5.20.2.

use std::fs;
use std::path::{Path, PathBuf};

use homura::nn::{LoadOptions, Module, StateDict};
use homura::{DType, Device, Tensor};
use homura_models::hub::{Repo, Revision, WeightFormat};
use homura_models::pretrained::{
    FromPretrained, LoadedModel, PretrainedConfig, PretrainedError, PretrainedOptions,
};
use homura_models::resnet::{ResNetConfig, ResNetForImageClassification, ResNetModel};
use serde_json::Value;

const HASH: &str = "0123456789abcdef0123456789abcdef01234567";
const HEAD: [&str; 2] = ["classifier.1.weight", "classifier.1.bias"];

fn golden(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(path)
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("pretrained")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A copy of the golden repository `name` in a scratch directory `dir`.
fn copy_repo(name: &str, dir: &str) -> PathBuf {
    let to = scratch(dir);
    for entry in fs::read_dir(golden(&format!("repos/{name}"))).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
    to
}

/// Changes the fields of `dir/config.json` (`Value::Null` removes one).
fn edit_config(dir: &Path, changes: &[(&str, Value)]) {
    let path = dir.join("config.json");
    let mut config: serde_json::Map<String, Value> =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    for (key, value) in changes {
        if value.is_null() {
            config.remove(*key);
        } else {
            config.insert((*key).to_owned(), value.clone());
        }
    }
    fs::write(path, serde_json::to_string(&config).unwrap()).unwrap();
}

fn remove(dir: &Path, names: &[&str]) {
    for name in names {
        fs::remove_file(dir.join(name)).unwrap();
    }
}

/// An index of `state_dict` with all keys in the shard `shard`.
fn write_index(path: &Path, state_dict: &StateDict, shard: &str, metadata: Value) {
    let weight_map: serde_json::Map<String, Value> = state_dict
        .keys()
        .map(|k| (k.to_owned(), Value::from(shard)))
        .collect();
    let index = serde_json::json!({ "metadata": metadata, "weight_map": weight_map });
    fs::write(path, serde_json::to_string(&index).unwrap()).unwrap();
}

fn options() -> PretrainedOptions {
    PretrainedOptions::default()
}

fn load<M: FromPretrained>(dir: &Path, options: PretrainedOptions) -> LoadedModel<M> {
    M::from_pretrained_with_info(&Repo::local(dir).unwrap(), options)
        .unwrap_or_else(|e| panic!("{e}"))
}

fn error<M: FromPretrained + std::fmt::Debug>(
    dir: &Path,
    options: PretrainedOptions,
) -> PretrainedError {
    M::from_pretrained(&Repo::local(dir).unwrap(), options).unwrap_err()
}

fn pixel_values() -> Tensor {
    homura::load(golden("resnet/inputs.safetensors"))
        .unwrap()
        .get("pixel_values")
        .unwrap()
        .clone()
}

fn expected(repo: &str, key: &str) -> Tensor {
    homura::load(golden(&format!("resnet/{repo}.safetensors")))
        .unwrap()
        .get(key)
        .unwrap()
        .clone()
}

/// The model of the golden repository `repo` loaded with the core API alone.
fn core_model(repo: &str) -> ResNetForImageClassification {
    let dir = golden(&format!("repos/{repo}"));
    let config = ResNetConfig::from_file(dir.join("config.json")).unwrap();
    let model = homura::with_device(Device::Meta, || ResNetForImageClassification::new(&config));
    let sd = homura::load(dir.join("model.safetensors")).unwrap();
    model
        .load_state_dict_with(&sd, LoadOptions::default().assign(true))
        .unwrap();
    model.eval();
    model
}

fn logits(model: &ResNetForImageClassification, x: &Tensor) -> Tensor {
    homura::with_no_grad(|| model.forward(x))
}

/// The values of a floating-point tensor (exactly).
fn values(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec::<f64>()
}

fn assert_bits(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    assert_eq!(values(a), values(b));
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for (a, e) in values(actual).iter().zip(values(expected)) {
        assert!((a - e).abs() <= tolerance * (1.0 + e.abs()), "{a} vs {e}");
    }
}

/// A cache with `org/mini` whose snapshot links the files of `dir` from `blobs`.
fn cache_of(dir: &Path, name: &str) -> PathBuf {
    let cache = scratch(name);
    let repo = cache.join("models--org--mini");
    let snapshot = repo.join("snapshots").join(HASH);
    fs::create_dir_all(&snapshot).unwrap();
    fs::create_dir_all(repo.join("blobs")).unwrap();
    fs::create_dir_all(repo.join("refs")).unwrap();
    fs::write(repo.join("refs/main"), HASH).unwrap();
    for (i, entry) in fs::read_dir(dir).unwrap().enumerate() {
        let entry = entry.unwrap();
        let blob = repo.join("blobs").join(format!("blob{i}"));
        fs::copy(entry.path(), &blob).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            format!("../../blobs/blob{i}"),
            snapshot.join(entry.file_name()),
        )
        .unwrap();
        #[cfg(not(unix))]
        fs::copy(&blob, snapshot.join(entry.file_name())).unwrap();
    }
    cache
}

/// Every way of loading a repository gives the model that the core API loads: the same outputs,
/// bit for bit, and the reference's within the tolerance of resnet_golden.rs.
#[test]
fn every_path_gives_the_core_model() {
    let x = pixel_values();
    for repo in [
        "basic",
        "basic_no_labels",
        "bottleneck",
        "bottleneck_downsample",
    ] {
        let core = logits(&core_model(repo), &x);
        assert_close(&core, &expected(repo, "eval_f32.logits"), 1e-5);
        let dir = golden(&format!("repos/{repo}"));
        let local = Repo::local(&dir).unwrap();
        let cache = cache_of(&dir, &format!("paths_{repo}"));
        let cached = Repo::cached_in(&cache, "org/mini", Revision::main()).unwrap();
        for repository in [&local, &cached] {
            for format in [
                None,
                Some(WeightFormat::Safetensors),
                Some(WeightFormat::PyTorch),
            ] {
                let options = match format {
                    Some(f) => options().format(f),
                    None => options(),
                };
                let loaded =
                    ResNetForImageClassification::from_pretrained_with_info(repository, options)
                        .unwrap();
                assert_bits(&logits(&loaded.model, &x), &core);
                let name = loaded.report.weights.path.file_name().unwrap().to_owned();
                let bin = format == Some(WeightFormat::PyTorch);
                assert_eq!(
                    name,
                    if bin {
                        "pytorch_model.bin"
                    } else {
                        "model.safetensors"
                    }
                );
                // SAFETY: the fixtures and the scratch copies aren't modified while the tests run.
                let mapped = unsafe {
                    ResNetForImageClassification::from_pretrained_mmap(repository, options)
                }
                .unwrap();
                assert_bits(&logits(&mapped.model, &x), &core);
                assert_eq!(mapped.report, loaded.report);
            }
            let model =
                ResNetForImageClassification::from_pretrained(repository, options()).unwrap();
            assert_bits(&logits(&model, &x), &core);
        }
    }
}

#[test]
fn loaded_models_are_in_eval_mode_and_trainable() {
    let LoadedModel {
        model,
        config,
        report,
        ..
    } = load::<ResNetForImageClassification>(&golden("repos/basic"), options());
    // Batch normalization uses the running statistics: two calls on different batches agree on
    // the first image.
    let x = pixel_values();
    let first = logits(&model, &x.narrow(0, 0, 1));
    assert_bits(&first, &logits(&model, &x).narrow(0, 0, 1));
    assert!(model.parameters().iter().all(|p| p.requires_grad()));
    assert_eq!(
        config,
        ResNetConfig::from_file(golden("repos/basic/config.json")).unwrap()
    );
    assert_eq!(config.common().num_labels(), 10);
    assert_eq!(report.weights.path, golden("repos/basic/model.safetensors"));
    assert_eq!((report.prefix_removed, report.prefix_added), (0, 0));
    assert!(report.unused_keys.is_empty() && report.config_warnings.is_empty());
    assert_eq!((report.dtype, report.converted), (Some(DType::F32), false));
    // Training still works.
    model.train();
    let loss = model.forward(&x).sum();
    loss.backward();
    assert!(model.parameters().iter().all(|p| p.grad().is_some()));
}

/// The base model from a classification repository and the classification model from a base
/// model's repository (check_m3_hub.py `base_model_prefix`).
#[test]
fn base_model_prefix() {
    let x = pixel_values();
    let full = homura::load(golden("repos/basic/model.safetensors")).unwrap();
    let loaded = load::<ResNetModel>(&golden("repos/basic"), options());
    let output = homura::with_no_grad(|| loaded.model.forward(&x));
    let core = core_model("basic");
    let core_output = homura::with_no_grad(|| core.resnet.forward(&x));
    assert_bits(&output.pooler_output, &core_output.pooler_output);
    assert_bits(&output.last_hidden_state, &core_output.last_hidden_state);
    // In the file's order.
    assert_eq!(
        loaded.report.unused_keys,
        ["classifier.1.bias", "classifier.1.weight"]
    );
    assert_eq!(loaded.report.prefix_removed, full.len() - 2);
    assert_eq!(loaded.report.prefix_added, 0);

    // A base model's repository: the keys of `ResNetModel.save_pretrained`.
    let dir = copy_repo("basic", "prefix_base");
    remove(&dir, &["model.safetensors", "pytorch_model.bin"]);
    edit_config(
        &dir,
        &[("architectures", serde_json::json!(["ResNetModel"]))],
    );
    let base = full.sub("resnet.");
    homura::save(&base, dir.join("model.safetensors")).unwrap();
    let loaded = load::<ResNetModel>(&dir, options());
    let output = homura::with_no_grad(|| loaded.model.forward(&x));
    assert_bits(&output.pooler_output, &core_output.pooler_output);
    assert_eq!(
        (loaded.report.prefix_removed, loaded.report.prefix_added),
        (0, 0)
    );
    // The classification model gets the prefix, but its head is missing.
    match error::<ResNetForImageClassification>(&dir, options()) {
        PretrainedError::Load(e) => {
            assert_eq!(e.missing_keys, HEAD);
            assert!(e.unexpected_keys.is_empty());
        }
        e => panic!("{e}"),
    }
    // With the head, every base key gets the prefix.
    let mut with_head = base.clone();
    for key in HEAD {
        with_head.insert(key, full.get(key).unwrap().clone());
    }
    homura::save(&with_head, dir.join("model.safetensors")).unwrap();
    let loaded = load::<ResNetForImageClassification>(&dir, options());
    assert_bits(&logits(&loaded.model, &x), &logits(&core, &x));
    assert_eq!(
        (loaded.report.prefix_removed, loaded.report.prefix_added),
        (0, base.len())
    );

    // Key by key: keys that renaming doesn't make the model's are left unused.
    let mut mixed = with_head.clone();
    mixed.insert("resnet.extra", homura::zeros([1]));
    mixed.insert("other", homura::zeros([1]));
    homura::save(&mixed, dir.join("model.safetensors")).unwrap();
    let loaded = load::<ResNetForImageClassification>(&dir, options());
    assert_eq!(loaded.report.unused_keys, ["resnet.extra", "other"]);
    // A key with and without the prefix.
    let mut twice = full.clone();
    let key = "embedder.embedder.convolution.weight";
    twice.insert(key, full.get(&format!("resnet.{key}")).unwrap().clone());
    homura::save(&twice, dir.join("model.safetensors")).unwrap();
    match error::<ResNetForImageClassification>(&dir, options()) {
        PretrainedError::Checkpoint(m) => assert!(m.contains(&format!("resnet.{key}")), "{m}"),
        e => panic!("{e}"),
    }
}

fn float_dtypes(model: &impl Module) -> Vec<DType> {
    let mut dtypes: Vec<DType> = model
        .state_dict()
        .values()
        .filter(|t| t.is_floating_point())
        .map(|t| t.dtype())
        .collect();
    dtypes.dedup();
    dtypes
}

fn count_dtypes(model: &impl Module) -> Vec<DType> {
    let mut dtypes: Vec<DType> = model
        .state_dict()
        .values()
        .filter(|t| !t.is_floating_point())
        .map(|t| t.dtype())
        .collect();
    dtypes.dedup();
    dtypes
}

/// The dtype of the model (check_m3_hub.py `dtypes`).
#[test]
fn dtypes() {
    let full = homura::load(golden("repos/basic/model.safetensors")).unwrap();
    let f64_weights = full.clone().map_keys(|k| Some(k.to_owned()));
    let f64_weights = {
        let mut sd = StateDict::new();
        for (k, t) in f64_weights.iter() {
            let t = if t.is_floating_point() {
                t.to(DType::F64)
            } else {
                t.clone()
            };
            sd.insert(k, t);
        }
        sd
    };
    let null = Value::Null;
    let name = |s: &str| Value::from(s);
    type Case<'a> = (
        Vec<(&'a str, Value)>,
        &'a StateDict,
        Option<DType>,
        DType,
        bool,
    );
    let cases: Vec<Case> = vec![
        (
            vec![("dtype", name("float32"))],
            &full,
            None,
            DType::F32,
            false,
        ),
        (
            vec![("dtype", name("float64"))],
            &full,
            None,
            DType::F64,
            true,
        ),
        (
            vec![("dtype", null.clone()), ("torch_dtype", name("float64"))],
            &full,
            None,
            DType::F64,
            true,
        ),
        (
            vec![("dtype", name("float64")), ("torch_dtype", name("float32"))],
            &full,
            None,
            DType::F64,
            true,
        ),
        (
            vec![("dtype", null.clone())],
            &f64_weights,
            None,
            DType::F64,
            false,
        ),
        (
            vec![("dtype", null.clone())],
            &full,
            None,
            DType::F32,
            false,
        ),
        (
            vec![("dtype", name("float32"))],
            &f64_weights,
            None,
            DType::F32,
            true,
        ),
        (
            vec![("dtype", null.clone())],
            &f64_weights,
            Some(DType::F32),
            DType::F32,
            true,
        ),
        (
            vec![("dtype", name("float32"))],
            &full,
            Some(DType::F64),
            DType::F64,
            true,
        ),
    ];
    let x = pixel_values();
    let core = logits(&core_model("basic"), &x.to(DType::F64).to(DType::F32));
    for (i, (fields, weights, dtype, expected, converted)) in cases.into_iter().enumerate() {
        let dir = copy_repo("basic", &format!("dtype_{i}"));
        remove(&dir, &["pytorch_model.bin"]);
        edit_config(&dir, &fields);
        homura::save(weights, dir.join("model.safetensors")).unwrap();
        let options = match dtype {
            Some(d) => options().dtype(d),
            None => options(),
        };
        for mapped in [false, true] {
            let loaded: LoadedModel<ResNetForImageClassification> = if mapped {
                // SAFETY: the scratch files aren't modified while the tests run.
                unsafe {
                    FromPretrained::from_pretrained_mmap(&Repo::local(&dir).unwrap(), options)
                }
                .unwrap()
            } else {
                load(&dir, options)
            };
            assert_eq!(float_dtypes(&loaded.model), [expected], "{i}");
            assert_eq!(count_dtypes(&loaded.model), [DType::I64], "{i}");
            assert_eq!(loaded.config.common().dtype, Some(expected), "{i}");
            assert_eq!(
                (loaded.report.dtype, loaded.report.converted),
                (Some(expected), converted),
                "{i}"
            );
            if expected == DType::F32
                && weights.get("classifier.1.bias").unwrap().dtype() == DType::F32
            {
                assert_bits(&logits(&loaded.model, &x), &core);
            }
        }
    }

    // Sharded, without a configuration dtype: the index's metadata, else the weights.
    let dir = copy_repo("basic", "dtype_index");
    remove(&dir, &["model.safetensors", "pytorch_model.bin"]);
    edit_config(&dir, &[("dtype", null.clone())]);
    homura::save(&full, dir.join("m-1.safetensors")).unwrap();
    let index = dir.join("model.safetensors.index.json");
    for (metadata, expected) in [
        (serde_json::json!({"dtype": "float64"}), DType::F64),
        (serde_json::json!({"total_size": 0}), DType::F32),
    ] {
        write_index(&index, &full, "m-1.safetensors", metadata);
        let loaded = load::<ResNetForImageClassification>(&dir, options());
        assert_eq!(float_dtypes(&loaded.model), [expected]);
        assert!(loaded.report.weights.sharded);
    }
    // The metadata isn't read when the configuration or the options have a dtype.
    write_index(
        &index,
        &full,
        "m-1.safetensors",
        serde_json::json!({"dtype": "int64"}),
    );
    match error::<ResNetForImageClassification>(&dir, options()) {
        PretrainedError::Checkpoint(m) => assert!(m.contains("\"int64\""), "{m}"),
        e => panic!("{e}"),
    }
    let loaded = load::<ResNetForImageClassification>(&dir, options().dtype(DType::F32));
    assert_eq!(float_dtypes(&loaded.model), [DType::F32]);
    edit_config(&dir, &[("dtype", name("float64"))]);
    let loaded = load::<ResNetForImageClassification>(&dir, options());
    assert_eq!(float_dtypes(&loaded.model), [DType::F64]);

    // A PyTorch file converted to F64.
    let loaded = load::<ResNetForImageClassification>(
        &golden("repos/bottleneck"),
        options().format(WeightFormat::PyTorch).dtype(DType::F64),
    );
    assert!(loaded.report.converted);
    let core = core_model("bottleneck");
    core.to(DType::F64);
    assert_bits(
        &logits(&loaded.model, &x.to(DType::F64)),
        &logits(&core, &x.to(DType::F64)),
    );
}

#[test]
#[should_panic(expected = "the dtype of a model must be floating point, found I64")]
fn integer_dtypes_are_rejected() {
    let _ = PretrainedOptions::default().dtype(DType::I64);
}

/// `transformers_weights`, sharded PyTorch files and which file is read (no fallback).
#[test]
fn weight_files() {
    let x = pixel_values();
    let core = logits(&core_model("basic"), &x);
    let dir = copy_repo("basic", "files_explicit");
    fs::create_dir_all(dir.join("sub")).unwrap();
    fs::rename(dir.join("model.safetensors"), dir.join("sub/w.safetensors")).unwrap();
    // pytorch_model.bin is there but not read.
    edit_config(
        &dir,
        &[("transformers_weights", Value::from("sub/w.safetensors"))],
    );
    for format in [WeightFormat::Safetensors, WeightFormat::PyTorch] {
        let loaded = load::<ResNetForImageClassification>(&dir, options().format(format));
        assert_eq!(loaded.report.weights.path, dir.join("sub/w.safetensors"));
        assert_bits(&logits(&loaded.model, &x), &core);
    }
    edit_config(&dir, &[("transformers_weights", Value::from("w.bin"))]);
    assert!(matches!(
        error::<ResNetForImageClassification>(&dir, options()),
        PretrainedError::Config(_)
    ));

    // An index of PyTorch files.
    let dir = copy_repo("basic", "files_bin_index");
    remove(&dir, &["model.safetensors"]);
    fs::rename(
        dir.join("pytorch_model.bin"),
        dir.join("pytorch_model-00001-of-00001.bin"),
    )
    .unwrap();
    let full = homura::load(golden("repos/basic/model.safetensors")).unwrap();
    write_index(
        &dir.join("pytorch_model.bin.index.json"),
        &full,
        "pytorch_model-00001-of-00001.bin",
        serde_json::json!({"total_size": 0}),
    );
    let loaded = load::<ResNetForImageClassification>(&dir, options());
    assert!(loaded.report.weights.sharded);
    assert_eq!(loaded.report.weights.format, WeightFormat::PyTorch);
    assert_bits(&logits(&loaded.model, &x), &core);

    // A corrupt model.safetensors is an error, although pytorch_model.bin is fine.
    let dir = copy_repo("basic", "files_corrupt");
    fs::write(
        dir.join("model.safetensors"),
        b"\x08\0\0\0\0\0\0\0{}garbage",
    )
    .unwrap();
    match error::<ResNetForImageClassification>(&dir, options()) {
        PretrainedError::Weights { path, .. } => assert_eq!(path, dir.join("model.safetensors")),
        e => panic!("{e}"),
    }
    // SAFETY: the scratch files aren't modified while the tests run.
    let mapped = unsafe {
        ResNetForImageClassification::from_pretrained_mmap(&Repo::local(&dir).unwrap(), options())
    };
    assert!(matches!(mapped, Err(PretrainedError::Weights { .. })));
    let loaded =
        load::<ResNetForImageClassification>(&dir, options().format(WeightFormat::PyTorch));
    assert_bits(&logits(&loaded.model, &x), &core);
}

/// The shards of a `transformers_weights` index in a subdirectory are files of the repository's
/// root, as transformers finds them (R117): the shard there is read, not one next to the index.
#[test]
fn shards_of_a_nested_index_are_at_the_root() {
    let full = homura::load(golden("repos/basic/model.safetensors")).unwrap();
    let marked = |mark: f64| {
        let mut sd = full.clone();
        sd.insert("classifier.1.bias", homura::zeros([10]) + mark);
        sd
    };
    // (index in sub/, shard at the root, shard in sub/) -> the bias loaded, or None for an error.
    let cases = [
        (true, true, false, Some(2.0)),
        (true, false, true, None),
        (true, true, true, Some(2.0)),
        (false, true, false, Some(2.0)),
    ];
    for (i, (nested, at_root, in_sub, expected)) in cases.into_iter().enumerate() {
        let dir = copy_repo("basic", &format!("nested_index_{i}"));
        remove(&dir, &["model.safetensors", "pytorch_model.bin"]);
        fs::create_dir_all(dir.join("sub")).unwrap();
        let index = if nested {
            "sub/w.safetensors.index.json"
        } else {
            "w.safetensors.index.json"
        };
        edit_config(&dir, &[("transformers_weights", Value::from(index))]);
        write_index(
            &dir.join(index),
            &full,
            "weights.safetensors",
            serde_json::json!({}),
        );
        if at_root {
            homura::save(&marked(2.0), dir.join("weights.safetensors")).unwrap();
        }
        if in_sub {
            homura::save(&marked(9.0), dir.join("sub/weights.safetensors")).unwrap();
        }
        let cache = cache_of_tree(&dir, &format!("nested_index_cache_{i}"));
        let cached = Repo::cached_in(&cache, "org/mini", Revision::main()).unwrap();
        for repo in [Repo::local(&dir).unwrap(), cached] {
            // SAFETY: the scratch files aren't modified while the test runs.
            let mapped =
                unsafe { ResNetForImageClassification::from_pretrained_mmap(&repo, options()) };
            let read = ResNetForImageClassification::from_pretrained_with_info(&repo, options());
            for result in [read, mapped] {
                match (result, expected) {
                    (Ok(loaded), Some(bias)) => {
                        let sd = loaded.model.state_dict();
                        assert_eq!(
                            values(sd.get("classifier.1.bias").unwrap()),
                            [bias; 10],
                            "{i}"
                        );
                        assert!(loaded.report.weights.sharded);
                    }
                    (Err(PretrainedError::Weights { .. }), None) => {}
                    (Ok(_), None) => panic!("case {i} loaded"),
                    (Err(e), _) => panic!("case {i}: {e}"),
                }
            }
        }
    }
}

/// A shard name with a Windows drive, which would replace the repository's directory when joined
/// there, is refused before anything is read, even where such a file exists (R119).
#[test]
fn shard_names_with_drives_are_refused() {
    let full = homura::load(golden("repos/basic/model.safetensors")).unwrap();
    for (i, index) in ["w.safetensors.index.json", "sub/w.safetensors.index.json"]
        .into_iter()
        .enumerate()
    {
        let dir = copy_repo("basic", &format!("drive_shard_{i}"));
        remove(&dir, &["model.safetensors", "pytorch_model.bin"]);
        fs::create_dir_all(dir.join("sub")).unwrap();
        edit_config(&dir, &[("transformers_weights", Value::from(index))]);
        write_index(
            &dir.join(index),
            &full,
            "C:weights.safetensors",
            serde_json::json!({}),
        );
        homura::save(&full, dir.join("C:weights.safetensors")).unwrap();
        let cache = cache_of_tree(&dir, &format!("drive_shard_cache_{i}"));
        let cached = Repo::cached_in(&cache, "org/mini", Revision::main()).unwrap();
        for repo in [Repo::local(&dir).unwrap(), cached] {
            // SAFETY: the scratch files aren't modified while the test runs.
            let mapped =
                unsafe { ResNetForImageClassification::from_pretrained_mmap(&repo, options()) };
            let read = ResNetForImageClassification::from_pretrained_with_info(&repo, options());
            for result in [read, mapped] {
                match result {
                    Err(PretrainedError::Weights {
                        error: homura::Error::Format(m),
                        ..
                    }) => {
                        assert!(m.contains("is not a file in the shards' directory"), "{m}");
                    }
                    Err(e) => panic!("{index}: {e}"),
                    Ok(_) => panic!("{index}: loaded"),
                }
            }
        }
    }
}

/// Like `cache_of`, for a directory with subdirectories: every file is linked from `blobs`.
fn cache_of_tree(dir: &Path, name: &str) -> PathBuf {
    let cache = scratch(name);
    let repo = cache.join("models--org--mini");
    let snapshot = repo.join("snapshots").join(HASH);
    fs::create_dir_all(repo.join("blobs")).unwrap();
    fs::create_dir_all(repo.join("refs")).unwrap();
    fs::write(repo.join("refs/main"), HASH).unwrap();
    let mut files = vec![];
    let mut dirs = vec![PathBuf::new()];
    while let Some(sub) = dirs.pop() {
        for entry in fs::read_dir(dir.join(&sub)).unwrap() {
            let entry = entry.unwrap();
            let rel = sub.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                dirs.push(rel);
            } else {
                files.push(rel);
            }
        }
    }
    for (i, rel) in files.iter().enumerate() {
        let blob = repo.join("blobs").join(format!("blob{i}"));
        fs::copy(dir.join(rel), &blob).unwrap();
        let link = snapshot.join(rel);
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        #[cfg(unix)]
        {
            let up = "../".repeat(rel.components().count() + 1);
            std::os::unix::fs::symlink(format!("{up}blobs/blob{i}"), &link).unwrap();
        }
        #[cfg(not(unix))]
        fs::copy(&blob, &link).unwrap();
    }
    cache
}

/// transformers doesn't read `_metadata`: a count that version 2 of BatchNorm2d requires may be
/// missing (check_m3_hub.py `loaded_models`).
#[test]
fn versions_are_not_read() {
    let dir = copy_repo("bottleneck_downsample", "no_count");
    remove(&dir, &["model.safetensors"]);
    fs::copy(
        golden("pretrained/no_count.bin"),
        dir.join("pytorch_model.bin"),
    )
    .unwrap();
    let key = "resnet.embedder.embedder.normalization.num_batches_tracked";
    let sd = homura::load(dir.join("pytorch_model.bin")).unwrap();
    assert!(sd.get(key).is_none());
    assert_eq!(
        sd.module_version("resnet.embedder.embedder.normalization"),
        Some(2)
    );
    let loaded = load::<ResNetForImageClassification>(&dir, options());
    let counts = loaded.model.state_dict();
    assert_eq!(counts.get(key).unwrap().to_vec::<i64>(), [0]);
    let x = pixel_values();
    let core = logits(&core_model("bottleneck_downsample"), &x);
    assert_bits(&logits(&loaded.model, &x), &core);
}

#[test]
fn errors() {
    // No config.json.
    let dir = copy_repo("basic", "errors_config");
    remove(&dir, &["config.json"]);
    match error::<ResNetForImageClassification>(&dir, options()) {
        PretrainedError::Hub(e) => {
            assert!(e.to_string().contains("has no file config.json"), "{e}")
        }
        e => panic!("{e}"),
    }
    // Another model type, an invalid configuration.
    fs::copy(golden("repos/basic/config.json"), dir.join("config.json")).unwrap();
    edit_config(&dir, &[("model_type", Value::from("bert"))]);
    assert!(matches!(
        error::<ResNetForImageClassification>(&dir, options()),
        PretrainedError::Config(_)
    ));
    edit_config(
        &dir,
        &[
            ("model_type", Value::from("resnet")),
            ("depths", serde_json::json!([1])),
        ],
    );
    assert!(matches!(
        error::<ResNetModel>(&dir, options()),
        PretrainedError::Config(_)
    ));
    // No weights.
    let dir = copy_repo("basic", "errors_weights");
    remove(&dir, &["model.safetensors", "pytorch_model.bin"]);
    let e = error::<ResNetForImageClassification>(&dir, options());
    assert!(matches!(&e, PretrainedError::Hub(_)), "{e}");
    assert!(e.to_string().contains("has no weights"), "{e}");
    // Weights of another configuration.
    let dir = copy_repo("basic", "errors_shapes");
    fs::copy(
        golden("repos/bottleneck/model.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();
    assert!(matches!(
        error::<ResNetForImageClassification>(&dir, options()),
        PretrainedError::Load(_)
    ));
    // The configuration warnings are reported.
    let dir = copy_repo("basic", "errors_labels");
    edit_config(
        &dir,
        &[
            ("num_labels", Value::from(10)),
            ("id2label", serde_json::json!({"0": "a"})),
        ],
    );
    let loaded = load::<ResNetForImageClassification>(&dir, options());
    assert_eq!(loaded.report.config_warnings.len(), 1);
    assert!(loaded.report.config_warnings[0].contains("`num_labels` (10)"));
}

/// Whether this process maps `path` (Linux lists the mapped files in /proc/self/maps).
#[cfg(target_os = "linux")]
fn is_mapped(path: &Path) -> bool {
    let path = fs::canonicalize(path).unwrap();
    let maps = fs::read_to_string("/proc/self/maps").unwrap();
    maps.lines()
        .any(|line| line.ends_with(path.to_str().unwrap()))
}

/// Only `from_pretrained_mmap` maps the weight files, and a model keeps them mapped while its
/// tensors use them (R103).
#[cfg(target_os = "linux")]
#[test]
fn only_the_mmap_entry_maps_files() {
    let x = pixel_values();
    let core = logits(&core_model("basic"), &x);
    for format in [WeightFormat::Safetensors, WeightFormat::PyTorch] {
        let dir = copy_repo("basic", &format!("mapped_{format:?}"));
        let options = options().format(format);
        let file = Repo::local(&dir)
            .unwrap()
            .weight_files(Some(format), None)
            .unwrap()
            .path;
        let model = load::<ResNetForImageClassification>(&dir, options).model;
        assert!(!is_mapped(&file));
        drop(model);
        // SAFETY: the scratch copy isn't modified while the test runs.
        let loaded = unsafe {
            ResNetForImageClassification::from_pretrained_mmap(&Repo::local(&dir).unwrap(), options)
        }
        .unwrap();
        assert!(is_mapped(&file), "{format:?}");
        assert_bits(&logits(&loaded.model, &x), &core);
        drop(loaded);
        assert!(!is_mapped(&file));
        // Converted tensors have memory of their own, but the counts of the batch normalizations
        // keep their dtype, and the mapping.
        // SAFETY: as above.
        let loaded = unsafe {
            ResNetForImageClassification::from_pretrained_mmap(
                &Repo::local(&dir).unwrap(),
                options.dtype(DType::F64),
            )
        }
        .unwrap();
        assert!(loaded.report.converted);
        assert!(is_mapped(&file), "{format:?}");
        let counts: Vec<Tensor> = loaded
            .model
            .state_dict()
            .values()
            .filter(|t| !t.is_floating_point())
            .cloned()
            .collect();
        drop(loaded);
        assert!(is_mapped(&file), "{format:?}");
        drop(counts);
        assert!(!is_mapped(&file), "{format:?}");
    }
}
