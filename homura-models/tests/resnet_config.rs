//! Reading `config.json` into a `ResNetConfig`: transformers' fields, defaults and label and
//! dtype rules (checked against transformers by tools/golden/check_m3_resnet.py), and the errors
//! for configurations the models can't be built from.

use homura::DType;
use homura_models::config::ConfigError;
use homura_models::resnet::{ResNetConfig, ResNetLayerType};

fn read(json: &str) -> ResNetConfig {
    ResNetConfig::from_json(json).unwrap_or_else(|e| panic!("{json}: {e}"))
}

fn error(json: &str) -> String {
    match ResNetConfig::from_json(json) {
        Err(e @ ConfigError::Invalid(_)) => e.to_string(),
        Err(e) => panic!("{json}: unexpected error {e}"),
        Ok(config) => panic!("{json}: read as {config:?}"),
    }
}

#[test]
fn missing_fields_have_transformers_defaults() {
    let config = read("{}");
    assert_eq!(config, ResNetConfig::default());
    assert_eq!(config.num_channels, 3);
    assert_eq!(config.embedding_size, 64);
    assert_eq!(config.hidden_sizes, [256, 512, 1024, 2048]);
    assert_eq!(config.depths, [3, 4, 6, 3]);
    assert_eq!(config.layer_type, ResNetLayerType::Bottleneck);
    assert_eq!(config.hidden_act, "relu");
    assert!(!config.downsample_in_first_stage && !config.downsample_in_bottleneck);
    assert_eq!(config.common.model_type, "resnet");
    assert_eq!(config.common.num_labels(), 2);
    assert_eq!(config.common.id2label[&1], "LABEL_1");
    assert_eq!(config.common.label2id["LABEL_0"], 0);
    assert_eq!(config.common.dtype, None);
    assert!(config.common.warnings.is_empty());
    // `null` is "not given", and other fields are ignored.
    assert_eq!(
        read(r#"{"depths": null, "dtype": null, "stage_names": ["stem"], "out_indices": [4]}"#),
        config
    );
}

#[test]
fn fields_are_read() {
    let config = read(
        r#"{"model_type": "resnet", "num_channels": 1, "embedding_size": 8,
            "hidden_sizes": [8, 16], "depths": [2, 1], "layer_type": "basic",
            "hidden_act": "gelu", "downsample_in_first_stage": true,
            "downsample_in_bottleneck": true, "architectures": ["ResNetModel"],
            "transformers_weights": "weights.safetensors", "transformers_version": "5.20.2"}"#,
    );
    assert_eq!(
        (
            config.num_channels,
            config.embedding_size,
            config.layer_type
        ),
        (1, 8, ResNetLayerType::Basic)
    );
    assert_eq!(
        (config.hidden_sizes.as_slice(), config.depths.as_slice()),
        (&[8, 16][..], &[2, 1][..])
    );
    assert_eq!(config.hidden_act, "gelu");
    assert!(config.downsample_in_first_stage && config.downsample_in_bottleneck);
    assert_eq!(
        config.common.architectures.as_deref(),
        Some(&["ResNetModel".to_owned()][..])
    );
    assert_eq!(
        config.common.transformers_weights.as_deref(),
        Some("weights.safetensors")
    );
    assert_eq!(
        config.common.transformers_version.as_deref(),
        Some("5.20.2")
    );
}

#[test]
fn labels_follow_transformers() {
    let labels = |json: &str| {
        let config = read(json);
        let names: Vec<(i64, String)> = config.common.id2label.clone().into_iter().collect();
        (names, config.common.warnings.len())
    };
    let named = |n: i64| {
        (0..n)
            .map(|i| (i, format!("LABEL_{i}")))
            .collect::<Vec<_>>()
    };
    assert_eq!(labels(r#"{"num_labels": 4}"#), (named(4), 0));
    assert_eq!(labels(r#"{"num_labels": 0}"#), (named(0), 0));
    // id2label's keys are strings in JSON; their number is the number of labels.
    let abc = r#""id2label": {"0": "a", "1": "b", "2": "c"}"#;
    let abc_names = vec![
        (0, "a".to_owned()),
        (1, "b".to_owned()),
        (2, "c".to_owned()),
    ];
    assert_eq!(labels(&format!("{{{abc}}}")), (abc_names.clone(), 0));
    assert_eq!(
        labels(&format!(r#"{{{abc}, "num_labels": 3}}"#)),
        (abc_names, 0)
    );
    // A num_labels that disagrees wins, with a warning, and renames the labels.
    assert_eq!(
        labels(&format!(r#"{{{abc}, "num_labels": 5}}"#)),
        (named(5), 1)
    );
    assert_eq!(
        labels(&format!(r#"{{{abc}, "num_labels": 0}}"#)),
        (named(0), 1)
    );
    // Keys need not be 0, 1, ...
    assert_eq!(
        labels(r#"{"id2label": {"2": "c", "0": "a"}}"#),
        (vec![(0, "a".to_owned()), (2, "c".to_owned())], 0)
    );
    let config = read(r#"{"id2label": {"0": "a", "1": "b"}, "label2id": {"a": 0, "b": "1"}}"#);
    // Without id2label, label2id is replaced too.
    assert_eq!(
        read(r#"{"label2id": {"a": "x"}}"#).common.label2id["LABEL_1"],
        1
    );
    assert_eq!(config.common.label2id["b"], 1);
    assert_eq!(config.common.num_labels(), 2);

    assert!(error(r#"{"num_labels": -1}"#).contains("`num_labels` is negative"));
    assert!(error(r#"{"num_labels": 2000000}"#).contains("more than the 1048576 labels"));
    assert!(error(r#"{"num_labels": 2.5}"#).contains("must be an integer"));
    assert!(error(r#"{"id2label": {"x": "a"}}"#).contains("isn't an integer"));
    assert!(error(r#"{"id2label": {"1": "a", "01": "b"}}"#).contains("the label 1 twice"));
    assert!(error(r#"{"id2label": {"0": 3}}"#).contains("must be a string"));
    assert!(error(r#"{"id2label": [1]}"#).contains("`id2label` must be"));
    assert!(
        error(r#"{"id2label": {"0": "a"}, "label2id": {"a": "x"}}"#).contains("must be an integer")
    );
}

#[test]
fn dtype_and_torch_dtype() {
    let dtype = |json: &str| read(json).common.dtype;
    assert_eq!(dtype(r#"{"torch_dtype": "float32"}"#), Some(DType::F32));
    assert_eq!(
        dtype(r#"{"dtype": "float64", "torch_dtype": "float16"}"#),
        Some(DType::F64)
    );
    assert_eq!(
        dtype(r#"{"dtype": null, "torch_dtype": "bfloat16"}"#),
        Some(DType::BF16)
    );
    for (name, expected) in [
        ("float", DType::F32),
        ("double", DType::F64),
        ("half", DType::F16),
        ("float16", DType::F16),
    ] {
        assert_eq!(dtype(&format!(r#"{{"dtype": "{name}"}}"#)), Some(expected));
    }
    assert!(error(r#"{"dtype": "int8"}"#).contains("must name a floating-point dtype"));
    assert!(error(r#"{"torch_dtype": "auto"}"#).contains("found \"auto\""));
    // `dtype` decides when given, even if `torch_dtype` is invalid.
    assert_eq!(
        dtype(r#"{"dtype": "float32", "torch_dtype": 7}"#),
        Some(DType::F32)
    );
    assert!(error(r#"{"torch_dtype": 7}"#).contains("must be a string"));
}

#[test]
fn invalid_configurations_are_errors() {
    let cases = [
        (
            r#"{"model_type": "bert"}"#,
            "for a model of type `bert`, not `resnet`",
        ),
        ("[1, 2]", "isn't a JSON object"),
        ("{", "isn't valid JSON"),
        (
            r#"{"layer_type": "wide"}"#,
            "must be \"basic\" or \"bottleneck\"",
        ),
        (
            r#"{"hidden_act": "mish"}"#,
            "`hidden_act` \"mish\" isn't one of",
        ),
        (
            r#"{"hidden_sizes": [], "depths": []}"#,
            "the same nonzero length",
        ),
        (
            r#"{"hidden_sizes": [8, 16], "depths": [1]}"#,
            "the same nonzero length",
        ),
        (
            r#"{"hidden_sizes": [8, 0], "depths": [1, 1]}"#,
            "a list of positive integers",
        ),
        (r#"{"depths": [1, 0, 1, 1]}"#, "a list of positive integers"),
        (
            r#"{"hidden_sizes": [8, 2], "depths": [1, 1]}"#,
            "at least 4",
        ),
        (
            r#"{"num_channels": 0}"#,
            "`num_channels` must be a positive integer",
        ),
        (
            r#"{"embedding_size": -8}"#,
            "`embedding_size` must be a positive integer",
        ),
        (
            r#"{"embedding_size": 8.0}"#,
            "`embedding_size` must be a positive integer",
        ),
        (r#"{"hidden_sizes": "64"}"#, "a list of positive integers"),
        (
            r#"{"downsample_in_first_stage": 1}"#,
            "must be true or false",
        ),
        (r#"{"architectures": "ResNetModel"}"#, "a list of strings"),
        (r#"{"transformers_weights": 3}"#, "must be a string"),
        (
            r#"{"transformers_weights": "w.bin"}"#,
            "must name a safetensors file",
        ),
        (
            r#"{"transformers_weights": "../w.safetensors"}"#,
            "must name a file inside the repository",
        ),
        // homura's limits, which keep a configuration file from making a model's construction
        // panic or take without end.
        (
            r#"{"embedding_size": 18446744073709551615}"#,
            "can't exceed 1048576",
        ),
        (r#"{"num_channels": 1048577}"#, "can't exceed 1048576"),
        (
            r#"{"hidden_sizes": [1048577], "depths": [1]}"#,
            "can't exceed 1048576",
        ),
        (
            r#"{"hidden_sizes": [8, 8], "depths": [32768, 32769]}"#,
            "can't add up to more than 65536 layers",
        ),
        (
            r#"{"hidden_sizes": [8, 8], "depths": [18446744073709551615, 2]}"#,
            "can't add up to more than 65536 layers",
        ),
    ];
    for (json, message) in cases {
        let found = error(json);
        assert!(found.starts_with("invalid configuration: "), "{found}");
        assert!(found.contains(message), "{json}: {found}");
    }
    // Basic layers have no quarter-channel convolutions.
    read(r#"{"layer_type": "basic", "hidden_sizes": [2, 3], "depths": [1, 1]}"#);
    // The limits themselves are valid, and models of them build on the meta device.
    read(r#"{"hidden_sizes": [8, 8], "depths": [32768, 32768]}"#);
    let widest = read(
        r#"{"num_channels": 1048576, "embedding_size": 1048576, "hidden_sizes": [1048576],
            "depths": [1], "num_labels": 1048576}"#,
    );
    for layer_type in [ResNetLayerType::Basic, ResNetLayerType::Bottleneck] {
        let config = widest.clone().layer_type(layer_type);
        let model = homura::with_device(homura::Device::Meta, || {
            homura_models::resnet::ResNetForImageClassification::new(&config)
        });
        drop(model);
    }
}

#[test]
fn files_are_read() {
    let dir = std::env::temp_dir().join(format!("homura-models-config-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.json");
    std::fs::write(&path, r#"{"model_type": "resnet", "num_labels": 3}"#).unwrap();
    assert_eq!(
        ResNetConfig::from_file(&path).unwrap().common.num_labels(),
        3
    );
    std::fs::remove_file(&path).unwrap();
    match ResNetConfig::from_file(&path) {
        Err(e @ ConfigError::Io { .. }) => {
            assert!(e.to_string().starts_with("can't read the configuration "));
            assert!(std::error::Error::source(&e).is_some());
        }
        other => panic!("{other:?}"),
    }
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn setters_change_one_field() {
    let config = ResNetConfig::default()
        .num_channels(1)
        .embedding_size(16)
        .hidden_sizes([16, 32])
        .depths(vec![1, 2])
        .layer_type(ResNetLayerType::Basic)
        .hidden_act("silu")
        .downsample_in_first_stage(true)
        .downsample_in_bottleneck(true)
        .num_labels(7)
        .dtype(DType::F64);
    assert!(config.validate().is_ok());
    assert_eq!(config.common.num_labels(), 7);
    assert_eq!(config.common.dtype, Some(DType::F64));
    assert_eq!(config.dtype(None).common.dtype, None);
    // Keeping the number of labels keeps their names.
    let mut config = read(r#"{"id2label": {"0": "cat", "1": "dog"}}"#);
    config.common.set_num_labels(2);
    assert_eq!(config.common.id2label[&1], "dog");
    let config = config.num_labels(3);
    assert_eq!(config.common.id2label[&1], "LABEL_1");
    assert_eq!(config.common.label2id.len(), 3);

    let invalid = ResNetConfig::default().hidden_sizes([8]);
    assert!(
        invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("the same nonzero length")
    );
    assert_eq!(ResNetLayerType::Bottleneck.to_string(), "bottleneck");
}
