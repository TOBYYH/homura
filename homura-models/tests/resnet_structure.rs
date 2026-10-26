//! The module trees of transformers' ResNet: for the configurations of microsoft/resnet-18 and
//! -50 and of the small repositories, the keys, shapes, dtypes, versions, numbers of parameters
//! and printed trees of models built on the meta device are transformers' (see
//! tools/golden/gen_m3_resnet.py).

use homura::Device;
use homura::nn::Module;
use homura_models::resnet::{ResNetConfig, ResNetForImageClassification, ResNetModel};
use serde_json::Value;

fn structures() -> serde_json::Map<String, Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/resnet/structure.json"
    );
    match serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap() {
        Value::Object(map) => map,
        _ => panic!("structure.json isn't an object"),
    }
}

/// transformers' `repr` with homura's spelling: the core layers' names, and pooling sizes given
/// as equal pairs printed as one number.
fn to_homura(text: &str) -> String {
    text.replace("ReLU(", "Relu(").replace(
        "AdaptiveAvgPool2d(output_size=(1, 1))",
        "AdaptiveAvgPool2d(output_size=1)",
    )
}

fn dtype_name(dtype: homura::DType) -> &'static str {
    match dtype {
        homura::DType::F32 => "float32",
        homura::DType::I64 => "int64",
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// A key with its shape and dtype name.
type Key = (String, Vec<usize>, String);

/// The keys with shapes and dtypes, the versions and the number of parameters of `model`.
fn tree(model: &dyn Module) -> (Vec<Key>, Vec<(String, u32)>, usize) {
    let sd = model.state_dict();
    let keys = sd
        .iter()
        .map(|(k, t)| {
            (
                k.to_owned(),
                t.shape().to_vec(),
                dtype_name(t.dtype()).to_owned(),
            )
        })
        .collect();
    let versions = sd
        .versions()
        .unwrap()
        .iter()
        .map(|(p, v)| (p.to_owned(), v))
        .collect();
    let num_parameters = model.parameters().iter().map(|p| p.num_elements()).sum();
    (keys, versions, num_parameters)
}

fn expected_keys(tree: &Value) -> Vec<Key> {
    tree["keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let shape = entry[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            (
                entry[0].as_str().unwrap().to_owned(),
                shape,
                entry[2].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

fn expected_versions(tree: &Value) -> Vec<(String, u32)> {
    let mut versions: Vec<_> = tree["versions"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(p, v)| (p.clone(), v.as_u64().unwrap() as u32))
        .collect();
    versions.sort();
    versions
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

#[test]
fn module_trees_are_transformers() {
    let structures = structures();
    assert_eq!(structures.len(), 6);
    for (name, structure) in &structures {
        let config = ResNetConfig::from_json(&structure["config"].to_string())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let (clf, base) = homura::with_device(Device::Meta, || {
            (
                ResNetForImageClassification::new(&config),
                ResNetModel::new(&config),
            )
        });
        let expected = &structure["classification"];

        let (keys, versions, num_parameters) = tree(&clf);
        assert_eq!(keys, expected_keys(expected), "{name}: keys");
        assert_eq!(
            sorted(versions),
            expected_versions(expected),
            "{name}: versions"
        );
        assert_eq!(
            num_parameters as u64,
            expected["num_parameters"].as_u64().unwrap(),
            "{name}"
        );
        let text = to_homura(expected["repr"].as_str().unwrap());
        assert_eq!(clf.display().to_string(), text, "{name}: display");

        // The base model is the classification model's `resnet`.
        let (keys, versions, num_parameters) = tree(&base);
        let base_keys: Vec<_> = expected_keys(expected)
            .into_iter()
            .filter_map(|(k, s, d)| Some((k.strip_prefix("resnet.")?.to_owned(), s, d)))
            .collect();
        assert_eq!(keys, base_keys, "{name}: base keys");
        let mut base_versions: Vec<_> = expected_versions(expected)
            .into_iter()
            .filter_map(|(p, v)| match p.as_str() {
                "resnet" => Some((String::new(), v)),
                _ => Some((p.strip_prefix("resnet.")?.to_owned(), v)),
            })
            .collect();
        base_versions.sort();
        assert_eq!(sorted(versions), base_versions, "{name}: base versions");
        assert_eq!(
            num_parameters as u64,
            structure["base"]["num_parameters"].as_u64().unwrap(),
            "{name}"
        );
        let lines: Vec<&str> = text.lines().collect();
        let start = lines
            .iter()
            .position(|l| *l == "  (resnet): ResNetModel(")
            .unwrap();
        let end = start + lines[start..].iter().position(|l| *l == "  )").unwrap();
        let mut block = vec!["ResNetModel(".to_owned()];
        block.extend(lines[start + 1..end].iter().map(|l| l[2..].to_owned()));
        block.push(")".to_owned());
        assert_eq!(
            base.display().to_string(),
            block.join("\n"),
            "{name}: base display"
        );
    }
}

#[test]
fn the_real_configurations_are_read() {
    let structures = structures();
    let config = ResNetConfig::from_json(&structures["resnet-50"]["config"].to_string()).unwrap();
    assert_eq!(config.common.num_labels(), 1000);
    assert_eq!(config.common.dtype, Some(homura::DType::F32));
    assert!(!config.downsample_in_bottleneck);
    assert_eq!(config.hidden_sizes, [256, 512, 1024, 2048]);
}
