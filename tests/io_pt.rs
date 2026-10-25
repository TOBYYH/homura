//! Loading PyTorch `.pt` files (M3 steps 3a and 3b) through the public entry points:
//! `homura::load`, `io::load_mmap`, `io::load_value` and `io::load_value_from_bytes`. The files are
//! `tests/golden/m3_pt/` from `tools/golden/gen_m3_pt.py`; the unit tests of `src/io` compare
//! everything they hold with the reference, these tests what the entry points make of it.

use std::fs;
use std::path::{Path, PathBuf};

use homura::io::{Dict, Key, Value, load_mmap, load_value, load_value_from_bytes};
use homura::nn::{
    BatchNorm2d, Conv2d, Linear, LoadOptions, Module, ModuleVersions, Sequential, StateDict,
};
use homura::{DType, Error, Tensor};

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/m3_pt")
        .join(name)
}

fn is_format_error<T: std::fmt::Debug>(result: homura::Result<T>, message: &str) -> bool {
    match result {
        Err(Error::Format(m)) => m.contains(message) || panic!("{m:?} lacks {message:?}"),
        other => panic!("expected a format error, got {other:?}"),
    }
}

type Load = fn(&Path) -> homura::Result<StateDict>;

/// `homura::load` and `load_mmap`.
const LOADS: [Load; 2] = [
    |p| homura::load(p),
    // SAFETY: nothing modifies the fixtures.
    |p| unsafe { load_mmap(p) },
];

/// The reference's tensors of a fixture, in order (see the generator).
fn reference(name: &str) -> Vec<Tensor> {
    let mut tensors = homura::load_safetensors(golden(&format!("{name}.safetensors"))).unwrap();
    tensors.sort_by_key(|(k, _)| k.parse::<usize>().unwrap());
    tensors.into_iter().map(|(_, t)| t).collect()
}

fn same(a: &Tensor, b: &Tensor) -> bool {
    a.dtype() == b.dtype() && a.shape() == b.shape() && a.equal(b).all().item::<bool>()
}

#[test]
fn load_reads_pt_state_dicts() {
    let versions: ModuleVersions = [("", 1), ("0", 1), ("1", 2), ("2", 1)]
        .into_iter()
        .collect();
    for load in LOADS {
        let sd = load(&golden("state_dict.pt")).unwrap();
        assert_eq!(
            sd.keys().collect::<Vec<_>>(),
            [
                "0.weight",
                "0.bias",
                "1.weight",
                "1.bias",
                "1.running_mean",
                "1.running_var",
                "1.num_batches_tracked",
                "2.weight",
                "2.bias",
            ]
        );
        for ((key, t), e) in sd.iter().zip(reference("state_dict")) {
            assert!(same(t, &e), "{key}");
            assert!(!t.requires_grad(), "{key}");
        }
        assert_eq!(sd.versions(), Some(&versions));
        // PyTorch's Sequential(Conv2d(1, 2, 1), BatchNorm2d(2), Linear(1, 2)) loads strictly.
        let model = Sequential::new()
            .append(Conv2d::new(1, 2, 1))
            .append(BatchNorm2d::new(2))
            .append(Linear::new(1, 2));
        model.load_state_dict(&sd).unwrap();
        assert!(same(&model.state_dict()["2.weight"], &sd["2.weight"]));

        let empty = load(&golden("empty_state_dict.pt")).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.module_version(""), Some(1));
        // A checkpoint of other values is not a state dict.
        assert!(is_format_error(
            load(&golden("values_p2.pt")),
            "expected a state dict (a dict of tensors), found a None for \"none\"; \
             homura::io::load_value reads other objects from .pt files"
        ));
    }
}

/// `pytorch_model.bin.index.json` and its `.pt` shards, as Hugging Face writes them.
#[test]
fn load_reads_sharded_pt_checkpoints() {
    let whole = homura::load(golden("state_dict.pt")).unwrap();
    for load in LOADS {
        let sd = load(&golden("sharded/pytorch_model.bin.index.json")).unwrap();
        assert_eq!(
            sd.keys().collect::<Vec<_>>(),
            whole.keys().collect::<Vec<_>>()
        );
        for (key, t) in sd.iter() {
            assert!(same(t, &whole[key]), "{key}");
        }
        // The shards are plain dicts, without versions.
        assert!(sd.versions().is_none());
    }
}

/// Views, shared storage and the same object saved twice stay so.
#[test]
fn tensors_keep_their_sharing() {
    for load in LOADS {
        let sd = load(&golden("layouts.pt"));
        // `same` is a list: not a state dict.
        assert!(is_format_error(sd, "found a list for \"same\""));
    }
    let Value::Dict(d) = load_value(golden("layouts.pt")).unwrap() else {
        panic!()
    };
    let tensor = |key: &str| match d.get(key) {
        Some(Value::Tensor(t)) => t.clone(),
        other => panic!("{key}: {other:?}"),
    };
    let (base, strided) = (tensor("base"), tensor("strided"));
    assert_eq!(strided.storage_offset(), 1);
    assert_eq!(strided.stride().as_slice(), &[3]);
    assert_eq!(
        strided.to_vec::<f32>(),
        [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0]
    );
    // A write through the base is seen through its view.
    homura::with_no_grad(|| {
        base.index(1).fill_(100.0);
    });
    assert_eq!(strided.to_vec::<f32>()[0], 100.0);
    assert_eq!(tensor("expanded").stride().as_slice(), &[0, 1]);
    assert!(tensor("param").is_same(&tensor("param_again")));
    assert!(tensor("param").requires_grad() && tensor("param").is_leaf());
    assert!(!tensor("frozen").requires_grad());
    assert!(tensor("requires_grad").requires_grad());
    let Some(Value::List(same_list)) = d.get("same") else {
        panic!()
    };
    let [Value::Tensor(a), Value::Tensor(b)] = &same_list[..] else {
        panic!()
    };
    assert!(a.is_same(b));
    assert_eq!(tensor("ints").dtype(), DType::I64);
    assert_eq!(tensor("ints").to_vec::<i64>(), [1, 2, 4, 5]);
}

#[test]
fn load_value_reads_checkpoints() {
    let Value::Dict(checkpoint) = load_value(golden("checkpoint.pt")).unwrap() else {
        panic!()
    };
    let model = StateDict::try_from(checkpoint.get("model").unwrap()).unwrap();
    assert_eq!(model.module_version("1"), Some(2));
    let Some(Value::Dict(optimizer)) = checkpoint.get("optimizer") else {
        panic!()
    };
    assert!(optimizer.get("state").is_some() && optimizer.get("param_groups").is_some());
    assert!(matches!(checkpoint.get("epoch"), Some(Value::Int(3))));
    // An optimizer's state dict, with zero-dimensional steps.
    let Value::Dict(adam) = load_value(golden("optimizer.pt")).unwrap() else {
        panic!()
    };
    let Some(Value::Dict(state)) = adam.get("state") else {
        panic!()
    };
    let Some(Value::Dict(first)) = state.get(0) else {
        panic!()
    };
    let Some(Value::Tensor(step)) = first.get("step") else {
        panic!()
    };
    assert_eq!(
        (step.num_dims(), step.dtype(), step.item::<f32>()),
        (0, DType::F32, 1.0)
    );
    // From bytes, the same.
    let bytes = fs::read(golden("checkpoint.pt")).unwrap();
    let Value::Dict(again) = load_value_from_bytes(&bytes).unwrap() else {
        panic!()
    };
    assert_eq!(again.len(), checkpoint.len());
}

#[test]
fn load_value_reads_only_pt_files() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("io_pt_value");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    homura::save(&Linear::new(1, 1).state_dict(), &path).unwrap();
    assert!(is_format_error(
        load_value(&path),
        "is not a PyTorch .pt file (a zip archive); read safetensors files and indexes with \
         homura::load"
    ));
    assert!(is_format_error(
        load_value_from_bytes(&fs::read(&path).unwrap()),
        "the data is not a PyTorch .pt file"
    ));
    assert!(is_format_error(
        load_value_from_bytes(b""),
        "the data is not a PyTorch .pt file"
    ));
    assert!(is_format_error(
        load_value(golden("legacy.pt")),
        "is in the format of PyTorch before 1.6, which is not supported"
    ));
    assert!(is_format_error(
        load_value_from_bytes(&fs::read(golden("legacy.pt")).unwrap()),
        "the data is in the format of PyTorch before 1.6"
    ));
    assert!(is_format_error(
        load_value(golden("module.pt")),
        "the global torch.nn.modules.linear.Linear is not supported"
    ));
}

/// Each BUILD binds `_metadata` when it runs, as `inst.__dict__.update(state)` does (R114).
#[test]
fn versions_follow_the_reference_build_semantics() {
    for (name, version) in [
        ("build_once", 1),
        ("build_state_rebound", 1),
        ("build_state_extended", 1),
        ("build_empty_state", 1),
        ("build_replaced", 2),
        ("build_metadata_changed", 2),
    ] {
        let sd = homura::load(golden(&format!("{name}.pt"))).unwrap();
        assert!(sd.is_empty(), "{name}");
        assert_eq!(sd.module_version(""), Some(version), "{name}");
    }
}

#[test]
fn load_rejects_other_pytorch_formats() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("io_pt");
    fs::create_dir_all(&dir).unwrap();
    assert!(is_format_error(
        homura::load(golden("legacy.pt")),
        "is in the format of PyTorch before 1.6, which is not supported"
    ));
    assert!(is_format_error(
        homura::load(golden("torchscript.pt")),
        "TorchScript archive (torch.jit.save)"
    ));
    // Malformed zip archives are errors.
    fs::write(dir.join("short.pt"), b"PK\x03\x04rest").unwrap();
    assert!(is_format_error(
        homura::load(dir.join("short.pt")),
        "invalid zip archive"
    ));
    let mut bytes = fs::read(golden("state_dict.pt")).unwrap();
    bytes.truncate(bytes.len() - 30);
    fs::write(dir.join("truncated.pt"), &bytes).unwrap();
    assert!(is_format_error(
        homura::load(dir.join("truncated.pt")),
        "invalid zip archive"
    ));
}

#[test]
fn values_carry_state_dict_versions() {
    let sd = BatchNorm2d::new(2).state_dict();
    let versions: ModuleVersions = [("", 2)].into_iter().collect();
    assert_eq!(sd.versions(), Some(&versions));
    let value = Value::from(&sd);
    let Value::Dict(dict) = &value else {
        panic!("{value:?}")
    };
    assert!(dict.is_ordered());
    assert_eq!(dict.versions(), Some(&versions));
    let back = StateDict::try_from(&value).unwrap();
    assert_eq!(
        back.keys().collect::<Vec<_>>(),
        sd.keys().collect::<Vec<_>>()
    );
    assert_eq!(back.versions(), Some(&versions));
    // The tensors are shared.
    assert!(back["weight"].is_same(&sd["weight"]));

    // Without num_batches_tracked, version 2 makes strict loading fail, as in PyTorch...
    let mut ordered = Dict::new_ordered();
    let mut plain = Dict::new();
    for (key, v) in dict.iter() {
        if *key != Key::from("num_batches_tracked") {
            ordered.insert(key.clone(), v.clone());
            plain.insert(key.clone(), v.clone());
        }
    }
    ordered.set_versions(dict.versions().cloned());
    let strict = StateDict::try_from(&Value::Dict(ordered)).unwrap();
    let error = BatchNorm2d::new(2).load_state_dict(&strict).unwrap_err();
    assert!(error.to_string().contains("num_batches_tracked"), "{error}");
    // ...while a plain dict has no versions, so the old layout loads.
    let old = StateDict::try_from(&Value::Dict(plain)).unwrap();
    assert!(old.versions().is_none());
    BatchNorm2d::new(2)
        .load_state_dict_with(&old, LoadOptions::default())
        .unwrap();
}

#[test]
fn only_dicts_of_tensors_are_state_dicts() {
    let check = |value: Value, message: &str| {
        assert!(is_format_error(StateDict::try_from(&value), message));
    };
    check(Value::List(vec![]), "found a list");
    let mut dict = Dict::new();
    dict.insert(0, Value::Tensor(homura::zeros([1])));
    check(Value::Dict(dict), "the key 0, which is not a string");
    let mut dict = Dict::new_ordered();
    dict.insert("w", Value::Float(1.0));
    check(Value::Dict(dict), "found a float for \"w\"");
}
