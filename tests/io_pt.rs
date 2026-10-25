//! Loading PyTorch `.pt` files (M3 step 3a): `homura::load` checks the archive and its pickle,
//! but can't rebuild tensors yet. The files are `tests/golden/m3_pt/` from
//! `tools/golden/gen_m3_pt.py`; the unit tests of `src/io` compare what they hold with the
//! reference.

use std::fs;
use std::path::{Path, PathBuf};

use homura::Error;
use homura::io::{Dict, Key, Value, load_mmap};
use homura::nn::{BatchNorm2d, LoadOptions, Module, ModuleVersions, StateDict};

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

#[test]
fn load_reads_pt_files_without_tensors() {
    for load in [
        (|p| homura::load(p)) as fn(&Path) -> homura::Result<StateDict>,
        // SAFETY: nothing modifies the fixture.
        |p| unsafe { load_mmap(p) },
    ] {
        let sd = load(&golden("empty_state_dict.pt")).unwrap();
        assert!(sd.is_empty());
        let versions: ModuleVersions = [("", 1)].into_iter().collect();
        assert_eq!(sd.versions(), Some(&versions));
        assert!(is_format_error(
            load(&golden("state_dict.pt")),
            "PyTorch .pt files with tensors can't be loaded yet"
        ));
        // A checkpoint of other values is not a state dict.
        assert!(is_format_error(
            load(&golden("values_p2.pt")),
            "expected a state dict (a dict of tensors), found a None for \"none\""
        ));
    }
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
