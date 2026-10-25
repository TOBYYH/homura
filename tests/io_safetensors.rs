//! safetensors files and the file entry points (M3 step 2): reading files that Python's
//! `safetensors` wrote (`tests/golden/m3_safetensors/` from `tools/golden/gen_m3_safetensors.py`),
//! streamed and mapped reads against in-memory ones, malformed files, `__metadata__`,
//! `homura::load` / `homura::save`, `io::load_mmap` and sharded checkpoints.
//! `homura_files_for_python` writes files that `tools/golden/check_m3_safetensors.py` reads with
//! Python.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use homura::io::{self, ShardOptions};
use homura::nn::{self, Module, StateDict};
use homura::optim::{Optimizer, Sgd};
use homura::{DType, Device, Error, Tensor};

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/m3_safetensors");
/// A PyTorch `.pt` state dict (from `tools/golden/gen_m3_pt.py`).
const PT_STATE_DICT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/golden/m3_pt/state_dict.pt"
);

/// A fresh, empty directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("io_safetensors")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The values of `gen_m3_safetensors.py`'s formula: flat index i, k = (i * 37 + 5) % 251; BOOL
/// is i % 3 == 0, U8 is k, other integers k - 125, floats (k - 125) / 8.
fn values(shape: &[usize], dtype: DType) -> Tensor {
    let n: usize = shape.iter().product();
    let k: Vec<i64> = (0..n as i64).map(|i| (i * 37 + 5) % 251).collect();
    let t = match dtype {
        DType::Bool => homura::tensor((0..n).map(|i| i % 3 == 0).collect::<Vec<_>>()),
        DType::U8 => homura::tensor(k).to(dtype),
        d if d.is_floating_point() => {
            homura::tensor(k.iter().map(|k| (k - 125) as f64 / 8.0).collect::<Vec<_>>())
                .to(DType::F64)
                .to(d)
        }
        d => homura::tensor(k.iter().map(|k| k - 125).collect::<Vec<_>>()).to(d),
    };
    t.reshape(shape)
}

/// Asserts that `a` has the dtype, shape and values of `b`.
#[track_caller]
fn assert_same(a: &Tensor, b: &Tensor) {
    assert_eq!((a.dtype(), a.shape()), (b.dtype(), b.shape()));
    assert!(a.equal(b).all().item::<bool>(), "{a:?} != {b:?}");
}

/// Asserts that two state dicts have the same keys in the same order and the same tensors.
#[track_caller]
fn assert_same_dict(a: &StateDict, b: &StateDict) {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for ((_, x), (_, y)) in a.iter().zip(b.iter()) {
        assert_same(x, y);
    }
}

const DTYPES: [(&str, DType); 10] = [
    ("bool", DType::Bool),
    ("u8", DType::U8),
    ("i8", DType::I8),
    ("i16", DType::I16),
    ("i32", DType::I32),
    ("i64", DType::I64),
    ("f16", DType::F16),
    ("bf16", DType::BF16),
    ("f32", DType::F32),
    ("f64", DType::F64),
];

/// The tensors of `dtypes.safetensors`, by name.
fn python_dtypes() -> BTreeMap<String, Tensor> {
    let mut out: BTreeMap<String, Tensor> = DTYPES
        .iter()
        .map(|&(name, d)| (name.to_string(), values(&[2, 3], d)))
        .collect();
    out.insert("scalar".into(), values(&[], DType::F32));
    out.insert("empty_i64".into(), values(&[0, 3], DType::I64));
    out.insert("empty_bool".into(), values(&[0], DType::Bool));
    out
}

/// A safetensors file from a header and data, padded as the writer pads it.
fn file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut header = header.to_string();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

/// `io::load_mmap` for files that no test modifies while they are mapped.
fn load_mmap(path: impl AsRef<Path>) -> homura::Result<StateDict> {
    // SAFETY: each test writes its files before loading them and replaces a file only after the
    // tensors mapping it are dropped, or not at all.
    unsafe { io::load_mmap(path) }
}

/// Asserts that two state dicts have the same keys, dtypes, shapes and data bytes.
#[track_caller]
fn assert_bitwise_same(a: &StateDict, b: &StateDict) {
    assert_eq!(
        homura::save_safetensors_to_bytes(a).unwrap(),
        homura::save_safetensors_to_bytes(b).unwrap()
    );
}

fn is_format_error<T>(r: homura::Result<T>, needle: &str) -> bool {
    matches!(r, Err(Error::Format(msg)) if msg.contains(needle))
}

#[test]
fn python_files_load_with_every_dtype() {
    let path = format!("{GOLDEN}/dtypes.safetensors");
    let expected = python_dtypes();
    let streamed = homura::load_safetensors(&path).unwrap();
    let in_memory = homura::load_safetensors_from_bytes(&fs::read(&path).unwrap()).unwrap();
    let state_dict = homura::load(&path).unwrap();
    let mapped = load_mmap(&path).unwrap();
    assert_bitwise_same(&mapped, &state_dict);
    assert_same_dict(&mapped, &state_dict);
    assert!(mapped.versions().is_none());
    assert_eq!(streamed.len(), expected.len());
    // All three keep the header order, which Python's writer sorts by dtype.
    let names: Vec<&str> = streamed.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(names[..2], ["empty_i64", "i64"]);
    assert_eq!(state_dict.keys().collect::<Vec<_>>(), names);
    assert!(state_dict.versions().is_none());
    for (((name, a), (name_b, b)), (name_c, c)) in
        streamed.iter().zip(&in_memory).zip(state_dict.iter())
    {
        assert_eq!((name, name_b.as_str()), (name_b, name_c));
        assert_same(a, &expected[name]);
        assert_same(b, &expected[name]);
        assert_same(c, &expected[name]);
        assert_eq!(a.device(), Device::Cpu);
        assert!(a.is_contiguous());
    }
    // Loaded BOOL data works in kernels: 2 of the 6 positions are true.
    for loaded in [&state_dict, &mapped] {
        assert_eq!(
            loaded.get("bool").unwrap().sum().item::<i64>(),
            2,
            "{:?}",
            loaded.get("bool")
        );
    }
    let metadata = io::safetensors_metadata(&path).unwrap();
    assert_eq!(
        metadata.keys().collect::<Vec<_>>(),
        ["format", "torch_git", "torch_version"]
    );
    assert_eq!(metadata["format"], "pt");
    assert_eq!(metadata["torch_version"], "2.14.1+cpu");
}

#[test]
fn header_order_differs_from_data_order() {
    // "b" comes first in the header but its data comes last; "c" is empty.
    let header = r#"{"b":{"dtype":"I16","shape":[2],"data_offsets":[8,12]},"c":{"dtype":"F32","shape":[0],"data_offsets":[8,8]},"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#;
    let mut data = 1.5f64.to_le_bytes().to_vec();
    data.extend([7, 0, 0xfe, 0xff]);
    let raw = file(header, &data);
    let dir = scratch("order");
    fs::write(dir.join("x.safetensors"), &raw).unwrap();
    for loaded in [
        homura::load_safetensors_from_bytes(&raw).unwrap(),
        homura::load(dir.join("x.safetensors"))
            .unwrap()
            .into_iter()
            .collect(),
        load_mmap(dir.join("x.safetensors"))
            .unwrap()
            .into_iter()
            .collect(),
    ] {
        let names: Vec<&str> = loaded.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(names, ["b", "c", "a"]);
        assert_eq!(loaded[0].1.to_vec::<i16>(), [7, -2]);
        assert_eq!(loaded[1].1.shape(), [0]);
        assert_eq!(loaded[2].1.to_vec::<f64>(), [1.5]);
    }
}

#[test]
fn unaligned_headers_load() {
    let path = format!("{GOLDEN}/unaligned.safetensors");
    let bytes = fs::read(&path).unwrap();
    assert_ne!(u64::from_le_bytes(bytes[..8].try_into().unwrap()) % 8, 0);
    let state_dict = homura::load(&path).unwrap();
    assert_eq!(state_dict.keys().collect::<Vec<_>>(), ["x", "y"]);
    assert_same(state_dict.get("x").unwrap(), &values(&[2], DType::F32));
    assert_same(state_dict.get("y").unwrap(), &values(&[3], DType::I16));
    assert!(io::safetensors_metadata(&path).unwrap().is_empty());
    // The data isn't aligned either, so mapping copies some tensors; the values are the same.
    assert_bitwise_same(&load_mmap(&path).unwrap(), &state_dict);
}

#[test]
fn python_sharded_checkpoints_load_in_weight_map_order() {
    let state_dict =
        homura::load(format!("{GOLDEN}/sharded/model.safetensors.index.json")).unwrap();
    assert_eq!(
        state_dict.keys().collect::<Vec<_>>(),
        ["a.bias", "a.weight", "b.count", "b.weight"]
    );
    assert_same(
        state_dict.get("a.weight").unwrap(),
        &values(&[4, 4], DType::F32),
    );
    assert_same(state_dict.get("a.bias").unwrap(), &values(&[4], DType::F32));
    assert_same(
        state_dict.get("b.weight").unwrap(),
        &values(&[2, 4], DType::F16),
    );
    assert_same(state_dict.get("b.count").unwrap(), &values(&[], DType::I64));
    assert!(state_dict.versions().is_none());
    let mapped = load_mmap(format!("{GOLDEN}/sharded/model.safetensors.index.json")).unwrap();
    assert_bitwise_same(&mapped, &state_dict);
    assert_same_dict(&mapped, &state_dict);
}

fn longer_by_one(bytes: &[u8]) -> Vec<u8> {
    let mut longer = bytes.to_vec();
    longer.push(0);
    longer
}

#[test]
fn truncated_and_corrupt_files_are_errors() {
    let bytes = fs::read(format!("{GOLDEN}/dtypes.safetensors")).unwrap();
    // Every proper prefix fails the size checks before any data is read.
    for cut in 0..bytes.len() {
        let r = homura::load_safetensors_from_bytes(&bytes[..cut]);
        assert!(matches!(r, Err(Error::Format(_))), "cut at {cut}");
    }
    let dir = scratch("truncated");
    let path = dir.join("cut.safetensors");
    for cut in [0, 7, 8, 9, 100, bytes.len() - 1] {
        fs::write(&path, &bytes[..cut]).unwrap();
        assert!(
            matches!(homura::load(&path), Err(Error::Format(_))),
            "cut at {cut}"
        );
        assert!(matches!(
            homura::load_safetensors(&path),
            Err(Error::Format(_))
        ));
        assert!(matches!(load_mmap(&path), Err(Error::Format(_))));
    }
    fs::write(&path, longer_by_one(&bytes)).unwrap();
    assert!(is_format_error(
        load_mmap(&path),
        "data section doesn't match"
    ));
    // A trailing byte, offsets past the data, and headers longer than the file or the limit.
    let longer = longer_by_one(&bytes);
    assert!(is_format_error(
        homura::load_safetensors_from_bytes(&longer),
        "data section doesn't match"
    ));
    let past = file(
        r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#,
        &[0; 8],
    );
    assert!(is_format_error(
        homura::load_safetensors_from_bytes(&past),
        "gap"
    ));
    for header_len in [1_000u64, 99_999_999, 100_000_001, u64::MAX] {
        let mut raw = header_len.to_le_bytes().to_vec();
        raw.extend_from_slice(b"{}");
        fs::write(&path, &raw).unwrap();
        for r in [
            homura::load_safetensors_from_bytes(&raw),
            homura::load_safetensors(&path),
            load_mmap(&path).map(|sd| sd.into_iter().collect()),
        ] {
            assert!(matches!(r, Err(Error::Format(_))), "{header_len}");
        }
        assert!(io::safetensors_metadata(&path).is_err());
    }
}

#[test]
fn non_canonical_bool_bytes_are_errors() {
    let header = r#"{"ok":{"dtype":"BOOL","shape":[2],"data_offsets":[0,2]},"x":{"dtype":"BOOL","shape":[2],"data_offsets":[2,4]}}"#;
    let dir = scratch("bool");
    for byte in [2u8, 255] {
        let raw = file(header, &[1, 0, 0, byte]);
        let path = dir.join("bool.safetensors");
        fs::write(&path, &raw).unwrap();
        let message = "x: BOOL data must consist of the bytes 0 and 1";
        assert!(is_format_error(
            homura::load_safetensors_from_bytes(&raw),
            message
        ));
        assert!(is_format_error(homura::load_safetensors(&path), message));
        assert!(is_format_error(homura::load(&path), message));
        // The mapped data is checked too, before a tensor uses it.
        assert!(is_format_error(load_mmap(&path), message));
        // Also when the file is a shard.
        let index = r#"{"weight_map": {"ok": "bool.safetensors", "x": "bool.safetensors"}}"#;
        fs::write(dir.join("index.json"), index).unwrap();
        assert!(is_format_error(
            homura::load(dir.join("index.json")),
            message
        ));
        assert!(is_format_error(load_mmap(dir.join("index.json")), message));
    }
    let raw = file(header, &[1, 0, 0, 1]);
    let loaded = homura::load_safetensors_from_bytes(&raw).unwrap();
    assert_eq!(loaded[1].1.to_vec::<bool>(), [false, true]);
    fs::write(dir.join("bool.safetensors"), &raw).unwrap();
    let mapped = load_mmap(dir.join("bool.safetensors")).unwrap();
    assert_eq!(mapped.get("x").unwrap().to_vec::<bool>(), [false, true]);
    assert_eq!(
        mapped.get("x").unwrap().logical_not().to_vec::<bool>(),
        [true, false]
    );
}

#[test]
fn metadata_is_written_and_read() {
    let dir = scratch("metadata");
    let state_dict: StateDict = [("w", values(&[2, 2], DType::F32))].into_iter().collect();
    let metadata = BTreeMap::from([
        ("format".to_string(), "pt".to_string()),
        ("note \"quoted\"\n".to_string(), "é\u{1}".to_string()),
    ]);
    let path = dir.join("with.safetensors");
    io::save_safetensors_with_metadata(&state_dict, &path, &metadata).unwrap();
    assert_eq!(io::safetensors_metadata(&path).unwrap(), metadata);
    assert_same_dict(&homura::load(&path).unwrap(), &state_dict);

    // An empty map is written as an empty `__metadata__`, as Python writes `metadata={}`.
    let empty = dir.join("empty.safetensors");
    io::save_safetensors_with_metadata(&state_dict, &empty, &BTreeMap::new()).unwrap();
    let raw = fs::read(&empty).unwrap();
    assert!(raw.windows(17).any(|w| w == br#""__metadata__":{}"#));
    assert!(io::safetensors_metadata(&empty).unwrap().is_empty());

    // `homura::save` and the older functions write none.
    let plain = dir.join("plain.safetensors");
    homura::save(&state_dict, &plain).unwrap();
    assert!(
        !fs::read(&plain)
            .unwrap()
            .windows(12)
            .any(|w| w == b"__metadata__")
    );
    assert!(io::safetensors_metadata(&plain).unwrap().is_empty());

    // Metadata that is not a map of strings is rejected when reading.
    let raw = file(r#"{"__metadata__":{"a":1}}"#, &[]);
    fs::write(&plain, &raw).unwrap();
    assert!(is_format_error(
        io::safetensors_metadata(&plain),
        "__metadata__"
    ));
    assert!(is_format_error(homura::load(&plain), "__metadata__"));
}

#[test]
fn state_dicts_round_trip() {
    let dir = scratch("round_trip");
    let model = nn::Sequential::new()
        .append(nn::Linear::new(3, 4))
        .append(nn::BatchNorm2d::new(4));
    let mut state_dict = model.state_dict();
    assert!(state_dict.versions().is_some());
    let transposed = values(&[3, 4], DType::F16).transpose(0, 1);
    assert!(!transposed.is_contiguous());
    state_dict.insert("transposed", transposed.clone());
    // Two keys for one tensor: each key gets its own copy.
    let weight = state_dict.get("0.weight").unwrap().clone();
    state_dict.insert("tied", weight.clone());
    let path = dir.join("model.SafeTensors");
    homura::save(&state_dict, &path).unwrap();
    // The header is padded so that the data starts 8-byte aligned.
    let raw = fs::read(&path).unwrap();
    assert!(u64::from_le_bytes(raw[..8].try_into().unwrap()).is_multiple_of(8));
    let loaded = homura::load(&path).unwrap();
    assert_same_dict(&loaded, &state_dict);
    assert!(loaded.versions().is_none());
    assert!(loaded.get("transposed").unwrap().is_contiguous());
    // The bytes match the older entry points'.
    let pairs: Vec<(&str, &Tensor)> = state_dict.iter().collect();
    assert_eq!(
        fs::read(&path).unwrap(),
        homura::save_safetensors_to_bytes(pairs).unwrap()
    );
    // The model loads what was saved.
    let other = nn::Sequential::new()
        .append(nn::Linear::new(3, 4))
        .append(nn::BatchNorm2d::new(4));
    let mut entries = loaded.clone();
    entries.remove("transposed");
    entries.remove("tied");
    other.load_state_dict(&entries).unwrap();
    assert_same_dict(&other.state_dict(), &model.state_dict());
    // Each key got its own copy of the tied tensor.
    let (a, b) = (loaded.get("0.weight").unwrap(), loaded.get("tied").unwrap());
    a.fill_(0.0);
    assert_same(b, &weight);
}

#[test]
fn save_picks_the_format_by_extension() {
    let dir = scratch("extension");
    let state_dict: StateDict = [("w", homura::ones([2]))].into_iter().collect();
    for name in ["model.pt", "model.pth", "model.bin", "model.PT"] {
        let path = dir.join(name);
        assert!(is_format_error(
            homura::save(&state_dict, &path),
            "can't be saved yet"
        ));
        assert!(!path.exists());
    }
    for name in ["model.txt", "model", "model.safetensors.json"] {
        let path = dir.join(name);
        assert!(is_format_error(
            homura::save(&state_dict, &path),
            "from its extension"
        ));
        assert!(!path.exists());
    }
    // Checks happen before the file is created.
    let path = dir.join("bad.safetensors");
    let meta: StateDict = [("m", homura::zeros_with([2], Device::Meta))]
        .into_iter()
        .collect();
    assert!(is_format_error(homura::save(&meta, &path), "meta tensor"));
    let reserved: StateDict = [("__metadata__", homura::ones([1]))].into_iter().collect();
    assert!(is_format_error(homura::save(&reserved, &path), "reserved"));
    let options = ShardOptions::default();
    assert!(is_format_error(
        io::save_sharded(&meta, dir.join("shards"), options),
        "meta tensor"
    ));
    assert!(!path.exists());
}

#[test]
fn load_tells_formats_by_content() {
    let dir = scratch("content");
    let state_dict: StateDict = [("w", values(&[3], DType::I32))].into_iter().collect();
    // A safetensors file under another name.
    let bytes = homura::save_safetensors_to_bytes([("w", state_dict.get("w").unwrap())]).unwrap();
    fs::write(dir.join("weights.bin"), &bytes).unwrap();
    assert_same_dict(&homura::load(dir.join("weights.bin")).unwrap(), &state_dict);
    // A zip archive is PyTorch's format, whatever the name.
    fs::copy(PT_STATE_DICT, dir.join("model.safetensors")).unwrap();
    let pt = homura::load(PT_STATE_DICT).unwrap();
    for loaded in [
        homura::load(dir.join("model.safetensors")).unwrap(),
        load_mmap(dir.join("model.safetensors")).unwrap(),
    ] {
        assert_eq!(loaded.len(), 9);
        assert_same_dict(&loaded, &pt);
    }
    assert_same_dict(&load_mmap(dir.join("weights.bin")).unwrap(), &state_dict);
    // A header length whose first byte is `{`: still safetensors, since its upper bytes are 0.
    let header = format!("{{{}}}", " ".repeat(b'{' as usize - 2));
    let mut raw = (header.len() as u64).to_le_bytes().to_vec();
    raw.extend_from_slice(header.as_bytes());
    assert_eq!(raw[0], b'{');
    fs::write(dir.join("brace.safetensors"), &raw).unwrap();
    assert!(
        homura::load(dir.join("brace.safetensors"))
            .unwrap()
            .is_empty()
    );
    // JSON without a weight map, and short or empty files.
    for (content, needle) in [
        (&b"{}"[..], "missing weight_map"),
        (b"{\"weight_map\": []}", "not an object"),
        (b"{\"weight_map\": ", "invalid JSON"),
        (b"", "missing header length"),
        (b"abc", "missing header length"),
    ] {
        fs::write(dir.join("x"), content).unwrap();
        assert!(
            is_format_error(homura::load(dir.join("x")), needle),
            "{content:?}"
        );
        assert!(
            is_format_error(load_mmap(dir.join("x")), needle),
            "{content:?}"
        );
    }
    assert!(matches!(
        homura::load(dir.join("missing")),
        Err(Error::Io(_))
    ));
    assert!(matches!(load_mmap(dir.join("missing")), Err(Error::Io(_))));
}

#[test]
fn mapped_tensors_are_copy_on_write() {
    let dir = scratch("mmap_cow");
    let path = dir.join("model.safetensors");
    let original: StateDict = [
        ("w", values(&[2, 3], DType::F32)),
        ("n", values(&[4], DType::I64)),
        ("m", values(&[3], DType::Bool)),
    ]
    .into_iter()
    .collect();
    homura::save(&original, &path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let mapped = load_mmap(&path).unwrap();
    let other = load_mmap(&path).unwrap();
    mapped.get("w").unwrap().mul_(2.0);
    mapped.get("n").unwrap().fill_(7);
    mapped.get("m").unwrap().fill_(true);
    assert_same(
        mapped.get("w").unwrap(),
        &values(&[2, 3], DType::F32).mul(2.0),
    );
    assert_eq!(mapped.get("n").unwrap().to_vec::<i64>(), [7; 4]);
    assert_eq!(mapped.get("m").unwrap().to_vec::<bool>(), [true; 3]);
    // The writes reach neither the file nor another mapping of it.
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_bitwise_same(&other, &original);
    assert_bitwise_same(&homura::load(&path).unwrap(), &original);
}

#[test]
fn mapped_tensors_outlive_their_state_dict() {
    let dir = scratch("mmap_lifetime");
    let path = dir.join("linear.safetensors");
    let source = nn::Linear::new(3, 2);
    homura::save(&source.state_dict(), &path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let mapped = load_mmap(&path).unwrap();
    let weight = mapped.get("weight").unwrap().clone();
    let transposed = weight.transpose(0, 1);
    // Loading with assign makes the parameters use the mapped data; copying doesn't.
    let model = nn::Linear::new(3, 2);
    let copied = nn::Linear::new(3, 2);
    model
        .load_state_dict_with(&mapped, nn::LoadOptions::default().assign(true))
        .unwrap();
    copied.load_state_dict(&mapped).unwrap();
    drop(mapped);
    assert_same(&transposed, &source.weight.transpose(0, 1));
    let x = homura::randn([4, 3]);
    assert_same(&model.forward(&x), &source.forward(&x));
    // A training step writes the parameters in place: the handles that share the mapped storage
    // see it, the file and the copies don't.
    let mut opt = Sgd::new(model.parameters(), 0.5);
    model.forward(&x).sum().backward();
    opt.step();
    assert!(!model.weight.equal(&*source.weight).all().item::<bool>());
    assert_same(&weight, &model.weight);
    assert_same(&transposed, &model.weight.transpose(0, 1));
    assert_same(&copied.weight, &source.weight);
    assert_eq!(fs::read(&path).unwrap(), bytes);
    drop((weight, transposed));
    // The parameters keep the mapping alive on their own.
    assert_eq!(model.forward(&x).shape(), [4, 2]);
    assert_bitwise_same(&homura::load(&path).unwrap(), &source.state_dict());
}

/// A state dict with tensors of 64 (`a`), 16 (`b`), 400 (`c`), 8 (`d`) and 0 (`e`) bytes.
fn shard_test_dict() -> StateDict {
    [
        ("a", values(&[4, 4], DType::F32)),
        ("b", values(&[8], DType::BF16)),
        ("c", values(&[50], DType::F64)),
        ("d", values(&[8], DType::Bool)),
        ("e", values(&[0], DType::I64)),
    ]
    .into_iter()
    .collect()
}

#[test]
fn sharded_saves_round_trip() {
    let dir = scratch("shards").join("new");
    let state_dict = shard_test_dict();
    let options = ShardOptions::default().max_shard_size(80).prefix("net");
    let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
    io::save_sharded(&state_dict, &dir, options.metadata(metadata.clone())).unwrap();
    // a + b fit in 80 bytes; c alone exceeds the limit; d and e share the last shard.
    let mut files: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    files.sort();
    assert_eq!(
        files,
        [
            "net-00001-of-00003.safetensors",
            "net-00002-of-00003.safetensors",
            "net-00003-of-00003.safetensors",
            "net.safetensors.index.json"
        ]
    );
    let index = fs::read_to_string(dir.join("net.safetensors.index.json")).unwrap();
    assert_eq!(
        index,
        r#"{
  "metadata": {
    "total_size": 488
  },
  "weight_map": {
    "a": "net-00001-of-00003.safetensors",
    "b": "net-00001-of-00003.safetensors",
    "c": "net-00002-of-00003.safetensors",
    "d": "net-00003-of-00003.safetensors",
    "e": "net-00003-of-00003.safetensors"
  }
}
"#
    );
    for file in &files[..3] {
        assert_eq!(io::safetensors_metadata(dir.join(file)).unwrap(), metadata);
    }
    let shard = homura::load(dir.join("net-00002-of-00003.safetensors")).unwrap();
    assert_eq!(shard.keys().collect::<Vec<_>>(), ["c"]);
    assert_same_dict(
        &homura::load(dir.join("net.safetensors.index.json")).unwrap(),
        &state_dict,
    );

    // One shard: only `<prefix>.safetensors`, without an index, here without metadata.
    let single = scratch("shards_single");
    io::save_sharded(&state_dict, &single, ShardOptions::default()).unwrap();
    let files: Vec<_> = fs::read_dir(&single)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(files, ["model.safetensors"]);
    assert_same_dict(
        &homura::load(single.join("model.safetensors")).unwrap(),
        &state_dict,
    );
    assert!(
        io::safetensors_metadata(single.join("model.safetensors"))
            .unwrap()
            .is_empty()
    );
    // Data exactly at the limit stays in one shard; one byte less splits off the last tensor
    // with data (the empty one follows it).
    let limit = scratch("shards_limit");
    io::save_sharded(
        &state_dict,
        &limit,
        ShardOptions::default().max_shard_size(488),
    )
    .unwrap();
    assert!(!limit.join("model.safetensors.index.json").exists());
    io::save_sharded(
        &state_dict,
        &limit,
        ShardOptions::default().max_shard_size(487),
    )
    .unwrap();
    let index = fs::read_to_string(limit.join("model.safetensors.index.json")).unwrap();
    assert!(index.contains(r#""c": "model-00001-of-00002.safetensors""#));
    assert!(index.contains(r#""d": "model-00002-of-00002.safetensors""#));
    // An empty state dict gives one file without tensors.
    let empty = StateDict::new();
    io::save_sharded(
        &empty,
        single.join("empty"),
        ShardOptions::default().max_shard_size(1),
    )
    .unwrap();
    assert!(
        homura::load(single.join("empty/model.safetensors"))
            .unwrap()
            .is_empty()
    );
    // Each tensor alone: every shard holds one tensor.
    let each = scratch("shards_each");
    io::save_sharded(
        &state_dict,
        &each,
        ShardOptions::default().max_shard_size(1),
    )
    .unwrap();
    assert_eq!(fs::read_dir(&each).unwrap().count(), 6);
    assert_same_dict(
        &homura::load(each.join("model.safetensors.index.json")).unwrap(),
        &state_dict,
    );
    // Loading keeps the order of the weight map, also when it is not sorted.
    let entries: Vec<(&str, &Tensor)> = state_dict.iter().collect();
    let reversed: StateDict = entries
        .into_iter()
        .rev()
        .map(|(k, t)| (k, t.clone()))
        .collect();
    let back = scratch("shards_reversed");
    io::save_sharded(
        &reversed,
        &back,
        ShardOptions::default().max_shard_size(100),
    )
    .unwrap();
    let index = back.join("model.safetensors.index.json");
    assert_same_dict(&homura::load(&index).unwrap(), &reversed);
}

/// The checkpoint that transformers' discovery in a local directory picks for the default
/// prefix (R106): `model.safetensors` first, then `model.safetensors.index.json`.
fn discover(dir: &Path) -> StateDict {
    let single = dir.join("model.safetensors");
    let path = if single.is_file() {
        single
    } else {
        dir.join("model.safetensors.index.json")
    };
    homura::load(path).unwrap()
}

/// The sorted names in `dir`.
fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// Three 16-byte tensors whose values depend on `v`.
fn version(v: i64) -> StateDict {
    [
        ("z", homura::full([4], v as f32)),
        ("a", homura::full([4], v as f32 + 0.5)),
        ("m", homura::tensor([v, -v])),
    ]
    .into_iter()
    .collect()
}

#[test]
fn sharded_saves_replace_earlier_layouts() {
    let dir = scratch("layouts");
    let save = |sd: &StateDict, limit: u64, prefix: &str| {
        let options = ShardOptions::default().max_shard_size(limit).prefix(prefix);
        io::save_sharded(sd, &dir, options)
    };
    let shards = |prefix: &str, n: usize| -> Vec<String> {
        (1..=n)
            .map(|i| format!("{prefix}-{i:05}-of-{n:05}.safetensors"))
            .collect()
    };
    // Files that no save of the prefix "model" may touch.
    let unrelated = [
        "notes.txt",
        "net.safetensors",
        "model-extra.safetensors",
        "model-a-of-2.safetensors",
        "model-00001-of-00002.safetensors.bak",
        "model-00001-of-x.safetensors",
        "model00001-of-00002.safetensors",
        "model-1-of-2.bin",
    ];
    for (i, name) in unrelated.iter().enumerate() {
        fs::write(dir.join(name), format!("unrelated {i}")).unwrap();
    }
    fs::create_dir(dir.join("model-00009-of-00009.safetensors")).unwrap();
    let expect = |mut ours: Vec<String>| {
        ours.extend(unrelated.iter().map(|s| s.to_string()));
        ours.push("model-00009-of-00009.safetensors".into());
        ours.sort();
        assert_eq!(listing(&dir), ours);
    };

    // A single file, then shards: the single file would be found first, so it goes.
    save(&version(1), 1000, "model").unwrap();
    expect(vec!["model.safetensors".into()]);
    save(&version(2), 20, "model").unwrap();
    let mut files = shards("model", 3);
    files.push("model.safetensors.index.json".into());
    expect(files);
    assert_same_dict(&discover(&dir), &version(2));
    // Fewer shards: the old numbered shards go.
    save(&version(3), 32, "model").unwrap();
    let mut files = shards("model", 2);
    files.push("model.safetensors.index.json".into());
    expect(files.clone());
    assert_same_dict(&discover(&dir), &version(3));
    // The same layout with new values.
    save(&version(4), 32, "model").unwrap();
    expect(files);
    assert_same_dict(&discover(&dir), &version(4));
    // Shards, then a single file: the index and its shards go.
    save(&version(5), 1000, "model").unwrap();
    expect(vec!["model.safetensors".into()]);
    assert_same_dict(&discover(&dir), &version(5));
    assert!(matches!(
        homura::load(dir.join("model.safetensors.index.json")),
        Err(Error::Io(_))
    ));

    // A state dict that can't be saved leaves the earlier checkpoint as it was, in both layouts.
    let mut meta = version(9);
    meta.insert("z", homura::zeros_with([4], Device::Meta));
    assert!(is_format_error(save(&meta, 20, "model"), "meta tensor"));
    expect(vec!["model.safetensors".into()]);
    assert_same_dict(&discover(&dir), &version(5));
    save(&version(6), 20, "model").unwrap();
    assert!(is_format_error(save(&meta, 1000, "model"), "meta tensor"));
    let mut files = shards("model", 3);
    files.push("model.safetensors.index.json".into());
    expect(files.clone());
    assert_same_dict(&discover(&dir), &version(6));

    // Another prefix only manages its own files.
    save(&version(7), 20, "enc").unwrap();
    let mut with_enc = files.clone();
    with_enc.extend(shards("enc", 3));
    with_enc.push("enc.safetensors.index.json".into());
    expect(with_enc);
    save(&version(8), 1000, "enc").unwrap();
    let mut with_enc = files;
    with_enc.push("enc.safetensors".into());
    expect(with_enc);
    assert_same_dict(
        &homura::load(dir.join("enc.safetensors")).unwrap(),
        &version(8),
    );
    assert_same_dict(&discover(&dir), &version(6));
    for (i, name) in unrelated.iter().enumerate() {
        assert_eq!(
            fs::read_to_string(dir.join(name)).unwrap(),
            format!("unrelated {i}")
        );
    }
}

#[test]
fn sharded_index_errors() {
    let dir = scratch("index_errors");
    let state_dict = shard_test_dict();
    io::save_sharded(
        &state_dict,
        &dir,
        ShardOptions::default().max_shard_size(80),
    )
    .unwrap();
    let shard = |n: usize| format!("model-0000{n}-of-00003.safetensors");
    let index = dir.join("model.safetensors.index.json");
    let original = fs::read_to_string(&index).unwrap();
    let check = |text: &str, needle: &str| {
        fs::write(&index, text).unwrap();
        let r = homura::load(&index);
        assert!(
            is_format_error(r, needle),
            "{needle}: {:?}",
            homura::load(&index).err()
        );
    };
    // A key the shard doesn't have, and a tensor the index doesn't list.
    check(&original.replace("\"b\"", "\"z\""), "has no tensor \"z\"");
    check(
        &original.replace("    \"b\": \"model-00001-of-00003.safetensors\",\n", ""),
        "doesn't list: [\"b\"]",
    );
    // A key listed for the wrong shard.
    check(
        &original.replace(
            &format!("\"c\": \"{}\"", shard(2)),
            &format!("\"c\": \"{}\"", shard(3)),
        ),
        "has no tensor \"c\"",
    );
    // Shard names that leave the directory.
    for name in [
        "../model-00002-of-00003.safetensors",
        "sub/x.safetensors",
        "a\\\\b",
        "..",
        ".",
        "",
    ] {
        let text = original.replace(&shard(2), name);
        check(&text, "is not a file in the index's directory");
    }
    let absolute = dir.join(shard(2));
    check(
        &original.replace(&shard(2), absolute.to_str().unwrap()),
        "is not a file in the index's directory",
    );
    check(
        &original.replace(&format!("\"{}\"", shard(2)), "2"),
        "not a file name",
    );
    check(
        r#"{"metadata": {}, "weight_map": "x"}"#,
        "weight_map is not an object",
    );
    check(r#"{"weight_map": {"a": "x", "a": "y"}}"#, "invalid JSON");
    // A shard that is an index, or a zip archive.
    check(
        &original.replace(&shard(2), "model.safetensors.index.json"),
        "is an index, not a shard",
    );
    fs::copy(PT_STATE_DICT, dir.join("zip.bin")).unwrap();
    // A .pt shard is read, but doesn't have the keys listed for it.
    check(
        &original.replace(&shard(2), "zip.bin"),
        "\"zip.bin\" has no tensor",
    );
    // A missing shard is an IO error.
    fs::write(&index, original.replace(&shard(2), "gone.safetensors")).unwrap();
    assert!(matches!(homura::load(&index), Err(Error::Io(_))));
    // The original still loads.
    fs::write(&index, &original).unwrap();
    assert_same_dict(&homura::load(&index).unwrap(), &state_dict);
}

#[test]
fn shard_options_check_their_arguments() {
    for (f, message) in [
        (
            Box::new(|| drop(ShardOptions::default().max_shard_size(0))) as Box<dyn Fn()>,
            "the maximum shard size must be positive",
        ),
        (
            Box::new(|| drop(ShardOptions::default().prefix(""))),
            "not \"\"",
        ),
        (
            Box::new(|| drop(ShardOptions::default().prefix("a/b"))),
            "not \"a/b\"",
        ),
        (
            Box::new(|| drop(ShardOptions::default().prefix("a\\b"))),
            "not \"a\\\\b\"",
        ),
        (
            Box::new(|| drop(ShardOptions::default().prefix(".."))),
            "not \"..\"",
        ),
        (
            Box::new(|| drop(ShardOptions::default().prefix("model-00001-of-00002"))),
            "\"model-00001-of-00002\" ends like a numbered shard",
        ),
        (
            Box::new(|| drop(ShardOptions::default().prefix("x-1-of-2"))),
            "\"x-1-of-2\" ends like a numbered shard",
        ),
        (
            Box::new(|| drop(ShardOptions::default().prefix("a-1-of-2-3-of-4"))),
            "ends like a numbered shard",
        ),
    ] {
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_err();
        let msg = match err.downcast_ref::<&str>() {
            Some(msg) => msg.to_string(),
            None => err.downcast_ref::<String>().unwrap().clone(),
        };
        assert!(msg.contains(message), "{msg}");
    }
    // Names that only resemble numbered shards are prefixes like any other.
    for prefix in [
        "model-00001-of-x",
        "model-1-of-",
        "model-of-2",
        "model1-of-2",
        "x--of-2",
        "-1-of-2x",
        "v2",
        "a-1",
    ] {
        drop(ShardOptions::default().prefix(prefix));
    }
}

/// The checkpoint `name` with one tensor `z` of value `v`, written by `save_safetensors` as
/// another tool would write it.
fn plain_file(dir: &Path, name: &str, v: i64) {
    homura::save_safetensors([("z", &homura::tensor([v]))], dir.join(name)).unwrap();
}

fn z_of(path: PathBuf) -> i64 {
    homura::load(path).unwrap()["z"].item::<i64>()
}

#[test]
fn cleanup_only_removes_shards_the_earlier_index_lists() {
    let dir = scratch("cleanup");
    let save = |sd: &StateDict, limit: u64, prefix: &str| {
        let options = ShardOptions::default().max_shard_size(limit).prefix(prefix);
        io::save_sharded(sd, &dir, options).unwrap();
    };
    // A prefix that would name another prefix's shards is refused before anything is written,
    // in either order of the two saves.
    let refused = |prefix: &'static str| {
        std::panic::catch_unwind(|| ShardOptions::default().prefix(prefix)).is_err()
    };
    assert!(refused("model-00001-of-00002"));
    assert_eq!(listing(&dir), Vec::<String>::new());
    save(&version(88), 1000, "model");
    assert!(refused("model-00001-of-00002"));
    assert_eq!(listing(&dir), ["model.safetensors"]);

    // A file named like a shard of `model` that no index of `model` lists is not touched when
    // `model` changes layouts, as long as no new file has its name.
    plain_file(&dir, "model-00001-of-00002.safetensors", 77);
    save(&version(1), 20, "model");
    save(&version(2), 1000, "model");
    assert_eq!(
        listing(&dir),
        ["model-00001-of-00002.safetensors", "model.safetensors"]
    );
    assert_eq!(z_of(dir.join("model-00001-of-00002.safetensors")), 77);
    assert_same_dict(&discover(&dir), &version(2));
    // A new shard with that name overwrites it, as any save overwrites its own file names; the
    // next layout change removes it with the shards the index lists.
    save(&version(3), 32, "model");
    assert_same_dict(&discover(&dir), &version(3));
    save(&version(4), 1000, "model");
    assert_eq!(listing(&dir), ["model.safetensors"]);

    // An index lists names of other prefixes, other files and leaving names, and a missing
    // shard: only its own numbered shards go.
    let others = [
        "notes.txt",
        "enc-00001-of-00002.safetensors",
        "model-00001-of-x.safetensors",
        "model-00007-of-00009.safetensors",
    ];
    for (i, name) in others.iter().enumerate() {
        plain_file(&dir, name, i as i64);
    }
    let outside = dir.parent().unwrap().join("cleanup-outside.safetensors");
    plain_file(dir.parent().unwrap(), "cleanup-outside.safetensors", 5);
    let index = r#"{"weight_map": {
        "a": "notes.txt", "b": "enc-00001-of-00002.safetensors",
        "c": "model-00001-of-x.safetensors", "d": "../cleanup-outside.safetensors",
        "e": "model-00007-of-00009.safetensors", "f": "model-00002-of-00009.safetensors",
        "g": "model-00003-of-00009.safetensors", "h": "model--of-00009.safetensors"}}"#;
    // A directory with a listed name is left alone.
    fs::create_dir(dir.join("model-00003-of-00009.safetensors")).unwrap();
    plain_file(&dir, "model--of-00009.safetensors", 4);
    fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
    fs::remove_file(dir.join("model.safetensors")).unwrap();
    save(&version(5), 1000, "model");
    let mut expected = vec![
        "enc-00001-of-00002.safetensors",
        "model--of-00009.safetensors",
        "model-00003-of-00009.safetensors",
        "model-00001-of-x.safetensors",
        "model.safetensors",
        "notes.txt",
    ];
    expected.sort();
    assert_eq!(listing(&dir), expected);
    assert_eq!(z_of(outside.clone()), 5);
    assert_same_dict(&discover(&dir), &version(5));

    // Without a readable earlier index, numbered files are left alone; the index itself goes.
    save(&version(6), 20, "model");
    fs::write(dir.join("model.safetensors.index.json"), "not json").unwrap();
    save(&version(7), 1000, "model");
    assert!(!dir.join("model.safetensors.index.json").exists());
    assert!(dir.join("model-00003-of-00003.safetensors").exists());
    assert_same_dict(&discover(&dir), &version(7));

    // The same for another main prefix, next to `model`.
    plain_file(&dir, "enc-00002-of-00009.safetensors", 9);
    save(&version(10), 20, "enc");
    save(&version(11), 1000, "enc");
    assert!(!dir.join("enc-00001-of-00003.safetensors").exists());
    assert!(!dir.join("enc.safetensors.index.json").exists());
    assert_eq!(z_of(dir.join("enc-00002-of-00009.safetensors")), 9);
    assert_same_dict(
        &homura::load(dir.join("enc.safetensors")).unwrap(),
        &version(11),
    );
    assert_same_dict(&discover(&dir), &version(7));
}

/// Symbolic links among the files to remove are removed themselves, not their targets.
#[cfg(unix)]
#[test]
fn cleanup_removes_links_not_their_targets() {
    use std::os::unix::fs::symlink;
    let dir = scratch("links");
    let outside = scratch("links_outside");
    plain_file(&outside, "target.safetensors", 3);
    fs::create_dir(outside.join("target_dir")).unwrap();
    symlink(
        outside.join("target.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();
    let options = ShardOptions::default().max_shard_size(20);
    io::save_sharded(&version(1), &dir, options).unwrap();
    assert!(fs::symlink_metadata(dir.join("model.safetensors")).is_err());
    assert_eq!(z_of(outside.join("target.safetensors")), 3);
    // An old shard that is a link to a directory.
    fs::remove_file(dir.join("model-00003-of-00003.safetensors")).unwrap();
    symlink(
        outside.join("target_dir"),
        dir.join("model-00003-of-00003.safetensors"),
    )
    .unwrap();
    io::save_sharded(&version(2), &dir, ShardOptions::default()).unwrap();
    assert_eq!(listing(&dir), ["model.safetensors"]);
    assert!(outside.join("target_dir").is_dir());
    assert_same_dict(&discover(&dir), &version(2));
}

/// Writes files for `tools/golden/check_m3_safetensors.py` to read with Python, into
/// `target/tmp/m3_safetensors/`: `metadata.safetensors` (every dtype of the formula as
/// "<dtype>" with shape [2, 3], a transposed F32 "transposed" ([3, 2], saved contiguous), a
/// zero-dimensional "scalar" and an empty "empty", with `__metadata__` {"format": "pt",
/// "writer": "homura"}) and `sharded/` (`save_sharded` of the same tensors with a 40-byte limit
/// and the same metadata). `to_sharded/` and `to_single/` hold the same tensors saved over an
/// earlier checkpoint of zeros in the other layout: zeros as one file then the tensors as shards
/// with a 40-byte limit, and zeros as shards then the tensors as one file (both without metadata).
#[test]
fn homura_files_for_python() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("m3_safetensors");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut state_dict: StateDict = DTYPES
        .iter()
        .map(|&(name, d)| (name, values(&[2, 3], d)))
        .collect();
    state_dict.insert("transposed", values(&[2, 3], DType::F32).transpose(0, 1));
    state_dict.insert("scalar", values(&[], DType::F64));
    state_dict.insert("empty", values(&[0, 2], DType::I16));
    let metadata = BTreeMap::from([
        ("format".to_string(), "pt".to_string()),
        ("writer".to_string(), "homura".to_string()),
    ]);
    io::save_safetensors_with_metadata(&state_dict, dir.join("metadata.safetensors"), &metadata)
        .unwrap();
    let options = ShardOptions::default()
        .max_shard_size(40)
        .metadata(metadata);
    io::save_sharded(&state_dict, dir.join("sharded"), options).unwrap();
    assert!(dir.join("sharded/model.safetensors.index.json").exists());
    let zeros: StateDict = state_dict
        .iter()
        .map(|(k, t)| (k, homura::zeros_like(t)))
        .collect();
    let (single, shards) = (
        ShardOptions::default(),
        ShardOptions::default().max_shard_size(40),
    );
    io::save_sharded(&zeros, dir.join("to_sharded"), single.clone()).unwrap();
    io::save_sharded(&state_dict, dir.join("to_sharded"), shards.clone()).unwrap();
    io::save_sharded(&zeros, dir.join("to_single"), shards).unwrap();
    io::save_sharded(&state_dict, dir.join("to_single"), single).unwrap();
}
