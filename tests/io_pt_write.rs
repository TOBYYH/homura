//! Writing PyTorch `.pt` files (M3 step 5) through the public entry points: `homura::save`,
//! `io::save_value` and `io::save_value_to_bytes`, read back with `homura::load`,
//! `io::load_mmap` and `io::load_value`. The unit tests of `src/io` check the bytes against
//! Python's pickler and the archive layout; `homura_files_for_python` writes the files that
//! `tools/golden/check_m3_pt_written.py` reads with PyTorch.

use std::fs;
use std::path::{Path, PathBuf};

use homura::io::{Dict, Key, Value, load_mmap, load_value, load_value_from_bytes};
use homura::io::{save_value, save_value_to_bytes};
use homura::nn::{
    BatchNorm2d, Conv2d, Linear, Module, ModuleVersions, Relu, Sequential, StateDict,
};
use homura::{DType, Device, Error, Tensor};

/// A fresh, empty directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("io_pt_write")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn is_format_error<T: std::fmt::Debug>(result: homura::Result<T>, message: &str) -> bool {
    match result {
        Err(Error::Format(m)) => m.contains(message) || panic!("{m:?} lacks {message:?}"),
        other => panic!("expected a format error, got {other:?}"),
    }
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

/// Asserts that `a` has the dtype, shape, layout, `requires_grad` and values of `b`.
#[track_caller]
fn assert_same(a: &Tensor, b: &Tensor) {
    assert_eq!((a.dtype(), a.shape()), (b.dtype(), b.shape()));
    assert_eq!(
        (a.stride(), a.storage_offset()),
        (b.stride(), b.storage_offset())
    );
    assert_eq!(a.requires_grad(), b.requires_grad());
    let bits = |t: &Tensor| homura::save_safetensors_to_bytes([("t", &t.detach())]).unwrap();
    assert_eq!(bits(a), bits(b), "{a:?} != {b:?}");
}

/// One tensor of each dtype named after it, then views of one storage, a zero-dimensional tensor,
/// an empty one and one that requires grad.
fn layouts() -> StateDict {
    let mut sd: StateDict = DTYPES
        .iter()
        .map(|&(name, d)| (name, values(&[2, 3], d)))
        .collect();
    let base = values(&[3, 4], DType::F32);
    sd.insert("base", base.clone());
    sd.insert("transposed", base.transpose(0, 1));
    sd.insert("slice", base.index((1.., 2..)));
    sd.insert("expanded", base.index((0, ..1)).expand([3, 4]));
    sd.insert("scalar", values(&[], DType::F64));
    sd.insert("empty", values(&[0, 2], DType::I16));
    sd.insert(
        "grad",
        values(&[2], DType::F32).requires_grad_(true).clone(),
    );
    sd
}

#[test]
fn state_dicts_round_trip_with_every_extension() {
    let dir = scratch("extensions");
    let sd = layouts();
    let model = Sequential::new()
        .append(Conv2d::new(1, 2, 1))
        .append(BatchNorm2d::new(2));
    let with_versions = model.state_dict();
    for name in ["model.pt", "model.pth", "model.bin", "model.PT"] {
        let path = dir.join(name);
        homura::save(&sd, &path).unwrap();
        // SAFETY: nothing else uses the file, which is not changed while the tensors live.
        let mapped = unsafe { load_mmap(&path) }.unwrap();
        for back in [homura::load(&path).unwrap(), mapped] {
            assert_eq!(
                back.keys().collect::<Vec<_>>(),
                sd.keys().collect::<Vec<_>>()
            );
            for ((key, a), (_, b)) in back.iter().zip(&sd) {
                assert_same(a, b);
                assert_eq!(a.requires_grad(), key == "grad", "{key}");
            }
            assert_eq!(back.versions(), None);
            // The views share one storage again: writing the base changes them.
            back["base"].index((1, 2)).fill_(100.0);
            assert_eq!(back["transposed"].index((2, 1)).item::<f32>(), 100.0);
            assert_eq!(back["slice"].index((0, 0)).item::<f32>(), 100.0);
            back["base"].index((0, 0)).fill_(-7.0);
            assert_eq!(back["expanded"].index((2, 3)).item::<f32>(), -7.0);
            assert_eq!(
                back["f32"].sum().item::<f32>(),
                sd["f32"].sum().item::<f32>()
            );
        }
        homura::save(&with_versions, &path).unwrap();
        let back = homura::load(&path).unwrap();
        let expected: ModuleVersions = [("", 1), ("0", 1), ("1", 2)].into_iter().collect();
        assert_eq!(back.versions(), Some(&expected));
        let fresh = Sequential::new()
            .append(Conv2d::new(1, 2, 1))
            .append(BatchNorm2d::new(2));
        fresh.load_state_dict(&back).unwrap();
        for ((_, a), (_, b)) in fresh.state_dict().iter().zip(&with_versions) {
            assert_same(a, b);
        }
    }
}

/// Every scalar and container that `Value` has, in the shapes of a training checkpoint.
fn checkpoint() -> Value {
    let model = Sequential::new()
        .append(Linear::new(3, 4))
        .append(Relu)
        .append(BatchNorm2d::new(4));
    let shared = values(&[2], DType::F64);
    let mut state: Dict = Dict::new();
    state.insert(
        0,
        [("step", values(&[], DType::F32))]
            .into_iter()
            .collect::<Dict>(),
    );
    state.insert(1, Value::None);
    let ints = [
        0,
        1,
        255,
        256,
        65535,
        65536,
        -1,
        -(1 << 31),
        (1 << 31) - 1,
        1 << 31,
        i64::MAX,
        i64::MIN,
    ];
    let mut ordered = Dict::new_ordered();
    ordered.insert("z", 1);
    ordered.insert("a", 2);
    [
        ("model", Value::from(&model.state_dict())),
        ("state", Value::from(state)),
        ("ints", Value::list(ints)),
        (
            "floats",
            Value::tuple([0.5, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
        ),
        ("strings", Value::list(["", "名前", "a\nb"])),
        (
            "flags",
            Value::tuple([Value::from(true), false.into(), Value::None]),
        ),
        ("ordered", Value::from(ordered)),
        ("list_1000", Value::list(0..1000)),
        ("list_1001", Value::list(0..1001)),
        (
            "dict_1001",
            Value::from((0..1001).map(|i| (i, i * 2)).collect::<Dict>()),
        ),
        ("same", Value::list([&shared, &shared])),
        (
            "nested",
            Value::tuple([Value::tuple([Value::list::<Value>([])])]),
        ),
    ]
    .into_iter()
    .collect::<Dict>()
    .into()
}

/// Compares two values: equal structure and scalars (floats bitwise), and tensors as
/// [`assert_same`].
#[track_caller]
fn assert_same_value(a: &Value, b: &Value) {
    match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => assert_same(a, b),
        (Value::Float(a), Value::Float(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        (Value::List(a), Value::List(b)) | (Value::Tuple(a), Value::Tuple(b)) => {
            assert_eq!(a.len(), b.len());
            a.iter().zip(b).for_each(|(a, b)| assert_same_value(a, b));
        }
        (Value::Dict(a), Value::Dict(b)) => {
            assert_eq!(a.is_ordered(), b.is_ordered());
            assert_eq!(a.versions(), b.versions());
            let keys = |d: &Dict| d.iter().map(|(k, _)| k.clone()).collect::<Vec<Key>>();
            assert_eq!(keys(a), keys(b));
            a.iter()
                .zip(b.iter())
                .for_each(|((_, a), (_, b))| assert_same_value(a, b));
        }
        (a, b) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
    }
}

#[test]
fn checkpoints_round_trip() {
    let dir = scratch("checkpoints");
    let v = checkpoint();
    let path = dir.join("checkpoint.pt");
    save_value(&v, &path).unwrap();
    let bytes = save_value_to_bytes(&v).unwrap();
    // Equal values give equal bytes, in a file or in memory.
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(save_value_to_bytes(&v).unwrap(), bytes);
    for back in [
        load_value(&path).unwrap(),
        load_value_from_bytes(&bytes).unwrap(),
    ] {
        assert_same_value(&back, &v);
        let Value::Dict(back) = back else {
            unreachable!()
        };
        // The same tensor in two places is one tensor again, as in Python.
        let Some(Value::List(same)) = back.get("same") else {
            unreachable!()
        };
        let [Value::Tensor(a), Value::Tensor(b)] = &same[..] else {
            unreachable!()
        };
        assert!(a.is_same(b));
        // The model's state dict loads strictly, with its versions.
        let sd = StateDict::try_from(back.get("model").unwrap()).unwrap();
        assert_eq!(sd.module_version("2"), Some(2));
        Sequential::new()
            .append(Linear::new(3, 4))
            .append(Relu)
            .append(BatchNorm2d::new(4))
            .load_state_dict(&sd)
            .unwrap();
    }
}

#[test]
fn failed_checks_leave_no_file() {
    let dir = scratch("checks");
    let meta: StateDict = [
        ("ok", homura::ones([1])),
        ("m", homura::zeros_with([2], Device::Meta)),
    ]
    .into_iter()
    .collect();
    let mut deep = Value::list::<Value>([]);
    for _ in 0..256 {
        deep = Value::list([deep]);
    }
    for name in ["model.pt", "model.bin"] {
        let path = dir.join(name);
        assert!(is_format_error(
            homura::save(&meta, &path),
            r#"can't save a meta tensor: it has no data (at ["m"])"#
        ));
        assert!(!path.exists());
        assert!(is_format_error(
            save_value(&deep, &path),
            "nested more than 256 levels deep"
        ));
        assert!(!path.exists());
        // An existing file is left as it was.
        fs::write(&path, b"old").unwrap();
        assert!(is_format_error(homura::save(&meta, &path), "meta tensor"));
        assert_eq!(fs::read(&path).unwrap(), b"old");
    }
    assert!(is_format_error(save_value_to_bytes(&deep), "levels deep"));
    // 256 levels are written and read back.
    let Value::List(items) = deep else {
        unreachable!()
    };
    let deep = items.into_iter().next().unwrap();
    load_value_from_bytes(&save_value_to_bytes(&deep).unwrap()).unwrap();
}

/// Writes `target/tmp/m3_pt_written/` for `tools/golden/check_m3_pt_written.py`:
///
/// - `layouts.pt`: [`layouts`], with `homura::save`;
/// - `checkpoint.pt`: [`checkpoint`], with `io::save_value`; the model's tensors are zeros and
///   ones but for the linear layer, whose weight and bias are the formula's values;
/// - `model.pt`: the state dict of `Sequential(Conv2d(1, 2, 1), BatchNorm2d(2), Linear(1, 2))`
///   with the formula's values in every parameter and buffer of floats, `num_batches_tracked`
///   5, and its versions.
#[test]
fn homura_files_for_python() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("m3_pt_written");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    homura::save(&layouts(), dir.join("layouts.pt")).unwrap();
    let mut v = checkpoint();
    let Value::Dict(d) = &mut v else {
        unreachable!()
    };
    let mut model = StateDict::try_from(d.get("model").unwrap()).unwrap();
    model.insert("0.weight", values(&[4, 3], DType::F32));
    model.insert("0.bias", values(&[4], DType::F32));
    d.insert("model", &model);
    save_value(&v, dir.join("checkpoint.pt")).unwrap();
    let model = Sequential::new()
        .append(Conv2d::new(1, 2, 1))
        .append(BatchNorm2d::new(2))
        .append(Linear::new(1, 2));
    let sd: StateDict = model
        .state_dict()
        .iter()
        .map(|(k, t)| {
            let t = if t.dtype().is_floating_point() {
                values(&t.shape(), t.dtype())
            } else {
                homura::full_like(t, 5)
            };
            (k.to_string(), t)
        })
        .collect();
    model.load_state_dict(&sd).unwrap();
    homura::save(&model.state_dict(), dir.join("model.pt")).unwrap();
}
