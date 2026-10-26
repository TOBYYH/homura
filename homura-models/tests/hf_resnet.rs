//! The M3 acceptance (step 4b): microsoft/resnet-18 and microsoft/resnet-50, loaded with
//! `from_pretrained` from a Hugging Face cache, give transformers' logits
//! (tests/golden/hf_resnet_logits.safetensors, written by tools/golden/gen_m3_hf_resnet.py with
//! transformers 5.20.2).
//!
//! The repositories (about 300 MB) aren't part of the tests: tools/golden/fetch_hf_repos.py
//! downloads them at pinned commits into target/hf-cache, and the test, ignored by default, runs in
//! release builds (it returns at once in debug builds, where a forward pass takes minutes):
//!
//! ```text
//! HF_HUB_CACHE=$PWD/target/hf-cache cargo test --release -p homura-models --test hf_resnet -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use homura::nn::StateDict;
use homura::{DType, Tensor};
use homura_models::hub::{Repo, Revision, WeightFormat};
use homura_models::pretrained::{FromPretrained, LoadedModel, PretrainedOptions};
use homura_models::resnet::ResNetForImageClassification;

const MODELS: [&str; 2] = ["microsoft/resnet-18", "microsoft/resnet-50"];

fn golden(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(path)
}

/// FNV-1a, 64 bits.
fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The input of gen_m3_hf_resnet.py, `((i·7919 + 13) mod 1021) / 1024 − 0.5`, in F64 (exact in
/// F32 too).
fn pixel_values() -> Tensor {
    let values: Vec<f64> = (0..2 * 3 * 224 * 224_i64)
        .map(|i| ((i * 7919 + 13) % 1021) as f64 / 1024.0 - 0.5)
        .collect();
    Tensor::from_vec(values, [2, 3, 224, 224])
}

/// Checks the bytes of `x` (little-endian) against the digest and length that the generator
/// recorded for its input in this dtype.
fn check_input(x: &Tensor, metadata: &BTreeMap<String, String>, name: &str) {
    let bytes: Vec<u8> = match x.dtype() {
        DType::F32 => x
            .to_vec::<f32>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::F64 => x
            .to_vec::<f64>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        dtype => panic!("no input in {dtype:?}"),
    };
    assert_eq!(
        bytes.len().to_string(),
        metadata[&format!("input.{name}.bytes")]
    );
    assert_eq!(
        format!("{:016x}", fnv1a64(bytes)),
        metadata[&format!("input.{name}.fnv1a64")],
        "the {name} input isn't the generator's"
    );
}

/// The indices of the 5 largest values of each row of a `[n, classes]` tensor, largest first.
fn top5(logits: &Tensor) -> Vec<Vec<usize>> {
    let [rows, classes] = logits.shape()[..] else {
        panic!("logits of shape {:?}", logits.shape());
    };
    let values = logits.to(DType::F64).to_vec::<f64>();
    (0..rows)
        .map(|r| {
            let row = &values[r * classes..(r + 1) * classes];
            let mut indices: Vec<usize> = (0..classes).collect();
            indices.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            indices.truncate(5);
            indices
        })
        .collect()
}

/// The largest `|found - expected| / (1 + |expected|)`, or an error if a value of either isn't
/// finite (a NaN would otherwise drop out of the maximum, and a negative NaN out of the top 5).
fn error(found: &Tensor, expected: &Tensor) -> Result<f64, String> {
    assert_eq!(found.shape(), expected.shape());
    assert_eq!(found.dtype(), expected.dtype());
    let found = found.to(DType::F64).to_vec::<f64>();
    let expected = expected.to(DType::F64).to_vec::<f64>();
    for (what, values) in [("found", &found), ("expected", &expected)] {
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(format!("non-finite {what} logit {} at {i}", values[i]));
        }
    }
    Ok(found
        .iter()
        .zip(&expected)
        .map(|(f, e)| (f - e).abs() / (1.0 + e.abs()))
        .fold(0.0, f64::max))
}

fn assert_bits(what: &str, a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    let bits = |t: &Tensor| -> Vec<u64> {
        t.to(DType::F64)
            .to_vec::<f64>()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    assert!(bits(a) == bits(b), "{what}: the logits differ");
}

/// The largest errors measured on x86-64 (`--nocapture` prints them) are 2.9e-6 in F32 and
/// 4.6e-15 in F64; the reference's own F32 logits are 3.3e-6 from its F64 logits.
const F32_TOLERANCE: f64 = 1e-5;
const F64_TOLERANCE: f64 = 1e-13;

/// How the model is loaded.
#[derive(Clone, Copy, Debug)]
enum Reading {
    Safetensors,
    PyTorch,
    /// `model.safetensors` mapped by `from_pretrained_mmap`.
    Mapped,
}

fn load(
    repo: &Repo,
    path: Reading,
    dtype: Option<DType>,
) -> LoadedModel<ResNetForImageClassification> {
    let options = match dtype {
        Some(dtype) => PretrainedOptions::default().dtype(dtype),
        None => PretrainedOptions::default(),
    };
    match path {
        Reading::Safetensors => ResNetForImageClassification::from_pretrained_with_info(
            repo,
            options.format(WeightFormat::Safetensors),
        ),
        Reading::PyTorch => ResNetForImageClassification::from_pretrained_with_info(
            repo,
            options.format(WeightFormat::PyTorch),
        ),
        // SAFETY: the mapped weight file is a read-only blob of the cache that
        // fetch_hf_repos.py wrote, and nothing modifies the cache while the tests run.
        Reading::Mapped => unsafe {
            ResNetForImageClassification::from_pretrained_mmap(repo, options)
        },
    }
    .unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

#[test]
#[ignore = "needs the repositories of tools/golden/fetch_hf_repos.py and a release build: \
            HF_HUB_CACHE=$PWD/target/hf-cache cargo test --release -p homura-models \
            --test hf_resnet -- --ignored"]
fn microsoft_resnets_give_the_logits_of_transformers() {
    if cfg!(debug_assertions) {
        eprintln!("skipped: the acceptance of microsoft/resnet-18 and -50 runs in release builds");
        return;
    }
    assert!(
        std::env::var_os("HF_HUB_CACHE").is_some(),
        "set HF_HUB_CACHE to the cache of tools/golden/fetch_hf_repos.py (target/hf-cache)"
    );
    let file = golden("hf_resnet_logits.safetensors");
    let expected: StateDict = homura::load(&file).unwrap();
    let metadata = homura::io::safetensors_metadata(&file).unwrap();
    let x64 = pixel_values();
    let x32 = x64.to(DType::F32);
    check_input(&x32, &metadata, "f32");
    check_input(&x64, &metadata, "f64");

    for repo_id in MODELS {
        let name = repo_id.split_once('/').unwrap().1;
        let repo = Repo::cached(repo_id, Revision::main()).unwrap();
        assert_eq!(
            repo.commit(),
            Some(metadata[&format!("{name}.commit")].as_str())
        );
        for (dtype, x, label, tolerance) in [
            (None, &x32, "f32", F32_TOLERANCE),
            (Some(DType::F64), &x64, "f64", F64_TOLERANCE),
        ] {
            let want = expected.get(&format!("{name}.{label}.logits")).unwrap();
            let mut first: Option<Tensor> = None;
            for path in [Reading::Safetensors, Reading::PyTorch, Reading::Mapped] {
                let start = Instant::now();
                let LoadedModel {
                    model,
                    config,
                    report,
                    ..
                } = load(&repo, path, dtype);
                let loaded = start.elapsed();
                let bin = matches!(path, Reading::PyTorch);
                assert_eq!(
                    report.weights.path.file_name().unwrap(),
                    if bin {
                        "pytorch_model.bin"
                    } else {
                        "model.safetensors"
                    }
                );
                assert!(!report.weights.sharded);
                assert_eq!(
                    (report.prefix_removed, report.prefix_added),
                    (0, 0),
                    "{name}"
                );
                assert!(report.unused_keys.is_empty(), "{:?}", report.unused_keys);
                assert!(
                    report.config_warnings.is_empty(),
                    "{:?}",
                    report.config_warnings
                );
                assert_eq!(report.converted, dtype.is_some());
                assert_eq!(config.common.dtype, Some(dtype.unwrap_or(DType::F32)));
                assert_eq!(config.common.num_labels(), 1000);
                let start = Instant::now();
                let logits = homura::with_no_grad(|| model.forward(x));
                let forward = start.elapsed();
                let e =
                    error(&logits, want).unwrap_or_else(|e| panic!("{name} {label} {path:?}: {e}"));
                println!(
                    "{name} {label} {path:?}: error {e:.2e}, load {:.2} s, forward {:.2} s",
                    loaded.as_secs_f64(),
                    forward.as_secs_f64()
                );
                assert!(
                    e <= tolerance,
                    "{name} {label} {path:?}: error {e:e} above {tolerance:e}"
                );
                assert_eq!(top5(&logits), top5(want), "{name} {label} {path:?}");
                match &first {
                    None => first = Some(logits),
                    Some(first) => assert_bits(&format!("{name} {label} {path:?}"), &logits, first),
                }
            }
        }
    }
}

/// The comparison of the acceptance refuses non-finite logits wherever they are, also outside the
/// top 5 and as negative NaN, which `total_cmp` sorts below every finite value. It needs no
/// repository, so it runs in every build.
#[test]
fn logit_comparisons_refuse_non_finite_values() {
    let expected: Vec<f32> = (0..2 * 10).map(|i| (i % 10) as f32 - 4.5).collect();
    let expected = Tensor::from_vec(expected, [2, 10]);
    let with = |t: &Tensor, i: usize, v: f32| {
        let mut values = t.to_vec::<f32>();
        values[i] = v;
        Tensor::from_vec(values, [2, 10])
    };
    // Controls: equal logits, and a finite error above the tolerance that is reported as such.
    assert_eq!(error(&expected, &expected), Ok(0.0));
    let off = error(&with(&expected, 3, 0.0), &expected).unwrap();
    assert!(off > F32_TOLERANCE, "{off}");
    // The smallest logit of the first image (index 0) is outside its top 5.
    assert!(!top5(&expected)[0].contains(&0));
    let negative_nan = f32::from_bits(f32::NAN.to_bits() | 0x8000_0000);
    assert!(negative_nan.is_nan() && negative_nan.is_sign_negative());
    for (name, v) in [
        ("NaN", f32::NAN),
        ("negative NaN", negative_nan),
        ("infinity", f32::INFINITY),
        ("negative infinity", f32::NEG_INFINITY),
    ] {
        let bad = with(&expected, 0, v);
        if v.is_sign_negative() {
            // Below every finite value: the top 5 alone wouldn't notice.
            assert_eq!(top5(&bad), top5(&expected), "{name} changes the top 5");
        }
        let found = error(&bad, &expected);
        assert!(
            found
                .as_ref()
                .is_err_and(|e| e.starts_with("non-finite found")),
            "{name}: {found:?}"
        );
        let found = error(&expected, &bad);
        assert!(
            found
                .as_ref()
                .is_err_and(|e| e.starts_with("non-finite expected")),
            "{name}: {found:?}"
        );
        let found = error(&bad.to(DType::F64), &expected.to(DType::F64));
        assert!(found.is_err(), "{name} in F64: {found:?}");
    }
}
