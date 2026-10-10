//! Correct-behaviour regressions discovered during the M0 review.
//!
//! These are explicitly ignored because this review does not change implementations.
//! Run `cargo test --test m0_regressions -- --ignored --test-threads=1` to reproduce.
//! Also run in release mode to exercise unchecked arithmetic. Unsafe layouts are tested only
//! on Meta, and invalid BOOL data is never read as Rust bool.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::{DType, Device};

fn rejects(f: impl FnOnce()) {
    assert!(
        catch_unwind(AssertUnwindSafe(f)).is_err(),
        "invalid input was accepted"
    );
}

fn file(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
#[ignore = "M0: BOOL file bytes are copied without validating Rust bool representation"]
fn invalid_bool_representation_is_rejected_before_typed_reads() {
    let bytes = file(
        r#"{"x":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}}"#,
        &[2],
    );
    // Do not call to_vec, Debug, comparisons or other typed reads on this input.
    assert!(homura::load_safetensors_from_bytes(&bytes).is_err());
}

#[test]
#[ignore = "M0: as_strided bounds arithmetic wraps in release builds; test only on Meta"]
fn as_strided_rejects_overflowing_span() {
    let x = homura::zeros_with([1], (DType::U8, Device::Meta));
    rejects(|| {
        x.as_strided([2], [usize::MAX], 0);
    });
}

#[test]
#[ignore = "M0: Tensor::empty_on byte count wraps in release builds; test only on Meta"]
fn allocation_rejects_overflowing_byte_count() {
    rejects(|| {
        homura::zeros_with([usize::MAX / 8 + 1], (DType::F64, Device::Meta));
    });
}

#[test]
#[ignore = "M0: writable check detects stride zero but misses nonzero overlapping strides"]
fn as_strided_nonzero_internal_overlap_is_rejected() {
    let x = homura::zeros_with([3], DType::I64);
    let overlapping = x.as_strided([2, 2], [1, 1], 0);
    rejects(|| {
        overlapping.copy_(&homura::tensor([[1, 2], [3, 4]]));
    });
}

#[test]
#[ignore = "M0: repeat with all repetitions equal to one returns a view"]
fn repeat_always_copies_storage() {
    let x = homura::arange(6).view([2, 3]);
    x.repeat([1, 1]).fill_(99);
    assert_eq!(x.to_vec::<i64>(), [0, 1, 2, 3, 4, 5]);
}

#[test]
#[ignore = "M0: empty chunk returns one view rather than the requested number"]
fn chunk_empty_dimension_returns_requested_chunks() {
    let pieces = homura::zeros([0, 3]).chunk(4, 0);
    assert_eq!(pieces.len(), 4);
    assert!(pieces.iter().all(|t| t.shape() == [0, 3]));
}

#[test]
#[ignore = "M0: arange converts integer bounds to f64 before computing the length"]
fn arange_preserves_i64_bounds_above_two_to_the_53() {
    let start = 1i64 << 53;
    assert_eq!(
        homura::arange_step(start, start + 3, 1).to_vec::<i64>(),
        [start, start + 1, start + 2]
    );
}

#[test]
#[ignore = "M0: randint routes i64 samples through f64 and loses integer precision"]
fn randint_single_value_interval_preserves_large_integer() {
    let low = (1i64 << 53) + 1;
    assert_eq!(homura::randint(low, low + 1, [8]).to_vec::<i64>(), [low; 8]);
}

#[test]
#[ignore = "M0: high-low overflows i64 for valid wide randint intervals"]
fn randint_accepts_wide_i64_interval() {
    let values = homura::randint(i64::MIN, i64::MAX, [16]).to_vec::<i64>();
    assert!(values.iter().all(|&v| v < i64::MAX));
}

#[test]
#[ignore = "M0: half uniform samples can round up to the excluded upper endpoint"]
fn half_rand_keeps_the_upper_bound_exclusive() {
    homura::manual_seed(42);
    for dtype in [DType::F16, DType::BF16] {
        let values = homura::rand_with([100_000], dtype).to_vec::<f32>();
        assert!(
            values.iter().all(|&v| (0.0..1.0).contains(&v)),
            "{dtype} produced a value outside [0, 1)"
        );
    }
}

#[test]
#[ignore = "M0: normal_ accepts a negative standard deviation"]
fn normal_rejects_negative_standard_deviation() {
    rejects(|| {
        homura::zeros([2]).normal_(0.0, -1.0);
    });
}

#[test]
#[ignore = "M0: uniform_ accepts a reversed interval"]
fn uniform_rejects_reversed_bounds() {
    rejects(|| {
        homura::zeros([2]).uniform_(2.0, 1.0);
    });
}

#[test]
#[ignore = "M0: Meta random_tensor returns before dtype and interval validation"]
fn meta_random_validates_parameters() {
    rejects(|| {
        homura::rand_with([2], (DType::I64, Device::Meta));
    });
    rejects(|| {
        homura::randint_with(4, 4, [2], Device::Meta);
    });
}

#[test]
#[ignore = "M0: empty variance with a negative correction returns zero instead of NaN"]
fn empty_variance_is_nan_even_with_negative_correction() {
    assert!(homura::zeros([0]).var(-1).item::<f32>().is_nan());
    assert!(homura::zeros([0]).std(-1).item::<f32>().is_nan());
}

#[test]
#[ignore = "M0: addbmm allows batch broadcasting although PyTorch requires equal batches"]
fn addbmm_rejects_unequal_batch_counts() {
    rejects(|| {
        homura::zeros([2, 4]).addbmm(&homura::ones([1, 2, 3]), &homura::ones([2, 3, 4]), 1, 1);
    });
}

#[test]
#[ignore = "M0: addbmm uses generic integer sum, incorrectly promoting its result to I64"]
fn addbmm_preserves_integer_dtype() {
    let input = homura::ones_with([2, 4], DType::I32);
    let a = homura::ones_with([1, 2, 3], DType::I32);
    let b = homura::ones_with([1, 3, 4], DType::I32);
    let result = input.addbmm(&a, &b, 1, 1);
    assert_eq!(result.dtype(), DType::I32);
    assert_eq!(result.to_vec::<i32>(), [4; 8]);
}

#[test]
#[ignore = "M0: scaled_sum skips input device/dtype checks when beta is zero"]
fn addmm_beta_zero_still_validates_input_device_and_dtype() {
    let a = homura::ones([2, 3]);
    let b = homura::ones([3, 4]);
    rejects(|| {
        homura::zeros_with([2, 4], Device::Meta).addmm(&a, &b, 0, 1);
    });
    rejects(|| {
        homura::zeros_with([2, 4], DType::I64).addmm(&a, &b, 0, 1);
    });
}

#[test]
#[ignore = "M0: Rust saturating float casts disagree with PyTorch narrow integer casts"]
fn finite_float_to_u8_cast_matches_pytorch() {
    assert_eq!(
        homura::tensor([300.0, -1.0]).to(DType::U8).to_vec::<u8>(),
        [44, 255]
    );
}

#[test]
#[ignore = "M0: scalar flatten returns before dimension validation"]
fn scalar_flatten_checks_dimensions() {
    rejects(|| {
        homura::tensor(1.0).flatten(10, 10);
    });
}

#[test]
#[ignore = "M0: full optimizes all zero values to zeroed memory, losing the sign of -0.0"]
fn full_preserves_negative_zero() {
    assert!(homura::full([1], -0.0).item::<f32>().is_sign_negative());
}

#[test]
#[ignore = "M0: safetensors does not validate global data coverage, duplicate names or metadata"]
fn safetensors_rejects_invalid_global_structure() {
    let invalid = [
        (
            "overlap",
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            vec![0],
        ),
        (
            "hole",
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
            vec![0, 0],
        ),
        (
            "trailing payload",
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            vec![0, 0],
        ),
        (
            "duplicate name",
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
            vec![0, 0],
        ),
        (
            "invalid metadata",
            r#"{"__metadata__":{"version":1}}"#,
            vec![],
        ),
    ];
    let accepted: Vec<_> = invalid
        .iter()
        .filter(|(_, header, payload)| {
            homura::load_safetensors_from_bytes(&file(header, payload)).is_ok()
        })
        .map(|(name, _, _)| *name)
        .collect();
    assert!(
        accepted.is_empty(),
        "accepted invalid structures: {accepted:?}"
    );
}

#[test]
#[ignore = "M0: JSON numbers are parsed with f64::from_str, accepting leading zeros"]
fn safetensors_json_numbers_follow_json_grammar() {
    let header = r#"{"a":{"dtype":"U8","shape":[01],"data_offsets":[0,1]}}"#;
    assert!(homura::load_safetensors_from_bytes(&file(header, &[0])).is_err());
}

#[test]
#[ignore = "M0: save accepts reserved __metadata__ as a tensor name, then load discards it"]
fn safetensors_save_rejects_reserved_metadata_name() {
    assert!(homura::save_safetensors_to_bytes([("__metadata__", &homura::tensor(1))]).is_err());
}

#[test]
#[ignore = "M0: concurrent add_ has separate read/compute/write critical sections"]
fn concurrent_increments_do_not_lose_updates() {
    let x = homura::zeros_with([128], DType::I64);
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let (x, barrier) = (&x, &barrier);
            scope.spawn(move || {
                for _ in 0..100 {
                    barrier.wait();
                    x.add_(1);
                }
            });
        }
    });
    assert_eq!(x.to_vec::<i64>(), [800; 128]);
}

#[test]
#[ignore = "M0: golden comparisons cast I64 to F64, masking differences above 2^53"]
fn golden_comparison_detects_adjacent_large_i64_values() {
    let expected = homura::tensor([1i64 << 53]);
    let actual = homura::tensor([(1i64 << 53) + 1]);
    rejects(|| {
        common::assert_close(&actual, &expected, "exact i64 comparison");
    });
}
