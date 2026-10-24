//! M0 boundary coverage independent of the committed PyTorch golden fixtures.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::{DType, Device, Ellipsis, Error, NewAxis, Shape, Strides, Tensor, TensorIndex};

fn panics(f: impl FnOnce()) {
    assert!(catch_unwind(AssertUnwindSafe(f)).is_err());
}

#[test]
fn dtype_metadata_and_promotion_matrix() {
    let sizes = [1, 1, 1, 2, 4, 8, 2, 2, 4, 8];
    for (d, size) in DType::ALL.into_iter().zip(sizes) {
        assert_eq!(d.size_in_bytes(), size);
        assert_eq!(d.to_string(), d.name());
        for e in DType::ALL {
            assert_eq!(homura::promote_types(d, e), homura::promote_types(e, d));
            assert!(homura::can_cast(d, homura::promote_types(d, e)));
        }
    }
    assert_eq!(homura::promote_types(DType::U8, DType::I8), DType::I16);
    assert_eq!(homura::promote_types(DType::F16, DType::BF16), DType::F32);
    assert!(!DType::Bool.is_integral());
    assert!(!DType::U8.is_signed());
    assert!(DType::I8.is_signed());
}

#[test]
fn zero_dim_and_scalar_promotion() {
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let x = homura::ones_with([2], dtype);
        assert_eq!(x.add(homura::full_with((), 2.0, DType::F64)).dtype(), dtype);
        assert_eq!(x.mul(2.0).dtype(), dtype);
    }
    let i = homura::ones_with([2], DType::I32);
    assert_eq!(
        i.add(homura::full_with((), 2.0, DType::F64)).dtype(),
        DType::F64
    );
    assert_eq!(i.div(2).dtype(), DType::F32);
    assert_eq!(homura::tensor([true]).add(1).dtype(), DType::I64);
}

#[test]
fn shape_and_strides_inline_and_heap_values() {
    for rank in [0, 1, 8, 9, 16] {
        let values = vec![2; rank];
        let shape = Shape::new(&values);
        assert_eq!(shape.num_dims(), rank);
        assert_eq!(shape.num_elements(), 1usize << rank);
        assert_eq!(shape.clone().to_vec(), values);
        assert_eq!(Strides::new(&values).as_slice(), values);
    }
    assert_eq!(Shape::scalar().num_elements(), 1);
    let t = homura::arange(8).view([1, 1, 1, 1, 1, 1, 1, 2, 4]);
    assert_eq!(
        t.permute([8, 7, 6, 5, 4, 3, 2, 1, 0]).sum().item::<i64>(),
        28
    );
}

#[test]
fn device_and_error_interfaces() {
    for device in [Device::Cpu, Device::Meta, Device::Rocm(u32::MAX)] {
        assert_eq!(device.name().parse::<Device>().unwrap(), device);
    }
    for bad in ["", "CPU", "cpu:0", "meta:0", "rocm:", "rocm:4294967296"] {
        let error = bad.parse::<Device>().unwrap_err();
        assert!(matches!(error, Error::Parse(_)));
        assert!(error.to_string().starts_with("parse error:"));
    }
    let io = Error::from(std::io::Error::from(std::io::ErrorKind::NotFound));
    assert!(std::error::Error::source(&io).is_some());
    assert!(std::error::Error::source(&Error::Format("bad".into())).is_none());
    Device::Cpu.synchronize();
    Device::Meta.synchronize();
}

#[test]
fn guards_restore_after_unwind_and_are_thread_local() {
    let initial = homura::default_device();
    panics(|| {
        let _outer = homura::device_scope(Device::Meta);
        let _grad = homura::no_grad();
        assert_eq!(homura::default_device(), Device::Meta);
        assert!(!homura::is_grad_enabled());
        {
            let _inner = homura::device_scope(Device::Cpu);
            let _enabled = homura::enable_grad();
            assert_eq!(homura::default_device(), Device::Cpu);
            assert!(homura::is_grad_enabled());
        }
        let child = std::thread::spawn(|| (homura::default_device(), homura::is_grad_enabled()));
        assert_eq!(child.join().unwrap(), (initial, true));
        panic!("exercise guard unwinding");
    });
    assert_eq!(homura::default_device(), initial);
    assert!(homura::is_grad_enabled());
}

#[test]
fn empty_creation_views_and_broadcasting() {
    for dtype in DType::ALL {
        let t = homura::zeros_with([2, 0, 3], dtype);
        assert_eq!(t.num_elements(), 0);
        assert!(t.is_contiguous());
        assert_eq!(t.to(dtype).shape(), [2, 0, 3]);
        assert_eq!(t.deep_clone().shape(), [2, 0, 3]);
        assert_eq!(t.transpose(0, 2).reshape([0, 6]).shape(), [0, 6]);
        assert_eq!(t.add(homura::ones_with([1, 3], dtype)).shape(), [2, 0, 3]);
        assert_eq!(t.flip([0, 2]).shape(), [2, 0, 3]);
        assert_eq!(t.repeat([2, 3, 1]).shape(), [4, 0, 3]);
    }
    assert_eq!(homura::eye(0).shape(), [0, 0]);
    assert_eq!(homura::linspace(1.0, 2.0, 0).shape(), [0]);
    assert_eq!(homura::arange_step(3, 3, -1).shape(), [0]);
    assert_eq!(homura::zeros([0]).view([-1, 3]).shape(), [0, 3]);
}

#[test]
fn empty_reduction_identities_and_validation() {
    let x = homura::zeros([2, 0, 3]);
    assert_eq!(x.sum_dim(1, false).to_vec::<f32>(), [0.0; 6]);
    assert_eq!(x.prod_dim(1, true).to_vec::<f32>(), [1.0; 6]);
    assert!(
        x.mean_dim(1, false)
            .to_vec::<f32>()
            .iter()
            .all(|v| v.is_nan())
    );
    assert!(
        x.var_dim(1, 0, false)
            .to_vec::<f32>()
            .iter()
            .all(|v| v.is_nan())
    );
    assert!(x.all().item::<bool>());
    assert!(!x.any().item::<bool>());
    panics(|| {
        x.max();
    });
    panics(|| {
        x.min_dim(1, false);
    });
    panics(|| {
        x.argmax();
    });
    panics(|| {
        x.sum_dim([1, -2], false);
    });
    assert_eq!(x.max_dim(0, false).values.shape(), [0, 3]);
    assert_eq!(x.softmax(1).shape(), [2, 0, 3]);
    assert_eq!(x.log_softmax(1).shape(), [2, 0, 3]);
}

#[test]
fn scalar_reductions_views_and_softmax() {
    let t = homura::tensor(3.0);
    for dim in [0, -1] {
        assert_eq!(t.sum_dim(dim, true).shape(), []);
        let (v, i) = t.max_dim(dim, true).into();
        assert_eq!(v.item::<f32>(), 3.0);
        assert_eq!(i.item::<i64>(), 0);
        assert_eq!(t.softmax(dim).item::<f32>(), 1.0);
        assert_eq!(t.log_softmax(dim).item::<f32>(), 0.0);
        assert_eq!(t.transpose(dim, dim).shape(), []);
    }
    assert_eq!(t.unsqueeze(-1).squeeze().shape(), []);
    assert_eq!(t.var(0).item::<f32>(), 0.0);
    assert!(t.var(1).item::<f32>().is_nan());
    assert_eq!(t.flatten(0, -1).shape(), [1]);
}

#[test]
fn noncontiguous_offset_kernels() {
    let source = homura::arange(24).view([2, 3, 4]);
    let x = source.narrow(2, 1, 2).permute([1, 0, 2]);
    assert_eq!(
        x.to_vec::<i64>(),
        [1, 2, 13, 14, 5, 6, 17, 18, 9, 10, 21, 22]
    );
    assert_eq!(x.sum_dim([0, 2], false).to_vec::<i64>(), [33, 105]);
    assert_eq!(
        x.max_dim(2, false).values.to_vec::<i64>(),
        [2, 14, 6, 18, 10, 22]
    );
    assert_eq!(
        x.pow(2).to_vec::<i64>(),
        [1, 4, 169, 196, 25, 36, 289, 324, 81, 100, 441, 484]
    );
    let weights = homura::tensor([2, 3]);
    assert_eq!(x.matmul(&weights).to_vec::<i64>(), [8, 68, 28, 88, 48, 108]);
    assert!(x.to(DType::F16).is_contiguous());
}

#[test]
fn view_strides_and_contiguous_alias_contract() {
    let source = homura::arange(12).view([3, 4]);
    let strided = source.index((.., (0..4).step_by(2)));
    assert_eq!(strided.stride(), [4, 2]);
    assert_eq!(strided.view([6]).stride(), [2]);
    strided.view([6]).fill_(9);
    assert_eq!(
        source.to_vec::<i64>(),
        [9, 1, 9, 3, 9, 5, 9, 7, 9, 9, 9, 11]
    );
    assert!(source.contiguous().is_same(&source));
    let copied = source.transpose(0, 1).contiguous();
    copied.fill_(-1);
    assert_eq!(source.index((0, 0)).item::<i64>(), 9);
    let expanded = homura::tensor([7]).expand([3, 4]);
    assert_eq!(expanded.to_vec::<i64>(), [7; 12]);
    panics(|| {
        expanded.copy_(&homura::ones([3, 4]));
    });
}

#[test]
fn as_strided_offset_bounds_and_independent_copy() {
    let x = homura::arange(10);
    let y = x.as_strided([2, 3], [4, 1], 1);
    assert_eq!(y.to_vec::<i64>(), [1, 2, 3, 5, 6, 7]);
    y.copy_(&homura::tensor([20, 30, 40]));
    assert_eq!(x.to_vec::<i64>(), [0, 20, 30, 40, 4, 20, 30, 40, 8, 9]);
    panics(|| {
        x.as_strided([2, 3], [4, 1], 5);
    });
    panics(|| {
        x.as_strided([2, 3], [1], 0);
    });
}

#[test]
fn overlap_copy_transpose_and_broadcast() {
    let x = homura::arange(9).view([3, 3]);
    x.copy_(&x.transpose(0, 1));
    assert_eq!(x.to_vec::<i64>(), [0, 3, 6, 1, 4, 7, 2, 5, 8]);
    x.copy_(&x.narrow(0, 0, 1));
    assert_eq!(x.to_vec::<i64>(), [0, 3, 6, 0, 3, 6, 0, 3, 6]);
    let y = homura::arange(6);
    y.narrow(0, 0, 5).copy_(&y.narrow(0, 1, 5));
    assert_eq!(y.to_vec::<i64>(), [1, 2, 3, 4, 5, 5]);
}

#[test]
fn rejected_writes_leave_destination_unchanged() {
    let x = homura::tensor([1, 2, 3]);
    panics(|| {
        x.add_(homura::zeros([2, 3]));
    });
    panics(|| {
        x.copy_(&homura::tensor([1, 2]));
    });
    panics(|| {
        x.add_(0.5);
    });
    assert_eq!(x.to_vec::<i64>(), [1, 2, 3]);
    // A panic while holding the execution lock must not disable later kernels.
    panics(|| {
        homura::arange(3).index(homura::tensor([3]));
    });
    assert_eq!(homura::tensor([2, 3]).mul(4).to_vec::<i64>(), [8, 12]);
}

#[test]
fn indexing_scalars_empty_indices_and_mixed_views() {
    let x = homura::arange(24).view([2, 3, 4]);
    assert_eq!(x.index((Ellipsis, NewAxis)).shape(), [2, 3, 4, 1]);
    assert_eq!(x.index((NewAxis, Ellipsis, -1)).shape(), [1, 2, 3]);
    let idx = homura::tensor_with([0i64; 0], DType::I64);
    assert_eq!(x.index((.., &idx)).shape(), [2, 0, 4]);
    assert_eq!(x.index_select(1, &idx).shape(), [2, 0, 4]);
    assert_eq!(
        x.index((.., &homura::tensor(1))).to_vec::<i64>(),
        [4, 5, 6, 7, 16, 17, 18, 19]
    );
    let gather = x.index((.., &homura::tensor([2, 0]), 1..3));
    assert_eq!(gather.to_vec::<i64>(), [9, 10, 1, 2, 21, 22, 13, 14]);
    gather.zero_();
    assert_eq!(x.sum().item::<i64>(), 276);
    assert_eq!(x.index_select(1, &homura::tensor(1)).shape(), [2, 1, 4]);
}

#[test]
fn indexing_validation() {
    let x = homura::arange(4);
    panics(|| {
        x.index((Ellipsis, Ellipsis));
    });
    panics(|| {
        x.index(homura::tensor([true]));
    });
    panics(|| {
        x.index_select(0, &homura::tensor([-1]));
    });
    panics(|| {
        x.index_select(0, &homura::tensor([[0]]));
    });
    panics(|| {
        x.index(TensorIndex::Slice {
            start: None,
            end: None,
            step: 0,
        });
    });
    panics(|| {
        x.index(TensorIndex::Slice {
            start: None,
            end: None,
            step: -1,
        });
    });
    panics(|| {
        x.narrow(0, 3, 2);
    });
    panics(|| {
        x.expand([-1, -1]);
    });
    panics(|| {
        x.view([-1, -1]);
    });
    panics(|| {
        x.permute([0, 0]);
    });
}

#[test]
fn matmul_zero_inner_dimension_and_empty_batches() {
    for dtype in [
        DType::I8,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let a = homura::zeros_with([2, 0], dtype);
        let b = homura::zeros_with([0, 3], dtype);
        let product = a.matmul(&b);
        assert_eq!(product.shape(), [2, 3]);
        assert_eq!(product.to(DType::I64).to_vec::<i64>(), [0; 6]);
        assert_eq!(
            homura::zeros_with([0], dtype)
                .matmul(&homura::zeros_with([0], dtype))
                .to(DType::I64)
                .item::<i64>(),
            0
        );
    }
}

#[test]
fn batched_matmul_broadcast_matches_explicit_loops() {
    let a = homura::arange(12).view([2, 1, 2, 3]);
    let b = homura::arange(24).view([1, 4, 3, 2]);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), [2, 4, 2, 2]);
    for i in 0..2 {
        for j in 0..4 {
            for row in 0..2 {
                for col in 0..2 {
                    let expected: i64 = (0..3)
                        .map(|k| (i * 6 + row * 3 + k) * (j * 6 + k * 2 + col))
                        .sum();
                    assert_eq!(c.index((i, j, row, col)).item::<i64>(), expected);
                }
            }
        }
    }
    assert_eq!(
        homura::zeros([0, 2, 3])
            .matmul(&homura::ones([1, 3, 4]))
            .shape(),
        [0, 2, 4]
    );
}

#[test]
fn boolean_nan_and_first_tie_reductions() {
    let x = homura::tensor([[1.0, f64::NAN, f64::NAN], [4.0, 4.0, 2.0]]);
    assert_eq!(x.argmax_dim(1, false).to_vec::<i64>(), [1, 0]);
    assert_eq!(x.argmin_dim(1, false).to_vec::<i64>(), [1, 2]);
    assert!(x.max().item::<f32>().is_nan());
    let u = homura::tensor([0u8, 2]);
    assert_eq!(u.any().dtype(), DType::U8);
    assert_eq!(u.any().item::<u8>(), 1);
    assert_eq!(u.all().item::<u8>(), 0);
    assert_eq!(homura::tensor([true, false, true]).prod().item::<i64>(), 0);
    assert_eq!(homura::tensor([i64::MAX, 1]).sum().item::<i64>(), i64::MIN);
}

#[test]
fn shape_operators_on_noncontiguous_inputs() {
    let t = homura::arange(6).view([2, 3]).transpose(0, 1);
    let cat = homura::concat(&[&homura::arange(0), &t, &t], -1);
    assert_eq!(cat.to_vec::<i64>(), [0, 3, 0, 3, 1, 4, 1, 4, 2, 5, 2, 5]);
    assert_eq!(homura::stack(&[&t, &t], 0).shape(), [2, 3, 2]);
    assert_eq!(t.triu(1).to_vec::<i64>(), [0, 3, 0, 0, 0, 0]);
    assert_eq!(t.tril(-1).to_vec::<i64>(), [0, 0, 1, 0, 2, 5]);
    let parts = t.split_with_sizes(&[1, 0, 2], 0);
    assert_eq!(
        parts.iter().map(Tensor::shape).collect::<Vec<_>>(),
        vec![
            Shape::from([1, 2]),
            Shape::from([0, 2]),
            Shape::from([2, 2])
        ]
    );
    assert_eq!(homura::concat(&parts, 0).to_vec::<i64>(), t.to_vec::<i64>());
    panics(|| {
        homura::stack(&[&t, &t.transpose(0, 1)], 0);
    });
    panics(|| {
        t.split_with_sizes(&[2, 2], 0);
    });
    panics(|| {
        t.flip([0, -2]);
    });
}

#[test]
fn meta_shape_dtype_parity_for_operator_families() {
    let x = homura::arange(24)
        .to(DType::F32)
        .view([2, 3, 4])
        .transpose(0, 1);
    let meta = x.to(Device::Meta);
    let ops: [fn(&Tensor) -> Tensor; 18] = [
        |t| t.exp(),
        |t| t.logical_not(),
        |t| t.pow(2),
        |t| t.div(2),
        |t| t.greater(1),
        |t| t.logical_and(true),
        |t| t.sum_dim([0, 2], true),
        |t| t.var_dim(-1, 1, false),
        |t| t.argmax_dim(-1, false),
        |t| t.all_dim(0, false),
        |t| t.softmax(-1),
        |t| t.log_softmax(1),
        |t| t.repeat([2, 1, 1]),
        |t| t.flip([0, 2]),
        |t| t.triu(1),
        |t| t.flatten(1, -1),
        |t| homura::concat(&[t, t], 0),
        |t| homura::stack(&[t, t], -1),
    ];
    for op in ops {
        let (cpu, meta) = (op(&x), op(&meta));
        assert_eq!((meta.shape(), meta.dtype()), (cpu.shape(), cpu.dtype()));
        assert_eq!(meta.device(), Device::Meta);
    }
    let idx = homura::tensor([2, 0]);
    assert_eq!(
        meta.index_select(-1, &idx.to(Device::Meta)).shape(),
        x.index_select(-1, &idx).shape()
    );
    panics(|| {
        homura::zeros_with([2], (DType::Bool, Device::Meta)).neg();
    });
    panics(|| {
        meta.sum_dim(4, false);
    });
}

#[test]
fn softmax_stability_and_special_values() {
    let x = homura::tensor([[10000., 10001., 10002.], [f64::NAN, 1.0, 2.0]]);
    let p = x.softmax(-1).to_vec::<f32>();
    assert!((p[0] - 0.09003057).abs() < 1e-6);
    assert!((p[1] - 0.24472848).abs() < 1e-6);
    assert!((p[2] - 0.66524094).abs() < 1e-6);
    assert!(p[3..].iter().all(|v| v.is_nan()));
    let y = homura::tensor([0., f64::NEG_INFINITY]);
    assert_eq!(y.softmax(0).to_vec::<f32>(), [1.0, 0.0]);
    assert_eq!(y.log_softmax(0).to_vec::<f32>(), [0.0, f32::NEG_INFINITY]);
}

#[test]
fn operators_with_owned_borrowed_and_scalar_operands() {
    let x = homura::tensor([2, 4]);
    let y = homura::tensor([1, 2]);
    assert_eq!((x.clone() + y.clone()).to_vec::<i64>(), [3, 6]);
    assert_eq!((x.clone() - &y).to_vec::<i64>(), [1, 2]);
    assert_eq!((&x * y.clone()).to_vec::<i64>(), [2, 8]);
    assert_eq!((8i64 / &x).to_vec::<f32>(), [4.0, 2.0]);
    assert_eq!((8i64 - x.clone()).to_vec::<i64>(), [6, 4]);
    assert_eq!((-x.clone()).to_vec::<i64>(), [-2, -4]);
    assert_eq!(
        homura::where_(&homura::tensor([true, false]), 7, &x).to_vec::<i64>(),
        [7, 4]
    );
}

#[test]
fn safetensors_unicode_empty_scalar_and_disk_roundtrip() {
    let empty = homura::zeros_with([0, 3], DType::BF16);
    let scalar = homura::tensor(i64::MAX);
    let unicode = "weight.焰😀\"\\\n";
    let bytes =
        homura::save_safetensors_to_bytes([(unicode, &empty), ("scalar", &scalar)]).unwrap();
    let loaded = homura::load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded[0].0, unicode);
    assert_eq!(loaded[0].1.shape(), [0, 3]);
    assert_eq!(loaded[1].1.item::<i64>(), i64::MAX);
    let path = std::env::temp_dir().join(format!(
        "homura-m0-review-{}.safetensors",
        std::process::id()
    ));
    homura::save_safetensors([("scalar", &scalar)], &path).unwrap();
    let disk = homura::load_safetensors(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(disk[0].1.item::<i64>(), i64::MAX);
    assert!(matches!(
        homura::load_safetensors("/path/that/does/not/exist/homura"),
        Err(Error::Io(_))
    ));
}

fn file(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn safetensors_malformed_headers_fail_without_panics() {
    for header in [
        "[]",
        "null",
        "{",
        "{\"x\":{}}",
        r#"{"x":{"dtype":"F8","shape":[1],"data_offsets":[0,1]}}"#,
        r#"{"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0]}}"#,
        r#"{"x":{"dtype":"F64","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,2]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,99]}}"#,
        r#"{"\ud800":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
    ] {
        let result = catch_unwind(|| homura::load_safetensors_from_bytes(&file(header, &[0])));
        assert!(result.is_ok(), "parser panicked for {header}");
        assert!(result.unwrap().is_err(), "parser accepted {header}");
    }
    let nested = format!("{}0{}", "[".repeat(130), "]".repeat(130));
    assert!(homura::load_safetensors_from_bytes(&file(&nested, &[])).is_err());
    let mut invalid_utf8 = 1u64.to_le_bytes().to_vec();
    invalid_utf8.push(0xff);
    assert!(homura::load_safetensors_from_bytes(&invalid_utf8).is_err());
    let empty = homura::save_safetensors_to_bytes(std::iter::empty::<(&str, &Tensor)>()).unwrap();
    assert!(
        homura::load_safetensors_from_bytes(&empty)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn formatting_high_rank_nonfinite_and_noncontiguous() {
    let t = homura::arange(6).view([2, 3]).transpose(0, 1);
    assert_eq!(
        format!("{t}"),
        "tensor([[0, 3],\n        [1, 4],\n        [2, 5]])"
    );
    let t = homura::tensor([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    assert_eq!(format!("{t:?}"), "tensor([nan, inf, -inf])");
    assert!(
        format!("{:?}", homura::zeros_with([2; 9], Device::Meta))
            .contains("size=[2, 2, 2, 2, 2, 2, 2, 2, 2]")
    );
}

#[test]
fn composite_in_place_operations_preserve_aliases_and_values() {
    let x = homura::tensor([2.0, 4.0]);
    let alias = x.clone();
    let a = homura::tensor([3.0, 6.0]);
    let b = homura::tensor([2.0, 2.0]);
    x.addcmul_(&a, &b, 1).addcdiv_(&a, &b, 2);
    assert_eq!(alias.to_vec::<f32>(), [11.0, 22.0]);
    x.lerp_(&homura::zeros([2]), 0.5)
        .pow_(2)
        .sub_(0.25)
        .mul_(2)
        .div_(4);
    assert_eq!(alias.to_vec::<f32>(), [15.0, 60.375]);
    x.clamp_(0, 20)
        .masked_fill_(&homura::tensor([true, false]), -1);
    assert_eq!(alias.to_vec::<f32>(), [-1.0, 20.0]);
}
