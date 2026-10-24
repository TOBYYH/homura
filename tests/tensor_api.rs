//! Behaviour of the tensor API that golden data doesn't cover: reference semantics, views,
//! in-place rules, the meta device and error messages.

use homura::{DType, Device, Ellipsis, NewAxis, Tensor};

fn vals(t: &Tensor) -> Vec<f32> {
    t.to(DType::F32).to_vec::<f32>()
}

#[test]
fn clone_is_shallow_and_deep_clone_copies() {
    let x = homura::zeros([3]);
    let y = x.clone();
    y.add_(1.0);
    assert_eq!(vals(&x), [1.0, 1.0, 1.0]);
    assert!(x.is_same(&y));
    let z = x.deep_clone();
    z.add_(1.0);
    assert_eq!(vals(&x), [1.0, 1.0, 1.0]);
    assert_eq!(vals(&z), [2.0, 2.0, 2.0]);
    assert!(!x.is_same(&z));
}

#[test]
fn views_share_storage() {
    let x = homura::arange(12).view([3, 4]);
    x.narrow(1, 1, 2).fill_(0);
    x.index((Ellipsis, -1)).fill_(-1);
    x.select(0, 2).mul_(10);
    assert_eq!(x.to_vec::<i64>(), [0, 0, 0, -1, 4, 0, 0, -1, 80, 0, 0, -10]);
    // A transposed view is not contiguous; reshape copies it, view refuses.
    let t = x.transpose(0, 1);
    assert!(!t.is_contiguous());
    let r = t.reshape([12]);
    r.fill_(5);
    assert_eq!(x.sum().item::<i64>(), 0 - 1 + 4 - 1 + 80 - 10);
    // reshape of a contiguous tensor is a view.
    x.reshape([2, 6]).zero_();
    assert_eq!(x.to_vec::<i64>(), [0; 12]);
}

#[test]
#[should_panic(expected = "view size is not compatible")]
fn view_of_transposed_panics() {
    homura::zeros([2, 3]).transpose(0, 1).view([6]);
}

#[test]
fn split_and_chunk_are_views() {
    let x = homura::zeros([2, 5]);
    let parts = x.split(2, 1);
    assert_eq!(
        parts.iter().map(|p| p.size(1)).collect::<Vec<_>>(),
        [2, 2, 1]
    );
    parts[2].fill_(7.0);
    assert_eq!(x.sum().item::<f32>(), 14.0);
    assert_eq!(x.chunk(2, 1).len(), 2);
}

#[test]
fn in_place_on_itself_and_overlapping_copy() {
    let x = homura::tensor([1., 2., 3.]);
    x.add_(&x);
    assert_eq!(vals(&x), [2.0, 4.0, 6.0]);
    x.mul_(x.index(0..1));
    assert_eq!(vals(&x), [4.0, 8.0, 12.0]);
    // Partially overlapping source and destination behave as if the source was copied first.
    let y = homura::arange(6);
    y.narrow(0, 1, 5).copy_(&y.narrow(0, 0, 5));
    assert_eq!(y.to_vec::<i64>(), [0, 0, 1, 2, 3, 4]);
}

#[test]
fn in_place_casts_and_broadcasts() {
    let x = homura::zeros_with([2, 3], DType::F16);
    x.add_(homura::tensor([1., 2., 3.]));
    assert_eq!(x.dtype(), DType::F16);
    assert_eq!(vals(&x), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    let i = homura::tensor([1, 2, 3]);
    i.mul_(2);
    assert_eq!(i.to_vec::<i64>(), [2, 4, 6]);
    let mut j = i.clone();
    j += 1;
    j -= &homura::tensor([1, 1, 1]);
    assert_eq!(i.to_vec::<i64>(), [2, 4, 6]);
    i.copy_(&homura::tensor([1.9, -1.9, 0.5]));
    assert_eq!(i.to_vec::<i64>(), [1, -1, 0]);
}

#[test]
#[should_panic(expected = "result type f32 can't be cast to the desired output type i64")]
fn in_place_float_into_int_panics() {
    homura::tensor([1, 2]).add_(1.5);
}

#[test]
#[should_panic(expected = "result type f32 can't be cast to the desired output type i64")]
fn in_place_true_division_into_int_panics() {
    homura::tensor([1, 2]).div_(2);
}

#[test]
#[should_panic(expected = "output with shape [3] doesn't match the broadcast shape [2, 3]")]
fn in_place_cannot_grow_panics() {
    homura::zeros([3]).add_(homura::zeros([2, 3]));
}

#[test]
#[should_panic(expected = "more than one element of the written-to tensor")]
fn in_place_on_expanded_panics() {
    homura::zeros([1, 3]).expand([4, 3]).fill_(1.0);
}

#[test]
fn operators() {
    let x = homura::tensor([1., 2., 3.]);
    assert_eq!(vals(&(&x + 1)), [2.0, 3.0, 4.0]);
    assert_eq!(vals(&(2.0 * &x)), [2.0, 4.0, 6.0]);
    assert_eq!(vals(&(1.0 - &x)), [0.0, -1.0, -2.0]);
    assert_eq!(vals(&(-&x)), [-1.0, -2.0, -3.0]);
    assert_eq!(vals(&(&x / 2.0f32)), [0.5, 1.0, 1.5]);
    assert_eq!(vals(&((&x * &x).sqrt())), [1.0, 2.0, 3.0]);
    // Owned operands work too, and the result type is known without annotations.
    assert_eq!((x.clone() * 0.5).relu().sum().item::<f32>(), 3.0);
    assert_eq!((homura::tensor([2, 4]) / 4).dtype(), DType::F32);
    let mut acc = homura::zeros([3]);
    let alias = acc.clone();
    acc += &x;
    acc *= 2.0;
    assert_eq!(vals(&alias), [2.0, 4.0, 6.0]);
}

#[test]
fn creation_dtypes() {
    assert_eq!(homura::tensor([1.0, 2.0]).dtype(), DType::F32);
    assert_eq!(homura::tensor([1.0f32]).dtype(), DType::F32);
    assert_eq!(homura::tensor([1, 2]).dtype(), DType::I64);
    assert_eq!(homura::tensor([1u8]).dtype(), DType::U8);
    assert_eq!(
        homura::tensor(vec![vec![1i16, 2], vec![3, 4]]).shape(),
        [2, 2]
    );
    assert_eq!(
        homura::tensor_with([1.0, 2.0], DType::F64).dtype(),
        DType::F64
    );
    assert_eq!(
        Tensor::from_slice(&[1.0f64, 2.0], [2, 1]).dtype(),
        DType::F64
    );
    assert_eq!(homura::full([2], 3).dtype(), DType::I64);
    assert_eq!(homura::full([2], true).dtype(), DType::Bool);
    assert_eq!(homura::arange(3.0).dtype(), DType::F32);
    assert_eq!(homura::zeros(()).num_dims(), 0);
    let x = homura::ones_with([2], (DType::BF16, Device::Cpu));
    assert_eq!(
        (x.new_zeros([3]).dtype(), x.new_full([1], 2).dtype()),
        (DType::BF16, DType::BF16)
    );
    assert_eq!(homura::zeros_like(&x).shape(), [2]);
}

#[test]
#[should_panic(expected = "nested data is ragged")]
fn ragged_data_panics() {
    homura::tensor(vec![vec![1, 2], vec![3], vec![4, 5, 6]]);
}

#[test]
fn reading_data() {
    let x = homura::tensor_with([1.5, 2.5], DType::BF16);
    assert_eq!(x.to_vec::<f32>(), [1.5, 2.5]);
    assert_eq!(homura::tensor(7).item::<i64>(), 7);
    assert!(homura::tensor([[true]]).item::<bool>());
}

#[test]
#[should_panic(expected = "can't read a i64 tensor as f32")]
fn reading_wrong_type_panics() {
    homura::tensor([1, 2]).to_vec::<f32>();
}

#[test]
#[should_panic(expected = "a Tensor with 2 elements cannot be converted to Scalar")]
fn item_of_vector_panics() {
    homura::tensor([1, 2]).item::<i64>();
}

#[test]
fn shape_queries() {
    let x = homura::zeros([2, 3, 4]);
    assert_eq!(x.size(-1), 4);
    assert_eq!(x.num_dims(), 3);
    assert_eq!(x.num_elements(), 24);
    assert_eq!(x.stride(), [12, 4, 1]);
    assert_eq!(x.permute([2, 0, 1]).stride(), [1, 12, 4]);
    assert_eq!(x.narrow(1, 1, 2).storage_offset(), 4);
    assert_eq!(x.unsqueeze(-1).shape(), [2, 3, 4, 1]);
    assert_eq!(x.unsqueeze(0).squeeze().shape(), [2, 3, 4]);
    assert_eq!(
        x.index((NewAxis, .., NewAxis))
            .squeeze_dim([0, 2, 3])
            .shape(),
        [2, 3, 4]
    );
    assert_eq!(x.flatten(0, -1).shape(), [24]);
    assert_eq!(x.expand([5, -1, -1, -1]).shape(), [5, 2, 3, 4]);
}

#[test]
#[should_panic(expected = "Dimension out of range (expected to be in range of [-3, 2], but got 3)")]
fn dim_out_of_range_panics() {
    homura::zeros([2, 3, 4]).sum_dim(3, false);
}

#[test]
#[should_panic(
    expected = "The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1"
)]
fn broadcast_mismatch_panics() {
    let _ = homura::zeros([2, 3]) + homura::zeros([4]);
}

#[test]
#[should_panic(expected = "mat1 and mat2 shapes cannot be multiplied (2x3 and 4x5)")]
fn matmul_mismatch_panics() {
    homura::zeros([2, 3]).matmul(&homura::zeros([4, 5]));
}

#[test]
#[should_panic(
    expected = "Expected all tensors to be on the same device, but found at least two devices, cpu and meta!"
)]
fn device_mismatch_panics() {
    let _ = homura::zeros([2]) + homura::zeros_with([2], Device::Meta);
}

#[test]
fn indexing() {
    let x = homura::arange(10);
    let get = |t: Tensor| t.to_vec::<i64>();
    assert_eq!(get(x.index((-5..-1).step_by(2))), [5, 7]);
    assert_eq!(get(x.index((0..).step_by(3))), [0, 3, 6, 9]);
    assert_eq!(get(x.index((2..8).step_by(3))), [2, 5]);
    assert_eq!(get(x.index((2..=8).step_by(3))), [2, 5, 8]);
    assert_eq!(get(x.index(-3..)), [7, 8, 9]);
    assert_eq!(get(x.index(..-1)), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(get(x.index(..=-1)), (0..10).collect::<Vec<i64>>());
    // A reversed range is empty, like Python's x[7:3].
    let (start, end) = (7, 3);
    assert_eq!(get(x.index(start..end)), Vec::<i64>::new());
    assert_eq!(
        get(x.index(homura::tensor([[1, -1], [0, 0]]))),
        [1, 9, 0, 0]
    );
    assert_eq!(x.index(-1).item::<i64>(), 9);
    // Writing through an index view.
    x.index((1..9).step_by(2)).fill_(0);
    assert_eq!(get(x), [0, 0, 2, 0, 4, 0, 6, 0, 8, 9]);
}

#[test]
#[should_panic(expected = "too many indices for tensor of dimension 1")]
fn too_many_indices_panics() {
    homura::arange(3).index((0, 0));
}

#[test]
#[should_panic(expected = "index 3 is out of bounds for dimension 0 with size 3")]
fn tensor_index_out_of_range_panics() {
    homura::arange(3).index(homura::tensor([3]));
}

#[test]
fn meta_device_tracks_shapes_and_dtypes() {
    let x = homura::randn_with([4, 3], Device::Meta);
    let w = homura::zeros_with([5, 3], (DType::F16, Device::Meta));
    let y = x.matmul(&w.to(DType::F32).transpose(0, 1)).softmax(-1);
    assert_eq!(
        (y.shape(), y.dtype(), y.device()),
        ([4, 5].into(), DType::F32, Device::Meta)
    );
    let (v, i) = y.max_dim(1, true);
    assert_eq!((v.shape(), i.dtype()), ([4, 1].into(), DType::I64));
    assert_eq!((&x * 2).greater(0).dtype(), DType::Bool);
    assert_eq!(homura::concat(&[&x, &x], 0).shape(), [8, 3]);
    assert_eq!(
        x.index((.., &homura::tensor([0, 2]).to(Device::Meta)))
            .shape(),
        [4, 2]
    );
    x.add_(1.0).zero_();
    assert_eq!(format!("{x:?}"), "tensor(..., device=meta, size=[4, 3])");
    let _scope = homura::device_scope(Device::Meta);
    assert_eq!(homura::arange(5).device(), Device::Meta);
}

#[test]
#[should_panic(expected = "meta tensors have no data")]
fn reading_meta_panics() {
    homura::zeros_with([2], Device::Meta).to_vec::<f32>();
}

#[test]
#[should_panic(expected = "Cannot copy out of meta tensor; no data!")]
fn copying_out_of_meta_panics() {
    homura::zeros_with([2], Device::Meta).to(Device::Cpu);
}

#[test]
#[should_panic(expected = "the ROCm backend is not implemented yet")]
fn rocm_is_not_implemented_yet() {
    homura::zeros_with([2], Device::Rocm(0));
}

#[test]
fn safetensors_roundtrip_all_dtypes() {
    let base = homura::tensor([[1.5, -2.0, 0.0], [3.25, 100.0, -0.5]]);
    let tensors: Vec<(String, Tensor)> = DType::ALL
        .iter()
        .map(|&d| (format!("t.{d}"), base.to(d)))
        .chain([("transposed".to_string(), base.transpose(0, 1))])
        .chain([("scalar".to_string(), homura::tensor(4.0))])
        .collect();
    let bytes = homura::save_safetensors_to_bytes(tensors.iter().map(|(k, v)| (k, v))).unwrap();
    assert_eq!(&bytes[8..9], b"{");
    let loaded = homura::load_safetensors_from_bytes(&bytes).unwrap();
    assert_eq!(loaded.len(), tensors.len());
    for ((k1, t1), (k2, t2)) in tensors.iter().zip(&loaded) {
        assert_eq!(k1, k2);
        assert_eq!((t1.shape(), t1.dtype()), (t2.shape(), t2.dtype()));
        assert!(t1.equal(t2).all().item::<bool>(), "{k1}");
    }
}

#[test]
fn safetensors_rejects_bad_data() {
    assert!(homura::load_safetensors_from_bytes(b"\x08").is_err());
    let mut bad = 100u64.to_le_bytes().to_vec();
    bad.extend_from_slice(b"{}");
    assert!(homura::load_safetensors_from_bytes(&bad).is_err());
    let header = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
    let mut short = (header.len() as u64).to_le_bytes().to_vec();
    short.extend_from_slice(header);
    short.extend_from_slice(&[0; 4]);
    assert!(homura::load_safetensors_from_bytes(&short).is_err());
    let meta = homura::zeros_with([1], Device::Meta);
    assert!(homura::save_safetensors_to_bytes([("m", &meta)]).is_err());
    let x = homura::zeros([1]);
    assert!(homura::save_safetensors_to_bytes([("a", &x), ("a", &x)]).is_err());
}

#[test]
fn grad_mode_and_threads() {
    assert!(homura::is_grad_enabled());
    {
        let _g = homura::no_grad();
        assert!(!homura::is_grad_enabled());
    }
    assert!(!homura::zeros([1]).requires_grad());
    // Tensors can be shared across threads.
    let x = homura::ones([100]);
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let x = x.clone();
            std::thread::spawn(move || (&x * i as f64).sum().item::<f32>())
        })
        .collect();
    let sums: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(sums, [0.0, 100.0, 200.0, 300.0]);
}

/// What an operation with a shape, dimension or index argument gave, compared across the integer
/// types of the argument.
#[derive(Debug, PartialEq)]
struct Outcome {
    shape: Vec<usize>,
    stride: String,
    offset: usize,
    values: Vec<f32>,
    grad_fn: Option<&'static str>,
    shares_storage: bool,
}

/// Runs `op` on a fresh `[2, 3, 4]` tensor in a graph and records what it gave.
fn outcome(op: impl Fn(&Tensor) -> Tensor) -> Outcome {
    let leaf = homura::arange(24).to(DType::F32);
    leaf.requires_grad_(true);
    let x = leaf.mul(1.0).view([2, 3, 4]);
    let y = op(&x);
    let values = vals(&y);
    x.detach().add_(100.0);
    Outcome {
        shape: y.shape().to_vec(),
        stride: format!("{:?}", y.stride()),
        offset: y.storage_offset(),
        grad_fn: y.grad_fn().map(|g| g.name()),
        shares_storage: vals(&y) != values,
        values,
    }
}

macro_rules! shape_cases {
    ($t:ty) => {{
        let (rows, cols, one): ($t, $t, $t) = (6, 4, 1);
        let sizes = [rows, cols];
        vec![
            outcome(|x| x.view(sizes)),
            outcome(|x| x.view(&sizes)),
            outcome(|x| x.view(&sizes[..])),
            outcome(|x| x.view(sizes.to_vec())),
            outcome(|x| x.reshape(sizes)),
            outcome(|x| x.transpose(0, 2).reshape(sizes)),
            outcome(|x| x.index((.., .., ..one)).expand([2, 3, 5] as [$t; 3])),
            outcome(|x| x.view((rows, cols))),
        ]
    }};
}

#[test]
fn shape_arguments_take_i64_i32_and_usize() {
    let expected = shape_cases!(i64);
    assert_eq!(shape_cases!(i32), expected);
    assert_eq!(shape_cases!(usize), expected);
    assert!(expected[4].shares_storage && !expected[5].shares_storage);
    // `-1` with signed types, and literals, which infer as `i32`.
    let inferred = outcome(|x| x.view([-1, 4]));
    assert_eq!(inferred, expected[0]);
    assert_eq!(outcome(|x| x.view([-1i64, 4])), inferred);
    assert_eq!(outcome(|x| x.reshape(vec![6, -1])), expected[4]);
    let column = |x: &Tensor| x.index((.., .., ..1));
    assert_eq!(outcome(|x| column(x).expand([-1, -1, 5])), expected[6]);
    // Tuples mix types; `()` is the zero-dimensional shape.
    let (batch, seq) = (2usize, 3usize);
    let full = outcome(|x| x.view([2, 3, 4]));
    assert_eq!(outcome(|x| x.view((batch, seq, -1))), full);
    assert_eq!(outcome(|x| x.view((batch, 3i32, 4i64))), full);
    assert_eq!(outcome(|x| x.view((24usize,))), outcome(|x| x.view([24])));
    assert_eq!(
        outcome(|x| x.view((1, 1, 1, 1, 1, batch, seq, -1))),
        outcome(|x| x.view([1, 1, 1, 1, 1, 2, 3, 4]))
    );
    let scalar = outcome(|x| x.sum().view(()));
    assert_eq!(scalar.shape, []);
    assert_eq!(outcome(|x| x.sum().view(homura::Shape::scalar())), scalar);
    assert_eq!(outcome(|x| x.sum().view([] as [i64; 0])), scalar);
}

macro_rules! dims_cases {
    ($t:ty) => {{
        let (a, b, c): ($t, $t, $t) = (2, 0, 1);
        vec![
            outcome(|x| x.permute([a, b, c])),
            outcome(|x| x.permute(&[a, b, c])),
            outcome(|x| x.permute(&[a, b, c][..])),
            outcome(|x| x.permute(vec![a, b, c])),
            outcome(|x| x.sum_dim(c, true)),
            outcome(|x| x.sum_dim([b, a], false)),
            outcome(|x| x.mean_dim(a, false)),
            outcome(|x| x.flip([c])),
            outcome(|x| x.index((.., ..1)).squeeze_dim(c)),
            outcome(|x| x.index((.., ..1)).squeeze_dim([c])),
        ]
    }};
}

#[test]
fn dimension_arguments_take_i64_i32_and_usize() {
    let expected = dims_cases!(i64);
    assert_eq!(dims_cases!(i32), expected);
    assert_eq!(dims_cases!(usize), expected);
    // A single dimension and a list of one record different nodes, for every type.
    assert_eq!(expected[8].grad_fn, Some("SqueezeBackward1"));
    assert_eq!(expected[9].grad_fn, Some("SqueezeBackward2"));
    // Negative dimensions and literals.
    assert_eq!(outcome(|x| x.permute([-1, 0, 1])), expected[0]);
    assert_eq!(outcome(|x| x.permute([-1i64, 0, 1])), expected[0]);
    assert_eq!(outcome(|x| x.sum_dim(-2, true)), expected[4]);
    assert_eq!(outcome(|x| x.sum_dim(1, true)), expected[4]);
}

macro_rules! index_cases {
    ($t:ty) => {{
        let (zero, one, two, three): ($t, $t, $t, $t) = (0, 1, 2, 3);
        vec![
            outcome(|x| x.index(one)),
            outcome(|x| x.index((.., one))),
            outcome(|x| x.index((.., .., one..three))),
            outcome(|x| x.index((.., .., one..))),
            outcome(|x| x.index((.., .., ..three))),
            outcome(|x| x.index((.., .., one..=two))),
            outcome(|x| x.index((.., .., ..=two))),
            outcome(|x| x.index((.., .., (zero..three).step_by(2)))),
            outcome(|x| x.index((.., .., (zero..=three).step_by(2)))),
            outcome(|x| x.index((one.., .., (one..=three).step_by(2)))),
            outcome(|x| x.index((.., .., (one..two).step_by(3)))),
        ]
    }};
}

#[test]
fn index_items_take_i64_i32_and_usize() {
    let expected = index_cases!(i64);
    assert_eq!(index_cases!(i32), expected);
    assert_eq!(index_cases!(usize), expected);
    assert_eq!(expected[2].grad_fn, Some("SliceBackward0"));
    // Negative positions and literals.
    assert_eq!(outcome(|x| x.index((.., -2))), expected[1]);
    let (start, end) = (1, -1);
    assert_eq!(outcome(|x| x.index((.., .., start..end))), expected[2]);
    assert_eq!(outcome(|x| x.index((.., .., -3..=-2))), expected[5]);
    assert_eq!(
        outcome(|x| x.index((.., .., (-4..-1).step_by(2)))),
        expected[7]
    );
    assert_eq!(outcome(|x| x.index(1)), expected[0]);
}

/// The message of the panic in `f`.
fn panic_message(f: impl FnOnce()) -> String {
    let payload =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("didn't panic");
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload.downcast::<&str>().map(|m| m.to_string()).unwrap(),
    }
}

#[test]
fn usize_above_i64_max_panics() {
    let big = i64::MAX as usize + 1;
    let x = homura::zeros([2, 3]);
    let message = format!("dimension or index {big} is too large");
    let check = |name: &str, f: &dyn Fn()| assert_eq!(panic_message(f), message, "{name}");
    check("view", &|| drop(x.view([big])));
    check("view(&[..])", &|| drop(x.view(&[big][..])));
    check("view(vec)", &|| drop(x.view(vec![1, big])));
    check("view(tuple)", &|| drop(x.view((1i64, big))));
    check("expand", &|| drop(x.expand([big, 3])));
    check("permute", &|| drop(x.permute([big, 0])));
    check("sum_dim", &|| drop(x.sum_dim(big, false)));
    check("index", &|| drop(x.index(big)));
    check("index(range)", &|| drop(x.index(0..big)));
    check("index(range from)", &|| drop(x.index(big..)));
    check("index(..=)", &|| drop(x.index(..=big)));
    check("index(step_by)", &|| drop(x.index((big..=big).step_by(2))));
    check("index(step_by ..)", &|| {
        drop(x.index((big..big + 4).step_by(2)))
    });
    // The last position of a stepped `usize` range may be what doesn't fit.
    assert_eq!(
        panic_message(|| drop(x.index((0..usize::MAX).step_by(2)))),
        format!("dimension or index {} is too large", usize::MAX - 1)
    );
    // An inclusive end of `i64::MAX` fits and reaches past every dimension.
    assert_eq!(x.index(..=i64::MAX as usize).shape(), [2, 3]);
}

#[test]
fn stepped_ranges_at_the_limits_of_their_type() {
    use homura::IndexItem;
    let x = homura::arange(4);
    let get = |t: Tensor| t.to_vec::<i64>();
    let none = Vec::<i64>::new();
    // Open stepped ranges take `i64` only, so unsuffixed literals infer as `i64`, as in 0.3.0:
    // advancing an open `i32` range past `i32::MAX` would overflow inside the standard library
    // (review round 85, R109).
    assert_eq!(get(x.index((0..).step_by(1usize << 32))), [0]);
    assert_eq!(get(x.index((0i64..).step_by(1usize << 32))), [0]);
    assert_eq!(get(x.index((2_147_483_646..).step_by(1))), none);
    assert_eq!(get(x.index((2_147_483_646i64..).step_by(1))), none);
    assert_eq!(get(x.index((1..).step_by(2))), [1, 3]);
    // Finite ranges of narrow types stop at their end without overflowing.
    assert_eq!(get(x.index((i32::MAX - 1..i32::MAX).step_by(1))), none);
    assert_eq!(get(x.index((i32::MAX - 1..=i32::MAX).step_by(1))), none);
    assert_eq!(
        get(x.index((i32::MAX - 1..=i32::MAX).step_by(1usize << 32))),
        none
    );
    assert_eq!(get(x.index((0i32..4).step_by(1usize << 32))), [0]);
    assert_eq!(get(x.index((1usize..=3).step_by(2))), [1, 3]);
    // The last position of a finite range is checked even when the number of positions doesn't
    // fit in a `usize` (R110), and so is the step.
    let too_large = |v: u128| format!("dimension or index {v} is too large");
    let max = usize::MAX as u128;
    let message = |f: &dyn Fn() -> Tensor| panic_message(|| drop(f()));
    assert_eq!(
        message(&|| x.index((0..=usize::MAX).step_by(1))),
        too_large(max)
    );
    assert_eq!(
        message(&|| x.index((0..=usize::MAX).step_by(2))),
        too_large(max - 1)
    );
    assert_eq!(
        message(&|| x.index((0..usize::MAX).step_by(1))),
        too_large(max - 1)
    );
    assert_eq!(
        message(&|| x.index((usize::MAX..=usize::MAX).step_by(1))),
        too_large(max)
    );
    assert_eq!(message(&|| x.index(usize::MAX..)), too_large(max));
    assert_eq!(message(&|| x.index(usize::MAX)), too_large(max));
    let all = (i64::MIN..=i64::MAX).step_by(usize::MAX);
    assert_eq!(message(&|| x.index(all.clone())), too_large(max));
    // Ends that fit reach past every dimension.
    let end = i64::MAX as usize;
    assert_eq!(get(x.index((0..=end).step_by(1))), [0, 1, 2, 3]);
    assert_eq!(get(x.index((0..=end).step_by(2))), [0, 2]);
    assert_eq!(get(x.index((i64::MIN..=i64::MAX).step_by(1))), [0, 1, 2, 3]);
    assert_eq!(
        format!("{:?}", (0..=end).step_by(1).into_index()),
        "Slice { start: Some(0), end: None, step: 1 }"
    );
    assert_eq!(
        format!("{:?}", (0..=end).step_by(2).into_index()),
        format!(
            "Slice {{ start: Some(0), end: Some({}), step: 2 }}",
            i64::MAX
        )
    );
}
