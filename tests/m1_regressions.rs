//! Regression checks for the issues found while reviewing M1. Run in debug and release profiles.

use homura::{DType, Device, Tensor};

const DEVICES: [Device; 2] = [Device::Cpu, Device::Meta];

/// Element values with NaN compared by bit pattern, so that NaN equals NaN.
fn bits(t: &Tensor) -> Vec<u64> {
    t.to(DType::F64)
        .to_vec::<f64>()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// An empty `[4, 0]` tensor whose first stride is so large that walking the first dimension with
/// it overflows. Empty layouts touch no memory, so `as_strided` accepts any strides for them.
fn empty_with_huge_stride(device: Device) -> Tensor {
    homura::zeros_with([0], device).as_strided([4, 0], [isize::MAX as usize, 0], 0)
}

#[test]
fn matmul_with_an_empty_result_allocates_nothing_per_column() {
    // The output and both operands have no elements, but a row of `n` accumulators would not
    // fit in memory.
    let n = isize::MAX as usize / 8 + 1;
    for device in DEVICES {
        let a = homura::zeros_with([0, 0], (DType::I64, device));
        let b = homura::zeros_with([0, n], (DType::I64, device));
        assert_eq!(a.matmul(&b).shape(), [0, n]);
        let a = homura::zeros_with([0, 2, 0], device);
        let b = homura::zeros_with([1, 0, n], device);
        assert_eq!(a.matmul(&b).shape(), [0, 2, n]);
        let input = homura::zeros_with([0, n], device);
        let m = homura::zeros_with([0, 0], device);
        assert_eq!(input.addmm(&m, &b.select(0, 0), 1, 1).shape(), [0, n]);
    }
}

#[test]
fn matmul_with_an_empty_inner_dimension_ignores_operand_strides() {
    for device in DEVICES {
        let a =
            homura::zeros_with([0], device).as_strided([4, 2, 0], [isize::MAX as usize, 0, 0], 0);
        let b = homura::zeros_with([4, 0, 3], device);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), [4, 2, 3]);
        if device == Device::Cpu {
            assert_eq!(c.to_vec::<f32>(), [0.0; 24]);
        }
    }
}

#[test]
fn inclusive_ranges_up_to_i64_max_reach_the_end() {
    for device in DEVICES {
        let x = homura::arange_with(3, device);
        assert_eq!(x.index(..=i64::MAX).shape(), [3]);
        assert_eq!(x.index(0..=i64::MAX).shape(), [3]);
        assert_eq!(x.index(1..=i64::MAX).shape(), [2]);
        assert_eq!(x.index(i64::MAX..=i64::MAX).shape(), [0]);
        assert_eq!(x.index((0..=i64::MAX).step_by(2)).shape(), [2]);
        assert_eq!(x.index(..=-1).shape(), [3]);
        assert_eq!(x.index(..=-2).shape(), [2]);
        assert_eq!(x.index(-3..=-2).shape(), [2]);
    }
    let x = homura::arange(3);
    assert_eq!(x.index(..=i64::MAX).to_vec::<i64>(), [0, 1, 2]);
    assert_eq!(x.index(1..=i64::MAX).to_vec::<i64>(), [1, 2]);
    assert_eq!(x.index((0..=i64::MAX).step_by(2)).to_vec::<i64>(), [0, 2]);
}

#[test]
fn empty_splits_and_chunks_are_distinct_views() {
    for device in DEVICES {
        let x = homura::zeros_with([0, 3], device);
        let pieces = x.chunk(4, 0);
        assert_eq!(pieces.len(), 4);
        for (i, p) in pieces.iter().enumerate() {
            assert_eq!(p.shape(), [0, 3]);
            assert!(!p.is_same(&x));
            for q in &pieces[i + 1..] {
                assert!(!p.is_same(q));
            }
        }
        let split = x.split(0, 0);
        assert_eq!(split.len(), 1);
        assert_eq!(split[0].shape(), [0, 3]);
        assert!(!split[0].is_same(&x));
    }
}

type Op = fn(&Tensor) -> Tensor;

#[test]
fn reductions_and_softmax_of_empty_layouts_ignore_their_strides() {
    for device in DEVICES {
        let x = empty_with_huge_stride(device);
        let plain = homura::zeros_with([4, 0], device);
        let ops: [(&str, Op); 10] = [
            ("sum_dim", |t| t.sum_dim(1, false)),
            ("prod_dim", |t| t.prod_dim(1, false)),
            ("mean_dim", |t| t.mean_dim(1, true)),
            ("var_dim", |t| t.var_dim(1, 1, false)),
            ("std_dim", |t| t.std_dim(1, 0, false)),
            ("all_dim", |t| t.all_dim(1, false)),
            ("softmax", |t| t.softmax(1)),
            ("log_softmax", |t| t.log_softmax(1)),
            ("triu", |t| t.unsqueeze(1).triu(0)),
            ("max_dim over the nonempty dimension", |t| {
                t.max_dim(0, false).0
            }),
        ];
        for (name, op) in ops {
            let (got, expected) = (op(&x), op(&plain));
            assert_eq!(got.shape(), expected.shape(), "{name} on {device}");
            assert_eq!(got.dtype(), expected.dtype(), "{name} on {device}");
            if device == Device::Cpu {
                assert_eq!(bits(&got), bits(&expected), "{name}");
            }
        }
    }
    let x = empty_with_huge_stride(Device::Cpu);
    assert_eq!(x.sum_dim(1, false).to_vec::<f32>(), [0.0; 4]);
    assert_eq!(x.prod_dim(1, false).to_vec::<f32>(), [1.0; 4]);
}

#[test]
fn indexing_empty_layouts_checks_indices_without_computing_slice_offsets() {
    let x = empty_with_huge_stride(Device::Cpu);
    let index = |v: &[i64]| homura::tensor(v.to_vec());
    assert_eq!(x.index_select(0, &index(&[3])).shape(), [1, 0]);
    assert_eq!(x.index_select(0, &index(&[3, 0, 2])).shape(), [3, 0]);
    assert_eq!(x.index_select(0, &index(&[])).shape(), [0, 0]);
    assert_eq!(x.index((&index(&[3]), ..)).shape(), [1, 0]);
    assert_eq!(x.index((&index(&[-1]), ..)).shape(), [1, 0]);
    let grid = homura::tensor([[3i64, -4], [0, 1]]);
    assert_eq!(x.index((&grid, ..)).shape(), [2, 2, 0]);
    assert_eq!(x.index((&grid, ..)).to_vec::<f32>(), [0.0; 0]);
    // Invalid indices are still rejected, even though nothing is copied.
    for bad in [
        || {
            empty_with_huge_stride(Device::Cpu).index_select(0, &homura::tensor([4i64]));
        },
        || {
            empty_with_huge_stride(Device::Cpu).index_select(0, &homura::tensor([-1i64]));
        },
        || {
            empty_with_huge_stride(Device::Cpu).index((&homura::tensor([-5i64]), ..));
        },
    ] {
        assert!(std::panic::catch_unwind(bad).is_err());
    }
    let x = empty_with_huge_stride(Device::Meta);
    assert_eq!(
        x.index_select(0, &index(&[3]).to(Device::Meta)).shape(),
        [1, 0]
    );
    assert_eq!(x.index((&grid.to(Device::Meta), ..)).shape(), [2, 2, 0]);
}
