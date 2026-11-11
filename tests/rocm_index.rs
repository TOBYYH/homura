//! Indexing on a ROCm GPU against the CPU (M4 step 4b).
//!
//! - index_select, gather, index_add, scatter_add, advanced indexing (integer indices, masks,
//!   broadcasting, non-adjacent dimensions, indices on the CPU, on the GPU or both) and
//!   index_put with and without accumulation: every dtype, strided operands and indices, results
//!   on the GPU with the CPU's bits. Repeated positions add up in the CPU's order (values that
//!   give another sum in any other order) and overwrite with the last one.
//! - Masks whose coordinates need several tiles and chunks of the device scan.
//! - Positions out of range: the CPU's message for the position the CPU reports (the first in
//!   the order of the indices and of their logical positions, not of memory), also when the
//!   result is empty, for Embedding and NLL, and with the target left as it was.
#![cfg(feature = "rocm")]

#[path = "common/rocm.rs"]
mod rocm;

use homura::nn::functional as F;
use homura::{DType, Device, Tensor};
use rocm::{DTYPES, GPU, Words, assert_bitwise, assert_same, layouts, values};

/// The message of the panic of `f`, which must panic.
fn panic_message(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("no panic");
    match e.downcast::<String>() {
        Ok(s) => *s,
        Err(e) => e.downcast_ref::<&str>().unwrap().to_string(),
    }
}

#[track_caller]
fn assert_panics(expected: &str, f: impl FnOnce()) {
    assert_eq!(panic_message(f), expected);
}

fn ints(v: &[i64]) -> Tensor {
    Tensor::from_slice(v, [v.len()])
}

/// `n` positions in `0..size` (with repeats), from `seed`.
fn random_positions(n: usize, size: usize, seed: u64) -> Vec<i64> {
    let mut w = Words::new(seed);
    (0..n).map(|_| (w.next() % size as u64) as i64).collect()
}

/// An index tensor of `shape` with the given values (row-major), in three layouts: contiguous
/// I64, I32, and I64 as a transposed (or strided, for one dimension) view.
fn index_layouts(v: &[i64], shape: &[usize]) -> Vec<(&'static str, Tensor)> {
    let t = Tensor::from_slice(v, [v.len()])
        .reshape(shape.iter().map(|&n| n as i64).collect::<Vec<_>>());
    let strided = if shape.len() >= 2 {
        let nd = shape.len();
        t.transpose(0, nd as i64 - 1)
            .contiguous()
            .transpose(0, nd as i64 - 1)
    } else {
        // Every other element of a longer tensor.
        let doubled: Vec<i64> = v.iter().flat_map(|&x| [x, -7]).collect();
        Tensor::from_slice(&doubled, [doubled.len()])
            .reshape([-1, 2])
            .select(1, 0)
            .reshape(shape.iter().map(|&n| n as i64).collect::<Vec<_>>())
    };
    assert_eq!(strided.to_vec::<i64>(), t.to_vec::<i64>());
    vec![
        ("I64", t.clone()),
        ("I32", t.to(DType::I32)),
        ("strided I64", strided),
    ]
}

#[test]
fn index_select_matches_the_cpu() {
    for (s, &dtype) in DTYPES.iter().enumerate() {
        for pair in layouts(dtype, 10 + s as u64) {
            let shape = pair.cpu.shape();
            for dim in 0..shape.len() {
                let n = shape[dim];
                let picks: Vec<i64> = if n == 0 {
                    vec![]
                } else {
                    vec![n as i64 - 1, 0, n as i64 - 1, (n / 2) as i64]
                };
                for (iname, index) in index_layouts(&picks, &[picks.len()]) {
                    let what = format!("{dtype} {} dim {dim} {iname}", pair.name);
                    let cpu = pair.cpu.index_select(dim as i64, &index);
                    let gpu = pair.gpu.index_select(dim as i64, &index.to(GPU));
                    assert_same(&gpu, &cpu, &what);
                }
            }
        }
    }
}

/// A tensor of `shape` of `dtype` in three layouts on the CPU (contiguous, permuted, narrowed
/// with an offset) and the same views on the GPU.
fn operand_layouts(
    dtype: DType,
    shape: &[usize],
    seed: u64,
) -> Vec<(&'static str, Tensor, Tensor)> {
    let n: usize = shape.iter().product();
    let dims: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    let rev: Vec<i64> = dims.iter().rev().copied().collect();
    let nd = shape.len() as i64;
    let perm: Vec<i64> = (0..nd).rev().collect();
    let base = values(dtype, n, seed);
    let wide = values(dtype, n * 2 + 3, seed + 1);
    let mut out = vec![];
    let mut add = |name: &'static str, cpu: Tensor| {
        let gpu = match name {
            "contiguous" => base.to(GPU).reshape(dims.clone()),
            "permuted" => base.to(GPU).reshape(rev.clone()).permute(perm.clone()),
            _ => wide
                .to(GPU)
                .narrow(0, 3, n * 2)
                .reshape([n as i64, 2])
                .select(1, 1)
                .reshape(dims.clone()),
        };
        assert_eq!(gpu.stride(), cpu.stride(), "{name}");
        out.push((name, cpu, gpu));
    };
    add("contiguous", base.reshape(dims.clone()));
    add("permuted", base.reshape(rev.clone()).permute(perm.clone()));
    add(
        "strided",
        wide.narrow(0, 3, n * 2)
            .reshape([n as i64, 2])
            .select(1, 1)
            .reshape(dims.clone()),
    );
    out
}

#[test]
fn gather_matches_the_cpu() {
    let shape = [4, 5, 6];
    for (s, &dtype) in DTYPES.iter().enumerate() {
        for (sname, cpu, gpu) in operand_layouts(dtype, &shape, 100 + s as u64) {
            for dim in 0..3 {
                // Longer than the source along `dim`, shorter in one other dimension.
                let mut ishape = shape.to_vec();
                ishape[dim] = 7;
                ishape[(dim + 1) % 3] -= 1;
                let n: usize = ishape.iter().product();
                let picks = random_positions(n, shape[dim], 7 * s as u64 + dim as u64);
                for (iname, index) in index_layouts(&picks, &ishape) {
                    let what = format!("{dtype} {sname} dim {dim} {iname}");
                    let c = cpu.gather(dim as i64, &index);
                    let g = gpu.gather(dim as i64, &index.to(GPU));
                    assert_same(&g, &c, &what);
                }
            }
        }
    }
}

#[test]
fn scatter_add_and_index_add_match_the_cpu() {
    let shape = [4, 5, 6];
    for (s, &dtype) in DTYPES.iter().enumerate() {
        for (sname, cpu, gpu) in operand_layouts(dtype, &shape, 200 + s as u64) {
            for dim in 0..3 {
                // scatter_add: an index smaller than self in the other dimensions, a source
                // larger than the index.
                let mut ishape = shape.to_vec();
                ishape[dim] = 9;
                ishape[(dim + 2) % 3] -= 2;
                let mut sshape = ishape.clone();
                sshape[(dim + 1) % 3] += 1;
                sshape[dim] += 2;
                let n: usize = ishape.iter().product();
                let m: usize = sshape.iter().product();
                let picks = random_positions(n, shape[dim], 11 * s as u64 + dim as u64);
                let src = values(dtype, m, 300 + s as u64)
                    .reshape(sshape.iter().map(|&k| k as i64).collect::<Vec<_>>());
                for (iname, index) in index_layouts(&picks, &ishape) {
                    let what = format!("scatter_add {dtype} {sname} dim {dim} {iname}");
                    let c = cpu.scatter_add(dim as i64, &index, &src);
                    let g = gpu.scatter_add(dim as i64, &index.to(GPU), &src.to(GPU));
                    assert_bitwise(&g, &c, &what);
                }
                // index_add: a vector of 9 positions with repeats.
                let picks = random_positions(9, shape[dim], 13 * s as u64 + dim as u64);
                let mut sshape = shape.to_vec();
                sshape[dim] = 9;
                let m: usize = sshape.iter().product();
                let src = values(dtype, m, 400 + s as u64)
                    .reshape(sshape.iter().map(|&k| k as i64).collect::<Vec<_>>());
                let src_t = src.transpose(0, 2).contiguous().transpose(0, 2);
                for (iname, index) in index_layouts(&picks, &[9]) {
                    for (srcname, source) in [("source", &src), ("transposed source", &src_t)] {
                        let what = format!("index_add {dtype} {sname} dim {dim} {iname} {srcname}");
                        let c = cpu.index_add(dim as i64, &index, source);
                        let g = gpu.index_add(dim as i64, &index.to(GPU), &source.to(GPU));
                        assert_bitwise(&g, &c, &what);
                    }
                }
            }
        }
    }
    // A zero-dimensional index and source.
    let x = homura::zeros([3]);
    let (i, v) = (homura::tensor(2i64), homura::tensor([5.0f32]));
    let c = x.index_add(0, &i, &v);
    let g = x.to(GPU).index_add(0, &i.to(GPU), &v.to(GPU));
    assert_same(&g, &c, "0-dim index_add");
}

/// Values whose f64 sum depends on the order: 2^53 + 1 + 1 - 2^53 is 0 from the left (the ones
/// are lost) and 2 when the ones come first.
fn order_sensitive(n: usize) -> Vec<f64> {
    let big = 2f64.powi(53);
    (0..n)
        .map(|i| match i % 4 {
            0 => big,
            3 => -big,
            _ => 1.0,
        })
        .collect()
}

#[test]
fn repeated_positions_add_in_the_cpus_order() {
    // Every element goes to one of two targets; within a target, the order of the source decides
    // the sum.
    let n = 4000;
    let v = order_sensitive(n);
    let src = Tensor::from_slice(&v, [n]);
    let picks: Vec<i64> = (0..n).map(|i| ((i / 4) % 2) as i64).collect();
    let index = ints(&picks);
    let reversed = ints(&picks.iter().rev().copied().collect::<Vec<_>>());
    let src_rev = src.flip([0]);
    let x = homura::zeros_with([2], DType::F64);
    for (name, i, s) in [("forward", &index, &src), ("reversed", &reversed, &src_rev)] {
        let c = x.scatter_add(0, i, s);
        let g = x.to(GPU).scatter_add(0, &i.to(GPU), &s.to(GPU));
        assert_same(&g, &c, &format!("scatter_add {name}"));
        let c = x.index_add(0, i, s);
        let g = x.to(GPU).index_add(0, &i.to(GPU), &s.to(GPU));
        assert_same(&g, &c, &format!("index_add {name}"));
        let c = x.index_put_accumulate((i,), s);
        let g = x.to(GPU).index_put_accumulate((&i.to(GPU),), &s.to(GPU));
        assert_same(&g, &c, &format!("index_put accumulate {name}"));
    }
    // The order matters: the two directions give different sums.
    let forward = x.scatter_add(0, &index, &src).to_vec::<f64>();
    let backward = x.scatter_add(0, &reversed, &src_rev).to_vec::<f64>();
    assert_ne!(
        forward, backward,
        "the same pairs in the other order sum differently"
    );
    let shuffled: Vec<f64> = {
        // [1, 1, 2^53, -2^53] sums to 2.
        let mut w = v.clone();
        w.swap(0, 2);
        w
    };
    let other = x.scatter_add(0, &index, &Tensor::from_slice(&shuffled, [n]));
    assert_ne!(
        other.to_vec::<f64>(),
        forward,
        "a reordered group sums differently"
    );
    // Many more pairs than one tile of the sort, and keys that need several passes.
    let big = 3 * 2048 + 17;
    let w = order_sensitive(big);
    let picks = random_positions(big, 70_000, 5);
    let x = homura::zeros_with([70_000], DType::F64);
    let c = x.scatter_add(0, &ints(&picks), &Tensor::from_slice(&w, [big]));
    let g = x.to(GPU).scatter_add(
        0,
        &ints(&picks).to(GPU),
        &Tensor::from_slice(&w, [big]).to(GPU),
    );
    assert_same(&g, &c, "scatter_add with several tiles");
}

/// The cases of advanced indexing on a [5, 6, 7] target: (name, indices).
fn advanced_cases() -> Vec<(&'static str, Vec<homura::TensorIndex>)> {
    use homura::IndexItem;
    use homura::TensorIndex as I;
    let t = |v: &[i64]| I::Tensor(ints(v));
    let m = |shape: [usize; 2], seed: u64| {
        let mut w = Words::new(seed);
        let v: Vec<bool> = (0..shape[0] * shape[1])
            .map(|_| !w.next().is_multiple_of(3))
            .collect();
        I::Tensor(Tensor::from_slice(&v, shape))
    };
    let all = || (..).into_index();
    vec![
        (
            "one index with repeats and negatives",
            vec![t(&[4, -1, 0, -5, 2, 4])],
        ),
        (
            "two broadcast indices",
            vec![
                I::Tensor(Tensor::from_slice(&[1i64, -2], [2, 1])),
                t(&[0, 5, -6]),
            ],
        ),
        ("non-adjacent", vec![t(&[3, 1]), all(), t(&[6, -7])]),
        ("after a slice", vec![(1..4).into_index(), t(&[5, 0, 5])]),
        ("mask over two dimensions", vec![m([5, 6], 1)]),
        ("mask then index", vec![all(), m([6, 7], 2)]),
        (
            "transposed index",
            vec![I::Tensor(
                Tensor::from_slice(&[0i64, 1, 2, 3, 4, 0], [2, 3]).transpose(0, 1),
            )],
        ),
    ]
}

/// The case with every index tensor on `device`, or (with `mixed`) alternately on the CPU and
/// on the GPU.
fn place(items: &[homura::TensorIndex], device: Device, mixed: bool) -> Vec<homura::TensorIndex> {
    use homura::TensorIndex as I;
    let mut k = 0;
    items
        .iter()
        .map(|item| match item {
            I::Tensor(t) => {
                k += 1;
                let d = if mixed && k % 2 == 0 {
                    Device::Cpu
                } else {
                    device
                };
                I::Tensor(t.to(d))
            }
            other => other.clone(),
        })
        .collect()
}

#[test]
fn advanced_indexing_matches_the_cpu() {
    for (s, &dtype) in DTYPES.iter().enumerate() {
        for (sname, cpu, gpu) in operand_layouts(dtype, &[5, 6, 7], 500 + s as u64) {
            for (name, items) in advanced_cases() {
                let c = cpu.index(items.clone());
                for (where_, device, mixed) in [
                    ("GPU indices", GPU, false),
                    ("CPU indices", Device::Cpu, false),
                    ("mixed indices", GPU, true),
                ] {
                    let what = format!("{dtype} {sname} {name}, {where_}");
                    let g = gpu.index(place(&items, device, mixed));
                    assert_same(&g, &c, &what);
                }
            }
        }
    }
}

#[test]
fn index_put_matches_the_cpu() {
    for (s, &dtype) in DTYPES.iter().enumerate() {
        for (sname, cpu, gpu) in operand_layouts(dtype, &[5, 6, 7], 600 + s as u64) {
            // index_put takes tensors and full slices only.
            for (name, items) in advanced_cases()
                .into_iter()
                .filter(|(n, _)| *n != "after a slice")
            {
                let shape = cpu.index(items.clone()).shape();
                let n: usize = shape.iter().product();
                let v = values(dtype, n, 700 + s as u64)
                    .reshape(shape.iter().map(|&k| k as i64).collect::<Vec<_>>());
                for (where_, device, mixed) in [("GPU indices", GPU, false), ("mixed", GPU, true)] {
                    let placed = place(&items, device, mixed);
                    let what = format!("{dtype} {sname} {name}, {where_}");
                    // Overwriting: repeated positions keep the last value.
                    let c = cpu.index_put(items.clone(), &v);
                    let g = gpu.index_put(placed.clone(), &v.to(GPU));
                    assert_same(&g, &c, &format!("index_put {what}"));
                    let c = cpu.index_put_accumulate(items.clone(), &v);
                    let g = gpu.index_put_accumulate(placed.clone(), &v.to(GPU));
                    assert_bitwise(&g, &c, &format!("index_put accumulate {what}"));
                    // A broadcast value.
                    let one = v.reshape([-1]).narrow(0, 0, n.min(1));
                    if n > 0 {
                        let c = cpu.index_put(items.clone(), &one.reshape(()));
                        let g = gpu.index_put(placed.clone(), &one.reshape(()).to(GPU));
                        assert_same(&g, &c, &format!("index_put broadcast {what}"));
                    }
                }
            }
        }
    }
    // In place, on the GPU tensor itself.
    let x = homura::arange(10.0).to(GPU);
    x.index_put_(
        (&ints(&[3, 3, 7]).to(GPU),),
        &homura::tensor([1.0f32, 2.0, 3.0]).to(GPU),
    );
    assert_eq!(x.device(), GPU);
    assert_eq!(
        x.to_vec::<f32>(),
        [0.0, 1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 3.0, 8.0, 9.0]
    );
}

#[test]
fn large_masks_compact_in_order() {
    let mut w = Words::new(42);
    // More true elements than one tile of the count (16384) and than one chunk of the scan of
    // the tiles' counts, in several layouts.
    let (rows, cols) = (3001, 1709);
    let v: Vec<bool> = (0..rows * cols).map(|_| w.next() % 5 < 3).collect();
    let mask = Tensor::from_slice(&v, [rows, cols]);
    let x =
        homura::arange_with((rows * cols) as i64, DType::I64).reshape([rows as i64, cols as i64]);
    let gx = x.to(GPU);
    for (name, m, cx, gxx) in [
        ("contiguous", mask.clone(), x.clone(), gx.clone()),
        (
            "transposed",
            mask.transpose(0, 1),
            x.transpose(0, 1),
            gx.transpose(0, 1),
        ),
    ] {
        let c = cx.index(&m);
        let g = gxx.index(m.to(GPU));
        assert_same(&g, &c, &format!("{name} mask"));
    }
    for fill in [false, true] {
        let m = homura::full_with([rows, cols], fill, DType::Bool);
        let c = x.index(&m);
        let g = gx.index(m.to(GPU));
        assert_same(&g, &c, &format!("all {fill}"));
    }
    // A mask of one dimension and one of three.
    let m1 = mask.reshape([-1]).narrow(0, 0, 1000);
    let x1 = homura::arange(1000.0);
    assert_same(&x1.to(GPU).index(m1.to(GPU)), &x1.index(&m1), "1-D mask");
    let m3 = mask.reshape([-1]).narrow(0, 0, 60).reshape([3, 4, 5]);
    let x3 = homura::arange(120.0).reshape([3, 4, 10]).narrow(2, 3, 5);
    let c = x3.index((&m3,));
    assert_eq!(c.num_dims(), 1, "{:?}", c.shape());
    assert_same(
        &x3.to(GPU).index((&m3.to(GPU),)),
        &c,
        "3-D mask of a strided target",
    );
}

#[test]
fn positions_out_of_range_report_what_the_cpu_reports() {
    let x = homura::arange(30.0).reshape([5, 6]);
    let g = x.to(GPU);
    let on = |t: &Tensor, d: Device| t.to(d);
    for d in [Device::Cpu, GPU] {
        let xd = x.to(d);
        // The first offending element in order: a negative position comes first here.
        assert_panics(
            "index -1 is out of bounds for dimension 0 with size 5",
            || drop(xd.index_select(0, &on(&ints(&[1, -1, 9]), d))),
        );
        assert_panics(
            "index 6 is out of bounds for dimension 1 with size 6",
            || drop(xd.gather(1, &on(&Tensor::from_slice(&[0i64, 6, 7, 0], [2, 2]), d))),
        );
        // The logical order of a transposed index, not its memory order.
        let t = Tensor::from_slice(&[0i64, 9, 8, 0], [2, 2]).transpose(0, 1);
        assert_eq!(t.to_vec::<i64>(), [0, 8, 9, 0]);
        assert_panics(
            "index 8 is out of bounds for dimension 0 with size 5",
            || drop(xd.narrow(1, 0, 2).gather(0, &on(&t, d))),
        );
        assert_panics(
            "index 7 is out of bounds for dimension 0 with size 5",
            || drop(xd.index_add(0, &on(&ints(&[0, 7, -2]), d), &homura::ones([3, 6]).to(d))),
        );
        assert_panics(
            "index -2 is out of bounds for dimension 0 with size 5",
            || {
                drop(xd.scatter_add(
                    0,
                    &on(&Tensor::from_slice(&[0i64, -2], [2, 1]), d),
                    &homura::ones([2, 1]).to(d),
                ))
            },
        );
        // Advanced indexing checks the indices in order: the second one's offence is reported
        // only when the first has none, whatever the logical positions.
        assert_panics(
            "index 5 is out of bounds for dimension 0 with size 5",
            || drop(xd.index((&on(&ints(&[0, 0, 5]), d), &on(&ints(&[6, 0, 0]), d)))),
        );
        assert_panics(
            "index -7 is out of bounds for dimension 1 with size 6",
            || drop(xd.index((&on(&ints(&[0, 1]), d), &on(&ints(&[-7, 1]), d)))),
        );
        // Broadcast views: positions of the index itself, before broadcasting.
        let b = Tensor::from_slice(&[1i64, 9], [2, 1]).expand([2, 3]);
        assert_panics(
            "index 9 is out of bounds for dimension 0 with size 5",
            || drop(xd.index((&on(&b, d),))),
        );
        // Checked even when the result is empty.
        assert_panics(
            "index 9 is out of bounds for dimension 1 with size 6",
            || drop(xd.index((&on(&ints(&[]), d).reshape([0, 1]), &on(&ints(&[9]), d)))),
        );
    }
    // Indices on both devices: the CPU's or the GPU's offence, whichever index comes first.
    assert_panics(
        "index 7 is out of bounds for dimension 0 with size 5",
        || drop(g.index((&ints(&[7]), &ints(&[9]).to(GPU)))),
    );
    assert_panics(
        "index 8 is out of bounds for dimension 0 with size 5",
        || drop(g.index((&ints(&[8]).to(GPU), &ints(&[9])))),
    );
    assert_panics(
        "index 9 is out of bounds for dimension 1 with size 6",
        || drop(g.index((&ints(&[0]).to(GPU), &ints(&[9])))),
    );
    assert_panics(
        "index 9 is out of bounds for dimension 1 with size 6",
        || drop(g.index((&ints(&[0]), &ints(&[9]).to(GPU)))),
    );
}

#[test]
fn failed_writes_leave_the_target_unchanged() {
    let x = homura::arange(6.0).to(GPU);
    let before = x.to_vec::<f32>();
    assert_panics(
        "index 6 is out of bounds for dimension 0 with size 6",
        || {
            x.index_put_((&ints(&[1, 6]).to(GPU),), &homura::ones([2]).to(GPU));
        },
    );
    assert_panics(
        "index 9 is out of bounds for dimension 0 with size 6",
        || {
            x.index_add_(0, &ints(&[9]).to(GPU), &homura::ones([1]).to(GPU));
        },
    );
    assert_panics(
        "index 9 is out of bounds for dimension 0 with size 6",
        || {
            x.scatter_add_(0, &ints(&[2, 9]).to(GPU), &homura::ones([2]).to(GPU));
        },
    );
    assert_eq!(x.to_vec::<f32>(), before);
}

/// Sorts whose workspace size overflows an i64 (indices and sources of 2^58 elements, views of
/// one element with zero strides) fail before any kernel is queued, with the target unchanged;
/// the sizes once wrapped to negative numbers, which gave no workspace at all (review round 131,
/// R154).
#[test]
fn sorts_too_large_to_size_fail_before_any_kernel() {
    let n = 1usize << 58;
    let x = homura::arange(4.0).to(GPU);
    let before = x.to_vec::<f32>();
    let index = ints(&[1]).to(GPU).expand([n]);
    let src = homura::ones([1]).to(GPU).expand([n]);
    assert_panics("the workspace is too large", || {
        x.index_add_(0, &index, &src);
    });
    assert_panics("the workspace is too large", || {
        x.scatter_add_(0, &index, &src);
    });
    assert_eq!(x.to_vec::<f32>(), before);
}

#[test]
fn embedding_and_nll_check_on_the_device() {
    let weight = homura::arange(12.0).reshape([4, 3]);
    let input = Tensor::from_slice(&[0i64, 3, 1, 3], [2, 2]);
    for dtype in [DType::I64, DType::I32] {
        let i = input.to(dtype);
        let c = F::embedding(&i, &weight, None);
        let g = F::embedding(&i.to(GPU), &weight.to(GPU), None);
        assert_same(&g, &c, &format!("embedding {dtype}"));
        let bad = Tensor::from_slice(&[0i64, 4], [2]).to(dtype);
        assert_panics("index out of range in self", || {
            drop(F::embedding(&bad.to(GPU), &weight.to(GPU), None))
        });
    }
    let logits = homura::randn([4, 5]);
    let target = ints(&[1, -100, 4, 0]);
    let c = F::nll_loss(&logits, &target, None, -100, F::Reduction::Sum);
    let g = F::nll_loss(
        &logits.to(GPU),
        &target.to(GPU),
        None,
        -100,
        F::Reduction::Sum,
    );
    assert_same(&g, &c, "nll_loss with an ignored target");
    // The first target out of range that isn't ignored.
    let bad = ints(&[1, -100, 7, 5]);
    for d in [Device::Cpu, GPU] {
        assert_panics("Target 7 is out of bounds.", || {
            drop(F::nll_loss(
                &logits.to(d),
                &bad.to(d),
                None,
                -100,
                F::Reduction::Mean,
            ))
        });
    }
}

/// Keys that need every pass of the radix sort (4 of 8 bits for a target of 2^25 elements) and
/// more tiles than one chunk of the scan of their digit counts.
#[test]
fn large_scatters_sort_every_digit() {
    let n = 200_000;
    let size = (1 << 25) + 5;
    let picks = random_positions(n, size, 77);
    let src = values(DType::U8, n, 78);
    let x = values(DType::U8, size, 79);
    let c = x.scatter_add(0, &ints(&picks), &src);
    let g = x
        .to(GPU)
        .scatter_add(0, &ints(&picks).to(GPU), &src.to(GPU));
    assert_same(&g, &c, "scatter_add into 2^25 elements");
    // Many repeats of few targets, in f64: long runs added in order.
    let picks = random_positions(n, 300, 80);
    let v = order_sensitive(n);
    let x = homura::zeros_with([300], DType::F64);
    let s = Tensor::from_slice(&v, [n]);
    let c = x.index_add(0, &ints(&picks), &s);
    let g = x.to(GPU).index_add(0, &ints(&picks).to(GPU), &s.to(GPU));
    assert_same(&g, &c, "index_add with long groups");
}

/// Indices on the CPU are copied to the GPU before its lock is taken: indexing a GPU tensor with
/// them while another thread keeps the CPU busy (and writes the indices) finishes, and each
/// result is one that a consistent snapshot of the indices gives.
#[test]
fn cpu_indices_are_snapshots_taken_outside_the_gpu_lock() {
    let x = homura::arange_with(40, DType::I64).reshape([8, 5]).to(GPU);
    let index = ints(&[1, 2, 3]);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writer = {
        let (index, stop) = (index.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut k = 0i64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                // Every write leaves three equal positions, all in range.
                index.fill_(k % 8);
                // CPU work that holds the CPU's lock.
                drop(homura::ones([64, 64]).matmul(&homura::ones([64, 64])));
                k += 1;
            }
        })
    };
    for _ in 0..200 {
        let rows = x
            .index((&index, &ints(&[0, 1, 2]).to(GPU)))
            .to(Device::Cpu)
            .to_vec::<i64>();
        // Rows of one snapshot: the same row r for the three columns, or the initial [1, 2, 3].
        let r = rows[0] / 5;
        let consistent = rows == [5 * r, 5 * r + 1, 5 * r + 2] || rows == [5, 11, 17];
        assert!(consistent, "{rows:?}");
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    writer.join().unwrap();
}

/// A zero-dimensional integer tensor on the GPU selects like an integer (its value is read back)
/// and gives a view; out of range, the CPU's message.
#[test]
fn zero_dimensional_gpu_indices_select() {
    let x = homura::arange(12.0).reshape([3, 4]);
    let g = x.to(GPU);
    for i in [2i64, -1] {
        let c = x.index(homura::tensor(i));
        let v = g.index(homura::tensor(i).to(GPU));
        assert_same(&v, &c, &format!("x[{i}]"));
    }
    let cpu = panic_message(|| drop(x.index(homura::tensor(3i64))));
    let gpu = panic_message(|| drop(g.index(homura::tensor(3i64).to(GPU))));
    assert_eq!(gpu, cpu);
}
