//! Every operation that runs a kernel reaches the ROCm arm of the dispatch layer for ROCm
//! tensors, instead of treating them like meta tensors (which would give results without data).
//! The tensors here are placeholders without memory (`storage::fake_rocm`), so every kernel must
//! panic; an operation that returns has skipped its kernel.

use crate::nn::functional::{self as F, MaxPool2dOptions};
use crate::storage::fake_rocm;
use crate::{DType, Device, Tensor};

const GPU: Device = Device::Rocm(0);

/// The message of the panic of `f`, which must panic.
fn panic_message<R>(f: impl FnOnce() -> R) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fake_rocm::with(f);
    }))
    .expect_err("returned without running a kernel");
    match e.downcast::<String>() {
        Ok(s) => *s,
        Err(e) => e.downcast_ref::<&str>().unwrap().to_string(),
    }
}

/// The kernels that the ROCm backend implements: with the `rocm` feature they reach the
/// placeholder storage and panic there.
const IMPLEMENTED: [&str; 36] = [
    "write_host",
    "read_host",
    "copy",
    "fill",
    "upload",
    "download",
    "unary",
    "binary",
    "binary_scalar",
    "fused",
    "where_",
    "triangle",
    "flip",
    "uniform",
    "normal",
    "randint",
    "reduce",
    "var",
    "arg_reduce",
    "softmax",
    "exclusive_cumprod",
    "index_select",
    "gather",
    "index_add",
    "scatter_add",
    "index_overwrite",
    "index_positions",
    "mask_coordinates",
    "out_of_range",
    "max_pool2d",
    "max_pool2d_backward",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool2d_backward",
    "unfold_backward",
    "gemm",
    "addr",
];

/// Asserts that `f` panics in the ROCm arm of the dispatch layer, in `kernel`.
#[track_caller]
fn reaches<R>(kernel: &str, f: impl FnOnce() -> R) {
    let message = panic_message(f);
    let expected = if cfg!(feature = "rocm") && IMPLEMENTED.contains(&kernel) {
        format!("internal error: {kernel} on ROCm storage without device memory")
    } else if cfg!(feature = "rocm") {
        format!("{kernel} is not implemented on ROCm yet (rocm:0)")
    } else {
        "homura was built without ROCm support; enable the `rocm` feature".into()
    };
    assert_eq!(message, expected, "for {kernel}");
}

fn gpu(shape: &[usize]) -> Tensor {
    crate::zeros_with(shape, GPU)
}

fn cpu_index(values: &[i64]) -> Tensor {
    crate::tensor(values.to_vec())
}

fn gpu_index(n: usize) -> Tensor {
    crate::zeros_with([n], (DType::I64, GPU))
}

#[test]
fn creation_and_copies() {
    reaches("fill", || crate::full_with([2], 1.0, GPU));
    reaches("fill", || crate::ones_with([2], GPU));
    reaches("write_host", || crate::arange_with(4, GPU));
    reaches("write_host", || {
        drop(crate::tensor_with([1.0f32, 2.0], GPU))
    });
    reaches("copy", || gpu(&[2, 3]).transpose(0, 1).contiguous());
    reaches("copy", || crate::concat(&[gpu(&[2]), gpu(&[3])], 0));
    reaches("copy", || F::pad(&gpu(&[2, 2]), &[1, 1], 0.0));
    reaches("fill", || {
        gpu(&[2]).fill_(3.0);
    });
}

#[test]
fn elementwise() {
    reaches("unary", || gpu(&[2]).exp());
    // Integers are promoted to the default float dtype: the cast (`cast_data`) comes first.
    reaches("copy", || crate::zeros_with([2], (DType::I64, GPU)).exp());
    // A scalar operand of a non-zero value is made on the device (`scalar_data`) first.
    reaches("fill", || gpu(&[2]).add(1.5));
    reaches("binary", || gpu(&[2]).add(gpu(&[2])));
    // Half precision products keep a scalar operand in f32.
    reaches("binary_scalar", || {
        crate::zeros_with([2], (DType::F16, GPU)).mul(2.0)
    });
    reaches("where_", || {
        let c = crate::zeros_with([2], (DType::Bool, GPU));
        crate::where_(&c, gpu(&[2]), gpu(&[2]))
    });
    reaches("triangle", || gpu(&[2, 2]).triu(0));
    reaches("flip", || gpu(&[2, 2]).flip([0]));
    // The optimizers' steps run fused kernels on the parameters' device.
    reaches("fused", || {
        use crate::optim::Optimizer;
        let p = crate::nn::Parameter::new(gpu(&[2]));
        p.set_grad(Some(gpu(&[2])));
        crate::optim::Sgd::new([p], 0.1).step();
    });
}

/// Random sampling on a ROCm device computes the samples there (M4 step 3, item 4). Without the
/// `rocm` feature it stops at the device like every kernel.
#[test]
fn random_sampling() {
    let host_path = !cfg!(feature = "rocm");
    reaches(if host_path { "write_host" } else { "uniform" }, || {
        crate::rand_with([3], GPU)
    });
    reaches(if host_path { "write_host" } else { "normal" }, || {
        crate::randn_with([3], GPU)
    });
    // With rejection (a range that doesn't divide 2^64) and without.
    for (low, high) in [(0, 3), (0, 4)] {
        reaches(if host_path { "write_host" } else { "randint" }, || {
            crate::randint_with(low, high, [3], GPU)
        });
    }
    reaches(if host_path { "write_host" } else { "uniform" }, || {
        gpu(&[2]).uniform_(0.0, 1.0);
    });
}

#[test]
fn reductions_and_matmul() {
    reaches("reduce", || gpu(&[2, 3]).sum());
    reaches("arg_reduce", || gpu(&[2, 3]).argmax_dim(1, false));
    reaches("var", || gpu(&[2, 3]).var(1));
    reaches("softmax", || gpu(&[2, 3]).softmax(1));
    reaches("exclusive_cumprod", || {
        crate::ops::exclusive_cumprod(&gpu(&[2, 3]), 1, false)
    });
    reaches("gemm", || gpu(&[2, 3]).matmul(&gpu(&[3, 4])));
    reaches("gemm", || {
        gpu(&[2, 4]).addmm(&gpu(&[2, 3]), &gpu(&[3, 4]), 1, 2)
    });
    reaches("gemm", || {
        gpu(&[3, 4]).addbmm(&gpu(&[2, 3, 3]), &gpu(&[2, 3, 4]), 1, 2)
    });
    reaches("addr", || gpu(&[2, 3]).addr(&gpu(&[2]), &gpu(&[3]), 1, 2));
    // Without the input (a zero beta), and for every dtype: an elementwise kernel.
    reaches("addr", || {
        let b = |n| crate::zeros_with([n], (DType::Bool, GPU));
        b(1).addr(&b(2), &b(3), false, true)
    });
    // Half precision products come with M4 step 5b.
    let half = || crate::zeros_with([2, 2], (DType::F16, GPU));
    let message = panic_message(|| half().matmul(&half()));
    let expected = if cfg!(feature = "rocm") {
        "gemm is not implemented on ROCm yet (rocm:0)"
    } else {
        "homura was built without ROCm support; enable the `rocm` feature"
    };
    assert_eq!(message, expected);
}

#[test]
fn indexing_and_pooling() {
    reaches("index_select", || {
        drop(gpu(&[3, 2]).index_select(0, &gpu_index(2)))
    });
    // Advanced indexing copies CPU indices to the GPU first (a snapshot), then checks them and
    // computes the positions there; GPU indices are checked there directly.
    reaches(
        if cfg!(feature = "rocm") {
            "upload"
        } else {
            "index_positions"
        },
        || drop(gpu(&[3, 2]).index(cpu_index(&[0, 2]))),
    );
    reaches("index_positions", || gpu(&[3, 2]).index(gpu_index(2)));
    reaches("mask_coordinates", || {
        gpu(&[3, 2]).index(crate::zeros_with([3, 2], (DType::Bool, GPU)))
    });
    reaches("gather", || gpu(&[3]).gather(0, &gpu_index(2)));
    // index_add and scatter_add copy `self` first; the kernels themselves come after.
    reaches("copy", || {
        drop(gpu(&[3]).index_add(0, &gpu_index(2), &gpu(&[2])))
    });
    reaches("copy", || {
        drop(gpu(&[3]).scatter_add(0, &gpu_index(2), &gpu(&[2])))
    });
    // Made inside the closures: without the `rocm` feature, making them panics already.
    let operands = || (gpu(&[3]).data(), gpu_index(2).data(), gpu(&[2]).data());
    reaches("index_add", || {
        let (out, index, src) = operands();
        super::index_add(&out, 0, &index, &src, super::Bounds::NonNegative)
    });
    reaches("index_overwrite", || {
        let (out, index, src) = operands();
        super::index_overwrite(&out, 0, &index, &src)
    });
    reaches("scatter_add", || {
        let (out, index, src) = operands();
        super::scatter_add(&out, 0, &index, &src)
    });
    reaches("out_of_range", || {
        drop(F::embedding(&gpu_index(2), &gpu(&[3, 2]), None))
    });
    reaches("out_of_range", || {
        drop(F::nll_loss(
            &gpu(&[2, 3]),
            &gpu_index(2),
            None,
            -100,
            F::Reduction::Mean,
        ))
    });
    let pool = MaxPool2dOptions::new([2, 2]);
    reaches("max_pool2d", || {
        drop(F::max_pool2d(&gpu(&[1, 1, 4, 4]), pool))
    });
    reaches("adaptive_avg_pool2d", || {
        drop(F::adaptive_avg_pool2d(&gpu(&[1, 1, 4, 4]), [2, 2]))
    });
    reaches("max_pool2d_backward", || {
        let indices = crate::zeros_with([1, 1, 2, 2], (DType::I64, GPU));
        let input = gpu(&[1, 1, 4, 4]);
        let g = F::max_pool2d_geometry(&input, &pool);
        drop(crate::ops::max_pool2d_backward(
            &gpu(&[1, 1, 2, 2]),
            &input,
            &indices,
            &g,
        ))
    });
    reaches("adaptive_avg_pool2d_backward", || {
        drop(crate::ops::adaptive_avg_pool2d_backward(
            &gpu(&[1, 1, 2, 2]),
            &gpu(&[1, 1, 4, 4]),
        ))
    });
    reaches("unfold_backward", || {
        drop(crate::ops::unfold_backward(
            &gpu(&[3, 2]),
            &crate::Shape::new(&[4]),
            0,
            2,
            1,
        ))
    });
}

/// Copies and conversions between the CPU and ROCm pass the device-pair checks and reach the
/// transfers (M4 step 2b); only data out of meta tensors is refused.
#[test]
fn device_pairs_reach_the_transfers() {
    reaches("copy", || {
        gpu(&[2]).copy_(&gpu(&[2]));
    });
    reaches("copy", || gpu(&[2]).to(DType::F16));
    reaches("upload", || crate::zeros([2]).to(GPU));
    reaches("upload", || {
        gpu(&[2]).copy_(&crate::zeros([2]));
    });
    reaches("download", || gpu(&[2]).to(Device::Cpu));
    reaches("download", || {
        crate::zeros([2]).copy_(&gpu(&[2]));
    });
    let m = panic_message(|| {
        gpu(&[2]).copy_(&crate::zeros_with([2], Device::Meta));
    });
    assert_eq!(m, "Cannot copy out of meta tensor; no data!");
}

#[test]
fn reading_data() {
    reaches("read_host", || format!("{:?}", gpu(&[2])));
    reaches("read_host", || gpu(&[2]).to_vec::<f32>());
    reaches("read_host", || gpu(&[]).item::<f32>());
}
