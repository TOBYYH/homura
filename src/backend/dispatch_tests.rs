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

/// Asserts that `f` panics in the ROCm arm of the dispatch layer, in `kernel`.
#[track_caller]
fn reaches<R>(kernel: &str, f: impl FnOnce() -> R) {
    let message = panic_message(f);
    let expected = if cfg!(feature = "rocm") {
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
}

#[test]
fn reductions_and_matmul() {
    reaches("reduce", || gpu(&[2, 3]).sum());
    reaches("arg_reduce", || gpu(&[2, 3]).argmax_dim(1, false));
    reaches("var", || gpu(&[2, 3]).var(1));
    reaches("softmax", || gpu(&[2, 3]).softmax(1));
    reaches("matmul", || gpu(&[2, 3]).matmul(&gpu(&[3, 4])));
}

#[test]
fn indexing_and_pooling() {
    reaches("index_select", || {
        drop(gpu(&[3, 2]).index_select(0, &gpu_index(2)))
    });
    // Advanced indexing with CPU indices reads them, then gathers on the GPU; GPU indices are
    // read from the GPU.
    reaches("index_select", || {
        drop(gpu(&[3, 2]).index(cpu_index(&[0, 2])))
    });
    reaches("read_host", || gpu(&[3, 2]).index(gpu_index(2)));
    reaches("read_host", || gpu(&[3]).gather(0, &gpu_index(2)));
    reaches("read_host", || {
        drop(gpu(&[3]).index_add(0, &gpu_index(2), &gpu(&[2])))
    });
    reaches("read_host", || {
        drop(gpu(&[3]).scatter_add(0, &gpu_index(2), &gpu(&[2])))
    });
    let pool = MaxPool2dOptions::new([2, 2]);
    reaches("max_pool2d", || {
        drop(F::max_pool2d(&gpu(&[1, 1, 4, 4]), pool))
    });
    reaches("adaptive_avg_pool2d", || {
        drop(F::adaptive_avg_pool2d(&gpu(&[1, 1, 4, 4]), [2, 2]))
    });
}

/// Copies and conversions involving ROCm are refused by the device-pair checks until the
/// transfers exist (M4 step 2b), before any kernel.
#[test]
fn device_pairs_are_refused() {
    let m = panic_message(|| {
        gpu(&[2]).copy_(&gpu(&[2]));
    });
    assert_eq!(m, "copy from rocm:0 to rocm:0 is not supported");
    let m = panic_message(|| gpu(&[2]).to(DType::F16));
    assert_eq!(m, "copying from rocm:0 to rocm:0 is not supported yet");
    let m = panic_message(|| crate::zeros([2]).to(GPU));
    assert_eq!(m, "copying from cpu to rocm:0 is not supported yet");
}

#[test]
fn reading_data() {
    reaches("read_host", || format!("{:?}", gpu(&[2])));
    let m = panic_message(|| gpu(&[2]).to_vec::<f32>());
    assert_eq!(m, "to_vec: reading data from rocm:0 is not supported yet");
}
