//! A ROCm device's first use (its context: stream and probe) can run out of memory, and the
//! `try_` functions return that (M4.5 step 2b, item 7). A binary of its own with one test, so
//! that nothing else creates the context first. Needs a ROCm GPU; built only with the `rocm`
//! feature.
#![cfg(feature = "rocm")]

use homura::{DType, Device, Error};

unsafe extern "C" {
    /// The ROCm layer's testing hook: the calling thread's next `n` stream creations fail with
    /// `HMR_OUT_OF_MEMORY` (csrc/homura_rocm.h).
    fn hmr_test_fail_stream_create(n: i32);
}

const GPU: Device = Device::Rocm(0);

fn is_out_of_memory(e: Error) -> bool {
    matches!(e, Error::OutOfMemory(e) if e.device() == GPU)
}

#[test]
fn a_failed_first_use_is_out_of_memory_and_is_retried() {
    // SAFETY: a testing hook without preconditions.
    unsafe { hmr_test_fail_stream_create(3) };
    assert!(is_out_of_memory(
        homura::try_zeros_with([4], GPU).unwrap_err()
    ));
    let x = homura::arange(4);
    assert!(is_out_of_memory(x.try_to(GPU).unwrap_err()));
    assert!(is_out_of_memory(x.try_to((DType::F16, GPU)).unwrap_err()));
    assert_eq!(GPU.memory_stats().allocated, 0);
    // The failure isn't cached: the next use creates the context.
    let y = x.try_to(GPU).unwrap();
    assert_eq!(y.to_vec::<i64>(), [0, 1, 2, 3]);
    assert_eq!(x.to_vec::<i64>(), [0, 1, 2, 3]);
}
