//! Tests of the ROCm layer itself. They need a ROCm GPU.

use std::ffi::CString;
use std::mem::{align_of, offset_of, size_of};
use std::sync::{Arc, Barrier, Mutex};

use super::context::{TestSlot, create_in};
use super::*;

#[test]
fn the_abi_matches_the_library() {
    // SAFETY: all-zero bytes are a valid `hmr_abi_info`.
    let mut abi: ffi::hmr_abi_info = unsafe { std::mem::zeroed() };
    // SAFETY: `abi` is a valid output location.
    unsafe { ffi::hmr_abi_check(&mut abi) };
    type P = ffi::hmr_device_properties;
    assert_eq!(abi.abi_info_size as usize, size_of::<ffi::hmr_abi_info>());
    assert_eq!(abi.properties_size as usize, size_of::<P>());
    assert_eq!(abi.properties_align as usize, align_of::<P>());
    let offsets = [
        offset_of!(P, name),
        offset_of!(P, arch),
        offset_of!(P, multiprocessor_count),
        offset_of!(P, warp_size),
        offset_of!(P, total_memory),
    ];
    assert_eq!(abi.properties_offsets.map(|o| o as usize), offsets);
    let statuses = [
        ffi::HMR_OK,
        ffi::HMR_OUT_OF_MEMORY,
        ffi::HMR_INVALID,
        ffi::HMR_HIP_ERROR,
        ffi::HMR_INTERNAL_ERROR,
    ];
    assert_eq!(abi.status_values, statuses);
    assert_eq!(
        (abi.name_len, abi.arch_len),
        (ffi::HMR_NAME_LEN, ffi::HMR_ARCH_LEN)
    );
}

#[test]
fn the_library_was_built_for_the_configured_architectures() {
    // SAFETY: a static NUL-terminated string.
    let archs = unsafe { CStr::from_ptr(ffi::hmr_built_archs()) };
    // The list `build.rs` compiled for, after trimming and removing duplicates, not the raw
    // `HOMURA_OFFLOAD_ARCH` (review round 118, R138).
    assert_eq!(archs.to_str().unwrap(), env!("HOMURA_ROCM_OFFLOAD_ARCHS"));
    if option_env!("HOMURA_OFFLOAD_ARCH").is_none() {
        assert_eq!(archs.to_str().unwrap(), "gfx1201");
    }
    // An expectation independent of `build.rs`, which tools/check_rocm_build.py sets for the
    // lists it builds with (review round 119, R141): read when the test runs, not compiled in.
    if let Ok(expected) = std::env::var("HOMURA_TEST_EXPECTED_ARCHS") {
        assert_eq!(archs.to_str().unwrap(), expected);
    }
}

#[test]
fn devices_and_their_properties() {
    let n = device_count();
    assert!(n >= 1, "these tests need a ROCm GPU");
    for i in 0..n as u32 {
        let p = properties(i);
        assert!(!p.name.is_empty());
        assert!(p.arch.starts_with("gfx12"), "{}", p.arch);
        assert!(!p.arch.contains(':'), "{}", p.arch);
        assert_eq!(p.warp_size, 32);
        assert!(p.compute_units > 0 && p.compute_units.is_multiple_of(2));
        let m = mem_get_info(i);
        assert_eq!(m.total, p.total_memory);
        assert!(0 < m.free && m.free <= m.total);
    }
}

#[test]
fn a_missing_device_is_an_error_with_a_message() {
    // SAFETY: all-zero bytes are a valid `hmr_device_properties`.
    let mut p: ffi::hmr_device_properties = unsafe { std::mem::zeroed() };
    // SAFETY: `p` is a valid output location.
    let f = call(unsafe { ffi::hmr_device_properties_get(99, &mut p) }).unwrap_err();
    assert_eq!(f.status, ffi::HMR_INVALID);
    let n = device_count();
    let s = if n == 1 { "" } else { "s" };
    assert_eq!(
        f.message,
        format!("rocm:99 is not available: {n} ROCm device{s} found")
    );
}

#[test]
fn error_messages_are_per_thread() {
    let barrier = Arc::new(Barrier::new(2));
    let threads: Vec<_> = ["first", "second"]
        .into_iter()
        .map(|text| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let message = CString::new(text).unwrap();
                // SAFETY: a NUL-terminated string.
                let status = unsafe { ffi::hmr_test_fail(message.as_ptr()) };
                // Both threads have failed before either reads its message.
                barrier.wait();
                let f = call(status).unwrap_err();
                assert_eq!((f.status, f.message.as_str()), (ffi::HMR_INVALID, text));
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

fn current_device() -> i32 {
    let mut d = -1;
    // SAFETY: `d` is a valid output location.
    call(unsafe { ffi::hmr_test_current_device(&mut d) }).unwrap();
    d
}

#[test]
fn entry_points_leave_the_current_device_alone() {
    // A new thread works on its first call, whatever its current device.
    std::thread::spawn(|| {
        let before = current_device();
        synchronize(0);
        properties(0);
        mem_get_info(0);
        exec(0, || ());
        assert_eq!(current_device(), before);
    })
    .join()
    .unwrap();
    // With a single GPU there is no other device to switch to: the guard must not fail or
    // change anything either way.
    if device_count() > 1 {
        std::thread::spawn(|| {
            // SAFETY: plain HIP call.
            call(unsafe { ffi::hmr_test_set_current_device(1) }).unwrap();
            synchronize(0);
            mem_get_info(0);
            assert_eq!(current_device(), 1);
        })
        .join()
        .unwrap();
    }
}

fn live_streams() -> i32 {
    // SAFETY: reads a counter of this thread.
    unsafe { ffi::hmr_test_live_streams() }
}

#[test]
fn a_failed_context_creation_is_not_cached() {
    let slot: TestSlot = Mutex::new(None);
    // SAFETY: affects this thread's next stream creation only.
    unsafe { ffi::hmr_test_fail_stream_create(1) };
    let f = create_in(&slot, 0).unwrap_err();
    assert_eq!(f.status, ffi::HMR_OUT_OF_MEMORY);
    assert!(f.message.contains("injected for testing"), "{}", f.message);
    assert!(slot.lock().unwrap().is_none());
    let ctx = create_in(&slot, 0).unwrap();
    assert_eq!(ctx.index(), 0);
    ctx.synchronize();
    // The second call finds it.
    assert!(Arc::ptr_eq(&create_in(&slot, 0).unwrap(), &ctx));
}

/// A failure after the stream was created and the probe kernel ran (review round 118, R136)
/// destroys the stream, on the creating device; a retry then succeeds, and destroying that
/// context destroys its stream too.
#[test]
fn a_late_failure_destroys_the_stream() {
    let slot: TestSlot = Mutex::new(None);
    let before = live_streams();
    let device = current_device();
    // SAFETY: affects this thread's next context creation only.
    unsafe { ffi::hmr_test_fail_after_probe(1) };
    let f = create_in(&slot, 0).unwrap_err();
    assert_eq!(
        (f.status, f.message.as_str()),
        (ffi::HMR_OUT_OF_MEMORY, "host allocation failed")
    );
    assert!(slot.lock().unwrap().is_none());
    assert_eq!(live_streams(), before);
    assert_eq!(current_device(), device);
    let ctx = create_in(&slot, 0).unwrap();
    assert_eq!(live_streams(), before + 1);
    ctx.synchronize();
    drop(ctx);
    *slot.lock().unwrap() = None;
    assert_eq!(live_streams(), before);
}

/// A failure after the context exists but before it is registered (the architecture query for
/// the warning) is returned, not raised, and destroys the stream (review round 120, R142).
#[test]
fn a_failure_after_creation_destroys_the_context() {
    let slot: TestSlot = Mutex::new(None);
    let before = live_streams();
    let device = current_device();
    // SAFETY: affects this thread's next property query only.
    unsafe { ffi::hmr_test_fail_properties(1) };
    let f = create_in(&slot, 0).unwrap_err();
    assert_eq!(
        (f.status, f.message.as_str()),
        (
            ffi::HMR_HIP_ERROR,
            "hipGetDeviceProperties(&p, device): invalid resource handle (injected for testing)"
        )
    );
    assert!(slot.lock().unwrap().is_none());
    assert_eq!(live_streams(), before);
    assert_eq!(current_device(), device);
    let ctx = create_in(&slot, 0).unwrap();
    assert_eq!(live_streams(), before + 1);
    assert!(Arc::ptr_eq(&ctx, slot.lock().unwrap().as_ref().unwrap()));
    ctx.synchronize();
    drop(ctx);
    *slot.lock().unwrap() = None;
    assert_eq!(live_streams(), before);
}

#[test]
fn the_execution_lock_is_reentrant_per_device() {
    let value = exec(0, || exec(0, || 7));
    assert_eq!(value, 7);
    // The CPU lock comes first in the lock order.
    assert_eq!(crate::backend::cpu::exec(|| exec(0, || 8)), 8);
}

#[cfg(debug_assertions)]
#[test]
fn taking_the_cpu_lock_inside_a_rocm_lock_is_a_lock_order_violation() {
    let r = std::panic::catch_unwind(|| exec(0, || crate::backend::cpu::exec(|| ())));
    let message = *r.unwrap_err().downcast::<String>().unwrap();
    assert!(message.contains("lock order violation"), "{message}");
    // Nothing stays held: both locks work afterwards.
    exec(0, || ());
    crate::backend::cpu::exec(|| ());
}
