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
    type D = ffi::hmr_tensor_desc;
    assert_eq!(abi.desc_size as usize, size_of::<D>());
    assert_eq!(abi.desc_align as usize, align_of::<D>());
    let offsets = [
        offset_of!(D, data),
        offset_of!(D, offset),
        offset_of!(D, dtype),
        offset_of!(D, ndim),
        offset_of!(D, sizes),
        offset_of!(D, strides),
    ];
    assert_eq!(abi.desc_offsets.map(|o| o as usize), offsets);
    // The dtype codes are homura's DType order.
    let codes: Vec<i32> = crate::DType::ALL.iter().map(|&d| d as i32).collect();
    assert_eq!(abi.dtype_values.to_vec(), codes);
    assert_eq!(abi.max_dims, ffi::HMR_MAX_DIMS);
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

/// The memory tests share rocm:0's allocator; they run one at a time so that each sees its own
/// effect on the statistics and the cache.
static MEMORY: Mutex<()> = Mutex::new(());

fn memory_lock() -> std::sync::MutexGuard<'static, ()> {
    MEMORY.lock().unwrap_or_else(|e| e.into_inner())
}

fn rocm_storage(nbytes: usize) -> crate::storage::Storage {
    crate::storage::Storage::new(nbytes, crate::Device::Rocm(0))
}

fn read_back(s: &crate::storage::Storage) -> Vec<u8> {
    let b = s.rocm_buffer();
    let mut out = vec![0x5a; b.nbytes()];
    b.context().copy_to_host(&mut out, b.ptr()).unwrap();
    out
}

/// New storage reads as zero, also when its block is one that earlier data was written to.
#[test]
fn rocm_storage_is_zeroed_also_when_its_block_is_reused() {
    let _lock = memory_lock();
    let first = rocm_storage(4096);
    let (ptr, size) = (first.rocm_buffer().ptr(), first.rocm_buffer().block_size());
    assert_eq!(size, 4096);
    assert_eq!(read_back(&first), vec![0; 4096]);
    let b = first.rocm_buffer();
    b.context().copy_from_host(b.ptr(), &[0xab; 4096]).unwrap();
    assert_eq!(read_back(&first), vec![0xab; 4096]);
    drop(first);
    // 4000 bytes round to the same 4 KiB, which the freed block fits best.
    let second = rocm_storage(4000);
    assert_eq!(second.rocm_buffer().ptr(), ptr);
    assert_eq!(read_back(&second), vec![0; 4000]);
}

#[test]
fn zero_bytes_take_no_block() {
    let _lock = memory_lock();
    let before = memory_stats(0);
    let s = rocm_storage(0);
    assert_eq!((s.nbytes(), s.rocm_buffer().block_size()), (0, 0));
    assert!(s.rocm_buffer().ptr().is_null());
    assert_eq!(memory_stats(0), before);
}

/// When `hipMalloc` reports no memory, the cached segments are freed and the request is tried
/// again.
#[test]
fn a_refused_segment_releases_the_cache_and_retries() {
    let _lock = memory_lock();
    empty_cache(0);
    let base = memory_stats(0).reserved;
    drop(rocm_storage(64 << 20));
    assert_eq!(memory_stats(0).reserved, base + (64 << 20));
    // 128 MiB don't fit the cached 64 MiB block: a new segment is needed, whose first hipMalloc
    // fails. The cached segment is freed, and the second hipMalloc succeeds.
    memory::inject_malloc_failures(1);
    let s = rocm_storage(128 << 20);
    assert_eq!(memory_stats(0).reserved, base + (128 << 20));
    assert!(memory_stats(0).allocated >= 128 << 20);
    drop(s);
    empty_cache(0);
    assert_eq!(memory_stats(0).reserved, base);
}

/// A request larger than the device is refused (not a panic from `try_new`), and leaves the
/// device usable and the statistics as they were.
#[test]
fn a_request_larger_than_the_device_is_out_of_memory() {
    let _lock = memory_lock();
    let total = properties(0).total_memory as usize;
    let before = memory_stats(0);
    let r = crate::storage::Storage::try_new(2 * total, crate::Device::Rocm(0));
    assert_eq!(
        r.map(drop),
        Err(crate::storage::AllocError::OutOfMemory {
            device: crate::Device::Rocm(0),
            requested: 2 * total
        })
    );
    let after = memory_stats(0);
    assert_eq!(
        (after.allocated, after.peak),
        (before.allocated, before.peak)
    );
    assert_eq!(read_back(&rocm_storage(512)), vec![0; 512]);
    // The refused hipMalloc doesn't linger as this thread's last HIP error, which the probe of a
    // new context (a kernel launch, checked with hipGetLastError) would otherwise report.
    let _ = crate::storage::Storage::try_new(2 * total, crate::Device::Rocm(0));
    let slot: TestSlot = Mutex::new(None);
    let ctx = create_in(&slot, 0).unwrap();
    ctx.synchronize();
}

/// Queuing the zeroing of a new buffer can run out of memory too: the allocation reports it like
/// any other, the block goes back to the cache, and the next allocation works (review round 122,
/// R146).
#[test]
fn a_zeroing_that_runs_out_of_memory_is_out_of_memory() {
    let _lock = memory_lock();
    drop(rocm_storage(8192));
    let before = memory_stats(0);
    memory::inject_memset_failures(1);
    let r = crate::storage::Storage::try_new(8192, crate::Device::Rocm(0));
    assert_eq!(
        r.map(drop),
        Err(crate::storage::AllocError::OutOfMemory {
            device: crate::Device::Rocm(0),
            requested: 8192
        })
    );
    assert_eq!(memory_stats(0).allocated, before.allocated);
    assert_eq!(memory_stats(0).reserved, before.reserved);
    let s = rocm_storage(8192);
    assert_eq!(memory_stats(0).reserved, before.reserved);
    assert_eq!(read_back(&s), vec![0; 8192]);
}

/// The injected failures of `alloc_faults` apply to real ROCm allocations too.
#[test]
fn injected_allocation_failures_reach_rocm() {
    let _lock = memory_lock();
    crate::storage::alloc_faults::fail_nth(crate::Device::Rocm(0), 2);
    let first = crate::storage::Storage::try_new(512, crate::Device::Rocm(0));
    let second = crate::storage::Storage::try_new(512, crate::Device::Rocm(0));
    assert!(first.is_ok());
    assert!(matches!(
        second,
        Err(crate::storage::AllocError::OutOfMemory { requested: 512, .. })
    ));
}

/// Allocating zeroes the memory inside the device's execution lock, which may already be held
/// by this thread (reentrant), and after the CPU lock (the lock order).
#[test]
fn allocation_follows_the_lock_order() {
    let _lock = memory_lock();
    let a = exec(0, || rocm_storage(512));
    let b = crate::backend::cpu::exec(|| rocm_storage(512));
    assert_eq!((read_back(&a), read_back(&b)), (vec![0; 512], vec![0; 512]));
}

/// Dimensions merge as the CPU's `Plan` merges them.
#[test]
fn kernel_dimensions_merge_like_the_cpu_plan() {
    use super::copy::merge_dims;
    // Contiguous: one dimension; size-1 dimensions dropped.
    assert_eq!(
        merge_dims(&[2, 1, 3, 4], [&[12, 12, 4, 1]]),
        Some(vec![(24, [1])])
    );
    // Transposed in one operand only: nothing merges.
    assert_eq!(
        merge_dims(&[2, 3], [&[3, 1], &[1, 2]]),
        Some(vec![(2, [3, 1]), (3, [1, 2])])
    );
    // Broadcast (stride 0) merges with stride 0 neighbours.
    assert_eq!(merge_dims(&[4, 3, 5], [&[0, 0, 0]]), Some(vec![(60, [0])]));
    assert_eq!(merge_dims(&[0, 3], [&[3, 1]]), None);
    assert_eq!(merge_dims::<1>(&[], [&[]]), Some(vec![]));
    // `stride * size` doesn't fit in `usize`: not merged.
    let half = usize::MAX / 2 + 1;
    assert_eq!(merge_dims(&[2, 2], [&[0, half]]).unwrap().len(), 2);
}

/// More than 25 dimensions that can't be merged are refused, with the operation's name.
#[test]
#[should_panic(
    expected = "copy on ROCm supports at most 25 dimensions that can't be merged, got 26"
)]
fn too_many_dimensions_are_refused() {
    let _lock = memory_lock();
    let x = crate::zeros_with(vec![2usize; 26], crate::Device::Rocm(0));
    // Every other dimension reversed: no two neighbours are contiguous.
    let perm: Vec<i64> = (0..26).rev().collect();
    let _ = x.permute(perm).contiguous();
}

fn assert_out_of_memory(result: crate::Result<crate::Tensor>, device: crate::Device) {
    match result {
        Err(crate::Error::OutOfMemory(e)) => assert_eq!(e.device(), device),
        Err(e) => panic!("not out of memory: {e}"),
        Ok(_) => panic!("no failure"),
    }
}

/// `try_to` returns the refusal of every allocation it makes (the target, the contiguous copy on
/// the source device, the host staging of a conversion), leaves the source unchanged and leaks
/// nothing (M4.5 step 2b, item 7).
#[test]
fn try_to_returns_every_refused_allocation() {
    use crate::storage::alloc_faults::{clear, fail_nth};
    use crate::{DType, Device};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    let before = memory_stats(0).allocated;
    // A strided CPU tensor.
    let x = crate::arange(12).reshape([3, 4]).transpose(0, 1);
    let values = x.to_vec::<i64>();
    // CPU -> GPU: the target, then the contiguous copy (or the cast) on the CPU.
    fail_nth(gpu, 1);
    assert_out_of_memory(x.try_to(gpu), gpu);
    fail_nth(Device::Cpu, 1);
    assert_out_of_memory(x.try_to(gpu), Device::Cpu);
    fail_nth(Device::Cpu, 1);
    assert_out_of_memory(x.try_to((DType::F32, gpu)), Device::Cpu);
    clear();
    // GPU -> CPU: the target, then the contiguous copy on the GPU; with a cast, the staging.
    let g = x.to(gpu).transpose(0, 1);
    let g_values = x.transpose(0, 1).to_vec::<i64>();
    assert!(!g.is_contiguous());
    fail_nth(Device::Cpu, 1);
    assert_out_of_memory(g.try_to(Device::Cpu), Device::Cpu);
    fail_nth(gpu, 1);
    assert_out_of_memory(g.try_to(Device::Cpu), gpu);
    fail_nth(Device::Cpu, 2);
    assert_out_of_memory(g.try_to((DType::F16, Device::Cpu)), Device::Cpu);
    // On the GPU: the converted copy.
    fail_nth(gpu, 1);
    assert_out_of_memory(g.try_to(DType::F16), gpu);
    clear();
    // Nothing changed, everything still works, and the failures leaked no memory.
    assert_eq!(x.to_vec::<i64>(), values);
    assert_eq!(g.to_vec::<i64>(), g_values);
    assert_eq!(
        g.try_to((DType::F64, Device::Cpu))
            .unwrap()
            .to_vec::<f64>()
            .len(),
        12
    );
    drop(g);
    assert_eq!(memory_stats(0).allocated, before);
}

/// Between two ROCm devices, data goes through host staging, one device lock at a time. With one
/// GPU, the same device stands for both (the lock-order checker of debug builds would panic if
/// both locks were held).
#[test]
fn transfers_between_rocm_devices_go_through_the_host() {
    use crate::storage::alloc_faults::{clear, fail_nth};
    use crate::{DType, Device, Tensor};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    let src = crate::arange_with(12, (DType::F32, gpu))
        .reshape([3, 4])
        .transpose(0, 1);
    let dst = Tensor::empty_on(src.data().layout.shape.clone(), DType::F32, gpu).data();
    crate::backend::try_transfer_into(&dst, &src.data()).unwrap();
    assert_eq!(
        Tensor::from_data(dst.clone()).to_vec::<f32>(),
        src.to_vec::<f32>()
    );
    // The staging is CPU memory: its refusal is returned.
    fail_nth(Device::Cpu, 1);
    assert!(matches!(
        crate::backend::try_transfer_into(&dst, &src.data()),
        Err(crate::storage::AllocError::OutOfMemory {
            device: Device::Cpu,
            ..
        })
    ));
    clear();
}

/// Makes the calling thread's next `n` data operations of the ROCm layer fail before queueing
/// anything: out of memory, or with another HIP error (`hmr_test_fail_data`).
fn fail_data(n: i32, out_of_memory: bool) {
    // SAFETY: a testing hook without preconditions.
    unsafe { ffi::hmr_test_fail_data(n, i32::from(out_of_memory)) };
}

#[track_caller]
fn assert_out_of_memory_for(result: crate::Result<crate::Tensor>, requested: usize) {
    match result {
        Err(crate::Error::OutOfMemory(e)) => {
            assert_eq!(
                (e.device(), e.requested()),
                (crate::Device::Rocm(0), requested)
            )
        }
        Err(e) => panic!("not out of memory: {e}"),
        Ok(_) => panic!("no failure"),
    }
}

/// The message of the panic of `f`, which must panic.
#[track_caller]
fn panic_message(f: impl FnOnce()) -> String {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("no panic");
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(p) => p
            .downcast_ref::<&str>()
            .map_or_else(String::new, |s| s.to_string()),
    }
}

/// The `try_` functions return the device's lack of memory when the fill or the copy after
/// their allocation fails, not only when the allocation does; nothing leaks, the source stays as
/// it was, and the retry succeeds (review round 124, R148).
#[test]
fn try_functions_return_a_kernel_that_runs_out_of_memory() {
    use crate::{DType, Device};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    let src = crate::arange_with(6, (DType::F32, gpu));
    let strided = src.reshape([2, 3]).transpose(0, 1);
    let before = memory_stats(0).allocated;
    // The fill of a new tensor.
    fail_data(1, true);
    assert_out_of_memory_for(crate::try_ones_with([5], (DType::F32, gpu)), 20);
    fail_data(1, true);
    assert_out_of_memory_for(crate::try_full_with([5], 3.0, (DType::F16, gpu)), 10);
    // A conversion on the GPU.
    fail_data(1, true);
    assert_out_of_memory_for(src.try_to(DType::F64), 48);
    fail_data(1, true);
    assert_out_of_memory_for(strided.try_to(DType::I8), 6);
    // The contiguous copy of a strided GPU tensor on its way to the CPU.
    fail_data(1, true);
    assert_out_of_memory_for(strided.try_to(Device::Cpu), 24);
    // The transfers themselves (HIP's staging).
    fail_data(1, true);
    assert_out_of_memory_for(src.try_to(Device::Cpu), 24);
    fail_data(1, true);
    assert_out_of_memory_for(crate::arange(3).try_to(gpu), 24);
    fail_data(0, false);
    assert_eq!(memory_stats(0).allocated, before);

    assert_eq!(src.to_vec::<f32>(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let ones = crate::try_ones_with([5], (DType::F32, gpu)).unwrap();
    assert_eq!(ones.to_vec::<f32>(), [1.0; 5]);
    let full = crate::try_full_with([5], 3.0, (DType::F16, gpu)).unwrap();
    assert_eq!(full.to_vec::<f32>(), [3.0; 5]);
    let wide = src.try_to(DType::F64).unwrap();
    assert_eq!(wide.to_vec::<f64>(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let host = strided.try_to(Device::Cpu).unwrap();
    assert_eq!(host.to_vec::<f32>(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

/// The entry points that can't return a failure panic for it: running out of memory like a
/// refused allocation, other failures with HIP's message. A failed in-place write changes
/// neither the data nor the version (review round 124, R148 and R149).
#[test]
fn failed_data_operations_panic_and_in_place_writes_keep_their_version() {
    use crate::{DType, Device};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    let t = crate::arange_with(4, (DType::F32, gpu));
    let other = crate::full_with([4], 9.0, (DType::F32, gpu));
    let version = t.data().storage.version();
    for (out_of_memory, text) in [(true, "out of memory"), (false, "invalid resource handle")] {
        fail_data(1, out_of_memory);
        assert!(panic_message(|| drop(crate::ones_with([3], gpu))).contains(text));
        fail_data(1, out_of_memory);
        assert!(panic_message(|| drop(t.to(DType::F64))).contains(text));
        fail_data(1, out_of_memory);
        assert!(
            panic_message(|| {
                t.fill_(5.0);
            })
            .contains(text)
        );
        fail_data(1, out_of_memory);
        assert!(
            panic_message(|| {
                t.copy_(&other);
            })
            .contains(text)
        );
        fail_data(1, out_of_memory);
        assert!(panic_message(|| drop(t.to_vec::<f32>())).contains(text));
        fail_data(1, out_of_memory);
        assert!(panic_message(|| drop(crate::arange_with(3, gpu))).contains(text));
    }
    // Recorded writes: the failure comes back through the recording, which records nothing.
    let x = crate::ones([4]);
    x.requires_grad_(true);
    let y = x.to(gpu);
    for out_of_memory in [true, false] {
        fail_data(1, out_of_memory);
        let _ = panic_message(|| {
            y.copy_(&other);
        });
        fail_data(1, out_of_memory);
        let _ = panic_message(|| {
            y.fill_(2.0);
        });
    }
    fail_data(0, false);
    assert_eq!(y.grad_fn().unwrap().name(), "ToCopyBackward0");
    assert_eq!((y.version(), y.to_vec::<f32>()), (0, vec![1.0; 4]));
    y.copy_(&other);
    assert_eq!(y.grad_fn().unwrap().name(), "CopyBackwards");
    assert_eq!(y.version(), 1);

    assert_eq!(t.data().storage.version(), version);
    assert_eq!(t.to_vec::<f32>(), [0.0, 1.0, 2.0, 3.0]);
    t.fill_(5.0);
    assert_eq!(t.data().storage.version(), version + 1);
    t.copy_(&other);
    assert_eq!(t.data().storage.version(), version + 2);
    assert_eq!(t.to_vec::<f32>(), [9.0; 4]);
}
