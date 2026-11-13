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
    type G = ffi::hmr_gemm_desc;
    assert_eq!(abi.gemm_desc_size as usize, size_of::<G>());
    assert_eq!(abi.gemm_desc_align as usize, align_of::<G>());
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
    crate::storage::Storage::new(nbytes, crate::Device::Rocm(0), crate::storage::Fill::Zero)
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

/// Unset storage queues nothing in release builds (the block keeps what it held) and the poison
/// byte in debug builds; zeroed storage on the same block reads as zero again.
#[test]
fn unset_rocm_storage_is_poisoned_in_debug_builds_and_left_as_it_was_otherwise() {
    use crate::storage::{Fill, Storage};
    let _lock = memory_lock();
    let first = rocm_storage(4096);
    let ptr = first.rocm_buffer().ptr();
    let b = first.rocm_buffer();
    b.context().copy_from_host(b.ptr(), &[0xab; 4096]).unwrap();
    drop(first);
    let unset = Storage::new(4096, crate::Device::Rocm(0), Fill::Unset { poison: 0xcd });
    assert_eq!(unset.rocm_buffer().ptr(), ptr);
    let expected = if cfg!(debug_assertions) { 0xcd } else { 0xab };
    assert_eq!(read_back(&unset), vec![expected; 4096]);
    drop(unset);
    let zeroed = rocm_storage(4096);
    assert_eq!(zeroed.rocm_buffer().ptr(), ptr);
    assert_eq!(read_back(&zeroed), vec![0; 4096]);
}

/// `Tensor::uninit_on` poisons by dtype in debug builds: NaN for floats, -1 / 255 for
/// integers, `true` for Bool (a valid bool byte).
#[test]
fn uninit_tensors_are_poisoned_by_dtype_in_debug_builds() {
    use crate::{DType, Device, Shape, Tensor};
    if !cfg!(debug_assertions) {
        return;
    }
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    let new = |dtype| Tensor::uninit_on(Shape::new(&[3]), dtype, gpu);
    assert!(new(DType::F32).to_vec::<f32>().iter().all(|v| v.is_nan()));
    assert!(new(DType::F64).to_vec::<f64>().iter().all(|v| v.is_nan()));
    assert!(
        new(DType::F16)
            .to_vec::<half::f16>()
            .iter()
            .all(|v| v.is_nan())
    );
    assert!(
        new(DType::BF16)
            .to_vec::<half::bf16>()
            .iter()
            .all(|v| v.is_nan())
    );
    assert_eq!(new(DType::I64).to_vec::<i64>(), vec![-1; 3]);
    assert_eq!(new(DType::I32).to_vec::<i32>(), vec![-1; 3]);
    assert_eq!(new(DType::U8).to_vec::<u8>(), vec![255; 3]);
    assert_eq!(new(DType::Bool).to_vec::<bool>(), vec![true; 3]);
}

/// Creating zeros never relies on what a reused block held: `zeros` and `full(0)` skip the fill
/// on zeroed memory, `full(-0.0)` keeps its sign.
#[test]
fn zeros_are_zero_on_blocks_that_held_data() {
    use crate::{DType, Device};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    for dtype in [DType::F32, DType::F16, DType::I64, DType::U8, DType::Bool] {
        let dirty = crate::full_with([1024], 1, (dtype, gpu));
        let ptr = dirty.data().storage.rocm_buffer().ptr();
        drop(dirty);
        let z = crate::zeros_with([1024], (dtype, gpu));
        assert_eq!(z.data().storage.rocm_buffer().ptr(), ptr, "{dtype}");
        assert!(
            z.to(DType::F64).to_vec::<f64>().iter().all(|&v| v == 0.0),
            "{dtype}"
        );
        drop(z);
        let f = crate::full_with([1024], 0, (dtype, gpu));
        assert!(
            f.to(DType::F64).to_vec::<f64>().iter().all(|&v| v == 0.0),
            "{dtype}"
        );
    }
    let n = crate::full_with([1024], -0.0, (DType::F32, gpu)).to_vec::<f32>();
    assert!(n.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()));
}

/// Non-zero values are written over memory that isn't zeroed first (poisoned in debug builds):
/// every element of a reused block holds the value, and `-0.0` its sign.
#[test]
fn nonzero_creation_overwrites_every_element_of_a_reused_block() {
    use crate::{DType, Device};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    for dtype in [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I64,
        DType::U8,
    ] {
        let dirty = crate::full_with([1000], 7, (dtype, gpu));
        let ptr = dirty.data().storage.rocm_buffer().ptr();
        drop(dirty);
        let t = crate::full_with([1000], 3, (dtype, gpu));
        assert_eq!(t.data().storage.rocm_buffer().ptr(), ptr, "{dtype}");
        assert_eq!(t.to(DType::F64).to_vec::<f64>(), vec![3.0; 1000], "{dtype}");
        drop(t);
        let ones = crate::ones_with([1000], (dtype, gpu));
        assert_eq!(
            ones.to(DType::F64).to_vec::<f64>(),
            vec![1.0; 1000],
            "{dtype}"
        );
    }
    let dirty = crate::full_with([1000], false, (DType::Bool, gpu));
    drop(dirty);
    assert_eq!(
        crate::full_with([1000], true, (DType::Bool, gpu)).to_vec::<bool>(),
        vec![true; 1000]
    );
    drop(crate::full_with([1000], 5.0, (DType::F32, gpu)));
    let n = crate::full_with([1000], -0.0, (DType::F32, gpu)).to_vec::<f32>();
    assert!(n.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()));
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
    let r = crate::storage::Storage::try_new(
        2 * total,
        crate::Device::Rocm(0),
        crate::storage::Fill::Zero,
    );
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
    let _ = crate::storage::Storage::try_new(
        2 * total,
        crate::Device::Rocm(0),
        crate::storage::Fill::Zero,
    );
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
    let r =
        crate::storage::Storage::try_new(8192, crate::Device::Rocm(0), crate::storage::Fill::Zero);
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
    drop(s);
    // Unset storage queues no fill in release builds, so it can't fail this way there.
    memory::inject_memset_failures(1);
    let unset = crate::storage::Fill::Unset { poison: 0xff };
    let r = crate::storage::Storage::try_new(8192, crate::Device::Rocm(0), unset);
    assert_eq!(r.is_err(), cfg!(debug_assertions));
    memory::inject_memset_failures(0);
}

/// The injected failures of `alloc_faults` apply to real ROCm allocations too.
#[test]
fn injected_allocation_failures_reach_rocm() {
    let _lock = memory_lock();
    crate::storage::alloc_faults::fail_nth(crate::Device::Rocm(0), 2);
    let first =
        crate::storage::Storage::try_new(512, crate::Device::Rocm(0), crate::storage::Fill::Zero);
    let second =
        crate::storage::Storage::try_new(512, crate::Device::Rocm(0), crate::storage::Fill::Zero);
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
    let dst = Tensor::uninit_on(src.data().layout.shape.clone(), DType::F32, gpu).data();
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

/// Pseudo-random values of a float dtype on the CPU: a mix of magnitudes, signs, zeros and
/// values that overflow or vanish in F16.
fn mixed_floats(dtype: crate::DType, n: usize, seed: u64) -> crate::Tensor {
    let mut s = seed;
    let v: Vec<f64> = (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let e = ((s >> 3) % 40) as i32 - 20;
            match i % 17 {
                0 => 0.0,
                1 => -0.0,
                2 => 7e4,
                3 => 1e-7,
                _ => (u * 2.0 - 1.0) * 2f64.powi(e),
            }
        })
        .collect();
    crate::Tensor::from_slice(&v, [n]).to(dtype)
}

/// The same bits on both devices (a NaN only has to meet a NaN).
#[track_caller]
fn assert_same_bits(gpu: &crate::Tensor, cpu: &crate::Tensor, what: &str) {
    let (g, c) = (
        gpu.to((crate::DType::F64, crate::Device::Cpu))
            .to_vec::<f64>(),
        cpu.to(crate::DType::F64).to_vec::<f64>(),
    );
    for (i, (a, b)) in g.iter().zip(&c).enumerate() {
        assert!(
            a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
            "{what}: element {i}: gpu {a:e}, cpu {b:e}"
        );
    }
}

/// `binary_scalar` in its three forms, `s / a` included (only the formulas use it), on every
/// float dtype, with scalars that round differently in f32 and in the dtype.
#[test]
fn scalar_products_and_quotients_match_the_cpu() {
    use crate::backend::cpu;
    use crate::ops::BinaryOp;
    use crate::{DType, Device, Tensor};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let a = mixed_floats(dtype, 1031, 1);
        let ag = a.to(gpu);
        for s in [
            2.5,
            1.0 / 3.0,
            1e-5,
            70000.0,
            1e300,
            -0.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            for (op, first) in [
                (BinaryOp::Mul, false),
                (BinaryOp::Div, false),
                (BinaryOp::Div, true),
            ] {
                let (out, outg) = (
                    Tensor::uninit_on(a.shape(), dtype, Device::Cpu),
                    Tensor::uninit_on(a.shape(), dtype, gpu),
                );
                cpu::binary_scalar(op, &out.data(), &a.data(), s, first);
                super::binary_scalar(op, &outg.data(), &ag.data(), s, first);
                assert_same_bits(&outg, &out, &format!("{dtype} {op:?} {s} first={first}"));
            }
        }
    }
}

/// Every fused step on every float dtype, with coefficients on both sides of Lerp's 0.5 and
/// ones that round in the dtype (alpha) or only in f32 (the others): the FMA positions and the
/// coefficients' roundings are the CPU's.
#[test]
fn fused_steps_match_the_cpu() {
    use crate::backend::{Fused, cpu};
    use crate::{DType, Device, Tensor};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let [a, b, c] = [2, 3, 4].map(|seed| mixed_floats(dtype, 1031, seed));
        let [ag, bg, cg] = [&a, &b, &c].map(|t| t.to(gpu));
        let coefficients = [
            0.1,
            -0.3,
            0.49999,
            0.5,
            0.9,
            1.0 + 1e-4,
            -1.7,
            1.0 / 3.0,
            1e-30,
        ];
        for v in coefficients {
            for op in [
                Fused::ScaledAdd(v),
                Fused::Lerp(v),
                Fused::Addcmul(v),
                Fused::Addcdiv(v),
            ] {
                let three = matches!(op, Fused::Addcmul(_) | Fused::Addcdiv(_));
                let out = Tensor::uninit_on(a.shape(), dtype, Device::Cpu);
                let outg = Tensor::uninit_on(a.shape(), dtype, gpu);
                let (cd, cgd) = (c.data(), cg.data());
                cpu::fused(op, &out.data(), &a.data(), &b.data(), three.then_some(&cd));
                super::fused(
                    op,
                    &outg.data(),
                    &ag.data(),
                    &bg.data(),
                    three.then_some(&cgd),
                );
                assert_same_bits(&outg, &out, &format!("{dtype} {op:?}"));
            }
        }
    }
}

/// In-place elementwise operations on the GPU bump the version as often as on the CPU, and an
/// operation whose kernel fails changes neither the data nor the version.
#[test]
fn in_place_elementwise_versions_match_the_cpu() {
    use crate::{DType, Device};
    let _lock = memory_lock();
    for device in [Device::Cpu, Device::Rocm(0)] {
        let t = crate::arange_with(6, (DType::F32, device));
        let u = crate::ones_with([6], device);
        let v0 = t.data().storage.version();
        t.add_(&u).mul_(2.0).sub_(1.0).clamp_(0.0, 5.0);
        let bumps = t.data().storage.version() - v0;
        assert_eq!(bumps, 4, "{device}");
        assert_eq!(t.to_vec::<f32>(), [1.0, 3.0, 5.0, 5.0, 5.0, 5.0]);
    }
    let gpu = Device::Rocm(0);
    let t = crate::arange_with(6, (DType::F32, gpu));
    let v0 = t.data().storage.version();
    fail_data(1, false);
    assert!(
        panic_message(|| {
            t.add_(1.0);
        })
        .contains("invalid resource handle")
    );
    assert_eq!(t.data().storage.version(), v0);
    assert_eq!(t.to_vec::<f32>(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// The shapes of `PLAN_SHAPES` in `tests/rocm_reduce.rs` reach the plans they are meant to:
/// lanes per output, chunks, and grids of more than one round (65536 blocks).
#[test]
fn reduction_plans_cover_every_path() {
    use crate::{DType, Device};
    let _lock = memory_lock();
    let rounds = |shape: &[usize], group: i64, dims: &[i64]| {
        let nout: usize = shape
            .iter()
            .enumerate()
            .filter(|(d, _)| !dims.contains(&(*d as i64)))
            .map(|(_, &n)| n)
            .product();
        nout.div_ceil(256 / group as usize) > 1 << 16
    };
    // (shape, reduced dimensions, lanes per output, chunks, more than one round of the grid)
    type Case = (&'static [usize], &'static [i64], i64, i64, bool);
    let cases: [Case; 11] = [
        (&[300, 512], &[1], 256, 1, false),
        (&[4, 100000], &[1], 256, 24, false),
        (&[9, 2, 3000], &[0, 2], 256, 6, false),
        (&[5000, 100], &[1], 32, 1, false),
        (&[64, 3, 100], &[0, 2], 32, 12, false),
        (&[200000, 2], &[1], 1, 1, false),
        (&[100000, 3], &[0], 1, 4000, false),
        (&[7, 9, 11], &[0, 2], 1, 4, false),
        (&[70000, 256], &[1], 256, 1, true),
        (&[(1 << 24) + 1000, 2], &[1], 1, 1, true),
        (&[5, 1], &[1], 1, 1, false),
    ];
    for (shape, dims, group, splits, many_rounds) in cases {
        let t = crate::Tensor::uninit_on(crate::Shape::new(shape), DType::F32, Device::Rocm(0));
        let mask: Vec<bool> = (0..shape.len())
            .map(|d| dims.contains(&(d as i64)))
            .collect();
        let r = super::reduce::describe("test", &t.data(), &mask).expect("outputs");
        let (g, s, chunk) = super::reduce::plan(&r);
        assert_eq!((g, s), (group, splits), "{shape:?} {dims:?}");
        let nred: usize = shape
            .iter()
            .enumerate()
            .filter(|(d, _)| dims.contains(&(*d as i64)))
            .map(|(_, &n)| n)
            .product();
        assert!(
            chunk * s >= nred as i64 && chunk * (s - 1) < nred as i64,
            "{shape:?}"
        );
        assert_eq!(rounds(shape, group, dims), many_rounds, "{shape:?}");
    }
    // Chunks that cut rows: [9, 2, 3000] folds rows of 3000 in chunks of 4500, [64, 3, 100] rows
    // of 100 in chunks of 534.
    let t = crate::Tensor::uninit_on(
        crate::Shape::new(&[64, 3, 100]),
        DType::F32,
        Device::Rocm(0),
    );
    let r = super::reduce::describe("test", &t.data(), &[true, false, true]).unwrap();
    assert_eq!(super::reduce::plan(&r).2, 534);
}

/// The exclusive cumulative product (only the gradient of `prod` uses it) multiplies each line
/// in the CPU's order: the same bits on every float dtype, layout and direction.
#[test]
fn exclusive_cumprod_matches_the_cpu() {
    use crate::{DType, Device};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let base = mixed_floats(dtype, 2 * 3 * 50, 7).mul(3.0);
        let views = [
            base.reshape([6, 50]),
            base.reshape([50, 6]).transpose(0, 1),
            base.reshape([2, 3, 50]).narrow(2, 1, 40),
            base.narrow(0, 0, 1),
        ];
        for x in views {
            let xg = x.to(gpu);
            for dim in 0..x.num_dims() {
                for reverse in [false, true] {
                    let (c, g) = (
                        crate::ops::exclusive_cumprod(&x, dim, reverse),
                        crate::ops::exclusive_cumprod(&xg, dim, reverse),
                    );
                    assert_eq!(g.device(), gpu);
                    assert_same_bits(
                        &g,
                        &c,
                        &format!("{dtype} {:?} dim {dim} {reverse}", x.shape()),
                    );
                }
            }
        }
    }
}

/// The shapes of `SOFTMAX_PLAN_SHAPES` in `tests/rocm_reduce.rs` reach the softmax plans they
/// are meant to: rows kept in registers (up to 16 elements per lane) or not, for 256, 32 and 1
/// lanes per row, and rows split across blocks.
#[test]
fn softmax_plans_cover_every_path() {
    use crate::{DType, Device};
    let _lock = memory_lock();
    // (shape, dimension, split, lanes per row, elements per lane in registers)
    type Case = (&'static [usize], usize, bool, i32, i32);
    let cases: [Case; 11] = [
        (&[300, 4096], 1, false, 256, 16),
        (&[300, 4097], 1, false, 256, 0),
        (&[3000, 1000], 1, false, 256, 4),
        (&[5000, 100], 1, false, 32, 4),
        (&[5000, 255], 1, false, 32, 8),
        (&[100000, 16], 1, false, 1, 16),
        (&[20, 7], 0, false, 1, 0),
        (&[4, 100000], 1, true, 256, 0),
        (&[100000, 3], 0, true, 1, 0),
        (&[1, 8192], 1, true, 256, 0),
        (&[], 0, false, 1, 1),
    ];
    for (shape, dim, split, group, cache) in cases {
        let t = crate::Tensor::uninit_on(crate::Shape::new(shape), DType::F32, Device::Rocm(0));
        let (s, g, c) = super::reduce::softmax_plan(&t.data(), dim);
        assert_eq!(s, split, "{shape:?} {dim}");
        assert_eq!(g, group, "{shape:?} {dim}");
        if !split {
            assert_eq!(c, cache, "{shape:?} {dim}");
        }
    }
}

/// The host picks the first check that found a position out of range, in the order of the
/// checks, whatever the positions: a slot holds a 64-bit position (`u64::MAX` for none), so
/// positions near 2^63 can't collide with the sentinel (review round 115, R132).
#[test]
fn the_first_check_with_a_violation_wins() {
    let slots = |words: &[u64]| -> Vec<u8> { words.iter().flat_map(|w| w.to_ne_bytes()).collect() };
    let none = u64::MAX;
    let big = (1u64 << 63) - 1;
    let s = slots(&[none, 0, 1 << 60, 7, 0, 9]);
    assert_eq!(
        index::first_violation(&s, &[true, true, true]),
        Some((1, 7))
    );
    // A check that didn't run (an empty index) doesn't count, whatever its slot holds.
    assert_eq!(
        index::first_violation(&s, &[true, false, true]),
        Some((2, 9))
    );
    let s = slots(&[big, -5i64 as u64, 0, 1]);
    assert_eq!(index::first_violation(&s, &[true, true]), Some((0, -5)));
    let s = slots(&[none, 3, none, 4]);
    assert_eq!(index::first_violation(&s, &[true, true]), None);
}

/// The workspace sizes of the sorts (index_add, scatter_add, max_pool2d backward), of
/// overwriting and of masks: exactly the C formula up to the largest size whose byte count fits
/// in an i64, and a failure beyond it rather than a wrapped or negative size (review round 131,
/// R154). The model recomputes the formula in i128.
#[test]
fn workspace_sizes_fail_instead_of_overflowing() {
    let ceil = |n: i128, d: i128| (n + d - 1) / d;
    // 4 words per pair, a count per digit and 2048-pair tile, the scan's words (two per chunk of
    // 4096 counts, and the total).
    let sort = |n: i128| -> i128 {
        if n == 0 {
            return 0;
        }
        let counts = 256 * ceil(n, 2048);
        8 * (4 * n + counts + 2 * ceil(counts, 4096) + 1)
    };
    let mask = |n: i128| -> i128 {
        let tiles = ceil(n, 16384);
        8 * (tiles + 2 * ceil(tiles, 4096) + 1)
    };
    let max = i64::MAX as i128;
    // The largest n whose sort workspace fits.
    let (mut lo, mut hi) = (0i128, max);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if sort(mid) <= max {
            lo = mid
        } else {
            hi = mid - 1
        }
    }
    let largest = lo as i64;
    type Sizing = fn(i64) -> Result<i64, Failure>;
    let sizings: [(&str, Sizing); 3] = [
        ("index_add", |n| {
            let mut b = -1;
            // SAFETY: a valid output location.
            call(unsafe { ffi::hmr_index_add_workspace(n, 0, &mut b) }).map(|()| b)
        }),
        ("scatter_add", |n| {
            let mut b = -1;
            // SAFETY: as above.
            call(unsafe { ffi::hmr_scatter_add_workspace(n, &mut b) }).map(|()| b)
        }),
        ("max_pool2d_backward", |n| {
            let mut b = -1;
            // SAFETY: as above.
            call(unsafe { ffi::hmr_max_pool2d_backward_workspace(n, &mut b) }).map(|()| b)
        }),
    ];
    let too_large = |name: &str, n: i64, r: Result<i64, Failure>| match r {
        Err(f) => assert_eq!(f.message, "the workspace is too large", "{name} {n}"),
        Ok(b) => panic!("{name} {n}: {b} bytes"),
    };
    for (name, sizing) in sizings {
        for n in [
            0,
            1,
            2047,
            2048,
            2049,
            (1 << 20) + 1,
            1 << 40,
            largest - 1,
            largest,
        ] {
            assert_eq!(
                sizing(n).map(i128::from).ok(),
                Some(sort(n.into())),
                "{name} {n}"
            );
        }
        for n in [largest + 1, 1 << 58, 1 << 59, 1 << 61, 1 << 62, i64::MAX] {
            too_large(name, n, sizing(n));
        }
    }
    let overwrite = |size: i64| {
        let mut b = -1;
        // SAFETY: a valid output location.
        call(unsafe { ffi::hmr_index_overwrite_workspace(size, &mut b) }).map(|()| b)
    };
    assert_eq!(overwrite(i64::MAX / 8).ok(), Some(i64::MAX / 8 * 8));
    too_large(
        "index_overwrite",
        i64::MAX / 8 + 1,
        overwrite(i64::MAX / 8 + 1),
    );
    too_large("index_overwrite", i64::MAX, overwrite(i64::MAX));
    for n in [0, 1, 16384, 16385, 1 << 40, i64::MAX] {
        let (mut b, mut total) = (-1, -1);
        // SAFETY: valid output locations.
        call(unsafe { ffi::hmr_mask_workspace(n, &mut b, &mut total) }).unwrap();
        assert_eq!(i128::from(b), mask(n.into()), "mask {n}");
    }
}

/// The value at a logical position of an index tensor, read by the second step of a check, for
/// positions as large as a tensor's logical shape allows (below 2^63), in views of two elements
/// with zero strides; the kernels compute offsets in 64 bits.
#[test]
fn values_are_read_at_large_logical_positions() {
    let _lock = memory_lock();
    use crate::tensor::{Layout, TensorData};
    let base = crate::tensor([11i64, 22]).to(crate::Device::Rocm(0));
    let data = base.data();
    let view = |shape: &[usize], strides: &[usize]| TensorData {
        layout: Layout {
            shape: crate::Shape::new(shape),
            strides: crate::Strides::new(strides),
            offset: 0,
        },
        ..data.clone()
    };
    let cases: [(&[usize], &[usize], u64, i64); 5] = [
        (&[2, 1 << 60], &[1, 0], 1 << 60, 22),
        (&[2, 1 << 60], &[1, 0], (1 << 60) - 1, 11),
        (&[2, 1 << 60], &[1, 0], (1 << 61) - 1, 22),
        (&[1 << 62, 2], &[0, 1], (1 << 63) - 1, 22),
        (&[1 << 62, 2], &[0, 1], (1 << 63) - 2, 11),
    ];
    let slots = crate::Tensor::uninit_on(crate::Shape::new(&[2]), crate::DType::I64, base.device());
    for (shape, strides, position, expected) in cases {
        let index = view(shape, strides);
        let [desc] = expect(copy::descs_with_dims(
            "test",
            [&index],
            [0],
            &shape
                .iter()
                .zip(strides)
                .map(|(&n, &s)| (n, [s as i64]))
                .collect::<Vec<_>>(),
        ));
        write_host(&slots.data(), &[position as i64, 0]);
        let ctx = copy::buffer("test", &data).context();
        let slot = copy::buffer("test", &slots.data()).ptr().cast();
        // SAFETY: the description lies inside `base`'s two elements (zero strides elsewhere);
        // `slot` is two words of device memory.
        expect(call(unsafe {
            ffi::hmr_test_index_value(ctx.raw(), &desc, slot)
        }));
        let got = read_host::<i64>(&slots.data());
        assert_eq!(got, [position as i64, expected], "{shape:?} at {position}");
    }
}

/// The geometry of a max pooling of `input` with these options (no ceil mode).
fn pool_geometry(
    input: &crate::Tensor,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> crate::ops::Pool2d {
    let o = crate::nn::functional::MaxPool2dOptions {
        stride: Some(stride),
        padding,
        dilation,
        ..crate::nn::functional::MaxPool2dOptions::new(kernel)
    };
    crate::nn::functional::max_pool2d_geometry(input, &o)
}

/// The backward of max_pool2d checks the indices on the GPU and reports the first offending one
/// in row-major order, with the CPU's message.
#[test]
fn max_pool2d_backward_checks_its_indices() {
    let _lock = memory_lock();
    for device in [crate::Device::Cpu, crate::Device::Rocm(0)] {
        let grad = crate::ones_with([1, 2, 2, 2], device);
        let input = crate::zeros_with([1, 2, 4, 4], device);
        let indices = crate::tensor([0i64, 5, 16, 3, 1, -1, 2, 99])
            .reshape([1, 2, 2, 2])
            .to(device);
        let g = pool_geometry(&input, [2, 2], [2, 2], [0, 0], [1, 1]);
        let message =
            panic_message(|| drop(crate::ops::max_pool2d_backward(&grad, &input, &indices, &g)));
        assert_eq!(
            message, "max_pool2d index 16 is outside a plane of 16 elements",
            "{device}"
        );
    }
}

/// The backward of max_pool2d takes the gather path for the indices of the forward pass, and the
/// sort path for indices inside the plane but outside their windows, which only
/// internal callers can give; forced on the same input, both paths give the CPU's bits (step
/// 4c). The check and the gather are run with both their coordinate types.
#[test]
fn max_pool2d_backward_chooses_its_path() {
    for wide in [false, true] {
        // SAFETY: a testing hook without preconditions.
        unsafe { ffi::hmr_test_wide_pool(wide as i32) };
        let result = std::panic::catch_unwind(pool_paths);
        // SAFETY: as above.
        unsafe { ffi::hmr_test_wide_pool(0) };
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }
}

fn pool_paths() {
    use super::index::{FORCE_POOL_PATH, LAST_POOL_PATH, PoolPath};
    use crate::{DType, Device, Tensor};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    let bits = |t: &Tensor| t.to(Device::Cpu).to(DType::F64).to_vec::<f64>();
    let run = |grad: &Tensor, x: &Tensor, indices: &Tensor, g: &crate::ops::Pool2d, force| {
        FORCE_POOL_PATH.set(force);
        LAST_POOL_PATH.set(None);
        let out = crate::ops::max_pool2d_backward(grad, x, indices, g);
        FORCE_POOL_PATH.set(None);
        (bits(&out), LAST_POOL_PATH.get())
    };
    let geometries = [
        ([3, 3], [2, 2], [1, 1], [1, 1]),
        ([2, 3], [1, 2], [1, 0], [2, 1]),
        ([3, 3], [2, 2], [0, 0], [2, 2]),
        ([2, 2], [3, 3], [0, 0], [1, 1]),
    ];
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        for (kernel, stride, padding, dilation) in geometries {
            let what = format!("{dtype} {kernel:?} {stride:?} {padding:?} {dilation:?}");
            // Few distinct values: ties, so windows overlap on the same maxima.
            let x = crate::randint_with(0, 3, [2, 3, 11, 13], gpu).to(dtype);
            let g = pool_geometry(&x, kernel, stride, padding, dilation);
            let (y, indices) = crate::ops::max_pool2d_with_indices(&x, g);
            let grad = crate::randn_with(y.shape(), gpu).to(dtype);
            let cpu = crate::ops::max_pool2d_backward(
                &grad.to(Device::Cpu),
                &x.to(Device::Cpu),
                &indices.to(Device::Cpu),
                &g,
            );
            let (chosen, path) = run(&grad, &x, &indices, &g, None);
            assert_eq!(path, Some(PoolPath::Gather), "{what}");
            assert_eq!(chosen, bits(&cpu), "{what}");
            let (sorted, path) = run(&grad, &x, &indices, &g, Some(PoolPath::Sort));
            assert_eq!(path, Some(PoolPath::Sort), "{what}");
            assert_eq!(sorted, chosen, "{what}");
            // The gather writes every element from zero: it doesn't need a zero-filled output.
            let out = crate::full_with(x.shape(), f64::NAN, (dtype, gpu));
            crate::backend::max_pool2d_backward(&out.data(), &grad.data(), &indices.data(), &g);
            assert_eq!(LAST_POOL_PATH.get(), Some(PoolPath::Gather), "{what}");
            assert_eq!(bits(&out), chosen, "{what}: into NaNs");
        }
    }
    // One index inside the plane but outside its window, before or after it along the
    // rows or the columns (the windows of 2 x 2 / 2 over 4 x 4 start at 0, 2, 8 and 10): the
    // check sends it to the sort path, which matches the CPU; gathering would drop it.
    for device in [Device::Cpu, gpu] {
        let x = crate::zeros_with([1, 1, 4, 4], device);
        let g = pool_geometry(&x, [2, 2], [2, 2], [0, 0], [1, 1]);
        let grad = crate::tensor([1.0f32, 2.0, 3.0, 4.0])
            .reshape([1, 1, 2, 2])
            .to(device);
        for picked in [
            [2i64, 2, 8, 10],
            [0, 1, 8, 10],
            [8, 2, 8, 10],
            [0, 2, 4, 10],
        ] {
            let indices = crate::tensor(picked).reshape([1, 1, 2, 2]).to(device);
            let mut expected = vec![0.0; 16];
            for (k, i) in picked.into_iter().enumerate() {
                expected[i as usize] += (k + 1) as f64;
            }
            if device == Device::Cpu {
                let out = crate::ops::max_pool2d_backward(&grad, &x, &indices, &g);
                assert_eq!(bits(&out), expected, "{picked:?}");
                continue;
            }
            let (chosen, path) = run(&grad, &x, &indices, &g, None);
            assert_eq!(path, Some(PoolPath::Sort), "{picked:?}");
            assert_eq!(chosen, expected, "{picked:?}");
            let (gathered, _) = run(&grad, &x, &indices, &g, Some(PoolPath::Gather));
            assert_ne!(
                gathered, expected,
                "{picked:?}: the case must tell the paths apart"
            );
        }
        if device == Device::Cpu {
            continue;
        }
        // Between the positions of a dilated window (0, 2, 4): the gather visits only the
        // windows that cover a position, so this goes to the sort too (review round 135, R159).
        let row = crate::zeros_with([1, 1, 1, 5], device);
        let dilated = pool_geometry(&row, [1, 3], [1, 1], [0, 0], [1, 2]);
        let one = crate::tensor([5.0f32]).reshape([1, 1, 1, 1]);
        let between = crate::tensor([1i64]).reshape([1, 1, 1, 1]);
        let (got, path) = run(&one.to(device), &row, &between.to(device), &dilated, None);
        assert_eq!(path, Some(PoolPath::Sort));
        assert_eq!(got, [0.0, 5.0, 0.0, 0.0, 0.0]);
        let cpu = crate::ops::max_pool2d_backward(&one, &row.to(Device::Cpu), &between, &dilated);
        assert_eq!(bits(&cpu), got);
        let (gathered, _) = run(
            &one.to(device),
            &row,
            &between.to(device),
            &dilated,
            Some(PoolPath::Gather),
        );
        assert_ne!(gathered, got, "the case must tell the paths apart");
        // With an index outside the plane as well, that one is reported, as on the CPU.
        let bad = crate::tensor([15i64, 16, 5, 10])
            .reshape([1, 1, 2, 2])
            .to(device);
        let message = panic_message(|| drop(crate::ops::max_pool2d_backward(&grad, &x, &bad, &g)));
        assert_eq!(
            message,
            "max_pool2d index 16 is outside a plane of 16 elements"
        );
    }
}

/// The gather visits exactly the windows that cover each input position, however large the
/// dilation (review round 135, R159): its own walk, counted on the host over a plane, against the
/// windows of `Pool2d::window` that hold each position. The span of a dilated window would visit
/// 4,328,587,264 pairs for 512 x 512 with kernel 2 and dilation 256, against 262,144.
#[test]
fn max_pool2d_gather_visits_only_covering_windows() {
    let covering = |g: &crate::ops::Pool2d, d: usize, n: usize| -> i64 {
        (0..g.output[d])
            .map(|o| {
                let (first, end) = g.window(d, o, n);
                (first..end).step_by(g.dilation[d]).count() as i64
            })
            .sum()
    };
    let mut cases = vec![
        (
            [512, 512],
            [2, 2],
            [1, 1],
            [0, 0],
            [256, 256],
            Some(262_144),
        ),
        ([64, 64], [3, 3], [1, 1], [0, 0], [16, 16], Some(9_216)),
        ([9, 9], [3, 3], [2, 2], [0, 0], [2, 2], Some(81)),
        ([112, 112], [3, 3], [2, 2], [1, 1], [1, 1], Some(27_889)),
        ([9, 17], [2, 3], [1, 2], [1, 1], [1, 3], None),
        ([20, 30], [5, 4], [3, 2], [2, 1], [7, 1], None),
    ];
    for kernel in [[2, 2], [3, 2], [3, 3]] {
        for stride in [[1, 2], [2, 1], [3, 3]] {
            for padding in [[0, 0], [1, 0], [1, 1]] {
                for dilation in [[1, 1], [2, 1], [3, 4]] {
                    cases.push(([9, 11], kernel, stride, padding, dilation, None));
                }
            }
        }
    }
    let mut counted = 0;
    for (shape, kernel, stride, padding, dilation, expected) in cases {
        let what = format!("{shape:?} {kernel:?} {stride:?} {padding:?} {dilation:?}");
        let x = crate::zeros_with([1, 1, shape[0], shape[1]], crate::Device::Meta);
        let o = crate::nn::functional::MaxPool2dOptions {
            stride: Some(stride),
            padding,
            dilation,
            ..crate::nn::functional::MaxPool2dOptions::new(kernel)
        };
        // Geometries PyTorch rejects, or with windows of only padding, are skipped.
        let Ok(g) = std::panic::catch_unwind(|| crate::nn::functional::max_pool2d_geometry(&x, &o))
        else {
            assert!(expected.is_none(), "{what}: rejected");
            continue;
        };
        let exact = covering(&g, 0, shape[0]) * covering(&g, 1, shape[1]);
        if let Some(e) = expected {
            assert_eq!(exact, e, "{what}");
        }
        let words = super::pool::geometry_words(&g);
        let mut count = 0;
        // SAFETY: valid pointers; a host-only testing hook.
        expect(call(unsafe {
            ffi::hmr_test_pool_candidates(
                words.as_ptr(),
                shape[0] as i64,
                shape[1] as i64,
                g.output[0] as i64,
                g.output[1] as i64,
                &mut count,
            )
        }));
        assert_eq!(count, exact, "{what}");
        counted += 1;
    }
    assert!(counted >= 60, "{counted} geometries counted");
}

/// The coordinates of a mask with more tiles than one chunk of the device scan (4096 tiles of
/// 16384 elements), against a mask whose true elements are known: every 7th.
#[test]
fn masks_beyond_one_chunk_of_the_scan_compact_in_order() {
    let _lock = memory_lock();
    let (rows, cols) = (10_000usize, 7_000usize);
    let mask = crate::zeros_with([rows, cols], (crate::DType::Bool, crate::Device::Rocm(0)));
    mask.reshape([-1, 7]).select(1, 0).fill_(true);
    let coords = mask_coordinates(&mask.data());
    assert_eq!(coords.len(), 2);
    let r = read_host::<i64>(&coords[0]);
    let c = read_host::<i64>(&coords[1]);
    assert_eq!(r.len(), rows * cols / 7);
    for (k, (&r, &c)) in r.iter().zip(&c).enumerate() {
        let linear = 7 * k;
        assert_eq!(
            (r as usize, c as usize),
            (linear / cols, linear % cols),
            "{k}"
        );
    }
}

/// Timings of the indexing kernels on representative shapes (M4.14 item 13: the synchronization
/// of the device checks), printed; run on demand with
/// `cargo test --release --features rocm --lib -- --ignored indexing_timings --nocapture`.
#[test]
#[ignore = "a measurement, run on demand"]
fn indexing_timings() {
    use crate::backend::Bounds;
    use crate::{DType, Device, Shape, Tensor};
    let _lock = memory_lock();
    let gpu = Device::Rocm(0);
    let time = |name: &str, f: &dyn Fn()| {
        for _ in 0..3 {
            f();
        }
        synchronize(0);
        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        synchronize(0);
        let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("{name:<64} {us:>10.1} us");
    };
    let tokens = crate::randint_with(0, 32000, [2048], gpu);
    // Embedding lookup: Llama's table, 2048 tokens.
    let table = crate::zeros_with([32000, 4096], (DType::F16, gpu));
    let picked = Tensor::uninit_on(Shape::new(&[2048, 4096]), DType::F16, gpu);
    for bounds in [Bounds::NonNegative, Bounds::Valid] {
        time(
            &format!("index_select [32000, 4096] f16, 2048 rows, {bounds:?}"),
            &|| {
                crate::backend::index_select(
                    &picked.data(),
                    &table.data(),
                    0,
                    &tokens.data(),
                    bounds,
                )
            },
        );
    }
    time(
        "out_of_range of 2048 tokens (check + read-back only)",
        &|| {
            let _ = crate::backend::out_of_range(&tokens.data(), 0, 32000, None);
        },
    );
    // Its backward: 2048 rows added into the table, f32.
    let table32 = crate::zeros_with([32000, 4096], (DType::F32, gpu));
    let rows = crate::ones_with([2048, 4096], (DType::F32, gpu));
    for bounds in [Bounds::NonNegative, Bounds::Valid] {
        time(
            &format!("index_add 2048 rows into [32000, 4096] f32, {bounds:?}"),
            &|| crate::backend::index_add(&table32.data(), 0, &tokens.data(), &rows.data(), bounds),
        );
    }
    // NLL over Llama's logits: gather and its backward, a scatter_add.
    let logits = crate::zeros_with([2048, 32000], (DType::F32, gpu));
    let target = tokens.reshape([2048, 1]);
    let one = Tensor::uninit_on(Shape::new(&[2048, 1]), DType::F32, gpu);
    time("gather [2048, 32000] f32 at [2048, 1]", &|| {
        crate::backend::gather(&one.data(), &logits.data(), 1, &target.data())
    });
    time("scatter_add [2048, 1] into [2048, 32000] f32", &|| {
        crate::backend::scatter_add(&logits.data(), 1, &target.data(), &one.data())
    });
    // The max pooling of ResNet-50's stem (3x3, stride 2, padding 1; batch 32) and of
    // ResNet-18's at batch 1, and its backward on both paths.
    use super::index::{FORCE_POOL_PATH, PoolPath};
    let o = crate::nn::functional::MaxPool2dOptions {
        stride: Some([2, 2]),
        padding: [1, 1],
        ..crate::nn::functional::MaxPool2dOptions::new([3, 3])
    };
    // And the case of review round 135 (R159): a dilation far larger than the kernel.
    let r159 = crate::nn::functional::MaxPool2dOptions {
        stride: Some([1, 1]),
        dilation: [256, 256],
        ..crate::nn::functional::MaxPool2dOptions::new([2, 2])
    };
    for (shape, o, name) in [
        ([32usize, 64, 112, 112], o, "3x3/2"),
        ([1, 64, 112, 112], o, "3x3/2"),
        ([1, 1, 512, 512], r159, "2x2/1, dilation 256"),
    ] {
        let x = crate::randn_with(shape, gpu);
        let (y, indices) = crate::nn::functional::max_pool2d_with_indices(&x, o);
        let geometry = crate::nn::functional::max_pool2d_geometry(&x, &o);
        time(&format!("max_pool2d {shape:?} f32 {name}"), &|| {
            drop(crate::nn::functional::max_pool2d_with_indices(&x, o))
        });
        let g = crate::randn_with(y.shape(), gpu);
        for (path, name) in [
            (PoolPath::Gather, "check, gather"),
            (PoolPath::Sort, "check, sort, add"),
        ] {
            FORCE_POOL_PATH.set(Some(path));
            time(&format!("  its backward ({name})"), &|| {
                drop(crate::ops::max_pool2d_backward(&g, &x, &indices, &geometry))
            });
            FORCE_POOL_PATH.set(None);
        }
    }
    time("adaptive_avg_pool2d [32, 2048, 7, 7] -> [2, 3]", &|| {
        drop(crate::nn::functional::adaptive_avg_pool2d(
            &crate::zeros_with([32, 2048, 7, 7], gpu),
            [2, 3],
        ))
    });
    // Masks of 2^20 to 2^26 elements, half true.
    for e in [20, 24, 26] {
        let m = crate::rand_with([1usize << e], gpu).less(0.5);
        time(&format!("mask coordinates of 2^{e} elements"), &|| {
            drop(mask_coordinates(&m.data()))
        });
    }
}

/// The description of `x @ y` (3-D GPU tensors) for the library.
fn gemm_desc(x: &crate::Tensor, y: &crate::Tensor) -> ffi::hmr_gemm_desc {
    let out = crate::Tensor::uninit_on(
        crate::Shape::new(&[x.size(0), x.size(1), y.size(2)]),
        x.dtype(),
        x.device(),
    );
    let g = crate::backend::Gemm {
        a: &x.data(),
        b: &y.data(),
        sum_batches: false,
        scale: None,
    };
    super::matmul::describe(&out.data(), &g)
}

#[test]
fn gemm_selection_classifies_the_layouts() {
    let _lock = memory_lock();
    let gpu = crate::Device::Rocm(0);
    let t = |shape: &[usize]| crate::zeros_with(shape.to_vec(), gpu);
    // [B, M, K] and [B, K, N], each stored row-major (N) or transposed (T), or neither.
    let n = |b: usize, r: usize, c: usize| t(&[b, r, c]);
    let tr = |b: usize, r: usize, c: usize| t(&[b, c, r]).transpose(1, 2);
    let strided = |b: usize, r: usize, c: usize| t(&[b, r, c, 2]).select(3, 1);
    let cases = [
        (n(2, 3, 4), n(2, 4, 5), ffi::HMR_GEMM_NN),
        (n(2, 3, 4), tr(2, 4, 5), ffi::HMR_GEMM_NT),
        (tr(2, 3, 4), n(2, 4, 5), ffi::HMR_GEMM_TN),
        (tr(2, 3, 4), tr(2, 4, 5), ffi::HMR_GEMM_TT),
        (strided(2, 3, 4), n(2, 4, 5), ffi::HMR_GEMM_STRIDED),
        (n(2, 3, 4), strided(2, 4, 5), ffi::HMR_GEMM_STRIDED),
        // A dimension of 1 has any stride: a column of B is both N and T, a row of A with a
        // stride of 2 along k is T (not N).
        (tr(1, 3, 4), n(1, 4, 1), ffi::HMR_GEMM_TN),
        // A column of B stored transposed (column stride 4) is N too.
        (n(1, 3, 4), tr(1, 4, 1), ffi::HMR_GEMM_NN),
        (strided(1, 1, 4), tr(1, 4, 5), ffi::HMR_GEMM_TT),
    ];
    for (i, (x, y, layout)) in cases.iter().enumerate() {
        let d = gemm_desc(x, y);
        assert_eq!(super::matmul::select(&d).1, *layout, "case {i}");
    }
    assert_eq!(super::matmul::path_name(ffi::HMR_GEMM_GENERIC), "generic");
}

#[test]
fn gemm_paths_can_be_forced_and_refuse_what_they_dont_take() {
    let _lock = memory_lock();
    let gpu = crate::Device::Rocm(0);
    let x = crate::zeros_with([1, 2, 3], gpu);
    let y = crate::zeros_with([1, 3, 4], gpu);
    let d = gemm_desc(&x, &y);
    let message = |f: &dyn Fn()| {
        let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("panics");
        e.downcast_ref::<String>().cloned().unwrap_or_default()
    };
    super::matmul::force_path(Some(ffi::HMR_GEMM_GENERIC));
    assert_eq!(super::matmul::select(&d).0, ffi::HMR_GEMM_GENERIC);
    super::matmul::force_path(Some(ffi::HMR_GEMM_PATHS));
    let m = message(&|| {
        let _ = super::matmul::select(&d);
    });
    super::matmul::force_path(None);
    assert_eq!(
        m,
        format!(
            "matrix product path {0} can't take this description: unknown path {0}",
            ffi::HMR_GEMM_PATHS
        )
    );
    assert!(message(&|| drop(super::matmul::path_name(ffi::HMR_GEMM_PATHS))).contains("unknown"));
    // Half precision has no path in step 5a; the library refuses it too.
    let mut h = d;
    h.dtype = crate::DType::F16 as i32;
    assert_eq!(
        message(&|| {
            let _ = super::matmul::select(&h);
        }),
        format!(
            "no ROCm matrix product path for dtype {}",
            crate::DType::F16 as i32
        )
    );
}

/// A description of a contiguous F32 product `[batch, m, k] @ [batch, k, n]` in `layout` (only
/// the sizes, strides and alignment matter to the selection; nothing is read).
fn sized_desc(layout: i32, [batch, m, k, n]: [i64; 4]) -> ffi::hmr_gemm_desc {
    // SAFETY: all-zero bytes are a valid description.
    let mut d: ffi::hmr_gemm_desc = unsafe { std::mem::zeroed() };
    d.dtype = crate::DType::F32 as i32;
    (d.batch, d.m, d.n, d.k) = (batch, m, n, k);
    let fake = 256 as *const std::ffi::c_void;
    (d.a, d.b, d.c, d.out) = (fake, fake, fake, fake.cast_mut());
    d.a_strides = match layout {
        ffi::HMR_GEMM_TN | ffi::HMR_GEMM_TT => [m * k, 1, m],
        _ => [m * k, k, 1],
    };
    d.b_strides = match layout {
        ffi::HMR_GEMM_NT | ffi::HMR_GEMM_TT => [k * n, 1, k],
        _ => [k * n, n, 1],
    };
    d.out_strides = [m * n, n, 1];
    d
}

/// The selection rules: rocm-exp's measured preferences (LOG.md, the S series' configuration
/// table) where the S paths take the description, the generic path otherwise.
#[test]
fn gemm_selection_follows_the_measured_rules() {
    use ffi::*;
    let (nn, nt, tn) = (HMR_GEMM_NN, HMR_GEMM_NT, HMR_GEMM_TN);
    let cases = [
        // NN: M <= 16 narrow; else S8a-1, split-K with a long k and 16 < M <= 128 (S7g-1, which
        // rocm-exp chose for 16 < M <= 128 and 8192^3, is slower in this build: step 5a item 5).
        (nn, [1, 1, 4096, 4096], HMR_GEMM_S08C_NARROW_T256),
        (nn, [1, 16, 11008, 4096], HMR_GEMM_S08C_NARROW_T256),
        (nn, [1, 128, 4096, 4096], HMR_GEMM_S08A_S7A5_UNAL),
        (nn, [1, 128, 11008, 4096], HMR_GEMM_S08C_SPLITK_T128),
        (nn, [1, 1000, 1000, 1000], HMR_GEMM_S08A_S7A5_UNAL),
        (nn, [1, 4096, 4096, 4096], HMR_GEMM_S08A_S7A5_UNAL),
        (nn, [1, 8192, 8192, 8192], HMR_GEMM_S08A_S7A5_UNAL),
        // Batches: no split-K (measured with one batch only).
        (nn, [2, 1, 4096, 4096], HMR_GEMM_S08A_S7A5_UNAL),
        (nn, [2, 128, 11008, 4096], HMR_GEMM_S08A_S7A5_UNAL),
        // NT.
        (nt, [1, 1, 4096, 4096], HMR_GEMM_S08B_NARROW_NT),
        (nt, [3, 16, 4096, 11008], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 128, 11008, 4096], HMR_GEMM_S08B_SPLITK_NT_T128),
        (nt, [1, 128, 4096, 4096], HMR_GEMM_S08B_S7G1_NT),
        (nt, [1, 1024, 1024, 1024], HMR_GEMM_S08B_S8A1_NT),
        // The narrow paths need k >= 32 M (NN) or 16 M (TN); NT needs the bound of its table for
        // the r rows of its instance (M rounded up to a power of two) and n < 4096, < 8192 or
        // larger (round 138, R161); below, the layout's tile path.
        (nn, [1, 1, 32, 4096], HMR_GEMM_S08C_NARROW_T256),
        (nn, [1, 1, 31, 4096], HMR_GEMM_S08A_S7A5_UNAL),
        (nn, [1, 4, 128, 4096], HMR_GEMM_S08C_NARROW_T256),
        (nn, [1, 4, 127, 4096], HMR_GEMM_S08A_S7A5_UNAL),
        (nn, [1, 16, 512, 256], HMR_GEMM_S08C_NARROW_T256),
        (nn, [1, 16, 511, 256], HMR_GEMM_S08A_S7A5_UNAL),
        (nt, [1, 1, 16, 4096], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 1, 15, 4096], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 1, 15, 256], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 2, 16, 4096], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 3, 24, 256], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 3, 23, 256], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 3, 40, 4096], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 3, 39, 4096], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 4, 24, 11008], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 4, 23, 11008], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 5, 64, 1024], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 5, 63, 1024], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 5, 80, 4096], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 5, 79, 4096], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 8, 80, 4096], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 9, 80, 4096], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 9, 192, 256], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 9, 144, 256], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 9, 352, 4096], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 9, 144, 4096], HMR_GEMM_S08B_S8A1_NT),
        (nt, [3, 16, 192, 2048], HMR_GEMM_S08B_NARROW_NT),
        (nt, [3, 16, 191, 2048], HMR_GEMM_S08B_S8A1_NT),
        (nt, [3, 16, 352, 4096], HMR_GEMM_S08B_NARROW_NT),
        (nt, [3, 16, 351, 4096], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 16, 256, 4096], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 16, 256, 8191], HMR_GEMM_S08B_S8A1_NT),
        (nt, [1, 16, 256, 8192], HMR_GEMM_S08B_NARROW_NT),
        (nt, [1, 16, 255, 11008], HMR_GEMM_S08B_S8A1_NT),
        (tn, [1, 16, 256, 4096], HMR_GEMM_S08B_NARROW_TN),
        (tn, [1, 16, 255, 4096], HMR_GEMM_S08B_S7G1_TN),
        (tn, [1, 2, 32, 4096], HMR_GEMM_S08B_NARROW_TN),
        (tn, [1, 2, 31, 4096], HMR_GEMM_S08B_S7G1_TN),
        // TN.
        (tn, [1, 16, 4096, 4096], HMR_GEMM_S08B_NARROW_TN),
        (tn, [2, 16, 4096, 4096], HMR_GEMM_S08B_S7G1_TN),
        (tn, [1, 128, 4096, 4096], HMR_GEMM_S08B_S7G1_TN),
        (tn, [1, 2048, 2048, 2048], HMR_GEMM_S08B_S7G1_TN),
        // TT and sizes past the S series' int range: the generic path.
        (HMR_GEMM_TT, [1, 64, 64, 64], HMR_GEMM_GENERIC),
        (nn, [1, 1 << 31, 1, 1], HMR_GEMM_GENERIC),
        (nn, [1, 1, 1 << 31, 1], HMR_GEMM_GENERIC),
        // Only K past the range (TN keeps both leading dimensions small).
        (tn, [1, 2, 1 << 31, 2], HMR_GEMM_GENERIC),
    ];
    for (layout, size, path) in cases {
        let d = sized_desc(layout, size);
        let (got, l) = super::matmul::select(&d);
        assert_eq!(l, layout, "{size:?}");
        assert_eq!(
            super::matmul::path_name(got),
            super::matmul::path_name(path),
            "{size:?}"
        );
    }
    // What the S series doesn't take: other dtypes, k == 0, sums of batches.
    let mut d = sized_desc(nn, [1, 64, 64, 64]);
    d.dtype = crate::DType::F64 as i32;
    assert_eq!(super::matmul::select(&d).0, HMR_GEMM_GENERIC);
    let d = sized_desc(nn, [1, 64, 0, 64]);
    assert_eq!(super::matmul::select(&d).0, HMR_GEMM_GENERIC);
    let mut d = sized_desc(nn, [3, 64, 64, 64]);
    d.sum_batches = 1;
    assert_eq!(super::matmul::select(&d).0, HMR_GEMM_GENERIC);
    d.batch = 1;
    assert_eq!(super::matmul::select(&d).0, HMR_GEMM_S08A_S7A5_UNAL);
}

/// `hmr_gemm_fold` of a description.
fn folded(mut d: ffi::hmr_gemm_desc) -> ffi::hmr_gemm_desc {
    // SAFETY: a valid description; the call only rewrites its sizes and strides.
    super::expect(super::call(unsafe { ffi::hmr_gemm_fold(&mut d) }));
    d
}

/// Batches that share b become the rows of one batch when the rows of a, the output and a read
/// input continue from batch to batch (PyTorch's folding of `[..., m, k] @ [k, n]` into one mm);
/// every other description is left as it is.
#[test]
fn gemm_fold_merges_batches_that_share_b() {
    use ffi::*;
    let size = |d: &hmr_gemm_desc| [d.batch, d.m, d.n, d.k];
    // [4, 3, 5] @ [5, 7] broadcast: one batch of 12 rows, b as it was.
    let mut d = sized_desc(HMR_GEMM_NN, [4, 3, 5, 7]);
    d.b_strides[0] = 0;
    let f = folded(d);
    assert_eq!(size(&f), [1, 12, 7, 5]);
    assert_eq!(
        (&f.a_strides[1..], &f.out_strides[1..]),
        (&[5, 1][..], &[7, 1][..])
    );
    assert_eq!(f.b_strides, d.b_strides);
    // Single rows: the batch stride is the row stride (here a row of a padded batch).
    let mut d = sized_desc(HMR_GEMM_NN, [6, 1, 8, 9]);
    (d.b_strides[0], d.a_strides) = (0, [10, 3, 1]);
    let f = folded(d);
    assert_eq!(size(&f), [1, 6, 9, 8]);
    assert_eq!(f.a_strides[1..], [10, 1][..]);
    // A read input: whole batches, or one row for all (a bias) fold; one [m, n] matrix for all
    // batches doesn't (its rows restart), nor does a description whose input isn't read.
    let mut d = sized_desc(HMR_GEMM_NN, [4, 3, 5, 7]);
    (d.b_strides[0], d.scaled, d.reads_input) = (0, 1, 1);
    d.c_strides = [21, 7, 1];
    assert_eq!((folded(d).m, folded(d).c_strides[1]), (12, 7));
    d.c_strides = [0, 0, 1];
    assert_eq!((folded(d).m, folded(d).c_strides[1]), (12, 0));
    d.c_strides = [0, 7, 1];
    assert_eq!(size(&folded(d)), size(&d));
    d.reads_input = 0;
    assert_eq!(folded(d).m, 12);
    // Left as they are: b not shared, a's batches not continuing (broadcast, padded, or TN),
    // a padded output, a sum of batches, one batch, and more rows than an i64 counts.
    let mut shared = sized_desc(HMR_GEMM_NN, [4, 3, 5, 7]);
    shared.b_strides[0] = 0;
    let unchanged = [
        sized_desc(HMR_GEMM_NN, [4, 3, 5, 7]),
        hmr_gemm_desc {
            a_strides: [0, 5, 1],
            ..shared
        },
        hmr_gemm_desc {
            a_strides: [16, 5, 1],
            ..shared
        },
        hmr_gemm_desc {
            b_strides: [0, 1, 5],
            a_strides: [15, 1, 3],
            ..shared
        },
        hmr_gemm_desc {
            out_strides: [22, 7, 1],
            ..shared
        },
        hmr_gemm_desc {
            sum_batches: 1,
            ..shared
        },
        hmr_gemm_desc { batch: 1, ..shared },
        hmr_gemm_desc {
            batch: 1 << 40,
            m: 1 << 40,
            ..shared
        },
    ];
    for d in unchanged {
        let f = folded(d);
        assert_eq!(
            (size(&f), f.a_strides, f.out_strides),
            (size(&d), d.a_strides, d.out_strides),
            "{:?} {:?}",
            size(&d),
            d.a_strides
        );
    }
    // The folded description is chosen for: four rows of a long k take the narrow path.
    let mut d = sized_desc(HMR_GEMM_NN, [4, 1, 4096, 11008]);
    d.b_strides[0] = 0;
    assert_eq!(super::matmul::select(&d).0, HMR_GEMM_S08A_S7A5_UNAL);
    assert_eq!(
        super::matmul::select(&folded(d)).0,
        HMR_GEMM_S08C_NARROW_T256
    );
    // Sizes or strides the library refuses are refused here too.
    d.m = -1;
    let e = std::panic::catch_unwind(|| folded(d)).expect_err("refused");
    assert!(
        e.downcast_ref::<String>()
            .is_some_and(|m| m.contains("negative size"))
    );
}
