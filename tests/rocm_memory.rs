//! ROCm device memory through the public API: statistics, the cache, running out of memory and
//! threads. Needs a ROCm GPU; built only with the `rocm` feature. The tests share rocm:0's
//! allocator, so they run one at a time.
#![cfg(feature = "rocm")]

use std::sync::{Mutex, MutexGuard};

use homura::{DType, Device, Tensor};

const GPU: Device = Device::Rocm(0);
const MIB: u64 = 1 << 20;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

fn zeros(n: usize) -> Tensor {
    homura::zeros_with([n], (DType::U8, GPU))
}

#[test]
fn tensors_count_in_the_statistics_and_the_cache() {
    let _lock = lock();
    GPU.empty_cache();
    // The peak is cumulative: other tests of this binary may have raised it.
    GPU.reset_peak_memory_stats();
    let base = GPU.memory_stats();
    assert_eq!((base.allocated, base.reserved, base.peak), (0, 0, 0));

    // Small requests round up to 512 bytes, from a 2 MiB segment.
    let a = zeros(1000);
    let s = GPU.memory_stats();
    assert_eq!(s.allocated, 1024);
    assert_eq!(s.reserved, 2 * MIB);
    assert!(s.peak >= 1024);
    // Large ones round up to 2 MiB, in a segment of their own.
    let b = zeros(3 << 20);
    let s = GPU.memory_stats();
    assert_eq!(s.allocated, 1024 + 4 * MIB);
    let reserved = s.reserved;

    // Freed memory stays cached and is reused.
    drop(b);
    assert_eq!(GPU.memory_stats().allocated, 1024);
    assert_eq!(GPU.memory_stats().reserved, reserved);
    let b = zeros(4 << 20);
    assert_eq!(GPU.memory_stats().reserved, reserved);
    drop(b);

    // The peak stays until reset to the current value.
    assert_eq!(GPU.memory_stats().peak, 1024 + 4 * MIB);
    GPU.reset_peak_memory_stats();
    assert_eq!(GPU.memory_stats().peak, 1024);

    // Emptying the cache releases the large segment; the small one is still in use.
    GPU.empty_cache();
    assert_eq!(GPU.memory_stats().reserved, 2 * MIB);
    drop(a);
    GPU.empty_cache();
    let s = GPU.memory_stats();
    assert_eq!((s.allocated, s.reserved), (0, 0));
    // Empty tensors take no memory.
    let e = homura::zeros_with([0, 3], GPU);
    assert_eq!(GPU.memory_stats().allocated, 0);
    drop(e);
}

#[test]
fn a_request_larger_than_the_gpu_panics_and_leaves_it_usable() {
    let _lock = lock();
    let total = GPU.properties().unwrap().total_memory;
    let before = GPU.memory_stats().allocated;
    let n = (2 * total) as usize;
    let e = std::panic::catch_unwind(|| zeros(n)).unwrap_err();
    let message = e.downcast_ref::<String>().unwrap();
    let start = format!(
        "ROCm out of memory. Tried to allocate {:.2} MiB. rocm:0 has a total capacity of {:.2} \
         GiB of which ",
        n as f64 / MIB as f64,
        total as f64 / (1u64 << 30) as f64
    );
    assert!(message.starts_with(&start), "{message}");
    assert!(
        message.contains(" MiB is allocated by homura, and ")
            && message.ends_with(" MiB is reserved but unallocated."),
        "{message}"
    );
    assert_eq!(GPU.memory_stats().allocated, before);
    let x = zeros(1 << 20);
    assert_eq!(x.device(), GPU);
}

/// Temporaries allocated and dropped at a high rate reuse the same cached memory: once the cache
/// has a segment for each size of a repeating pattern, nothing more is reserved.
#[test]
fn temporaries_reuse_the_cache() {
    let _lock = lock();
    let _held = (zeros(4096), zeros(8 << 20));
    let pattern = |i: usize| {
        drop(zeros(1 + i % 4096));
        if i.is_multiple_of(100) {
            drop(zeros(1 + (i % 7) * (1 << 20)));
        }
    };
    (0..700).for_each(pattern);
    let reserved = GPU.memory_stats().reserved;
    (0..5000).for_each(pattern);
    assert_eq!(GPU.memory_stats().reserved, reserved);
}

/// Tensors allocated on some threads and dropped on others leave nothing allocated.
#[test]
fn tensors_are_allocated_and_dropped_on_many_threads() {
    let _lock = lock();
    let before = GPU.memory_stats().allocated;
    let (send, receive) = std::sync::mpsc::channel::<Tensor>();
    let producers: Vec<_> = (0..8)
        .map(|t| {
            let send = send.clone();
            std::thread::spawn(move || {
                for i in 0..200 {
                    let x = zeros(1 + (t * 7919 + i * 104_729) % (3 << 20));
                    if i % 2 == 0 {
                        send.send(x).unwrap();
                    }
                }
            })
        })
        .collect();
    drop(send);
    let consumer = std::thread::spawn(move || receive.iter().count());
    for p in producers {
        p.join().unwrap();
    }
    assert_eq!(consumer.join().unwrap(), 8 * 100);
    assert_eq!(GPU.memory_stats().allocated, before);
}
