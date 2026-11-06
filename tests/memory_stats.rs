//! The CPU's memory statistics, exactly. This binary has a single test, so that no other test
//! allocates while it measures.

use homura::{DType, Device, MemoryStats};

#[test]
fn cpu_memory_statistics() {
    let cpu = || Device::Cpu.memory_stats();
    let base = cpu();
    assert_eq!(base.reserved, base.allocated);

    // A tensor's buffer counts unrounded, as allocated and reserved alike.
    let x = homura::zeros_with([1000], DType::F32);
    let s = cpu();
    assert_eq!(s.allocated, base.allocated + 4000);
    assert_eq!(s.reserved, s.allocated);
    assert!(s.peak >= s.allocated);

    // A view shares the buffer; a copy has its own.
    let view = x.narrow(0, 0, 10);
    assert_eq!(cpu().allocated, base.allocated + 4000);
    let copy = view.contiguous().to(DType::F64);
    assert_eq!(cpu().allocated, base.allocated + 4000 + 80);
    drop((view, copy));

    // Freed memory is returned at once: the CPU has no cache to empty.
    let y = homura::zeros_with([2500], DType::F64);
    let peak = base.allocated + 4000 + 20_000;
    assert_eq!(cpu().peak.max(peak), cpu().peak);
    drop((x, y));
    let s = cpu();
    assert_eq!((s.allocated, s.reserved), (base.allocated, base.allocated));
    Device::Cpu.empty_cache();
    assert_eq!(cpu().allocated, base.allocated);

    // The peak stays until it is reset to the current value.
    assert!(cpu().peak >= peak);
    Device::Cpu.reset_peak_memory_stats();
    assert_eq!(cpu().peak, base.allocated);
    let z = homura::zeros_with([16], DType::U8);
    assert_eq!(cpu().peak, base.allocated + 16);
    drop(z);

    // Data mapped from a file doesn't count; the meta device has nothing.
    let dir = std::env::temp_dir().join(format!("homura-memory-stats-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.safetensors");
    // Bytes, which are mapped at any offset (wider elements may be copied when misaligned).
    let t = homura::zeros_with([4096], DType::U8);
    homura::save_safetensors([("t", &t)], &path).unwrap();
    drop(t);
    let before = cpu().allocated;
    // SAFETY: nothing modifies the file while the tensors live.
    let mapped = unsafe { homura::io::load_mmap(&path) }.unwrap();
    assert_eq!(cpu().allocated, before);
    drop(mapped);
    std::fs::remove_dir_all(&dir).unwrap();
    let _meta = homura::zeros_with([1 << 30], Device::Meta);
    assert_eq!(Device::Meta.memory_stats(), MemoryStats::default());
}
