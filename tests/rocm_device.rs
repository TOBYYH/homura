//! The ROCm device API: counting devices, their properties and memory, synchronizing. Needs a
//! ROCm GPU; built only with the `rocm` feature.
#![cfg(feature = "rocm")]

use std::process::Command;

use homura::Device;

fn panic_message<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> String {
    let Err(e) = std::panic::catch_unwind(f) else {
        panic!("didn't panic")
    };
    match e.downcast::<String>() {
        Ok(s) => *s,
        Err(e) => e.downcast_ref::<&str>().unwrap().to_string(),
    }
}

#[test]
fn devices_are_counted_and_described() {
    let n = Device::rocm_count();
    assert!(n >= 1, "these tests need a ROCm GPU");
    for i in 0..n as u32 {
        let d = Device::Rocm(i);
        let p = d.properties().unwrap();
        assert_eq!(p.warp_size, 32);
        assert!(p.arch == "gfx1201" || p.arch == "gfx1200", "{}", p.arch);
        let m = d.mem_get_info().unwrap();
        assert_eq!(m.total, p.total_memory);
        assert!(m.free <= m.total);
        d.synchronize();
        assert_eq!(d.name(), format!("rocm:{i}"));
    }
}

/// The properties agree with `rocminfo`, when it is installed: the GPU agents' names,
/// architectures and compute unit counts, in order.
#[test]
fn properties_agree_with_rocminfo() {
    let Ok(out) = Command::new("rocminfo").output() else {
        eprintln!("rocminfo not found; skipped");
        return;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut gpus = vec![];
    for agent in text.split("Agent ").skip(1) {
        let field = |name: &str| {
            agent
                .lines()
                .find_map(|l| l.trim().strip_prefix(name).map(|v| v.trim().to_string()))
        };
        if field("Device Type:").as_deref() != Some("GPU") {
            continue;
        }
        let cus = field("Compute Unit:").unwrap().parse::<u32>().unwrap();
        gpus.push((
            field("Name:").unwrap(),
            field("Marketing Name:").unwrap(),
            cus,
        ));
    }
    assert_eq!(gpus.len(), Device::rocm_count());
    for (i, (arch, name, cus)) in gpus.into_iter().enumerate() {
        let p = Device::Rocm(i as u32).properties().unwrap();
        assert_eq!((p.arch, p.name, p.compute_units), (arch, name, cus));
    }
}

#[test]
fn a_missing_device_panics() {
    let n = Device::rocm_count();
    let expected = format!(
        "rocm:{n} is not available: {n} ROCm device{} found",
        if n == 1 { "" } else { "s" }
    );
    let d = Device::Rocm(n as u32);
    assert_eq!(panic_message(|| d.properties()), expected);
    assert_eq!(panic_message(|| d.mem_get_info()), expected);
    assert_eq!(panic_message(|| d.synchronize()), expected);
    assert_eq!(panic_message(|| homura::zeros_with([2], d)), expected);
}

#[test]
fn devices_work_from_many_threads() {
    let threads: Vec<_> = (0..8)
        .map(|_| std::thread::spawn(|| Device::Rocm(0).synchronize()))
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}
