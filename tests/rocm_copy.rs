//! Moving data on and off a ROCm GPU (M4 step 2b): transfers, dtype conversions, `copy_`,
//! creation, reading back and display, compared with the CPU byte for byte. Needs a ROCm GPU;
//! built only with the `rocm` feature.
#![cfg(feature = "rocm")]

#[path = "common/rocm.rs"]
mod rocm;

use homura::{DType, Device, Tensor};
use rocm::{DTYPES, GPU, assert_same, bits, large_layouts, layouts};

#[test]
fn every_conversion_matches_the_cpu_in_every_layout() {
    for (k, &from) in DTYPES.iter().enumerate() {
        for pair in layouts(from, 10 + k as u64) {
            // The move itself keeps the bits and the shape, and so does a copy on the GPU.
            assert_same(&pair.gpu, &pair.cpu, &format!("{from} {}", pair.name));
            let copy = pair.gpu.deep_clone();
            assert_same(&copy, &pair.cpu, &format!("{from} {} copied", pair.name));
            for &to in &DTYPES {
                let what = format!("{from} -> {to}, {}", pair.name);
                let cpu = pair.cpu.to(to);
                let gpu = pair.gpu.to(to);
                assert_same(&gpu, &cpu, &what);
                if to != from {
                    assert!(gpu.is_contiguous(), "{what}: not contiguous");
                }
            }
        }
    }
}

/// NaNs of both signs with random payloads (and signalling ones) from every float dtype to every
/// dtype: the payload rules of the CPU (x86 `cvtsd2ss`, `half`'s conversions) hold on the GPU.
#[test]
fn nan_payloads_convert_like_the_cpu() {
    let mut w = rocm::Words::new(99);
    let n = 1 << 16;
    let f64s: Vec<f64> = (0..n)
        .map(|_| {
            let r = w.next();
            // Exponent all ones, a nonzero mantissa, either sign.
            f64::from_bits((r & 0x800F_FFFF_FFFF_FFFF) | 0x7FF0_0000_0000_0001)
        })
        .collect();
    let f32s: Vec<f32> = (0..n)
        .map(|_| f32::from_bits((w.next() as u32 & 0x807F_FFFF) | 0x7F80_0001))
        .collect();
    let f16s: Vec<half::f16> = (0..n)
        .map(|_| half::f16::from_bits((w.next() as u16 & 0x83FF) | 0x7C01))
        .collect();
    let bf16s: Vec<half::bf16> = (0..n)
        .map(|_| half::bf16::from_bits((w.next() as u16 & 0x807F) | 0x7F81))
        .collect();
    for cpu in [
        Tensor::from_slice(&f64s, [n]),
        Tensor::from_slice(&f32s, [n]),
        Tensor::from_slice(&f16s, [n]),
        Tensor::from_slice(&bf16s, [n]),
    ] {
        let gpu = cpu.to(GPU);
        for &to in &DTYPES {
            assert_same(
                &gpu.to(to),
                &cpu.to(to),
                &format!("NaNs {} -> {to}", cpu.dtype()),
            );
        }
    }
}

#[test]
fn conversions_of_large_tensors_take_several_grid_rounds() {
    for (from, to) in [
        (DType::F32, DType::BF16),
        (DType::F64, DType::F16),
        (DType::I64, DType::F32),
        (DType::U8, DType::U8),
    ] {
        for pair in large_layouts(from, 7) {
            let what = format!("{from} -> {to}, {}", pair.name);
            assert_same(&pair.gpu.to(to), &pair.cpu.to(to), &what);
        }
    }
}

/// The counterexamples of the plan (step 2b, item 4; revision 8), spelled out.
#[test]
fn conversion_counterexamples() {
    let f = |v: f64, dtype: DType| homura::tensor([v]).to((DType::F32, GPU)).to(dtype);
    assert_eq!(f(300.0, DType::U8).to_vec::<u8>(), [44]);
    assert_eq!(f(-129.9, DType::I8).to_vec::<i8>(), [127]);
    assert_eq!(f(f64::INFINITY, DType::I32).to_vec::<i32>(), [-1]);
    assert_eq!(f(f64::INFINITY, DType::I64).to_vec::<i64>(), [i64::MAX]);
    assert_eq!(f(1e30, DType::I16).to_vec::<i16>(), [-1]);
    assert_eq!(f(f64::NAN, DType::I32).to_vec::<i32>(), [0]);
    let h = homura::tensor([-2.5f32]).to((DType::F16, GPU));
    assert_eq!(h.to(DType::U8).to_vec::<u8>(), [254]);
    let g = |v: f64, dtype: DType| homura::tensor_with([v], (DType::F64, GPU)).to(dtype);
    let p = |e: i32| 2f64.powi(e);
    // Through f32: torch 2.14.1 gives these.
    assert_eq!(
        g(1.0 + p(-8) + p(-21), DType::F32)
            .to(DType::BF16)
            .to_vec::<half::bf16>()[0]
            .to_bits(),
        0x3f81
    );
    assert_eq!(
        g(1.0 + p(-8) + p(-40), DType::BF16).to_vec::<half::bf16>()[0].to_bits(),
        0x3f80
    );
    assert_eq!(
        g(1.0 + p(-11) + p(-40), DType::F16).to_vec::<half::f16>()[0].to_bits(),
        0x3c00
    );
    let i = homura::tensor_with([(1i64 << 24) + (1 << 16) + 1], GPU);
    assert_eq!(i.to(DType::BF16).to_vec::<half::bf16>()[0].to_f64(), p(24));
    // Bool data stays 0 or 1.
    let b = homura::tensor_with([0.0, -0.0, 0.1, f64::NAN, -3.0], GPU).to(DType::Bool);
    assert_eq!(b.to_vec::<bool>(), [false, false, true, true, true]);
    assert_eq!(b.to(DType::U8).to_vec::<u8>(), [0, 0, 1, 1, 1]);
}

#[test]
fn moves_between_the_cpu_and_the_gpu() {
    for (k, &dtype) in DTYPES.iter().enumerate() {
        for pair in layouts(dtype, 100 + k as u64) {
            let what = format!("{dtype}, {}", pair.name);
            // CPU -> GPU of a strided tensor: contiguous on the GPU, same bits.
            let up = pair.cpu.to(GPU);
            assert_same(&up, &pair.cpu, &what);
            assert!(up.is_contiguous());
            // GPU -> CPU of a strided tensor.
            let down = pair.gpu.to(Device::Cpu);
            assert_eq!(down.device(), Device::Cpu);
            assert_eq!(bits(&down), bits(&pair.cpu), "{what}");
            // Round trip.
            assert_eq!(bits(&up.to(Device::Cpu)), bits(&pair.cpu), "{what}");
            // Between devices with a new dtype, both ways.
            for &to in &[DType::F16, DType::I32, DType::Bool, DType::F64] {
                let expected = pair.cpu.to(to);
                assert_same(
                    &pair.cpu.to((to, GPU)),
                    &expected,
                    &format!("{what} -> {to}"),
                );
                let down = pair.gpu.to((to, Device::Cpu));
                assert_eq!(down.device(), Device::Cpu);
                assert_eq!(bits(&down), bits(&expected), "{what} -> {to} on the cpu");
            }
            // Between two ROCm devices (the same one twice: there is one GPU here) the data goes
            // through the host; nothing changes, so `to` returns the tensor itself.
            assert!(pair.gpu.to(GPU).is_same(&pair.gpu));
        }
    }
}

#[test]
fn copy_into_strided_targets() {
    for (k, &from) in DTYPES.iter().enumerate() {
        for &to in &[DType::F32, DType::I8, DType::BF16, from] {
            // A transposed target, a broadcast source of another dtype, on the same device.
            let src = rocm::values(from, 5, 300 + k as u64);
            let cpu = homura::zeros_with([5, 3], to).transpose(0, 1);
            let gpu = homura::zeros_with([5, 3], (to, GPU)).transpose(0, 1);
            cpu.copy_(&src.reshape([1, 5]));
            gpu.copy_(&src.to(GPU).reshape([1, 5]));
            let what = format!("{from} -> {to}");
            assert_same(&gpu, &cpu, &what);
            // From the CPU into a GPU target, and from the GPU into a CPU target.
            let gpu = homura::zeros_with([5, 3], (to, GPU)).transpose(0, 1);
            gpu.copy_(&src.reshape([1, 5]));
            assert_same(&gpu, &cpu, &format!("{what}, cpu source"));
            let back = homura::zeros_with([5, 3], to).transpose(0, 1);
            back.copy_(&src.to(GPU).reshape([1, 5]));
            assert_eq!(bits(&back), bits(&cpu), "{what}, gpu source");
        }
    }
    // Copying a tensor onto an overlapping part of itself goes through a copy.
    let x = homura::arange_with(6, GPU);
    x.narrow(0, 1, 5).copy_(&x.narrow(0, 0, 5));
    assert_eq!(x.to_vec::<i64>(), [0, 0, 1, 2, 3, 4]);
}

#[test]
fn creation_on_the_gpu() {
    let cases: Vec<(&str, Tensor, Tensor)> = vec![
        (
            "zeros",
            homura::zeros_with([2, 3], GPU),
            homura::zeros([2, 3]),
        ),
        (
            "ones i16",
            homura::ones_with([4], (DType::I16, GPU)),
            homura::ones_with([4], DType::I16),
        ),
        (
            "full -0.0",
            homura::full_with([3], -0.0, GPU),
            homura::full([3], -0.0),
        ),
        (
            "full f16",
            homura::full_with([2, 2], 1.0 / 3.0, (DType::F16, GPU)),
            homura::full_with([2, 2], 1.0 / 3.0, DType::F16),
        ),
        (
            "full bool",
            homura::full_with([3], true, GPU),
            homura::full([3], true),
        ),
        ("arange", homura::arange_with(7, GPU), homura::arange(7)),
        (
            "arange_step f64",
            homura::arange_step_with(0.5, 3.0, 0.25, (DType::F64, GPU)),
            homura::arange_step_with(0.5, 3.0, 0.25, DType::F64),
        ),
        (
            "linspace",
            homura::linspace_with(-1.0, 1.0, 9, GPU),
            homura::linspace(-1.0, 1.0, 9),
        ),
        ("eye", homura::eye_with(3, GPU), homura::eye(3)),
        (
            "tensor",
            homura::tensor_with([[1u8, 2], [3, 4]], GPU),
            homura::tensor([[1u8, 2], [3, 4]]),
        ),
        (
            "from_slice",
            Tensor::from_slice(&[1.5f32, -2.0, 3.25], [3]).to(GPU),
            Tensor::from_slice(&[1.5f32, -2.0, 3.25], [3]),
        ),
        (
            "zeros_like",
            homura::zeros_like(&homura::ones_with([2], GPU)),
            homura::zeros([2]),
        ),
        (
            "empty",
            homura::zeros_with([0, 4], GPU),
            homura::zeros([0, 4]),
        ),
    ];
    for (what, gpu, cpu) in &cases {
        assert_same(gpu, cpu, what);
    }
    // Reading back.
    assert_eq!(homura::full_with([], 2.5, GPU).item::<f32>(), 2.5);
    let h = homura::full_with([2], 1.5, (DType::BF16, GPU));
    assert_eq!(h.to_vec::<f32>(), [1.5, 1.5]);
    // fill_ on a strided GPU tensor.
    let x = homura::zeros_with([3, 4], GPU);
    x.transpose(0, 1).narrow(0, 1, 2).fill_(7.0);
    let y = homura::zeros([3, 4]);
    y.transpose(0, 1).narrow(0, 1, 2).fill_(7.0);
    assert_same(&x, &y, "fill_");
}

/// `to_vec`, `item` and `Display` of strided GPU tensors read them directly (through a contiguous
/// copy on the GPU), without moving them to the CPU first.
#[test]
fn strided_gpu_tensors_read_back_directly() {
    let cpu = homura::arange_with(12, DType::F32).reshape([3, 4]);
    let gpu = cpu.to(GPU);
    let (ct, gt) = (cpu.transpose(0, 1), gpu.transpose(0, 1));
    assert!(!gt.is_contiguous());
    assert_eq!(gt.to_vec::<f32>(), ct.to_vec::<f32>());
    assert_eq!(gpu.select(1, 2).select(0, 1).item::<f32>(), 6.0);
    let h = gpu.to(DType::F16).transpose(0, 1).narrow(0, 1, 2);
    assert_eq!(
        h.to_vec::<f32>(),
        cpu.transpose(0, 1).narrow(0, 1, 2).to_vec::<f32>()
    );
    let shown = format!("{gt}").replace(", device=rocm:0", "");
    assert_eq!(shown, format!("{ct}"));
    let b = homura::full_with([2, 3], true, GPU)
        .expand([4, 2, 3])
        .transpose(0, 2);
    assert_eq!(b.to_vec::<bool>(), [true; 24]);
}

#[test]
fn gpu_tensors_display_like_cpu_tensors() {
    for t in [
        homura::arange_with(5, GPU).reshape([1, 5]),
        homura::linspace_with(0.0, 1.0, 4, (DType::F16, GPU)),
        homura::arange_with(6, GPU).reshape([2, 3]).transpose(0, 1),
        homura::zeros_with([0], GPU),
        homura::full_with([], true, GPU),
    ] {
        let cpu = format!("{}", t.to(Device::Cpu));
        let gpu = format!("{t}");
        // As PyTorch puts it: after the data, before the dtype.
        assert!(gpu.contains(", device=rocm:0"), "{gpu}");
        let strip = |s: &str| s.replace(", device=rocm:0", "");
        assert_eq!(strip(&gpu), cpu);
        assert_eq!(strip(&format!("{t:?}")), format!("{:?}", t.to(Device::Cpu)));
    }
}

#[test]
fn operands_on_different_devices_are_refused() {
    let r = std::panic::catch_unwind(|| homura::zeros_with([2], GPU).add(homura::zeros([2])));
    let e = r.expect_err("adding tensors on two devices");
    let m = e.downcast_ref::<String>().unwrap();
    assert!(
        m.starts_with("Expected all tensors to be on the same device"),
        "{m}"
    );
}

#[test]
fn try_functions_report_running_out_of_memory() {
    let too_much = 1usize << 42; // 4 TiB
    let e = homura::try_zeros_with([too_much], (DType::U8, GPU)).unwrap_err();
    let homura::Error::OutOfMemory(e) = e else {
        panic!("not out of memory: {e}")
    };
    assert_eq!((e.device(), e.requested()), (GPU, too_much));
    assert!(
        e.to_string()
            .starts_with("ROCm out of memory. Tried to allocate 4194304.00 MiB.")
    );
    assert!(homura::try_ones_with([too_much / 2], (DType::I16, GPU)).is_err());
    assert!(homura::try_full_with([too_much / 8], 1.0, (DType::F64, GPU)).is_err());
    // The device still works, and so do the try functions.
    let t = homura::try_full_with([2], 3, (DType::I32, GPU)).unwrap();
    assert_eq!(t.to_vec::<i32>(), [3, 3]);
    assert_eq!(homura::try_zeros_with([2], GPU).unwrap().device(), GPU);
    assert_eq!(
        homura::try_ones_with([2], GPU).unwrap().to_vec::<f32>(),
        [1.0, 1.0]
    );
    // try_to of a big CPU tensor: the expanded view is fine on the CPU, its GPU copy isn't.
    let big = homura::zeros_with([1], DType::U8).expand([too_much]);
    let e = big.try_to(GPU).unwrap_err();
    assert!(
        matches!(e, homura::Error::OutOfMemory(ref e) if e.device() == GPU),
        "{e}"
    );
    let x = homura::arange(4);
    assert_eq!(x.try_to(GPU).unwrap().to_vec::<i64>(), [0, 1, 2, 3]);
}

/// Every f32 bit pattern converted to f16 and bf16, and every f16 and bf16 pattern to every
/// dtype, on the GPU and on the CPU. About a minute in a release build; run it with
/// `cargo test --release --features rocm --test rocm_copy -- --ignored`.
#[test]
#[ignore = "exhaustive; run in release with --ignored"]
fn half_conversions_match_the_cpu_on_every_input() {
    const CHUNK: u64 = 1 << 26;
    for start in (0..1u64 << 32).step_by(CHUNK as usize) {
        let v: Vec<f32> = (start..start + CHUNK)
            .map(|b| f32::from_bits(b as u32))
            .collect();
        let cpu = Tensor::from_slice(&v, [CHUNK as usize]);
        let gpu = cpu.to(GPU);
        for dtype in [DType::F16, DType::BF16] {
            let g = gpu.to(dtype).to(Device::Cpu);
            let c = cpu.to(dtype);
            let (g, c) = if dtype == DType::F16 {
                let f = |t: &Tensor| {
                    t.to_vec::<half::f16>()
                        .iter()
                        .map(|h| h.to_bits())
                        .collect::<Vec<_>>()
                };
                (f(&g), f(&c))
            } else {
                let f = |t: &Tensor| {
                    t.to_vec::<half::bf16>()
                        .iter()
                        .map(|h| h.to_bits())
                        .collect::<Vec<_>>()
                };
                (f(&g), f(&c))
            };
            if let Some(i) = (0..g.len()).find(|&i| g[i] != c[i]) {
                panic!(
                    "f32 {:#010x} -> {dtype}: gpu {:#06x}, cpu {:#06x}",
                    start + i as u64,
                    g[i],
                    c[i]
                );
            }
        }
    }
    let all: Vec<u16> = (0..=u16::MAX).collect();
    for half in [DType::F16, DType::BF16] {
        let cpu = match half {
            DType::F16 => Tensor::from_slice(
                &all.iter()
                    .map(|&b| half::f16::from_bits(b))
                    .collect::<Vec<_>>(),
                [all.len()],
            ),
            _ => Tensor::from_slice(
                &all.iter()
                    .map(|&b| half::bf16::from_bits(b))
                    .collect::<Vec<_>>(),
                [all.len()],
            ),
        };
        let gpu = cpu.to(GPU);
        for &to in &DTYPES {
            assert_same(&gpu.to(to), &cpu.to(to), &format!("every {half} -> {to}"));
        }
    }
}
