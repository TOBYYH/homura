//! Helpers for GPU differential tests: inputs with special and pseudo-random values of every
//! dtype, the same tensor on the CPU and on the GPU in a range of layouts, and comparisons of the
//! raw bits of the results.
//!
//! Include it with `#[path = "common/rocm.rs"] mod rocm;` from a test that runs on a ROCm GPU.

#![allow(dead_code)]

use half::{bf16, f16};
use homura::{DType, Device, Tensor};

pub const GPU: Device = Device::Rocm(0);

pub const DTYPES: [DType; 10] = [
    DType::Bool,
    DType::U8,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

/// A deterministic stream of 64-bit words (SplitMix64).
pub struct Words(u64);

impl Words {
    pub fn new(seed: u64) -> Words {
        Words(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn p(e: i32) -> f64 {
    2f64.powi(e)
}

/// Values every conversion test covers: zeros, infinities, NaNs with payloads, subnormals,
/// limits, the values next to rounding ties, and the counterexamples of the M4 plan (step 2b,
/// item 4, and plan revision 8).
pub fn special_f64() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -2.5,
        2.5,
        255.9,
        -0.5,
        300.0,
        -129.9,
        1e30,
        -1e30,
        65504.0,
        65520.0,
        -65536.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7FF0_0000_0000_0001), // signalling NaN
        f64::from_bits(0xFFF4_0000_0000_1234), // quiet NaN with a payload
        f64::from_bits(0x7FF8_0000_2000_0000), // payload below f32's mantissa
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE / 3.0, // subnormal
        f64::from_bits(1),
        f64::MAX,
        f64::MIN,
        f32::MAX as f64,
        f32::MIN_POSITIVE as f64,
        f32::from_bits(1) as f64,        // smallest f32 subnormal
        f32::from_bits(1) as f64 / 2.0,  // half of it: rounds to 0 (tie to even)
        f32::from_bits(1) as f64 * 0.75, // rounds up to the subnormal
        6.0e-8,                          // around the smallest f16 subnormal
        p(-24) * 1.5,
        p(-133), // smallest bf16 subnormal
        9.2e18,
        -9.3e18,
        p(63),
        -p(63),
        p(63) - 1024.0,
        p(31),
        -p(31) - 1.0,
        p(53) + 1.0, // not an f64: rounds to 2^53
        1.0 + p(-8) + p(-21),
        1.0 + p(-8) + p(-40),
        1.0 + p(-11) + p(-40),
        1.0 + p(-11),
        1.0 + p(-8),
        1.0 + p(-24),
        1.0 + p(-24) + p(-52),
        -(1.0 + p(-8) + p(-21)),
    ];
    v.extend((0..40).map(|i| (i as f64 - 20.0) * 0.37));
    v
}

pub fn special_i64() -> Vec<i64> {
    let mut v = vec![
        0,
        1,
        -1,
        2,
        127,
        128,
        -128,
        -129,
        255,
        256,
        -255,
        300,
        32767,
        32768,
        -32769,
        65535,
        65536,
        i32::MAX as i64,
        i32::MIN as i64,
        i32::MAX as i64 + 1,
        i64::MAX,
        i64::MIN,
        i64::MAX - 1,
        (1 << 24) + 1,
        (1 << 24) + (1 << 16) + 1,
        (1 << 53) + 1,
        (1 << 53) + (1 << 30) + 1,
        65504,
        65520,
        70000,
    ];
    v.extend((0..40).map(|i| i * 7 - 140));
    v
}

/// `n` values of `dtype` on the CPU: the special values first (converted with the CPU's casts
/// where the dtype can't hold them), then pseudo-random bit patterns (every pattern of a float
/// dtype is possible, NaN payloads and subnormals included).
pub fn values(dtype: DType, n: usize, seed: u64) -> Tensor {
    let mut w = Words::new(seed);
    macro_rules! build {
        ($t:ty, $special:expr, $random:expr) => {{
            let special: Vec<$t> = $special;
            let v: Vec<$t> = (0..n)
                .map(|i| {
                    if i < special.len() {
                        special[i]
                    } else {
                        let r = w.next();
                        $random(r)
                    }
                })
                .collect();
            Tensor::from_slice(&v, [n])
        }};
    }
    let floats = special_f64();
    let ints = special_i64();
    match dtype {
        DType::Bool => build!(bool, vec![false, true], |r: u64| r & 1 == 1),
        DType::U8 => build!(u8, ints.iter().map(|&v| v as u8).collect(), |r: u64| r
            as u8),
        DType::I8 => build!(i8, ints.iter().map(|&v| v as i8).collect(), |r: u64| r
            as i8),
        DType::I16 => build!(i16, ints.iter().map(|&v| v as i16).collect(), |r: u64| r
            as i16),
        DType::I32 => build!(i32, ints.iter().map(|&v| v as i32).collect(), |r: u64| r
            as i32),
        DType::I64 => build!(i64, ints.clone(), |r: u64| r as i64),
        DType::F16 => build!(
            f16,
            floats.iter().map(|&v| f16::from_f64(v)).collect(),
            |r: u64| f16::from_bits(r as u16)
        ),
        DType::BF16 => build!(
            bf16,
            floats.iter().map(|&v| bf16::from_f64(v)).collect(),
            |r: u64| bf16::from_bits(r as u16)
        ),
        DType::F32 => build!(f32, floats.iter().map(|&v| v as f32).collect(), |r: u64| {
            f32::from_bits(r as u32)
        }),
        DType::F64 => build!(f64, floats.clone(), |r: u64| f64::from_bits(r)),
    }
}

/// The raw bits of every element of `t` (on any device), in row-major order: 0 or 1 for `Bool`,
/// the two's complement bits for integers, the IEEE bits for floats.
pub fn bits(t: &Tensor) -> Vec<u64> {
    let t = t.to(Device::Cpu);
    match t.dtype() {
        DType::Bool => t.to_vec::<bool>().into_iter().map(u64::from).collect(),
        DType::U8 => t.to_vec::<u8>().into_iter().map(u64::from).collect(),
        DType::I8 => t
            .to_vec::<i8>()
            .into_iter()
            .map(|v| v as u8 as u64)
            .collect(),
        DType::I16 => t
            .to_vec::<i16>()
            .into_iter()
            .map(|v| v as u16 as u64)
            .collect(),
        DType::I32 => t
            .to_vec::<i32>()
            .into_iter()
            .map(|v| v as u32 as u64)
            .collect(),
        DType::I64 => t.to_vec::<i64>().into_iter().map(|v| v as u64).collect(),
        DType::F16 => t
            .to_vec::<f16>()
            .into_iter()
            .map(|v| v.to_bits() as u64)
            .collect(),
        DType::BF16 => t
            .to_vec::<bf16>()
            .into_iter()
            .map(|v| v.to_bits() as u64)
            .collect(),
        DType::F32 => t
            .to_vec::<f32>()
            .into_iter()
            .map(|v| v.to_bits() as u64)
            .collect(),
        DType::F64 => t.to_vec::<f64>().into_iter().map(f64::to_bits).collect(),
    }
}

/// Asserts that `gpu` is on the GPU and has the shape, dtype and bits of `cpu`.
#[track_caller]
pub fn assert_same(gpu: &Tensor, cpu: &Tensor, what: &str) {
    assert_eq!(gpu.device(), GPU, "{what}: device");
    assert_eq!(gpu.shape(), cpu.shape(), "{what}: shape");
    assert_eq!(gpu.dtype(), cpu.dtype(), "{what}: dtype");
    let (g, c) = (bits(gpu), bits(cpu));
    if let Some(i) = (0..g.len()).find(|&i| g[i] != c[i]) {
        panic!(
            "{what}: element {i} differs: gpu {:#x}, cpu {:#x} ({} of {} differ)",
            g[i],
            c[i],
            (0..g.len()).filter(|&i| g[i] != c[i]).count(),
            g.len()
        );
    }
}

/// A tensor on the CPU and the same tensor on the GPU: the same values in the same layout.
pub struct Pair {
    pub name: &'static str,
    pub cpu: Tensor,
    pub gpu: Tensor,
}

/// A layout of the generator: its name, the number of elements of the base tensor, and the view
/// of the base.
type View = (&'static str, usize, fn(&Tensor) -> Tensor);

const VIEWS: [View; 11] = [
    ("contiguous [3, 5]", 15, |t| t.reshape([3, 5])),
    ("contiguous with an offset [7]", 10, |t| t.narrow(0, 2, 7)),
    ("transposed [5, 3]", 15, |t| {
        t.reshape([3, 5]).transpose(0, 1)
    }),
    ("broadcast [4, 3, 5] (stride 0)", 15, |t| {
        t.reshape([1, 3, 5]).expand([4, 3, 5])
    }),
    ("narrow with an offset [2, 3]", 20, |t| {
        t.reshape([4, 5]).narrow(0, 1, 2).narrow(1, 1, 3)
    }),
    ("0-dim", 1, |t| t.reshape(())),
    ("empty [0, 3]", 0, |t| t.reshape([0, 3])),
    ("odd [1031]", 1031, |t| t.clone()),
    ("permuted [5, 4, 3, 2]", 120, |t| {
        t.reshape([2, 3, 4, 5]).permute([3, 2, 1, 0])
    }),
    ("every other [600]", 1200, |t| {
        t.reshape([600, 2]).select(1, 1)
    }),
    ("past one block [3, 257]", 771, |t| {
        t.reshape([3, 257]).transpose(0, 1)
    }),
];

/// `values(dtype, n, seed)` in every layout of the generator, on the CPU and on the GPU (the
/// GPU's from the base moved to the GPU, viewed the same way).
pub fn layouts(dtype: DType, seed: u64) -> Vec<Pair> {
    VIEWS
        .iter()
        .map(|&(name, n, view)| {
            let base = values(dtype, n, seed);
            let cpu = view(&base);
            let gpu = view(&base.to(GPU));
            assert_eq!(gpu.stride(), cpu.stride(), "{name}");
            assert_eq!(gpu.storage_offset(), cpu.storage_offset(), "{name}");
            Pair { name, cpu, gpu }
        })
        .collect()
}

/// Layouts whose kernels need more than one round of the launch grid (65536 blocks of 256
/// threads): a contiguous and a strided one of 2^24 + 1000 elements.
pub fn large_layouts(dtype: DType, seed: u64) -> Vec<Pair> {
    let n = (1 << 24) + 1000;
    let base = values(dtype, 2 * n, seed);
    let gpu_base = base.to(GPU);
    vec![
        Pair {
            name: "large contiguous",
            cpu: base.narrow(0, 0, n),
            gpu: gpu_base.narrow(0, 0, n),
        },
        Pair {
            name: "large strided",
            cpu: base.reshape([n, 2]).select(1, 1),
            gpu: gpu_base.reshape([n, 2]).select(1, 1),
        },
    ]
}
