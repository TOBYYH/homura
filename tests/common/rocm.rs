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

/// 2^e as an f64, exactly, for -1074 <= e <= 1023. Built from the bits: `2f64.powi(e)` computes
/// the reciprocal of 2^-e for negative `e`, which overflows below 2^-1023 when it isn't
/// constant-folded (debug builds), giving 0 (review round 126, R151).
pub fn pow2(e: i32) -> f64 {
    assert!(
        (-1074..=1023).contains(&e),
        "2^{e} is not a finite nonzero f64"
    );
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

fn p(e: i32) -> f64 {
    pow2(e)
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

/// Whether element `i` of the bits of a `dtype` tensor is a NaN.
fn is_nan_bits(dtype: DType, b: u64) -> bool {
    match dtype {
        DType::F16 => f16::from_bits(b as u16).is_nan(),
        DType::BF16 => bf16::from_bits(b as u16).is_nan(),
        DType::F32 => f32::from_bits(b as u32).is_nan(),
        DType::F64 => f64::from_bits(b).is_nan(),
        _ => false,
    }
}

/// [`assert_same`] for the bitwise class of the M4 plan (M4.12): the same bytes, except that a
/// NaN only has to be a NaN (its sign and payload may differ, as they do between x86 and the GPU
/// when two NaNs meet).
#[track_caller]
pub fn assert_bitwise(gpu: &Tensor, cpu: &Tensor, what: &str) {
    assert_eq!(gpu.device(), GPU, "{what}: device");
    assert_eq!(gpu.shape(), cpu.shape(), "{what}: shape");
    assert_eq!(gpu.dtype(), cpu.dtype(), "{what}: dtype");
    let dtype = cpu.dtype();
    let (g, c) = (bits(gpu), bits(cpu));
    let same = |i: usize| g[i] == c[i] || (is_nan_bits(dtype, g[i]) && is_nan_bits(dtype, c[i]));
    if let Some(i) = (0..g.len()).find(|&i| !same(i)) {
        panic!(
            "{what}: element {i} differs: gpu {:#x}, cpu {:#x} ({} of {} differ)",
            g[i],
            c[i],
            (0..g.len()).filter(|&i| !same(i)).count(),
            g.len()
        );
    }
}

/// The elements of a floating point tensor (on any device) as `f64`, exactly.
pub fn floats(t: &Tensor) -> Vec<f64> {
    t.to((DType::F64, Device::Cpu)).to_vec::<f64>()
}

/// The unit roundoff of the type a floating point dtype computes in: f32 for F16, BF16 and F32,
/// f64 for F64.
pub fn unit_roundoff(dtype: DType) -> f64 {
    if dtype == DType::F64 { p(-53) } else { p(-24) }
}

/// The spacing of `dtype` at `|v|` (a finite value of it): the distance from the representable
/// value of largest magnitude not above `|v|` to the next one, the smallest subnormal at 0 and
/// below the normal range. Computed from the exponent field, which `v`'s f64 bits share with its
/// dtype (review round 126, R151: `floor(log2(v))` rounds up just below powers of two).
pub fn ulp(dtype: DType, v: f64) -> f64 {
    let (man, min_exp) = match dtype {
        DType::F16 => (10, -14),
        DType::BF16 => (7, -126),
        DType::F32 => (23, -126),
        DType::F64 => (52, -1022),
        d => panic!("ulp of {d}"),
    };
    assert!(v.is_finite(), "ulp of {v}");
    let biased = (v.abs().to_bits() >> 52) as i32;
    // Zero and f64 subnormals are below every dtype's normal range.
    let e = if biased == 0 { -1023 } else { biased - 1023 };
    pow2(e.max(min_exp) - man)
}

/// The distance in representable values of `dtype` between two finite values of it.
pub fn ulp_distance(dtype: DType, a: f64, b: f64) -> u64 {
    fn ordered(bits: u64, sign_bit: u64) -> i128 {
        if bits & sign_bit != 0 {
            -((bits & !sign_bit) as i128)
        } else {
            bits as i128
        }
    }
    let (x, y, s) = match dtype {
        DType::F16 => (
            f16::from_f64(a).to_bits() as u64,
            f16::from_f64(b).to_bits() as u64,
            0x8000,
        ),
        DType::BF16 => (
            bf16::from_f64(a).to_bits() as u64,
            bf16::from_f64(b).to_bits() as u64,
            0x8000,
        ),
        DType::F32 => (
            (a as f32).to_bits() as u64,
            (b as f32).to_bits() as u64,
            1 << 31,
        ),
        DType::F64 => (a.to_bits(), b.to_bits(), 1 << 63),
        d => panic!("ulp distance of {d}"),
    };
    (ordered(x, s) - ordered(y, s)).unsigned_abs() as u64
}

/// The largest ulp distance between the GPU's and the CPU's elements, and how many elements
/// differ at all.
#[derive(Clone, Copy, Debug, Default)]
pub struct UlpStats {
    pub elements: usize,
    pub differ: usize,
    pub max_ulp: u64,
}

impl UlpStats {
    pub fn merge(&mut self, other: UlpStats) {
        self.elements += other.elements;
        self.differ += other.differ;
        self.max_ulp = self.max_ulp.max(other.max_ulp);
    }
}

/// The transcendental class of the M4 plan (M4.12): both results rounded to the output dtype,
/// `|gpu - cpu| <= k * u_comp * m[i] + ulp_out(max(|gpu|, |cpu|))`, where `m[i]` is the size of
/// the largest intermediate of the operation at element `i` and `u_comp` the unit roundoff of
/// the output dtype's computation type ([`unit_roundoff`]), which the elementwise kernels compute
/// in. A NaN must meet a NaN and an infinity the same infinity. Returns the statistics, or the
/// first violation.
pub fn check_transcendental(
    gpu: &Tensor,
    cpu: &Tensor,
    m: &[f64],
    k: f64,
) -> Result<UlpStats, String> {
    check_transcendental_in(gpu, cpu, m, k, unit_roundoff(cpu.dtype()))
}

/// [`check_transcendental`] for an operation that computes in another type than the output
/// dtype's computation type, with that type's unit roundoff `u_comp`: normal sampling computes
/// in f64 for every dtype (review round 126, R150).
pub fn check_transcendental_in(
    gpu: &Tensor,
    cpu: &Tensor,
    m: &[f64],
    k: f64,
    u_comp: f64,
) -> Result<UlpStats, String> {
    let extra: Vec<f64> = m.iter().map(|&m| k * u_comp * m).collect();
    check_within(gpu, cpu, &extra).map_err(|e| format!("{e} (k = {k}, u_comp = {u_comp:e})"))
}

/// The general form of the tolerance classes of the M4 plan (M4.12): both results rounded to
/// the output dtype, `|gpu - cpu| <= extra[i] + ulp_out(max(|gpu|, |cpu|))`, where `extra[i]` is
/// the class's bound before the output's rounding at element `i`. A NaN must meet a NaN and an
/// infinity the same infinity. Returns the statistics, or the first violation.
pub fn check_within(gpu: &Tensor, cpu: &Tensor, extra: &[f64]) -> Result<UlpStats, String> {
    if gpu.device() != GPU || gpu.shape() != cpu.shape() || gpu.dtype() != cpu.dtype() {
        return Err(format!(
            "gpu {:?} {} {} vs cpu {:?} {}",
            gpu.shape(),
            gpu.dtype(),
            gpu.device(),
            cpu.shape(),
            cpu.dtype()
        ));
    }
    let dtype = cpu.dtype();
    let (g, c) = (floats(gpu), floats(cpu));
    assert_eq!(extra.len(), c.len());
    let mut stats = UlpStats {
        elements: c.len(),
        ..UlpStats::default()
    };
    for i in 0..c.len() {
        let (x, y) = (g[i], c[i]);
        if x.is_nan() || y.is_nan() || x.is_infinite() || y.is_infinite() {
            if !(x.is_nan() && y.is_nan() || x == y) {
                return Err(format!(
                    "element {i}: gpu {x:e}, cpu {y:e} (classification)"
                ));
            }
            continue;
        }
        if x == y {
            continue;
        }
        let bound = extra[i] + ulp(dtype, x.abs().max(y.abs()));
        // A NaN bound rejects too.
        let within = matches!(
            (x - y).abs().partial_cmp(&bound),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        );
        if !within {
            return Err(format!(
                "element {i}: gpu {x:e}, cpu {y:e}: |diff| {:e} > bound {bound:e}",
                (x - y).abs()
            ));
        }
        stats.differ += 1;
        stats.max_ulp = stats.max_ulp.max(ulp_distance(dtype, x, y));
    }
    Ok(stats)
}
