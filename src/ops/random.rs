//! Random number generation.
//!
//! A single process-wide Philox4x32-10 generator (the counter-based generator also offered by
//! rocRAND) produces all random numbers. Sequences are reproducible after [`manual_seed`] but
//! differ from PyTorch's.

use std::sync::Mutex;

use crate::backend::cpu::dispatch_float;
use crate::dtype::sealed::Sealed;
use crate::shape::IntoShape;
use crate::{DType, Device, Shape, Tensor, TensorOptions};

/// PyTorch's default seed.
const DEFAULT_SEED: u64 = 67280421310721;

struct Generator {
    seed: u64,
    /// Number of 128-bit Philox blocks handed out since seeding.
    offset: u64,
}

static GENERATOR: Mutex<Generator> = Mutex::new(Generator {
    seed: DEFAULT_SEED,
    offset: 0,
});

/// One Philox4x32-10 block.
fn philox(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    const M0: u64 = 0xD251_1F53;
    const M1: u64 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    let (mut c, mut k) = (counter, key);
    for round in 0..10 {
        if round > 0 {
            k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
        }
        let p0 = M0 * c[0] as u64;
        let p1 = M1 * c[2] as u64;
        c = [
            (p1 >> 32) as u32 ^ c[1] ^ k[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ c[3] ^ k[1],
            p0 as u32,
        ];
    }
    c
}

/// Seeds the global random number generator.
pub fn manual_seed(seed: u64) {
    let mut g = GENERATOR.lock().unwrap_or_else(|e| e.into_inner());
    g.seed = seed;
    g.offset = 0;
}

/// Reserves `blocks` Philox blocks and returns a function producing them.
fn reserve(blocks: u64) -> impl Fn(u64) -> [u32; 4] {
    let (seed, start) = {
        let mut g = GENERATOR.lock().unwrap_or_else(|e| e.into_inner());
        let start = g.offset;
        g.offset += blocks;
        (g.seed, start)
    };
    let key = [seed as u32, (seed >> 32) as u32];
    move |i| {
        let n = start + i;
        philox([n as u32, (n >> 32) as u32, 0, 0], key)
    }
}

/// A uniform `f64` in `[0, 1)` from 53 random bits.
fn unit_f64(hi: u32, lo: u32) -> f64 {
    let bits = ((hi as u64) << 21) ^ (lo as u64 >> 11);
    bits as f64 * (1.0 / (1u64 << 53) as f64)
}

/// A uniform value in `[0, 1)` from the top `bits` bits of `x`.
fn unit_bits(x: u32, bits: u32) -> f64 {
    (x >> (32 - bits)) as f64 / (1u64 << bits) as f64
}

/// The significand precision of a floating point dtype.
///
/// A uniform `[0, 1)` sample of a dtype narrower than `f64` uses this many random bits, so every
/// sample is exactly representable and can't round up to 1.
fn float_digits(dtype: DType) -> u32 {
    match dtype {
        DType::F16 => 11,
        DType::BF16 => 8,
        DType::F32 => 24,
        _ => 53,
    }
}

enum Dist {
    Uniform(f64, f64),
    Normal(f64, f64),
    Int(i64, i64),
}

impl Dist {
    /// Panics if the parameters or the dtype are invalid. Runs before anything else, so rejected
    /// calls neither advance the generator nor behave differently on the meta device.
    fn validate(&self, dtype: DType) {
        match *self {
            Dist::Uniform(..) | Dist::Normal(..) if !dtype.is_floating_point() => {
                panic!("random floating point sampling is not implemented for {dtype} tensors")
            }
            Dist::Uniform(from, to) => {
                // As in PyTorch: both bounds and the width `to - from` (which overflows to
                // infinity for too wide `f64` intervals) must be finite values of `dtype`.
                let max = dtype.float_max();
                for (name, v) in [("from", from), ("to", to)] {
                    if !(-max..=max).contains(&v) {
                        panic!(
                            "uniform_ expects {name} to be a finite value of {dtype}, but got \
                             {name}={v}"
                        );
                    }
                }
                if from > to {
                    panic!(
                        "uniform_ expects to return a [from, to) range, but found from={from} > \
                         to={to}"
                    );
                }
                if to - from > max {
                    panic!(
                        "uniform_ expects to - from <= {max} for {dtype}, but got from={from} \
                         and to={to}"
                    );
                }
            }
            Dist::Normal(_, std) => {
                if std.is_nan() || std < 0.0 {
                    panic!("normal expects std >= 0.0, but found std {std}");
                }
            }
            Dist::Int(low, high) => {
                if low >= high {
                    panic!(
                        "random_ expects 'from' to be less than 'to', but got from={low} >= \
                         to={high}"
                    );
                }
                // Floating point dtypes are limited to the integers they all represent exactly,
                // so samples stay inside `[from, to)` and uniform after conversion. PyTorch only
                // warns outside this range (and may then return values outside the interval).
                let (min, max) = if dtype.is_floating_point() {
                    let exact = 1i64 << float_digits(dtype);
                    (-exact, exact)
                } else {
                    dtype.int_range()
                };
                if low < min || high - 1 > max {
                    panic!(
                        "random_ expects [from, to) to lie inside [{min}, {max}] for {dtype}, but \
                         got from={low} and to={high}"
                    );
                }
            }
        }
    }
}

/// Samples `n` floating point values for a `dtype` tensor, computed in `f64`.
fn sample_float(n: usize, dist: &Dist, dtype: DType) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    match *dist {
        Dist::Uniform(from, to) => {
            // f64 uses two words per value, narrower types one.
            let per_block = if dtype == DType::F64 { 2 } else { 4 };
            let bits = float_digits(dtype);
            let block = reserve(n.div_ceil(per_block) as u64);
            for i in 0..n.div_ceil(per_block) {
                let b = block(i as u64);
                for j in 0..per_block {
                    let u = if per_block == 2 {
                        unit_f64(b[2 * j], b[2 * j + 1])
                    } else {
                        unit_bits(b[j], bits)
                    };
                    out.push(from + u * (to - from));
                }
            }
        }
        Dist::Normal(mean, std) => {
            // Box-Muller: each block gives two normals.
            let block = reserve(n.div_ceil(2) as u64);
            for i in 0..n.div_ceil(2) {
                let b = block(i as u64);
                let u1 = 1.0 - unit_f64(b[0], b[1]);
                let u2 = unit_f64(b[2], b[3]);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                out.push(mean + std * r * theta.cos());
                out.push(mean + std * r * theta.sin());
            }
        }
        Dist::Int(..) => unreachable!("integer distributions use sample_int"),
    }
    out.truncate(n);
    out
}

/// Samples `n` integers uniformly from `[low, high)`, which must be nonempty.
///
/// Uses rejection sampling on 64-bit words, so every value is equally likely even for ranges
/// that don't divide `2^64`.
fn sample_int(n: usize, low: i64, high: i64) -> Vec<i64> {
    // `high - low` is in `1..=u64::MAX`, which may not fit in `i64`.
    let range = (high as i128 - low as i128) as u64;
    // Words below `2^64 mod range` would make small results more likely.
    let threshold = range.wrapping_neg() % range;
    let mut out = Vec::with_capacity(n);
    let mut words: Vec<u64> = vec![];
    while out.len() < n {
        if words.is_empty() {
            // Each block holds two words; usually no word is rejected, so reserve one block per
            // two values still needed.
            let blocks = (n - out.len()).div_ceil(2) as u64;
            let block = reserve(blocks);
            for i in (0..blocks).rev() {
                let b = block(i);
                words.push(((b[2] as u64) << 32) | b[3] as u64);
                words.push(((b[0] as u64) << 32) | b[1] as u64);
            }
        }
        let r = words.pop().unwrap();
        if r >= threshold {
            out.push((low as i128 + (r % range) as i128) as i64);
        }
    }
    out
}

/// Fills a new tensor with samples from `dist`.
fn random_tensor(shape: Shape, dtype: DType, device: Device, dist: Dist) -> Tensor {
    dist.validate(dtype);
    if device == Device::Meta {
        return Tensor::empty_on(shape, dtype, device);
    }
    let n = shape.num_elements();
    match dist {
        Dist::Int(low, high) => {
            let values = sample_int(n, low, high);
            Tensor::from_host(&values, shape, device).to(dtype)
        }
        _ => {
            let values = sample_float(n, &dist, dtype);
            dispatch_float!(dtype, T => {
                let v: Vec<T> = values
                    .iter()
                    .map(|&x| match dist {
                        // Rounding to `T` can reach the excluded upper bound; PyTorch maps it to
                        // the lower bound in that case too.
                        Dist::Uniform(from, to) if from < to && T::from_f64(x).to_f64() >= to => {
                            T::from_f64(from)
                        }
                        _ => T::from_f64(x),
                    })
                    .collect();
                Tensor::from_host(&v, shape, device)
            })
        }
    }
}

/// Uniform samples from `[0, 1)` with the default dtype and device.
pub fn rand(shape: impl IntoShape) -> Tensor {
    rand_with(shape, TensorOptions::default())
}

/// [`rand`] with the given dtype and/or device.
pub fn rand_with(shape: impl IntoShape, options: impl Into<TensorOptions>) -> Tensor {
    let o = options.into();
    random_tensor(
        shape.into_shape(),
        o.float_dtype(),
        o.device(),
        Dist::Uniform(0.0, 1.0),
    )
}

/// Samples from the standard normal distribution with the default dtype and device.
pub fn randn(shape: impl IntoShape) -> Tensor {
    randn_with(shape, TensorOptions::default())
}

/// [`randn`] with the given dtype and/or device.
pub fn randn_with(shape: impl IntoShape, options: impl Into<TensorOptions>) -> Tensor {
    let o = options.into();
    random_tensor(
        shape.into_shape(),
        o.float_dtype(),
        o.device(),
        Dist::Normal(0.0, 1.0),
    )
}

/// Uniform integers from `[low, high)` as an `I64` tensor on the default device.
///
/// # Panics
///
/// Panics if the interval is empty or doesn't lie inside the range of the dtype. For floating
/// point dtypes that range is the integers they represent exactly: `±2^11` for `F16`, `±2^8` for
/// `BF16`, `±2^24` for `F32` and `±2^53` for `F64`.
pub fn randint(low: i64, high: i64, shape: impl IntoShape) -> Tensor {
    randint_with(low, high, shape, TensorOptions::default())
}

/// [`randint`] with the given dtype and/or device (the dtype defaults to `I64`).
pub fn randint_with(
    low: i64,
    high: i64,
    shape: impl IntoShape,
    options: impl Into<TensorOptions>,
) -> Tensor {
    let o = options.into();
    let dtype = o.dtype.unwrap_or(DType::I64);
    random_tensor(shape.into_shape(), dtype, o.device(), Dist::Int(low, high))
}

/// [`rand`] with the shape, dtype and device of `t`.
pub fn rand_like(t: &Tensor) -> Tensor {
    rand_with(t.shape(), (t.dtype(), t.device()))
}

/// [`randn`] with the shape, dtype and device of `t`.
pub fn randn_like(t: &Tensor) -> Tensor {
    randn_with(t.shape(), (t.dtype(), t.device()))
}

impl Tensor {
    /// Fills `self` with uniform samples from `[from, to)`.
    ///
    /// # Panics
    ///
    /// Panics if `from > to`, or if `from`, `to` or `to - from` is not a finite value of the dtype
    /// of `self`, as PyTorch does.
    pub fn uniform_(&self, from: f64, to: f64) -> &Self {
        crate::autograd::check_inplace(self, &[]);
        let r = random_tensor(
            self.shape(),
            self.dtype(),
            self.device(),
            Dist::Uniform(from, to),
        );
        self.write_copy(&r);
        self
    }

    /// Fills `self` with samples from a normal distribution.
    pub fn normal_(&self, mean: f64, std: f64) -> &Self {
        crate::autograd::check_inplace(self, &[]);
        let r = random_tensor(
            self.shape(),
            self.dtype(),
            self.device(),
            Dist::Normal(mean, std),
        );
        self.write_copy(&r);
        self
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};

    use super::*;

    #[test]
    fn philox_known_answer() {
        // Known-answer vectors from the Random123 distribution (kat_vectors).
        assert_eq!(
            philox([0; 4], [0; 2]),
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
        );
        assert_eq!(
            philox([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
        );
        assert_eq!(
            philox(
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0xa4093822, 0x299f31d0]
            ),
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
        );
    }

    #[test]
    fn unit_ranges() {
        assert_eq!(unit_f64(0, 0), 0.0);
        assert!(unit_f64(u32::MAX, u32::MAX) < 1.0);
        for bits in [8, 11, 24] {
            assert!(unit_bits(u32::MAX, bits) < 1.0);
        }
        assert_eq!(
            f16::from_f64(unit_bits(u32::MAX, 11)).to_f64(),
            1.0 - 2f64.powi(-11)
        );
        assert_eq!(
            bf16::from_f64(unit_bits(u32::MAX, 8)).to_f64(),
            1.0 - 2f64.powi(-8)
        );
    }

    #[test]
    fn int_sampling_covers_extreme_ranges() {
        assert!(
            sample_int(64, i64::MIN, i64::MAX)
                .iter()
                .all(|&v| v < i64::MAX)
        );
        assert_eq!(sample_int(4, i64::MAX - 1, i64::MAX), [i64::MAX - 1; 4]);
        let v = sample_int(1000, -2, 1);
        assert!(v.iter().all(|x| (-2..1).contains(x)));
        assert!((-2..1).all(|k| v.contains(&k)));
    }
}
