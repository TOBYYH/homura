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

/// A uniform `f32` in `[0, 1)` from 24 random bits.
fn unit_f32(x: u32) -> f64 {
    (x >> 8) as f64 * (1.0 / (1u32 << 24) as f64)
}

enum Dist {
    Uniform(f64, f64),
    Normal(f64, f64),
    Int(i64, i64),
}

/// Samples `n` values; floats are computed in `f64` and returned as such.
fn sample(n: usize, dist: &Dist, dtype: DType) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    match *dist {
        Dist::Uniform(from, to) => {
            // f64 uses two words per value, narrower types one.
            let per_block = if dtype == DType::F64 { 2 } else { 4 };
            let block = reserve(n.div_ceil(per_block) as u64);
            for i in 0..n.div_ceil(per_block) {
                let b = block(i as u64);
                for j in 0..per_block {
                    let u = if per_block == 2 {
                        unit_f64(b[2 * j], b[2 * j + 1])
                    } else {
                        unit_f32(b[j])
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
        Dist::Int(low, high) => {
            let range = (high - low) as u64;
            let block = reserve(n.div_ceil(2) as u64);
            for i in 0..n.div_ceil(2) {
                let b = block(i as u64);
                for j in 0..2 {
                    let r = ((b[2 * j] as u64) << 32) | b[2 * j + 1] as u64;
                    out.push((low + (r % range) as i64) as f64);
                }
            }
        }
    }
    out.truncate(n);
    out
}

/// Fills a new tensor with samples from `dist`.
fn random_tensor(shape: Shape, dtype: DType, device: Device, dist: Dist) -> Tensor {
    if device == Device::Meta {
        return Tensor::empty_on(shape, dtype, device);
    }
    let n = shape.num_elements();
    match dist {
        Dist::Int(low, high) => {
            if low >= high {
                panic!(
                    "random_ expects 'from' to be less than 'to', but got from={low} >= to={high}"
                );
            }
            let values: Vec<i64> = sample(n, &dist, dtype)
                .into_iter()
                .map(|v| v as i64)
                .collect();
            Tensor::from_host(&values, shape, device).to(dtype)
        }
        _ => {
            if !dtype.is_floating_point() {
                panic!("random floating point sampling is not implemented for {dtype} tensors");
            }
            let values = sample(n, &dist, dtype);
            dispatch_float!(dtype, T => {
                let v: Vec<T> = values.iter().map(|&x| T::from_f64(x)).collect();
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
    pub fn uniform_(&self, from: f64, to: f64) -> &Self {
        let r = random_tensor(
            self.shape(),
            self.dtype(),
            self.device(),
            Dist::Uniform(from, to),
        );
        self.copy_(&r)
    }

    /// Fills `self` with samples from a normal distribution.
    pub fn normal_(&self, mean: f64, std: f64) -> &Self {
        let r = random_tensor(
            self.shape(),
            self.dtype(),
            self.device(),
            Dist::Normal(mean, std),
        );
        self.copy_(&r)
    }
}

#[cfg(test)]
mod tests {
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
        assert!(unit_f32(u32::MAX) < 1.0);
    }
}
