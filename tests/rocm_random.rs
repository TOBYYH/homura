//! Random sampling on a ROCm GPU (M4 step 3, item 4): uniform samples, integers and Dropout's
//! masks equal the CPU's bit for bit for the same seed, also when calls on both devices
//! alternate; normal samples are within the transcendental bound of M4.12.
#![cfg(feature = "rocm")]

#[path = "common/rocm.rs"]
mod rocm;

use std::sync::Mutex;

use homura::nn::functional::dropout;
use homura::{DType, Device, Tensor};
use rocm::{GPU, UlpStats, assert_bitwise, check_transcendental, check_transcendental_in, floats};

/// The generator is global: tests that seed it run one at a time.
static GENERATOR: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    GENERATOR.lock().unwrap_or_else(|e| e.into_inner())
}

const FLOATS: [DType; 4] = [DType::F16, DType::BF16, DType::F32, DType::F64];
const SIZES: [usize; 8] = [0, 1, 2, 3, 5, 1031, 4096, (1 << 20) + 3];

/// `f(device)` after seeding, on the CPU and on the GPU.
fn both(seed: u64, f: impl Fn(Device) -> Tensor) -> (Tensor, Tensor) {
    homura::manual_seed(seed);
    let cpu = f(Device::Cpu);
    homura::manual_seed(seed);
    let gpu = f(GPU);
    (gpu, cpu)
}

#[test]
fn uniform_samples_are_the_cpus() {
    let _g = lock();
    for dtype in FLOATS {
        for n in SIZES {
            let (g, c) = both(7, |d| homura::rand_with([n], (dtype, d)));
            assert_bitwise(&g, &c, &format!("rand {dtype} [{n}]"));
            // The excluded upper bound: in F16 and BF16 many samples of [1, 1 + 2^-3) round up
            // to it and become the lower bound.
            for (from, to) in [(-3.0, 5.0), (1.0, 1.125), (-1e-3, 0.0), (0.0, 65504.0)] {
                let (g, c) = both(8, |d| {
                    homura::zeros_with([n], (dtype, d))
                        .uniform_(from, to)
                        .clone()
                });
                assert_bitwise(&g, &c, &format!("uniform_ {dtype} [{n}] [{from}, {to})"));
            }
        }
        // A strided tensor: sampled contiguously, then written into the view.
        let (g, c) = both(9, |d| {
            let t = homura::zeros_with([4, 6], (dtype, d)).transpose(0, 1);
            t.uniform_(0.0, 1.0);
            t
        });
        assert_bitwise(&g, &c, &format!("uniform_ {dtype} strided"));
    }
}

#[test]
fn integer_samples_are_the_cpus() {
    let _g = lock();
    let ranges = [
        (0, 2),                    // a power of two: no rejection
        (0, 1 << 40),              // also
        (-5, 6),                   // rare rejection
        (0, 3),                    // also
        (i64::MIN, 1),             // about half of the words rejected
        (i64::MIN, i64::MAX),      // almost the whole range
        (-(1 << 62) - 1, 1 << 62), // about a quarter rejected
        (i64::MAX - 1, i64::MAX),  // a single value
    ];
    for (low, high) in ranges {
        for n in SIZES {
            let (g, c) = both(6, |d| homura::randint_with(low, high, [n], d));
            assert_bitwise(&g, &c, &format!("randint [{low}, {high}) [{n}]"));
            // The next sample shows that both advanced the generator by the same blocks.
            let (g, c) = both(6, |d| {
                homura::randint_with(low, high, [n], d);
                homura::rand_with([5], (DType::F64, Device::Cpu)).to(GPU)
            });
            assert_bitwise(&g, &c, &format!("after randint [{low}, {high}) [{n}]"));
        }
    }
    for dtype in [
        DType::U8,
        DType::I16,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let low = if dtype == DType::U8 { 0 } else { -100 };
        let (g, c) = both(10, |d| homura::randint_with(low, 100, [3, 333], (dtype, d)));
        assert_bitwise(&g, &c, &format!("randint {dtype}"));
    }
    // The reviewer's probe (round 114, R129): three blocks for one value, then this F64 sample.
    homura::manual_seed(6);
    homura::randint_with(i64::MIN, 1, [1], GPU);
    let next = homura::rand_with([1], DType::F64).item::<f64>();
    assert_eq!(next, 0.6159043529544902);
}

#[test]
fn alternating_devices_follow_one_sequence() {
    let _g = lock();
    let run = |devices: [Device; 4]| {
        homura::manual_seed(42);
        let a = homura::randint_with(i64::MIN, 1, [7], devices[0]).to(Device::Cpu);
        let b = homura::rand_with([9], (DType::F32, devices[1])).to(Device::Cpu);
        let x = homura::ones_with([1001], (DType::F32, devices[2]));
        let c = dropout(&x, 0.3, true).to(Device::Cpu);
        let d = homura::randint_with(0, 7, [13], devices[3]).to(Device::Cpu);
        [a, b, c, d]
    };
    let cpu = run([Device::Cpu; 4]);
    for devices in [
        [GPU; 4],
        [GPU, Device::Cpu, GPU, Device::Cpu],
        [Device::Cpu, GPU, Device::Cpu, GPU],
    ] {
        let mixed = run(devices);
        for (i, (m, c)) in mixed.iter().zip(&cpu).enumerate() {
            assert_bitwise(&m.to(GPU), c, &format!("{devices:?}: call {i}"));
        }
    }
}

#[test]
fn dropout_masks_are_the_cpus() {
    let _g = lock();
    for dtype in FLOATS {
        for p in [0.1, 0.5, 0.9] {
            let (g, c) = both(11, |d| {
                let x = homura::ones_with([3, 1031], (dtype, d)).mul(3.0);
                dropout(&x, p, true)
            });
            assert_bitwise(&g, &c, &format!("dropout {dtype} p={p}"));
        }
    }
}

/// Normal sampling computes in f64 for every dtype (Box-Muller), so its bound has f64's unit
/// roundoff plus one rounding to the output dtype (review round 126, R150).
const NORMAL_U: f64 = f64::EPSILON / 2.0;
const NORMAL_K: f64 = 6.0;

/// M4.12's `M = |mean| + |std| r (1 + 2 pi)` for each of `n` normal samples drawn after seeding
/// with `seed`, with each block's `r` from the CPU's unrounded f64 samples: the same seed with an
/// even length gives every block's pair (cos, sin), whatever the dtype, as the CPU reserves the
/// same blocks for every dtype (review round 126, R150).
fn normal_sizes(seed: u64, n: usize, mean: f64, std: f64) -> Vec<f64> {
    homura::manual_seed(seed);
    let y = floats(homura::zeros_with([n + n % 2], DType::F64).normal_(mean, std));
    (0..n)
        .map(|i| {
            let pair = i - i % 2;
            let (a, b) = ((y[pair] - mean) / std, (y[pair + 1] - mean) / std);
            // Margins for the rounding of y and of this computation, in f64.
            let r =
                (a * a + b * b).sqrt() * (1.0 + 2f64.powi(-40)) + 2f64.powi(-40) * mean.abs() / std;
            mean.abs() + std * r * (1.0 + 2.0 * std::f64::consts::PI)
        })
        .collect()
}

#[test]
fn normal_samples_stay_within_the_bound() {
    let _g = lock();
    for dtype in FLOATS {
        let mut stats = UlpStats::default();
        for (mean, std) in [(0.0, 1.0), (3.0, 0.5), (-1e3, 10.0), (0.0, 1e-3)] {
            for n in [1, 2, 7, 4096, (1 << 20) + 1] {
                let (g, c) = both(12, |d| {
                    homura::zeros_with([n], (dtype, d))
                        .normal_(mean, std)
                        .clone()
                });
                let m = normal_sizes(12, n, mean, std);
                match check_transcendental_in(&g, &c, &m, NORMAL_K, NORMAL_U) {
                    Ok(s) => stats.merge(s),
                    Err(e) => panic!("normal {dtype} ({mean}, {std}) [{n}]: {e}"),
                }
                // One element off by twice the bound's first term plus 3 output ulps fails; for
                // the narrower dtypes also 16 output ulps, which the bound of their computation
                // type (F32) would accept.
                let ys = floats(&c);
                let i = n / 2;
                let off = |delta: f64| {
                    let mut v = ys.clone();
                    v[i] += delta;
                    Tensor::from_slice(&v, [n]).to(dtype)
                };
                let u_out = rocm::ulp(dtype, ys[i]);
                let mut deltas = vec![2.0 * NORMAL_K * NORMAL_U * m[i] + 3.0 * u_out];
                if dtype != DType::F64 {
                    deltas.push(16.0 * u_out);
                }
                for delta in deltas {
                    let what = format!("normal {dtype} ({mean}, {std}) [{n}] off by {delta:e}");
                    let bad = off(delta);
                    assert!(
                        check_transcendental_in(&g, &bad, &m, NORMAL_K, NORMAL_U).is_err(),
                        "{what}"
                    );
                }
            }
        }
        println!(
            "ulp {:<12} {dtype:<5} k={NORMAL_K}: max {} ulp, {} of {} elements differ",
            "normal", stats.max_ulp, stats.differ, stats.elements
        );
        // Both advance the generator by the same blocks.
        let (g, c) = both(13, |d| {
            homura::randn_with([5], (dtype, d));
            homura::rand_with([3], DType::F64).to(GPU)
        });
        assert_bitwise(&g, &c, &format!("after randn {dtype}"));
    }
}

/// The reviewer's probe: an F32 normal sample 16 ulp off at 1 (M for r = 1) fails the f64 bound
/// and passed the bound of F32's computation type.
#[test]
fn the_normal_bound_uses_f64() {
    let cpu = Tensor::from_slice(&[1f32], [1]);
    let off = Tensor::from_slice(&[1.0 + 16.0 * f32::EPSILON], [1]).to(GPU);
    let m = [1.0 + 2.0 * std::f64::consts::PI];
    assert!(check_transcendental_in(&off, &cpu, &m, NORMAL_K, NORMAL_U).is_err());
    assert!(check_transcendental(&off, &cpu, &m, NORMAL_K).is_ok());
    let one_ulp = Tensor::from_slice(&[1.0 + f32::EPSILON], [1]).to(GPU);
    assert!(check_transcendental_in(&one_ulp, &cpu, &m, NORMAL_K, NORMAL_U).is_ok());
}

#[test]
fn integer_sampling_needs_no_more_memory_than_its_result() {
    let _g = lock();
    // The workspace and output are freed afterwards: nothing stays allocated.
    let before = GPU.memory_stats().allocated;
    let t = homura::randint_with(i64::MIN, 1, [100_000], GPU);
    drop(t);
    assert_eq!(GPU.memory_stats().allocated, before);
    let _ = Tensor::from_slice(&[0i64], [1]);
}
