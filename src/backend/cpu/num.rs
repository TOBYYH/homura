//! Numeric traits used by the CPU kernels.

use half::{bf16, f16};

use crate::Element;

/// The type floating point math is done in: `f32` or `f64`. Most math uses the inherent methods
/// of the concrete type; this trait adds what generic kernels need on top.
pub(crate) trait Float: Copy + 'static {
    const ZERO: Self;
    const ONE: Self;
    /// Converts a constant.
    fn of(v: f64) -> Self;
    fn erf(self) -> Self;
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn of(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn erf(self) -> Self {
        erf(self as f64) as f32
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn of(v: f64) -> Self {
        v
    }
    #[inline]
    fn erf(self) -> Self {
        erf(self)
    }
}

/// A floating point element type and the type its math is done in. Like PyTorch's CPU kernels,
/// `f16` and `bf16` are computed in `f32`.
pub(crate) trait FloatElement: Element {
    type C: Float;
    fn up(self) -> Self::C;
    fn down(c: Self::C) -> Self;
}

impl FloatElement for f32 {
    type C = f32;
    #[inline]
    fn up(self) -> f32 {
        self
    }
    #[inline]
    fn down(c: f32) -> Self {
        c
    }
}

impl FloatElement for f64 {
    type C = f64;
    #[inline]
    fn up(self) -> f64 {
        self
    }
    #[inline]
    fn down(c: f64) -> Self {
        c
    }
}

impl FloatElement for f16 {
    type C = f32;
    #[inline]
    fn up(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn down(c: f32) -> Self {
        f16::from_f32(c)
    }
}

impl FloatElement for bf16 {
    type C = f32;
    #[inline]
    fn up(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn down(c: f32) -> Self {
        bf16::from_f32(c)
    }
}

/// Integer element types. Arithmetic wraps on overflow, like PyTorch.
pub(crate) trait IntElement: Element + Ord {
    const ZERO: Self;
    const ONE: Self;
    fn wadd(self, o: Self) -> Self;
    fn wsub(self, o: Self) -> Self;
    fn wmul(self, o: Self) -> Self;
    fn wneg(self) -> Self;
    fn wabs(self) -> Self;
}

macro_rules! impl_int {
    ($($t:ty: |$a:ident| $abs:expr),*) => {$(
        impl IntElement for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            #[inline]
            fn wadd(self, o: Self) -> Self {
                self.wrapping_add(o)
            }
            #[inline]
            fn wsub(self, o: Self) -> Self {
                self.wrapping_sub(o)
            }
            #[inline]
            fn wmul(self, o: Self) -> Self {
                self.wrapping_mul(o)
            }
            #[inline]
            fn wneg(self) -> Self {
                self.wrapping_neg()
            }
            #[inline]
            fn wabs(self) -> Self {
                let $a = self;
                $abs
            }
        }
    )*};
}

impl_int!(
    u8: |a| a,
    i8: |a| a.wrapping_abs(),
    i16: |a| a.wrapping_abs(),
    i32: |a| a.wrapping_abs(),
    i64: |a| a.wrapping_abs()
);

/// Integer power with wrapping multiplication. Negative exponents give `1 / base` truncated to an
/// integer, as PyTorch does for integer tensors.
pub(crate) fn int_pow<T: IntElement>(base: T, exp: T) -> T {
    let e = crate::dtype::sealed::Sealed::to_i64(exp);
    if e < 0 {
        let b = crate::dtype::sealed::Sealed::to_i64(base);
        return match b {
            1 => T::ONE,
            -1 => {
                if e % 2 == 0 {
                    T::ONE
                } else {
                    base
                }
            }
            _ => T::ZERO,
        };
    }
    let (mut result, mut b, mut e) = (T::ONE, base, e as u64);
    while e > 0 {
        if e & 1 == 1 {
            result = result.wmul(b);
        }
        b = b.wmul(b);
        e >>= 1;
    }
    result
}

/// The error function, within 3 ulp in `f64`. (`f64::erf` is not stable yet.)
pub(crate) fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let ax = x.abs();
    let r = if ax < 1.5 {
        // erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (2n+1)!!, a series of positive
        // terms (no cancellation).
        let x2 = ax * ax;
        let mut term = ax;
        let mut sum = ax;
        let mut n = 0.0;
        while term > 1e-17 * sum {
            n += 1.0;
            term *= 2.0 * x2 / (2.0 * n + 1.0);
            sum += term;
        }
        std::f64::consts::FRAC_2_SQRT_PI * (-x2).exp() * sum
    } else if ax < 6.0 {
        // Continued fraction erfc(x) = exp(-x^2) / sqrt(pi) / (x + (1/2)/(x + 1/(x + ...))).
        let mut f = ax;
        for k in (1..=80).rev() {
            f = ax + (k as f64 * 0.5) / f;
        }
        1.0 - (-ax * ax).exp() / (std::f64::consts::PI.sqrt() * f)
    } else {
        1.0
    };
    r.copysign(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_values() {
        let cases = [
            (0.0, 0.0),
            (1e-10, 1.1283791670955126e-10),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (2.0, 0.9953222650189527),
            (2.4999, 0.9995928300996666),
            (2.5, 0.999593047982555),
            (3.0, 0.9999779095030014),
            (4.0, 0.9999999845827421),
            (-1.5, -0.9661051464753108),
        ];
        for (x, want) in cases {
            let got = erf(x);
            assert!(
                (got - want).abs() <= 3.0 * f64::EPSILON * want.abs(),
                "erf({x}) = {got}"
            );
        }
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert!(erf(f64::NAN).is_nan());
    }

    #[test]
    fn integer_power() {
        assert_eq!(int_pow(3i64, 4), 81);
        assert_eq!(int_pow(-2i32, 3), -8);
        assert_eq!(int_pow(2i8, 7), -128);
        assert_eq!(int_pow(-1i64, -3), -1);
        assert_eq!(int_pow(5i64, -1), 0);
    }
}
