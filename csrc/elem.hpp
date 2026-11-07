// Element types and the conversions between them, shared by the kernels: they mirror homura's
// Element::cast (src/dtype.rs) bit for bit, including NaN payloads:
// - a float source is converted to f64 exactly (a NaN keeps its payload and becomes quiet), then
//   to the target: integers saturate to i64 (NaN is 0) and wrap, f32 rounds to nearest even
//   (a NaN keeps the top of its payload and becomes quiet), f16 and bf16 go through f32 and
//   then round to nearest even, bool is `!= 0` (NaN is true);
// - an integer source is converted to i64, then wraps to integers, rounds to f32 or f64, goes
//   through f32 to f16 and bf16, or becomes `!= 0`;
// - a bool source is 0 or 1.
// Equal dtypes copy the bytes. The f16 and bf16 roundings from f32 are those of the `half`
// crate's software conversions, which equal its hardware ones on every f32 input.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

#include "homura_rocm.h"

namespace hmr {
namespace {

enum Kind { kBool, kInt, kFloat };

template <int DT>
struct Elem;
template <>
struct Elem<HMR_BOOL> {
    using T = uint8_t;
    static constexpr Kind kind = kBool;
};
template <>
struct Elem<HMR_U8> {
    using T = uint8_t;
    static constexpr Kind kind = kInt;
};
template <>
struct Elem<HMR_I8> {
    using T = int8_t;
    static constexpr Kind kind = kInt;
};
template <>
struct Elem<HMR_I16> {
    using T = int16_t;
    static constexpr Kind kind = kInt;
};
template <>
struct Elem<HMR_I32> {
    using T = int32_t;
    static constexpr Kind kind = kInt;
};
template <>
struct Elem<HMR_I64> {
    using T = int64_t;
    static constexpr Kind kind = kInt;
};
template <>
struct Elem<HMR_F16> {
    using T = uint16_t;
    static constexpr Kind kind = kFloat;
};
template <>
struct Elem<HMR_BF16> {
    using T = uint16_t;
    static constexpr Kind kind = kFloat;
};
template <>
struct Elem<HMR_F32> {
    using T = float;
    static constexpr Kind kind = kFloat;
};
template <>
struct Elem<HMR_F64> {
    using T = double;
    static constexpr Kind kind = kFloat;
};

__device__ inline uint64_t bits64(double v) {
    return static_cast<uint64_t>(__double_as_longlong(v));
}
__device__ inline double from_bits64(uint64_t b) {
    return __longlong_as_double(static_cast<long long>(b));
}

// 2^e as a double, for -1022 <= e <= 1023.
__device__ inline double pow2(int e) { return from_bits64(static_cast<uint64_t>(e + 1023) << 52); }

// An IEEE binary format with `man` mantissa bits and exponent bias `bias`, as f64: exact, with
// the arithmetic done on integers and powers of two so that no denormal mode matters. A NaN
// becomes a quiet f64 NaN with the payload shifted to the top (`half`'s f16_to_f64,
// bf16_to_f64, and Rust's f32 as f64).
__device__ inline double decode(uint64_t sign, uint32_t exp, uint64_t mantissa, int man,
                                int bias, uint32_t exp_max) {
    uint64_t s = sign << 63;
    if (exp == exp_max) {
        if (mantissa == 0) return from_bits64(s | 0x7FF0000000000000ull);
        return from_bits64(s | 0x7FF8000000000000ull | (mantissa << (52 - man)));
    }
    double mag;
    if (exp == 0) {
        // Subnormal (or zero): mantissa * 2^(1 - bias - man), split so each factor is normal.
        mag = static_cast<double>(mantissa) * pow2(1 - bias) * pow2(-man);
    } else {
        mag = static_cast<double>(mantissa | (1ull << man)) * pow2(static_cast<int>(exp) - bias) *
              pow2(-man);
    }
    return from_bits64(bits64(mag) | s);
}

template <int S>
__device__ inline double to_f64(typename Elem<S>::T x) {
    if constexpr (S == HMR_F64) {
        return x;
    } else if constexpr (S == HMR_F32) {
        uint32_t b = __float_as_uint(x);
        return decode(b >> 31, (b >> 23) & 0xFF, b & 0x7FFFFF, 23, 127, 0xFF);
    } else if constexpr (S == HMR_F16) {
        return decode(x >> 15, (x >> 10) & 0x1F, x & 0x3FF, 10, 15, 0x1F);
    } else {
        static_assert(S == HMR_BF16);
        return decode(x >> 15, (x >> 7) & 0xFF, x & 0x7F, 7, 127, 0xFF);
    }
}

// Rust's `v as f32`: round to nearest even; a NaN keeps the top 23 bits of its mantissa and
// becomes quiet, as with x86's cvtsd2ss, which the CPU results come from. The hardware conversion
// does the same (tests/rocm_copy.rs, nan_payloads_convert_like_the_cpu).
__device__ inline float f64_to_f32(double v) { return __double2float_rn(v); }

// `half`'s f32_to_f16_fallback: round to nearest even; a NaN keeps the top 10 bits of its
// mantissa and becomes quiet.
__device__ inline uint16_t f32_to_f16(float f) {
    uint32_t x = __float_as_uint(f);
    uint32_t sign = x & 0x80000000u;
    uint32_t exp = x & 0x7F800000u;
    uint32_t man = x & 0x007FFFFFu;
    if (exp == 0x7F800000u) {
        uint32_t nan_bit = man == 0 ? 0 : 0x0200u;
        return static_cast<uint16_t>((sign >> 16) | 0x7C00u | nan_bit | (man >> 13));
    }
    uint32_t half_sign = sign >> 16;
    int32_t half_exp = static_cast<int32_t>(exp >> 23) - 127 + 15;
    if (half_exp >= 0x1F) return static_cast<uint16_t>(half_sign | 0x7C00u);
    if (half_exp <= 0) {
        if (14 - half_exp > 24) return static_cast<uint16_t>(half_sign);
        uint32_t m = man | 0x00800000u;
        uint32_t half_man = m >> (14 - half_exp);
        uint32_t round_bit = 1u << (13 - half_exp);
        if ((m & round_bit) != 0 && (m & (3 * round_bit - 1)) != 0) half_man += 1;
        return static_cast<uint16_t>(half_sign | half_man);
    }
    uint32_t h = half_sign | (static_cast<uint32_t>(half_exp) << 10) | (man >> 13);
    uint32_t round_bit = 0x00001000u;
    if ((man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0) h += 1;
    return static_cast<uint16_t>(h);
}

// `half`'s f32_to_bf16: round to nearest even; a NaN keeps the top 7 bits of its mantissa and
// becomes quiet.
__device__ inline uint16_t f32_to_bf16(float f) {
    uint32_t x = __float_as_uint(f);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((x >> 16) | 0x0040u);
    uint32_t round_bit = 0x00008000u;
    if ((x & round_bit) != 0 && (x & (3 * round_bit - 1)) != 0) {
        return static_cast<uint16_t>((x >> 16) + 1);
    }
    return static_cast<uint16_t>(x >> 16);
}

// Rust's `v as i64`: toward zero, saturating, NaN is 0.
__device__ inline int64_t f64_to_i64(double v) {
    if (v != v) return 0;
    if (v >= 0x1p63) return INT64_MAX;
    if (v <= -0x1p63) return INT64_MIN;
    return static_cast<int64_t>(v);
}

// Two's complement truncation of `v` to the integer type T.
template <class T>
__device__ inline T wrap(int64_t v) {
    using U = std::make_unsigned_t<T>;
    U u = static_cast<U>(static_cast<uint64_t>(v));
    T t;
    __builtin_memcpy(&t, &u, sizeof t);
    return t;
}

template <int D>
__device__ inline typename Elem<D>::T from_f32(float f) {
    if constexpr (D == HMR_F16) {
        return f32_to_f16(f);
    } else {
        static_assert(D == HMR_BF16);
        return f32_to_bf16(f);
    }
}

template <int D>
__device__ inline typename Elem<D>::T from_f64(double v) {
    if constexpr (D == HMR_F64) {
        return v;
    } else if constexpr (D == HMR_F32) {
        return f64_to_f32(v);
    } else if constexpr (D == HMR_F16 || D == HMR_BF16) {
        return from_f32<D>(f64_to_f32(v));
    } else if constexpr (D == HMR_BOOL) {
        return v != 0.0;  // NaN too
    } else {
        return wrap<typename Elem<D>::T>(f64_to_i64(v));
    }
}

template <int D>
__device__ inline typename Elem<D>::T from_i64(int64_t v) {
    if constexpr (D == HMR_F64) {
        return static_cast<double>(v);
    } else if constexpr (D == HMR_F32) {
        return static_cast<float>(v);
    } else if constexpr (D == HMR_F16 || D == HMR_BF16) {
        return from_f32<D>(static_cast<float>(v));
    } else if constexpr (D == HMR_BOOL) {
        return v != 0;
    } else {
        return wrap<typename Elem<D>::T>(v);
    }
}

template <int D>
__device__ inline typename Elem<D>::T from_bool(bool v) {
    if constexpr (D == HMR_F64) {
        return v ? 1.0 : 0.0;
    } else if constexpr (D == HMR_F32) {
        return v ? 1.0f : 0.0f;
    } else if constexpr (D == HMR_F16) {
        return v ? 0x3C00 : 0;
    } else if constexpr (D == HMR_BF16) {
        return v ? 0x3F80 : 0;
    } else {
        return v ? 1 : 0;
    }
}

template <int D, int S>
__device__ inline typename Elem<D>::T convert(typename Elem<S>::T x) {
    if constexpr (D == S) {
        return x;
    } else if constexpr (Elem<S>::kind == kFloat) {
        return from_f64<D>(to_f64<S>(x));
    } else if constexpr (Elem<S>::kind == kInt) {
        return from_i64<D>(static_cast<int64_t>(x));
    } else {
        return from_bool<D>(x != 0);
    }
}

}  // namespace
}  // namespace hmr
