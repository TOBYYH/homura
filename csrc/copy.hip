// Copies, dtype conversions and fills of strided tensors.
//
// The conversions mirror homura's Element::cast (src/dtype.rs) bit for bit, including NaN
// payloads:
// - a float source is converted to f64 exactly (a NaN keeps its payload and becomes quiet), then
//   to the target: integers saturate to i64 (NaN is 0) and wrap, f32 rounds to nearest even
//   (a NaN keeps the top of its payload and becomes quiet), f16 and bf16 go through f32 and
//   then round to nearest even, bool is `!= 0` (NaN is true);
// - an integer source is converted to i64, then wraps to integers, rounds to f32 or f64, goes
//   through f32 to f16 and bf16, or becomes `!= 0`;
// - a bool source is 0 or 1.
// Equal dtypes copy the bytes. The f16 and bf16 roundings from f32 are those of the `half`
// crate's software conversions, which equal its hardware ones on every f32 input.
#include <cstdint>
#include <type_traits>

#include "common.hpp"

using hmr::guarded;

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

// The element offsets of linear index `i` (row-major over dst's sizes) in dst and in src.
__device__ inline void offsets(const hmr_tensor_desc &dst, const hmr_tensor_desc &src, int64_t i,
                               int64_t &od, int64_t &os) {
    od = dst.offset;
    os = src.offset;
    for (int k = dst.ndim - 1; k > 0; --k) {
        int64_t size = dst.sizes[k];
        int64_t q = i / size;
        int64_t idx = i - q * size;
        i = q;
        od += idx * dst.strides[k];
        os += idx * src.strides[k];
    }
    if (dst.ndim > 0) {
        od += i * dst.strides[0];
        os += i * src.strides[0];
    }
}

template <int D, int S>
__global__ void copy_kernel(hmr_tensor_desc dst, hmr_tensor_desc src, int64_t n) {
    auto *out = static_cast<typename Elem<D>::T *>(dst.data);
    const auto *in = static_cast<const typename Elem<S>::T *>(src.data);
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        int64_t od, os;
        offsets(dst, src, i, od, os);
        out[od] = convert<D, S>(in[os]);
    }
}

// Both operands flat and contiguous from their offsets: no index arithmetic.
template <int D, int S>
__global__ void copy_flat_kernel(typename Elem<D>::T *out, const typename Elem<S>::T *in,
                                 int64_t n) {
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        out[i] = convert<D, S>(in[i]);
    }
}

template <class T>
__global__ void fill_kernel(hmr_tensor_desc dst, T value, int64_t n) {
    auto *out = static_cast<T *>(dst.data);
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        int64_t od, os;
        offsets(dst, dst, i, od, os);
        out[od] = value;
    }
}

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 1 << 16;

dim3 blocks_for(int64_t n) {
    int64_t b = (n + kThreads - 1) / kThreads;
    return dim3(static_cast<unsigned>(b < kMaxBlocks ? b : kMaxBlocks));
}

bool is_flat(const hmr_tensor_desc &d) {
    return d.ndim == 0 || (d.ndim == 1 && d.strides[0] == 1);
}

int element_size(int dtype) {
    switch (dtype) {
        case HMR_BOOL:
        case HMR_U8:
        case HMR_I8:
            return 1;
        case HMR_I16:
        case HMR_F16:
        case HMR_BF16:
            return 2;
        case HMR_I32:
        case HMR_F32:
            return 4;
        default:
            return 8;
    }
}

// Checks a description from the Rust side; returns its number of elements in `n`.
hmr_status check_desc(const hmr_tensor_desc &d, int64_t &n) {
    if (d.dtype < HMR_BOOL || d.dtype > HMR_F64) {
        return fail(HMR_INVALID, "invalid dtype " + std::to_string(d.dtype));
    }
    if (d.ndim < 0 || d.ndim > HMR_MAX_DIMS) {
        return fail(HMR_INVALID, "invalid number of dimensions " + std::to_string(d.ndim));
    }
    n = 1;
    for (int k = 0; k < d.ndim; ++k) {
        if (d.sizes[k] < 0 || d.strides[k] < 0) return fail(HMR_INVALID, "negative size or stride");
        n *= d.sizes[k];
    }
    return HMR_OK;
}

template <int D, int S>
void launch_copy(const hmr_tensor_desc &dst, const hmr_tensor_desc &src, int64_t n,
                 hipStream_t stream) {
    if (is_flat(dst) && is_flat(src)) {
        auto *out = static_cast<typename Elem<D>::T *>(dst.data) + dst.offset;
        const auto *in = static_cast<const typename Elem<S>::T *>(src.data) + src.offset;
        hipLaunchKernelGGL((copy_flat_kernel<D, S>), blocks_for(n), dim3(kThreads), 0, stream, out,
                           in, n);
    } else {
        hipLaunchKernelGGL((copy_kernel<D, S>), blocks_for(n), dim3(kThreads), 0, stream, dst, src,
                           n);
    }
}

template <int D>
void launch_copy_from(const hmr_tensor_desc &dst, const hmr_tensor_desc &src, int64_t n,
                      hipStream_t stream) {
    switch (src.dtype) {
        case HMR_BOOL: return launch_copy<D, HMR_BOOL>(dst, src, n, stream);
        case HMR_U8: return launch_copy<D, HMR_U8>(dst, src, n, stream);
        case HMR_I8: return launch_copy<D, HMR_I8>(dst, src, n, stream);
        case HMR_I16: return launch_copy<D, HMR_I16>(dst, src, n, stream);
        case HMR_I32: return launch_copy<D, HMR_I32>(dst, src, n, stream);
        case HMR_I64: return launch_copy<D, HMR_I64>(dst, src, n, stream);
        case HMR_F16: return launch_copy<D, HMR_F16>(dst, src, n, stream);
        case HMR_BF16: return launch_copy<D, HMR_BF16>(dst, src, n, stream);
        case HMR_F32: return launch_copy<D, HMR_F32>(dst, src, n, stream);
        default: return launch_copy<D, HMR_F64>(dst, src, n, stream);
    }
}

void launch_copy_any(const hmr_tensor_desc &dst, const hmr_tensor_desc &src, int64_t n,
                     hipStream_t stream) {
    switch (dst.dtype) {
        case HMR_BOOL: return launch_copy_from<HMR_BOOL>(dst, src, n, stream);
        case HMR_U8: return launch_copy_from<HMR_U8>(dst, src, n, stream);
        case HMR_I8: return launch_copy_from<HMR_I8>(dst, src, n, stream);
        case HMR_I16: return launch_copy_from<HMR_I16>(dst, src, n, stream);
        case HMR_I32: return launch_copy_from<HMR_I32>(dst, src, n, stream);
        case HMR_I64: return launch_copy_from<HMR_I64>(dst, src, n, stream);
        case HMR_F16: return launch_copy_from<HMR_F16>(dst, src, n, stream);
        case HMR_BF16: return launch_copy_from<HMR_BF16>(dst, src, n, stream);
        case HMR_F32: return launch_copy_from<HMR_F32>(dst, src, n, stream);
        default: return launch_copy_from<HMR_F64>(dst, src, n, stream);
    }
}

template <class T>
void launch_fill(const hmr_tensor_desc &dst, uint64_t bits, int64_t n, hipStream_t stream) {
    T value;
    __builtin_memcpy(&value, &bits, sizeof value);
    hipLaunchKernelGGL((fill_kernel<T>), blocks_for(n), dim3(kThreads), 0, stream, dst, value, n);
}

}  // namespace
}  // namespace hmr

extern "C" {

hmr_status hmr_copy(hmr_context_t ctx, const hmr_tensor_desc *dst, const hmr_tensor_desc *src) {
    return guarded([&] {
        int64_t n = 0, ns = 0;
        if (hmr_status s = hmr::check_desc(*dst, n)) return s;
        if (hmr_status s = hmr::check_desc(*src, ns)) return s;
        bool same_sizes = dst->ndim == src->ndim;
        for (int k = 0; same_sizes && k < dst->ndim; ++k) {
            same_sizes = dst->sizes[k] == src->sizes[k];
        }
        if (!same_sizes) return hmr::fail(HMR_INVALID, "copy operands have different sizes");
        if (n == 0) return HMR_OK;
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(ctx->device)) return s;
        if (hmr_status s = hmr::injected_data_failure("hipGetLastError()")) return s;
        // The launch is checked with hipGetLastError, which must report this launch only.
        (void)hipGetLastError();
        hmr::launch_copy_any(*dst, *src, n, ctx->stream);
        HMR_HIP(hipGetLastError());
        return HMR_OK;
    });
}

hmr_status hmr_fill(hmr_context_t ctx, const hmr_tensor_desc *dst, uint64_t bits) {
    return guarded([&] {
        int64_t n = 0;
        if (hmr_status s = hmr::check_desc(*dst, n)) return s;
        if (n == 0) return HMR_OK;
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(ctx->device)) return s;
        if (hmr_status s = hmr::injected_data_failure("hipGetLastError()")) return s;
        (void)hipGetLastError();
        switch (hmr::element_size(dst->dtype)) {
            case 1: hmr::launch_fill<uint8_t>(*dst, bits, n, ctx->stream); break;
            case 2: hmr::launch_fill<uint16_t>(*dst, bits, n, ctx->stream); break;
            case 4: hmr::launch_fill<uint32_t>(*dst, bits, n, ctx->stream); break;
            default: hmr::launch_fill<uint64_t>(*dst, bits, n, ctx->stream); break;
        }
        HMR_HIP(hipGetLastError());
        return HMR_OK;
    });
}

hmr_status hmr_copy_device(hmr_context_t ctx, void *dst, const void *src, uint64_t nbytes) {
    return guarded([&] {
        if (nbytes == 0) return HMR_OK;
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(ctx->device)) return s;
        if (hmr_status s = hmr::injected_data_failure("hipMemcpyAsync(DeviceToDevice)")) return s;
        HMR_HIP(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDeviceToDevice, ctx->stream));
        return HMR_OK;
    });
}

}  // extern "C"
