// Elementwise kernels: unary and binary ops, binary ops with a scalar, the optimizers' fused
// steps, where, triangle and flip.
//
// They compute what the CPU reference backend computes (src/backend/cpu/mod.rs), operation by
// operation in the same order: floating point math in f32 for F16, BF16 and F32 and in f64 for
// F64, rounding to the dtype once at the end; fused multiply-adds only where the CPU calls
// mul_add (the library is compiled with -ffp-contract=off); wrapping integer arithmetic. The
// IEEE operations (+, -, *, /, sqrt, fma) are correctly rounded on both sides, so their results
// are bitwise equal; exp, log, sin, cos, tanh, log1p and pow come from the device library and
// may differ from the CPU's libm in the last place.
#include <cmath>
#include <tuple>
#include <utility>

#include "elem.hpp"
#include "kernel.hpp"

using hmr::guarded;

namespace hmr {
namespace {

// The type a floating point dtype computes in, as the CPU's FloatElement::C.
template <int DT>
struct Comp {
    using C = float;
};
template <>
struct Comp<HMR_F64> {
    using C = double;
};

// Exact widening to the computation type (half's to_f32).
template <int DT>
__device__ inline typename Comp<DT>::C up(typename Elem<DT>::T v) {
    if constexpr (DT == HMR_F16) {
        return static_cast<float>(__builtin_bit_cast(_Float16, v));
    } else if constexpr (DT == HMR_BF16) {
        return __uint_as_float(static_cast<uint32_t>(v) << 16);
    } else {
        return v;
    }
}

// Rounding to the dtype, to nearest even (half's from_f32).
template <int DT>
__device__ inline typename Elem<DT>::T down(typename Comp<DT>::C c) {
    if constexpr (DT == HMR_F16 || DT == HMR_BF16) {
        return from_f32<DT>(c);
    } else {
        return c;
    }
}

// The math functions of the CPU kernels for f32 and f64 (Rust's f32::exp is expf, and so on).
__device__ inline float m_exp(float x) { return expf(x); }
__device__ inline double m_exp(double x) { return exp(x); }
__device__ inline float m_log(float x) { return logf(x); }
__device__ inline double m_log(double x) { return log(x); }
__device__ inline float m_log1p(float x) { return log1pf(x); }
__device__ inline double m_log1p(double x) { return log1p(x); }
__device__ inline float m_sqrt(float x) { return sqrtf(x); }
__device__ inline double m_sqrt(double x) { return sqrt(x); }
__device__ inline float m_sin(float x) { return sinf(x); }
__device__ inline double m_sin(double x) { return sin(x); }
__device__ inline float m_cos(float x) { return cosf(x); }
__device__ inline double m_cos(double x) { return cos(x); }
__device__ inline float m_tanh(float x) { return tanhf(x); }
__device__ inline double m_tanh(double x) { return tanh(x); }
__device__ inline float m_pow(float x, float y) { return powf(x, y); }
__device__ inline double m_pow(double x, double y) { return pow(x, y); }
__device__ inline float m_fma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ inline double m_fma(double a, double b, double c) { return fma(a, b, c); }
__device__ inline float m_abs(float x) { return fabsf(x); }
__device__ inline double m_abs(double x) { return fabs(x); }

template <class C>
__device__ inline bool is_nan(C x) {
    return x != x;
}

// cpu::num::erf, step by step in f64; its exp is the device library's.
__device__ inline double erf_f64(double x) {
    if (x != x) return x;
    double ax = fabs(x);
    double r;
    if (ax < 1.5) {
        double x2 = ax * ax;
        double term = ax;
        double sum = ax;
        double n = 0.0;
        while (term > 1e-17 * sum) {
            n += 1.0;
            term *= 2.0 * x2 / (2.0 * n + 1.0);
            sum += term;
        }
        r = M_2_SQRTPI * exp(-x2) * sum;
    } else if (ax < 6.0) {
        double f = ax;
        for (int k = 80; k >= 1; --k) f = ax + (static_cast<double>(k) * 0.5) / f;
        r = 1.0 - exp(-ax * ax) / (sqrt(M_PI) * f);
    } else {
        r = 1.0;
    }
    return copysign(r, x);
}

// Float::erf: f32 goes through f64.
__device__ inline float m_erf(float x) { return static_cast<float>(erf_f64(x)); }
__device__ inline double m_erf(double x) { return erf_f64(x); }

template <int V>
struct Tag {
    static constexpr int v = V;
};

template <int>
constexpr bool kNever = false;

// The float unary ops of cpu::unary.
template <int OP, int DT>
struct FloatUnary {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    __device__ T operator()(T v) const { return down<DT>(f(up<DT>(v))); }
    __device__ static C f(C x) {
        const C one = 1;
        const C zero = 0;
        const C half = 0.5;
        if constexpr (OP == HMR_UNARY_NEG) {
            return -x;
        } else if constexpr (OP == HMR_UNARY_ABS) {
            return m_abs(x);
        } else if constexpr (OP == HMR_UNARY_RELU) {
            return is_nan(x) || x > zero ? x : zero;
        } else if constexpr (OP == HMR_UNARY_SIGN) {
            // NaN and -0.0 give +0.0, as in PyTorch.
            return x > zero ? one : x < zero ? -one : zero;
        } else if constexpr (OP == HMR_UNARY_EXP) {
            return m_exp(x);
        } else if constexpr (OP == HMR_UNARY_LOG) {
            return m_log(x);
        } else if constexpr (OP == HMR_UNARY_SQRT) {
            return m_sqrt(x);
        } else if constexpr (OP == HMR_UNARY_RSQRT) {
            return one / m_sqrt(x);
        } else if constexpr (OP == HMR_UNARY_RECIPROCAL) {
            return one / x;
        } else if constexpr (OP == HMR_UNARY_SIN) {
            return m_sin(x);
        } else if constexpr (OP == HMR_UNARY_COS) {
            return m_cos(x);
        } else if constexpr (OP == HMR_UNARY_ERF) {
            return m_erf(x);
        } else if constexpr (OP == HMR_UNARY_TANH) {
            return m_tanh(x);
        } else if constexpr (OP == HMR_UNARY_SIGMOID) {
            return one / (one + m_exp(-x));
        } else if constexpr (OP == HMR_UNARY_LOG_SIGMOID) {
            // PyTorch's CPU kernel: min(x, 0) - log1p(e^-|x|).
            C min = is_nan(x) || x < zero ? x : zero;
            return min - m_log1p(m_exp(-m_abs(x)));
        } else if constexpr (OP == HMR_UNARY_SILU) {
            return x / (one + m_exp(-x));
        } else if constexpr (OP == HMR_UNARY_GELU) {
            const C frac_1_sqrt_2 = static_cast<C>(M_SQRT1_2);
            return half * x * (one + m_erf(x * frac_1_sqrt_2));
        } else if constexpr (OP == HMR_UNARY_GELU_TANH) {
            // sqrt(2 / pi) in f64 (correctly rounded on both sides), then converted.
            const C k = static_cast<C>(sqrt(2.0 / M_PI));
            const C c = static_cast<C>(0.044715);
            return half * x * (one + m_tanh(k * (x + c * x * x * x)));
        } else {
            static_assert(kNever<OP>, "not a float unary op");
        }
    }
};

// Wrapping integer arithmetic on the bits, in u64 (two's complement), truncated to T.
template <class T>
__device__ inline uint64_t u64(T v) {
    return static_cast<uint64_t>(static_cast<int64_t>(v));
}
template <class T>
__device__ inline T trunc(uint64_t v) {
    return wrap<T>(static_cast<int64_t>(v));
}
template <class T>
__device__ inline T wneg(T x) {
    return trunc<T>(0 - u64(x));
}

// The integer unary ops of cpu::unary (Neg, Abs, Relu, Sign).
template <int OP, int DT>
struct IntUnary {
    using T = typename Elem<DT>::T;
    __device__ T operator()(T x) const {
        if constexpr (OP == HMR_UNARY_NEG) {
            return wneg(x);
        } else if constexpr (OP == HMR_UNARY_ABS) {
            if constexpr (std::is_unsigned_v<T>) {
                return x;
            } else {
                return x < 0 ? wneg(x) : x;
            }
        } else if constexpr (OP == HMR_UNARY_RELU) {
            return x > 0 ? x : T{0};
        } else if constexpr (OP == HMR_UNARY_SIGN) {
            return x > 0 ? T{1} : x < 0 ? wneg(T{1}) : T{0};
        } else {
            static_assert(kNever<OP>, "not an integer unary op");
        }
    }
};

// Element::to_bool: nonzero (NaN is true).
template <int DT>
__device__ inline bool to_bool(typename Elem<DT>::T v) {
    if constexpr (Elem<DT>::kind == kFloat) {
        return up<DT>(v) != 0;
    } else {
        return v != 0;
    }
}

template <int DT>
struct LogicalNot {
    __device__ uint8_t operator()(typename Elem<DT>::T v) const { return !to_bool<DT>(v); }
};

template <class T>
struct Identity {
    __device__ T operator()(T v) const { return v; }
};

// Comparisons of two elements of one dtype (floats compare as their exact f32 / f64 values).
template <int OP, int DT>
struct Compare {
    using T = typename Elem<DT>::T;
    __device__ uint8_t operator()(T a, T b) const {
        if constexpr (Elem<DT>::kind == kFloat) {
            return cmp(up<DT>(a), up<DT>(b));
        } else {
            return cmp(a, b);
        }
    }
    template <class V>
    __device__ static bool cmp(V x, V y) {
        if constexpr (OP == HMR_BINARY_GREATER) {
            return x > y;
        } else if constexpr (OP == HMR_BINARY_GREATER_EQUAL) {
            return x >= y;
        } else if constexpr (OP == HMR_BINARY_LESS) {
            return x < y;
        } else if constexpr (OP == HMR_BINARY_LESS_EQUAL) {
            return x <= y;
        } else if constexpr (OP == HMR_BINARY_EQUAL) {
            return x == y;
        } else if constexpr (OP == HMR_BINARY_NOT_EQUAL) {
            return x != y;
        } else {
            static_assert(kNever<OP>, "not a comparison");
        }
    }
};

// Logical ops, and the arithmetic of bools: Add and Maximum are or, Mul and Minimum are and.
template <bool OR>
struct BoolBinary {
    __device__ uint8_t operator()(uint8_t a, uint8_t b) const {
        if constexpr (OR) {
            return a != 0 || b != 0;
        } else {
            return a != 0 && b != 0;
        }
    }
};

template <class T>
__device__ inline T wmul(T a, T b) {
    return trunc<T>(u64(a) * u64(b));
}

// num::int_pow: wrapping powers; a negative exponent gives 1 / base truncated.
template <class T>
__device__ inline T int_pow(T base, T exponent) {
    int64_t e = static_cast<int64_t>(exponent);
    if (e < 0) {
        int64_t b = static_cast<int64_t>(base);
        if (b == 1) return T{1};
        if (b == -1) return e % 2 == 0 ? T{1} : base;
        return T{0};
    }
    T result = 1;
    T b = base;
    uint64_t ue = static_cast<uint64_t>(e);
    while (ue > 0) {
        if (ue & 1) result = wmul(result, b);
        b = wmul(b, b);
        ue >>= 1;
    }
    return result;
}

template <int OP, int DT>
struct IntBinary {
    using T = typename Elem<DT>::T;
    __device__ T operator()(T x, T y) const {
        if constexpr (OP == HMR_BINARY_ADD) {
            return trunc<T>(u64(x) + u64(y));
        } else if constexpr (OP == HMR_BINARY_SUB) {
            return trunc<T>(u64(x) - u64(y));
        } else if constexpr (OP == HMR_BINARY_MUL) {
            return wmul(x, y);
        } else if constexpr (OP == HMR_BINARY_POW) {
            return int_pow(x, y);
        } else if constexpr (OP == HMR_BINARY_MAXIMUM) {
            return x > y ? x : y;
        } else if constexpr (OP == HMR_BINARY_MINIMUM) {
            return x < y ? x : y;
        } else {
            static_assert(kNever<OP>, "not an integer binary op");
        }
    }
};

template <int OP, int DT>
struct FloatBinary {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    __device__ T operator()(T u, T v) const {
        C x = up<DT>(u);
        C y = up<DT>(v);
        return down<DT>(f(x, y));
    }
    __device__ static C f(C x, C y) {
        if constexpr (OP == HMR_BINARY_ADD) {
            return x + y;
        } else if constexpr (OP == HMR_BINARY_SUB) {
            return x - y;
        } else if constexpr (OP == HMR_BINARY_MUL) {
            return x * y;
        } else if constexpr (OP == HMR_BINARY_DIV) {
            return x / y;
        } else if constexpr (OP == HMR_BINARY_POW) {
            return m_pow(x, y);
        } else if constexpr (OP == HMR_BINARY_MAXIMUM) {
            return is_nan(x) || x > y ? x : y;
        } else if constexpr (OP == HMR_BINARY_MINIMUM) {
            return is_nan(x) || x < y ? x : y;
        } else {
            static_assert(kNever<OP>, "not a float binary op");
        }
    }
};

// cpu::binary_scalar: the scalar in the computation type, one rounding.
template <int DT>
struct ScalarOp {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    C s;
    int mode;  // 0: x * s, 1: x / s, 2: s / x
    __device__ T operator()(T v) const {
        C x = up<DT>(v);
        return down<DT>(mode == 0 ? x * s : mode == 1 ? x / s : s / x);
    }
};

// cpu::fused, with the coefficient already in the computation type.
template <int OP, int DT>
struct Fused2 {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    C coefficient;
    bool small;
    __device__ T operator()(T u, T v) const {
        C x = up<DT>(u);
        C y = up<DT>(v);
        if constexpr (OP == HMR_FUSED_SCALED_ADD) {
            return down<DT>(m_fma(y, coefficient, x));
        } else if constexpr (OP == HMR_FUSED_LERP) {
            return down<DT>(m_fma(coefficient, y - x, small ? x : y));
        } else {
            static_assert(kNever<OP>, "not a two-operand fused op");
        }
    }
};

template <int OP, int DT>
struct Fused3 {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    C value;
    __device__ T operator()(T u, T v, T w) const {
        C x = up<DT>(u);
        C y = up<DT>(v);
        C z = up<DT>(w);
        if constexpr (OP == HMR_FUSED_ADDCMUL) {
            return down<DT>(m_fma(value * y, z, x));
        } else if constexpr (OP == HMR_FUSED_ADDCDIV) {
            return down<DT>(x + value * y / z);
        } else {
            static_assert(kNever<OP>, "not a three-operand fused op");
        }
    }
};

template <class T>
struct Select {
    __device__ T operator()(uint8_t c, T a, T b) const { return c != 0 ? a : b; }
};

// The general kernel: operands with arbitrary strides, a grid-stride loop over the output.
template <class F, class O, class... I>
struct Map {
    static constexpr int N = 1 + sizeof...(I);

    template <size_t... K>
    __device__ static void at(const Operands<N> &ops, const int64_t (&o)[N], const F &f,
                              std::index_sequence<K...>) {
        static_cast<O *>(ops.d[0].data)[o[0]] =
            f(static_cast<const I *>(ops.d[K + 1].data)[o[K + 1]]...);
    }

    template <size_t... K>
    static void launch_flat(const Operands<N> &ops, int64_t n, const F &f, hipStream_t stream,
                            std::index_sequence<K...>);
};

template <class F, class O, class... I>
__global__ void map_kernel(Operands<1 + sizeof...(I)> ops, int64_t n, F f) {
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        int64_t o[1 + sizeof...(I)];
        offsets_n(ops, i, o);
        Map<F, O, I...>::at(ops, o, f, std::index_sequence_for<I...>{});
    }
}

// All operands flat and contiguous from their offsets: no index arithmetic.
template <class F, class O, class... I>
__global__ void map_flat_kernel(int64_t n, F f, O *out, const I *...in) {
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        out[i] = f(in[i]...);
    }
}

// Four consecutive elements, for vector loads and stores.
template <class T>
struct alignas(4 * sizeof(T)) Vec4 {
    T v[4];
};

// map_flat_kernel with four elements per thread, for operands whose first elements are aligned
// to four elements: each operand is read and written with one vector access (step 3, item 1:
// about 5% more bandwidth than one element per thread once the data is in DRAM, and twice as
// much for F16 in the cache, target/m4-step3/bench). The last group of fewer than four reads
// indices clamped into range, which needs no predicated loads, and guards only its stores.
template <class F, class O, class... I>
__global__ void map_vec4_kernel(int64_t n, F f, O *out, const I *...in) {
    int64_t groups = (n + 3) / 4;
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; g < groups;
         g += step) {
        int64_t i = 4 * g;
        if (i + 4 <= n) {
            Vec4<O> r;
            auto apply = [&](const auto &...x) {
                for (int j = 0; j < 4; ++j) r.v[j] = f(x.v[j]...);
            };
            apply(*reinterpret_cast<const Vec4<I> *>(in + i)...);
            *reinterpret_cast<Vec4<O> *>(out + i) = r;
        } else {
            for (int j = 0; j < 4; ++j) {
                int64_t k = i + j < n ? i + j : n - 1;
                O r = f(in[k]...);
                if (i + j < n) out[i + j] = r;
            }
        }
    }
}

template <class T>
bool aligned4(const T *p) {
    return reinterpret_cast<uintptr_t>(p) % (4 * sizeof(T)) == 0;
}

template <class F, class O, class... I>
template <size_t... K>
void Map<F, O, I...>::launch_flat(const Operands<N> &ops, int64_t n, const F &f,
                                  hipStream_t stream, std::index_sequence<K...>) {
    O *out = static_cast<O *>(ops.d[0].data) + ops.d[0].offset;
    std::tuple<const I *...> in{
        (static_cast<const I *>(ops.d[K + 1].data) + ops.d[K + 1].offset)...};
    if (aligned4(out) && (aligned4(std::get<K>(in)) && ...)) {
        hipLaunchKernelGGL((map_vec4_kernel<F, O, I...>), blocks_for((n + 3) / 4), dim3(kThreads),
                           0, stream, n, f, out, std::get<K>(in)...);
    } else {
        hipLaunchKernelGGL((map_flat_kernel<F, O, I...>), blocks_for(n), dim3(kThreads), 0,
                           stream, n, f, out, std::get<K>(in)...);
    }
}

// Queues out = f(inputs...) over the operands' elements.
template <class F, class O, class... I>
bool launch_map(const Operands<1 + sizeof...(I)> &ops, int64_t n, F f, hipStream_t stream) {
    bool flat = true;
    for (const hmr_tensor_desc &d : ops.d) flat = flat && is_flat(d);
    if (flat) {
        Map<F, O, I...>::launch_flat(ops, n, f, stream, std::index_sequence_for<I...>{});
    } else {
        hipLaunchKernelGGL((map_kernel<F, O, I...>), blocks_for(n), dim3(kThreads), 0, stream,
                           ops, n, f);
    }
    return true;
}

template <class T>
__global__ void triangle_kernel(Operands<2> ops, int64_t n, int64_t diagonal, int upper) {
    const hmr_tensor_desc &out = ops.d[0];
    int64_t rows = out.sizes[out.ndim - 2];
    int64_t cols = out.sizes[out.ndim - 1];
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        int64_t o[2];
        offsets_n(ops, i, o);
        int64_t c = i % cols;
        int64_t r = (i / cols) % rows;
        int64_t d = c - r;
        bool keep = upper ? d >= diagonal : d <= diagonal;
        static_cast<T *>(ops.d[0].data)[o[0]] =
            keep ? static_cast<const T *>(ops.d[1].data)[o[1]] : T{};
    }
}

// Calls body(Tag<V>{}) for the V of `Vs` equal to `value`; false if there is none.
template <int... Vs, class B>
bool select(int value, std::integer_sequence<int, Vs...>, B &&body) {
    bool result = false;
    (void)((value == Vs ? (result = body(Tag<Vs>{}), true) : false) || ...);
    return result;
}

using AllDtypes = std::integer_sequence<int, HMR_BOOL, HMR_U8, HMR_I8, HMR_I16, HMR_I32, HMR_I64,
                                        HMR_F16, HMR_BF16, HMR_F32, HMR_F64>;
using IntDtypes = std::integer_sequence<int, HMR_U8, HMR_I8, HMR_I16, HMR_I32, HMR_I64>;
using FloatDtypes = std::integer_sequence<int, HMR_F16, HMR_BF16, HMR_F32, HMR_F64>;

// The unsigned type of an element size, for kernels that only move elements.
template <int SIZE>
struct Bits;
template <>
struct Bits<1> {
    using T = uint8_t;
};
template <>
struct Bits<2> {
    using T = uint16_t;
};
template <>
struct Bits<4> {
    using T = uint32_t;
};
template <>
struct Bits<8> {
    using T = uint64_t;
};

template <class B>
bool for_size(int dtype, B &&body) {
    return select(element_size(dtype), std::integer_sequence<int, 1, 2, 4, 8>{}, body);
}

bool unary(int op, const Operands<2> &ops, int64_t n, hipStream_t s) {
    int dtype = ops.d[1].dtype;
    int out = ops.d[0].dtype;
    if (op == HMR_UNARY_LOGICAL_NOT) {
        if (out != HMR_BOOL) return false;
        return select(dtype, AllDtypes{}, [&](auto t) {
            constexpr int DT = decltype(t)::v;
            return launch_map<LogicalNot<DT>, uint8_t, typename Elem<DT>::T>(ops, n, {}, s);
        });
    }
    if (out != dtype) return false;
    if (dtype == HMR_BOOL) {
        if (op != HMR_UNARY_SIGN) return false;
        return launch_map<Identity<uint8_t>, uint8_t, uint8_t>(ops, n, {}, s);
    }
    bool int_op = op == HMR_UNARY_NEG || op == HMR_UNARY_ABS || op == HMR_UNARY_RELU ||
                  op == HMR_UNARY_SIGN;
    if (select(dtype, IntDtypes{}, [](auto) { return true; })) {
        if (!int_op) return false;
        return select(dtype, IntDtypes{}, [&](auto t) {
            constexpr int DT = decltype(t)::v;
            using T = typename Elem<DT>::T;
            return select(op, std::integer_sequence<int, HMR_UNARY_NEG, HMR_UNARY_ABS,
                                                    HMR_UNARY_RELU, HMR_UNARY_SIGN>{},
                          [&](auto o) {
                              constexpr int OP = decltype(o)::v;
                              return launch_map<IntUnary<OP, DT>, T, T>(ops, n, {}, s);
                          });
        });
    }
    return select(dtype, FloatDtypes{}, [&](auto t) {
        constexpr int DT = decltype(t)::v;
        using T = typename Elem<DT>::T;
        return select(op, std::make_integer_sequence<int, HMR_UNARY_LOGICAL_NOT>{}, [&](auto o) {
            constexpr int OP = decltype(o)::v;
            return launch_map<FloatUnary<OP, DT>, T, T>(ops, n, {}, s);
        });
    });
}

bool binary(int op, const Operands<3> &ops, int64_t n, hipStream_t s) {
    int dtype = ops.d[1].dtype;
    int out = ops.d[0].dtype;
    if (ops.d[2].dtype != dtype) return false;
    if (op >= HMR_BINARY_GREATER && op <= HMR_BINARY_NOT_EQUAL) {
        if (out != HMR_BOOL) return false;
        return select(dtype, AllDtypes{}, [&](auto t) {
            constexpr int DT = decltype(t)::v;
            using T = typename Elem<DT>::T;
            return select(op, std::integer_sequence<int, HMR_BINARY_GREATER,
                                                    HMR_BINARY_GREATER_EQUAL, HMR_BINARY_LESS,
                                                    HMR_BINARY_LESS_EQUAL, HMR_BINARY_EQUAL,
                                                    HMR_BINARY_NOT_EQUAL>{},
                          [&](auto o) {
                              constexpr int OP = decltype(o)::v;
                              return launch_map<Compare<OP, DT>, uint8_t, T, T>(ops, n, {}, s);
                          });
        });
    }
    if (out != dtype) return false;
    if (op == HMR_BINARY_LOGICAL_AND || op == HMR_BINARY_LOGICAL_OR) {
        if (dtype != HMR_BOOL) return false;
        if (op == HMR_BINARY_LOGICAL_OR) {
            return launch_map<BoolBinary<true>, uint8_t, uint8_t, uint8_t>(ops, n, {}, s);
        }
        return launch_map<BoolBinary<false>, uint8_t, uint8_t, uint8_t>(ops, n, {}, s);
    }
    if (dtype == HMR_BOOL) {
        if (op == HMR_BINARY_ADD || op == HMR_BINARY_MAXIMUM) {
            return launch_map<BoolBinary<true>, uint8_t, uint8_t, uint8_t>(ops, n, {}, s);
        }
        if (op == HMR_BINARY_MUL || op == HMR_BINARY_MINIMUM) {
            return launch_map<BoolBinary<false>, uint8_t, uint8_t, uint8_t>(ops, n, {}, s);
        }
        return false;
    }
    using IntOps = std::integer_sequence<int, HMR_BINARY_ADD, HMR_BINARY_SUB, HMR_BINARY_MUL,
                                         HMR_BINARY_POW, HMR_BINARY_MAXIMUM, HMR_BINARY_MINIMUM>;
    if (select(dtype, IntDtypes{}, [](auto) { return true; })) {
        return select(dtype, IntDtypes{}, [&](auto t) {
            constexpr int DT = decltype(t)::v;
            using T = typename Elem<DT>::T;
            return select(op, IntOps{}, [&](auto o) {
                constexpr int OP = decltype(o)::v;
                return launch_map<IntBinary<OP, DT>, T, T, T>(ops, n, {}, s);
            });
        });
    }
    return select(dtype, FloatDtypes{}, [&](auto t) {
        constexpr int DT = decltype(t)::v;
        using T = typename Elem<DT>::T;
        return select(op, std::make_integer_sequence<int, HMR_BINARY_GREATER>{}, [&](auto o) {
            constexpr int OP = decltype(o)::v;
            return launch_map<FloatBinary<OP, DT>, T, T, T>(ops, n, {}, s);
        });
    });
}

bool binary_scalar(int op, const Operands<2> &ops, int64_t n, double scalar, bool scalar_first,
                   hipStream_t s) {
    if (ops.d[0].dtype != ops.d[1].dtype) return false;
    int mode;
    if (op == HMR_BINARY_MUL) {
        mode = 0;
    } else if (op == HMR_BINARY_DIV) {
        mode = scalar_first ? 2 : 1;
    } else {
        return false;
    }
    return select(ops.d[0].dtype, FloatDtypes{}, [&](auto t) {
        constexpr int DT = decltype(t)::v;
        using T = typename Elem<DT>::T;
        // Rust's `s as f32` (to nearest even) for the f32 computation type.
        ScalarOp<DT> f{static_cast<typename Comp<DT>::C>(scalar), mode};
        return launch_map<ScalarOp<DT>, T, T>(ops, n, f, s);
    });
}

bool fused(int op, const Operands<4> &ops, bool has_c, int64_t n, double coefficient, bool small,
           hipStream_t s) {
    int dtype = ops.d[0].dtype;
    int operands = has_c ? 4 : 3;
    for (int k = 1; k < operands; ++k) {
        if (ops.d[k].dtype != dtype) return false;
    }
    bool three = op == HMR_FUSED_ADDCMUL || op == HMR_FUSED_ADDCDIV;
    bool two = op == HMR_FUSED_SCALED_ADD || op == HMR_FUSED_LERP;
    if ((three && !has_c) || (two && has_c) || (!two && !three)) return false;
    return select(dtype, FloatDtypes{}, [&](auto t) {
        constexpr int DT = decltype(t)::v;
        using T = typename Elem<DT>::T;
        using C = typename Comp<DT>::C;
        // The coefficient is a value of C, so this conversion is exact.
        C coeff = static_cast<C>(coefficient);
        if (three) {
            return select(op, std::integer_sequence<int, HMR_FUSED_ADDCMUL, HMR_FUSED_ADDCDIV>{},
                          [&](auto o) {
                              constexpr int OP = decltype(o)::v;
                              return launch_map<Fused3<OP, DT>, T, T, T, T>(ops, n,
                                                                            Fused3<OP, DT>{coeff},
                                                                            s);
                          });
        }
        Operands<3> ab{{ops.d[0], ops.d[1], ops.d[2]}};
        return select(op, std::integer_sequence<int, HMR_FUSED_SCALED_ADD, HMR_FUSED_LERP>{},
                      [&](auto o) {
                          constexpr int OP = decltype(o)::v;
                          return launch_map<Fused2<OP, DT>, T, T, T>(
                              ab, n, Fused2<OP, DT>{coeff, small}, s);
                      });
    });
}

}  // namespace
}  // namespace hmr

extern "C" {

hmr_status hmr_unary(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                     const hmr_tensor_desc *src) {
    return guarded([&] {
        hmr::Operands<2> ops{{*out, *src}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n)) return s;
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "unary", [&](hipStream_t s) {
            return hmr::unary(op, ops, n, s);
        });
    });
}

hmr_status hmr_binary(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                      const hmr_tensor_desc *a, const hmr_tensor_desc *b) {
    return guarded([&] {
        hmr::Operands<3> ops{{*out, *a, *b}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n)) return s;
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "binary", [&](hipStream_t s) {
            return hmr::binary(op, ops, n, s);
        });
    });
}

hmr_status hmr_binary_scalar(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                             const hmr_tensor_desc *a, double scalar, int32_t scalar_first) {
    return guarded([&] {
        hmr::Operands<2> ops{{*out, *a}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n)) return s;
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "binary_scalar", [&](hipStream_t s) {
            return hmr::binary_scalar(op, ops, n, scalar, scalar_first != 0, s);
        });
    });
}

hmr_status hmr_fused(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                     const hmr_tensor_desc *a, const hmr_tensor_desc *b,
                     const hmr_tensor_desc *c, double coefficient, int32_t small) {
    return guarded([&] {
        // Without `c`, its slot repeats `b` so that every slot holds a valid description.
        hmr::Operands<4> ops{{*out, *a, *b, c != nullptr ? *c : *b}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n)) return s;
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "fused", [&](hipStream_t s) {
            return hmr::fused(op, ops, c != nullptr, n, coefficient, small != 0, s);
        });
    });
}

hmr_status hmr_where(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *cond,
                     const hmr_tensor_desc *x, const hmr_tensor_desc *y) {
    return guarded([&] {
        hmr::Operands<4> ops{{*out, *cond, *x, *y}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n)) return s;
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "where", [&](hipStream_t s) {
            if (cond->dtype != HMR_BOOL || x->dtype != out->dtype || y->dtype != out->dtype) {
                return false;
            }
            return hmr::for_size(out->dtype, [&](auto size) {
                using T = typename hmr::Bits<decltype(size)::v>::T;
                return hmr::launch_map<hmr::Select<T>, T, uint8_t, T, T>(ops, n, {}, s);
            });
        });
    });
}

hmr_status hmr_triangle(hmr_context_t ctx, const hmr_tensor_desc *out,
                        const hmr_tensor_desc *src, int64_t diagonal, int32_t upper) {
    return guarded([&] {
        hmr::Operands<2> ops{{*out, *src}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n)) return s;
        if (out->ndim < 2) return hmr::fail(HMR_INVALID, "triangle needs two dimensions");
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "triangle", [&](hipStream_t s) {
            if (src->dtype != out->dtype) return false;
            return hmr::for_size(out->dtype, [&](auto size) {
                using T = typename hmr::Bits<decltype(size)::v>::T;
                hipLaunchKernelGGL((hmr::triangle_kernel<T>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, ops, n, diagonal, upper);
                return true;
            });
        });
    });
}

hmr_status hmr_flip(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *src) {
    return guarded([&] {
        hmr::Operands<2> ops{{*out, *src}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n, /*negative_inputs=*/true)) return s;
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "flip", [&](hipStream_t s) {
            if (src->dtype != out->dtype) return false;
            return hmr::for_size(out->dtype, [&](auto size) {
                using T = typename hmr::Bits<decltype(size)::v>::T;
                return hmr::launch_map<hmr::Identity<T>, T, T>(ops, n, {}, s);
            });
        });
    });
}

}  // extern "C"
