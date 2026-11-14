// Helpers for launching kernels over tensor descriptions, shared by the kernel sources.
#pragma once

#include <string>
#include <utility>

#include "common.hpp"

namespace hmr {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 1 << 16;

// n / d rounded up, for n >= 0 and d > 0, without forming n + d - 1 (which overflows for the
// largest sizes; review round 131, R154).
__host__ __device__ inline constexpr int64_t ceil_div(int64_t n, int64_t d) { return n / d + (n % d != 0); }

// Byte and word counts of workspaces, computed without signed overflow (which is undefined, and
// whose wrapped results once passed for small or negative sizes; review round 131, R154): each
// returns false once a step overflows int64_t.
inline bool checked_mul(int64_t a, int64_t b, int64_t &r) {
    return !__builtin_mul_overflow(a, b, &r);
}
inline bool checked_add(int64_t a, int64_t b, int64_t &r) {
    return !__builtin_add_overflow(a, b, &r);
}

inline dim3 blocks_for(int64_t n) {
    int64_t b = ceil_div(n, kThreads);
    return dim3(static_cast<unsigned>(b < kMaxBlocks ? b : kMaxBlocks));
}

inline bool is_flat(const hmr_tensor_desc &d) {
    return d.ndim == 0 || (d.ndim == 1 && d.strides[0] == 1);
}

// One element read at every index: an input broadcast from a single element, which the Rust
// side's dimension merging (copy::merge_dims) hands over as one dimension of stride 0. The flat
// paths read its element once per thread (review/M4.md, uniform operands).
inline bool is_uniform(const hmr_tensor_desc &d) {
    return d.ndim == 1 && d.strides[0] == 0;
}

inline int element_size(int dtype) {
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

// Checks a description from the Rust side; returns its number of elements in `n`. Strides may
// be negative only with `negative_strides`.
inline hmr_status check_desc(const hmr_tensor_desc &d, int64_t &n, bool negative_strides = false) {
    if (d.dtype < HMR_BOOL || d.dtype > HMR_F64) {
        return fail(HMR_INVALID, "invalid dtype " + std::to_string(d.dtype));
    }
    if (d.ndim < 0 || d.ndim > HMR_MAX_DIMS) {
        return fail(HMR_INVALID, "invalid number of dimensions " + std::to_string(d.ndim));
    }
    n = 1;
    for (int k = 0; k < d.ndim; ++k) {
        if (d.sizes[k] < 0) return fail(HMR_INVALID, "negative size");
        if (d.strides[k] < 0 && !negative_strides) return fail(HMR_INVALID, "negative stride");
        n *= d.sizes[k];
    }
    return HMR_OK;
}

// Descriptions of a kernel's operands, passed by value; operand 0 is the output, whose sizes
// all operands have.
template <int N>
struct Operands {
    hmr_tensor_desc d[N];
};

// The element offsets of linear index `i` (row-major over the output's sizes) in each operand.
template <int N>
__device__ inline void offsets_n(const Operands<N> &ops, int64_t i, int64_t (&o)[N]) {
    for (int k = 0; k < N; ++k) o[k] = ops.d[k].offset;
    const hmr_tensor_desc &out = ops.d[0];
    for (int dim = out.ndim - 1; dim > 0; --dim) {
        int64_t size = out.sizes[dim];
        int64_t q = i / size;
        int64_t idx = i - q * size;
        i = q;
        for (int k = 0; k < N; ++k) o[k] += idx * ops.d[k].strides[dim];
    }
    if (out.ndim > 0) {
        for (int k = 0; k < N; ++k) o[k] += i * ops.d[k].strides[0];
    }
}

// Checks the descriptions of `ops` (as check_desc) and that they all have the output's sizes;
// returns the number of elements in `n`. The inputs' strides may be negative only with
// `negative_inputs`.
template <int N>
hmr_status check_operands(const Operands<N> &ops, int64_t &n, bool negative_inputs = false) {
    if (hmr_status s = check_desc(ops.d[0], n)) return s;
    for (int k = 1; k < N; ++k) {
        int64_t m = 0;
        if (hmr_status s = check_desc(ops.d[k], m, negative_inputs)) return s;
        bool same = ops.d[k].ndim == ops.d[0].ndim;
        for (int j = 0; same && j < ops.d[0].ndim; ++j) {
            same = ops.d[k].sizes[j] == ops.d[0].sizes[j];
        }
        if (!same) return fail(HMR_INVALID, "kernel operands have different sizes");
    }
    return HMR_OK;
}

template <int V>
struct Tag {
    static constexpr int v = V;
};

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

// Queues the kernels that `body` launches on the context's stream, on its device, and checks
// the launches. `body` returns false (and queues nothing) for an op the dtypes don't support.
// A failure injected by hmr_test_fail_data comes before anything is queued.
template <class F>
hmr_status launch_on(hmr_context_t ctx, const char *what, F body) {
    DeviceGuard guard;
    if (hmr_status s = guard.enter(ctx->device)) return s;
    if (hmr_status s = injected_data_failure("hipGetLastError()")) return s;
    // The launch is checked with hipGetLastError, which must report this launch only.
    (void)hipGetLastError();
    if (!body(ctx->stream)) return fail(HMR_INVALID, std::string(what) + ": unsupported dtype");
    HMR_HIP(hipGetLastError());
    return HMR_OK;
}

}  // namespace
}  // namespace hmr
