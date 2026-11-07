// Reductions (sum, mean, prod, max, min, all, any), var and std, the arg reductions, softmax and
// log-softmax, and the exclusive cumulative product: the counterparts of src/backend/cpu/reduce.rs
// and cpu::softmax, with the same results except for the order of floating point additions and
// multiplications (accumulated in f64, as on the CPU) and the device library's exp and log.
//
// Integer sums and products, all, any, max, min and the arg reductions give the CPU's bits; so
// does the exclusive cumulative product, which multiplies each line in the CPU's order. The
// reduction plans are in reduce.hpp.
#include <cmath>

#include "elem.hpp"
#include "reduce.hpp"

using hmr::guarded;

namespace hmr {
namespace {

// Rounds a float fold's f64 result to the dtype; with MEAN, divides it by the count first.
template <int DT, bool MEAN>
struct RoundF64 {
    using T = typename Elem<DT>::T;
    __device__ T operator()(double a, int64_t count) const {
        return from_f64<DT>(MEAN ? a / static_cast<double>(count) : a);
    }
};

struct I64Result {
    using T = int64_t;
    __device__ T operator()(int64_t a, int64_t) const { return a; }
};

struct TruthResult {
    using T = uint8_t;
    __device__ T operator()(uint32_t a, int64_t) const { return a != 0; }
};

template <int DT>
struct BestValue {
    using T = typename Elem<DT>::T;
    __device__ T operator()(Best<DT> a, int64_t) const { return a.v; }
};

// The result of the arg reductions: the index and, if `values` is given, the value.
template <int DT>
struct ArgSink {
    hmr_tensor_desc indices;
    hmr_tensor_desc values;
    bool has_values;
    __device__ void put(int64_t o, Best<DT> a, int64_t) const {
        static_cast<int64_t *>(indices.data)[indices.offset + o] = a.i;
        if (has_values) static_cast<typename Elem<DT>::T *>(values.data)[values.offset + o] = a.v;
    }
};

template <int DT>
struct VarResult {
    using T = typename Elem<DT>::T;
    double divisor;
    bool take_sqrt;
    __device__ T operator()(double sq, int64_t count) const {
        // An empty reduction has no mean, whatever the correction.
        double v = count == 0 ? NAN : sq / divisor;
        return from_f64<DT>(take_sqrt ? sqrt(v) : v);
    }
};

template <class Op, class F>
void reduce_into(const ReduceArgs &a, const hmr_tensor_desc &out, Op op, F f, hipStream_t s) {
    launch_reduce(a, op, OutSink<F>{out, f}, s);
}

bool reduce(int op, const hmr_tensor_desc &out, ReduceArgs a, hipStream_t s) {
    int dtype = a.kept.dtype;
    bool is_float = select(dtype, FloatDtypes{}, [](auto) { return true; });
    switch (op) {
        case HMR_REDUCE_SUM:
        case HMR_REDUCE_PROD:
        case HMR_REDUCE_MEAN:
            if (is_float) {
                if (out.dtype != dtype) return false;
                return select(dtype, FloatDtypes{}, [&](auto t) {
                    constexpr int DT = decltype(t)::v;
                    if (op == HMR_REDUCE_PROD) {
                        reduce_into(a, out, FloatFold<DT, true>{}, RoundF64<DT, false>{}, s);
                    } else if (op == HMR_REDUCE_SUM) {
                        reduce_into(a, out, FloatFold<DT, false>{}, RoundF64<DT, false>{}, s);
                    } else {
                        reduce_into(a, out, FloatFold<DT, false>{}, RoundF64<DT, true>{}, s);
                    }
                    return true;
                });
            }
            if (op == HMR_REDUCE_MEAN || out.dtype != HMR_I64) return false;
            return select(dtype, AllDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                if (op == HMR_REDUCE_PROD) {
                    reduce_into(a, out, IntFold<DT, true>{}, I64Result{}, s);
                } else {
                    reduce_into(a, out, IntFold<DT, false>{}, I64Result{}, s);
                }
                return true;
            });
        case HMR_REDUCE_MAX:
        case HMR_REDUCE_MIN:
            if (out.dtype != dtype) return false;
            return select(dtype, AllDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                if (op == HMR_REDUCE_MAX) {
                    reduce_into(a, out, Extreme<DT, true>{}, BestValue<DT>{}, s);
                } else {
                    reduce_into(a, out, Extreme<DT, false>{}, BestValue<DT>{}, s);
                }
                return true;
            });
        case HMR_REDUCE_ALL:
        case HMR_REDUCE_ANY:
            if (out.dtype != (dtype == HMR_U8 ? HMR_U8 : HMR_BOOL)) return false;
            return select(dtype, AllDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                if (op == HMR_REDUCE_ALL) {
                    reduce_into(a, out, Truth<DT, true>{}, TruthResult{}, s);
                } else {
                    reduce_into(a, out, Truth<DT, false>{}, TruthResult{}, s);
                }
                return true;
            });
        default:
            return false;
    }
}

// Checks the descriptions of a reduction and builds its arguments; `out` (may be null) must
// be contiguous with the kept sizes.
hmr_status reduce_args(const hmr_tensor_desc *out, const hmr_tensor_desc &kept,
                       const hmr_tensor_desc &red, ReduceArgs &a) {
    int64_t n = 0;
    if (hmr_status s = check_desc(kept, n)) return s;
    if (hmr_status s = check_desc(red, n)) return s;
    if (red.dtype != kept.dtype) return fail(HMR_INVALID, "reduction descriptions differ");
    if (out != nullptr) {
        if (hmr_status s = check_desc(*out, n)) return s;
        bool same = out->ndim == kept.ndim;
        int64_t stride = 1;
        for (int k = kept.ndim - 1; same && k >= 0; --k) {
            same = out->sizes[k] == kept.sizes[k] &&
                   (out->sizes[k] == 1 || out->strides[k] == stride);
            stride *= kept.sizes[k];
        }
        if (!same) {
            return fail(HMR_INVALID, "a reduction's output is not contiguous with its sizes");
        }
    }
    a.kept = kept;
    a.red = red;
    a.plan = make_plan(kept, red);
    a.partials = nullptr;
    return HMR_OK;
}

// The workspace layout: var's means when its plan has chunks (16-byte aligned), then the
// partials.
int64_t means_bytes(const ReducePlan &p, bool var) {
    return var && p.splits > 1 ? (p.nout * 8 + 15) / 16 * 16 : 0;
}

hmr_status take_workspace(ReduceArgs &a, bool var, void *workspace, int64_t bytes) {
    int64_t need = means_bytes(a.plan, var) + plan_workspace(a.plan);
    if (need > 0 && (workspace == nullptr || bytes < need)) {
        return fail(HMR_INVALID, "a reduction's workspace is too small");
    }
    a.partials = need > 0 ? static_cast<char *>(workspace) + means_bytes(a.plan, var) : nullptr;
    return HMR_OK;
}

// Softmax (or log-softmax) in f64 like the CPU: the maximum of each row (NaN ignored), the sum
// of exp(x - max), then each output, exp(x - max) / sum or x - max - log(sum). `rows` holds the
// output's and the source's descriptions over the other dimensions; the row elements are
// `so` / `si` apart. Two plans (softmax_plan), both depending only on the sizes and strides:
// - rows: one row per group of G lanes (G as in make_plan), lane l taking the elements
//   l, l + G, ... in order; with CACHE > 0 (at most CACHE elements per lane) a lane keeps its
//   inputs and then their exponentials in registers, so the source is read once and exp runs
//   once per element, with the same operations in the same order (the same bits) as without;
// - split: few long rows, whose maximum and sum are reductions of reduce.hpp (with chunks),
//   then one thread per element writes the output (exp computed again).
struct RowMaxOp {
    using Acc = double;
    __device__ Acc combine(Acc a, Acc b) const { return b > a ? b : a; }
};
struct RowSumOp {
    using Acc = double;
    __device__ Acc combine(Acc a, Acc b) const { return a + b; }
};

template <int DT, int G, int CACHE, bool LOG>
__global__ void __launch_bounds__(kThreads)
    softmax_kernel(Operands<2> rows, int64_t nrows, int64_t n, int64_t so, int64_t si) {
    using T = typename Elem<DT>::T;
    constexpr int P = kThreads / G;
    int lane = threadIdx.x % G;
    int slot = threadIdx.x / G;
    T *out = static_cast<T *>(rows.d[0].data);
    const T *src = static_cast<const T *>(rows.d[1].data);
    for (int64_t tile = blockIdx.x; tile * P < nrows; tile += gridDim.x) {
        int64_t r = tile * P + slot;
        bool valid = r < nrows;
        int64_t o[2] = {0, 0};
        if (valid) offsets_n(rows, r, o);
        int64_t len = valid ? n : 0;
        double max = -INFINITY;
        double sum = 0.0;
        if constexpr (CACHE > 0) {
            double v[CACHE];
#pragma unroll
            for (int j = 0; j < CACHE; ++j) {
                int64_t k = lane + int64_t{j} * G;
                v[j] = k < len ? widen<DT>(src[o[1] + k * si]) : -INFINITY;
                if (k < len) max = v[j] > max ? v[j] : max;
            }
            max = group_combine<G>(RowMaxOp{}, max);
#pragma unroll
            for (int j = 0; j < CACHE; ++j) {
                int64_t k = lane + int64_t{j} * G;
                if (k < len) {
                    double e = exp(v[j] - max);
                    sum += e;
                    if constexpr (!LOG) v[j] = e;
                }
            }
            sum = group_combine<G>(RowSumOp{}, sum);
            double log_sum = log(sum);
#pragma unroll
            for (int j = 0; j < CACHE; ++j) {
                int64_t k = lane + int64_t{j} * G;
                if (k < len) {
                    out[o[0] + k * so] = from_f64<DT>(LOG ? v[j] - max - log_sum : v[j] / sum);
                }
            }
        } else {
            for (int64_t k = lane; k < len; k += G) {
                double v = widen<DT>(src[o[1] + k * si]);
                max = v > max ? v : max;
            }
            max = group_combine<G>(RowMaxOp{}, max);
            for (int64_t k = lane; k < len; k += G) sum += exp(widen<DT>(src[o[1] + k * si]) - max);
            sum = group_combine<G>(RowSumOp{}, sum);
            double log_sum = log(sum);
            for (int64_t k = lane; k < len; k += G) {
                double x = widen<DT>(src[o[1] + k * si]);
                out[o[0] + k * so] = from_f64<DT>(LOG ? x - max - log_sum : exp(x - max) / sum);
            }
        }
    }
}

// The split plan's reductions: the maximum of each row (NaN ignored, as the CPU's fold), and
// the sum of exp(x - max) with the maxima from the first.
template <int DT>
struct RowMax {
    using T = typename Elem<DT>::T;
    using Acc = double;
    __device__ Acc init() const { return -INFINITY; }
    __device__ Acc load(T x, int64_t, int64_t) const { return widen<DT>(x); }
    __device__ Acc combine(Acc a, Acc b) const { return b > a ? b : a; }
};

template <int DT>
struct RowExpSum {
    using T = typename Elem<DT>::T;
    using Acc = double;
    const double *maxs;
    __device__ Acc init() const { return 0.0; }
    __device__ Acc load(T x, int64_t o, int64_t) const { return exp(widen<DT>(x) - maxs[o]); }
    __device__ Acc combine(Acc a, Acc b) const { return a + b; }
};

struct F64Sink {
    double *values;
    __device__ void put(int64_t o, double a, int64_t) const { values[o] = a; }
};

// The split plan's outputs, one thread per element.
template <int DT, bool LOG>
__global__ void __launch_bounds__(kThreads)
    softmax_normalize_kernel(Operands<2> rows, int64_t nrows, int64_t n, int64_t so, int64_t si,
                             const double *maxs, const double *sums) {
    using T = typename Elem<DT>::T;
    T *out = static_cast<T *>(rows.d[0].data);
    const T *src = static_cast<const T *>(rows.d[1].data);
    for (int64_t i = blockIdx.x * int64_t{kThreads} + threadIdx.x; i < nrows * n;
         i += int64_t{gridDim.x} * kThreads) {
        int64_t r = i / n;
        int64_t k = i - r * n;
        int64_t o[2];
        offsets_n(rows, r, o);
        double x = widen<DT>(src[o[1] + k * si]);
        double max = maxs[r], sum = sums[r];
        out[o[0] + k * so] = from_f64<DT>(LOG ? x - max - log(sum) : exp(x - max) / sum);
    }
}

// The registers a lane of the rows plan may keep: CACHE elements, at most 16 (145 to 155 VGPRs
// for G = 256 on gfx1201, without scratch; 32 would spill).
constexpr int kSoftmaxMaxCache = 16;

struct SoftmaxPlan {
    bool split;
    int group;
    int cache;         // 0: not cached
    ReduceArgs rows;   // the split plan's reductions over the rows
    int64_t nrows;
};

// The plan of a softmax: split when the reduction of its rows (the source's rows as kept
// dimensions, the row as the reduced one) is cut into chunks, else one row per group, cached
// when a lane has at most kSoftmaxMaxCache elements (rounded up to a power of two).
SoftmaxPlan softmax_plan(const hmr_tensor_desc &src_rows, int64_t n, int64_t si) {
    SoftmaxPlan p{};
    hmr_tensor_desc red{};
    red.data = src_rows.data;
    red.dtype = src_rows.dtype;
    red.ndim = 1;
    red.sizes[0] = n;
    red.strides[0] = si;
    p.rows = ReduceArgs{src_rows, red, make_plan(src_rows, red), nullptr};
    p.nrows = p.rows.plan.nout;
    p.split = p.rows.plan.splits > 1;
    p.group = p.rows.plan.group;
    int64_t per = (n + p.group - 1) / p.group;
    p.cache = 0;
    for (int c = 1; c <= kSoftmaxMaxCache; c *= 2) {
        if (per <= c) {
            p.cache = c;
            break;
        }
    }
    return p;
}

// The split plan's workspace: the maxima and the sums of the rows (16-byte aligned), then the
// partials of their reductions.
int64_t softmax_rows_bytes(const SoftmaxPlan &p) { return (p.nrows * 8 + 15) / 16 * 16; }

int64_t softmax_workspace(const SoftmaxPlan &p) {
    return p.split ? 2 * softmax_rows_bytes(p) + plan_workspace(p.rows.plan) : 0;
}

template <int DT, bool LOG>
void launch_softmax(SoftmaxPlan p, const Operands<2> &rows, int64_t n, int64_t so, int64_t si,
                    void *workspace, hipStream_t s) {
    if (p.split) {
        char *ws = static_cast<char *>(workspace);
        double *maxs = reinterpret_cast<double *>(ws);
        double *sums = reinterpret_cast<double *>(ws + softmax_rows_bytes(p));
        p.rows.partials = ws + 2 * softmax_rows_bytes(p);
        launch_reduce(p.rows, RowMax<DT>{}, F64Sink{maxs}, s);
        launch_reduce(p.rows, RowExpSum<DT>{maxs}, F64Sink{sums}, s);
        hipLaunchKernelGGL((softmax_normalize_kernel<DT, LOG>), blocks_for(p.nrows * n),
                           dim3(kThreads), 0, s, rows, p.nrows, n, so, si, maxs, sums);
        return;
    }
    with_group(p.group, [&](auto g) {
        constexpr int G = decltype(g)::v;
        int64_t tiles = (p.nrows + kThreads / G - 1) / (kThreads / G);
        dim3 grid(static_cast<unsigned>(tiles < kMaxBlocks ? tiles : kMaxBlocks));
        auto run = [&](auto c) {
            constexpr int C = decltype(c)::v;
            hipLaunchKernelGGL((softmax_kernel<DT, G, C, LOG>), grid, dim3(kThreads), 0, s, rows,
                               p.nrows, n, so, si);
        };
        switch (p.cache) {
            case 1: run(Tag<1>{}); break;
            case 2: run(Tag<2>{}); break;
            case 4: run(Tag<4>{}); break;
            case 8: run(Tag<8>{}); break;
            case 16: run(Tag<16>{}); break;
            default: run(Tag<0>{}); break;
        }
    });
}

// The exclusive cumulative product of one line per thread, as the CPU: accumulated in f32 for
// F16 and BF16 and in f64 for F32 and F64, each output rounded from the accumulator.
template <int DT>
struct CumAcc {
    using A = double;
};
template <>
struct CumAcc<HMR_F16> {
    using A = float;
};
template <>
struct CumAcc<HMR_BF16> {
    using A = float;
};

template <int DT>
__global__ void __launch_bounds__(kThreads)
    cumprod_kernel(Operands<2> lines, int64_t nlines, int64_t n, int64_t so, int64_t si,
                   bool reverse) {
    using T = typename Elem<DT>::T;
    using A = typename CumAcc<DT>::A;
    T *out = static_cast<T *>(lines.d[0].data);
    const T *src = static_cast<const T *>(lines.d[1].data);
    for (int64_t l = blockIdx.x * int64_t{kThreads} + threadIdx.x; l < nlines;
         l += int64_t{gridDim.x} * kThreads) {
        int64_t o[2];
        offsets_n(lines, l, o);
        A acc = 1;
        for (int64_t kk = 0; kk < n; ++kk) {
            int64_t k = reverse ? n - 1 - kk : kk;
            if constexpr (std::is_same_v<A, float>) {
                out[o[0] + k * so] = from_f32<DT>(acc);
                acc = acc * static_cast<float>(widen<DT>(src[o[1] + k * si]));
            } else {
                out[o[0] + k * so] = static_cast<T>(acc);
                acc = acc * widen<DT>(src[o[1] + k * si]);
            }
        }
    }
}

}  // namespace
}  // namespace hmr

extern "C" {

hmr_status hmr_reduce_workspace(const hmr_tensor_desc *kept, const hmr_tensor_desc *red,
                                int32_t var, int64_t *bytes) {
    return guarded([&] {
        hmr::ReduceArgs a;
        if (hmr_status s = hmr::reduce_args(nullptr, *kept, *red, a)) return s;
        *bytes = hmr::means_bytes(a.plan, var != 0) + hmr::plan_workspace(a.plan);
        return HMR_OK;
    });
}

hmr_status hmr_test_reduce_plan(const hmr_tensor_desc *kept, const hmr_tensor_desc *red,
                                int64_t *group, int64_t *splits, int64_t *chunk) {
    return guarded([&] {
        hmr::ReduceArgs a;
        if (hmr_status s = hmr::reduce_args(nullptr, *kept, *red, a)) return s;
        *group = a.plan.group;
        *splits = a.plan.splits;
        *chunk = a.plan.chunk;
        return HMR_OK;
    });
}

hmr_status hmr_reduce(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                      const hmr_tensor_desc *kept, const hmr_tensor_desc *red, void *workspace,
                      int64_t workspace_bytes) {
    return guarded([&] {
        hmr::ReduceArgs a;
        if (hmr_status s = hmr::reduce_args(out, *kept, *red, a)) return s;
        if (hmr_status s = hmr::take_workspace(a, false, workspace, workspace_bytes)) return s;
        if (a.plan.nout == 0) return HMR_OK;
        return hmr::launch_on(ctx, "reduce",
                              [&](hipStream_t s) { return hmr::reduce(op, *out, a, s); });
    });
}

hmr_status hmr_var(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *kept,
                   const hmr_tensor_desc *red, double correction, int32_t take_sqrt,
                   void *workspace, int64_t workspace_bytes) {
    return guarded([&] {
        hmr::ReduceArgs a;
        if (hmr_status s = hmr::reduce_args(out, *kept, *red, a)) return s;
        if (hmr_status s = hmr::take_workspace(a, true, workspace, workspace_bytes)) return s;
        if (a.plan.nout == 0) return HMR_OK;
        return hmr::launch_on(ctx, "var", [&](hipStream_t s) {
            if (out->dtype != kept->dtype) return false;
            double *means = static_cast<double *>(workspace);
            double divisor = static_cast<double>(a.plan.nred) - correction;
            divisor = divisor > 0.0 ? divisor : 0.0;
            return hmr::select(out->dtype, hmr::FloatDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                using Result = hmr::VarResult<DT>;
                hmr::OutSink<Result> sink{*out, Result{divisor, take_sqrt != 0}};
                hmr::launch_deviations<DT>(a, sink, means, s);
                return true;
            });
        });
    });
}

hmr_status hmr_arg_reduce(hmr_context_t ctx, const hmr_tensor_desc *values,
                          const hmr_tensor_desc *indices, const hmr_tensor_desc *kept,
                          const hmr_tensor_desc *red, int32_t is_max, void *workspace,
                          int64_t workspace_bytes) {
    return guarded([&] {
        hmr::ReduceArgs a;
        if (hmr_status s = hmr::reduce_args(indices, *kept, *red, a)) return s;
        if (values != nullptr) {
            hmr::ReduceArgs b;
            if (hmr_status s = hmr::reduce_args(values, *kept, *red, b)) return s;
        }
        if (hmr_status s = hmr::take_workspace(a, false, workspace, workspace_bytes)) return s;
        if (a.plan.nred == 0) return hmr::fail(HMR_INVALID, "arg reduction of an empty dimension");
        if (a.plan.nout == 0) return HMR_OK;
        return hmr::launch_on(ctx, "arg_reduce", [&](hipStream_t s) {
            if (indices->dtype != HMR_I64) return false;
            if (values != nullptr && values->dtype != kept->dtype) return false;
            return hmr::select(kept->dtype, hmr::AllDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                hmr::ArgSink<DT> sink{*indices, values != nullptr ? *values : *indices,
                                      values != nullptr};
                if (is_max != 0) {
                    hmr::launch_reduce(a, hmr::Extreme<DT, true>{}, sink, s);
                } else {
                    hmr::launch_reduce(a, hmr::Extreme<DT, false>{}, sink, s);
                }
                return true;
            });
        });
    });
}

// Checks the rows of a softmax; their number in `nrows`.
static hmr_status softmax_rows(const hmr::Operands<2> &rows, int64_t n, int64_t out_stride,
                               int64_t src_stride, int64_t &nrows) {
    if (hmr_status s = hmr::check_operands(rows, nrows)) return s;
    if (n < 0 || out_stride < 0 || src_stride < 0) {
        return hmr::fail(HMR_INVALID, "invalid softmax row");
    }
    return HMR_OK;
}

hmr_status hmr_softmax_workspace(const hmr_tensor_desc *src_rows, int64_t n, int64_t src_stride,
                                 int64_t *bytes) {
    return guarded([&] {
        hmr::Operands<2> rows{{*src_rows, *src_rows}};
        int64_t nrows = 0;
        if (hmr_status s = softmax_rows(rows, n, 0, src_stride, nrows)) return s;
        bool empty = nrows == 0 || n == 0;
        *bytes = empty ? 0 : hmr::softmax_workspace(hmr::softmax_plan(*src_rows, n, src_stride));
        return HMR_OK;
    });
}

hmr_status hmr_test_softmax_plan(const hmr_tensor_desc *src_rows, int64_t n, int64_t src_stride,
                                 int32_t *split, int32_t *group, int32_t *cache) {
    return guarded([&] {
        hmr::Operands<2> rows{{*src_rows, *src_rows}};
        int64_t nrows = 0;
        if (hmr_status s = softmax_rows(rows, n, 0, src_stride, nrows)) return s;
        hmr::SoftmaxPlan p = hmr::softmax_plan(*src_rows, n, src_stride);
        *split = p.split;
        *group = p.group;
        *cache = p.cache;
        return HMR_OK;
    });
}

hmr_status hmr_softmax(hmr_context_t ctx, const hmr_tensor_desc *out_rows,
                       const hmr_tensor_desc *src_rows, int64_t n, int64_t out_stride,
                       int64_t src_stride, int32_t log, void *workspace,
                       int64_t workspace_bytes) {
    return guarded([&] {
        hmr::Operands<2> rows{{*out_rows, *src_rows}};
        int64_t nrows = 0;
        if (hmr_status s = softmax_rows(rows, n, out_stride, src_stride, nrows)) return s;
        if (nrows == 0 || n == 0) return HMR_OK;
        hmr::SoftmaxPlan plan = hmr::softmax_plan(*src_rows, n, src_stride);
        int64_t need = hmr::softmax_workspace(plan);
        if (need > 0 && (workspace == nullptr || workspace_bytes < need)) {
            return hmr::fail(HMR_INVALID, "a softmax's workspace is too small");
        }
        return hmr::launch_on(ctx, "softmax", [&](hipStream_t s) {
            if (out_rows->dtype != src_rows->dtype) return false;
            return hmr::select(out_rows->dtype, hmr::FloatDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                if (log != 0) {
                    hmr::launch_softmax<DT, true>(plan, rows, n, out_stride, src_stride,
                                                  workspace, s);
                } else {
                    hmr::launch_softmax<DT, false>(plan, rows, n, out_stride, src_stride,
                                                   workspace, s);
                }
                return true;
            });
        });
    });
}

hmr_status hmr_exclusive_cumprod(hmr_context_t ctx, const hmr_tensor_desc *out_lines,
                                 const hmr_tensor_desc *src_lines, int64_t n, int64_t out_stride,
                                 int64_t src_stride, int32_t reverse) {
    return guarded([&] {
        hmr::Operands<2> lines{{*out_lines, *src_lines}};
        int64_t nlines = 0;
        if (hmr_status s = hmr::check_operands(lines, nlines)) return s;
        if (n < 0 || out_stride < 0 || src_stride < 0) {
            return hmr::fail(HMR_INVALID, "invalid cumulative product line");
        }
        if (nlines == 0 || n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "exclusive_cumprod", [&](hipStream_t s) {
            if (out_lines->dtype != src_lines->dtype) return false;
            return hmr::select(out_lines->dtype, hmr::FloatDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                hipLaunchKernelGGL((hmr::cumprod_kernel<DT>), hmr::blocks_for(nlines),
                                   dim3(hmr::kThreads), 0, s, lines, nlines, n, out_stride,
                                   src_stride, reverse != 0);
                return true;
            });
        });
    });
}

}  // extern "C"
