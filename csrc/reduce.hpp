// The reduction machinery shared by every kernel that folds the elements of a tensor into fewer
// outputs: sum, mean, prod, max, min, all, any, var and std, the arg reductions, and BatchNorm's
// statistics (step 6c), which must give the same bits as mean and var on the same input.
//
// A reduction is described by three tensor descriptions (built by src/backend/rocm/reduce.rs):
// - `out`: the kept dimensions with the output's contiguous strides;
// - `kept`: the same dimensions with the source's strides, and the source's data and offset;
// - `red`: the reduced dimensions with the source's strides.
// Neighbouring dimensions are merged first. The reduced elements of an output are visited in
// their logical (row-major) order j = 0 .. nred - 1, which the arg reductions report.
//
// The plan (make_plan) depends only on these sizes and strides, never on the device, so a given
// reduction always adds in the same order: the result is reproducible across runs, processes
// and GPUs of the supported architectures.
//
// - Each output is folded by a group of G lanes (G = 1, 32 or 256): lane l of the group takes
//   the elements l, l + G, l + 2G, ... of each row (the last reduced dimension) of its range, in
//   that order, into its accumulator; the group's accumulators are then combined as a binary
//   tree in shared memory (lane l with lane l + s, for s = G/2, G/4, ..., 1).
// - When the outputs alone give too few lanes, the range 0 .. nred - 1 is cut into `splits`
//   chunks of `chunk` elements, each folded by its own group into a partial result in the
//   workspace; a second kernel combines the partials of each output in chunk order.
//
// An op folds elements into an accumulator (Op::Acc, at most 16 bytes):
//   Acc init() const;                              the empty fold
//   Acc load(T x, int64_t o, int64_t j) const;     element j of output o
//   Acc combine(Acc a, Acc b) const;               a's elements come before b's
// and a sink writes the result of output o:
//   void put(int64_t o, Acc a, int64_t count) const;   count = nred
// Integer sums and products wrap, and max / min / the arg reductions carry the element's index,
// so these results don't depend on the order of the combinations at all; floating point sums and
// products are accumulated in f64, and only those depend on the plan.
#pragma once

#include <algorithm>

#include "kernel.hpp"

namespace hmr {
namespace {

// Lanes in flight that make a reduction worth splitting further, and the elements each lane
// folds at least before a split pays off.
constexpr int64_t kReduceTargetLanes = int64_t{1} << 17;
constexpr int64_t kReduceMinPerLane = 16;
constexpr int64_t kReduceMaxSplits = 4096;
// The workspace bytes of one partial result (every Acc fits).
constexpr int64_t kReducePartialBytes = 16;

struct ReducePlan {
    int64_t nout;    // outputs
    int64_t nred;    // elements folded into each output
    int64_t row;     // the size of the last reduced dimension (1 without reduced dimensions)
    int64_t step;    // its source stride
    int group;       // lanes per output: 1, 32 or 256
    int64_t splits;  // chunks of the reduced range
    int64_t chunk;   // elements per chunk
};

inline int64_t product(const hmr_tensor_desc &d) {
    int64_t n = 1;
    for (int k = 0; k < d.ndim; ++k) n *= d.sizes[k];
    return n;
}

// The plan of a reduction; `out` and `kept` have the same sizes (checked by the entries).
inline ReducePlan make_plan(const hmr_tensor_desc &kept, const hmr_tensor_desc &red) {
    ReducePlan p{};
    p.nout = product(kept);
    p.nred = product(red);
    p.row = red.ndim > 0 ? red.sizes[red.ndim - 1] : 1;
    p.step = red.ndim > 0 ? red.strides[red.ndim - 1] : 0;
    if (p.row == 0) p.row = 1;  // nothing to fold
    // Groups read contiguous rows together; other layouts fold one output per lane, which reads
    // neighbouring outputs together when the kept dimensions are the contiguous ones.
    p.group = p.step == 1 && p.row >= 256 ? 256 : p.step == 1 && p.row >= 32 ? 32 : 1;
    p.splits = 1;
    if (p.nout > 0 && p.nred > 0) {
        int64_t lanes = p.nout * p.group;
        if (lanes < kReduceTargetLanes) {
            int64_t want = ceil_div(kReduceTargetLanes, lanes);
            int64_t worth = p.nred / (p.group * kReduceMinPerLane);
            p.splits = std::max<int64_t>(1, std::min({want, worth, kReduceMaxSplits}));
        }
    }
    p.chunk = p.splits > 1 ? ceil_div(p.nred, p.splits) : p.nred;
    if (p.splits > 1) p.splits = ceil_div(p.nred, p.chunk);
    return p;
}

// The workspace bytes of a plan: one partial result per output and chunk when it splits.
inline int64_t plan_workspace(const ReducePlan &p) {
    return p.splits > 1 ? p.splits * p.nout * kReducePartialBytes : 0;
}

struct ReduceArgs {
    hmr_tensor_desc kept;
    hmr_tensor_desc red;
    ReducePlan plan;
    void *partials;
};

// The source offset of output o (its kept index decomposed over the kept dimensions).
__device__ inline int64_t kept_offset(const hmr_tensor_desc &kept, int64_t o) {
    int64_t off = kept.offset;
    for (int d = kept.ndim - 1; d >= 0; --d) {
        int64_t size = kept.sizes[d];
        int64_t q = o / size;
        off += (o - q * size) * kept.strides[d];
        o = q;
    }
    return off;
}

// The source offset of row q: q decomposed over the reduced dimensions but the last.
__device__ inline int64_t row_offset(const hmr_tensor_desc &red, int64_t q) {
    int64_t off = 0;
    for (int d = red.ndim - 2; d >= 0; --d) {
        int64_t size = red.sizes[d];
        int64_t r = q / size;
        off += (q - r * size) * red.strides[d];
        q = r;
    }
    return off;
}

// Combines the accumulators of each group of G lanes of the block (all threads call it) and
// returns the group's result in its lane 0.
template <int G, class Op>
__device__ inline typename Op::Acc group_combine(const Op &op, typename Op::Acc acc) {
    if constexpr (G == 1) {
        return acc;
    } else {
        using Acc = typename Op::Acc;
        __shared__ Acc buf[kThreads];
        int t = threadIdx.x;
        int lane = t % G;
        buf[t] = acc;
        __syncthreads();
        for (int s = G / 2; s > 0; s >>= 1) {
            if (lane < s) buf[t] = op.combine(buf[t], buf[t + s]);
            __syncthreads();
        }
        Acc result = buf[t - lane];
        __syncthreads();
        return result;
    }
}

// Folds the elements [j0, j1) of output o (at source offset `base`) for lane `lane` of a group
// of G.
template <int G, class Op>
__device__ inline typename Op::Acc fold_range(const Op &op, const ReduceArgs &a, int64_t o,
                                              int64_t base, int64_t j0, int64_t j1, int lane) {
    using T = typename Op::T;
    const T *src = static_cast<const T *>(a.kept.data);
    const int64_t row = a.plan.row, step = a.plan.step;
    typename Op::Acc acc = op.init();
    for (int64_t q = j0 / row; q * row < j1; ++q) {
        int64_t first = q * row;
        int64_t lo = j0 > first ? j0 - first : 0;
        int64_t hi = j1 - first < row ? j1 - first : row;
        int64_t r = base + row_offset(a.red, q);
        for (int64_t i = lo + lane; i < hi; i += G) {
            acc = op.combine(acc, op.load(src[r + i * step], o, first + i));
        }
    }
    return acc;
}

// One block holds kThreads / G groups, one output each; blockIdx.y is the chunk. Without a
// split the sink gets the result, else it goes to the partials (chunk-major).
template <int G, class Op, class Sink>
__global__ void __launch_bounds__(kThreads) reduce_kernel(ReduceArgs a, Op op, Sink sink) {
    using Acc = typename Op::Acc;
    constexpr int P = kThreads / G;
    const ReducePlan &p = a.plan;
    int lane = threadIdx.x % G;
    int slot = threadIdx.x / G;
    int64_t c = blockIdx.y;
    int64_t j0 = c * p.chunk;
    int64_t j1 = j0 + p.chunk < p.nred ? j0 + p.chunk : p.nred;
    for (int64_t tile = blockIdx.x; tile * P < p.nout; tile += gridDim.x) {
        int64_t o = tile * P + slot;
        Acc acc = op.init();
        if (o < p.nout) acc = fold_range<G>(op, a, o, kept_offset(a.kept, o), j0, j1, lane);
        acc = group_combine<G>(op, acc);
        if (lane == 0 && o < p.nout) {
            if (p.splits == 1) {
                sink.put(o, acc, p.nred);
            } else {
                static_cast<Acc *>(a.partials)[c * p.nout + o] = acc;
            }
        }
    }
}

// Combines the partials of each output in chunk order.
template <class Op, class Sink>
__global__ void __launch_bounds__(kThreads) combine_kernel(ReduceArgs a, Op op, Sink sink) {
    using Acc = typename Op::Acc;
    const ReducePlan &p = a.plan;
    const Acc *partials = static_cast<const Acc *>(a.partials);
    for (int64_t o = blockIdx.x * int64_t{kThreads} + threadIdx.x; o < p.nout;
         o += int64_t{gridDim.x} * kThreads) {
        Acc acc = op.init();
        for (int64_t c = 0; c < p.splits; ++c) acc = op.combine(acc, partials[c * p.nout + o]);
        sink.put(o, acc, p.nred);
    }
}

// Calls body(Tag<G>{}) for the plan's lanes per output.
template <class B>
void with_group(int group, B &&body) {
    if (group == 256) {
        body(Tag<256>{});
    } else if (group == 32) {
        body(Tag<32>{});
    } else {
        body(Tag<1>{});
    }
}

// The grid of reduce_kernel: tiles of kThreads / G outputs (at most kMaxBlocks, the kernel loops
// over the rest), times the chunks.
inline dim3 plan_grid(const ReducePlan &p, int group) {
    int64_t tiles = (p.nout + kThreads / group - 1) / (kThreads / group);
    return dim3(static_cast<unsigned>(tiles < kMaxBlocks ? tiles : kMaxBlocks),
                static_cast<unsigned>(p.splits));
}

// Queues the reduction `a` (with a.plan set and a.partials holding plan_workspace bytes) on `s`.
template <class Op, class Sink>
void launch_reduce(const ReduceArgs &a, const Op &op, const Sink &sink, hipStream_t s) {
    static_assert(sizeof(typename Op::Acc) <= kReducePartialBytes, "accumulator too large");
    const ReducePlan &p = a.plan;
    if (p.nout == 0) return;
    with_group(p.group, [&](auto g) {
        constexpr int G = decltype(g)::v;
        hipLaunchKernelGGL((reduce_kernel<G, Op, Sink>), plan_grid(p, G), dim3(kThreads), 0, s, a,
                           op, sink);
    });
    if (p.splits > 1) {
        hipLaunchKernelGGL((combine_kernel<Op, Sink>), blocks_for(p.nout), dim3(kThreads), 0, s, a,
                           op, sink);
    }
}

// Exact widening of a float element to f64 (a NaN stays a NaN; its payload doesn't matter to a
// sum).
template <int DT>
__device__ inline double widen(typename Elem<DT>::T x) {
    if constexpr (DT == HMR_F16) {
        return static_cast<double>(static_cast<float>(__builtin_bit_cast(_Float16, x)));
    } else if constexpr (DT == HMR_BF16) {
        return static_cast<double>(__uint_as_float(static_cast<uint32_t>(x) << 16));
    } else {
        return static_cast<double>(x);
    }
}

// The value an element compares by: f64 for floats (exact), the element itself otherwise.
template <int DT>
__device__ inline auto key(typename Elem<DT>::T x) {
    if constexpr (Elem<DT>::kind == kFloat) {
        return widen<DT>(x);
    } else {
        return x;
    }
}

// Sum (or product) in f64 of float elements, as the CPU's fold.
template <int DT, bool PROD>
struct FloatFold {
    using T = typename Elem<DT>::T;
    using Acc = double;
    __device__ Acc init() const { return PROD ? 1.0 : 0.0; }
    __device__ Acc load(T x, int64_t, int64_t) const { return widen<DT>(x); }
    __device__ Acc combine(Acc a, Acc b) const { return PROD ? a * b : a + b; }
};

// Wrapping sum (or product) in i64 of integer and bool elements.
template <int DT, bool PROD>
struct IntFold {
    using T = typename Elem<DT>::T;
    using Acc = int64_t;
    __device__ Acc init() const { return PROD ? 1 : 0; }
    __device__ Acc load(T x, int64_t, int64_t) const { return static_cast<int64_t>(x); }
    __device__ Acc combine(Acc a, Acc b) const {
        uint64_t ua = static_cast<uint64_t>(a), ub = static_cast<uint64_t>(b);
        return static_cast<int64_t>(PROD ? ua * ub : ua + ub);
    }
};

// all / any of the elements' truth values (a float NaN is true).
template <int DT, bool ALL>
struct Truth {
    using T = typename Elem<DT>::T;
    using Acc = uint32_t;
    __device__ Acc init() const { return ALL ? 1 : 0; }
    __device__ Acc load(T x, int64_t, int64_t) const {
        if constexpr (DT == HMR_F16 || DT == HMR_BF16) {
            return (x & 0x7FFF) != 0;
        } else {
            return x != 0;
        }
    }
    __device__ Acc combine(Acc a, Acc b) const { return ALL ? (a & b) : (a | b); }
};

// The maximum (or minimum) and its index: the larger (smaller) value wins, the first NaN wins
// over everything, and of equal values (+0 and -0 included) the first wins, as the CPU's
// `replaces`. An index below 0 marks the empty fold.
template <int DT>
struct Best {
    typename Elem<DT>::T v;
    int64_t i;
};

template <int DT, bool MAX>
struct Extreme {
    using T = typename Elem<DT>::T;
    using Acc = Best<DT>;
    __device__ static bool is_nan(T x) {
        if constexpr (Elem<DT>::kind == kFloat) {
            double k = widen<DT>(x);
            return k != k;
        } else {
            return false;
        }
    }
    __device__ Acc init() const { return Acc{T{}, -1}; }
    __device__ Acc load(T x, int64_t, int64_t j) const { return Acc{x, j}; }
    __device__ Acc combine(Acc a, Acc b) const {
        if (a.i < 0) return b;
        if (b.i < 0) return a;
        bool an = is_nan(a.v), bn = is_nan(b.v);
        if (an || bn) return an && (!bn || a.i < b.i) ? a : b;
        auto ka = key<DT>(a.v), kb = key<DT>(b.v);
        if (MAX ? kb > ka : kb < ka) return b;
        if (MAX ? ka > kb : ka < kb) return a;
        return a.i < b.i ? a : b;
    }
};

// The mean of each output in f64 (its sum divided by the count), into `means`: the first pass
// of var, and of BatchNorm's statistics.
struct MeanSink {
    double *means;
    __device__ void put(int64_t o, double a, int64_t count) const {
        means[o] = a / static_cast<double>(count);
    }
};

// The squared deviations of the elements from the mean of their output, `means[o]`, or `mean`
// when `means` is null: (x - mean)^2 in f64, as the CPU computes them.
template <int DT>
struct SquaredDeviations {
    using T = typename Elem<DT>::T;
    using Acc = double;
    const double *means;
    double mean;
    __device__ Acc init() const { return 0.0; }
    __device__ Acc load(T x, int64_t o, int64_t) const {
        double d = widen<DT>(x) - (means != nullptr ? means[o] : mean);
        return d * d;
    }
    __device__ Acc combine(Acc a, Acc b) const { return a + b; }
};

// A plan without chunks in one kernel: each group folds its output's sum, divides it into the
// mean, and folds the squared deviations from that mean, with the same folds and tree as the
// two reduce_kernel passes of launch_deviations, so the bits are the same; the sink gets the
// sum of squares.
template <int G, int DT, class Sink>
__global__ void __launch_bounds__(kThreads) deviations_kernel(ReduceArgs a, Sink sink) {
    constexpr int P = kThreads / G;
    const ReducePlan &p = a.plan;
    int lane = threadIdx.x % G;
    int slot = threadIdx.x / G;
    FloatFold<DT, false> fold;
    for (int64_t tile = blockIdx.x; tile * P < p.nout; tile += gridDim.x) {
        int64_t o = tile * P + slot;
        bool valid = o < p.nout;
        int64_t base = valid ? kept_offset(a.kept, o) : 0;
        double sum = valid ? fold_range<G>(fold, a, o, base, 0, p.nred, lane) : fold.init();
        sum = group_combine<G>(fold, sum);
        SquaredDeviations<DT> dev{nullptr, sum / static_cast<double>(p.nred)};
        double sq = valid ? fold_range<G>(dev, a, o, base, 0, p.nred, lane) : dev.init();
        sq = group_combine<G>(dev, sq);
        if (lane == 0 && valid) sink.put(o, sq, p.nred);
    }
}

// The sum of squared deviations of each output from its mean (var, and BatchNorm's statistics
// in step 6c), into `sink`: one kernel without chunks, else the means into `means` (nout
// doubles of device memory) and then the deviations, as two reductions.
template <int DT, class Sink>
void launch_deviations(const ReduceArgs &a, const Sink &sink, double *means, hipStream_t s) {
    const ReducePlan &p = a.plan;
    if (p.nout == 0) return;
    if (p.splits == 1) {
        with_group(p.group, [&](auto g) {
            constexpr int G = decltype(g)::v;
            hipLaunchKernelGGL((deviations_kernel<G, DT, Sink>), plan_grid(p, G), dim3(kThreads),
                               0, s, a, sink);
        });
    } else {
        launch_reduce(a, FloatFold<DT, false>{}, MeanSink{means}, s);
        launch_reduce(a, SquaredDeviations<DT>{means, 0.0}, sink, s);
    }
}

// Writes F(acc, count) of each output to the contiguous `out` (F::T elements).
template <class F>
struct OutSink {
    hmr_tensor_desc out;
    F f;
    template <class Acc>
    __device__ void put(int64_t o, Acc a, int64_t count) const {
        static_cast<typename F::T *>(out.data)[out.offset + o] = f(a, count);
    }
};

}  // namespace
}  // namespace hmr
