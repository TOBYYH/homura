// Indexing kernels: the checks of index tensors, index_select and gather, the positions of
// advanced indexing, the overwriting and accumulating scatters (index_overwrite, index_add,
// scatter_add, and the backward of max_pool2d, which is a scatter of its gradient), and the
// coordinates of a mask. The counterparts of src/backend/cpu/misc.rs and cpu/pool.rs.
//
// Every result is the CPU's, bit for bit:
// - A check reports the first offending element in row-major order of the index tensor's
//   logical positions: the threads that find one take the minimum of their positions with an
//   atomic minimum, and only then a single thread reads the value at the winning position, so
//   the value always belongs to the reported position (review round 114, R132).
// - When several positions are written, the last one wins: each target keeps the largest source
//   position (an atomic maximum of integers), and only that source writes it.
// - Accumulating scatters group the pairs (target, source position) with the stable radix sort of
//   sort.hpp and add each group in the source order, from the target's value: floats in f64,
//   rounded once (the backward of max_pool2d adds in the dtype, rounding every addition, as the
//   CPU does), integers wrapping in i64, bools as a logical or. No floating point atomics.
// - The coordinates of a mask are compacted with the device scan of scan.hpp, in row-major order.
#include "elem.hpp"
#include "kernel.hpp"
#include "sort.hpp"

using hmr::guarded;

namespace hmr {
namespace {

constexpr uint64_t kNoPosition = ~uint64_t{0};

template <int N>
struct Word;
template <>
struct Word<1> {
    using T = uint8_t;
};
template <>
struct Word<2> {
    using T = uint16_t;
};
template <>
struct Word<4> {
    using T = uint32_t;
};
template <>
struct Word<8> {
    using T = uint64_t;
};

using WordSizes = std::integer_sequence<int, 1, 2, 4, 8>;
using IndexDtypes = std::integer_sequence<int, HMR_I32, HMR_I64>;

template <class I>
__device__ inline int64_t load_index(const hmr_tensor_desc &d, int64_t offset) {
    return static_cast<int64_t>(static_cast<const I *>(d.data)[offset]);
}

// The elements of `d` in row-major order: its own sizes, its own strides.
__device__ inline int64_t element_offset(const hmr_tensor_desc &d, int64_t i) {
    int64_t o = d.offset;
    for (int dim = d.ndim - 1; dim >= 0; --dim) {
        int64_t size = d.sizes[dim];
        int64_t q = i / size;
        o += (i - q * size) * d.strides[dim];
        i = q;
    }
    return o;
}

// A description with the sizes of `sizes` and the data, offset and strides of `of`, the stride of
// dimension `zero` (if any) replaced by 0: walking it gives the offset of `of` at the same
// coordinates, without the coordinate along `zero`.
hmr_tensor_desc walk(const hmr_tensor_desc &sizes, const hmr_tensor_desc &of, int zero) {
    hmr_tensor_desc w = of;
    w.ndim = sizes.ndim;
    for (int k = 0; k < sizes.ndim; ++k) {
        w.sizes[k] = sizes.sizes[k];
        w.strides[k] = k == zero ? 0 : of.strides[k];
    }
    return w;
}

// A description with the sizes of `sizes` whose offset is the coordinate along `dim` times
// `stride`, plus `offset`.
hmr_tensor_desc axis(const hmr_tensor_desc &sizes, int dim, int64_t stride, int64_t offset,
                     void *data, int dtype) {
    hmr_tensor_desc w = sizes;
    w.data = data;
    w.dtype = dtype;
    w.offset = offset;
    for (int k = 0; k < sizes.ndim; ++k) w.strides[k] = k == dim ? stride : 0;
    return w;
}

// ---- Checks ----

__global__ void slot_init_kernel(uint64_t *slot) {
    slot[0] = kNoPosition;
    slot[1] = 0;
}

template <class I>
__global__ void check_kernel(hmr_tensor_desc d, int64_t n, int64_t low, int64_t high,
                             int has_exempt, int64_t exempt, unsigned long long *slot) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        // Positions after a violation already found can't win.
        if (static_cast<uint64_t>(i) > __atomic_load_n(slot, __ATOMIC_RELAXED)) return;
        int64_t v = load_index<I>(d, element_offset(d, i));
        bool bad = (v < low || v >= high) && !(has_exempt && v == exempt);
        if (bad) atomicMin(slot, static_cast<unsigned long long>(i));
    }
}

template <class I>
__global__ void slot_value_kernel(hmr_tensor_desc d, uint64_t *slot) {
    uint64_t p = slot[0];
    if (p == kNoPosition) return;
    slot[1] = static_cast<uint64_t>(load_index<I>(d, element_offset(d, static_cast<int64_t>(p))));
}

hmr_status check_index_desc(const hmr_tensor_desc &d, int64_t &n) {
    if (hmr_status s = check_desc(d, n)) return s;
    if (d.dtype != HMR_I32 && d.dtype != HMR_I64) {
        return fail(HMR_INVALID, "index tensors must be I32 or I64");
    }
    return HMR_OK;
}

// ---- index_select and gather ----

// out[o0] = src[o1 + v * src_stride] for the index element v at o2, wrapped by `size` when
// negative and `wrap_negative`.
template <class W, class I>
__global__ void gather_kernel(Operands<3> ops, int64_t n, int64_t src_stride, int64_t size,
                              int wrap_negative) {
    const W *src = static_cast<const W *>(ops.d[1].data);
    W *out = static_cast<W *>(ops.d[0].data);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t o[3];
        offsets_n(ops, i, o);
        int64_t v = load_index<I>(ops.d[2], o[2]);
        if (wrap_negative && v < 0) v += size;
        out[o[0]] = src[o[1] + v * src_stride];
    }
}

hmr_status launch_gather(hmr_context_t ctx, const char *what, Operands<3> ops, int64_t n,
                         int64_t src_stride, int64_t size, bool wrap_negative) {
    if (n == 0) return HMR_OK;
    return launch_on(ctx, what, [&](hipStream_t s) {
        if (ops.d[0].dtype != ops.d[1].dtype) return false;
        return select(element_size(ops.d[0].dtype), WordSizes{}, [&](auto w) {
            using W = typename Word<decltype(w)::v>::T;
            return select(ops.d[2].dtype, IndexDtypes{}, [&](auto it) {
                using I = typename Elem<decltype(it)::v>::T;
                hipLaunchKernelGGL((gather_kernel<W, I>), blocks_for(n), dim3(kThreads), 0, s,
                                   ops, n, src_stride, size, wrap_negative ? 1 : 0);
                return true;
            });
        });
    });
}

// ---- Positions of advanced indexing ----

template <class I>
__global__ void positions_kernel(Operands<2> ops, int64_t n, int64_t size, int64_t stride,
                                 int first) {
    int64_t *out = static_cast<int64_t *>(ops.d[0].data);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t o[2];
        offsets_n(ops, i, o);
        int64_t v = load_index<I>(ops.d[1], o[1]);
        if (v < 0) v += size;
        int64_t p = v * stride;
        out[o[0]] = first ? p : out[o[0]] + p;
    }
}

// ---- index_overwrite ----

__global__ void zero_kernel(uint64_t *v, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        v[i] = 0;
    }
}

// winner[j] = 1 + the largest source position i with positions[i] == j (0: none).
__global__ void winner_kernel(hmr_tensor_desc positions, int64_t m, unsigned long long *winner) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < m;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t j = load_index<int64_t>(positions, element_offset(positions, i));
        atomicMax(&winner[j], static_cast<unsigned long long>(i + 1));
    }
}

// Over the elements of src (operand 0): operand 1 walks the target without the indexed
// dimension, operand 2 is the position's offset in `positions` (its coordinate along the indexed
// dimension); the source writes its target if it is the winner.
template <class W>
__global__ void overwrite_kernel(Operands<3> ops, int64_t n, int64_t out_stride,
                                 int64_t positions_offset, const uint64_t *winner) {
    const W *src = static_cast<const W *>(ops.d[0].data);
    W *out = static_cast<W *>(ops.d[1].data);
    const int64_t *positions = static_cast<const int64_t *>(ops.d[2].data);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t o[3];
        offsets_n(ops, i, o);
        int64_t j = positions[o[2]];
        uint64_t k = static_cast<uint64_t>(o[2] - positions_offset);
        if (winner[j] == k + 1) out[o[1] + j * out_stride] = src[o[0]];
    }
}

// ---- Accumulation, as cpu::add_at ----

template <int DT, Kind K = Elem<DT>::kind>
struct Acc {
    // Floating point: f64, rounded once.
    using T = typename Elem<DT>::T;
    using A = double;
    __device__ static A init(T x) { return to_f64<DT>(x); }
    __device__ static A add(A a, T x) { return a + to_f64<DT>(x); }
    __device__ static T finish(A a) { return from_f64<DT>(a); }
};

template <int DT>
struct Acc<DT, kInt> {
    using T = typename Elem<DT>::T;
    using A = int64_t;
    __device__ static A init(T x) { return static_cast<int64_t>(x); }
    __device__ static A add(A a, T x) {
        return static_cast<int64_t>(static_cast<uint64_t>(a) +
                                    static_cast<uint64_t>(static_cast<int64_t>(x)));
    }
    __device__ static T finish(A a) { return wrap<T>(a); }
};

template <int DT>
struct Acc<DT, kBool> {
    using T = uint8_t;
    using A = int64_t;
    __device__ static A init(T x) { return x != 0; }
    __device__ static A add(A a, T x) {
        return static_cast<int64_t>(static_cast<uint64_t>(a) + (x != 0 ? 1u : 0u));
    }
    __device__ static T finish(A a) { return a != 0; }
};

// The backward of max_pool2d: in the dtype, rounding every addition (cpu::max_pool2d_backward).
template <int DT>
struct RoundingAcc {
    using T = typename Elem<DT>::T;
    using A = T;
    __device__ static A init(T x) { return x; }
    __device__ static A add(A a, T x) { return down<DT>(up<DT>(a) + up<DT>(x)); }
    __device__ static T finish(A a) { return a; }
};

// keys[i] = the target slice of source slice i (index element i), vals[i] = i.
template <class I>
__global__ void slice_keys_kernel(hmr_tensor_desc index, int64_t m, uint64_t *keys,
                                  uint64_t *vals) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < m;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        keys[i] = static_cast<uint64_t>(load_index<I>(index, element_offset(index, i)));
        vals[i] = static_cast<uint64_t>(i);
    }
}

// index_add over the runs of equal sorted keys (target slices) times the elements of a slice:
// thread (t, e) works only if sorted position t starts a run, and adds element e of the source
// slices of the run, in their order, into element e of the target slice. Operands 0 and 1 walk
// out's and src's slices (the indexed dimension of size 1); only the touched slices are visited.
template <class Ac>
__global__ void index_add_kernel(Operands<2> ops, int64_t slice, int64_t m, int64_t out_stride,
                                 int64_t src_stride, const uint64_t *keys,
                                 const uint64_t *sources) {
    using T = typename Ac::T;
    T *out = static_cast<T *>(ops.d[0].data);
    const T *src = static_cast<const T *>(ops.d[1].data);
    int64_t n = m * slice;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t t = i / slice, e = i - t * slice;
        uint64_t key = keys[t];
        if (t > 0 && keys[t - 1] == key) continue;
        int64_t o[2];
        offsets_n(ops, e, o);
        T *target = out + o[0] + static_cast<int64_t>(key) * out_stride;
        typename Ac::A acc = Ac::init(*target);
        for (int64_t u = t; u < m && keys[u] == key; ++u) {
            acc = Ac::add(acc, src[o[1] + static_cast<int64_t>(sources[u]) * src_stride]);
        }
        *target = Ac::finish(acc);
    }
}

// scatter_add: keys[p] = the offset of the target of index position p from the start of out,
// vals[p] = p. Operand 0 is the index, operand 1 walks out without the scattered dimension.
template <class I>
__global__ void scatter_keys_kernel(Operands<2> ops, int64_t n, int64_t out_offset,
                                    int64_t out_stride, uint64_t *keys, uint64_t *vals) {
    for (int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < n;
         p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t o[2];
        offsets_n(ops, p, o);
        int64_t v = load_index<I>(ops.d[0], o[0]);
        keys[p] = static_cast<uint64_t>(o[1] - out_offset + v * out_stride);
        vals[p] = static_cast<uint64_t>(p);
    }
}

// The backward of max_pool2d: keys[p] = plane * plane_size + index, vals[p] = p.
__global__ void pool_keys_kernel(hmr_tensor_desc indices, int64_t n, int64_t outputs_per_plane,
                                 int64_t plane_size, uint64_t *keys, uint64_t *vals) {
    for (int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < n;
         p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t v = load_index<int64_t>(indices, element_offset(indices, p));
        keys[p] = static_cast<uint64_t>((p / outputs_per_plane) * plane_size + v);
        vals[p] = static_cast<uint64_t>(p);
    }
}

// Adds each run of equal sorted keys into its target, out[out.offset + key], in the order of the
// run; `src` is read at the row-major position in vals (its own sizes and strides).
template <class Ac>
__global__ void run_add_kernel(const uint64_t *keys, const uint64_t *vals, int64_t n,
                               hmr_tensor_desc out, hmr_tensor_desc src) {
    using T = typename Ac::T;
    T *o = static_cast<T *>(out.data) + out.offset;
    const T *s = static_cast<const T *>(src.data);
    for (int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; t < n;
         t += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        uint64_t key = keys[t];
        if (t > 0 && keys[t - 1] == key) continue;
        typename Ac::A acc = Ac::init(o[key]);
        for (int64_t u = t; u < n && keys[u] == key; ++u) {
            acc = Ac::add(acc, s[element_offset(src, static_cast<int64_t>(vals[u]))]);
        }
        o[key] = Ac::finish(acc);
    }
}

template <template <int> class AccOf, class Seq, class F>
bool with_acc(int dtype, Seq seq, F body) {
    return select(dtype, seq, [&](auto t) {
        constexpr int DT = decltype(t)::v;
        body(AccOf<DT>{});
        return true;
    });
}

template <int DT>
using CpuAcc = Acc<DT>;

// ---- Coordinates of a mask ----

constexpr int kMaskItems = 16;
constexpr int64_t kMaskTile = static_cast<int64_t>(kScanThreads) * kMaskItems;

inline int64_t mask_tiles(int64_t n) { return ceil_div(n, kMaskTile); }

__device__ inline uint32_t mask_thread_count(const hmr_tensor_desc &mask, int64_t n,
                                             int64_t first) {
    const uint8_t *m = static_cast<const uint8_t *>(mask.data);
    uint32_t c = 0;
    for (int k = 0; k < kMaskItems; ++k) {
        int64_t i = first + k;
        if (i < n && m[element_offset(mask, i)] != 0) ++c;
    }
    return c;
}

__global__ void mask_count_kernel(hmr_tensor_desc mask, int64_t n, uint64_t *counts) {
    int64_t first = static_cast<int64_t>(blockIdx.x) * kMaskTile +
                    static_cast<int64_t>(threadIdx.x) * kMaskItems;
    uint64_t total;
    (void)block_exclusive_scan<kScanThreads>(mask_thread_count(mask, n, first), total);
    if (threadIdx.x == 0) counts[blockIdx.x] = total;
}

// out[d * count + k] = coordinate d of the k-th true element.
__global__ void mask_write_kernel(hmr_tensor_desc mask, int64_t n, const uint64_t *starts,
                                  int64_t *out, int64_t count) {
    int64_t first = static_cast<int64_t>(blockIdx.x) * kMaskTile +
                    static_cast<int64_t>(threadIdx.x) * kMaskItems;
    uint64_t total;
    uint64_t k = starts[blockIdx.x] +
                 block_exclusive_scan<kScanThreads>(mask_thread_count(mask, n, first), total);
    const uint8_t *m = static_cast<const uint8_t *>(mask.data);
    for (int j = 0; j < kMaskItems; ++j) {
        int64_t i = first + j;
        if (i >= n || m[element_offset(mask, i)] == 0) continue;
        int64_t rest = i;
        for (int d = mask.ndim - 1; d >= 0; --d) {
            int64_t size = mask.sizes[d];
            int64_t q = rest / size;
            out[d * count + static_cast<int64_t>(k)] = rest - q * size;
            rest = q;
        }
        ++k;
    }
}

// The failure of a workspace whose size overflows int64_t: like a storage whose byte count
// overflows (D22), the caller panics.
hmr_status too_large() { return fail(HMR_INVALID, "the workspace is too large"); }

// The workspace bytes of the scatters, a sort of `pairs` >= 0 pairs, into `bytes`.
hmr_status scatter_bytes(int64_t pairs, int64_t &bytes) {
    int64_t words = 0;
    if (pairs > 0 && !sort_workspace_words(pairs, words)) return too_large();
    return checked_mul(8, words, bytes) ? HMR_OK : too_large();
}

// The workspace bytes of a mask of `n` >= 0 elements: a count per tile and the scan's words.
hmr_status mask_bytes(int64_t n, int64_t &bytes) {
    int64_t tiles = mask_tiles(n);
    return checked_mul(8, tiles + scan_aux_words(tiles), bytes) ? HMR_OK : too_large();
}

// The workspace bytes of overwriting along a dimension of `size` >= 0: a winner per position.
hmr_status overwrite_bytes(int64_t size, int64_t &bytes) {
    return checked_mul(8, size, bytes) ? HMR_OK : too_large();
}

hmr_status check_workspace(int64_t need, int64_t have, void *ws) {
    if (need < 0) return fail(HMR_INVALID, "a negative workspace size");
    if (need > 0 && (have < need || ws == nullptr)) {
        return fail(HMR_INVALID, "the workspace is too small");
    }
    return HMR_OK;
}

}  // namespace
}  // namespace hmr

extern "C" {

hmr_status hmr_index_check(hmr_context_t ctx, const hmr_tensor_desc *index, int64_t low,
                           int64_t high, int32_t has_exempt, int64_t exempt, uint64_t *slot) {
    return guarded([&] {
        int64_t n = 0;
        if (hmr_status s = hmr::check_index_desc(*index, n)) return s;
        return hmr::launch_on(ctx, "index check", [&](hipStream_t s) {
            hipLaunchKernelGGL(hmr::slot_init_kernel, dim3(1), dim3(1), 0, s, slot);
            if (n == 0) return true;
            return hmr::select(index->dtype, hmr::IndexDtypes{}, [&](auto it) {
                using I = typename hmr::Elem<decltype(it)::v>::T;
                hipLaunchKernelGGL((hmr::check_kernel<I>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, *index, n, low, high, has_exempt,
                                   exempt, reinterpret_cast<unsigned long long *>(slot));
                hipLaunchKernelGGL((hmr::slot_value_kernel<I>), dim3(1), dim3(1), 0, s, *index,
                                   slot);
                return true;
            });
        });
    });
}

hmr_status hmr_test_index_value(hmr_context_t ctx, const hmr_tensor_desc *index,
                                uint64_t *slot) {
    return guarded([&] {
        int64_t n = 0;
        if (hmr_status s = hmr::check_index_desc(*index, n)) return s;
        return hmr::launch_on(ctx, "index value", [&](hipStream_t s) {
            return hmr::select(index->dtype, hmr::IndexDtypes{}, [&](auto it) {
                using I = typename hmr::Elem<decltype(it)::v>::T;
                hipLaunchKernelGGL((hmr::slot_value_kernel<I>), dim3(1), dim3(1), 0, s, *index,
                                   slot);
                return true;
            });
        });
    });
}

hmr_status hmr_index_select(hmr_context_t ctx, const hmr_tensor_desc *out,
                            const hmr_tensor_desc *src, const hmr_tensor_desc *index, int32_t dim,
                            int32_t wrap_negative) {
    return guarded([&] {
        int64_t n = 0, ns = 0, ni = 0;
        if (hmr_status s = hmr::check_desc(*out, n)) return s;
        if (hmr_status s = hmr::check_desc(*src, ns)) return s;
        if (hmr_status s = hmr::check_index_desc(*index, ni)) return s;
        if (src->ndim != out->ndim || dim < 0 || dim >= out->ndim || index->ndim != 1 ||
            index->sizes[0] != out->sizes[dim]) {
            return hmr::fail(HMR_INVALID, "invalid index_select");
        }
        hmr::Operands<3> ops{{*out, hmr::walk(*out, *src, dim),
                              hmr::axis(*out, dim, index->strides[0], index->offset, index->data,
                                        index->dtype)}};
        return hmr::launch_gather(ctx, "index_select", ops, n, src->strides[dim],
                                  src->sizes[dim], wrap_negative != 0);
    });
}

hmr_status hmr_gather(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *src,
                      const hmr_tensor_desc *index, int32_t dim) {
    return guarded([&] {
        int64_t n = 0, ns = 0, ni = 0;
        if (hmr_status s = hmr::check_desc(*out, n)) return s;
        if (hmr_status s = hmr::check_desc(*src, ns)) return s;
        if (hmr_status s = hmr::check_index_desc(*index, ni)) return s;
        bool same = index->ndim == out->ndim && src->ndim == out->ndim;
        for (int k = 0; same && k < out->ndim; ++k) same = index->sizes[k] == out->sizes[k];
        if (!same || dim < 0 || dim >= out->ndim) return hmr::fail(HMR_INVALID, "invalid gather");
        hmr::Operands<3> ops{{*out, hmr::walk(*out, *src, dim), *index}};
        return hmr::launch_gather(ctx, "gather", ops, n, src->strides[dim], src->sizes[dim],
                                  false);
    });
}

hmr_status hmr_index_positions(hmr_context_t ctx, const hmr_tensor_desc *out,
                               const hmr_tensor_desc *coord, int64_t size, int64_t stride,
                               int32_t first) {
    return guarded([&] {
        hmr::Operands<2> ops{{*out, *coord}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n)) return s;
        int64_t m = 0;
        if (hmr_status s = hmr::check_index_desc(*coord, m)) return s;
        if (out->dtype != HMR_I64) return hmr::fail(HMR_INVALID, "positions must be I64");
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "index positions", [&](hipStream_t s) {
            return hmr::select(coord->dtype, hmr::IndexDtypes{}, [&](auto it) {
                using I = typename hmr::Elem<decltype(it)::v>::T;
                hipLaunchKernelGGL((hmr::positions_kernel<I>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, ops, n, size, stride, first);
                return true;
            });
        });
    });
}

hmr_status hmr_index_overwrite_workspace(int64_t size, int64_t *bytes) {
    return guarded([&] {
        if (size < 0) return hmr::fail(HMR_INVALID, "negative size");
        return hmr::overwrite_bytes(size, *bytes);
    });
}

hmr_status hmr_index_overwrite(hmr_context_t ctx, const hmr_tensor_desc *out,
                               const hmr_tensor_desc *src, const hmr_tensor_desc *positions,
                               int32_t dim, void *workspace, int64_t workspace_bytes) {
    return guarded([&] {
        int64_t n = 0, no = 0, m = 0;
        if (hmr_status s = hmr::check_desc(*src, n)) return s;
        if (hmr_status s = hmr::check_desc(*out, no)) return s;
        if (hmr_status s = hmr::check_desc(*positions, m)) return s;
        bool ok = src->ndim == out->ndim && dim >= 0 && dim < out->ndim &&
                  positions->dtype == HMR_I64 && positions->ndim == 1 &&
                  positions->strides[0] == 1 && positions->sizes[0] == src->sizes[dim] &&
                  src->dtype == out->dtype;
        for (int k = 0; ok && k < out->ndim; ++k) ok = k == dim || src->sizes[k] == out->sizes[k];
        if (!ok) return hmr::fail(HMR_INVALID, "invalid index_overwrite");
        int64_t size = out->sizes[dim];
        int64_t need = 0;
        if (hmr_status s = hmr::overwrite_bytes(size, need)) return s;
        if (hmr_status s = hmr::check_workspace(need, workspace_bytes, workspace)) return s;
        if (n == 0) return HMR_OK;
        auto *winner = static_cast<uint64_t *>(workspace);
        hmr::Operands<3> ops{{*src, hmr::walk(*src, *out, dim),
                              hmr::axis(*src, dim, 1, positions->offset, positions->data,
                                        HMR_I64)}};
        return hmr::launch_on(ctx, "index_overwrite", [&](hipStream_t s) {
            return hmr::select(hmr::element_size(src->dtype), hmr::WordSizes{}, [&](auto w) {
                using W = typename hmr::Word<decltype(w)::v>::T;
                hipLaunchKernelGGL(hmr::zero_kernel, hmr::blocks_for(size), dim3(hmr::kThreads),
                                   0, s, winner, size);
                hipLaunchKernelGGL(hmr::winner_kernel, hmr::blocks_for(m), dim3(hmr::kThreads),
                                   0, s, *positions, m,
                                   reinterpret_cast<unsigned long long *>(winner));
                hipLaunchKernelGGL((hmr::overwrite_kernel<W>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, ops, n, out->strides[dim],
                                   positions->offset, winner);
                return true;
            });
        });
    });
}

hmr_status hmr_index_add_workspace(int64_t m, int64_t size, int64_t *bytes) {
    return guarded([&] {
        if (m < 0 || size < 0) return hmr::fail(HMR_INVALID, "negative size");
        return hmr::scatter_bytes(m, *bytes);
    });
}

hmr_status hmr_index_add(hmr_context_t ctx, const hmr_tensor_desc *out,
                         const hmr_tensor_desc *src, const hmr_tensor_desc *index, int32_t dim,
                         void *workspace, int64_t workspace_bytes) {
    return guarded([&] {
        int64_t n = 0, ns = 0, m = 0;
        if (hmr_status s = hmr::check_desc(*out, n)) return s;
        if (hmr_status s = hmr::check_desc(*src, ns)) return s;
        if (hmr_status s = hmr::check_index_desc(*index, m)) return s;
        bool ok = src->ndim == out->ndim && dim >= 0 && dim < out->ndim &&
                  src->sizes[dim] == m && src->dtype == out->dtype;
        for (int k = 0; ok && k < out->ndim; ++k) ok = k == dim || src->sizes[k] == out->sizes[k];
        if (!ok) return hmr::fail(HMR_INVALID, "invalid index_add");
        int64_t size = out->sizes[dim];
        int64_t need = 0;
        if (hmr_status s = hmr::scatter_bytes(m, need)) return s;
        if (hmr_status s = hmr::check_workspace(need, workspace_bytes, workspace)) return s;
        if (ns == 0) return HMR_OK;
        hmr::SortSpace sp = hmr::sort_space(static_cast<uint64_t *>(workspace), m);
        // The slices: out and src with the indexed dimension reduced to one position.
        hmr_tensor_desc os = *out, ss = hmr::walk(*out, *src, -1);
        os.sizes[dim] = 1;
        ss.sizes[dim] = 1;
        int64_t slice = ns / m;
        hmr::Operands<2> ops{{os, ss}};
        return hmr::launch_on(ctx, "index_add", [&](hipStream_t s) {
            bool keyed = hmr::select(index->dtype, hmr::IndexDtypes{}, [&](auto it) {
                using I = typename hmr::Elem<decltype(it)::v>::T;
                hipLaunchKernelGGL((hmr::slice_keys_kernel<I>), hmr::blocks_for(m),
                                   dim3(hmr::kThreads), 0, s, *index, m, sp.keys[0], sp.vals[0]);
                return true;
            });
            if (!keyed) return false;
            int cur = hmr::radix_sort(sp, static_cast<uint64_t>(size), s);
            return hmr::with_acc<hmr::CpuAcc>(out->dtype, hmr::AllDtypes{}, [&](auto acc) {
                hipLaunchKernelGGL((hmr::index_add_kernel<decltype(acc)>), hmr::blocks_for(ns),
                                   dim3(hmr::kThreads), 0, s, ops, slice, m, out->strides[dim],
                                   src->strides[dim], sp.keys[cur], sp.vals[cur]);
            });
        });
    });
}

hmr_status hmr_scatter_add_workspace(int64_t n, int64_t *bytes) {
    return guarded([&] {
        if (n < 0) return hmr::fail(HMR_INVALID, "negative size");
        return hmr::scatter_bytes(n, *bytes);
    });
}

hmr_status hmr_scatter_add(hmr_context_t ctx, const hmr_tensor_desc *out,
                           const hmr_tensor_desc *src, const hmr_tensor_desc *index, int32_t dim,
                           void *workspace, int64_t workspace_bytes) {
    return guarded([&] {
        int64_t no = 0, ns = 0, n = 0;
        if (hmr_status s = hmr::check_desc(*out, no)) return s;
        if (hmr_status s = hmr::check_desc(*src, ns)) return s;
        if (hmr_status s = hmr::check_index_desc(*index, n)) return s;
        bool ok = index->ndim == out->ndim && src->ndim == out->ndim && dim >= 0 &&
                  dim < out->ndim && src->dtype == out->dtype;
        for (int k = 0; ok && k < out->ndim; ++k) {
            ok = (k == dim || index->sizes[k] <= out->sizes[k]) && index->sizes[k] <= src->sizes[k];
        }
        if (!ok) return hmr::fail(HMR_INVALID, "invalid scatter_add");
        int64_t need = 0;
        if (hmr_status s = hmr::scatter_bytes(n, need)) return s;
        if (hmr_status s = hmr::check_workspace(need, workspace_bytes, workspace)) return s;
        if (n == 0 || no == 0) return HMR_OK;
        // The keys are offsets from out's first element: below 1 + the largest offset.
        uint64_t bound = 1;
        for (int k = 0; k < out->ndim; ++k) {
            bound += static_cast<uint64_t>(out->sizes[k] - 1) * out->strides[k];
        }
        hmr::SortSpace sp = hmr::sort_space(static_cast<uint64_t *>(workspace), n);
        hmr::Operands<2> ops{{*index, hmr::walk(*index, *out, dim)}};
        // src is read at the index's positions with its own strides.
        hmr_tensor_desc src_at = hmr::walk(*index, *src, -1);
        return hmr::launch_on(ctx, "scatter_add", [&](hipStream_t s) {
            bool keyed = hmr::select(index->dtype, hmr::IndexDtypes{}, [&](auto it) {
                using I = typename hmr::Elem<decltype(it)::v>::T;
                hipLaunchKernelGGL((hmr::scatter_keys_kernel<I>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, ops, n, out->offset,
                                   out->strides[dim], sp.keys[0], sp.vals[0]);
                return true;
            });
            if (!keyed) return false;
            int cur = hmr::radix_sort(sp, bound, s);
            return hmr::with_acc<hmr::CpuAcc>(out->dtype, hmr::AllDtypes{}, [&](auto acc) {
                hipLaunchKernelGGL((hmr::run_add_kernel<decltype(acc)>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, sp.keys[cur], sp.vals[cur], n, *out,
                                   src_at);
            });
        });
    });
}

hmr_status hmr_max_pool2d_backward_workspace(int64_t n, int64_t *bytes) {
    return hmr_scatter_add_workspace(n, bytes);
}

hmr_status hmr_max_pool2d_backward(hmr_context_t ctx, const hmr_tensor_desc *out,
                                   const hmr_tensor_desc *grad, const hmr_tensor_desc *indices,
                                   void *workspace, int64_t workspace_bytes) {
    return guarded([&] {
        hmr::Operands<2> gi{{*grad, *indices}};
        int64_t n = 0, no = 0;
        if (hmr_status s = hmr::check_operands(gi, n)) return s;
        if (hmr_status s = hmr::check_desc(*out, no)) return s;
        int nd = out->ndim;
        bool ok = nd >= 2 && grad->ndim == nd && indices->dtype == HMR_I64 &&
                  grad->dtype == out->dtype;
        for (int k = 0; ok && k < nd - 2; ++k) ok = grad->sizes[k] == out->sizes[k];
        // The output is contiguous: plane p holds elements p * plane_size onwards.
        int64_t expected = 1;
        for (int k = nd - 1; ok && k >= 0; --k) {
            ok = out->sizes[k] == 1 || out->strides[k] == expected;
            expected *= out->sizes[k];
        }
        if (!ok) return hmr::fail(HMR_INVALID, "invalid max_pool2d_backward");
        int64_t need = 0;
        if (hmr_status s = hmr::scatter_bytes(n, need)) return s;
        if (hmr_status s = hmr::check_workspace(need, workspace_bytes, workspace)) return s;
        if (n == 0 || no == 0) return HMR_OK;
        int64_t plane_size = out->sizes[nd - 2] * out->sizes[nd - 1];
        int64_t per_plane = grad->sizes[nd - 2] * grad->sizes[nd - 1];
        hmr::SortSpace sp = hmr::sort_space(static_cast<uint64_t *>(workspace), n);
        return hmr::launch_on(ctx, "max_pool2d_backward", [&](hipStream_t s) {
            hipLaunchKernelGGL(hmr::pool_keys_kernel, hmr::blocks_for(n), dim3(hmr::kThreads), 0,
                               s, *indices, n, per_plane, plane_size, sp.keys[0], sp.vals[0]);
            int cur = hmr::radix_sort(sp, static_cast<uint64_t>(no), s);
            return hmr::with_acc<hmr::RoundingAcc>(out->dtype, hmr::FloatDtypes{}, [&](auto acc) {
                hipLaunchKernelGGL((hmr::run_add_kernel<decltype(acc)>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, sp.keys[cur], sp.vals[cur], n, *out,
                                   *grad);
            });
        });
    });
}

hmr_status hmr_mask_workspace(int64_t n, int64_t *bytes, int64_t *total_word) {
    return guarded([&] {
        if (n < 0) return hmr::fail(HMR_INVALID, "negative size");
        int64_t tiles = hmr::mask_tiles(n);
        *total_word = tiles + 2 * hmr::scan_chunks(tiles);
        return hmr::mask_bytes(n, *bytes);
    });
}

hmr_status hmr_mask_count(hmr_context_t ctx, const hmr_tensor_desc *mask, void *workspace,
                          int64_t workspace_bytes) {
    return guarded([&] {
        int64_t n = 0;
        if (hmr_status s = hmr::check_desc(*mask, n)) return s;
        if (mask->dtype != HMR_BOOL) return hmr::fail(HMR_INVALID, "a mask must be BOOL");
        int64_t tiles = hmr::mask_tiles(n);
        int64_t need = 0;
        if (hmr_status s = hmr::mask_bytes(n, need)) return s;
        if (hmr_status s = hmr::check_workspace(need, workspace_bytes, workspace)) return s;
        auto *counts = static_cast<uint64_t *>(workspace);
        return hmr::launch_on(ctx, "mask count", [&](hipStream_t s) {
            if (tiles > 0) {
                hipLaunchKernelGGL(hmr::mask_count_kernel, dim3(static_cast<unsigned>(tiles)),
                                   dim3(hmr::kScanThreads), 0, s, *mask, n, counts);
            }
            hmr::device_exclusive_scan(counts, tiles, counts + tiles, s);
            return true;
        });
    });
}

hmr_status hmr_mask_coordinates(hmr_context_t ctx, const hmr_tensor_desc *mask, void *workspace,
                                int64_t workspace_bytes, const hmr_tensor_desc *out) {
    return guarded([&] {
        int64_t n = 0, no = 0;
        if (hmr_status s = hmr::check_desc(*mask, n)) return s;
        if (hmr_status s = hmr::check_desc(*out, no)) return s;
        bool ok = mask->dtype == HMR_BOOL && out->dtype == HMR_I64 && out->ndim == 2 &&
                  out->sizes[0] == mask->ndim && out->strides[1] == 1 &&
                  (out->sizes[0] <= 1 || out->strides[0] == out->sizes[1]);
        if (!ok) return hmr::fail(HMR_INVALID, "invalid mask coordinates");
        int64_t tiles = hmr::mask_tiles(n);
        int64_t need = 0;
        if (hmr_status s = hmr::mask_bytes(n, need)) return s;
        if (hmr_status s = hmr::check_workspace(need, workspace_bytes, workspace)) return s;
        if (no == 0) return HMR_OK;
        auto *starts = static_cast<const uint64_t *>(workspace);
        int64_t *coords = static_cast<int64_t *>(out->data) + out->offset;
        return hmr::launch_on(ctx, "mask coordinates", [&](hipStream_t s) {
            hipLaunchKernelGGL(hmr::mask_write_kernel, dim3(static_cast<unsigned>(tiles)),
                               dim3(hmr::kScanThreads), 0, s, *mask, n, starts, coords,
                               out->sizes[1]);
            return true;
        });
    });
}

}  // extern "C"
