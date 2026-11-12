// Pooling kernels and the backward of unfold: max_pool2d, its check and gathering backward,
// adaptive_avg_pool2d and its backward, unfold_backward (the backward of max_pool2d for indices
// outside their windows is a scatter, in index.hip). The counterparts of
// src/backend/cpu/pool.rs and cpu::unfold_backward, with the CPU's bits: each output element
// computes what the CPU computes for it, in the same order and type.
//
// The CPU's backward kernels add each gradient element into the positions it covers, one at a
// time in output order. Here each position of the result instead reads the gradient elements
// that cover it, in that same output order, starting from zero, so the sums (rounded after every
// addition, as on the CPU) are the same, every element is written, and nothing is atomic.
#include <cstdint>
#include <limits>
#include <type_traits>

#include "elem.hpp"
#include "kernel.hpp"

using hmr::guarded;

namespace hmr {
namespace {

// Pool2d::window: the input positions [first, end) (every `dilation`) of output position `o`.
__device__ inline void pool_window(int64_t o, int64_t kernel, int64_t stride, int64_t padding,
                                   int64_t dilation, int64_t n, int64_t &first, int64_t &end) {
    int64_t start = o * stride - padding;
    int64_t e = start + (kernel - 1) * dilation + 1;
    if (e > n) e = n;
    first = start < 0 ? start + (-start + dilation - 1) / dilation * dilation : start;
    end = e > first ? e : first;
}

// AdaptiveWindows::window: [floor(i * input / output), ceil((i + 1) * input / output)).
__device__ inline void adaptive_window(int64_t i, int64_t output, int64_t input, int64_t &start,
                                       int64_t &end) {
    start = (i / output) * input + ((i % output) * input) / output;
    end = 1 + ((i + 1) * input - 1) / output;
}

// The geometry of a 2-D pooling: kernel, stride, padding and dilation for the height and the
// width.
struct Geometry {
    int64_t kernel[2], stride[2], padding[2], dilation[2];
};

// The geometry from the C ABI's eight words (hmr_max_pool2d), checked.
hmr_status read_geometry(const int64_t *geometry, Geometry &g) {
    for (int d = 0; d < 2; ++d) {
        g.kernel[d] = geometry[d];
        g.stride[d] = geometry[2 + d];
        g.padding[d] = geometry[4 + d];
        g.dilation[d] = geometry[6 + d];
        if (g.kernel[d] < 1 || g.stride[d] < 1 || g.padding[d] < 0 || g.dilation[d] < 1) {
            return fail(HMR_INVALID, "invalid max_pool2d geometry");
        }
    }
    return HMR_OK;
}

template <int DT, Kind K = Elem<DT>::kind>
struct MaxOf {
    // Floating point: the candidate starts at -inf, a value replaces it if greater or NaN.
    using T = typename Elem<DT>::T;
    __device__ static T lowest() { return down<DT>(-INFINITY); }
    __device__ static bool replaces(T v, T best) {
        auto x = up<DT>(v);
        return x > up<DT>(best) || x != x;
    }
};

template <int DT>
struct MaxOf<DT, kInt> {
    using T = typename Elem<DT>::T;
    __device__ static T lowest() { return std::numeric_limits<T>::min(); }
    __device__ static bool replaces(T v, T best) { return v > best; }
};

// Over the elements of out (operand 0), whose last two dimensions are the output's height and
// width; operand 1 is the indices, operand 2 walks the input's planes.
template <int DT>
__global__ void max_pool_kernel(Operands<3> ops, int64_t n, Geometry g, int64_t height,
                                int64_t width, int64_t sh, int64_t sw) {
    using T = typename Elem<DT>::T;
    using M = MaxOf<DT>;
    const hmr_tensor_desc &od = ops.d[0];
    int64_t out_w = od.sizes[od.ndim - 1], out_h = od.sizes[od.ndim - 2];
    T *out = static_cast<T *>(od.data);
    int64_t *indices = static_cast<int64_t *>(ops.d[1].data);
    const T *x = static_cast<const T *>(ops.d[2].data);
    for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t o[3];
        offsets_n(ops, e, o);
        int64_t c = e % out_w, r = (e / out_w) % out_h;
        int64_t h0, h1, w0, w1;
        pool_window(r, g.kernel[0], g.stride[0], g.padding[0], g.dilation[0], height, h0, h1);
        pool_window(c, g.kernel[1], g.stride[1], g.padding[1], g.dilation[1], width, w0, w1);
        T best = M::lowest();
        int64_t index = h0 * width + w0;
        for (int64_t ih = h0; ih < h1; ih += g.dilation[0]) {
            for (int64_t iw = w0; iw < w1; iw += g.dilation[1]) {
                T v = x[o[2] + ih * sh + iw * sw];
                if (M::replaces(v, best)) {
                    best = v;
                    index = ih * width + iw;
                }
            }
        }
        out[o[0]] = best;
        indices[o[1]] = index;
    }
}

constexpr uint64_t kNoPosition = ~uint64_t{0};

// hmr_test_wide_pool: the int64_t coordinates even where int32_t ones fit.
thread_local bool wide_pool = false;

// The geometry with the coordinate type I: int32_t where every coordinate of a pooling fits in
// it (fits_in with INT32_MAX), else int64_t: divisions of int64_t are slow on the GPU.
template <class I>
struct GeometryOf {
    I kernel[2], stride[2], padding[2], dilation[2];
};

template <class I>
GeometryOf<I> geometry_of(const Geometry &g) {
    GeometryOf<I> r;
    for (int d = 0; d < 2; ++d) {
        r.kernel[d] = static_cast<I>(g.kernel[d]);
        r.stride[d] = static_cast<I>(g.stride[d]);
        r.padding[d] = static_cast<I>(g.padding[d]);
        r.dilation[d] = static_cast<I>(g.dilation[d]);
    }
    return r;
}

// The windows along one dimension, for the gathering backward: output position o's window has
// the input positions o * stride - padding + k * dilation, 0 <= k < kernel.
template <class I>
struct Axis {
    I kernel, stride, padding, dilation;
};

template <class I>
Axis<I> axis_of(const Geometry &g, int d) {
    Axis<I> r;
    r.kernel = static_cast<I>(g.kernel[d]);
    r.stride = static_cast<I>(g.stride[d]);
    r.padding = static_cast<I>(g.padding[d]);
    r.dilation = static_cast<I>(g.dilation[d]);
    return r;
}

// A walk over the output positions (of `out`) whose windows cover input position i along one
// dimension, in increasing order: all of them, and no others (review round 135, R159).
// - With a dilation of 1, the range of those whose windows reach i, o * stride - padding <= i <=
//   o * stride - padding + kernel - 1: every one covers i. No division per position.
// - Otherwise, o = (i + padding - k * dilation) / stride for k = kernel - 1 ... 0 where that
//   divides and lies in the output (o rises as k falls): at most `kernel` positions, however
//   large the dilation; the range would hold about (kernel - 1) * dilation / stride.
// `o` is the current position; `last` ends the range, `k` the next k to try.
template <class I>
struct Walk {
    I o, last, k;
};

template <class I>
__host__ __device__ inline bool walk_dilated(const Axis<I> &a, I i, I out, Walk<I> &w) {
    using U = std::make_unsigned_t<I>;
    for (I k = w.k - 1; k >= 0; --k) {
        I t = i + a.padding - k * a.dilation;
        if (t < 0 || static_cast<U>(t) % static_cast<U>(a.stride) != 0) continue;
        I o = static_cast<I>(static_cast<U>(t) / static_cast<U>(a.stride));
        // The later k give later positions.
        if (o >= out) return false;
        w.o = o;
        w.k = k;
        return true;
    }
    return false;
}

// Starts `w` at the first position; false if there is none.
template <class I>
__host__ __device__ inline bool walk_first(const Axis<I> &a, I i, I out, Walk<I> &w) {
    using U = std::make_unsigned_t<I>;
    if (a.dilation != 1) {
        w.k = a.kernel;
        return walk_dilated(a, i, out, w);
    }
    U s = static_cast<U>(a.stride);
    I top = i + a.padding, bottom = top - (a.kernel - 1);
    w.last = static_cast<I>(static_cast<U>(top) / s);
    if (w.last > out - 1) w.last = out - 1;
    w.o = bottom <= 0 ? 0 : static_cast<I>((static_cast<U>(bottom) + s - 1) / s);
    return w.o <= w.last;
}

// `a` known to have a dilation of 1, for the compiler.
template <class I>
__host__ __device__ inline Axis<I> undilated(Axis<I> a) {
    a.dilation = 1;
    return a;
}

// Moves `w` to the next position; false after the last.
template <class I>
__host__ __device__ inline bool walk_next(const Axis<I> &a, I i, I out, Walk<I> &w) {
    if (a.dilation != 1) return walk_dilated(a, i, out, w);
    return ++w.o <= w.last;
}

// Whether the coordinates of a pooling of an input of `in` positions per dimension into `out`
// fit in [-max, max]: the sizes, the positions per plane, and every position in a window,
// relative or not (bounded by out * stride + padding + in + (kernel - 1) * dilation).
bool fits_in(int64_t max, const Geometry &g, const int64_t (&in)[2], const int64_t (&out)[2]) {
    int64_t a = 0, b = 0;
    if (!checked_mul(in[0], in[1], a) || a > max || !checked_mul(out[0], out[1], b) || b > max) {
        return false;
    }
    for (int d = 0; d < 2; ++d) {
        int64_t span = 0, reach = 0;
        if (!checked_mul(g.kernel[d] - 1, g.dilation[d], span) ||
            !checked_mul(out[d], g.stride[d], reach) || !checked_add(reach, span, reach) ||
            !checked_add(reach, g.padding[d], reach) || !checked_add(reach, in[d], reach) ||
            reach > max) {
            return false;
        }
    }
    return true;
}

// The largest offset inside a plane of `d` (its last two dimensions, which have elements): the
// sum of |stride| * (size - 1), or INT64_MAX if that overflows.
int64_t plane_span(const hmr_tensor_desc &d) {
    int64_t span = 0;
    for (int k = d.ndim - 2; k < d.ndim; ++k) {
        int64_t stride = d.strides[k] < 0 ? -d.strides[k] : d.strides[k], part = 0;
        if (d.strides[k] == INT64_MIN || !checked_mul(stride, d.sizes[k] - 1, part) ||
            !checked_add(span, part, span)) {
            return INT64_MAX;
        }
    }
    return span;
}

// Whether input position `i` of dimension `d` is in the window of output position `o`: one of
// o * stride - padding + k * dilation, 0 <= k < kernel. Pool2d::window gives the same positions,
// those inside the input, so the indices of the forward pass are. The gather finds exactly these
// windows (`Walk`), so an index between the positions of a dilated window goes to the sort
// (review round 135, R159).
template <class I>
__device__ inline bool in_window(const GeometryOf<I> &g, int d, I i, I o) {
    using U = std::make_unsigned_t<I>;
    I t = i - (o * g.stride[d] - g.padding[d]);
    if (t < 0 || t > (g.kernel[d] - 1) * g.dilation[d]) return false;
    return g.dilation[d] == 1 || static_cast<U>(t) % static_cast<U>(g.dilation[d]) == 0;
}

// A description of the planes of `of` (its leading dimensions, all but the last two), with the
// sizes of `sizes`' planes: walking it by plane number gives the offset of each plane of `of`.
hmr_tensor_desc plane_desc(const hmr_tensor_desc &sizes, const hmr_tensor_desc &of) {
    hmr_tensor_desc w = of;
    w.ndim = sizes.ndim - 2;
    for (int k = 0; k < w.ndim; ++k) {
        w.sizes[k] = sizes.sizes[k];
        w.strides[k] = of.strides[k];
    }
    return w;
}

// The offset in the indices (planes as `planes`, strides sh and sw within a plane) of row-major
// position e of an output of out_h x out_w = per_plane positions per plane.
__device__ inline int64_t pool_offset(const Operands<1> &planes, int64_t e, int64_t per_plane,
                                      int64_t out_w, int64_t sh, int64_t sw) {
    int64_t p = e / per_plane, rest = e - p * per_plane, r = rest / out_w, c = rest - r * out_w;
    int64_t o[1];
    offsets_n(planes, p, o);
    return o[0] + r * sh + c * sw;
}

// The kernels over planes x rows x columns run blocks of kTileW columns x kTileH rows (a
// wavefront of 32 covers a row), on a grid of (columns, rows, planes) tiles of at most 65535 blocks
// each, and stride over the rest. The coordinates come from the position of the thread, without
// divisions, and the offsets of a plane are computed once per block.
constexpr int kTileW = 32, kTileH = 8;

inline dim3 tile_blocks(int64_t planes, int64_t h, int64_t w) {
    auto cap = [](int64_t b) { return static_cast<unsigned>(b < 1 ? 1 : b > 65535 ? 65535 : b); };
    return dim3(cap(ceil_div(w, kTileW)), cap(ceil_div(h, kTileH)), cap(planes));
}

// The check of max_pool2d's indices for its backward: slot[0] = the first position, in row-major
// order, whose index is outside the plane of `size` elements (kNoPosition for none), slot[1] its
// index, slot[2] = 1 if an index inside the plane is outside the window of its output position.
__global__ void pool_slot_init_kernel(uint64_t *slot) {
    slot[0] = kNoPosition;
    slot[1] = 0;
    slot[2] = 0;
}

// Over the `count` planes of out_h x out_w indices.
template <class I>
__global__ void pool_check_kernel(Operands<1> planes, int64_t count, GeometryOf<I> g, I out_h,
                                  I out_w, int64_t sh, int64_t sw, I width, int64_t size,
                                  unsigned long long *slot) {
    const int64_t *indices = static_cast<const int64_t *>(planes.d[0].data);
    bool outside = false;
    for (int64_t p = blockIdx.z; p < count; p += gridDim.z) {
        int64_t o[1];
        offsets_n(planes, p, o);
        for (I r = static_cast<I>(blockIdx.y) * kTileH + static_cast<I>(threadIdx.y); r < out_h;
             r += static_cast<I>(gridDim.y) * kTileH) {
            for (I c = static_cast<I>(blockIdx.x) * kTileW + static_cast<I>(threadIdx.x);
                 c < out_w; c += static_cast<I>(gridDim.x) * kTileW) {
                int64_t v = indices[o[0] + r * sh + c * sw];
                if (v < 0 || v >= size) {
                    int64_t e = (p * out_h + r) * out_w + c;
                    atomicMin(slot, static_cast<unsigned long long>(e));
                    continue;
                }
                I at = static_cast<I>(v), y = at / width, x = at - y * width;
                if (!in_window(g, 0, y, r) || !in_window(g, 1, x, c)) outside = true;
            }
        }
    }
    if (outside) atomicOr(slot + 2, 1ull);
}

__global__ void pool_slot_value_kernel(Operands<1> planes, int64_t per_plane, int64_t out_w,
                                       int64_t sh, int64_t sw, uint64_t *slot) {
    uint64_t p = slot[0];
    if (p == kNoPosition) return;
    int64_t o = pool_offset(planes, static_cast<int64_t>(p), per_plane, out_w, sh, sw);
    slot[1] = static_cast<uint64_t>(static_cast<const int64_t *>(planes.d[0].data)[o]);
}

// The backward of max_pool2d for indices inside their windows, over the `count`
// planes of height x width elements of the contiguous `out` (the input's shape): each gathers,
// from zero, the elements of the gradient (its planes are operand 0 of `planes`; out_h x out_w,
// strides gh and gw) whose index (operand 1; strides ih and iw) is its position, rounding every
// addition in the dtype like the CPU's scatter. The output positions whose windows cover it
// (`Walk`: all of them, for every kh and kw) are enumerated in the CPU's order, the gradient's
// row-major order: rows rising in the outer loop, columns in the inner one.
template <int DT, class I, bool Dilated>
__global__ void max_pool_gather_kernel(Operands<2> planes, typename Elem<DT>::T *out,
                                       int64_t count, Axis<I> rows, Axis<I> cols, I height,
                                       I width, I out_h, I out_w, I gh, I gw, I ih, I iw) {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    int64_t per_plane = static_cast<int64_t>(height) * width;
    for (int64_t p = blockIdx.z; p < count; p += gridDim.z) {
        int64_t o[2];
        offsets_n(planes, p, o);
        const T *grad = static_cast<const T *>(planes.d[0].data) + o[0];
        const int64_t *indices = static_cast<const int64_t *>(planes.d[1].data) + o[1];
        T *plane = out + p * per_plane;
        for (I y = static_cast<I>(blockIdx.y) * kTileH + static_cast<I>(threadIdx.y); y < height;
             y += static_cast<I>(gridDim.y) * kTileH) {
            for (I x = static_cast<I>(blockIdx.x) * kTileW + static_cast<I>(threadIdx.x);
                 x < width; x += static_cast<I>(gridDim.x) * kTileW) {
                I at = y * width + x;
                T acc = down<DT>(C(0));
                Walk<I> r, c;
                // Without a dilation the walks are ranges, in plain loops.
                for (bool row = Dilated ? walk_first(rows, y, out_h, r)
                                        : walk_first(undilated(rows), y, out_h, r);
                     row; row = Dilated ? walk_next(rows, y, out_h, r) : ++r.o <= r.last) {
                    for (bool col = Dilated ? walk_first(cols, x, out_w, c)
                                            : walk_first(undilated(cols), x, out_w, c);
                         col; col = Dilated ? walk_next(cols, x, out_w, c) : ++c.o <= c.last) {
                        // Both loads, then a choice rather than a branch: no load waits for
                        // another.
                        int64_t index = indices[r.o * ih + c.o * iw];
                        T sum = down<DT>(up<DT>(acc) + up<DT>(grad[r.o * gh + c.o * gw]));
                        acc = index == at ? sum : acc;
                    }
                }
                plane[at] = acc;
            }
        }
    }
}

// Over the elements of out (operand 0); operand 1 walks the input's planes.
template <int DT>
__global__ void adaptive_avg_kernel(Operands<2> ops, int64_t n, int64_t height, int64_t width,
                                    int64_t sh, int64_t sw) {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    const hmr_tensor_desc &od = ops.d[0];
    int64_t out_w = od.sizes[od.ndim - 1], out_h = od.sizes[od.ndim - 2];
    T *out = static_cast<T *>(od.data);
    const T *x = static_cast<const T *>(ops.d[1].data);
    for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t o[2];
        offsets_n(ops, e, o);
        int64_t c = e % out_w, r = (e / out_w) % out_h;
        int64_t h0, h1, w0, w1;
        adaptive_window(r, out_h, height, h0, h1);
        adaptive_window(c, out_w, width, w0, w1);
        C sum = 0;
        for (int64_t ih = h0; ih < h1; ++ih) {
            for (int64_t iw = w0; iw < w1; ++iw) sum += up<DT>(x[o[1] + ih * sh + iw * sw]);
        }
        C mean = sum / static_cast<C>(h1 - h0) / static_cast<C>(w1 - w0);
        out[o[0]] = down<DT>(mean);
    }
}

// The output positions `lo..=hi` whose adaptive windows may cover input position `i`: the exact
// range, widened by one on each side (the caller checks each window).
__device__ inline void covering(int64_t i, int64_t output, int64_t input, int64_t &lo,
                                int64_t &hi) {
    lo = i * output / input - 1;
    hi = ((i + 1) * output + input - 1) / input;
    if (lo < 0) lo = 0;
    if (hi > output - 1) hi = output - 1;
}

// Over the elements of out (operand 0, the input's shape); operand 1 walks the gradient's
// planes, whose height and width are out_h and out_w with strides gh and gw.
template <int DT>
__global__ void adaptive_avg_backward_kernel(Operands<2> ops, int64_t n, int64_t out_h,
                                             int64_t out_w, int64_t gh, int64_t gw) {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    const hmr_tensor_desc &od = ops.d[0];
    int64_t width = od.sizes[od.ndim - 1], height = od.sizes[od.ndim - 2];
    T *out = static_cast<T *>(od.data);
    const T *grad = static_cast<const T *>(ops.d[1].data);
    // A window size as the dtype, then the computation type.
    auto size = [](int64_t k) { return up<DT>(down<DT>(static_cast<C>(k))); };
    for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t o[2];
        offsets_n(ops, e, o);
        int64_t iw = e % width, ih = (e / width) % height;
        int64_t r_lo, r_hi, c_lo, c_hi;
        covering(ih, out_h, height, r_lo, r_hi);
        covering(iw, out_w, width, c_lo, c_hi);
        T acc = down<DT>(C(0));
        for (int64_t r = r_lo; r <= r_hi; ++r) {
            int64_t h0, h1;
            adaptive_window(r, out_h, height, h0, h1);
            if (ih < h0 || ih >= h1) continue;
            for (int64_t c = c_lo; c <= c_hi; ++c) {
                int64_t w0, w1;
                adaptive_window(c, out_w, width, w0, w1);
                if (iw < w0 || iw >= w1) continue;
                T v = grad[o[1] + r * gh + c * gw];
                T delta = down<DT>(up<DT>(down<DT>(up<DT>(v) / size(h1 - h0))) / size(w1 - w0));
                acc = down<DT>(up<DT>(acc) + up<DT>(delta));
            }
        }
        out[o[0]] = acc;
    }
}

// Over the elements of out (operand 0, x's shape); operand 1 walks the gradient without the
// unfolded dimension and the window's, operand 2 is the coordinate along the unfolded one.
template <int DT>
__global__ void unfold_backward_kernel(Operands<3> ops, int64_t n, int64_t size, int64_t step,
                                       int64_t windows, int64_t gd, int64_t gl) {
    using T = typename Elem<DT>::T;
    using C = typename Comp<DT>::C;
    T *out = static_cast<T *>(ops.d[0].data);
    const T *grad = static_cast<const T *>(ops.d[1].data);
    for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t o[3];
        offsets_n(ops, e, o);
        int64_t i = o[2];
        T value = down<DT>(C(0));
        if (step >= size) {
            int64_t w = i / step, j = i % step;
            if (w < windows && j < size) value = grad[o[1] + w * gd + j * gl];
        } else {
            int64_t first = i >= size ? (i - size) / step + 1 : 0;
            int64_t last = i / step < windows - 1 ? i / step : windows - 1;
            for (int64_t w = first; w <= last; ++w) {
                value = down<DT>(up<DT>(value) + up<DT>(grad[o[1] + w * gd + (i - w * step) * gl]));
            }
        }
        out[o[0]] = value;
    }
}

// A description with the sizes of `sizes` and the data and offset of `of`, walking `of`'s planes:
// its strides for the leading dimensions, 0 for the last two.
hmr_tensor_desc planes_walk(const hmr_tensor_desc &sizes, const hmr_tensor_desc &of) {
    hmr_tensor_desc w = of;
    w.ndim = sizes.ndim;
    for (int k = 0; k < sizes.ndim; ++k) {
        w.sizes[k] = sizes.sizes[k];
        w.strides[k] = k < sizes.ndim - 2 ? of.strides[k] : 0;
    }
    return w;
}

bool same_planes(const hmr_tensor_desc &a, const hmr_tensor_desc &b) {
    if (a.ndim != b.ndim || a.ndim < 2) return false;
    for (int k = 0; k < a.ndim - 2; ++k) {
        if (a.sizes[k] != b.sizes[k]) return false;
    }
    return true;
}

}  // namespace
}  // namespace hmr

extern "C" {

hmr_status hmr_max_pool2d(hmr_context_t ctx, const hmr_tensor_desc *out,
                          const hmr_tensor_desc *indices, const hmr_tensor_desc *input,
                          const int64_t *geometry) {
    return guarded([&] {
        hmr::Operands<2> oi{{*out, *indices}};
        int64_t n = 0, nx = 0;
        if (hmr_status s = hmr::check_operands(oi, n)) return s;
        if (hmr_status s = hmr::check_desc(*input, nx)) return s;
        if (!hmr::same_planes(*out, *input) || indices->dtype != HMR_I64 ||
            out->dtype != input->dtype) {
            return hmr::fail(HMR_INVALID, "invalid max_pool2d");
        }
        hmr::Geometry g;
        if (hmr_status s = hmr::read_geometry(geometry, g)) return s;
        if (n == 0) return HMR_OK;
        int nd = input->ndim;
        hmr::Operands<3> ops{{*out, *indices, hmr::planes_walk(*out, *input)}};
        return hmr::launch_on(ctx, "max_pool2d", [&](hipStream_t s) {
            auto body = [&](auto t) {
                constexpr int DT = decltype(t)::v;
                hipLaunchKernelGGL((hmr::max_pool_kernel<DT>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, ops, n, g, input->sizes[nd - 2],
                                   input->sizes[nd - 1], input->strides[nd - 2],
                                   input->strides[nd - 1]);
                return true;
            };
            return hmr::select(out->dtype, hmr::FloatDtypes{}, body) ||
                   hmr::select(out->dtype, hmr::IntDtypes{}, body);
        });
    });
}

hmr_status hmr_max_pool2d_check(hmr_context_t ctx, const hmr_tensor_desc *indices,
                                int64_t height, int64_t width, const int64_t *geometry,
                                uint64_t *slot) {
    return guarded([&] {
        hmr::Operands<1> ops{{*indices}};
        int64_t n = 0;
        if (hmr_status s = hmr::check_operands(ops, n)) return s;
        hmr::Geometry g;
        if (hmr_status s = hmr::read_geometry(geometry, g)) return s;
        int nd = indices->ndim;
        if (nd < 2 || indices->dtype != HMR_I64 || height < 0 || width < 0) {
            return hmr::fail(HMR_INVALID, "invalid max_pool2d check");
        }
        int64_t in[2] = {height, width}, out[2] = {indices->sizes[nd - 2], indices->sizes[nd - 1]};
        if (!hmr::fits_in(INT64_MAX / 2, g, in, out)) {
            return hmr::fail(HMR_INVALID, "max_pool2d geometry too large");
        }
        int64_t size = height * width, per_plane = out[0] * out[1];
        int64_t sh = indices->strides[nd - 2], sw = indices->strides[nd - 1];
        hmr::Operands<1> planes{{hmr::plane_desc(*indices, *indices)}};
        int64_t count = per_plane == 0 ? 0 : n / per_plane;
        // Half the range: the loops over the tiles step past the end by up to 65535 tiles.
        bool narrow = !hmr::wide_pool && hmr::fits_in(INT32_MAX / 2, g, in, out);
        return hmr::launch_on(ctx, "max_pool2d check", [&](hipStream_t s) {
            hipLaunchKernelGGL(hmr::pool_slot_init_kernel, dim3(1), dim3(1), 0, s, slot);
            if (n == 0) return true;
            auto ull = reinterpret_cast<unsigned long long *>(slot);
            auto check = [&](auto i) {
                using I = decltype(i);
                hipLaunchKernelGGL((hmr::pool_check_kernel<I>),
                                   hmr::tile_blocks(count, out[0], out[1]),
                                   dim3(hmr::kTileW, hmr::kTileH), 0, s, planes, count,
                                   hmr::geometry_of<I>(g),
                                   static_cast<I>(out[0]), static_cast<I>(out[1]), sh, sw,
                                   static_cast<I>(width), size, ull);
            };
            if (narrow) {
                check(int32_t{});
            } else {
                check(int64_t{});
            }
            hipLaunchKernelGGL(hmr::pool_slot_value_kernel, dim3(1), dim3(1), 0, s, planes,
                               per_plane, out[1], sh, sw, slot);
            return true;
        });
    });
}

hmr_status hmr_max_pool2d_gather_backward(hmr_context_t ctx, const hmr_tensor_desc *out,
                                          const hmr_tensor_desc *grad,
                                          const hmr_tensor_desc *indices,
                                          const int64_t *geometry) {
    return guarded([&] {
        hmr::Operands<2> gi{{*grad, *indices}};
        int64_t n = 0, ng = 0;
        if (hmr_status s = hmr::check_desc(*out, n)) return s;
        if (hmr_status s = hmr::check_operands(gi, ng)) return s;
        hmr::Geometry g;
        if (hmr_status s = hmr::read_geometry(geometry, g)) return s;
        int nd = out->ndim;
        bool ok = hmr::same_planes(*out, *grad) && grad->ndim == nd && indices->ndim == nd &&
                  indices->dtype == HMR_I64 && grad->dtype == out->dtype;
        for (int k = 0; ok && k < nd; ++k) ok = indices->sizes[k] == grad->sizes[k];
        // The output is contiguous: element e is at out.offset + e.
        int64_t expected = 1;
        for (int k = nd - 1; ok && k >= 0; --k) {
            ok = out->sizes[k] == 1 || out->strides[k] == expected;
            expected *= out->sizes[k];
        }
        if (!ok) return hmr::fail(HMR_INVALID, "invalid max_pool2d_backward");
        int64_t in[2] = {out->sizes[nd - 2], out->sizes[nd - 1]};
        int64_t outs[2] = {grad->sizes[nd - 2], grad->sizes[nd - 1]};
        if (!hmr::fits_in(INT64_MAX / 2, g, in, outs)) {
            return hmr::fail(HMR_INVALID, "max_pool2d geometry too large");
        }
        if (n == 0) return HMR_OK;
        hmr::Operands<2> planes{{hmr::plane_desc(*out, *grad), hmr::plane_desc(*out, *indices)}};
        int64_t count = n / (in[0] * in[1]);
        // Half the range: the loops over the tiles step past the end by up to 65535 tiles. The
        // offsets inside a plane of the gradient and the indices must fit too.
        bool narrow = !hmr::wide_pool && hmr::fits_in(INT32_MAX / 2, g, in, outs) &&
                      hmr::plane_span(*grad) <= INT32_MAX / 2 &&
                      hmr::plane_span(*indices) <= INT32_MAX / 2;
        return hmr::launch_on(ctx, "max_pool2d_backward", [&](hipStream_t s) {
            return hmr::select(out->dtype, hmr::FloatDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                using T = typename hmr::Elem<DT>::T;
                T *o = static_cast<T *>(out->data) + out->offset;
                bool dilated = g.dilation[0] != 1 || g.dilation[1] != 1;
                auto gather = [&](auto i, auto d) {
                    using I = decltype(i);
                    hipLaunchKernelGGL((hmr::max_pool_gather_kernel<DT, I, decltype(d)::value>),
                                       hmr::tile_blocks(count, in[0], in[1]),
                                       dim3(hmr::kTileW, hmr::kTileH), 0, s, planes, o, count,
                                       hmr::axis_of<I>(g, 0), hmr::axis_of<I>(g, 1),
                                       static_cast<I>(in[0]),
                                       static_cast<I>(in[1]), static_cast<I>(outs[0]),
                                       static_cast<I>(outs[1]),
                                       static_cast<I>(grad->strides[nd - 2]),
                                       static_cast<I>(grad->strides[nd - 1]),
                                       static_cast<I>(indices->strides[nd - 2]),
                                       static_cast<I>(indices->strides[nd - 1]));
                };
                using Plain = std::false_type;
                using Dilated = std::true_type;
                if (narrow && dilated) {
                    gather(int32_t{}, Dilated{});
                } else if (narrow) {
                    gather(int32_t{}, Plain{});
                } else if (dilated) {
                    gather(int64_t{}, Dilated{});
                } else {
                    gather(int64_t{}, Plain{});
                }
                return true;
            });
        });
    });
}

hmr_status hmr_adaptive_avg_pool2d(hmr_context_t ctx, const hmr_tensor_desc *out,
                                   const hmr_tensor_desc *input) {
    return guarded([&] {
        int64_t n = 0, nx = 0;
        if (hmr_status s = hmr::check_desc(*out, n)) return s;
        if (hmr_status s = hmr::check_desc(*input, nx)) return s;
        if (!hmr::same_planes(*out, *input) || out->dtype != input->dtype) {
            return hmr::fail(HMR_INVALID, "invalid adaptive_avg_pool2d");
        }
        if (n == 0) return HMR_OK;
        int nd = input->ndim;
        hmr::Operands<2> ops{{*out, hmr::planes_walk(*out, *input)}};
        return hmr::launch_on(ctx, "adaptive_avg_pool2d", [&](hipStream_t s) {
            return hmr::select(out->dtype, hmr::FloatDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                hipLaunchKernelGGL((hmr::adaptive_avg_kernel<DT>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, ops, n, input->sizes[nd - 2],
                                   input->sizes[nd - 1], input->strides[nd - 2],
                                   input->strides[nd - 1]);
                return true;
            });
        });
    });
}

hmr_status hmr_adaptive_avg_pool2d_backward(hmr_context_t ctx, const hmr_tensor_desc *out,
                                            const hmr_tensor_desc *grad) {
    return guarded([&] {
        int64_t n = 0, ng = 0;
        if (hmr_status s = hmr::check_desc(*out, n)) return s;
        if (hmr_status s = hmr::check_desc(*grad, ng)) return s;
        if (!hmr::same_planes(*out, *grad) || out->dtype != grad->dtype) {
            return hmr::fail(HMR_INVALID, "invalid adaptive_avg_pool2d_backward");
        }
        int nd = grad->ndim;
        if (n == 0) return HMR_OK;
        if (ng == 0) return hmr::fail(HMR_INVALID, "adaptive_avg_pool2d_backward without windows");
        hmr::Operands<2> ops{{*out, hmr::planes_walk(*out, *grad)}};
        return hmr::launch_on(ctx, "adaptive_avg_pool2d_backward", [&](hipStream_t s) {
            return hmr::select(out->dtype, hmr::FloatDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                hipLaunchKernelGGL((hmr::adaptive_avg_backward_kernel<DT>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, ops, n, grad->sizes[nd - 2],
                                   grad->sizes[nd - 1], grad->strides[nd - 2],
                                   grad->strides[nd - 1]);
                return true;
            });
        });
    });
}

hmr_status hmr_unfold_backward(hmr_context_t ctx, const hmr_tensor_desc *out,
                               const hmr_tensor_desc *grad, int32_t dim, int64_t size,
                               int64_t step) {
    return guarded([&] {
        int64_t n = 0, ng = 0;
        if (hmr_status s = hmr::check_desc(*out, n)) return s;
        if (hmr_status s = hmr::check_desc(*grad, ng)) return s;
        int nd = out->ndim;
        bool ok = nd >= 1 && grad->ndim == nd + 1 && dim >= 0 && dim < nd && size >= 0 &&
                  step >= 1 && grad->sizes[nd] == size && out->dtype == grad->dtype;
        for (int k = 0; ok && k < nd; ++k) ok = k == dim || grad->sizes[k] == out->sizes[k];
        if (!ok) return hmr::fail(HMR_INVALID, "invalid unfold_backward");
        if (n == 0) return HMR_OK;
        // The gradient walked at out's coordinates but the unfolded one.
        hmr_tensor_desc g = *grad;
        g.ndim = nd;
        for (int k = 0; k < nd; ++k) {
            g.sizes[k] = out->sizes[k];
            g.strides[k] = k == dim ? 0 : grad->strides[k];
        }
        hmr_tensor_desc i = *out;
        i.offset = 0;
        for (int k = 0; k < nd; ++k) i.strides[k] = k == dim ? 1 : 0;
        hmr::Operands<3> ops{{*out, g, i}};
        return hmr::launch_on(ctx, "unfold_backward", [&](hipStream_t s) {
            return hmr::select(out->dtype, hmr::FloatDtypes{}, [&](auto t) {
                constexpr int DT = decltype(t)::v;
                hipLaunchKernelGGL((hmr::unfold_backward_kernel<DT>), hmr::blocks_for(n),
                                   dim3(hmr::kThreads), 0, s, ops, n, size, step,
                                   grad->sizes[dim], grad->strides[dim], grad->strides[nd]);
                return true;
            });
        });
    });
}

void hmr_test_wide_pool(int32_t on) { hmr::wide_pool = on != 0; }

hmr_status hmr_test_pool_candidates(const int64_t *geometry, int64_t height, int64_t width,
                                    int64_t out_h, int64_t out_w, int64_t *count) {
    return guarded([&] {
        hmr::Geometry g;
        if (hmr_status s = hmr::read_geometry(geometry, g)) return s;
        int64_t in[2] = {height, width}, out[2] = {out_h, out_w}, sums[2] = {0, 0};
        if (height < 0 || width < 0 || out_h < 0 || out_w < 0 ||
            !hmr::fits_in(INT64_MAX / 2, g, in, out)) {
            return hmr::fail(HMR_INVALID, "invalid pool candidates");
        }
        // The positions of the two dimensions pair up, so the count is the product of their sums.
        for (int d = 0; d < 2; ++d) {
            hmr::Axis<int64_t> a = hmr::axis_of<int64_t>(g, d);
            hmr::Walk<int64_t> w;
            for (int64_t i = 0; i < in[d]; ++i) {
                for (bool more = hmr::walk_first(a, i, out[d], w); more;
                     more = hmr::walk_next(a, i, out[d], w)) {
                    ++sums[d];
                }
            }
        }
        if (!hmr::checked_mul(sums[0], sums[1], *count)) {
            return hmr::fail(HMR_INVALID, "too many pool candidates");
        }
        return HMR_OK;
    });
}

}  // extern "C"
