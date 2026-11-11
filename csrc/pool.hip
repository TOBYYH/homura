// Pooling kernels and the backward of unfold: max_pool2d, adaptive_avg_pool2d and its
// backward, unfold_backward (the backward of max_pool2d is a scatter, in index.hip). The
// counterparts of src/backend/cpu/pool.rs and cpu::unfold_backward, with the CPU's bits: each
// output element computes what the CPU computes for it, in the same order and type.
//
// The CPU's backward kernels add each gradient element into the positions it covers, one at a
// time in output order. Here each position of the result instead reads the gradient elements
// that cover it, in that same output order, starting from zero, so the sums (rounded after every
// addition, as on the CPU) are the same, every element is written, and nothing is atomic.
#include <limits>

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
        for (int d = 0; d < 2; ++d) {
            g.kernel[d] = geometry[d];
            g.stride[d] = geometry[2 + d];
            g.padding[d] = geometry[4 + d];
            g.dilation[d] = geometry[6 + d];
            if (g.kernel[d] < 1 || g.stride[d] < 1 || g.padding[d] < 0 || g.dilation[d] < 1) {
                return hmr::fail(HMR_INVALID, "invalid max_pool2d geometry");
            }
        }
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

}  // extern "C"
