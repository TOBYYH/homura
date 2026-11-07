// Copies, dtype conversions and fills of strided tensors. The conversions are those of elem.hpp,
// which mirror homura's Element::cast.
#include "elem.hpp"
#include "kernel.hpp"

using hmr::guarded;

namespace hmr {
namespace {

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
