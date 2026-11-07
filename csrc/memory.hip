// Device memory and the copies between it and host memory.
#include <cstdint>

#include "common.hpp"

using hmr::guarded;

extern "C" {

hmr_status hmr_malloc(int32_t device, uint64_t nbytes, void **out) {
    return guarded([&] {
        *out = nullptr;
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(device)) return s;
        void *ptr = nullptr;
        // On failure, hip_fail also clears the thread's last HIP error.
        HMR_HIP(hipMalloc(&ptr, nbytes));
        *out = ptr;
        return HMR_OK;
    });
}

hmr_status hmr_free(int32_t device, void *ptr) {
    return guarded([&] {
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(device)) return s;
        HMR_HIP(hipFree(ptr));
        return HMR_OK;
    });
}

hmr_status hmr_memset_zero(hmr_context_t ctx, void *ptr, uint64_t nbytes) {
    return guarded([&] {
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(ctx->device)) return s;
        HMR_HIP(hipMemsetAsync(ptr, 0, nbytes, ctx->stream));
        return HMR_OK;
    });
}

hmr_status hmr_copy_to_host(hmr_context_t ctx, void *dst, const void *src, uint64_t nbytes) {
    return guarded([&] {
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(ctx->device)) return s;
        if (hmr_status s = hmr::injected_data_failure("hipMemcpyAsync(DeviceToHost)")) return s;
        HMR_HIP(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDeviceToHost, ctx->stream));
        HMR_HIP(hipStreamSynchronize(ctx->stream));
        return HMR_OK;
    });
}

hmr_status hmr_copy_from_host(hmr_context_t ctx, void *dst, const void *src, uint64_t nbytes) {
    return guarded([&] {
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(ctx->device)) return s;
        if (hmr_status s = hmr::injected_data_failure("hipMemcpyAsync(HostToDevice)")) return s;
        HMR_HIP(hipMemcpyAsync(dst, src, nbytes, hipMemcpyHostToDevice, ctx->stream));
        HMR_HIP(hipStreamSynchronize(ctx->stream));
        return HMR_OK;
    });
}

}  // extern "C"
