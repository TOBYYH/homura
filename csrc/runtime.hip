// Device queries, contexts and the error state of the ROCm layer.
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <string>

#include "common.hpp"

#ifndef HMR_OFFLOAD_ARCHS
#error "build.rs defines HMR_OFFLOAD_ARCHS as the list of target architectures"
#endif

struct hmr_context {
    int device;
    hipStream_t stream;
};

namespace hmr {
namespace {

// The calling thread's last error: `static_error` if set (fail_static), else `last_error`.
thread_local std::string last_error;
thread_local const char *static_error = nullptr;

// Testing hooks, per thread so that a test injecting failures doesn't fail other threads'
// context creations: failures to inject, and the streams this thread created and hasn't
// destroyed.
thread_local int failing_stream_creations = 0;
thread_local int failing_after_probe = 0;
thread_local int failing_properties = 0;
thread_local int live_streams = 0;

// Destroys `stream`, neither allocating nor touching the last error. The count of live streams
// only goes down if HIP did destroy it (review round 119, R139).
hipError_t destroy_stream(hipStream_t stream) noexcept {
    hipError_t e = hipStreamDestroy(stream);
    if (e == hipSuccess) --live_streams;
    return e;
}

// Owns a stream until `release`, and destroys it otherwise: on every failure path of the context
// creation, exceptions included. Declared after the DeviceGuard, so that the stream is destroyed
// while its device is still current.
class OwnedStream {
  public:
    OwnedStream() = default;
    OwnedStream(const OwnedStream &) = delete;
    OwnedStream &operator=(const OwnedStream &) = delete;
    // Runs on a failure path: the error of that failure is the one to report, so a failure to
    // destroy the stream is ignored (review round 119, R139).
    ~OwnedStream() {
        if (stream_) (void)destroy_stream(stream_);
    }

    hmr_status create() {
        hipStream_t stream = nullptr;
        HMR_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        stream_ = stream;
        ++live_streams;
        return HMR_OK;
    }

    hipStream_t get() const { return stream_; }

    hipStream_t release() {
        hipStream_t stream = stream_;
        stream_ = nullptr;
        return stream;
    }

  private:
    hipStream_t stream_ = nullptr;
};

// The architecture of a gcnArchName without its target features ("gfx90a:sramecc+:xnack-").
std::string base_arch(const char *gcn_arch_name) {
    std::string arch(gcn_arch_name);
    return arch.substr(0, arch.find(':'));
}

bool built_for(const std::string &arch) {
    std::string list = HMR_OFFLOAD_ARCHS;
    size_t start = 0;
    while (start <= list.size()) {
        size_t end = list.find(';', start);
        if (end == std::string::npos) end = list.size();
        if (list.compare(start, end - start, arch) == 0) return true;
        start = end + 1;
    }
    return false;
}

hmr_status check_device(int device) {
    int count = 0;
    HMR_HIP(hipGetDeviceCount(&count));
    if (device < 0 || device >= count) {
        return fail(HMR_INVALID, "rocm:" + std::to_string(device) + " is not available: " +
                                     std::to_string(count) + " ROCm device" +
                                     (count == 1 ? "" : "s") + " found");
    }
    return HMR_OK;
}

__global__ void probe_kernel(uint32_t *out) { *out = 0x686d7200u + threadIdx.x + 1; }

// Runs a one-thread kernel on `stream` and checks what it wrote, to fail early (at context
// creation) if the code objects can't run on this GPU.
hmr_status probe(hipStream_t stream) {
    uint32_t *d = nullptr;
    HMR_HIP(hipMalloc(&d, sizeof(uint32_t)));
    uint32_t value = 0;
    hipError_t e = hipSuccess;
    hipLaunchKernelGGL(probe_kernel, dim3(1), dim3(1), 0, stream, d);
    e = hipGetLastError();
    if (e == hipSuccess) e = hipMemcpyAsync(&value, d, sizeof value, hipMemcpyDeviceToHost, stream);
    if (e == hipSuccess) e = hipStreamSynchronize(stream);
    hipError_t freed = hipFree(d);
    if (e != hipSuccess) return hip_fail(e, "running the probe kernel");
    if (freed != hipSuccess) return hip_fail(freed, "hipFree(probe buffer)");
    if (value != 0x686d7201u) {
        return fail(HMR_INTERNAL_ERROR, "the probe kernel wrote " + std::to_string(value));
    }
    return HMR_OK;
}

}  // namespace

hmr_status fail(hmr_status status, std::string message) {
    last_error = std::move(message);
    static_error = nullptr;
    return status;
}

hmr_status fail_static(hmr_status status, const char *message) noexcept {
    static_error = message;
    return status;
}

const char *last_error_text() noexcept { return static_error ? static_error : last_error.c_str(); }

hmr_status hip_status(hipError_t error) {
    return error == hipErrorOutOfMemory ? HMR_OUT_OF_MEMORY : HMR_HIP_ERROR;
}

hmr_status hip_fail(hipError_t error, const char *call) noexcept {
    try {
        return fail(hip_status(error), std::string(call) + ": " + hipGetErrorString(error));
    } catch (...) {
        return fail_static(hip_status(error), hipGetErrorString(error));
    }
}

DeviceGuard::~DeviceGuard() {
    if (switched_) {
        // Nothing to report from a destructor: hipSetDevice to a device that was current a moment
        // ago doesn't fail.
        (void)hipSetDevice(previous_);
    }
}

hmr_status DeviceGuard::enter(int device) {
    HMR_HIP(hipGetDevice(&previous_));
    if (previous_ != device) {
        HMR_HIP(hipSetDevice(device));
        switched_ = true;
    }
    return HMR_OK;
}

}  // namespace hmr

using hmr::fail;
using hmr::guarded;

extern "C" {

const char *hmr_last_error(void) { return hmr::last_error_text(); }

hmr_status hmr_device_count(int32_t *count) {
    return guarded([&] {
        int n = 0;
        hipError_t e = hipGetDeviceCount(&n);
        if (e == hipErrorNoDevice || e == hipErrorInsufficientDriver) {
            n = 0;
        } else {
            HMR_HIP(e);
        }
        *count = n;
        return HMR_OK;
    });
}

hmr_status hmr_device_properties_get(int32_t device, hmr_device_properties *out) {
    return guarded([&] {
        if (hmr_status s = hmr::check_device(device)) return s;
        if (hmr::failing_properties > 0) {
            --hmr::failing_properties;
            return fail(HMR_HIP_ERROR,
                        "hipGetDeviceProperties(&p, device): invalid resource handle (injected for "
                        "testing)");
        }
        hipDeviceProp_t p;
        HMR_HIP(hipGetDeviceProperties(&p, device));
        std::memset(out, 0, sizeof *out);
        std::strncpy(out->name, p.name, HMR_NAME_LEN - 1);
        std::string arch = hmr::base_arch(p.gcnArchName);
        std::strncpy(out->arch, arch.c_str(), HMR_ARCH_LEN - 1);
        out->multiprocessor_count = p.multiProcessorCount;
        out->warp_size = p.warpSize;
        out->total_memory = p.totalGlobalMem;
        return HMR_OK;
    });
}

hmr_status hmr_mem_get_info(int32_t device, uint64_t *free_bytes, uint64_t *total_bytes) {
    return guarded([&] {
        if (hmr_status s = hmr::check_device(device)) return s;
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(device)) return s;
        size_t f = 0, t = 0;
        HMR_HIP(hipMemGetInfo(&f, &t));
        *free_bytes = f;
        *total_bytes = t;
        return HMR_OK;
    });
}

hmr_status hmr_context_create(int32_t device, hmr_context_t *out) {
    return guarded([&] {
        *out = nullptr;
        if (hmr_status s = hmr::check_device(device)) return s;
        hipDeviceProp_t p;
        HMR_HIP(hipGetDeviceProperties(&p, device));
        std::string arch = hmr::base_arch(p.gcnArchName);
        if (!hmr::built_for(arch)) {
            return fail(HMR_INVALID, "rocm:" + std::to_string(device) + " is a " + arch +
                                         " GPU, but homura was built for " HMR_OFFLOAD_ARCHS
                                         "; rebuild it with HOMURA_OFFLOAD_ARCH set to include " +
                                         arch + " (if homura supports it)");
        }
        if (p.warpSize != 32) {
            return fail(HMR_INVALID, "rocm:" + std::to_string(device) + " has a wave size of " +
                                         std::to_string(p.warpSize) + "; homura needs wave32");
        }
        // The context is allocated before any device resource, which `stream` then owns until
        // the end: a failure (or exception) at any point leaves nothing behind.
        auto ctx = std::make_unique<hmr_context>();
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(device)) return s;
        hmr::OwnedStream stream;
        if (hmr::failing_stream_creations > 0) {
            --hmr::failing_stream_creations;
            return fail(HMR_OUT_OF_MEMORY,
                        "hipStreamCreateWithFlags(&stream, hipStreamNonBlocking): out of memory "
                        "(injected for testing)");
        }
        if (hmr_status s = stream.create()) return s;
        if (hmr_status s = hmr::probe(stream.get())) {
            // Without memory for the longer message, the probe's own stands, with its status
            // (review round 120, R143).
            try {
                return fail(s, "rocm:" + std::to_string(device) + ": " + hmr::last_error_text());
            } catch (const std::bad_alloc &) {
                return s;
            }
        }
        if (hmr::failing_after_probe > 0) {
            --hmr::failing_after_probe;
            throw std::bad_alloc();
        }
        ctx->device = device;
        ctx->stream = stream.release();
        *out = ctx.release();
        return HMR_OK;
    });
}

hmr_status hmr_context_destroy(hmr_context_t ctx) {
    return guarded([&] {
        if (!ctx) return HMR_OK;
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(ctx->device)) return s;
        std::unique_ptr<hmr_context> owned(ctx);
        hipError_t e = hmr::destroy_stream(owned->stream);
        if (e != hipSuccess) return hmr::hip_fail(e, "hipStreamDestroy(ctx->stream)");
        return HMR_OK;
    });
}

hmr_status hmr_context_synchronize(hmr_context_t ctx) {
    return guarded([&] {
        hmr::DeviceGuard guard;
        if (hmr_status s = guard.enter(ctx->device)) return s;
        HMR_HIP(hipStreamSynchronize(ctx->stream));
        return HMR_OK;
    });
}

const char *hmr_built_archs(void) { return HMR_OFFLOAD_ARCHS; }

void hmr_abi_check(hmr_abi_info *out) {
    std::memset(out, 0, sizeof *out);
    out->abi_info_size = sizeof(hmr_abi_info);
    out->properties_size = sizeof(hmr_device_properties);
    out->properties_align = alignof(hmr_device_properties);
    out->properties_offsets[0] = offsetof(hmr_device_properties, name);
    out->properties_offsets[1] = offsetof(hmr_device_properties, arch);
    out->properties_offsets[2] = offsetof(hmr_device_properties, multiprocessor_count);
    out->properties_offsets[3] = offsetof(hmr_device_properties, warp_size);
    out->properties_offsets[4] = offsetof(hmr_device_properties, total_memory);
    const int32_t statuses[5] = {HMR_OK, HMR_OUT_OF_MEMORY, HMR_INVALID, HMR_HIP_ERROR,
                                 HMR_INTERNAL_ERROR};
    std::memcpy(out->status_values, statuses, sizeof statuses);
    out->name_len = HMR_NAME_LEN;
    out->arch_len = HMR_ARCH_LEN;
}

void hmr_test_fail_stream_create(int32_t n) { hmr::failing_stream_creations = n; }

void hmr_test_fail_after_probe(int32_t n) { hmr::failing_after_probe = n; }

void hmr_test_fail_properties(int32_t n) { hmr::failing_properties = n; }

int32_t hmr_test_live_streams(void) { return hmr::live_streams; }

hmr_status hmr_test_current_device(int32_t *device) {
    return guarded([&] {
        int d = -1;
        HMR_HIP(hipGetDevice(&d));
        *device = d;
        return HMR_OK;
    });
}

hmr_status hmr_test_set_current_device(int32_t device) {
    return guarded([&] {
        HMR_HIP(hipSetDevice(device));
        return HMR_OK;
    });
}

hmr_status hmr_test_fail(const char *message) {
    return guarded([&] { return fail(HMR_INVALID, message); });
}

}  // extern "C"
