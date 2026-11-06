// Internals shared by the ROCm layer's translation units: error reporting and the device guard.
#pragma once

#include <hip/hip_runtime.h>

#include <exception>
#include <new>
#include <string>

#include "homura_rocm.h"

namespace hmr {

// Records `message` as the calling thread's last error and returns `status`.
hmr_status fail(hmr_status status, std::string message);

// The same for a message with static storage, without allocating: for the paths that must not
// throw, such as the exception handlers of `guarded` (review round 119, R140).
hmr_status fail_static(hmr_status status, const char *message) noexcept;

// The calling thread's last error, as hmr_last_error returns it.
const char *last_error_text() noexcept;

// The status for a failed HIP call: out of memory or a generic HIP error.
hmr_status hip_status(hipError_t error);

// Fails with `error`, the result of the HIP call `call` (its text). Doesn't throw: if the message
// can't be stored, the status stays the HIP error's, with HIP's own (static) text for it
// (review round 120, R143).
hmr_status hip_fail(hipError_t error, const char *call) noexcept;

// Fails with the HIP error of `expr` (text of the call) unless it is hipSuccess.
#define HMR_HIP(expr)                                                                              \
    do {                                                                                           \
        hipError_t hmr_error_ = (expr);                                                            \
        if (hmr_error_ != hipSuccess) return ::hmr::hip_fail(hmr_error_, #expr);                   \
    } while (0)

// Runs `body` (returning hmr_status), turning C++ exceptions into status codes so that none
// crosses the C ABI. The handlers don't need host memory, which may be what ran out: a message
// that needs it falls back to a static one (review round 119, R140).
template <class F>
hmr_status guarded(F body) noexcept {
    try {
        return body();
    } catch (const std::bad_alloc &) {
        return fail_static(HMR_OUT_OF_MEMORY, "host allocation failed");
    } catch (const std::exception &e) {
        try {
            return fail(HMR_INTERNAL_ERROR, std::string("internal error: ") + e.what());
        } catch (...) {
            return fail_static(HMR_INTERNAL_ERROR,
                               "internal error (and its message couldn't be stored: host "
                               "allocation failed)");
        }
    } catch (...) {
        return fail_static(HMR_INTERNAL_ERROR, "internal error: unknown exception");
    }
}

// Makes `device` the calling thread's current HIP device for the guard's lifetime and then
// restores the previous one, so that entry points never change what the caller sees (the current
// device is per thread). Call `enter` and check its status before using the device.
class DeviceGuard {
  public:
    DeviceGuard() = default;
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
    ~DeviceGuard();

    hmr_status enter(int device);

  private:
    int previous_ = -1;
    bool switched_ = false;
};

}  // namespace hmr
