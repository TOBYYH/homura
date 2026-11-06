// A host-only stand-in for the part of HIP that runtime.hip uses, for tools/check_rocm_host_failures.py:
// it records the streams that exist and the current device, and fails the calls a test asks it to.
// Not part of the library (build.rs compiles csrc/*.hip only, against the real HIP).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define __global__
#define hipLaunchKernelGGL(kernel, grid, block, shared, stream, ...) kernel(__VA_ARGS__)

struct dim3 {
    explicit dim3(unsigned) {}
};
inline struct {
    unsigned x = 0;
} threadIdx;

enum hipError_t {
    hipSuccess = 0,
    hipErrorOutOfMemory,
    hipErrorNoDevice,
    hipErrorInsufficientDriver,
    hipErrorInvalidHandle,
};
using hipStream_t = void *;
constexpr int hipStreamNonBlocking = 1;
constexpr int hipMemcpyDeviceToHost = 2;

struct hipDeviceProp_t {
    char name[256];
    char gcnArchName[256];
    int multiProcessorCount;
    int warpSize;
    size_t totalGlobalMem;
};

namespace fake {
// One device (0); the thread starts on device 1 so that a test sees whether it gets restored.
inline int current_device = 1;
inline int live_streams = 0;            // streams created and not destroyed
inline int destroyed_on = -1;           // the current device of the last hipStreamDestroy
inline hipError_t launch_error = hipSuccess;   // returned by hipGetLastError after a launch
inline hipError_t destroy_error = hipSuccess;  // returned by hipStreamDestroy (stream kept)
inline bool deny_new_after_destroy = false;    // set fake::deny_new from hipStreamDestroy on
inline bool deny_new_after_launch = false;     // set fake::deny_new from hipGetLastError on
inline bool deny_new = false;                  // operator new throws (host_failures.cpp)
inline intptr_t next_stream = 1;
}  // namespace fake

inline const char *hipGetErrorString(hipError_t e) {
    switch (e) {
        case hipSuccess: return "no error";
        case hipErrorOutOfMemory: return "out of memory";
        case hipErrorInvalidHandle: return "invalid resource handle";
        default: return "fake HIP error";
    }
}
inline hipError_t hipGetDeviceCount(int *n) {
    *n = 1;
    return hipSuccess;
}
inline hipError_t hipGetDevice(int *d) {
    *d = fake::current_device;
    return hipSuccess;
}
inline hipError_t hipSetDevice(int d) {
    if (d != 0 && d != 1) return hipErrorNoDevice;
    fake::current_device = d;
    return hipSuccess;
}
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t *p, int) {
    std::memset(p, 0, sizeof *p);
    std::strcpy(p->name, "fake GPU");
    std::strcpy(p->gcnArchName, "gfx1201");
    p->multiProcessorCount = 32;
    p->warpSize = 32;
    p->totalGlobalMem = 1 << 30;
    return hipSuccess;
}
inline hipError_t hipMemGetInfo(size_t *f, size_t *t) {
    *f = *t = 1 << 30;
    return hipSuccess;
}
inline hipError_t hipStreamCreateWithFlags(hipStream_t *s, unsigned) {
    *s = reinterpret_cast<hipStream_t>(fake::next_stream++);
    ++fake::live_streams;
    return hipSuccess;
}
inline hipError_t hipStreamDestroy(hipStream_t) {
    fake::destroyed_on = fake::current_device;
    if (fake::deny_new_after_destroy) fake::deny_new = true;
    if (fake::destroy_error != hipSuccess) return fake::destroy_error;
    --fake::live_streams;
    return hipSuccess;
}
inline hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
inline hipError_t hipGetLastError() {
    if (fake::deny_new_after_launch) fake::deny_new = true;
    return fake::launch_error;
}
template <class T>
hipError_t hipMalloc(T **p, size_t n) {
    *p = static_cast<T *>(std::malloc(n));
    return *p ? hipSuccess : hipErrorOutOfMemory;
}
inline hipError_t hipFree(void *p) {
    std::free(p);
    return hipSuccess;
}
inline hipError_t hipMemcpyAsync(void *dst, const void *src, size_t n, int, hipStream_t) {
    std::memcpy(dst, src, n);
    return hipSuccess;
}
