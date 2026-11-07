/*
 * The C ABI between homura's Rust code and its ROCm layer (libhomura_rocm.a).
 *
 * Plain C, depending only on <stdint.h> and <stddef.h>, so that bindgen generates the Rust side
 * (src/backend/rocm/ffi.rs, regenerated and compared by tools/gen_rocm_ffi.py). Conventions:
 *
 * - Every entry point returns an hmr_status. On failure, hmr_last_error() returns a message for
 *   the calling thread, valid until that thread's next call into this library.
 * - No C++ exception crosses this boundary.
 * - Entry points that use a device switch the calling thread's current HIP device to it and
 *   restore the previous one before returning, so callers (including user code calling HIP
 *   directly) never see it change.
 * - Enumerations are passed as int32_t with the values below, and hmr_abi_check() reports the
 *   layout of every struct so that the Rust side can check it against its own.
 */
#ifndef HOMURA_ROCM_H
#define HOMURA_ROCM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. */
#define HMR_OK 0
/* A device (or pinned host) allocation failed for lack of memory. */
#define HMR_OUT_OF_MEMORY 1
/* An argument is invalid, such as a device index out of range or an unsupported GPU. */
#define HMR_INVALID 2
/* The HIP runtime returned an error. */
#define HMR_HIP_ERROR 3
/* An error inside this library, such as an unexpected C++ exception. */
#define HMR_INTERNAL_ERROR 4

typedef int32_t hmr_status;

/* The message of the calling thread's last failure, or "" if there was none. */
const char *hmr_last_error(void);

/* Length of the character arrays in hmr_device_properties, including the terminating NUL. */
#define HMR_NAME_LEN 256
#define HMR_ARCH_LEN 64

typedef struct hmr_device_properties {
    /* Marketing name, such as "AMD Radeon AI PRO R9700". */
    char name[HMR_NAME_LEN];
    /* GPU architecture without target features, such as "gfx1201". */
    char arch[HMR_ARCH_LEN];
    /* hipDeviceProp_t::multiProcessorCount: workgroup processors (two compute units each) on
     * RDNA GPUs. */
    int32_t multiprocessor_count;
    int32_t warp_size;
    uint64_t total_memory;
} hmr_device_properties;

/* Number of visible ROCm devices; 0 (and HMR_OK) when there are none or no driver. */
hmr_status hmr_device_count(int32_t *count);

hmr_status hmr_device_properties_get(int32_t device, hmr_device_properties *out);

/* hipMemGetInfo on `device`. */
hmr_status hmr_mem_get_info(int32_t device, uint64_t *free_bytes, uint64_t *total_bytes);

/* A device's execution context: its stream. */
typedef struct hmr_context *hmr_context_t;

/*
 * Creates the context of `device`: checks that the GPU's architecture is one this library was
 * built for (HMR_INVALID otherwise), creates the stream and runs a probe kernel on it. On failure
 * nothing is left allocated.
 */
hmr_status hmr_context_create(int32_t device, hmr_context_t *out);

hmr_status hmr_context_destroy(hmr_context_t ctx);

/* Waits until all work queued on the context's stream has finished. */
hmr_status hmr_context_synchronize(hmr_context_t ctx);

/*
 * Device memory, which the Rust side's caching allocator (src/backend/caching.rs) requests in
 * segments and hands out in blocks.
 *
 * hmr_malloc: hipMalloc of `nbytes` on `device`; HMR_OUT_OF_MEMORY when the device has no
 * memory left (the failure doesn't linger as HIP's last error).
 * hmr_free: hipFree of a pointer from hmr_malloc.
 */
hmr_status hmr_malloc(int32_t device, uint64_t nbytes, void **out);
hmr_status hmr_free(int32_t device, void *ptr);

/* Queues the zeroing of `nbytes` at `ptr` on the context's stream (hipMemsetAsync). */
hmr_status hmr_memset_zero(hmr_context_t ctx, void *ptr, uint64_t nbytes);

/*
 * Copies `nbytes` between pageable host memory and the device, in the context's stream order,
 * and returns when the copy is complete.
 */
hmr_status hmr_copy_to_host(hmr_context_t ctx, void *dst, const void *src, uint64_t nbytes);
hmr_status hmr_copy_from_host(hmr_context_t ctx, void *dst, const void *src, uint64_t nbytes);

/* Element types, numbered as homura's DType. Bool elements are bytes holding 0 or 1, F16 and
 * BF16 elements are their 16-bit patterns. */
#define HMR_BOOL 0
#define HMR_U8 1
#define HMR_I8 2
#define HMR_I16 3
#define HMR_I32 4
#define HMR_I64 5
#define HMR_F16 6
#define HMR_BF16 7
#define HMR_F32 8
#define HMR_F64 9

/* The most dimensions a tensor description has (after the Rust side merges what it can). */
#define HMR_MAX_DIMS 25

/*
 * A strided tensor in device memory: the element at index (i_0, ..., i_{ndim-1}) is element
 * offset + sum_k i_k * strides[k] of the array of `dtype` elements at `data`. Only the first
 * `ndim` sizes and strides are used; ndim is 0 for a single element. Passed by value to kernels.
 */
typedef struct hmr_tensor_desc {
    void *data;
    int64_t offset;
    int32_t dtype;
    int32_t ndim;
    int64_t sizes[HMR_MAX_DIMS];
    int64_t strides[HMR_MAX_DIMS];
} hmr_tensor_desc;

/*
 * Queues dst[i] = src[i] for every index of dst, converting src's elements to dst's dtype as
 * homura's Element::cast does (the same bytes when the dtypes are equal). Both have the same
 * ndim and sizes; src's strides may be 0 (broadcasting). dst's elements are distinct and don't
 * partially overlap src's.
 */
hmr_status hmr_copy(hmr_context_t ctx, const hmr_tensor_desc *dst, const hmr_tensor_desc *src);

/* Queues setting every element of dst to the value whose bytes are the low bytes of `bits`
 * (little endian, as many as the element's size). */
hmr_status hmr_fill(hmr_context_t ctx, const hmr_tensor_desc *dst, uint64_t bits);

/* Queues copying `nbytes` from `src` to `dst`, both device memory of the context's device that
 * doesn't overlap (hipMemcpyAsync). */
hmr_status hmr_copy_device(hmr_context_t ctx, void *dst, const void *src, uint64_t nbytes);

/* The architectures this library was built for, separated by ';' (HOMURA_OFFLOAD_ARCH). */
const char *hmr_built_archs(void);

/* Layout of the structs and values of the enumerations, as compiled into the library. */
typedef struct hmr_abi_info {
    uint32_t abi_info_size;
    uint32_t properties_size;
    uint32_t properties_align;
    uint32_t properties_offsets[5];
    int32_t status_values[5];
    int32_t name_len;
    int32_t arch_len;
    uint32_t desc_size;
    uint32_t desc_align;
    uint32_t desc_offsets[6];
    int32_t dtype_values[10];
    int32_t max_dims;
} hmr_abi_info;

void hmr_abi_check(hmr_abi_info *out);

/*
 * Testing hooks, not part of the supported interface.
 *
 * hmr_test_fail_stream_create(n): the calling thread's next n stream creations fail with
 * HMR_OUT_OF_MEMORY, to test that a failed context creation is not cached and can be retried.
 * hmr_test_fail_after_probe(n): the calling thread's next n context creations fail with a host
 * allocation failure (std::bad_alloc) after the stream and the probe kernel succeeded, to test
 * that a late failure destroys the stream.
 * hmr_test_fail_properties(n): the calling thread's next n hmr_device_properties_get calls fail
 * with HMR_HIP_ERROR, to test that a context created before such a failure is destroyed.
 * hmr_test_fail_data(n, out_of_memory): the calling thread's next n data operations (hmr_copy,
 * hmr_fill, hmr_copy_device, hmr_copy_to_host and hmr_copy_from_host) fail before queueing any
 * work, as their HIP call would: with hipErrorOutOfMemory if out_of_memory is nonzero, else with
 * hipErrorInvalidHandle (review round 124, R148 and R149).
 * hmr_test_live_streams(): the number of streams the calling thread created and hasn't destroyed.
 * hmr_test_current_device(): the calling thread's current HIP device (hipGetDevice).
 * hmr_test_set_current_device(): hipSetDevice for the calling thread.
 * hmr_test_fail(): fails with HMR_INVALID and `message`, to test the thread-local messages.
 */
void hmr_test_fail_stream_create(int32_t n);
void hmr_test_fail_after_probe(int32_t n);
void hmr_test_fail_properties(int32_t n);
void hmr_test_fail_data(int32_t n, int32_t out_of_memory);
int32_t hmr_test_live_streams(void);
hmr_status hmr_test_current_device(int32_t *device);
hmr_status hmr_test_set_current_device(int32_t device);
hmr_status hmr_test_fail(const char *message);

#ifdef __cplusplus
}
#endif

#endif
