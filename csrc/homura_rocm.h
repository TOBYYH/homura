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
 * hmr_test_live_streams(): the number of streams the calling thread created and hasn't destroyed.
 * hmr_test_current_device(): the calling thread's current HIP device (hipGetDevice).
 * hmr_test_set_current_device(): hipSetDevice for the calling thread.
 * hmr_test_fail(): fails with HMR_INVALID and `message`, to test the thread-local messages.
 */
void hmr_test_fail_stream_create(int32_t n);
void hmr_test_fail_after_probe(int32_t n);
void hmr_test_fail_properties(int32_t n);
int32_t hmr_test_live_streams(void);
hmr_status hmr_test_current_device(int32_t *device);
hmr_status hmr_test_set_current_device(int32_t device);
hmr_status hmr_test_fail(const char *message);

#ifdef __cplusplus
}
#endif

#endif
