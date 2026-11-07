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

/*
 * Elementwise kernels. They compute what the CPU reference backend computes
 * (src/backend/cpu/mod.rs): floating point math in f32 for F16, BF16 and F32 and in f64 for F64,
 * rounding to the dtype once; wrapping integer arithmetic. All operands have the same ndim and
 * sizes; inputs may have stride 0 (broadcasting); the output's elements are distinct and don't
 * partially overlap an input's (an input may be the output itself). Each entry checks its
 * descriptions and returns HMR_INVALID for an op the dtypes don't support.
 */

/* Unary ops, numbered as homura's UnaryOp. */
#define HMR_UNARY_NEG 0
#define HMR_UNARY_ABS 1
#define HMR_UNARY_SIGN 2
#define HMR_UNARY_RELU 3
#define HMR_UNARY_EXP 4
#define HMR_UNARY_LOG 5
#define HMR_UNARY_SQRT 6
#define HMR_UNARY_RSQRT 7
#define HMR_UNARY_RECIPROCAL 8
#define HMR_UNARY_SIN 9
#define HMR_UNARY_COS 10
#define HMR_UNARY_ERF 11
#define HMR_UNARY_TANH 12
#define HMR_UNARY_SIGMOID 13
#define HMR_UNARY_LOG_SIGMOID 14
#define HMR_UNARY_SILU 15
#define HMR_UNARY_GELU 16
#define HMR_UNARY_GELU_TANH 17
#define HMR_UNARY_LOGICAL_NOT 18

/* Binary ops, numbered as homura's BinaryOp. */
#define HMR_BINARY_ADD 0
#define HMR_BINARY_SUB 1
#define HMR_BINARY_MUL 2
#define HMR_BINARY_DIV 3
#define HMR_BINARY_POW 4
#define HMR_BINARY_MAXIMUM 5
#define HMR_BINARY_MINIMUM 6
#define HMR_BINARY_GREATER 7
#define HMR_BINARY_GREATER_EQUAL 8
#define HMR_BINARY_LESS 9
#define HMR_BINARY_LESS_EQUAL 10
#define HMR_BINARY_EQUAL 11
#define HMR_BINARY_NOT_EQUAL 12
#define HMR_BINARY_LOGICAL_AND 13
#define HMR_BINARY_LOGICAL_OR 14

/* The optimizers' fused steps (homura's Fused). */
#define HMR_FUSED_SCALED_ADD 0
#define HMR_FUSED_LERP 1
#define HMR_FUSED_ADDCMUL 2
#define HMR_FUSED_ADDCDIV 3

/* out = op(src). LogicalNot writes Bool; the other ops write src's dtype. */
hmr_status hmr_unary(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                     const hmr_tensor_desc *src);

/* out = op(a, b) for a and b of one dtype. Comparisons and logical ops write Bool. */
hmr_status hmr_binary(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                      const hmr_tensor_desc *a, const hmr_tensor_desc *b);

/* out = a * s, a / s or (scalar_first) s / a for a floating point `a`, with `s` rounded to the
 * computation type (f32 for F16, BF16 and F32). Only HMR_BINARY_MUL and HMR_BINARY_DIV. */
hmr_status hmr_binary_scalar(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                             const hmr_tensor_desc *a, double s, int32_t scalar_first);

/*
 * The fused step `op` for floating point operands of one dtype; `c` is NULL for ScaledAdd and
 * Lerp. `coefficient` is already in the computation type: alpha (rounded to the dtype first),
 * the Lerp coefficient (weight, or weight - 1 when `small` is 0) or value.
 *   ScaledAdd: fma(b, coefficient, a);  Lerp: fma(coefficient, b - a, small ? a : b);
 *   Addcmul: fma(coefficient * b, c, a);  Addcdiv: a + coefficient * b / c.
 */
hmr_status hmr_fused(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                     const hmr_tensor_desc *a, const hmr_tensor_desc *b,
                     const hmr_tensor_desc *c, double coefficient, int32_t small);

/* out = cond ? x : y; cond is Bool, x and y have out's dtype. */
hmr_status hmr_where(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *cond,
                     const hmr_tensor_desc *x, const hmr_tensor_desc *y);

/* out = src where the column minus the row index (the last two dimensions, ndim >= 2) is at
 * least (upper) or at most `diagonal`, else 0. */
hmr_status hmr_triangle(hmr_context_t ctx, const hmr_tensor_desc *out,
                        const hmr_tensor_desc *src, int64_t diagonal, int32_t upper);

/* out = src for one dtype, where src's strides may be negative (flipped dimensions). */
hmr_status hmr_flip(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *src);

/*
 * Random sampling with the Philox4x32-10 counter of src/ops/random.rs: block `n` (counter
 * [n, n >> 32, 0, 0]) under the key [seed, seed >> 32]; the caller reserves the blocks of a call
 * (from `offset` on) as the CPU does and advances the generator. `out` is flat and contiguous.
 *
 * hmr_random_uniform: from + u * (to - from) in f64 for a float `out`, u from 53 bits (two words
 * per block) for F64 and from the dtype's significand bits (one word per element) otherwise,
 * rounded to the dtype; a result that rounds to `to` (with from < to) becomes `from`.
 * hmr_random_normal: Box-Muller in f64, two elements per block (cos, then sin), rounded to the
 * dtype.
 */
hmr_status hmr_random_uniform(hmr_context_t ctx, const hmr_tensor_desc *out, uint64_t seed,
                              uint64_t offset, double from, double to);
hmr_status hmr_random_normal(hmr_context_t ctx, const hmr_tensor_desc *out, uint64_t seed,
                             uint64_t offset, double mean, double std);

/* The Philox blocks of one tile of hmr_random_int. */
#define HMR_RANDOM_INT_TILE 256

/*
 * One batch of the CPU's integer rejection sampling (sample_int in src/ops/random.rs): the 64-bit
 * words of `blocks` blocks from `offset` on, two per block in the order [w0 w1] then [w2 w3];
 * the words >= `threshold`, in that order, become low + word % range (wrapping) and fill the I64
 * `out` until it is full, the rest are dropped. With threshold 0 every word is taken and
 * `workspace` may be NULL. Otherwise `workspace` holds 2 * tiles + 1 words of device memory,
 * tiles = ceil(blocks / HMR_RANDOM_INT_TILE), and the batch's number of words >= threshold
 * (which may exceed out's elements) is left in its last word.
 */
hmr_status hmr_random_int(hmr_context_t ctx, const hmr_tensor_desc *out, uint64_t seed,
                          uint64_t offset, uint64_t blocks, int64_t low, uint64_t range,
                          uint64_t threshold, void *workspace);

/*
 * Reductions (csrc/reduce.hip, the plans in csrc/reduce.hpp). A reduction is described by:
 * - `kept`: the kept dimensions with the source's strides, data, offset and dtype;
 * - `red`: the reduced dimensions with the source's strides (and dtype);
 * - the outputs: contiguous with the kept sizes.
 * The reduced elements of an output are folded in an order that depends only on these sizes
 * and strides. Some plans need device memory: hmr_reduce_workspace gives its size in bytes for
 * hmr_reduce and hmr_arg_reduce (var = 0) or hmr_var (var = 1), and `workspace` must hold at
 * least that many bytes (it may be NULL when it is 0).
 */
#define HMR_REDUCE_SUM 0
#define HMR_REDUCE_PROD 1
#define HMR_REDUCE_MEAN 2
#define HMR_REDUCE_MAX 3
#define HMR_REDUCE_MIN 4
#define HMR_REDUCE_ALL 5
#define HMR_REDUCE_ANY 6

hmr_status hmr_reduce_workspace(const hmr_tensor_desc *kept, const hmr_tensor_desc *red,
                                int32_t var, int64_t *bytes);

/* cpu::reduce: floats sum, multiply and average in f64; integers and bools sum and multiply
 * into I64 (wrapping); max and min keep the dtype (the first NaN wins, then the first of equal
 * values); all and any give BOOL (U8 for U8). Max and min need a nonempty reduction. */
hmr_status hmr_reduce(hmr_context_t ctx, int32_t op, const hmr_tensor_desc *out,
                      const hmr_tensor_desc *kept, const hmr_tensor_desc *red, void *workspace,
                      int64_t workspace_bytes);

/* cpu::var: the variance of float elements in f64 with two passes, divided by
 * max(0, n - correction) (NaN for n = 0), and its square root with take_sqrt. */
hmr_status hmr_var(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *kept,
                   const hmr_tensor_desc *red, double correction, int32_t take_sqrt,
                   void *workspace, int64_t workspace_bytes);

/* cpu::arg_reduce: the index (I64) along the one reduced dimension of the maximum (or minimum)
 * and, if `values` is not NULL, the value; the first NaN wins, then the first of equal values. */
hmr_status hmr_arg_reduce(hmr_context_t ctx, const hmr_tensor_desc *values,
                          const hmr_tensor_desc *indices, const hmr_tensor_desc *kept,
                          const hmr_tensor_desc *red, int32_t is_max, void *workspace,
                          int64_t workspace_bytes);

/* cpu::softmax along rows of `n` elements: `out_rows` and `src_rows` describe the other
 * dimensions (same sizes), the row elements are `out_stride` / `src_stride` apart. Few long rows
 * need a workspace of hmr_softmax_workspace bytes (NULL when it is 0). */
hmr_status hmr_softmax_workspace(const hmr_tensor_desc *src_rows, int64_t n, int64_t src_stride,
                                 int64_t *bytes);
hmr_status hmr_softmax(hmr_context_t ctx, const hmr_tensor_desc *out_rows,
                       const hmr_tensor_desc *src_rows, int64_t n, int64_t out_stride,
                       int64_t src_stride, int32_t log, void *workspace,
                       int64_t workspace_bytes);

/* cpu::exclusive_cumprod along lines of `n` elements, described as for hmr_softmax. */
hmr_status hmr_exclusive_cumprod(hmr_context_t ctx, const hmr_tensor_desc *out_lines,
                                 const hmr_tensor_desc *src_lines, int64_t n, int64_t out_stride,
                                 int64_t src_stride, int32_t reverse);

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
 * hmr_fill, hmr_copy_device, hmr_copy_to_host, hmr_copy_from_host, and the kernels, samplers
 * and reductions above) fail before queueing any work, as their HIP call would: with
 * hipErrorOutOfMemory if out_of_memory is nonzero, else with hipErrorInvalidHandle (review round
 * 124, R148 and R149).
 * hmr_test_reduce_plan(): the plan hmr_reduce chooses for a reduction (csrc/reduce.hpp): the
 * lanes per output, the number of chunks of the reduced range and their size, to test that the
 * tests reach every path. hmr_test_softmax_plan(): the same for hmr_softmax: whether it splits
 * its rows, the lanes per row and the elements a lane keeps in registers (0: none).
 * hmr_test_live_streams(): the number of streams the calling thread created and hasn't destroyed.
 * hmr_test_current_device(): the calling thread's current HIP device (hipGetDevice).
 * hmr_test_set_current_device(): hipSetDevice for the calling thread.
 * hmr_test_fail(): fails with HMR_INVALID and `message`, to test the thread-local messages.
 */
void hmr_test_fail_stream_create(int32_t n);
void hmr_test_fail_after_probe(int32_t n);
void hmr_test_fail_properties(int32_t n);
void hmr_test_fail_data(int32_t n, int32_t out_of_memory);
hmr_status hmr_test_reduce_plan(const hmr_tensor_desc *kept, const hmr_tensor_desc *red,
                                int64_t *group, int64_t *splits, int64_t *chunk);
hmr_status hmr_test_softmax_plan(const hmr_tensor_desc *src_rows, int64_t n, int64_t src_stride,
                                 int32_t *split, int32_t *group, int32_t *cache);
int32_t hmr_test_live_streams(void);
hmr_status hmr_test_current_device(int32_t *device);
hmr_status hmr_test_set_current_device(int32_t device);
hmr_status hmr_test_fail(const char *message);

#ifdef __cplusplus
}
#endif

#endif
