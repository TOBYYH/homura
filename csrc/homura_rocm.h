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

/*
 * addr: out[i][j] = beta * input[i][j] + alpha * v1[i] * v2[j], with the steps of the CPU's
 * kernel (src/backend/cpu/linalg.rs, addr), for every dtype. All four descriptions have the
 * output's two dimensions [n, m]: v1 with stride 0 along the columns, v2 along the rows, and
 * the input broadcast; `input` is NULL when beta is zero (it isn't read).
 *   floats (coefficients rounded to the dtype, in `alpha` and `beta`): a = alpha * v1[i]
 *     (skipped for alpha == 1), p = a * v2[j], out = p or (beta == 1 ? c : beta * c) + p, each
 *     step in the computation type (f32 for F16, BF16 and F32, f64 for F64) rounded to the dtype;
 *   integers (`alpha_int`, `beta_int`): wrapping, alpha * (v1[i] * v2[j]) + beta * c;
 *   Bool (`alpha_int`, `beta_int` 0 or 1): (v1[i] and alpha and v2[j]) or c.
 */
hmr_status hmr_addr(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *v1,
                    const hmr_tensor_desc *v2, const hmr_tensor_desc *input, double alpha,
                    double beta, int64_t alpha_int, int64_t beta_int);

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

/*
 * Indexing (csrc/index.hip). Index tensors are I32 or I64; positions are checked before they
 * are used (hmr_index_check, in the same execution lock), so the kernels take them as valid.
 */

/* The first element of `index`, in row-major order of its logical positions, that is outside
 * [low, high) and, if has_exempt, not equal to `exempt`: queues writing its position to slot[0]
 * (UINT64_MAX if there is none) and its value to slot[1]. `slot` is two words of device memory. */
hmr_status hmr_index_check(hmr_context_t ctx, const hmr_tensor_desc *index, int64_t low,
                           int64_t high, int32_t has_exempt, int64_t exempt, uint64_t *slot);

/* cpu::index_select: out (src's sizes but `index`'s length along dim) gets the slices of src
 * along dim at the positions of the 1-D `index`, negative ones counted from the end if
 * wrap_negative. */
hmr_status hmr_index_select(hmr_context_t ctx, const hmr_tensor_desc *out,
                            const hmr_tensor_desc *src, const hmr_tensor_desc *index, int32_t dim,
                            int32_t wrap_negative);

/* cpu::gather: out (index's sizes) gets src at each index position with the coordinate along
 * dim replaced by the index element. */
hmr_status hmr_gather(hmr_context_t ctx, const hmr_tensor_desc *out, const hmr_tensor_desc *src,
                      const hmr_tensor_desc *index, int32_t dim);

/* One indexed dimension of advanced indexing: out (I64) = (if first) or += coord * stride, the
 * coordinate wrapped by `size` when negative; `coord` has out's sizes (broadcast). */
hmr_status hmr_index_positions(hmr_context_t ctx, const hmr_tensor_desc *out,
                               const hmr_tensor_desc *coord, int64_t size, int64_t stride,
                               int32_t first);

/* cpu::index_overwrite: slice i of src along dim is copied to slice positions[i] of out, the last
 * of repeated positions winning. `positions` is a contiguous 1-D I64 tensor; the workspace has
 * hmr_index_overwrite_workspace bytes for out's size along dim. */
hmr_status hmr_index_overwrite_workspace(int64_t size, int64_t *bytes);
hmr_status hmr_index_overwrite(hmr_context_t ctx, const hmr_tensor_desc *out,
                               const hmr_tensor_desc *src, const hmr_tensor_desc *positions,
                               int32_t dim, void *workspace, int64_t workspace_bytes);

/* cpu::index_add: slice i of src along dim is added into slice index[i] of out, each element of
 * out summed in f64 (floats; integers wrap in i64, bools are or-ed) in the order of i and rounded
 * once. The workspace has hmr_index_add_workspace bytes for `m` positions (`size`, out's size
 * along dim, bounds them). */
hmr_status hmr_index_add_workspace(int64_t m, int64_t size, int64_t *bytes);
hmr_status hmr_index_add(hmr_context_t ctx, const hmr_tensor_desc *out,
                         const hmr_tensor_desc *src, const hmr_tensor_desc *index, int32_t dim,
                         void *workspace, int64_t workspace_bytes);

/* cpu::scatter_add: src at each index position is added into out at that position with the
 * coordinate along dim replaced by the index element, summed as hmr_index_add sums, in row-major
 * order of the index positions. The workspace has hmr_scatter_add_workspace bytes for the index's
 * number of elements. */
hmr_status hmr_scatter_add_workspace(int64_t n, int64_t *bytes);
hmr_status hmr_scatter_add(hmr_context_t ctx, const hmr_tensor_desc *out,
                           const hmr_tensor_desc *src, const hmr_tensor_desc *index, int32_t dim,
                           void *workspace, int64_t workspace_bytes);

/* cpu::max_pool2d_backward for any indices inside the plane (hmr_max_pool2d_gather_backward is
 * the path for indices inside their windows): each element of grad is added into
 * the contiguous out at the plane position `indices` (I64, grad's sizes, checked to lie in the
 * plane) holds for it, in the dtype, one at a time in row-major order of grad. The workspace has
 * hmr_max_pool2d_backward_workspace bytes for grad's number of elements. */
hmr_status hmr_max_pool2d_backward_workspace(int64_t n, int64_t *bytes);
hmr_status hmr_max_pool2d_backward(hmr_context_t ctx, const hmr_tensor_desc *out,
                                   const hmr_tensor_desc *grad, const hmr_tensor_desc *indices,
                                   void *workspace, int64_t workspace_bytes);

/* cpu::mask_coordinates in two calls: hmr_mask_count queues counting the true elements of the
 * BOOL mask, whose number it leaves at word total_word of the workspace (hmr_mask_workspace bytes
 * for the mask's number of elements); hmr_mask_coordinates then writes coordinate d of the k-th
 * true element, in row-major order, to out[d][k] (I64, contiguous, mask's rank x the number). */
hmr_status hmr_mask_workspace(int64_t n, int64_t *bytes, int64_t *total_word);
hmr_status hmr_mask_count(hmr_context_t ctx, const hmr_tensor_desc *mask, void *workspace,
                          int64_t workspace_bytes);
hmr_status hmr_mask_coordinates(hmr_context_t ctx, const hmr_tensor_desc *mask, void *workspace,
                                int64_t workspace_bytes, const hmr_tensor_desc *out);

/*
 * Pooling (csrc/pool.hip): the planes are all dimensions but the last two.
 */

/* cpu::max_pool2d: out and indices (I64, plane positions) for the windows of input; `geometry`
 * holds the kernel size, stride, padding and dilation, each for the height and the width. */
hmr_status hmr_max_pool2d(hmr_context_t ctx, const hmr_tensor_desc *out,
                          const hmr_tensor_desc *indices, const hmr_tensor_desc *input,
                          const int64_t *geometry);

/* The check of max_pool2d's indices (I64, the output's sizes) for its backward, in a plane of
 * height x width: slot[0] gets the first position, in row-major order, whose index is outside
 * the plane (UINT64_MAX for none), slot[1] that index, and slot[2] 1 if an index inside the
 * plane is outside the window (of `geometry`, as in hmr_max_pool2d, positions in the padding
 * included) of its output position, else 0. `slot` is three words of device memory. */
hmr_status hmr_max_pool2d_check(hmr_context_t ctx, const hmr_tensor_desc *indices,
                                int64_t height, int64_t width, const int64_t *geometry,
                                uint64_t *slot);

/* cpu::max_pool2d_backward for indices inside their windows of `geometry`
 * (hmr_max_pool2d_check found none outside): each element of out (the input's shape, contiguous)
 * gets, from zero, the elements of grad whose index in `indices` (I64, grad's sizes) is its
 * position, added in the dtype in row-major order of grad. Every element of out is written;
 * nothing is sorted. */
hmr_status hmr_max_pool2d_gather_backward(hmr_context_t ctx, const hmr_tensor_desc *out,
                                          const hmr_tensor_desc *grad,
                                          const hmr_tensor_desc *indices,
                                          const int64_t *geometry);

/* cpu::adaptive_avg_pool2d and its backward (out has the input's shape). */
hmr_status hmr_adaptive_avg_pool2d(hmr_context_t ctx, const hmr_tensor_desc *out,
                                   const hmr_tensor_desc *input);
hmr_status hmr_adaptive_avg_pool2d_backward(hmr_context_t ctx, const hmr_tensor_desc *out,
                                            const hmr_tensor_desc *grad);

/* cpu::unfold_backward: out has x's shape, grad that of x.unfold(dim, size, step). */
hmr_status hmr_unfold_backward(hmr_context_t ctx, const hmr_tensor_desc *out,
                               const hmr_tensor_desc *grad, int32_t dim, int64_t size,
                               int64_t step);

/*
 * Matrix products with their writeback (csrc/gemm.hip): cpu::gemm (src/backend/cpu/linalg.rs).
 *
 * For each output batch o, row i and column j:
 *   p = sum over k of a[b][i][k] * b[b][k][j], for the batch b = o, or summed over all batches
 *       (sum_batches, addbmm) after rounding each batch's product to the computation type;
 *   out[o][i][j] = scaled ? beta * c[o][i][j] + alpha * p : p, with the CPU's steps and roundings.
 * The operands are the addresses of element (0, 0, 0) with element strides for (batch, row,
 * column); a stride may be 0 (broadcasting) and any value for a dimension of size 0 or 1. The
 * output has batch batches (1 with sum_batches); its elements are distinct and don't overlap the
 * factors or the input. With k == 0 (or no batches to sum) the products are exact zeros.
 *
 * Floats accumulate in their computation type (f32 for F32, f64 for F64), with the fused
 * multiply-add of the path; sums of batches in f64. Integers wrap in 64 bits and are truncated
 * once. F16 and BF16 have no path yet (M4 step 5b), Bool none.
 */
typedef struct hmr_gemm_desc {
    int32_t dtype;
    /* Nonzero: the output is the sum of the batches' products (one output batch). */
    int32_t sum_batches;
    /* Nonzero: beta * c + alpha * p; zero: p, and c and the coefficients are unused. */
    int32_t scaled;
    /* Nonzero: c is read (scaled and beta != 0). */
    int32_t reads_input;
    int64_t batch, m, n, k;
    const void *a;
    const void *b;
    const void *c;
    void *out;
    /* Strides in elements: (batch, row, column) of a [batch, m, k], b [batch, k, n], c and out
     * [batches of the output, m, n]. */
    int64_t a_strides[3];
    int64_t b_strides[3];
    int64_t c_strides[3];
    int64_t out_strides[3];
    /* Floats: the coefficients' values, exact in the computation type. */
    double alpha;
    double beta;
    /* Integers: the coefficients. */
    int64_t alpha_int;
    int64_t beta_int;
} hmr_gemm_desc;

/* The paths of hmr_gemm. HMR_GEMM_GENERIC takes every description of F32, F64 and the integer
 * dtypes. The others are F32 kernels of rocm-exp's S series, named by its programs (the
 * configuration table in rocm-exp/LOG.md): NN, NT, TN for the layouts they take; NARROW for
 * M <= 16; SPLITK for those that split k into slices summed in a workspace. */
#define HMR_GEMM_GENERIC 0
#define HMR_GEMM_S08A_S7A5_UNAL 1
#define HMR_GEMM_S07G_BK16 2
#define HMR_GEMM_S08C_NARROW_T256 3
#define HMR_GEMM_S08C_SPLITK_T128 4
#define HMR_GEMM_S08B_S8A1_NT 5
#define HMR_GEMM_S08B_NARROW_NT 6
#define HMR_GEMM_S08B_SPLITK_NT_T128 7
#define HMR_GEMM_S08B_S7G1_NT 8
#define HMR_GEMM_S08B_S7G1_TN 9
#define HMR_GEMM_S08B_NARROW_TN 10
#define HMR_GEMM_PATHS 11

/* The storage of the factors, as the paths see it: each with a unit stride along k (N for a, T
 * for b) or along its other dimension (T for a, N for b); HMR_GEMM_STRIDED when one has neither. */
#define HMR_GEMM_NN 0
#define HMR_GEMM_NT 1
#define HMR_GEMM_TN 2
#define HMR_GEMM_TT 3
#define HMR_GEMM_STRIDED 4

/*
 * Rewrites a description whose batches share b (b's batch stride is 0, as for a weight
 * broadcast over the batches of `x @ w`) as one batch of batch * m rows, when the rows of
 * consecutive batches of a, the output and the read input continue each other, as PyTorch's
 * matmul folds `[..., m, k] @ [k, n]` into one mm. Other descriptions (and sums of batches) are
 * left as they are. Both forms describe the same product; the folded one has one product of
 * more rows to choose a path for. HMR_INVALID for a description hmr_gemm_select refuses for its
 * sizes or strides.
 */
hmr_status hmr_gemm_fold(hmr_gemm_desc *desc);

/*
 * The path hmr_gemm takes for a description, by fixed rules of its dtype, layout, sizes and
 * alignment (no timing), and the layout of its factors. HMR_INVALID for a description no path
 * takes, or that a path forced by hmr_test_force_gemm_path doesn't take (with the reason).
 */
hmr_status hmr_gemm_select(const hmr_gemm_desc *desc, int32_t *path, int32_t *layout);

/* The name of a path, for HOMURA_ROCM_GEMM_LOG; HMR_INVALID for an unknown one. */
hmr_status hmr_gemm_path_name(int32_t path, const char **name);

/* The bytes of workspace a path needs for a description (0: none). */
hmr_status hmr_gemm_workspace(const hmr_gemm_desc *desc, int32_t path, int64_t *bytes);

/* Queues the product on the context's stream along `path`, which hmr_gemm_select chose (or a
 * path that takes the description), with at least the workspace hmr_gemm_workspace reported. */
hmr_status hmr_gemm(hmr_context_t ctx, const hmr_gemm_desc *desc, int32_t path, void *workspace,
                    int64_t workspace_bytes);

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
    uint32_t gemm_desc_size;
    uint32_t gemm_desc_align;
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
 * hmr_fill, hmr_copy_device, hmr_copy_to_host, hmr_copy_from_host, and the kernels, samplers,
 * reductions, indexing, pooling and matrix product entries above) fail before queueing any
 * work, as their HIP call would: with hipErrorOutOfMemory if out_of_memory is nonzero, else with
 * hipErrorInvalidHandle (review round 124, R148 and R149).
 * hmr_test_reduce_plan(): the plan hmr_reduce chooses for a reduction (csrc/reduce.hpp): the
 * lanes per output, the number of chunks of the reduced range and their size, to test that the
 * tests reach every path. hmr_test_softmax_plan(): the same for hmr_softmax: whether it splits
 * its rows, the lanes per row and the elements a lane keeps in registers (0: none).
 * hmr_test_index_value(): the second step of hmr_index_check alone: if slot[0] (a logical
 * position of `index`, set by the caller) isn't UINT64_MAX, writes the value there to slot[1], to
 * test positions far beyond what a check could reach in a test (2^60 and more, in views with
 * zero strides).
 * hmr_test_wide_pool(on): while `on` is nonzero, hmr_max_pool2d_check and
 * hmr_max_pool2d_gather_backward on the calling thread use their int64_t coordinates even where
 * int32_t ones fit, to test them on tensors of a testable size.
 * hmr_test_pool_candidates(): the number of (output row, output column) pairs the gathering
 * backward of max_pool2d visits over all input positions of a plane of height x width with this
 * geometry and output size, counted on the host with the kernel's own walk, to test that it
 * visits exactly the covering windows (review round 135, R159).
 * hmr_test_force_gemm_path(path): hmr_gemm_select on the calling thread chooses `path` (-1: its
 * own rules), or fails if that path doesn't take the description, to test every path.
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
hmr_status hmr_test_index_value(hmr_context_t ctx, const hmr_tensor_desc *index,
                                uint64_t *slot);
void hmr_test_wide_pool(int32_t on);
void hmr_test_force_gemm_path(int32_t path);
hmr_status hmr_test_pool_candidates(const int64_t *geometry, int64_t height, int64_t width,
                                    int64_t out_h, int64_t out_w, int64_t *count);
int32_t hmr_test_live_streams(void);
hmr_status hmr_test_current_device(int32_t *device);
hmr_status hmr_test_set_current_device(int32_t device);
hmr_status hmr_test_fail(const char *message);

#ifdef __cplusplus
}
#endif

#endif
