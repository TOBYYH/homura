// Matrix products with their writeback (hmr_gemm_* in homura_rocm.h): the path selection, and
// the generic path, which takes every description of F32, F64 and the integer dtypes.
//
// The selection is a fixed function of the description (dtype, layout, sizes, alignment) and
// the architecture, so a path, and with it the result, is the same in every process. The fast
// paths of M4 step 5a (rocm-exp's S series) and 5b are added to it; each has a predicate of the
// descriptions it takes, and the generic path takes what none of them does.
//
// The generic path: 64 x 64 output tiles of 256 threads (4 x 4 elements each), the factors
// staged in LDS 16 steps of k at a time with any strides, 64-bit indices throughout. Each
// thread accumulates its elements in k order with the fused multiply-add of the computation
// type (wrapping 64-bit integers), then writes them back as cpu::gemm does.

#include <algorithm>
#include <string>

#include "gemm.hpp"

namespace hmr {
namespace {

// hmr_test_force_gemm_path: the path hmr_gemm_select chooses on this thread, or -1.
thread_local int32_t forced_gemm_path = -1;

const char *const kPathNames[HMR_GEMM_PATHS] = {
    "generic",          "s08a_s7a5_unal", "s07g_bk16",           "s08c_narrow_t256",
    "s08c_splitk_t128", "s08b_s8a1_nt",   "s08b_narrow_nt",      "s08b_splitk_nt_t128",
    "s08b_s7g1_nt",     "s08b_s7g1_tn",   "s08b_narrow_tn"};

constexpr int kTile = 64, kBk = 16, kSide = 16, kPer = kTile / kSide;
constexpr int64_t kMaxGridYZ = 65535, kMaxGridX = 0x7FFFFFFF;

bool is_float(int dtype) { return dtype == HMR_F32 || dtype == HMR_F64; }
bool is_int(int dtype) { return dtype >= HMR_U8 && dtype <= HMR_I64; }

template <int DT>
__device__ inline typename Acc<DT>::A widen(typename Elem<DT>::T v) {
    return static_cast<typename Acc<DT>::A>(v);
}

template <class A>
__device__ inline A mul_add(A x, A y, A acc) {
    if constexpr (std::is_same_v<A, float>) {
        return __builtin_fmaf(x, y, acc);
    } else if constexpr (std::is_same_v<A, double>) {
        return __builtin_fma(x, y, acc);
    } else {
        return static_cast<int64_t>(static_cast<uint64_t>(acc) +
                                    static_cast<uint64_t>(x) * static_cast<uint64_t>(y));
    }
}

// The sum of the batches' products: f64 for floats, wrapping 64 bits for integers.
template <int DT>
struct Sum {
    using S = int64_t;
    __device__ static S add(S s, int64_t p) {
        return static_cast<int64_t>(static_cast<uint64_t>(s) + static_cast<uint64_t>(p));
    }
    __device__ static int64_t round(S s) { return s; }
};
template <>
struct Sum<HMR_F32> {
    using S = double;
    __device__ static S add(S s, float p) { return s + static_cast<double>(p); }
    __device__ static float round(S s) { return __double2float_rn(s); }
};
template <>
struct Sum<HMR_F64> {
    using S = double;
    __device__ static S add(S s, double p) { return s + p; }
    __device__ static double round(S s) { return s; }
};

// One output tile of batch batch0 + blockIdx.z, row tile row0 + blockIdx.y, column tile
// blockIdx.x.
template <int DT, bool SumBatches>
__global__ __launch_bounds__(kSide *kSide) void gemm_generic(hmr_gemm_desc d, int64_t row0,
                                                             int64_t batch0) {
    using T = typename Elem<DT>::T;
    using A = typename Acc<DT>::A;
    // Padded by one so that the transposed staging stores don't share banks.
    __shared__ A as[kBk][kTile + 1];
    __shared__ A bs[kBk][kTile + 1];
    const int t = threadIdx.x, tx = t % kSide, ty = t / kSide;
    const int64_t i0 = (row0 + blockIdx.y) * kTile, j0 = static_cast<int64_t>(blockIdx.x) * kTile;
    const int64_t ob = batch0 + blockIdx.z;
    const T *pa = static_cast<const T *>(d.a), *pb = static_cast<const T *>(d.b);
    const int64_t *sa = d.a_strides, *sb = d.b_strides;
    typename Sum<DT>::S sum[kPer][kPer] = {};
    A acc[kPer][kPer];
    const int64_t first = SumBatches ? 0 : ob, end = SumBatches ? d.batch : ob + 1;
    for (int64_t bi = first; bi < end; ++bi) {
        for (int m = 0; m < kPer; ++m)
            for (int n = 0; n < kPer; ++n) acc[m][n] = A(0);
        const T *a = pa + bi * sa[0], *b = pb + bi * sb[0];
        for (int64_t k0 = 0; k0 < d.k; k0 += kBk) {
            for (int e = t; e < kTile * kBk; e += kSide * kSide) {
                int r = e / kBk, kk = e % kBk;
                int64_t i = i0 + r, k = k0 + kk;
                as[kk][r] = i < d.m && k < d.k ? widen<DT>(a[i * sa[1] + k * sa[2]]) : A(0);
            }
            for (int e = t; e < kTile * kBk; e += kSide * kSide) {
                int kk = e / kTile, col = e % kTile;
                int64_t j = j0 + col, k = k0 + kk;
                bs[kk][col] = j < d.n && k < d.k ? widen<DT>(b[k * sb[1] + j * sb[2]]) : A(0);
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < kBk; ++kk) {
                A x[kPer], y[kPer];
#pragma unroll
                for (int m = 0; m < kPer; ++m) x[m] = as[kk][ty + kSide * m];
#pragma unroll
                for (int n = 0; n < kPer; ++n) y[n] = bs[kk][tx + kSide * n];
#pragma unroll
                for (int m = 0; m < kPer; ++m)
#pragma unroll
                    for (int n = 0; n < kPer; ++n) acc[m][n] = mul_add(x[m], y[n], acc[m][n]);
            }
            __syncthreads();
        }
        if constexpr (SumBatches) {
            for (int m = 0; m < kPer; ++m)
                for (int n = 0; n < kPer; ++n) sum[m][n] = Sum<DT>::add(sum[m][n], acc[m][n]);
        }
    }
    const T *pc = static_cast<const T *>(d.c);
    T *po = static_cast<T *>(d.out);
    for (int m = 0; m < kPer; ++m) {
        int64_t i = i0 + ty + kSide * m;
        if (i >= d.m) continue;
        for (int n = 0; n < kPer; ++n) {
            int64_t j = j0 + tx + kSide * n;
            if (j >= d.n) continue;
            A p;
            if constexpr (SumBatches) {
                p = Sum<DT>::round(sum[m][n]);
            } else {
                p = acc[m][n];
            }
            const T *c = d.reads_input
                             ? pc + ob * d.c_strides[0] + i * d.c_strides[1] + j * d.c_strides[2]
                             : nullptr;
            po[ob * d.out_strides[0] + i * d.out_strides[1] + j * d.out_strides[2]] =
                write<DT>(d, p, c);
        }
    }
}

hmr_status check(const hmr_gemm_desc &d) {
    if (d.dtype == HMR_BOOL) return fail(HMR_INVALID, "matmul is not supported for Bool tensors");
    if (!is_float(d.dtype) && !is_int(d.dtype)) {
        return fail(HMR_INVALID, "no ROCm matrix product path for dtype " + std::to_string(d.dtype));
    }
    if (d.batch < 0 || d.m < 0 || d.n < 0 || d.k < 0) return fail(HMR_INVALID, "negative size");
    for (int x = 0; x < 3; ++x) {
        if (d.a_strides[x] < 0 || d.b_strides[x] < 0 || d.c_strides[x] < 0 ||
            d.out_strides[x] < 0) {
            return fail(HMR_INVALID, "negative stride");
        }
    }
    return HMR_OK;
}

// The row stride of `m` rows per batch read as one run of batch * m rows, if there is one: the
// batch stride itself for single rows, the row stride when each batch starts where the previous
// one's rows end.
bool rows_continue(int64_t batch_stride, int64_t row_stride, int64_t m, int64_t &rows) {
    int64_t span = 0;
    if (m == 1) {
        rows = batch_stride;
        return true;
    }
    if (checked_mul(m, row_stride, span) && span == batch_stride) {
        rows = row_stride;
        return true;
    }
    return false;
}

// hmr_gemm_fold: the batches as the rows of one batch when b is shared by all of them.
void fold(hmr_gemm_desc &d) {
    int64_t rows = 0, ra = 0, ro = 0, rc = 0;
    if (d.sum_batches || d.batch < 2 || d.m < 1 || d.b_strides[0] != 0) return;
    if (!checked_mul(d.batch, d.m, rows)) return;
    if (!rows_continue(d.a_strides[0], d.a_strides[1], d.m, ra)) return;
    if (!rows_continue(d.out_strides[0], d.out_strides[1], d.m, ro)) return;
    if (d.reads_input && !rows_continue(d.c_strides[0], d.c_strides[1], d.m, rc)) return;
    d.batch = 1;
    d.m = rows;
    d.a_strides[0] = d.out_strides[0] = d.c_strides[0] = 0;
    d.a_strides[1] = ra;
    d.out_strides[1] = ro;
    if (d.reads_input) d.c_strides[1] = rc;
}

// Why `path` doesn't take `d` (empty: it does).
std::string refusal(int32_t path, const hmr_gemm_desc &d) {
    if (path == HMR_GEMM_GENERIC) {
        if (ceil_div(d.n, kTile) > kMaxGridX) return "more columns than its grid holds";
        return "";
    }
    if (path > HMR_GEMM_GENERIC && path < HMR_GEMM_PATHS) return s_refusal(path, d, layout_of(d));
    return "unknown path " + std::to_string(path);
}

// The path the rules choose: the measured preference if it takes `d`, else the generic path.
int32_t choose(const hmr_gemm_desc &d) {
    int32_t preferred = s_preferred(d, layout_of(d));
    if (preferred >= 0 && refusal(preferred, d).empty()) return preferred;
    return HMR_GEMM_GENERIC;
}

// The workspace of `path` for `d`, or a failure if its size overflows.
hmr_status workspace_of(int32_t path, const hmr_gemm_desc &d, int64_t &bytes) {
    bytes = path == HMR_GEMM_GENERIC ? 0 : s_workspace(path, d);
    if (bytes < 0) return fail(HMR_INVALID, "the workspace is too large");
    return HMR_OK;
}

}  // namespace
}  // namespace hmr

hmr_status hmr_gemm_fold(hmr_gemm_desc *desc) {
    return hmr::guarded([&] {
        if (hmr_status s = hmr::check(*desc)) return s;
        hmr::fold(*desc);
        return HMR_OK;
    });
}

hmr_status hmr_gemm_select(const hmr_gemm_desc *desc, int32_t *path, int32_t *layout) {
    return hmr::guarded([&] {
        const hmr_gemm_desc &d = *desc;
        if (hmr_status s = hmr::check(d)) return s;
        *layout = hmr::layout_of(d);
        int32_t chosen = hmr::forced_gemm_path >= 0 ? hmr::forced_gemm_path : hmr::choose(d);
        std::string why = hmr::refusal(chosen, d);
        if (!why.empty()) {
            return hmr::fail(HMR_INVALID, "matrix product path " + std::to_string(chosen) +
                                              " can't take this description: " + why);
        }
        *path = chosen;
        return HMR_OK;
    });
}

hmr_status hmr_gemm_path_name(int32_t path, const char **name) {
    return hmr::guarded([&] {
        if (path < 0 || path >= HMR_GEMM_PATHS) {
            return hmr::fail(HMR_INVALID, "unknown matrix product path " + std::to_string(path));
        }
        *name = hmr::kPathNames[path];
        return HMR_OK;
    });
}

hmr_status hmr_gemm_workspace(const hmr_gemm_desc *desc, int32_t path, int64_t *bytes) {
    return hmr::guarded([&] {
        if (hmr_status s = hmr::check(*desc)) return s;
        std::string why = hmr::refusal(path, *desc);
        if (!why.empty()) return hmr::fail(HMR_INVALID, "matrix product path: " + why);
        return hmr::workspace_of(path, *desc, *bytes);
    });
}

hmr_status hmr_gemm(hmr_context_t ctx, const hmr_gemm_desc *desc, int32_t path, void *workspace,
                    int64_t workspace_bytes) {
    return hmr::guarded([&] {
        const hmr_gemm_desc &d = *desc;
        if (hmr_status s = hmr::check(d)) return s;
        std::string why = hmr::refusal(path, d);
        if (!why.empty()) return hmr::fail(HMR_INVALID, "matrix product path: " + why);
        int64_t needed = 0;
        if (hmr_status s = hmr::workspace_of(path, d, needed)) return s;
        if (workspace_bytes < needed || (needed > 0 && workspace == nullptr)) {
            return hmr::fail(HMR_INVALID, "the workspace is too small");
        }
        const int64_t batches = d.sum_batches ? 1 : d.batch;
        if (batches == 0 || d.m == 0 || d.n == 0) {
            return hmr::launch_on(ctx, "hmr_gemm", [](hipStream_t) { return true; });
        }
        if (path != HMR_GEMM_GENERIC) {
            return hmr::launch_on(ctx, "hmr_gemm", [&](hipStream_t s) {
                hmr::s_launch(path, d, workspace, s);
                return true;
            });
        }
        const int64_t rows = hmr::ceil_div(d.m, hmr::kTile), cols = hmr::ceil_div(d.n, hmr::kTile);
        return hmr::launch_on(ctx, "hmr_gemm", [&](hipStream_t s) {
            return hmr::select(d.dtype, std::integer_sequence<int, HMR_U8, HMR_I8, HMR_I16, HMR_I32,
                                                               HMR_I64, HMR_F32, HMR_F64>{},
                               [&](auto tag) {
                constexpr int DT = decltype(tag)::v;
                // Segments of at most kMaxGridYZ row tiles and output batches.
                for (int64_t z = 0; z < batches; z += hmr::kMaxGridYZ) {
                    int64_t nz = std::min(batches - z, hmr::kMaxGridYZ);
                    for (int64_t y = 0; y < rows; y += hmr::kMaxGridYZ) {
                        int64_t ny = std::min(rows - y, hmr::kMaxGridYZ);
                        dim3 grid(static_cast<unsigned>(cols), static_cast<unsigned>(ny),
                                  static_cast<unsigned>(nz));
                        if (d.sum_batches) {
                            hmr::gemm_generic<DT, true>
                                <<<grid, hmr::kSide * hmr::kSide, 0, s>>>(d, y, z);
                        } else {
                            hmr::gemm_generic<DT, false>
                                <<<grid, hmr::kSide * hmr::kSide, 0, s>>>(d, y, z);
                        }
                    }
                }
                return true;
            });
        });
    });
}

void hmr_test_force_gemm_path(int32_t path) { hmr::forced_gemm_path = path; }
