// The S series: F32 matrix product kernels ported from the programs rocm-exp measured and chose
// (rocm-exp/LOG.md, "S 系列的完整配置表"), with homura's writeback, batches and workspace.
//
// Each kernel keeps its program's structure (tiles, staging, prefetch, launch order, split-K);
// what the port changes:
// - the output goes through Direct::put (csrc/gemm.hpp), which applies the writeback of
//   cpu::gemm, at any output strides; split-K slices are summed in slice order by
//   reduce_slices, which applies it;
// - batches: grid.z for the kernels that don't split k over it; one launch per batch (with the
//   workspace reused in stream order) for those that do;
// - the workspace comes from the caller (the Rust side's caching allocator);
// - sizes and leading dimensions stay `int` as in the programs, so s_refusal only takes
//   descriptions whose sizes, leading dimensions and grids fit (the generic path takes the rest).
//
// Only the instances the programs launch are kept: S8a-1 and its NT / split-K relatives always
// stage B with float4 loads (4-byte alignment suffices on gfx12, rocm-exp S8a); S7g-1 and its
// NT / TN relatives choose float4 or scalar staging by lda % 4 and ldb % 4 at launch.

#include <algorithm>

#include "gemm.hpp"

namespace hmr {
namespace {

// Sizes and leading dimensions the int arithmetic of the kernels holds with room to spare
// (tile origins past the end, k slices rounded up).
constexpr int64_t kMaxDim = INT32_MAX / 2;
constexpr int64_t kMaxGridYZ = 65535;

// ---- S8a-1 (NN), S8b-1 (NT) and their split-K versions S8c-1, S8b-7 -------------------------
//
// 32 x 32 threads, 4 x 4 elements each (C tile 128 x 128), k steps of 32. A is read with
// wave-uniform scalar loads (rows clamped to M - 1), B staged in LDS through float4 registers
// loaded during the previous step's compute (one buffer). Grouped launch order of 8 block rows.
namespace s8a {

constexpr int kBx = 32, kBy = 32, kTm = 4, kTn = 4, kBk = 32;
constexpr int kRows = kBy * kTm, kCols = kBx * kTn, kT = kBx * kBy;
constexpr int kVq = kBk * kCols / kT / 4;  // float4 of B per thread per step
static_assert(kBk * kCols % (4 * kT) == 0, "B tile must divide over the threads");

// NT stores B transposed into LDS, with a pitch of 132 that keeps the b128 reads aligned.
template <bool NT>
constexpr int kPitch = NT ? kCols + 4 : kCols;

template <bool NT>
__device__ __forceinline__ void load_b(float4 (&v)[kVq], const float *__restrict__ B, int k0,
                                       int j0, int t, int K, int N, int ldb) {
    if constexpr (!NT) {
        if (k0 + kBk <= K && j0 + kCols <= N) {
#pragma unroll
            for (int q = 0; q < kVq; ++q) {
                int e = t + q * kT, r = e / (kCols / 4), c = (e % (kCols / 4)) * 4;
                v[q] = *reinterpret_cast<const float4 *>(B + size_t(k0 + r) * ldb + j0 + c);
            }
            return;
        }
#pragma unroll
        for (int q = 0; q < kVq; ++q) {
            int e = t + q * kT, r = e / (kCols / 4), c = (e % (kCols / 4)) * 4;
            const float *p = B + size_t(k0 + r) * ldb + j0 + c;
            if (k0 + r < K && j0 + c + 3 < N) {
                v[q] = *reinterpret_cast<const float4 *>(p);
            } else {
                v[q].x = k0 + r < K && j0 + c < N ? p[0] : 0.0f;
                v[q].y = k0 + r < K && j0 + c + 1 < N ? p[1] : 0.0f;
                v[q].z = k0 + r < K && j0 + c + 2 < N ? p[2] : 0.0f;
                v[q].w = k0 + r < K && j0 + c + 3 < N ? p[3] : 0.0f;
            }
        }
    } else {
        // B is stored N x K: a float4 along k.
        if (k0 + kBk <= K && j0 + kCols <= N) {
#pragma unroll
            for (int q = 0; q < kVq; ++q) {
                int e = t + q * kT, n = e / (kBk / 4), c = (e % (kBk / 4)) * 4;
                v[q] = *reinterpret_cast<const float4 *>(B + size_t(j0 + n) * ldb + k0 + c);
            }
            return;
        }
#pragma unroll
        for (int q = 0; q < kVq; ++q) {
            int e = t + q * kT, n = e / (kBk / 4), c = (e % (kBk / 4)) * 4;
            const float *p = B + size_t(j0 + n) * ldb + k0 + c;
            const bool okn = j0 + n < N;
            v[q].x = okn && k0 + c < K ? p[0] : 0.0f;
            v[q].y = okn && k0 + c + 1 < K ? p[1] : 0.0f;
            v[q].z = okn && k0 + c + 2 < K ? p[2] : 0.0f;
            v[q].w = okn && k0 + c + 3 < K ? p[3] : 0.0f;
        }
    }
}

template <bool NT>
__device__ __forceinline__ void store_b(float (*bs)[kPitch<NT>], const float4 (&v)[kVq], int t) {
#pragma unroll
    for (int q = 0; q < kVq; ++q) {
        if constexpr (!NT) {
            int e = t + q * kT, r = e / (kCols / 4), c = (e % (kCols / 4)) * 4;
            *reinterpret_cast<float4 *>(&bs[r][c]) = v[q];
        } else {
            int e = t + q * kT, n = e / (kBk / 4), c = (e % (kBk / 4)) * 4;
            bs[c][n] = v[q].x;
            bs[c + 1][n] = v[q].y;
            bs[c + 2][n] = v[q].z;
            bs[c + 3][n] = v[q].w;
        }
    }
}

template <int P>
__device__ __forceinline__ void compute(float (&acc)[kTm][kTn], const float (*bs)[P],
                                        const float *__restrict__ A, int i0, int ty, int tx,
                                        int k0, int M, int K, int lda) {
    const int tyu = __builtin_amdgcn_readfirstlane(ty);
    const float *ap[kTm];
#pragma unroll
    for (int m = 0; m < kTm; ++m) ap[m] = A + size_t(min(i0 + tyu + kBy * m, M - 1)) * lda + k0;
    const int kn = k0 + kBk <= K ? kBk : K - k0;
    if (kn == kBk) {
#pragma unroll
        for (int kk = 0; kk < kBk; ++kk) {
            float b[kTn];
#pragma unroll
            for (int n = 0; n < kTn; ++n) b[n] = bs[kk][tx * kTn + n];
#pragma unroll
            for (int m = 0; m < kTm; ++m)
#pragma unroll
                for (int n = 0; n < kTn; ++n) acc[m][n] = __builtin_fmaf(ap[m][kk], b[n], acc[m][n]);
        }
    } else {
        for (int kk = 0; kk < kn; ++kk) {
            float b[kTn];
#pragma unroll
            for (int n = 0; n < kTn; ++n) b[n] = bs[kk][tx * kTn + n];
#pragma unroll
            for (int m = 0; m < kTm; ++m)
#pragma unroll
                for (int n = 0; n < kTn; ++n) acc[m][n] = __builtin_fmaf(ap[m][kk], b[n], acc[m][n]);
        }
    }
}

// Split: blockIdx.z is the k slice (writing its partial products to `parts`, an M x N slice
// each) and `batch` the batch; otherwise blockIdx.z adds to `batch`.
template <bool NT, bool Split>
__global__ __launch_bounds__(kT) void kernel(const float *__restrict__ A_,
                                             const float *__restrict__ B_, hmr_gemm_desc d,
                                             float *__restrict__ parts, int M, int N, int K,
                                             int lda, int ldb, int64_t batch) {
    __shared__ float Bs[kBk][kPitch<NT>];
    const int tx = threadIdx.x, ty = threadIdx.y, t = ty * kBx + tx;
    const int64_t ob = Split ? batch : batch + blockIdx.z;
    const float *__restrict__ A = A_ + ob * d.a_strides[0];
    const float *__restrict__ B = B_ + ob * d.b_strides[0];
    if constexpr (Split) {
        const int per = (K + int(gridDim.z) * kBk - 1) / (int(gridDim.z) * kBk) * kBk;
        const int kb = int(blockIdx.z) * per, ke = min(K, kb + per);
        A += kb;
        B += NT ? size_t(kb) : size_t(kb) * ldb;
        K = max(ke - kb, 0);
    }
    constexpr int kGroup = 8;
    const int nbn = gridDim.x, nbm = gridDim.y, L = blockIdx.y * nbn + blockIdx.x;
    const int first = L / (kGroup * nbn) * kGroup, gs = min(kGroup, nbm - first);
    const int r = L - first * nbn;
    const int i0 = (first + r % gs) * kRows, j0 = (r / gs) * kCols;
    float acc[kTm][kTn] = {};
    float4 s[kVq];
    load_b<NT>(s, B, 0, j0, t, K, N, ldb);
    for (int k0 = 0; k0 < K; k0 += kBk) {
        store_b<NT>(Bs, s, t);
        __syncthreads();
        if (k0 + kBk < K) load_b<NT>(s, B, k0 + kBk, j0, t, K, N, ldb);
        compute(acc, Bs, A, i0, ty, tx, k0, M, K, lda);
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < kTm; ++m) {
        int i = i0 + ty + kBy * m;
#pragma unroll
        for (int n = 0; n < kTn; ++n) {
            int j = j0 + tx * kTn + n;
            if (i < M && j < N) {
                if constexpr (Split) {
                    parts[size_t(blockIdx.z) * M * N + size_t(i) * N + j] = acc[m][n];
                } else {
                    Direct{d, ob}.put(i, j, acc[m][n]);
                }
            }
        }
    }
}

// The slices of S8c-1 / S8b-7: enough for 128 blocks (two rounds of the resident places).
int splits(const hmr_gemm_desc &d) {
    const int64_t tiles = ceil_div(d.n, kCols) * ceil_div(d.m, kRows);
    const int64_t kt = ceil_div(d.k, kBk);
    return static_cast<int>(std::max<int64_t>(1, std::min(kt, ceil_div(128, tiles))));
}

}  // namespace s8a

// ---- S7g-1 (NN), S8b-2 (NT), S8b-3 (TN) ------------------------------------------------------
//
// 16 x 16 threads, 8 x 8 elements each (C tile 128 x 128, two 64-wide halves per thread), k
// steps of 16, A (transposed) and B both staged in LDS through float4 registers loaded during the
// previous step's compute. Grouped launch order of 8 block rows.
namespace s7g {

constexpr int kTx = 16, kTy = 16, kT = kTx * kTy, kBk = 16;
constexpr int kRows = 128, kCols = 128, kHalf = 64;
constexpr int kApitch = kRows + 4;
constexpr int kAq = kRows * kBk / 4 / kT;
constexpr int kBq = kBk * kCols / 4 / kT;
static_assert(kRows * kBk % (4 * kT) == 0 && kBk * kCols % (4 * kT) == 0, "tiles must divide");

// NT stores B transposed, with a pitch of 132.
template <int L>
constexpr int kBpitch = L == HMR_GEMM_NT ? kCols + 4 : kCols;

struct Stage {
    float4 a[kAq], b[kBq];
};

__device__ __forceinline__ float ld(const float *p, bool ok) { return ok ? *p : 0.0f; }

// The coordinates of float4 q of thread t: for A (NN, NT) row r of the tile and k c..c+3; for A
// in TN k and rows r..r+3; for B (NN, TN) k r and columns c..c+3; for B in NT column n and k
// c..c+3.
__device__ __forceinline__ void coords(int t, int q, int per_row, int &r, int &c) {
    int e = t + q * kT;
    r = e / per_row;
    c = (e % per_row) * 4;
}

template <int L, bool kVec>
__device__ __forceinline__ void load_tiles(Stage &s, const float *__restrict__ A,
                                           const float *__restrict__ B, int i0, int j0, int k0,
                                           int t, int M, int N, int K, int lda, int ldb) {
    constexpr bool AT = L == HMR_GEMM_TN, BT = L == HMR_GEMM_NT;
    if (kVec && i0 + kRows <= M && j0 + kCols <= N && k0 + kBk <= K) {
#pragma unroll
        for (int q = 0; q < kAq; ++q) {
            int r, c;
            if constexpr (AT) {
                coords(t, q, kRows / 4, r, c);  // r: k, c: rows
                s.a[q] = *reinterpret_cast<const float4 *>(A + size_t(k0 + r) * lda + i0 + c);
            } else {
                coords(t, q, kBk / 4, r, c);
                s.a[q] = *reinterpret_cast<const float4 *>(A + size_t(i0 + r) * lda + k0 + c);
            }
        }
#pragma unroll
        for (int q = 0; q < kBq; ++q) {
            int r, c;
            if constexpr (BT) {
                coords(t, q, kBk / 4, r, c);  // r: column, c: k
                s.b[q] = *reinterpret_cast<const float4 *>(B + size_t(j0 + r) * ldb + k0 + c);
            } else {
                coords(t, q, kCols / 4, r, c);
                s.b[q] = *reinterpret_cast<const float4 *>(B + size_t(k0 + r) * ldb + j0 + c);
            }
        }
        return;
    }
#pragma unroll
    for (int q = 0; q < kAq; ++q) {
        int r, c;
        if constexpr (AT) {
            coords(t, q, kRows / 4, r, c);
            const int k = k0 + r, i = i0 + c;
            const float *p = A + size_t(min(k, K - 1)) * lda + i;
            s.a[q].x = ld(p, k < K && i < M);
            s.a[q].y = ld(p + 1, k < K && i + 1 < M);
            s.a[q].z = ld(p + 2, k < K && i + 2 < M);
            s.a[q].w = ld(p + 3, k < K && i + 3 < M);
        } else {
            coords(t, q, kBk / 4, r, c);
            const int i = i0 + r, k = k0 + c;
            const float *p = A + size_t(min(i, M - 1)) * lda + k;
            s.a[q].x = ld(p, i < M && k < K);
            s.a[q].y = ld(p + 1, i < M && k + 1 < K);
            s.a[q].z = ld(p + 2, i < M && k + 2 < K);
            s.a[q].w = ld(p + 3, i < M && k + 3 < K);
        }
    }
#pragma unroll
    for (int q = 0; q < kBq; ++q) {
        int r, c;
        if constexpr (BT) {
            coords(t, q, kBk / 4, r, c);
            const int j = j0 + r, k = k0 + c;
            const float *p = B + size_t(min(j, N - 1)) * ldb + k;
            s.b[q].x = ld(p, j < N && k < K);
            s.b[q].y = ld(p + 1, j < N && k + 1 < K);
            s.b[q].z = ld(p + 2, j < N && k + 2 < K);
            s.b[q].w = ld(p + 3, j < N && k + 3 < K);
        } else {
            coords(t, q, kCols / 4, r, c);
            const int k = k0 + r, j = j0 + c;
            const float *p = B + size_t(min(k, K - 1)) * ldb + j;
            s.b[q].x = ld(p, k < K && j < N);
            s.b[q].y = ld(p + 1, k < K && j + 1 < N);
            s.b[q].z = ld(p + 2, k < K && j + 2 < N);
            s.b[q].w = ld(p + 3, k < K && j + 3 < N);
        }
    }
}

template <int L>
__device__ __forceinline__ void store_tiles(float (*as)[kApitch], float (*bs)[kBpitch<L>],
                                            const Stage &s, int t) {
#pragma unroll
    for (int q = 0; q < kAq; ++q) {
        int r, c;
        if constexpr (L == HMR_GEMM_TN) {
            coords(t, q, kRows / 4, r, c);
            *reinterpret_cast<float4 *>(&as[r][c]) = s.a[q];
        } else {
            coords(t, q, kBk / 4, r, c);
            as[c][r] = s.a[q].x;
            as[c + 1][r] = s.a[q].y;
            as[c + 2][r] = s.a[q].z;
            as[c + 3][r] = s.a[q].w;
        }
    }
#pragma unroll
    for (int q = 0; q < kBq; ++q) {
        int r, c;
        if constexpr (L == HMR_GEMM_NT) {
            coords(t, q, kBk / 4, r, c);
            bs[c][r] = s.b[q].x;
            bs[c + 1][r] = s.b[q].y;
            bs[c + 2][r] = s.b[q].z;
            bs[c + 3][r] = s.b[q].w;
        } else {
            coords(t, q, kCols / 4, r, c);
            *reinterpret_cast<float4 *>(&bs[r][c]) = s.b[q];
        }
    }
}

template <int P>
__device__ __forceinline__ void compute(float (&acc)[8][8], const float (*as)[kApitch],
                                        const float (*bs)[P], int tx, int ty) {
#pragma unroll
    for (int kk = 0; kk < kBk; ++kk) {
        float a[8], b[8];
        *reinterpret_cast<float4 *>(&a[0]) = *reinterpret_cast<const float4 *>(&as[kk][ty * 4]);
        *reinterpret_cast<float4 *>(&a[4]) =
            *reinterpret_cast<const float4 *>(&as[kk][kHalf + ty * 4]);
        *reinterpret_cast<float4 *>(&b[0]) = *reinterpret_cast<const float4 *>(&bs[kk][tx * 4]);
        *reinterpret_cast<float4 *>(&b[4]) =
            *reinterpret_cast<const float4 *>(&bs[kk][kHalf + tx * 4]);
#pragma unroll
        for (int m = 0; m < 8; ++m)
#pragma unroll
            for (int n = 0; n < 8; ++n) acc[m][n] = __builtin_fmaf(a[m], b[n], acc[m][n]);
    }
}

template <int L, bool kVec>
__global__ __launch_bounds__(kT) void kernel(const float *__restrict__ A_,
                                             const float *__restrict__ B_, hmr_gemm_desc d,
                                             int M, int N, int K, int lda, int ldb,
                                             int64_t batch) {
    __shared__ float As[kBk][kApitch];
    __shared__ float Bs[kBk][kBpitch<L>];
    const int t = threadIdx.x, tx = t % kTx, ty = t / kTx;
    const int64_t ob = batch + blockIdx.z;
    const float *__restrict__ A = A_ + ob * d.a_strides[0];
    const float *__restrict__ B = B_ + ob * d.b_strides[0];
    constexpr int kGroup = 8;
    const int nbn = gridDim.x, nbm = gridDim.y, Lb = blockIdx.y * nbn + blockIdx.x;
    const int first = Lb / (kGroup * nbn) * kGroup, gs = min(kGroup, nbm - first);
    const int r = Lb - first * nbn;
    const int i0 = (first + r % gs) * kRows, j0 = (r / gs) * kCols;
    float acc[8][8] = {};
    Stage s;
    load_tiles<L, kVec>(s, A, B, i0, j0, 0, t, M, N, K, lda, ldb);
    for (int k0 = 0; k0 < K; k0 += kBk) {
        store_tiles<L>(As, Bs, s, t);
        __syncthreads();
        if (k0 + kBk < K) load_tiles<L, kVec>(s, A, B, i0, j0, k0 + kBk, t, M, N, K, lda, ldb);
        compute(acc, As, Bs, tx, ty);
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int i = i0 + (m < 4 ? ty * 4 + m : kHalf + ty * 4 + m - 4);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int j = j0 + (n < 4 ? tx * 4 + n : kHalf + tx * 4 + n - 4);
            if (i < M && j < N) Direct{d, ob}.put(i, j, acc[m][n]);
        }
    }
}

}  // namespace s7g

// ---- S8c-2 (NN) and S8b-5 (TN): M <= 16 -----------------------------------------------------
//
// Blocks of 64 threads, 4 columns each (256 per block); the M x chunk slice of A is staged in LDS
// and read as a broadcast; B rows read as float4 (unrolled by 8). k is split over grid.y into
// slices of a multiple of 16 so that about 256 blocks run; with more than one slice the partial
// products go to the workspace and reduce_slices sums them in slice order.
namespace narrow {

constexpr int kT = 64, kCpt = 4, kBn = kT * kCpt;
template <int kM>
constexpr int kChunk = 1024 / kM < 64 ? 64 : 1024 / kM;
constexpr int kTargetBlocks = 256, kUnroll = 8;

// Without a workspace (one slice) the products are written back to output batch `batch` +
// blockIdx.z; with one, to slice blockIdx.y of `parts` (batch `batch`, one launch per batch).
template <bool TN, int kM>
__global__ __launch_bounds__(kT) void kernel(const float *__restrict__ A_,
                                             const float *__restrict__ B_, hmr_gemm_desc d,
                                             float *__restrict__ parts, int M, int N, int K,
                                             int lda, int ldb, int per, int64_t batch) {
    constexpr int kC = kChunk<kM>;
    __shared__ float As[kM][kC];
    const int64_t ob = batch + blockIdx.z;
    const float *__restrict__ A = A_ + ob * d.a_strides[0];
    const float *__restrict__ B = B_ + ob * d.b_strides[0];
    const int t = threadIdx.x, j = blockIdx.x * kBn + t * kCpt;
    const int kb = blockIdx.y * per, ke = min(K, kb + per);
    float acc[kM][kCpt] = {};
    const bool full = j + kCpt <= N;
    for (int c0 = kb; c0 < ke; c0 += kC) {
        const int cn = min(kC, ke - c0);
        __syncthreads();
        for (int e = t; e < kM * kC; e += kT) {
            if constexpr (TN) {
                // A stored K x M: m runs fastest for coalesced reads.
                const int m = e % kM, k = e / kM;
                As[m][k] = m < M && k < cn ? A[size_t(c0 + k) * lda + m] : 0.0f;
            } else {
                const int m = e / kC, k = e % kC;
                As[m][k] = m < M && k < cn ? A[size_t(m) * lda + c0 + k] : 0.0f;
            }
        }
        __syncthreads();
        const float *bp = B + size_t(c0) * ldb + j;
        int k = 0;
        if (full) {
            float4 b[kUnroll];
            for (; k + kUnroll <= cn; k += kUnroll) {
#pragma unroll
                for (int u = 0; u < kUnroll; ++u)
                    b[u] = *reinterpret_cast<const float4 *>(bp + size_t(k + u) * ldb);
#pragma unroll
                for (int u = 0; u < kUnroll; ++u)
#pragma unroll
                    for (int m = 0; m < kM; ++m) {
                        const float a = As[m][k + u];
                        acc[m][0] = __builtin_fmaf(a, b[u].x, acc[m][0]);
                        acc[m][1] = __builtin_fmaf(a, b[u].y, acc[m][1]);
                        acc[m][2] = __builtin_fmaf(a, b[u].z, acc[m][2]);
                        acc[m][3] = __builtin_fmaf(a, b[u].w, acc[m][3]);
                    }
            }
        }
        for (; k < cn; ++k) {
            float b[kCpt];
#pragma unroll
            for (int c = 0; c < kCpt; ++c) b[c] = j + c < N ? bp[size_t(k) * ldb + c] : 0.0f;
#pragma unroll
            for (int m = 0; m < kM; ++m)
#pragma unroll
                for (int c = 0; c < kCpt; ++c) acc[m][c] = __builtin_fmaf(As[m][k], b[c], acc[m][c]);
        }
    }
#pragma unroll
    for (int m = 0; m < kM; ++m)
        if (m < M)
#pragma unroll
            for (int c = 0; c < kCpt; ++c)
                if (j + c < N) {
                    if (parts) {
                        parts[size_t(blockIdx.y) * M * N + size_t(m) * N + j + c] = acc[m][c];
                    } else {
                        Direct{d, ob}.put(m, j + c, acc[m][c]);
                    }
                }
}

// The slices: (count, k per slice), as the program's splits_for and run.
void split(const hmr_gemm_desc &d, int &slices, int &per) {
    const int64_t cols = ceil_div(d.n, kBn);
    const int64_t s = std::max<int64_t>(1, std::min(ceil_div(d.k, 16), ceil_div(kTargetBlocks, cols)));
    const int64_t p = ceil_div(ceil_div(d.k, s), 16) * 16;
    per = static_cast<int>(p);
    slices = static_cast<int>(ceil_div(d.k, p));
}

}  // namespace narrow

// ---- S8b-6 (NT, M <= 16) --------------------------------------------------------------------
//
// Blocks of 256 threads (8 waves); each wave owns 4 columns. Lane l walks the float4 groups l,
// l + 32, ... along k of those 4 B rows and of each A row, then the wave adds its lanes' partial
// sums with shuffles. No LDS, no workspace.
namespace narrow_nt {

constexpr int kT = 256, kWaves = kT / 32, kCpw = 4;

template <int kM>
__global__ __launch_bounds__(kT) void kernel(const float *__restrict__ A_,
                                             const float *__restrict__ B_, hmr_gemm_desc d,
                                             int M, int N, int K, int lda, int ldb,
                                             int64_t batch) {
    const int64_t ob = batch + blockIdx.z;
    const float *__restrict__ A = A_ + ob * d.a_strides[0];
    const float *__restrict__ B = B_ + ob * d.b_strides[0];
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    const int n0 = (blockIdx.x * kWaves + w) * kCpw;
    if (n0 >= N) return;
    float acc[kM][kCpw] = {};
    const int kq = K / 4;
#pragma unroll 2
    for (int g = lane; g < kq; g += 32) {
        float4 a[kM], b[kCpw];
#pragma unroll
        for (int m = 0; m < kM; ++m)
            a[m] = m < M ? *reinterpret_cast<const float4 *>(A + size_t(m) * lda + 4 * g)
                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int c = 0; c < kCpw; ++c)
            b[c] = n0 + c < N ? *reinterpret_cast<const float4 *>(B + size_t(n0 + c) * ldb + 4 * g)
                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int m = 0; m < kM; ++m)
#pragma unroll
            for (int c = 0; c < kCpw; ++c) {
                float v = acc[m][c];
                v = __builtin_fmaf(a[m].x, b[c].x, v);
                v = __builtin_fmaf(a[m].y, b[c].y, v);
                v = __builtin_fmaf(a[m].z, b[c].z, v);
                acc[m][c] = __builtin_fmaf(a[m].w, b[c].w, v);
            }
    }
    for (int k = 4 * kq + lane; k < K; k += 32)
#pragma unroll
        for (int m = 0; m < kM; ++m)
#pragma unroll
            for (int c = 0; c < kCpw; ++c)
                if (m < M && n0 + c < N)
                    acc[m][c] =
                        __builtin_fmaf(A[size_t(m) * lda + k], B[size_t(n0 + c) * ldb + k], acc[m][c]);
#pragma unroll
    for (int m = 0; m < kM; ++m)
#pragma unroll
        for (int c = 0; c < kCpw; ++c) {
            float v = acc[m][c];
#pragma unroll
            for (int o = 16; o > 0; o /= 2) v += __shfl_xor(v, o, 32);
            if (lane == 0 && m < M && n0 + c < N) Direct{d, ob}.put(m, n0 + c, v);
        }
}

}  // namespace narrow_nt

// Sums the `slices` partial products of output batch `ob` in slice order and writes them back.
__global__ void reduce_slices(const float *__restrict__ parts, hmr_gemm_desc d, int64_t ob,
                              int M, int N, int slices) {
    const size_t n = size_t(M) * N;
    for (size_t e = size_t(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += size_t(gridDim.x) * blockDim.x) {
        float s = 0.0f;
        for (int z = 0; z < slices; ++z) s += parts[size_t(z) * n + e];
        Direct{d, ob}.put(static_cast<int64_t>(e / N), static_cast<int64_t>(e % N), s);
    }
}

// The leading dimensions of the factors in `layout`: the stride between rows of A as stored
// (M x K for N, K x M for T) and of B (K x N for N, N x K for T).
void leading(const hmr_gemm_desc &d, int layout, int64_t &lda, int64_t &ldb) {
    lda = layout == HMR_GEMM_TN ? d.a_strides[2] : d.a_strides[1];
    ldb = layout == HMR_GEMM_NT ? d.b_strides[2] : d.b_strides[1];
}

int layout_of_path(int32_t path) {
    switch (path) {
        case HMR_GEMM_S08B_S8A1_NT:
        case HMR_GEMM_S08B_NARROW_NT:
        case HMR_GEMM_S08B_SPLITK_NT_T128:
        case HMR_GEMM_S08B_S7G1_NT:
            return HMR_GEMM_NT;
        case HMR_GEMM_S08B_S7G1_TN:
        case HMR_GEMM_S08B_NARROW_TN:
            return HMR_GEMM_TN;
        default:
            return HMR_GEMM_NN;
    }
}

bool is_narrow(int32_t path) {
    return path == HMR_GEMM_S08C_NARROW_T256 || path == HMR_GEMM_S08B_NARROW_NT ||
           path == HMR_GEMM_S08B_NARROW_TN;
}

bool is_tiled_128(int32_t path) { return !is_narrow(path); }

int64_t workspace_slices(int32_t path, const hmr_gemm_desc &d) {
    if (path == HMR_GEMM_S08C_SPLITK_T128 || path == HMR_GEMM_S08B_SPLITK_NT_T128) {
        int s = s8a::splits(d);
        return s > 1 ? s : 0;
    }
    if (path == HMR_GEMM_S08C_NARROW_T256 || path == HMR_GEMM_S08B_NARROW_TN) {
        int slices = 0, per = 0;
        narrow::split(d, slices, per);
        return slices > 1 ? slices : 0;
    }
    return 0;
}

}  // namespace

std::string s_refusal(int32_t path, const hmr_gemm_desc &d, int layout) {
    if (d.dtype != HMR_F32) return "an F32 path";
    if (layout != layout_of_path(path)) return "another layout";
    if (d.k == 0) return "k == 0 (no product to compute)";
    if (d.sum_batches && d.batch != 1) return "a sum of batches (or of none)";
    if (d.m > kMaxDim || d.n > kMaxDim || d.k > kMaxDim) return "sizes past its int range";
    int64_t lda, ldb;
    leading(d, layout, lda, ldb);
    if (lda > kMaxDim || ldb > kMaxDim) return "leading dimensions past its int range";
    auto aligned = [](const void *p) { return reinterpret_cast<uintptr_t>(p) % 4 == 0; };
    if (!aligned(d.a) || !aligned(d.b)) return "factors not aligned to 4 bytes";
    if (is_narrow(path) && d.m > 16) return "M > 16";
    if (is_tiled_128(path)) {
        int64_t rows = ceil_div(d.m, 128), cols = ceil_div(d.n, 128), blocks = 0;
        if (rows > kMaxGridYZ) return "more row tiles than its grid holds";
        if (!checked_mul(rows, cols, blocks) || blocks > INT32_MAX) return "more tiles than its grid holds";
    }
    return "";
}

int32_t s_preferred(const hmr_gemm_desc &d, int layout) {
    if (d.dtype != HMR_F32) return -1;
    const bool one = d.batch == 1;
    // K long and too few column tiles for the GPU: split k (measured at 128 x 4096 x 11008).
    const bool long_k = d.k >= 8000 && ceil_div(d.n, 128) < 64;
    // The narrow paths need enough k: below these bounds the layout's tile path is faster
    // (M4 step 5a, batch 1). NN and TN: k >= 32 M and 16 M, measured at M = 1-5, 8, 9, 12, 16,
    // N = 256 and 4096, K = 16-512 (at most 17% or 0.9 us lost next to the bounds).
    const bool narrow_nn = d.m <= 16 && d.k >= 32 * d.m;
    const bool narrow_tn = d.m <= 16 && d.k >= 16 * d.m;
    // NT: the narrow kernel runs the instance of r rows (M rounded up to a power of two), with a
    // cost that grows with r and N but hardly with k, and its crossover with S8b-1 depends on N
    // unevenly; so a table of the measured bounds per r and N < 4096, < 8192, larger (round 138,
    // R161: M = 1-5, 8, 9, 12, 16, N = 256, 1024, 2048, 4096, 11008, K = 8-2048 in steps of 8
    // to 32 near the bounds; at most 4.3% or 1.6 us lost at those 550 points). Other N are
    // extrapolated by their class.
    constexpr int64_t kNarrowNt[5][3] = {
        {16, 16, 16}, {16, 16, 16}, {24, 40, 24}, {64, 80, 80}, {192, 352, 256}};
    int r = 0;
    while (r < 4 && (int64_t{1} << r) < d.m) ++r;
    const int ncls = d.n < 4096 ? 0 : d.n < 8192 ? 1 : 2;
    const bool narrow_nt = d.m <= 16 && d.k >= kNarrowNt[r][ncls];
    switch (layout) {
        case HMR_GEMM_NN: {
            if (d.m <= 16) return one && narrow_nn ? HMR_GEMM_S08C_NARROW_T256 : HMR_GEMM_S08A_S7A5_UNAL;
            // rocm-exp preferred S7g-1 for 16 < M <= 128 and for 8192^3, but its NN instance
            // in this build pairs only 359 of its 512 FMA packets (rocm-exp: 511): measured here,
            // S8a-1 is 1.06-1.35x faster at M = 32-128 and within 1% at 8192^3 (M4 step 5a,
            // item 5). S7g-1 NN stays available to forcing.
            if (d.m <= 128 && long_k && one) return HMR_GEMM_S08C_SPLITK_T128;
            return HMR_GEMM_S08A_S7A5_UNAL;
        }
        case HMR_GEMM_NT:
            if (narrow_nt) return HMR_GEMM_S08B_NARROW_NT;
            if (d.m <= 16) return HMR_GEMM_S08B_S8A1_NT;
            if (d.m <= 128) return long_k && one ? HMR_GEMM_S08B_SPLITK_NT_T128 : HMR_GEMM_S08B_S7G1_NT;
            return HMR_GEMM_S08B_S8A1_NT;
        case HMR_GEMM_TN:
            if (narrow_tn && one) return HMR_GEMM_S08B_NARROW_TN;
            return HMR_GEMM_S08B_S7G1_TN;
        default:
            return -1;
    }
}

int64_t s_workspace(int32_t path, const hmr_gemm_desc &d) {
    int64_t slices = workspace_slices(path, d), elements = 0, bytes = 0;
    if (slices == 0) return 0;
    if (!checked_mul(slices, d.m, elements) || !checked_mul(elements, d.n, elements) ||
        !checked_mul(elements, 4, bytes)) {
        return -1;
    }
    return bytes;
}

void s_launch(int32_t path, const hmr_gemm_desc &d, void *workspace, hipStream_t stream) {
    const int layout = layout_of_path(path);
    int64_t lda64, ldb64;
    leading(d, layout, lda64, ldb64);
    const int M = static_cast<int>(d.m), N = static_cast<int>(d.n), K = static_cast<int>(d.k);
    const int lda = static_cast<int>(lda64), ldb = static_cast<int>(ldb64);
    const float *A = static_cast<const float *>(d.a), *B = static_cast<const float *>(d.b);
    float *parts = static_cast<float *>(workspace);
    const int64_t batches = d.batch;
    // Batches over grid.z, at most kMaxGridYZ per launch: calls f(first batch, count).
    auto by_batches = [&](auto f) {
        for (int64_t z = 0; z < batches; z += kMaxGridYZ) f(z, std::min(batches - z, kMaxGridYZ));
    };
    const dim3 tiles(static_cast<unsigned>(ceil_div(d.n, 128)), static_cast<unsigned>(ceil_div(d.m, 128)));
    auto reduce = [&](int64_t b, int slices) {
        reduce_slices<<<1024, 256, 0, stream>>>(parts, d, b, M, N, slices);
    };
    switch (path) {
        case HMR_GEMM_S08A_S7A5_UNAL:
        case HMR_GEMM_S08B_S8A1_NT:
            by_batches([&](int64_t z, int64_t n) {
                dim3 grid(tiles.x, tiles.y, static_cast<unsigned>(n));
                dim3 block(s8a::kBx, s8a::kBy);
                if (path == HMR_GEMM_S08B_S8A1_NT) {
                    s8a::kernel<true, false><<<grid, block, 0, stream>>>(A, B, d, nullptr, M, N, K, lda, ldb, z);
                } else {
                    s8a::kernel<false, false><<<grid, block, 0, stream>>>(A, B, d, nullptr, M, N, K, lda, ldb, z);
                }
            });
            return;
        case HMR_GEMM_S08C_SPLITK_T128:
        case HMR_GEMM_S08B_SPLITK_NT_T128: {
            const int s = s8a::splits(d);
            const bool nt = path == HMR_GEMM_S08B_SPLITK_NT_T128;
            dim3 block(s8a::kBx, s8a::kBy);
            if (s == 1) {
                // One slice: S8a-1 / S8b-1 as they are.
                by_batches([&](int64_t z, int64_t n) {
                    dim3 grid(tiles.x, tiles.y, static_cast<unsigned>(n));
                    if (nt) {
                        s8a::kernel<true, false><<<grid, block, 0, stream>>>(A, B, d, nullptr, M, N, K, lda, ldb, z);
                    } else {
                        s8a::kernel<false, false><<<grid, block, 0, stream>>>(A, B, d, nullptr, M, N, K, lda, ldb, z);
                    }
                });
                return;
            }
            for (int64_t b = 0; b < batches; ++b) {
                dim3 grid(tiles.x, tiles.y, static_cast<unsigned>(s));
                if (nt) {
                    s8a::kernel<true, true><<<grid, block, 0, stream>>>(A, B, d, parts, M, N, K, lda, ldb, b);
                } else {
                    s8a::kernel<false, true><<<grid, block, 0, stream>>>(A, B, d, parts, M, N, K, lda, ldb, b);
                }
                reduce(b, s);
            }
            return;
        }
        case HMR_GEMM_S07G_BK16:
        case HMR_GEMM_S08B_S7G1_NT:
        case HMR_GEMM_S08B_S7G1_TN: {
            const bool vec = lda % 4 == 0 && ldb % 4 == 0;
            by_batches([&](int64_t z, int64_t n) {
                dim3 grid(tiles.x, tiles.y, static_cast<unsigned>(n));
                auto go = [&](auto kernel) {
                    kernel<<<grid, s7g::kT, 0, stream>>>(A, B, d, M, N, K, lda, ldb, z);
                };
                if (path == HMR_GEMM_S07G_BK16) {
                    vec ? go(s7g::kernel<HMR_GEMM_NN, true>) : go(s7g::kernel<HMR_GEMM_NN, false>);
                } else if (path == HMR_GEMM_S08B_S7G1_NT) {
                    vec ? go(s7g::kernel<HMR_GEMM_NT, true>) : go(s7g::kernel<HMR_GEMM_NT, false>);
                } else {
                    vec ? go(s7g::kernel<HMR_GEMM_TN, true>) : go(s7g::kernel<HMR_GEMM_TN, false>);
                }
            });
            return;
        }
        case HMR_GEMM_S08C_NARROW_T256:
        case HMR_GEMM_S08B_NARROW_TN: {
            int slices = 0, per = 0;
            narrow::split(d, slices, per);
            const bool tn = path == HMR_GEMM_S08B_NARROW_TN;
            const unsigned cols = static_cast<unsigned>(ceil_div(d.n, narrow::kBn));
            auto go = [&](auto kernel, float *p, unsigned nz, int64_t b) {
                kernel<<<dim3(cols, static_cast<unsigned>(slices), nz), narrow::kT, 0, stream>>>(
                    A, B, d, p, M, N, K, lda, ldb, per, b);
            };
            auto rows = [&](float *p, unsigned nz, int64_t b) {
                if (tn) {
                    if (d.m <= 1) go(narrow::kernel<true, 1>, p, nz, b);
                    else if (d.m <= 2) go(narrow::kernel<true, 2>, p, nz, b);
                    else if (d.m <= 4) go(narrow::kernel<true, 4>, p, nz, b);
                    else if (d.m <= 8) go(narrow::kernel<true, 8>, p, nz, b);
                    else go(narrow::kernel<true, 16>, p, nz, b);
                } else {
                    if (d.m <= 1) go(narrow::kernel<false, 1>, p, nz, b);
                    else if (d.m <= 2) go(narrow::kernel<false, 2>, p, nz, b);
                    else if (d.m <= 4) go(narrow::kernel<false, 4>, p, nz, b);
                    else if (d.m <= 8) go(narrow::kernel<false, 8>, p, nz, b);
                    else go(narrow::kernel<false, 16>, p, nz, b);
                }
            };
            if (slices == 1) {
                by_batches([&](int64_t z, int64_t n) { rows(nullptr, static_cast<unsigned>(n), z); });
                return;
            }
            for (int64_t b = 0; b < batches; ++b) {
                rows(parts, 1, b);
                reduce(b, slices);
            }
            return;
        }
        case HMR_GEMM_S08B_NARROW_NT: {
            const unsigned blocks =
                static_cast<unsigned>(ceil_div(d.n, narrow_nt::kWaves * narrow_nt::kCpw));
            by_batches([&](int64_t z, int64_t n) {
                dim3 grid(blocks, 1, static_cast<unsigned>(n));
                auto go = [&](auto kernel) {
                    kernel<<<grid, narrow_nt::kT, 0, stream>>>(A, B, d, M, N, K, lda, ldb, z);
                };
                if (d.m <= 1) go(narrow_nt::kernel<1>);
                else if (d.m <= 2) go(narrow_nt::kernel<2>);
                else if (d.m <= 4) go(narrow_nt::kernel<4>);
                else if (d.m <= 8) go(narrow_nt::kernel<8>);
                else go(narrow_nt::kernel<16>);
            });
            return;
        }
        default:
            return;
    }
}

}  // namespace hmr
