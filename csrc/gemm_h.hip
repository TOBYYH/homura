// The H series: F16 and BF16 matrix product kernels on the WMMA instructions of gfx12, ported
// from the programs rocm-exp measured and chose (rocm-exp/LOG-H.md, "H 系列的完整配置表", the
// F32 accumulation column), with homura's writeback, batches and workspace.
//
// Each kernel keeps its program's structure (block tile, waves, staging through LDS, register
// prefetch, K tail and edges, launch order); what the port changes:
// - batches: grid.z is the output batch, whose offsets are added to the factors' addresses in
//   64 bits when the kernel starts (the buffer ranges are a batch's matrices);
// - the output goes through Out::put8 (the writeback of cpu::gemm in f32, rounded to the dtype
//   once), with the program's mapping of accumulators to columns, at any output strides;
// - the programs' 32-bit element offsets and buffer ranges stay: h_refusal takes only factors
//   that span less than 2^31 bytes, whose rows don't overlap (the zero reads past a factor's
//   range stand for its rows past M or K), and grids that fit (the generic path takes the rest).
//
// Accumulation is in f32 (the WMMA instructions with an f32 accumulator), as the contract of
// D27 asks; rocm-exp's half precision accumulation variants are not ported.

#include <algorithm>

#include "gemm.hpp"

namespace hmr {
namespace {
namespace h {

typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef __bf16 b8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef unsigned short u8 __attribute__((ext_vector_type(8)));
// The loads and stores of 8 elements need only the elements' 2-byte alignment (gfx12 in the
// driver's default unaligned mode, rocm-exp H2).
typedef u8 u8a __attribute__((aligned(2)));
typedef __amdgpu_buffer_rsrc_t rsrc_t;

constexpr int64_t kMaxGridYZ = 65535;
// The factors' spans in bytes stay below this (32-bit offsets and buffer ranges).
constexpr int64_t kMaxSpanBytes = int64_t{1} << 31;

// One 16 x 16 x 16 product with an f32 accumulator. A and B fragments hold 8 elements per lane:
// A: i = t % 16, k = (t / 16) * 8 + e; B: j = t % 16, k as in A; the accumulator:
// i = t % 16, j = (t / 16) * 8 + e. The programs pass B's fragment first (their tiles are the
// transposes), which this keeps.
template <int DT>
__device__ __forceinline__ f8 mma(u8 b, u8 a, f8 c) {
    if constexpr (DT == HMR_F16) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(h8, b),
                                                                __builtin_bit_cast(h8, a), c);
    } else {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(b8, b),
                                                                 __builtin_bit_cast(b8, a), c);
    }
}

// Rounding to the dtype, to nearest even (the instructions: the programs' conversion).
template <int DT>
__device__ __forceinline__ uint16_t round_to(float f) {
    if constexpr (DT == HMR_F16) {
        return __builtin_bit_cast(uint16_t, static_cast<_Float16>(f));
    } else {
        return __builtin_bit_cast(uint16_t, static_cast<__bf16>(f));
    }
}

// The writeback's form, fixed for a block: SCALED (beta * c + alpha * p, or alpha * p) or the
// product itself; FAST, a block within the output whose rows have unit stride, with the product
// itself (the common case, whose writeback is the programs' own).
template <bool SCALED, bool FAST>
struct Form {
    static constexpr bool scaled = SCALED;
    static constexpr bool fast = FAST;
};

// The output of a product for output batch `ob`; SC: a scaled product (beta * c + alpha * p),
// a launch of its own (the scaled writeback's code, even unexecuted, slows the other kernels
// 2-6% on 1-block-per-CU shapes: M4 step 5b).
template <int DT, bool SC>
struct Out {
    hmr_gemm_desc d;
    int64_t ob;

    // The output element of the f32 product `p` with the input element at `c` (read when
    // reads_input), as write<DT> (csrc/gemm.hpp): alpha * p, then beta * c + that, in f32,
    // rounded to the dtype once.
    template <bool SCALED>
    __device__ __forceinline__ uint16_t value(float p, const uint16_t *c) const {
        if constexpr (!SCALED) {
            return round_to<DT>(p);
        } else {
            const float alpha = static_cast<float>(d.alpha), beta = static_cast<float>(d.beta);
            const float a = alpha == 1.0f ? p : p * alpha;
            if (!d.reads_input) return round_to<DT>(a);
            const float v = up<DT>(*c);
            return round_to<DT>(beta == 1.0f ? v + a : v * beta + a);
        }
    }

    // Elements (i, c + e) for e < 8, those within the output: one 16-byte store for 8 columns in
    // a row of unit stride (the programs' vector store), else one at a time in a short loop.
    // FAST: all 8 are within the output, in a row of unit stride.
    template <class F>
    __device__ __forceinline__ void put8(int64_t i, int64_t c, f8 v) const {
        if (!F::fast && i >= d.m) return;
        uint16_t *o = static_cast<uint16_t *>(d.out) + ob * d.out_strides[0] + i * d.out_strides[1];
        const uint16_t *in = F::scaled ? static_cast<const uint16_t *>(d.c) + ob * d.c_strides[0] + i * d.c_strides[1]
                                       : nullptr;
        const int64_t os = d.out_strides[2], cs = d.c_strides[2];
        if (F::fast || (c + 8 <= d.n && os == 1)) {
            u8 r;
            if constexpr (!F::scaled) {
                // The programs' conversion of a whole vector.
                if constexpr (DT == HMR_F16) {
                    r = __builtin_bit_cast(u8, __builtin_convertvector(v, h8));
                } else {
                    r = __builtin_bit_cast(u8, __builtin_convertvector(v, b8));
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) r[e] = value<true>(v[e], in + (c + e) * cs);
            }
            *reinterpret_cast<u8a *>(o + c) = r;
        } else {
            // A loop, not unrolled, that rotates the vector (no dynamic indexing): the rare case
            // is kept small.
#pragma unroll 1
            for (int e = 0; e < 8 && c + e < d.n; ++e) {
                o[(c + e) * os] = value<F::scaled>(v[0], in + (c + e) * cs);
                v = __builtin_shufflevector(v, v, 1, 2, 3, 4, 5, 6, 7, 0);
            }
        }
    }

    // Element (i, j), within the output.
    __device__ __forceinline__ void put1(int64_t i, int64_t j, float v) const {
        uint16_t *o = static_cast<uint16_t *>(d.out) + ob * d.out_strides[0] + i * d.out_strides[1] + j * d.out_strides[2];
        const uint16_t *in = static_cast<const uint16_t *>(d.c) + ob * d.c_strides[0] + i * d.c_strides[1] + j * d.c_strides[2];
        *o = value<SC>(v, in);
    }

    // Calls f(Form<SC, FAST>{}) for the block of rows i0.. i0 + rows and columns j0.. j0 + cols:
    // one uniform branch, so that each form's writeback is straight-line code.
    template <class F>
    __device__ __forceinline__ void form(int64_t i0, int64_t j0, int rows, int cols, F f) const {
        if constexpr (SC) {
            f(Form<true, false>{});
        } else if (i0 + rows <= d.m && j0 + cols <= d.n && d.out_strides[2] == 1) {
            f(Form<false, true>{});
        } else {
            f(Form<false, false>{});
        }
    }
};

// ---- Buffer loads of the any-shape kernels (rocm-exp H6a) ------------------------------------

// Elements of a rows x cols matrix with leading dimension ld, up to its last element.
__device__ __forceinline__ unsigned span(int rows, int cols, int ld) {
    return unsigned(rows - 1) * ld + cols;
}

// A buffer over the first n elements of p. Reads past the range are zero, per dword: a dword
// that straddles the end reads as zero too.
__device__ __forceinline__ rsrc_t rsrc(const uint16_t *p, unsigned n) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<uint16_t *>(p), 0, int(2 * n), 0x31004000);
}

// 8 consecutive elements from element offset off (zero past the range).
__device__ __forceinline__ u8 ld8(rsrc_t r, unsigned off) {
    return __builtin_bit_cast(u8, __builtin_amdgcn_raw_buffer_load_b128(r, int(2 * off), 0, 0));
}

// The same for a vector that may cross the end n of the range. Only the last element can be
// lost: at an even position d of the vector its dword straddles the end, so that dword is
// replaced by the element read on its own (the upper half, past the matrix, is zero), by
// selects, without branches.
__device__ __forceinline__ u8 ld8_end(rsrc_t r, unsigned off, unsigned n) {
    typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
    u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, int(2 * off), 0, 0);
    const unsigned last = __builtin_amdgcn_raw_buffer_load_b16(r, int(2 * (n - 1)), 0, 0);
    const unsigned d = n - 1 - off;
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = d == 2u * j ? last : v[j];
    return __builtin_bit_cast(u8, v);
}

// The elements of a vector along k from column kc that lie past K read as zero (by dwords: zero
// those past K, keep the lower half of the one K splits).
__device__ __forceinline__ void mask_k(u8 &v, int kc, int K) {
    typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
    u32x4 u = __builtin_bit_cast(u32x4, v);
    const int q = K - kc;  // elements to keep
#pragma unroll
    for (int j = 0; j < 4; ++j) u[j] = 2 * j >= q ? 0u : 2 * j + 1 == q ? u[j] & 0xffffu : u[j];
    v = __builtin_bit_cast(u8, u);
}

// A vector along k (NT's A and B): in the last K tile, the last element of the matrix read on
// its own and the elements past K zeroed.
template <bool LAST>
__device__ __forceinline__ u8 ldk(rsrc_t r, unsigned off, unsigned n, int kc, int K) {
    if constexpr (!LAST) {
        return ld8(r, off);
    } else {
        u8 v = ld8_end(r, off, n);
        mask_k(v, kc, K);
        return v;
    }
}

// The block of the grouped launch order (rocm-exp H2a-g4): groups of 4 block rows, column by
// column within a group. Gives the block's first row i0 and column j0.
__device__ __forceinline__ void grouped(int rows, int cols, unsigned &i0, unsigned &j0) {
    const int nbn = gridDim.x, nbm = gridDim.y, L = blockIdx.y * nbn + blockIdx.x;
    const int first = L / (4 * nbn) * 4, gs = min(4, nbm - first), pos = L - first * nbn;
    i0 = (first + pos % gs) * rows;
    j0 = (pos / gs) * cols;
}

// ---- 128 x 128, 4 waves, register prefetch (rocm-exp H4-a and its H6 relatives) --------------
//
// h04a (NN), h06ntn_4a (NT), h06tn_4a (TN) for M, N multiples of 128 and K of 32, with global
// loads; h06c_4a, h06fk_nt4a, h06fk_tn4a (EDGE) for any shape, with buffer loads and the last K
// tile peeled. Each wave holds 2 x 8 tiles of 16 x 16; per K step of 32, waves 0, 1 fetch B and
// waves 2, 3 A into registers during the previous step's products, then stage them in LDS as
// fragments (one buffer, two barriers per step).
namespace k4a {

constexpr int MR = 2, NR = 8, WY = 4;

// The next K step's stage data. NN: B's rows on waves 0, 1, A's fragments on waves 2, 3. NT: B
// is N x K, its fragments read directly (lane l of tile n is column 16 n + l). TN: A is K x M,
// 8 rows of A^T (lane l of tile r is row 8 l + r). Global loads (tile multiples).
template <int L>
__device__ __forceinline__ void fetch(u8 *st, const uint16_t *__restrict__ A,
                                      const uint16_t *__restrict__ B, unsigned i0, unsigned j0,
                                      int k0, int w, int lane, int g, int lda, int ldb) {
    auto at = [](const uint16_t *p, unsigned off) {
        return *reinterpret_cast<const u8a *>(reinterpret_cast<const char *>(p) + 2 * off);
    };
    if (w < 2) {
        if constexpr (L == HMR_GEMM_NT) {
#pragma unroll
            for (int n = 0; n < 8; ++n) st[n] = at(B, (j0 + n * 16 + lane) * ldb + k0 + w * 16 + g * 8);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned k = k0 + w * 16 + g * 8 + e;
                st[e] = at(B, k * ldb + j0 + lane * 8);
            }
        }
    } else {
        if constexpr (L == HMR_GEMM_TN) {
#pragma unroll
            for (int e = 0; e < 8; ++e) st[e] = at(A, (k0 + (w - 2) * 16 + g * 8 + e) * lda + i0 + lane * 8);
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) st[r] = at(A, (i0 + r * 16 + lane) * lda + k0 + (w - 2) * 16 + g * 8);
        }
    }
}

// The same with buffer loads (any shape); LAST is the last K tile.
template <int L, bool LAST>
__device__ __forceinline__ void fetch_edge(u8 *st, rsrc_t A, rsrc_t B, unsigned i0, unsigned j0,
                                           int k0, int w, int lane, int g, int K, int lda, int ldb,
                                           unsigned na, unsigned nb) {
    if constexpr (L == HMR_GEMM_NN) {
        // h06c_4a
        if (w < 2) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned k = k0 + w * 16 + g * 8 + e;
                st[e] = LAST ? ld8_end(B, k * ldb + j0 + lane * 8, nb) : ld8(B, k * ldb + j0 + lane * 8);
            }
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r)
                st[r] = LAST ? ld8_end(A, (i0 + r * 16 + lane) * lda + k0 + (w - 2) * 16 + g * 8, na)
                             : ld8(A, (i0 + r * 16 + lane) * lda + k0 + (w - 2) * 16 + g * 8);
        }
        if (LAST) {
            if (w < 2) {
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    if (k0 + w * 16 + g * 8 + e >= K) st[e] = u8{};
            } else {
#pragma unroll
                for (int r = 0; r < 8; ++r) mask_k(st[r], k0 + (w - 2) * 16 + g * 8, K);
            }
        }
    } else if constexpr (L == HMR_GEMM_NT) {
        // h06fk_nt4a
        if (w < 2) {
#pragma unroll
            for (int n = 0; n < 8; ++n)
                st[n] = ldk<LAST>(B, (j0 + n * 16 + lane) * ldb + k0 + w * 16 + g * 8, nb, k0 + w * 16 + g * 8, K);
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r)
                st[r] = ldk<LAST>(A, (i0 + r * 16 + lane) * lda + k0 + (w - 2) * 16 + g * 8, na,
                                  k0 + (w - 2) * 16 + g * 8, K);
        }
    } else {
        // h06fk_tn4a. Rows past K are zeroed explicitly: their offsets lie past both ranges only
        // while they fit in 32 bits, and with a large leading dimension they wrap onto real rows.
        const unsigned k = k0 + (w % 2) * 16 + g * 8;
        if (w < 2) {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                st[e] = LAST ? ld8_end(B, (k + e) * ldb + j0 + lane * 8, nb) : ld8(B, (k + e) * ldb + j0 + lane * 8);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                st[e] = LAST ? ld8_end(A, (k + e) * lda + i0 + lane * 8, na) : ld8(A, (k + e) * lda + i0 + lane * 8);
        }
        if (LAST) {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                if (k + e >= unsigned(K)) st[e] = u8{};
        }
    }
}

// Waits for the fetched data, transposes rows into fragments (B's in NN and TN, A^T's in TN) and
// writes the fragments to LDS.
template <int L>
__device__ __forceinline__ void stage(const u8 *st, u8 (*Af)[8][32], u8 (*Bf)[8][32], int w, int t) {
    if constexpr (L == HMR_GEMM_TN) {
        u8 f[8];
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 8; ++e) f[n][e] = st[e][n];
        if (w < 2) {
#pragma unroll
            for (int n = 0; n < 8; ++n) Bf[w][n][t] = f[n];
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) Af[w - 2][r][t] = f[r];
        }
    } else {
        if (w < 2) {
            if constexpr (L == HMR_GEMM_NT) {
#pragma unroll
                for (int n = 0; n < 8; ++n) Bf[w][n][t] = st[n];
            } else {
                u8 rb[8];
#pragma unroll
                for (int n = 0; n < 8; ++n)
#pragma unroll
                    for (int e = 0; e < 8; ++e) rb[n][e] = st[e][n];
#pragma unroll
                for (int n = 0; n < 8; ++n) Bf[w][n][t] = rb[n];
            }
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) Af[w - 2][r][t] = st[r];
        }
    }
}

template <int DT>
__device__ __forceinline__ void compute(f8 (&acc)[MR][NR], u8 (*Af)[8][32], u8 (*Bf)[8][32], int r0,
                                        int n0, int t) {
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        u8 ra[MR], rb[NR];
#pragma unroll
        for (int r = 0; r < MR; ++r) ra[r] = Af[kt][r0 + r][t];
#pragma unroll
        for (int n = 0; n < NR; ++n) rb[n] = Bf[kt][n0 + n][t];
#pragma unroll
        for (int r = 0; r < MR; ++r)
#pragma unroll
            for (int n = 0; n < NR; ++n) acc[r][n] = mma<DT>(rb[n], ra[r], acc[r][n]);
    }
}

// The wave's accumulators to the output: tile (r, n), element e of lane (lane, g) is row
// i0 + (r0 + r) * 16 + lane (TN: i0 + 8 lane + r0 + r) and column j0 + (g * 8 + e) * 8 + n0 + n
// (NT: j0 + (n0 + n) * 16 + g * 8 + e), 8 consecutive columns per store.
template <int DT, int L, class O>
__device__ __forceinline__ void write(const O &out, const f8 (&acc)[MR][NR], unsigned i0,
                                      unsigned j0, int r0, int n0, int lane, int g) {
    out.form(i0, j0, 128, 128, [&](auto form) {
        using F = decltype(form);
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            const int64_t i = L == HMR_GEMM_TN ? i0 + lane * 8 + r0 + r : i0 + (r0 + r) * 16 + lane;
            if constexpr (L == HMR_GEMM_NT) {
#pragma unroll
                for (int n = 0; n < NR; ++n) out.template put8<F>(i, j0 + (n0 + n) * 16 + g * 8, acc[r][n]);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    f8 v;
#pragma unroll
                    for (int n = 0; n < NR; ++n) v[n] = acc[r][n][e];
                    out.template put8<F>(i, j0 + (g * 8 + e) * 8 + n0, v);
                }
            }
        }
    });
}

template <int DT, bool SC, int L, bool EDGE>
__global__ __launch_bounds__(128) void kernel(const uint16_t *__restrict__ A_, const uint16_t *__restrict__ B_,
                                              hmr_gemm_desc d, int M, int N, int K, int lda, int ldb,
                                              int64_t batch0) {
    __shared__ u8 Af[2][8][32], Bf[2][8][32];
    const Out<DT, SC> out{d, batch0 + blockIdx.z};
    const uint16_t *__restrict__ A = A_ + out.ob * d.a_strides[0];
    const uint16_t *__restrict__ B = B_ + out.ob * d.b_strides[0];
    // w is wave-uniform: telling the compiler keeps the choice of buffer scalar (H6a).
    const int t = threadIdx.x, w = EDGE ? __builtin_amdgcn_readfirstlane(threadIdx.y) : threadIdx.y;
    const int lane = t % 16, g = t / 16;
    unsigned i0, j0;
    grouped(128, 128, i0, j0);
    const int r0 = (w % WY) * MR, n0 = (w / WY) * NR;
    f8 acc[MR][NR];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int n = 0; n < NR; ++n) acc[r][n] = f8(0.0f);
    u8 st[8];
    if constexpr (!EDGE) {
        fetch<L>(st, A, B, i0, j0, 0, w, lane, g, lda, ldb);
        for (int k0 = 0; k0 < K; k0 += 32) {
            stage<L>(st, Af, Bf, w, t);
            __syncthreads();
            // The next K step's loads stay in flight during this step's products.
            if (k0 + 32 < K) fetch<L>(st, A, B, i0, j0, k0 + 32, w, lane, g, lda, ldb);
            compute<DT>(acc, Af, Bf, r0, n0, t);
            __syncthreads();
        }
    } else {
        const unsigned na = L == HMR_GEMM_TN ? span(K, M, lda) : span(M, K, lda);
        const unsigned nb = L == HMR_GEMM_NT ? span(N, K, ldb) : span(K, N, ldb);
        const rsrc_t ra = rsrc(A, na), rb = rsrc(B, nb);
        if (32 < K) {
            fetch_edge<L, false>(st, ra, rb, i0, j0, 0, w, lane, g, K, lda, ldb, na, nb);
        } else {
            fetch_edge<L, true>(st, ra, rb, i0, j0, 0, w, lane, g, K, lda, ldb, na, nb);
        }
        int k0 = 0;
        for (; k0 + 64 < K; k0 += 32) {
            stage<L>(st, Af, Bf, w, t);
            __syncthreads();
            fetch_edge<L, false>(st, ra, rb, i0, j0, k0 + 32, w, lane, g, K, lda, ldb, na, nb);
            compute<DT>(acc, Af, Bf, r0, n0, t);
            __syncthreads();
        }
        // The last one or two K steps: the fetch, if any, is of the last K tile.
        for (; k0 < K; k0 += 32) {
            stage<L>(st, Af, Bf, w, t);
            __syncthreads();
            if (k0 + 32 < K) fetch_edge<L, true>(st, ra, rb, i0, j0, k0 + 32, w, lane, g, K, lda, ldb, na, nb);
            compute<DT>(acc, Af, Bf, r0, n0, t);
            __syncthreads();
        }
    }
    write<DT, L>(out, acc, i0, j0, r0, n0, lane, g);
}

}  // namespace k4a

// ---- 128 x 128 (4 waves) and 128 x 256 (8 waves), no prefetch (rocm-exp H2a-g4l, H5a-n) ------
//
// h02g_g4l (NN, 128 columns), h02n256 (NN) and h06tn_n256 (TN) for tile multiples (M of 128, N
// of the block's columns, K of 32), with global loads; h06c_n256 (NN) and h06fk_tn256 (TN) for
// any shape, with buffer loads and the last K tile peeled. Per K step of 32, half the waves read
// B's rows (transposed into fragments in registers) and two waves A, into LDS; an LDS-only
// barrier, the products, another barrier. Each wave holds 2 x 8 tiles.
namespace kg {

constexpr int MR = 2, NR = 8, WY = 4;

template <int DT>
__device__ __forceinline__ void compute(f8 (&acc)[MR][NR], u8 (*Af)[8][32], const u8 *Bf, int bn, int r0,
                                        int n0, int t) {
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        u8 ra[MR], rb[NR];
#pragma unroll
        for (int r = 0; r < MR; ++r) ra[r] = Af[kt][r0 + r][t];
#pragma unroll
        for (int n = 0; n < NR; ++n) rb[n] = Bf[(kt * bn + n0 + n) * 32 + t];
#pragma unroll
        for (int r = 0; r < MR; ++r)
#pragma unroll
            for (int n = 0; n < NR; ++n) acc[r][n] = mma<DT>(rb[n], ra[r], acc[r][n]);
    }
}

// The wave's accumulators to the output: tile (r, n), element e is row i0 + (r0 + r) * 16 + lane
// (TN: i0 + 8 lane + r0 + r) and column j0 + n0 / 8 * 128 + (g * 8 + e) * 8 + n0 % 8 + n.
template <int DT, int L, class O>
__device__ __forceinline__ void write(const O &out, const f8 (&acc)[MR][NR], unsigned i0,
                                      unsigned j0, int cols, int r0, int n0, int lane, int g) {
    out.form(i0, j0, 128, cols, [&](auto form) {
        using F = decltype(form);
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            const int64_t i = L == HMR_GEMM_TN ? i0 + lane * 8 + r0 + r : i0 + (r0 + r) * 16 + lane;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                f8 v;
#pragma unroll
                for (int n = 0; n < NR; ++n) v[n] = acc[r][n][e];
                out.template put8<F>(i, j0 + n0 / 8 * 128 + (g * 8 + e) * 8 + n0 % 8, v);
            }
        }
    });
}

// Tile multiples, WAVES 4 (h02g_g4l: 128 columns, B on waves 0, 1, A on 2, 3) or 8 (h02n256,
// h06tn_n256: 256 columns, B on waves 0-3, A on 4, 5). TN's A is K x M: rows of A^T transposed
// like B's (lane l of tile r is row 8 l + r).
template <int DT, bool SC, int L, int WAVES>
__global__ __launch_bounds__(32 * WAVES) void kernel(const uint16_t *__restrict__ A_,
                                                     const uint16_t *__restrict__ B_, hmr_gemm_desc d,
                                                     int M, int N, int K, int lda, int ldb, int64_t batch0) {
    constexpr int kCols = 32 * WAVES, kBw = WAVES / 2, kBn = WAVES == 8 ? 16 : 8;
    __shared__ u8 Af[2][8][32], Bf[2][kBn][32];
    const Out<DT, SC> out{d, batch0 + blockIdx.z};
    const uint16_t *__restrict__ A = A_ + out.ob * d.a_strides[0];
    const uint16_t *__restrict__ B = B_ + out.ob * d.b_strides[0];
    (void)M;
    (void)N;
    const int t = threadIdx.x, w = threadIdx.y, lane = t % 16, g = t / 16;
    unsigned i0, j0;
    grouped(128, kCols, i0, j0);
    const int r0 = (w % WY) * MR, n0 = (w / WY) * NR;
    auto at = [](const uint16_t *p, unsigned off) {
        return *reinterpret_cast<const u8a *>(reinterpret_cast<const char *>(p) + 2 * off);
    };
    f8 acc[MR][NR];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int n = 0; n < NR; ++n) acc[r][n] = f8(0.0f);
    for (int k0 = 0; k0 < K; k0 += 32) {
        if (w < kBw) {
            // WAVES 8: wave w reads K half w / 2 of columns half w % 2.
            const int kh = WAVES == 8 ? w / 2 : w, ch = WAVES == 8 ? w % 2 : 0;
            u8 rb[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned k = k0 + kh * 16 + g * 8 + e;
                const u8 row = at(B, k * ldb + j0 + ch * 128 + lane * 8);
#pragma unroll
                for (int n = 0; n < 8; ++n) rb[n][e] = row[n];
            }
#pragma unroll
            for (int n = 0; n < 8; ++n) Bf[kh][ch * 8 + n][t] = rb[n];
        } else if (w < kBw + 2) {
            if constexpr (L == HMR_GEMM_TN) {
                u8 fa[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned k = k0 + (w - kBw) * 16 + g * 8 + e;
                    const u8 row = at(A, k * lda + i0 + lane * 8);
#pragma unroll
                    for (int r = 0; r < 8; ++r) fa[r][e] = row[r];
                }
#pragma unroll
                for (int r = 0; r < 8; ++r) Af[w - kBw][r][t] = fa[r];
            } else {
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    Af[w - kBw][r][t] = at(A, (i0 + r * 16 + lane) * lda + k0 + (w - kBw) * 16 + g * 8);
            }
        }
        __barrier(__CLK_LOCAL_MEM_FENCE);
        compute<DT>(acc, Af, &Bf[0][0][0], kBn, r0, n0, t);
        __barrier(__CLK_LOCAL_MEM_FENCE);
    }
    write<DT, L>(out, acc, i0, j0, kCols, r0, n0, lane, g);
}

// One K tile of the any-shape kernels; LAST is the last one. h06c_n256 (NN): the elements past K
// zeroed, B's 8 row loads all issued before the first is used, w taken from threadIdx.y.
// h06fk_tn256 (TN): rows past K are past both ranges; w is made wave-uniform.
template <int DT, int L, bool LAST>
__device__ __forceinline__ void ktile(f8 (&acc)[MR][NR], u8 (*Af)[8][32], u8 (*Bf)[16][32], rsrc_t A,
                                      rsrc_t B, unsigned i0, unsigned j0, int k0, int w, int t, int K,
                                      int lda, int ldb, int r0, int n0, unsigned na, unsigned nb) {
    const int lane = t % 16, g = t / 16;
    if constexpr (L == HMR_GEMM_NN) {
        if (w < 4) {
            u8 rows[8], rb[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned k = k0 + w / 2 * 16 + g * 8 + e;
                const unsigned off = k * ldb + j0 + w % 2 * 128 + lane * 8;
                rows[e] = LAST ? ld8_end(B, off, nb) : ld8(B, off);
                if (LAST && k >= unsigned(K)) rows[e] = u8{};
            }
            // All 8 loads go out before the first is used (otherwise the compiler issues them in
            // pairs, with a wait for each pair).
            __asm__ volatile("" : "+v"(rows[0]), "+v"(rows[1]), "+v"(rows[2]), "+v"(rows[3]), "+v"(rows[4]),
                             "+v"(rows[5]), "+v"(rows[6]), "+v"(rows[7]));
#pragma unroll
            for (int e = 0; e < 8; ++e)
#pragma unroll
                for (int n = 0; n < 8; ++n) rb[n][e] = rows[e][n];
#pragma unroll
            for (int n = 0; n < 8; ++n) Bf[w / 2][w % 2 * 8 + n][t] = rb[n];
        } else if (w < 6) {
            const int kc = k0 + (w - 4) * 16 + g * 8;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const unsigned off = (i0 + r * 16 + lane) * lda + kc;
                u8 v = LAST ? ld8_end(A, off, na) : ld8(A, off);
                if (LAST) mask_k(v, kc, K);
                Af[w - 4][r][t] = v;
            }
        }
    } else {
        if (w < 4) {
            u8 rb[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned k = k0 + w / 2 * 16 + g * 8 + e;
                u8 row = LAST ? ld8_end(B, k * ldb + j0 + w % 2 * 128 + lane * 8, nb)
                              : ld8(B, k * ldb + j0 + w % 2 * 128 + lane * 8);
                // Rows past K zeroed explicitly (with a large leading dimension their offsets
                // wrap onto real rows; see k4a::fetch_edge).
                if (LAST && k >= unsigned(K)) row = u8{};
#pragma unroll
                for (int n = 0; n < 8; ++n) rb[n][e] = row[n];
            }
#pragma unroll
            for (int n = 0; n < 8; ++n) Bf[w / 2][w % 2 * 8 + n][t] = rb[n];
        } else if (w < 6) {
            u8 fa[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned k = k0 + (w - 4) * 16 + g * 8 + e;
                u8 row = LAST ? ld8_end(A, k * lda + i0 + lane * 8, na) : ld8(A, k * lda + i0 + lane * 8);
                if (LAST && k >= unsigned(K)) row = u8{};
#pragma unroll
                for (int r = 0; r < 8; ++r) fa[r][e] = row[r];
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) Af[w - 4][r][t] = fa[r];
        }
    }
    __barrier(__CLK_LOCAL_MEM_FENCE);
    compute<DT>(acc, Af, &Bf[0][0][0], 16, r0, n0, t);
    __barrier(__CLK_LOCAL_MEM_FENCE);
}

// h06c_n256 (NN) and h06fk_tn256 (TN): any shape, 128 x 256.
template <int DT, bool SC, int L>
__global__ __launch_bounds__(256) void edge(const uint16_t *__restrict__ A_, const uint16_t *__restrict__ B_,
                                            hmr_gemm_desc d, int M, int N, int K, int lda, int ldb,
                                            int64_t batch0) {
    __shared__ u8 Af[2][8][32], Bf[2][16][32];
    const Out<DT, SC> out{d, batch0 + blockIdx.z};
    const uint16_t *__restrict__ A = A_ + out.ob * d.a_strides[0];
    const uint16_t *__restrict__ B = B_ + out.ob * d.b_strides[0];
    const int t = threadIdx.x;
    const int w = L == HMR_GEMM_TN ? __builtin_amdgcn_readfirstlane(threadIdx.y) : threadIdx.y;
    const int lane = t % 16, g = t / 16;
    unsigned i0, j0;
    grouped(128, 256, i0, j0);
    const int r0 = (w % WY) * MR, n0 = (w / WY) * NR;
    f8 acc[MR][NR];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int n = 0; n < NR; ++n) acc[r][n] = f8(0.0f);
    const unsigned na = L == HMR_GEMM_TN ? span(K, M, lda) : span(M, K, lda), nb = span(K, N, ldb);
    const rsrc_t ra = rsrc(A, na), rb = rsrc(B, nb);
    // The first element of the last K tile (K > 0).
    const int klast = (K - 1) & ~31;
    for (int k0 = 0; k0 < klast; k0 += 32)
        ktile<DT, L, false>(acc, Af, Bf, ra, rb, i0, j0, k0, w, t, K, lda, ldb, r0, n0, na, nb);
    ktile<DT, L, true>(acc, Af, Bf, ra, rb, i0, j0, klast, w, t, K, lda, ldb, r0, n0, na, nb);
    write<DT, L>(out, acc, i0, j0, 256, r0, n0, lane, g);
}

}  // namespace kg

// ---- NT, 128 x 256, K step 64 (rocm-exp H6b-1k, H6f-k1) --------------------------------------
//
// h06ntk_n256 (K a multiple of 64, global loads) and h06fk_nt256 (any shape, buffer loads, the
// last K step peeled). B is N x K: per K step of 64 the 8 lanes of a group read the 64 elements
// of one column of the block (a whole cache line), 4 columns per load; chunk c of a column is
// half c % 2 of K tile c / 2. B's LDS layout keeps the 16 lanes of half a row-wise store on
// distinct banks; a fragment read stays 2 x 256 contiguous bytes. A is read in two batches of K
// tiles per wave (waves 4, 5). Lane l of tile n is column 16 n + l.
namespace knt {

constexpr int MR = 2, NR = 8, WY = 4;
constexpr int BK = 64, KT = BK / 16;
constexpr int kLpr = BK / 8, kRpi = 32 / kLpr, kNq = 64 / kRpi;
constexpr int kR = 16 / kLpr;
constexpr int kPG = 256 + 16 * kR, kTS = 256 + kPG, kKTS = 16 * kTS + 32 * kR;

__device__ __forceinline__ int boff(int kt, int n, int l, int g) { return kt * kKTS + n * kTS + 16 * l + g * kPG; }

template <int DT, bool SC, bool EDGE>
__global__ __launch_bounds__(256) void kernel(const uint16_t *__restrict__ A_, const uint16_t *__restrict__ B_,
                                              hmr_gemm_desc d, int M, int N, int K, int lda, int ldb,
                                              int64_t batch0) {
    __shared__ u8 Af[KT][8][32];
    __shared__ __attribute__((aligned(16))) char Bs[KT * kKTS];
    const Out<DT, SC> out{d, batch0 + blockIdx.z};
    const uint16_t *__restrict__ A = A_ + out.ob * d.a_strides[0];
    const uint16_t *__restrict__ B = B_ + out.ob * d.b_strides[0];
    const int t = threadIdx.x, w = EDGE ? __builtin_amdgcn_readfirstlane(threadIdx.y) : threadIdx.y;
    const int lane = t % 16, g = t / 16;
    unsigned i0, j0;
    grouped(128, 256, i0, j0);
    const int r0 = (w % WY) * MR, n0 = (w / WY) * NR;
    f8 acc[MR][NR];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int n = 0; n < NR; ++n) acc[r][n] = f8(0.0f);
    const unsigned na = EDGE ? span(M, K, lda) : 0, nb = EDGE ? span(N, K, ldb) : 0;
    const rsrc_t rsa = rsrc(A, na), rsb = rsrc(B, nb);
    // One K step; LAST loads the last one (EDGE).
    auto step = [&](int k0, auto last) {
        constexpr bool LAST = decltype(last)::v != 0;
        auto load = [&](rsrc_t r, const uint16_t *p, unsigned off, unsigned n, int kc) {
            if constexpr (EDGE) {
                return ldk<LAST>(r, off, n, kc, K);
            } else {
                return *reinterpret_cast<const u8a *>(reinterpret_cast<const char *>(p) + 2 * off);
            }
        };
        if (w < 4) {
            // Wave w reads columns 64 w .. 64 w + 63.
            const int c = t % kLpr;
#pragma unroll
            for (int q = 0; q < kNq; ++q) {
                const int jj = w * 64 + q * kRpi + t / kLpr;
                *reinterpret_cast<u8 *>(Bs + boff(c / 2, jj / 16, jj % 16, c % 2)) =
                    load(rsb, B, (j0 + jj) * ldb + k0 + c * 8, nb, k0 + c * 8);
            }
        } else if (w < 6) {
#pragma unroll
            for (int f = 0; f < KT / 2; ++f) {
                const int kc = k0 + ((w - 4) * KT / 2 + f) * 16 + g * 8;
                u8 ra[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) ra[r] = load(rsa, A, (i0 + r * 16 + lane) * lda + kc, na, kc);
#pragma unroll
                for (int r = 0; r < 8; ++r) Af[(w - 4) * KT / 2 + f][r][t] = ra[r];
            }
        }
        __barrier(__CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            u8 ra[MR], rb[NR];
#pragma unroll
            for (int r = 0; r < MR; ++r) ra[r] = Af[kt][r0 + r][t];
#pragma unroll
            for (int n = 0; n < NR; ++n) rb[n] = *reinterpret_cast<const u8 *>(Bs + boff(kt, n0 + n, lane, g));
#pragma unroll
            for (int r = 0; r < MR; ++r)
#pragma unroll
                for (int n = 0; n < NR; ++n) acc[r][n] = mma<DT>(rb[n], ra[r], acc[r][n]);
        }
        __barrier(__CLK_LOCAL_MEM_FENCE);
    };
    if constexpr (EDGE) {
        int k0 = 0;
        for (; k0 + BK < K; k0 += BK) step(k0, Tag<0>{});
        step(k0, Tag<1>{});
    } else {
        for (int k0 = 0; k0 < K; k0 += BK) step(k0, Tag<0>{});
    }
    out.form(i0, j0, 128, 256, [&](auto form) {
        using F = decltype(form);
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            const int64_t i = i0 + (r0 + r) * 16 + lane;
#pragma unroll
            for (int n = 0; n < NR; ++n) out.template put8<F>(i, j0 + (n0 + n) * 16 + g * 8, acc[r][n]);
        }
    });
}

}  // namespace knt

// ---- M = 128: double buffer, staggered K start (rocm-exp H6e-ngx, -gpx, -tgx, -tgbx) ---------
//
// h06nng_4x (NN), h06ntgp_4x (NT), h06tng_4x (TN) for M, N multiples of 128 and K of 32;
// h06tngb_4x (TN) for any M (rows of A from M on read as zero, rows of C from M on not written).
// 128 x 128 blocks of 4 waves, two LDS buffers: the next K step is staged into one while the
// other is computed, and the loads of the step after that stay in flight. Every thread reads 4
// rows or fragments of each factor; the K tile and the half of a fragment group follow from the
// wave. Block L's K loop starts at (L * 37) % (K / 32) * 32 and wraps (so the blocks don't all
// read the same rows of B at once): the order of the k steps, and so the rounding of the sum,
// depends on the block, the same in every run.
namespace k4x {

constexpr int MR = 2, NR = 8, WY = 4;
typedef unsigned short h4 __attribute__((ext_vector_type(4)));

// 8 consecutive elements of a row of TN's A (K x M) at element offset off, the first `left` of
// them below M (zero from M on); n is A's span. Branch-free: a buffer load (zero past n, by
// dwords), the last element read on its own, and dword selects for the elements from M on.
__device__ __forceinline__ u8 ld8b(const uint16_t *A, unsigned off, int left, unsigned n) {
    typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
    const rsrc_t r = rsrc(A, n);
    u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, int(2 * off), 0, 0);
    const unsigned last = __builtin_amdgcn_raw_buffer_load_b16(r, int(2 * (n - 1)), 0, 0);
    const unsigned d = n - 1 - off;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = d == 2u * j ? last : v[j];
        v[j] = 2 * j >= left ? 0u : 2 * j + 1 == left ? v[j] & 0xffffu : v[j];
    }
    return __builtin_bit_cast(u8, v);
}

// BM: TN with any M (h06tngb_4x), A read with ld8b.
template <int L, bool BM>
__device__ __forceinline__ void fetch(u8 *st, const uint16_t *__restrict__ A, const uint16_t *__restrict__ B,
                                      unsigned i0, unsigned j0, int k0, int w, int lane, int g, int lda,
                                      int ldb, int M, unsigned na) {
    auto at = [](const uint16_t *p, unsigned off) {
        return *reinterpret_cast<const u8a *>(reinterpret_cast<const char *>(p) + 2 * off);
    };
    const int kt = w / 2, gf = w % 2;
    if constexpr (L == HMR_GEMM_NT) {
#pragma unroll
        for (int q = 0; q < 4; ++q) st[q] = at(B, (j0 + (gf * 4 + q) * 16 + lane) * ldb + k0 + kt * 16 + g * 8);
    } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const unsigned k = k0 + kt * 16 + gf * 8 + g * 4 + e;
            st[e] = at(B, k * ldb + j0 + lane * 8);
        }
    }
    if constexpr (L == HMR_GEMM_TN) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const unsigned off = unsigned(k0 + kt * 16 + gf * 8 + g * 4 + e) * lda + i0 + lane * 8;
            st[4 + e] = BM ? ld8b(A, off, M - int(i0 + lane * 8), na) : at(A, off);
        }
    } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) st[4 + q] = at(A, (i0 + (gf * 4 + q) * 16 + lane) * lda + k0 + kt * 16 + g * 8);
    }
}

// Waits for the fetched data, transposes rows into half fragments (B's in NN and TN, A^T's in
// TN) and writes all to LDS.
template <int L>
__device__ __forceinline__ void stage(const u8 *st, u8 (*Af)[8][32], u8 (*Bf)[8][32], int w, int t) {
    const int kt = w / 2, gf = w % 2, lane = t % 16, g = t / 16;
    if constexpr (L == HMR_GEMM_NT) {
#pragma unroll
        for (int q = 0; q < 4; ++q) Bf[kt][gf * 4 + q][t] = st[q];
    } else {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            h4 v;
#pragma unroll
            for (int e = 0; e < 4; ++e) v[e] = st[e][n];
            reinterpret_cast<h4 *>(&Bf[kt][n][lane + 16 * gf])[g] = v;
        }
    }
    if constexpr (L == HMR_GEMM_TN) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            h4 v;
#pragma unroll
            for (int e = 0; e < 4; ++e) v[e] = st[4 + e][r];
            reinterpret_cast<h4 *>(&Af[kt][r][lane + 16 * gf])[g] = v;
        }
    } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) Af[kt][gf * 4 + q][t] = st[4 + q];
    }
}

template <int DT, bool SC, int L, bool BM>
__global__ __launch_bounds__(128) void kernel(const uint16_t *__restrict__ A_, const uint16_t *__restrict__ B_,
                                              hmr_gemm_desc d, int M, int N, int K, int lda, int ldb,
                                              int64_t batch0) {
    __shared__ u8 Af[2][2][8][32], Bf[2][2][8][32];
    const Out<DT, SC> out{d, batch0 + blockIdx.z};
    const uint16_t *__restrict__ A = A_ + out.ob * d.a_strides[0];
    const uint16_t *__restrict__ B = B_ + out.ob * d.b_strides[0];
    (void)N;
    const int t = threadIdx.x, w = threadIdx.y, lane = t % 16, g = t / 16;
    unsigned i0, j0;
    grouped(128, 128, i0, j0);
    const int r0 = (w % WY) * MR, n0 = (w / WY) * NR;
    // A's span (A is K x M), for BM.
    const unsigned na = BM ? unsigned(K - 1) * lda + M : 0;
    const int L0 = blockIdx.y * gridDim.x + blockIdx.x;
    const int ks = L0 * 37 % (K / 32) * 32;
    auto kw = [&](int k) { return k + ks < K ? k + ks : k + ks - K; };
    f8 acc[MR][NR];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int n = 0; n < NR; ++n) acc[r][n] = f8(0.0f);
    u8 st[8];
    fetch<L, BM>(st, A, B, i0, j0, kw(0), w, lane, g, lda, ldb, M, na);
    stage<L>(st, Af[0], Bf[0], w, t);
    __barrier(__CLK_LOCAL_MEM_FENCE);
    if (32 < K) fetch<L, BM>(st, A, B, i0, j0, kw(32), w, lane, g, lda, ldb, M, na);
    int p = 0;
    for (int k0 = 0; k0 < K; k0 += 32, p ^= 1) {
        // The other buffer takes the next K step while this one is computed; the loads of the
        // one after that stay in flight during the products.
        if (k0 + 32 < K) {
            stage<L>(st, Af[1 - p], Bf[1 - p], w, t);
            if (k0 + 64 < K) fetch<L, BM>(st, A, B, i0, j0, kw(k0 + 64), w, lane, g, lda, ldb, M, na);
        }
        k4a::compute<DT>(acc, Af[p], Bf[p], r0, n0, t);
        __barrier(__CLK_LOCAL_MEM_FENCE);
    }
    k4a::write<DT, L>(out, acc, i0, j0, r0, n0, lane, g);
}

}  // namespace k4x

// ---- NT, M <= 16 (rocm-exp H6e-w32) ----------------------------------------------------------
//
// h06ntw32: a 16 x 32 tile of C per block, for N a multiple of 32 and K of 256, any M (made for
// M <= 16). Per K step of 256 each of the 4 waves takes a quarter (64), reading its fragments of
// B and A directly from memory (no LDS in the K loop), two steps in flight; block L starts at
// (L * 37) % (K / 128) * 128 and wraps. The waves' partial tiles are summed in LDS in wave order.
namespace kw32 {

constexpr int NT = 2, NB = 16 * NT;
constexpr int KW = 64, KS = 4 * KW;
constexpr int kLdp = NB + 4;

struct Frag {
    u8 b[NT][4];
    u8 a[4];
};

// The fragments of K [k, k + 64): lane l of B tile n is row j0 + 16 n + l of B (N x K), lane l
// of A row arow of A (a buffer load: zero from row M on, or rows of A for offsets that wrap,
// either way only for rows of C that aren't written).
__device__ __forceinline__ void fetch(Frag &f, rsrc_t A, const uint16_t *__restrict__ B, unsigned arow,
                                      unsigned j0, int k, int lane, int g, int lda, int ldb) {
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int n = 0; n < NT; ++n)
            f.b[n][c] = *reinterpret_cast<const u8a *>(reinterpret_cast<const char *>(B) +
                                                       2 * ((j0 + n * 16 + lane) * ldb + k + c * 16 + g * 8));
#pragma unroll
    for (int c = 0; c < 4; ++c)
        f.a[c] = __builtin_bit_cast(
            u8, __builtin_amdgcn_raw_buffer_load_b128(A, int(2 * (arow * lda + k + c * 16 + g * 8)), 0, 0));
}

template <int DT>
__device__ __forceinline__ void compute(f8 (&acc)[NT], const Frag &f) {
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[n] = mma<DT>(f.b[n][c], f.a[c], acc[n]);
}

template <int DT, bool SC>
__global__ __launch_bounds__(128) void kernel(const uint16_t *__restrict__ A_, const uint16_t *__restrict__ B_,
                                              hmr_gemm_desc d, int M, int N, int K, int lda, int ldb,
                                              int64_t batch0) {
    __shared__ float Ps[4][16][kLdp];
    const Out<DT, SC> out{d, batch0 + blockIdx.z};
    const uint16_t *__restrict__ A = A_ + out.ob * d.a_strides[0];
    const uint16_t *__restrict__ B = B_ + out.ob * d.b_strides[0];
    (void)N;
    const int t = threadIdx.x, w = __builtin_amdgcn_readfirstlane(threadIdx.y), lane = t % 16, g = t / 16;
    const int L = blockIdx.y * gridDim.x + blockIdx.x, S = K / KS;
    const unsigned i0 = blockIdx.y * 16, j0 = blockIdx.x * NB;
    // A's span ends with row M - 1 (rows from M on are past the range); rounded up to a dword,
    // which h_refusal keeps within A (an even span).
    const unsigned na = unsigned(M - 1) * lda + K, arow = i0 + lane;
    const rsrc_t Ar = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint16_t *>(A), 0, int((2 * na + 3) & ~3u), 0x31004000);
    const int ks = L * 37 % (K / 128) * 128;
    // The wave's K offset in step s.
    auto kat = [&](int s) {
        const int k = ks + s * KS + w * KW;
        return k < K ? k : k - K;
    };
    f8 acc[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n] = f8(0.0f);
    Frag f0, f1;
    fetch(f0, Ar, B, arow, j0, kat(0), lane, g, lda, ldb);
    // Each step's loads are issued before the previous step's products (the scheduling barriers
    // keep the compiler from interleaving them); f0 holds step s on entry. Only the S steps are
    // computed: for an odd S the last one after the loop. (rocm-exp computed a step of zero A
    // there instead, which turns an infinite B into NaN.) The loop's last load of f0 repeats
    // step S - 1 when S is even and is not used.
    int s = 0;
    for (; s + 1 < S; s += 2) {
        fetch(f1, Ar, B, arow, j0, kat(s + 1), lane, g, lda, ldb);
        __builtin_amdgcn_sched_barrier(0);
        compute<DT>(acc, f0);
        fetch(f0, Ar, B, arow, j0, kat(min(s + 2, S - 1)), lane, g, lda, ldb);
        __builtin_amdgcn_sched_barrier(0);
        compute<DT>(acc, f1);
    }
    if (s < S) compute<DT>(acc, f0);
    // The 4 waves' partials (each over a quarter of every step) to LDS; summed in wave order.
    typedef float f32x4 __attribute__((ext_vector_type(4)));
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        float *q = &Ps[w][lane][n * 16 + g * 8];
        *reinterpret_cast<f32x4 *>(q) = f32x4{acc[n][0], acc[n][1], acc[n][2], acc[n][3]};
        *reinterpret_cast<f32x4 *>(q + 4) = f32x4{acc[n][4], acc[n][5], acc[n][6], acc[n][7]};
    }
    __barrier(__CLK_LOCAL_MEM_FENCE);
    constexpr int CG = NB / 8;  // 8-column groups per row
    const int tid = w * 32 + t, row = tid / CG, col = tid % CG * 8;
    if (tid < 16 * CG) {
        f8 v;
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = Ps[0][row][col + e];
#pragma unroll
        for (int z = 1; z < 4; ++z)
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] += Ps[z][row][col + e];
        out.form(i0, j0, 16, NB, [&](auto form) { out.template put8<decltype(form)>(i0 + row, j0 + col, v); });
    }
}

}  // namespace kw32

// ---- NN, M <= 16, split-K in one launch (rocm-exp H6c-w1) ------------------------------------
//
// h06w1_narrow: a block of 8 waves computes a 16 x 256 tile of C over a K slice (grid.y), f32
// partials into the workspace. Per K step of 64, wave w reads rows 8w .. 8w + 7 of B (lane t the
// 8 columns from 8t), transposes them into 8 half fragments in LDS with H5a-n's column
// permutation; each wave reads its 4 fragments of A itself (buffer loads: rows from M on read as
// zero). The last block of a column block to finish (a counter per column block, acq_rel at agent
// scope after a release fence) sums the slices in slice order, writes C and resets the counter.
namespace kw1 {

constexpr int kW = 8, kT = 32 * kW, kBn = 256, kBk = 64;
constexpr int kTargetBlocks = 256;

// One K step at k0; LAST: the step may cross K (zero past it).
template <int DT, bool LAST>
__device__ __forceinline__ void kstep(f8 (&acc)[2], u8 (*Bf)[16][32], rsrc_t A, rsrc_t B, unsigned j0, int k0,
                                      int w, int t, int K, int lda, int ldb, unsigned na, unsigned nb) {
    const int lane = t % 16, g = t / 16;
    {
        u8 rows[8], rb[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const unsigned k = k0 + 8 * w + e, off = k * ldb + j0 + 8 * t;
            rows[e] = LAST ? ld8_end(B, off, nb) : ld8(B, off);
            if (LAST && k >= unsigned(K)) rows[e] = u8{};
        }
        __asm__ volatile("" : "+v"(rows[0]), "+v"(rows[1]), "+v"(rows[2]), "+v"(rows[3]), "+v"(rows[4]),
                         "+v"(rows[5]), "+v"(rows[6]), "+v"(rows[7]));
#pragma unroll
        for (int e = 0; e < 8; ++e)
#pragma unroll
            for (int n = 0; n < 8; ++n) rb[n][e] = rows[e][n];
        // K octet w is half w % 2 of K tile w / 2.
#pragma unroll
        for (int n = 0; n < 8; ++n) Bf[w / 2][n + 8 * (t / 16)][16 * (w % 2) + t % 16] = rb[n];
    }
    u8 ra[4];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
        const int kc = k0 + 16 * kt + 8 * g;
        const unsigned off = unsigned(lane) * lda + kc;
        ra[kt] = LAST ? ld8_end(A, off, na) : ld8(A, off);
        if (LAST) mask_k(ra[kt], kc, K);
    }
    __barrier(__CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int kt = 0; kt < 4; ++kt)
#pragma unroll
        for (int q = 0; q < 2; ++q) acc[q] = mma<DT>(Bf[kt][2 * w + q][t], ra[kt], acc[q]);
    __barrier(__CLK_LOCAL_MEM_FENCE);
}

// The last block of a column block to finish sums its slices, in slice order, and writes C.
template <int DT, class O>
__device__ __forceinline__ void finish(const O &out, const float *__restrict__ P, unsigned *__restrict__ count,
                                       int M, int N) {
    __shared__ bool last;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (tid == 0) {
        last = __hip_atomic_fetch_add(&count[blockIdx.x], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) ==
               gridDim.y - 1;
    }
    __syncthreads();
    if (!last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const unsigned j0 = blockIdx.x * kBn;
    const int S = gridDim.y;
    const size_t n = size_t(M) * N;
    for (int e = tid; e < M * kBn; e += kT) {
        const int m = e / kBn;
        const unsigned j = j0 + e % kBn;
        if (j >= unsigned(N)) continue;
        float s = 0.0f;
        for (int z = 0; z < S; ++z) s += P[size_t(z) * n + size_t(m) * N + j];
        out.put1(m, j, s);
    }
    if (tid == 0) count[blockIdx.x] = 0;
}

template <int DT, bool SC>
__global__ __launch_bounds__(kT) void kernel(const uint16_t *__restrict__ A_, const uint16_t *__restrict__ B_,
                                             hmr_gemm_desc d, float *__restrict__ P, unsigned *__restrict__ count,
                                             int M, int N, int K, int lda, int ldb, int per, int64_t batch) {
    __shared__ u8 Bf[4][16][32];
    const Out<DT, SC> out{d, batch};
    const uint16_t *__restrict__ A = A_ + batch * d.a_strides[0];
    const uint16_t *__restrict__ B = B_ + batch * d.b_strides[0];
    // w is wave-uniform: tell the compiler (H6a-b4a).
    const int t = threadIdx.x, w = __builtin_amdgcn_readfirstlane(threadIdx.y), lane = t % 16, g = t / 16;
    const unsigned j0 = blockIdx.x * kBn;
    const int kb = blockIdx.y * per, ke = min(K, kb + per);
    const unsigned na = span(M, K, lda), nb = span(K, N, ldb);
    const rsrc_t ra = rsrc(A, na), rb = rsrc(B, nb);
    f8 acc[2] = {f8(0.0f), f8(0.0f)};
    int k0 = kb;
    for (; k0 + kBk < ke || (k0 + kBk == ke && ke < K); k0 += kBk)
        kstep<DT, false>(acc, Bf, ra, rb, j0, k0, w, t, K, lda, ldb, na, nb);
    if (k0 < ke) kstep<DT, true>(acc, Bf, ra, rb, j0, k0, w, t, K, lda, ldb, na, nb);
    // The accumulator (B's fragment passed first): lane l is row l of C, element e of half g is
    // column 8g + e of the tile, i.e. column 8 (8g + e + 16 (tile / 8)) + tile % 8 of the block.
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        if (lane >= M) break;
        const int tile = 2 * w + q;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const unsigned j = j0 + 8 * (8 * g + e + 16 * (tile / 8)) + tile % 8;
            if (j < unsigned(N)) P[(size_t(blockIdx.y) * M + lane) * N + j] = acc[q][e];
        }
    }
    finish<DT>(out, P, count, M, N);
}

__global__ void zero(unsigned *count, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) count[i] = 0;
}

// The slices: about kTargetBlocks blocks, each slice a multiple of kBk; `per` is a slice's K.
void split(const hmr_gemm_desc &d, int &slices, int &per) {
    const int64_t cols = ceil_div(d.n, kBn);
    const int64_t s = std::max<int64_t>(1, std::min(ceil_div(d.k, kBk), ceil_div(kTargetBlocks, cols)));
    per = static_cast<int>(ceil_div(ceil_div(d.k, s), kBk) * kBk);
    slices = static_cast<int>(ceil_div(d.k, per));
}

}  // namespace kw1

// ---- Pre-transposition into the workspace (rocm-exp H6e-z, H6a-z) ----------------------------

// h06tz_4a: At = A^T for TN's A (K x M, leading dimension lda); At is M x K with leading
// dimension ldt (K rounded up to 8, zero past K). Thread i = q M + m gathers A[8q .. 8q + 7][m]
// and writes them as one 16-byte store; consecutive threads take consecutive m.
__global__ __launch_bounds__(256) void transpose_a(const uint16_t *__restrict__ A, uint16_t *__restrict__ At, int M,
                                                   int K, int lda, int ldt) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x, m = i % unsigned(M), q = i / unsigned(M);
    if (q * 8 >= unsigned(ldt)) return;
    u8 v;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const unsigned k = q * 8 + e;
        v[e] = k < unsigned(K) ? A[size_t(k) * lda + m] : 0;
    }
    *reinterpret_cast<u8 *>(At + size_t(m) * ldt + q * 8) = v;
}

// h06fz_n256: D = S^T for S of R x Cn (leading dimension lds) into D of Cn x R (leading
// dimension ldd), through 64 x 64 tiles in LDS.
__global__ __launch_bounds__(256) void transpose_any(const uint16_t *__restrict__ S, uint16_t *__restrict__ D, int R,
                                                     int Cn, int lds, int ldd) {
    typedef unsigned short u16x4 __attribute__((ext_vector_type(4)));
    typedef u16x4 u16x4a __attribute__((aligned(2)));
    __shared__ unsigned short T[64][66];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const int r0 = blockIdx.y * 64, c0 = blockIdx.x * 64;
    const bool whole = r0 + 64 <= R && c0 + 64 <= Cn;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty + 16 * i;
        if (whole) {
            const u16x4 v = *reinterpret_cast<const u16x4a *>(S + size_t(r0 + r) * lds + c0 + 4 * tx);
#pragma unroll
            for (int e = 0; e < 4; ++e) T[r][4 * tx + e] = v[e];
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int c = c0 + 4 * tx + e;
                T[r][4 * tx + e] = r0 + r < R && c < Cn ? S[size_t(r0 + r) * lds + c] : 0;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = ty + 16 * i;
        u16x4 v;
#pragma unroll
        for (int e = 0; e < 4; ++e) v[e] = T[4 * tx + e][c];
        if (whole) {
            *reinterpret_cast<u16x4 *>(D + size_t(c0 + c) * ldd + r0 + 4 * tx) = v;
        } else if (c0 + c < Cn) {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                if (r0 + 4 * tx + e < R) D[size_t(c0 + c) * ldd + r0 + 4 * tx + e] = v[e];
        }
    }
}

// ---- Paths ---------------------------------------------------------------------------------

struct PathInfo {
    int32_t path;
    int layout;
    // The tile multiples the path needs (1: any).
    int m, n, k;
};

constexpr PathInfo kPaths[] = {
    {HMR_GEMM_H04A, HMR_GEMM_NN, 128, 128, 32},
    {HMR_GEMM_H06NTN_4A, HMR_GEMM_NT, 128, 128, 32},
    {HMR_GEMM_H06TN_4A, HMR_GEMM_TN, 128, 128, 32},
    {HMR_GEMM_H06C_4A, HMR_GEMM_NN, 1, 1, 1},
    {HMR_GEMM_H06FK_NT4A, HMR_GEMM_NT, 1, 1, 1},
    {HMR_GEMM_H06FK_TN4A, HMR_GEMM_TN, 1, 1, 1},
    {HMR_GEMM_H02G_G4L, HMR_GEMM_NN, 128, 128, 32},
    {HMR_GEMM_H02N256, HMR_GEMM_NN, 128, 256, 32},
    {HMR_GEMM_H06TN_N256, HMR_GEMM_TN, 128, 256, 32},
    {HMR_GEMM_H06C_N256, HMR_GEMM_NN, 1, 1, 1},
    {HMR_GEMM_H06FK_TN256, HMR_GEMM_TN, 1, 1, 1},
    {HMR_GEMM_H06NTK_N256, HMR_GEMM_NT, 128, 256, 64},
    {HMR_GEMM_H06FK_NT256, HMR_GEMM_NT, 1, 1, 1},
    {HMR_GEMM_H06NNG_4X, HMR_GEMM_NN, 128, 128, 32},
    {HMR_GEMM_H06NTGP_4X, HMR_GEMM_NT, 128, 128, 32},
    {HMR_GEMM_H06TNG_4X, HMR_GEMM_TN, 128, 128, 32},
    {HMR_GEMM_H06TNGB_4X, HMR_GEMM_TN, 1, 128, 32},
    {HMR_GEMM_H06NTW32, HMR_GEMM_NT, 1, 32, 256},
    {HMR_GEMM_H06W1_NARROW, HMR_GEMM_NN, 1, 1, 1},
    {HMR_GEMM_H06TZ_4A, HMR_GEMM_TN, 1, 1, 1},
    {HMR_GEMM_H06FZ_N256, HMR_GEMM_NT, 1, 1, 1},
};

// The leading dimension of a pre-transposed factor: its row length rounded up to 8.
int64_t padded(int64_t n) { return ceil_div(n, 8) * 8; }

// The columns of a path's block tile.
int64_t block_cols(int32_t path) {
    switch (path) {
        case HMR_GEMM_H02N256:
        case HMR_GEMM_H06TN_N256:
        case HMR_GEMM_H06C_N256:
        case HMR_GEMM_H06FK_TN256:
        case HMR_GEMM_H06NTK_N256:
        case HMR_GEMM_H06FK_NT256:
        case HMR_GEMM_H06FZ_N256:
            return 256;
        default:
            return 128;
    }
}

const PathInfo *info(int32_t path) {
    for (const PathInfo &p : kPaths)
        if (p.path == path) return &p;
    return nullptr;
}

// The stored matrices of the factors in `layout`: A is M x K (NN, NT) or K x M (TN), B is K x N
// (NN, TN) or N x K (NT); rows, columns and leading dimension. A dimension of size 1 (or 0)
// has any stride: the leading dimension is then taken as the row length.
struct Stored {
    int64_t rows, cols, ld;
};

void stored(const hmr_gemm_desc &d, int layout, Stored &a, Stored &b) {
    a = layout == HMR_GEMM_TN ? Stored{d.k, d.m, d.a_strides[2]} : Stored{d.m, d.k, d.a_strides[1]};
    b = layout == HMR_GEMM_NT ? Stored{d.n, d.k, d.b_strides[2]} : Stored{d.k, d.n, d.b_strides[1]};
    for (Stored *s : {&a, &b})
        if (s->rows <= 1) s->ld = s->cols;
}

// Why a factor stored as `s` doesn't suit the paths ("": it does).
const char *factor_refusal(const Stored &s) {
    if (s.ld < s.cols) return "rows of a factor overlap";
    int64_t span = 0, bytes = 0;
    if (!checked_mul(s.rows - 1, s.ld, span) || !checked_add(span, s.cols, span) ||
        !checked_mul(span, 2, bytes) || bytes >= kMaxSpanBytes) {
        return "a factor spans 2^31 bytes or more";
    }
    return "";
}

}  // namespace h
}  // namespace

std::string h_refusal(int32_t path, const hmr_gemm_desc &d, int layout) {
    const h::PathInfo *p = h::info(path);
    if (p == nullptr) return "unknown path " + std::to_string(path);
    if (d.dtype != HMR_F16 && d.dtype != HMR_BF16) return "an F16 or BF16 path";
    if (layout != p->layout) return "another layout";
    if (d.k == 0) return "k == 0 (no product to compute)";
    if (d.sum_batches && d.batch != 1) return "a sum of batches (or of none)";
    if (d.m % p->m || d.n % p->n || d.k % p->k) {
        return "sizes that aren't multiples of " + std::to_string(p->m) + ", " + std::to_string(p->n) +
               ", " + std::to_string(p->k);
    }
    h::Stored a, b;
    h::stored(d, layout, a, b);
    for (const h::Stored &s : {a, b})
        if (const char *why = h::factor_refusal(s); *why) return why;
    switch (path) {
        case HMR_GEMM_H06NTW32: {
            // A's range is rounded up to a dword: it must end with one.
            if (((a.rows - 1) * a.ld + a.cols) % 2) return "an odd span of A";
            if (ceil_div(d.m, 16) > h::kMaxGridYZ) return "more row tiles than its grid holds";
            return "";
        }
        case HMR_GEMM_H06W1_NARROW:
            return d.m > 16 ? "M > 16" : "";
        case HMR_GEMM_H06TZ_4A:
        case HMR_GEMM_H06FZ_N256: {
            // The transposed factor: M x padded(K) for A, K x padded(N) for B.
            const h::Stored t = path == HMR_GEMM_H06TZ_4A ? h::Stored{d.m, d.k, h::padded(d.k)}
                                                          : h::Stored{d.k, d.n, h::padded(d.n)};
            if (const char *why = h::factor_refusal(t); *why) return why;
            break;
        }
        default:
            break;
    }
    const int64_t rows = ceil_div(d.m, 128), cols = ceil_div(d.n, h::block_cols(path));
    int64_t blocks = 0;
    if (rows > h::kMaxGridYZ) return "more row tiles than its grid holds";
    if (!checked_mul(rows, cols, blocks) || blocks > INT32_MAX) return "more tiles than its grid holds";
    return "";
}

// The measured preferences (rocm-exp/LOG-H.md, "H 系列的完整配置表", F32 accumulation), as
// ordered rules; where a rule's path refuses the description, the selection falls to the generic
// path, so each rule names a path that takes the shapes it covers (the any-shape kernels take
// every size; the tile kernels are chosen only on their multiples). Measured: the squares
// 1000-8193 (8000 for NN only), and M = 1, 16, 128 with N, K in {4096, 11008}; between these
// the rules extrapolate by the inequalities below (M5 measures more). The paths with a workspace
// or one launch per batch (split-K, pre-transposition) are taken for one batch only, as measured.
int32_t h_preferred(const hmr_gemm_desc &d, int layout) {
    if (d.dtype != HMR_F16 && d.dtype != HMR_BF16) return -1;
    const bool one = d.batch == 1;
    // Tile multiples of the 128 x 128 and 128 x 256 kernels (M of 128, N of 128 / 256, K of 32).
    const bool t128 = d.m % 128 == 0 && d.n % 128 == 0 && d.k % 32 == 0;
    const bool t256 = t128 && d.n % 256 == 0;
    // Up to about 1000^2 outputs the 128 x 128 kernels win (1000^3, 1024^3), from 2048^2 on
    // the 128 x 256 ones.
    const bool small = d.m * d.n <= 1024 * 1024;
    // M = 128 with N, K in {4096, 11008}: the double-buffered kernels, except where the
    // configuration table names another.
    const bool m128 = d.m == 128 && t128;
    // Columns that 128-wide tiles cover with less padding than 256-wide ones (N mod 256 in
    // 1..128, e.g. 64, 128, 384): the 128 x 128 edge kernels, 1.2-1.8x faster there at M of
    // 16384 and 131072, K of 256 and 2048 (review round 142). NT only for K < 1024: from there
    // the 128 x 256 edge kernel is as fast or up to 10% faster at M = 131072 (below, at worst
    // 2.6% slower: 131072 x 64 x 512).
    const bool narrow = d.n % 256 != 0 && d.n % 256 <= 128;
    const bool wide_n = d.n == 11008 && d.k == 4096, long_k = d.n == 4096 && d.k == 11008;
    switch (layout) {
        case HMR_GEMM_NN:
            if (d.m <= 16) {
                // One launch of split-K at 1 x 4096 x 4096 only (twice as fast hot, even cold);
                // the 128 x 128 edge kernel at the other M <= 16 shapes.
                return one && d.m == 1 && d.n == 4096 && d.k == 4096 ? HMR_GEMM_H06W1_NARROW : HMR_GEMM_H06C_4A;
            }
            if (m128) return wide_n ? HMR_GEMM_H04A : HMR_GEMM_H06NNG_4X;
            if (small) return t128 ? HMR_GEMM_H04A : HMR_GEMM_H06C_4A;
            if (narrow) return HMR_GEMM_H06C_4A;
            if (t256) {
                // BF16 2048^3: H2a-g4l.
                return d.dtype == HMR_BF16 && d.m * d.n <= 2048 * 2048 ? HMR_GEMM_H02G_G4L : HMR_GEMM_H02N256;
            }
            return HMR_GEMM_H06C_N256;
        case HMR_GEMM_NT:
            // M <= 16: the narrow tile (N of 32, K of 256), else the 128 x 128 edge kernel.
            if (d.m <= 16) return d.n % 32 == 0 && d.k % 256 == 0 ? HMR_GEMM_H06NTW32 : HMR_GEMM_H06FK_NT4A;
            if (m128) return long_k ? HMR_GEMM_H06NTN_4A : HMR_GEMM_H06NTGP_4X;
            if (small) return t128 ? HMR_GEMM_H06NTN_4A : HMR_GEMM_H06FK_NT4A;
            if (narrow && d.k < 1024) return HMR_GEMM_H06FK_NT4A;
            // 8193^3: B pre-transposed (its odd leading dimension is slow to read; review 4 of
            // the H series), only at that shape.
            if (one && d.m == 8193 && d.n == 8193 && d.k == 8193) return HMR_GEMM_H06FZ_N256;
            // 2048^3 and 4096^3 (K of 64); 8192^3 and the non-multiples: the edge kernel.
            if (t256 && d.k % 64 == 0 && d.m * d.n <= 4096 * 4096) return HMR_GEMM_H06NTK_N256;
            return HMR_GEMM_H06FK_NT256;
        case HMR_GEMM_TN:
            if (d.m <= 16) {
                // N = 4096: A pre-transposed for the NN edge kernel; N = 11008 (K = 4096): the
                // double-buffered kernel for any M; elsewhere the 128 x 128 edge kernel.
                if (one && d.n == 4096 && (d.k == 4096 || d.k == 11008)) return HMR_GEMM_H06TZ_4A;
                if (wide_n) return HMR_GEMM_H06TNGB_4X;
                return HMR_GEMM_H06FK_TN4A;
            }
            if (m128) return wide_n ? HMR_GEMM_H06TN_4A : HMR_GEMM_H06TNG_4X;
            if (small) return t128 ? HMR_GEMM_H06TN_4A : HMR_GEMM_H06FK_TN4A;
            if (narrow) return HMR_GEMM_H06FK_TN4A;
            return t256 ? HMR_GEMM_H06TN_N256 : HMR_GEMM_H06FK_TN256;
        default:
            return -1;
    }
}

int64_t h_workspace(int32_t path, const hmr_gemm_desc &d) {
    int64_t elements = 0, bytes = 0;
    switch (path) {
        case HMR_GEMM_H06W1_NARROW: {
            // The slices' f32 partials, then a counter per column block.
            int slices = 0, per = 0;
            h::kw1::split(d, slices, per);
            if (!checked_mul(slices, d.m, elements) || !checked_mul(elements, d.n, elements) ||
                !checked_add(elements, ceil_div(d.n, h::kw1::kBn), elements) || !checked_mul(elements, 4, bytes)) {
                return -1;
            }
            return bytes;
        }
        case HMR_GEMM_H06TZ_4A:
            return checked_mul(d.m, h::padded(d.k), elements) && checked_mul(elements, 2, bytes) ? bytes : -1;
        case HMR_GEMM_H06FZ_N256:
            return checked_mul(d.k, h::padded(d.n), elements) && checked_mul(elements, 2, bytes) ? bytes : -1;
        default:
            return 0;
    }
}

void h_launch(int32_t path, const hmr_gemm_desc &d, void *workspace, hipStream_t stream) {
    const int layout = h::info(path)->layout;
    h::Stored a, b;
    h::stored(d, layout, a, b);
    const int M = static_cast<int>(d.m), N = static_cast<int>(d.n), K = static_cast<int>(d.k);
    const int lda = static_cast<int>(a.ld), ldb = static_cast<int>(b.ld);
    const uint16_t *A = static_cast<const uint16_t *>(d.a), *B = static_cast<const uint16_t *>(d.b);
    const dim3 tiles(static_cast<unsigned>(ceil_div(d.n, h::block_cols(path))),
                     static_cast<unsigned>(ceil_div(d.m, 128)));
    auto go = [&](auto kernel, dim3 block) {
        for (int64_t z = 0; z < d.batch; z += h::kMaxGridYZ) {
            const dim3 grid(tiles.x, tiles.y, static_cast<unsigned>(std::min(d.batch - z, h::kMaxGridYZ)));
            kernel<<<grid, block, 0, stream>>>(A, B, d, M, N, K, lda, ldb, z);
        }
    };
    auto typed = [&](auto dtype, auto scaled) {
        constexpr int DT = decltype(dtype)::v;
        constexpr bool SC = decltype(scaled)::v != 0;
        const dim3 b4(32, 4), b8(32, 8);
        switch (path) {
            case HMR_GEMM_H04A: return go(h::k4a::kernel<DT, SC, HMR_GEMM_NN, false>, b4);
            case HMR_GEMM_H06NTN_4A: return go(h::k4a::kernel<DT, SC, HMR_GEMM_NT, false>, b4);
            case HMR_GEMM_H06TN_4A: return go(h::k4a::kernel<DT, SC, HMR_GEMM_TN, false>, b4);
            case HMR_GEMM_H06C_4A: return go(h::k4a::kernel<DT, SC, HMR_GEMM_NN, true>, b4);
            case HMR_GEMM_H06FK_NT4A: return go(h::k4a::kernel<DT, SC, HMR_GEMM_NT, true>, b4);
            case HMR_GEMM_H06FK_TN4A: return go(h::k4a::kernel<DT, SC, HMR_GEMM_TN, true>, b4);
            case HMR_GEMM_H02G_G4L: return go(h::kg::kernel<DT, SC, HMR_GEMM_NN, 4>, b4);
            case HMR_GEMM_H02N256: return go(h::kg::kernel<DT, SC, HMR_GEMM_NN, 8>, b8);
            case HMR_GEMM_H06TN_N256: return go(h::kg::kernel<DT, SC, HMR_GEMM_TN, 8>, b8);
            case HMR_GEMM_H06C_N256: return go(h::kg::edge<DT, SC, HMR_GEMM_NN>, b8);
            case HMR_GEMM_H06FK_TN256: return go(h::kg::edge<DT, SC, HMR_GEMM_TN>, b8);
            case HMR_GEMM_H06NTK_N256: return go(h::knt::kernel<DT, SC, false>, b8);
            case HMR_GEMM_H06FK_NT256: return go(h::knt::kernel<DT, SC, true>, b8);
            case HMR_GEMM_H06NNG_4X: return go(h::k4x::kernel<DT, SC, HMR_GEMM_NN, false>, b4);
            case HMR_GEMM_H06NTGP_4X: return go(h::k4x::kernel<DT, SC, HMR_GEMM_NT, false>, b4);
            case HMR_GEMM_H06TNG_4X: return go(h::k4x::kernel<DT, SC, HMR_GEMM_TN, false>, b4);
            case HMR_GEMM_H06TNGB_4X: return go(h::k4x::kernel<DT, SC, HMR_GEMM_TN, true>, b4);
            case HMR_GEMM_H06NTW32: {
                const dim3 grid0(static_cast<unsigned>(d.n / h::kw32::NB), static_cast<unsigned>(ceil_div(d.m, 16)));
                for (int64_t z = 0; z < d.batch; z += h::kMaxGridYZ) {
                    const dim3 grid(grid0.x, grid0.y, static_cast<unsigned>(std::min(d.batch - z, h::kMaxGridYZ)));
                    h::kw32::kernel<DT, SC><<<grid, b4, 0, stream>>>(A, B, d, M, N, K, lda, ldb, z);
                }
                return;
            }
            case HMR_GEMM_H06W1_NARROW: {
                // One launch per batch, the workspace reused in stream order; the counters start
                // at zero, and each launch's finishing blocks reset theirs.
                int slices = 0, per = 0;
                h::kw1::split(d, slices, per);
                const int cols = static_cast<int>(ceil_div(d.n, h::kw1::kBn));
                float *P = static_cast<float *>(workspace);
                unsigned *count = reinterpret_cast<unsigned *>(P + int64_t{slices} * d.m * d.n);
                h::kw1::zero<<<1, 256, 0, stream>>>(count, cols);
                for (int64_t bi = 0; bi < d.batch; ++bi) {
                    h::kw1::kernel<DT, SC><<<dim3(cols, slices), dim3(32, h::kw1::kW), 0, stream>>>(
                        A, B, d, P, count, M, N, K, lda, ldb, per, bi);
                }
                return;
            }
            case HMR_GEMM_H06TZ_4A:
            case HMR_GEMM_H06FZ_N256: {
                // One batch at a time: its factor transposed into the workspace, then the NN
                // kernel (h06c_4a, h06c_n256) on it, the workspace reused in stream order.
                uint16_t *W = static_cast<uint16_t *>(workspace);
                const bool tz = path == HMR_GEMM_H06TZ_4A;
                const int ldw = static_cast<int>(h::padded(tz ? d.k : d.n));
                hmr_gemm_desc nn = d;
                if (tz) {
                    nn.a = W;
                    nn.a_strides[0] = 0;
                } else {
                    nn.b = W;
                    nn.b_strides[0] = 0;
                }
                const dim3 grid(tiles.x, tiles.y, 1);
                for (int64_t bi = 0; bi < d.batch; ++bi) {
                    if (tz) {
                        const unsigned threads = unsigned(M) * unsigned(ldw / 8);
                        h::transpose_a<<<(threads + 255) / 256, 256, 0, stream>>>(A + bi * d.a_strides[0], W, M, K, lda, ldw);
                        h::k4a::kernel<DT, SC, HMR_GEMM_NN, true><<<grid, b4, 0, stream>>>(W, B, nn, M, N, K, ldw, ldb, bi);
                    } else {
                        h::transpose_any<<<dim3((K + 63) / 64, (N + 63) / 64), 256, 0, stream>>>(
                            B + bi * d.b_strides[0], W, N, K, ldb, ldw);
                        h::kg::edge<DT, SC, HMR_GEMM_NN><<<grid, b8, 0, stream>>>(A, W, nn, M, N, K, lda, ldw, bi);
                    }
                }
                return;
            }
            default: return;
        }
    };
    if (d.dtype == HMR_F16) {
        d.scaled ? typed(Tag<HMR_F16>{}, Tag<1>{}) : typed(Tag<HMR_F16>{}, Tag<0>{});
    } else {
        d.scaled ? typed(Tag<HMR_BF16>{}, Tag<1>{}) : typed(Tag<HMR_BF16>{}, Tag<0>{});
    }
}

}  // namespace hmr
