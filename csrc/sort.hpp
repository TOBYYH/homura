// A stable least-significant-digit radix sort of (key, value) pairs of 64-bit words, for the
// kernels that group elements by their target (index_add, scatter_add, the backward of
// max_pool2d): after the sort, the pairs with equal keys are adjacent and keep their order, so a
// kernel can add each group in the order the CPU adds it, without atomic floating point
// additions.
//
// Each pass sorts by 8 bits of the key, from the lowest, and only as many passes run as the
// largest key needs. A pass has three kernels: each tile of kSortTile pairs counts its digits,
// device_exclusive_scan (scan.hpp) turns the counts, digit by digit and tile by tile, into each
// tile's first position for each digit, and each tile writes its pairs there. Within a tile the
// pairs are placed in their order: a round of 256 pairs (one per thread) finds, in each wave,
// the lanes with the same digit (eight ballots), and the waves' counts are added up in wave
// order. The positions are exact integer sums, so the result doesn't depend on the scheduling.
#pragma once

#include "scan.hpp"

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "sort.hpp assumes waves of 32 lanes (gfx12, DESIGN D24)"
#endif

namespace hmr {
namespace {

constexpr int kSortThreads = 256;
constexpr int kSortRounds = 8;
constexpr int64_t kSortTile = static_cast<int64_t>(kSortThreads) * kSortRounds;
constexpr int kRadix = 256;
constexpr int kWave = 32;
constexpr int kSortWaves = kSortThreads / kWave;
static_assert(kRadix == kSortThreads, "one digit per thread");

inline int64_t sort_tiles(int64_t n) { return ceil_div(n, kSortTile); }

// The words of a sort's workspace for `n` >= 0 pairs into `words`: two buffers of keys and of
// values, the counts of every digit in every tile, and the scan's auxiliary words; false if the
// number overflows int64_t. The counts can't overflow (kRadix / kSortTile < 1), only 4 * n and
// the sums.
inline bool sort_workspace_words(int64_t n, int64_t &words) {
    int64_t counts = kRadix * sort_tiles(n), pairs = 0, sum = 0;
    return checked_mul(4, n, pairs) && checked_add(pairs, counts, sum) &&
           checked_add(sum, scan_aux_words(counts), words);
}

// The passes for keys below `bound`: one per 8 bits of bound - 1.
inline int sort_passes(uint64_t bound) {
    int bits = 0;
    for (uint64_t m = bound > 0 ? bound - 1 : 0; m != 0; m >>= 1) ++bits;
    return (bits + 7) / 8;
}

// A sort's workspace: the caller writes the pairs to keys[0] and vals[0].
struct SortSpace {
    uint64_t *keys[2];
    uint64_t *vals[2];
    uint64_t *counts;
    uint64_t *aux;
    int64_t n;
    int64_t tiles;
};

inline SortSpace sort_space(uint64_t *ws, int64_t n) {
    SortSpace sp;
    sp.n = n;
    sp.tiles = sort_tiles(n);
    sp.keys[0] = ws;
    sp.keys[1] = ws + n;
    sp.vals[0] = ws + 2 * n;
    sp.vals[1] = ws + 3 * n;
    sp.counts = ws + 4 * n;
    sp.aux = sp.counts + kRadix * sp.tiles;
    return sp;
}

// counts[d * tiles + tile] = the number of keys of the tile whose digit (8 bits from `shift`) is d.
__global__ void radix_count_kernel(const uint64_t *keys, int64_t n, int shift, uint64_t *counts,
                                   int64_t tiles) {
    __shared__ uint32_t hist[kRadix];
    int t = threadIdx.x;
    hist[t] = 0;
    __syncthreads();
    int64_t base = static_cast<int64_t>(blockIdx.x) * kSortTile;
    for (int r = 0; r < kSortRounds; ++r) {
        int64_t i = base + r * kSortThreads + t;
        if (i < n) atomicAdd(&hist[(keys[i] >> shift) & (kRadix - 1)], 1u);
    }
    __syncthreads();
    counts[static_cast<int64_t>(t) * tiles + blockIdx.x] = hist[t];
}

// Writes the pairs of each tile at the starts that the scan of the counts gave, in order.
__global__ void radix_scatter_kernel(const uint64_t *keys_in, const uint64_t *vals_in,
                                     uint64_t *keys_out, uint64_t *vals_out, int64_t n, int shift,
                                     const uint64_t *starts, int64_t tiles) {
    __shared__ uint64_t next[kRadix];
    __shared__ uint32_t wave_count[kSortWaves][kRadix];
    __shared__ uint32_t round_count[kRadix];
    int t = threadIdx.x;
    int lane = t % kWave;
    int wave = t / kWave;
    next[t] = starts[static_cast<int64_t>(t) * tiles + blockIdx.x];
    for (int w = 0; w < kSortWaves; ++w) wave_count[w][t] = 0;
    __syncthreads();
    uint64_t before = (uint64_t{1} << lane) - 1;
    int64_t base = static_cast<int64_t>(blockIdx.x) * kSortTile;
    for (int r = 0; r < kSortRounds; ++r) {
        int64_t i = base + r * kSortThreads + t;
        bool valid = i < n;
        uint64_t key = valid ? keys_in[i] : 0;
        uint64_t val = valid ? vals_in[i] : 0;
        uint32_t d = static_cast<uint32_t>(key >> shift) & (kRadix - 1);
        // The valid lanes of this wave whose digit is d.
        uint64_t peers = __ballot(valid);
        for (int b = 0; b < 8; ++b) {
            bool bit = (d >> b) & 1;
            uint64_t m = __ballot(bit);
            peers &= bit ? m : ~m;
        }
        uint32_t rank = __popcll(peers & before);
        if (valid && rank == 0) wave_count[wave][d] = __popcll(peers);
        __syncthreads();
        // Thread t turns the waves' counts of digit t into their starts within the round.
        uint32_t sum = 0;
        for (int w = 0; w < kSortWaves; ++w) {
            uint32_t c = wave_count[w][t];
            wave_count[w][t] = sum;
            sum += c;
        }
        round_count[t] = sum;
        __syncthreads();
        if (valid) {
            uint64_t at = next[d] + wave_count[wave][d] + rank;
            keys_out[at] = key;
            vals_out[at] = val;
        }
        __syncthreads();
        next[t] += round_count[t];
        for (int w = 0; w < kSortWaves; ++w) wave_count[w][t] = 0;
        __syncthreads();
    }
}

// Sorts the pairs in keys[0] / vals[0] stably by their keys, all below `bound`, and returns
// which buffer holds the result.
inline int radix_sort(const SortSpace &sp, uint64_t bound, hipStream_t s) {
    int cur = 0;
    int passes = sort_passes(bound);
    int64_t ncounts = kRadix * sp.tiles;
    for (int p = 0; p < passes; ++p) {
        int shift = 8 * p;
        hipLaunchKernelGGL(radix_count_kernel, dim3(static_cast<unsigned>(sp.tiles)),
                           dim3(kSortThreads), 0, s, sp.keys[cur], sp.n, shift, sp.counts,
                           sp.tiles);
        device_exclusive_scan(sp.counts, ncounts, sp.aux, s);
        hipLaunchKernelGGL(radix_scatter_kernel, dim3(static_cast<unsigned>(sp.tiles)),
                           dim3(kSortThreads), 0, s, sp.keys[cur], sp.vals[cur],
                           sp.keys[1 - cur], sp.vals[1 - cur], sp.n, shift, sp.counts, sp.tiles);
        cur = 1 - cur;
    }
    return cur;
}

}  // namespace
}  // namespace hmr
