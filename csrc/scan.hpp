// Deterministic exclusive scans of counts across the device, for kernels that compact elements
// in order (randint's rejection sampling, the coordinates of a mask, the radix sort of sort.hpp).
// A device-wide compaction has three passes: each tile of a block's threads counts its items, a
// scan turns the counts into each tile's start, and each tile writes its items from its start
// plus the scan of its threads' counts. Every sum is an exact integer sum, so the result doesn't
// depend on the order of the additions.
//
// scan_counts_kernel is one block, which walks the counts 1024 at a time: enough for randint's
// few counts per batch. device_exclusive_scan scans long arrays (millions of counts) with every
// block of the device: chunks of 4096 counts are scanned in parallel, their totals by
// scan_counts_kernel, and the chunk starts added back.
#pragma once

#include "kernel.hpp"

namespace hmr {
namespace {

constexpr int kScanThreads = 1024;

// The exclusive prefix sums of `v` over the threads of a block of THREADS threads, and their
// total in `total`. Every thread must call it.
template <int THREADS>
__device__ inline uint64_t block_exclusive_scan(uint64_t v, uint64_t &total) {
    __shared__ uint64_t s[THREADS];
    int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int d = 1; d < THREADS; d <<= 1) {
        uint64_t x = t >= d ? s[t - d] : 0;
        __syncthreads();
        s[t] += x;
        __syncthreads();
    }
    uint64_t inclusive = s[t];
    total = s[THREADS - 1];
    __syncthreads();
    return inclusive - v;
}

// One block of kScanThreads threads: starts[i] = counts[0] + ... + counts[i - 1] for the `n`
// counts, and their sum in *total.
__global__ void scan_counts_kernel(const uint64_t *counts, uint64_t *starts, int64_t n,
                                   uint64_t *total) {
    uint64_t carry = 0;
    for (int64_t base = 0; base < n; base += kScanThreads) {
        int64_t i = base + threadIdx.x;
        uint64_t v = i < n ? counts[i] : 0;
        uint64_t chunk;
        uint64_t prefix = block_exclusive_scan<kScanThreads>(v, chunk);
        if (i < n) starts[i] = carry + prefix;
        carry += chunk;
    }
    if (threadIdx.x == 0) *total = carry;
}

constexpr int kChunkItems = 4;
constexpr int64_t kChunk = static_cast<int64_t>(kScanThreads) * kChunkItems;

inline int64_t scan_chunks(int64_t n) { return ceil_div(n, kChunk); }

// The words of device_exclusive_scan's auxiliary memory for `n` counts.
inline int64_t scan_aux_words(int64_t n) { return 2 * scan_chunks(n) + 1; }

// Each block scans one chunk of kChunk counts in place (exclusive, from 0) and stores its total.
__global__ void scan_chunks_kernel(uint64_t *v, int64_t n, uint64_t *totals) {
    int64_t base = static_cast<int64_t>(blockIdx.x) * kChunk +
                   static_cast<int64_t>(threadIdx.x) * kChunkItems;
    uint64_t x[kChunkItems];
    uint64_t sum = 0;
    for (int k = 0; k < kChunkItems; ++k) {
        int64_t i = base + k;
        x[k] = i < n ? v[i] : 0;
        sum += x[k];
    }
    uint64_t total;
    uint64_t prefix = block_exclusive_scan<kScanThreads>(sum, total);
    for (int k = 0; k < kChunkItems; ++k) {
        int64_t i = base + k;
        if (i < n) v[i] = prefix;
        prefix += x[k];
    }
    if (threadIdx.x == 0) totals[blockIdx.x] = total;
}

__global__ void add_chunk_starts_kernel(uint64_t *v, int64_t n, const uint64_t *starts) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        v[i] += starts[i / kChunk];
    }
}

// Replaces the `n` counts at `v` by their exclusive prefix sums and stores their sum at
// aux[2 * scan_chunks(n)]; `aux` holds scan_aux_words(n) words.
inline void device_exclusive_scan(uint64_t *v, int64_t n, uint64_t *aux, hipStream_t s) {
    int64_t chunks = scan_chunks(n);
    uint64_t *totals = aux, *starts = aux + chunks, *total = aux + 2 * chunks;
    if (n == 0) {
        hipLaunchKernelGGL(scan_counts_kernel, dim3(1), dim3(kScanThreads), 0, s, totals, starts,
                           int64_t{0}, total);
        return;
    }
    hipLaunchKernelGGL(scan_chunks_kernel, dim3(static_cast<unsigned>(chunks)),
                       dim3(kScanThreads), 0, s, v, n, totals);
    hipLaunchKernelGGL(scan_counts_kernel, dim3(1), dim3(kScanThreads), 0, s, totals, starts,
                       chunks, total);
    if (chunks > 1) {
        hipLaunchKernelGGL(add_chunk_starts_kernel, blocks_for(n), dim3(kThreads), 0, s, v, n,
                           starts);
    }
}

}  // namespace
}  // namespace hmr
