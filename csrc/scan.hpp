// Deterministic exclusive scans of counts across the device, for kernels that compact elements
// in order (randint's rejection sampling; the coordinates of a mask in step 4b). A device-wide
// scan has three passes: each tile of a block's threads counts its items, scan_counts_kernel
// turns the counts into each tile's start, and each tile writes its items from its start plus
// the scan of its threads' counts. Every sum is an exact integer sum, so the result doesn't
// depend on the order of the additions.
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

}  // namespace
}  // namespace hmr
