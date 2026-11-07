// Random sampling on the device with the Philox4x32-10 generator of src/ops/random.rs: the same
// counter, key, words and conversions, so that uniform samples, integers and Dropout's masks
// equal the CPU's bit for bit. Normal samples use the device library's log, cos and sin.
//
// Integer sampling rejects words below a threshold, so an element's word depends on how many
// words before it were rejected: each batch scans the accepted words' flags across the device
// (scan.hpp: tile counts, an exclusive scan of the counts, then a scan inside each tile), in the
// CPU's word order.
#include <cmath>

#include "elem.hpp"
#include "kernel.hpp"
#include "scan.hpp"

using hmr::guarded;

namespace hmr {
namespace {

struct Words4 {
    uint32_t w[4];
};

// One Philox4x32-10 block (random.rs: philox).
__device__ inline Words4 philox(uint64_t n, uint64_t seed) {
    constexpr uint64_t kM0 = 0xD2511F53u;
    constexpr uint64_t kM1 = 0xCD9E8D57u;
    constexpr uint32_t kW0 = 0x9E3779B9u;
    constexpr uint32_t kW1 = 0xBB67AE85u;
    uint32_t c0 = static_cast<uint32_t>(n);
    uint32_t c1 = static_cast<uint32_t>(n >> 32);
    uint32_t c2 = 0;
    uint32_t c3 = 0;
    uint32_t k0 = static_cast<uint32_t>(seed);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k0 += kW0;
            k1 += kW1;
        }
        uint64_t p0 = kM0 * c0;
        uint64_t p1 = kM1 * c2;
        uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ c1 ^ k0;
        uint32_t n1 = static_cast<uint32_t>(p1);
        uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ c3 ^ k1;
        uint32_t n3 = static_cast<uint32_t>(p0);
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    return Words4{{c0, c1, c2, c3}};
}

// A uniform f64 in [0, 1) from 53 random bits (random.rs: unit_f64); exact.
__device__ inline double unit_f64(uint32_t hi, uint32_t lo) {
    uint64_t bits = (static_cast<uint64_t>(hi) << 21) ^ (static_cast<uint64_t>(lo) >> 11);
    return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
}

// A uniform value in [0, 1) from the top `bits` bits of `x` (random.rs: unit_bits); exact.
__device__ inline double unit_bits(uint32_t x, int bits) {
    return static_cast<double>(x >> (32 - bits)) / static_cast<double>(1ull << bits);
}

template <int DT>
__global__ void uniform_kernel(typename Elem<DT>::T *out, int64_t n, uint64_t seed,
                               uint64_t offset, double from, double to) {
    constexpr int kPerBlock = DT == HMR_F64 ? 2 : 4;
    const int bits = DT == HMR_F16 ? 11 : DT == HMR_BF16 ? 8 : 24;
    int64_t blocks = (n + kPerBlock - 1) / kPerBlock;
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; b < blocks;
         b += step) {
        Words4 r = philox(offset + static_cast<uint64_t>(b), seed);
        for (int j = 0; j < kPerBlock; ++j) {
            int64_t i = b * kPerBlock + j;
            if (i >= n) break;
            double u = kPerBlock == 2 ? unit_f64(r.w[2 * j], r.w[2 * j + 1])
                                      : unit_bits(r.w[j], bits);
            double x = from + u * (to - from);
            typename Elem<DT>::T v = from_f64<DT>(x);
            // Rounding can reach the excluded upper bound; it becomes the lower bound then.
            if (from < to && to_f64<DT>(v) >= to) v = from_f64<DT>(from);
            out[i] = v;
        }
    }
}

template <int DT>
__global__ void normal_kernel(typename Elem<DT>::T *out, int64_t n, uint64_t seed,
                              uint64_t offset, double mean, double std) {
    int64_t blocks = (n + 1) / 2;
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; b < blocks;
         b += step) {
        Words4 w = philox(offset + static_cast<uint64_t>(b), seed);
        double u1 = 1.0 - unit_f64(w.w[0], w.w[1]);
        double u2 = unit_f64(w.w[2], w.w[3]);
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;
        out[2 * b] = from_f64<DT>(mean + std * r * cos(theta));
        if (2 * b + 1 < n) out[2 * b + 1] = from_f64<DT>(mean + std * r * sin(theta));
    }
}

// Word `j` (0 or 1) of a block: [w0 w1] or [w2 w3] as one 64-bit word (random.rs: sample_int).
__device__ inline uint64_t int_word(const Words4 &b, int j) {
    return (static_cast<uint64_t>(b.w[2 * j]) << 32) | b.w[2 * j + 1];
}

__device__ inline int64_t int_value(uint64_t word, int64_t low, uint64_t range) {
    return static_cast<int64_t>(static_cast<uint64_t>(low) + word % range);
}

// Without rejection, word w fills element w.
__global__ void int_direct_kernel(int64_t *out, int64_t n, uint64_t seed, uint64_t offset,
                                  int64_t low, uint64_t range) {
    int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; w < n;
         w += step) {
        Words4 b = philox(offset + static_cast<uint64_t>(w / 2), seed);
        out[w] = int_value(int_word(b, static_cast<int>(w % 2)), low, range);
    }
}

constexpr int kTile = HMR_RANDOM_INT_TILE;

// The accepted words of the block of thread `t` in tile `tile`, as two flags.
__device__ inline void int_flags(int64_t tile, uint64_t blocks, uint64_t seed, uint64_t offset,
                                 uint64_t threshold, Words4 &words, uint32_t &f0, uint32_t &f1) {
    uint64_t b = static_cast<uint64_t>(tile) * kTile + threadIdx.x;
    f0 = f1 = 0;
    if (b < blocks) {
        words = philox(offset + b, seed);
        f0 = int_word(words, 0) >= threshold;
        f1 = int_word(words, 1) >= threshold;
    }
}

// Pass 1: the number of accepted words of each tile.
__global__ void int_count_kernel(uint64_t *counts, uint64_t blocks, uint64_t seed,
                                 uint64_t offset, uint64_t threshold) {
    Words4 words;
    uint32_t f0, f1;
    int_flags(blockIdx.x, blocks, seed, offset, threshold, words, f0, f1);
    uint64_t total;
    block_exclusive_scan<kTile>(f0 + f1, total);
    if (threadIdx.x == 0) counts[blockIdx.x] = total;
}

// Pass 3: each accepted word goes to its position among the accepted words, if out has one.
__global__ void int_write_kernel(int64_t *out, int64_t n, const uint64_t *starts,
                                 uint64_t blocks, uint64_t seed, uint64_t offset,
                                 uint64_t threshold, int64_t low, uint64_t range) {
    Words4 words;
    uint32_t f0, f1;
    int_flags(blockIdx.x, blocks, seed, offset, threshold, words, f0, f1);
    uint64_t total;
    uint64_t pos = starts[blockIdx.x] + block_exclusive_scan<kTile>(f0 + f1, total);
    if (f0 && pos < static_cast<uint64_t>(n)) out[pos] = int_value(int_word(words, 0), low, range);
    pos += f0;
    if (f1 && pos < static_cast<uint64_t>(n)) out[pos] = int_value(int_word(words, 1), low, range);
}

template <class B>
bool for_float(int dtype, B &&body) {
    switch (dtype) {
        case HMR_F16: return body(std::integral_constant<int, HMR_F16>{});
        case HMR_BF16: return body(std::integral_constant<int, HMR_BF16>{});
        case HMR_F32: return body(std::integral_constant<int, HMR_F32>{});
        case HMR_F64: return body(std::integral_constant<int, HMR_F64>{});
        default: return false;
    }
}

// Checks that `out` is a flat description and returns its first element and length.
hmr_status flat_output(const hmr_tensor_desc &out, int64_t &n) {
    if (hmr_status s = check_desc(out, n)) return s;
    if (!is_flat(out)) return fail(HMR_INVALID, "random sampling needs a contiguous output");
    return HMR_OK;
}

}  // namespace
}  // namespace hmr

extern "C" {

hmr_status hmr_random_uniform(hmr_context_t ctx, const hmr_tensor_desc *out, uint64_t seed,
                              uint64_t offset, double from, double to) {
    return guarded([&] {
        int64_t n = 0;
        if (hmr_status s = hmr::flat_output(*out, n)) return s;
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "uniform", [&](hipStream_t s) {
            return hmr::for_float(out->dtype, [&](auto t) {
                constexpr int DT = decltype(t)::value;
                using T = typename hmr::Elem<DT>::T;
                T *data = static_cast<T *>(out->data) + out->offset;
                int64_t blocks = (n + (DT == HMR_F64 ? 1 : 3)) / (DT == HMR_F64 ? 2 : 4);
                hipLaunchKernelGGL((hmr::uniform_kernel<DT>), hmr::blocks_for(blocks),
                                   dim3(hmr::kThreads), 0, s, data, n, seed, offset, from, to);
                return true;
            });
        });
    });
}

hmr_status hmr_random_normal(hmr_context_t ctx, const hmr_tensor_desc *out, uint64_t seed,
                             uint64_t offset, double mean, double std) {
    return guarded([&] {
        int64_t n = 0;
        if (hmr_status s = hmr::flat_output(*out, n)) return s;
        if (n == 0) return HMR_OK;
        return hmr::launch_on(ctx, "normal", [&](hipStream_t s) {
            return hmr::for_float(out->dtype, [&](auto t) {
                constexpr int DT = decltype(t)::value;
                using T = typename hmr::Elem<DT>::T;
                T *data = static_cast<T *>(out->data) + out->offset;
                hipLaunchKernelGGL((hmr::normal_kernel<DT>), hmr::blocks_for((n + 1) / 2),
                                   dim3(hmr::kThreads), 0, s, data, n, seed, offset, mean, std);
                return true;
            });
        });
    });
}

hmr_status hmr_random_int(hmr_context_t ctx, const hmr_tensor_desc *out, uint64_t seed,
                          uint64_t offset, uint64_t blocks, int64_t low, uint64_t range,
                          uint64_t threshold, void *workspace) {
    return guarded([&] {
        int64_t n = 0;
        if (hmr_status s = hmr::flat_output(*out, n)) return s;
        if (out->dtype != HMR_I64) return hmr::fail(HMR_INVALID, "randint writes I64");
        if (range == 0) return hmr::fail(HMR_INVALID, "randint needs a nonempty range");
        if (threshold != 0 && workspace == nullptr) {
            return hmr::fail(HMR_INVALID, "randint with rejection needs a workspace");
        }
        if (n == 0 || blocks == 0) return HMR_OK;
        auto *data = static_cast<int64_t *>(out->data) + out->offset;
        return hmr::launch_on(ctx, "randint", [&](hipStream_t s) {
            if (threshold == 0) {
                hipLaunchKernelGGL(hmr::int_direct_kernel, hmr::blocks_for(n), dim3(hmr::kThreads),
                                   0, s, data, n, seed, offset, low, range);
                return true;
            }
            int64_t tiles =
                static_cast<int64_t>((blocks + HMR_RANDOM_INT_TILE - 1) / HMR_RANDOM_INT_TILE);
            if (tiles > INT32_MAX) return false;
            auto *counts = static_cast<uint64_t *>(workspace);
            uint64_t *starts = counts + tiles;
            uint64_t *total = starts + tiles;
            hipLaunchKernelGGL(hmr::int_count_kernel, dim3(static_cast<unsigned>(tiles)),
                               dim3(hmr::kTile), 0, s, counts, blocks, seed, offset, threshold);
            hipLaunchKernelGGL(hmr::scan_counts_kernel, dim3(1), dim3(hmr::kScanThreads), 0, s,
                               counts, starts, tiles, total);
            hipLaunchKernelGGL(hmr::int_write_kernel, dim3(static_cast<unsigned>(tiles)),
                               dim3(hmr::kTile), 0, s, data, n, starts, blocks, seed, offset,
                               threshold, low, range);
            return true;
        });
    });
}

}  // extern "C"
