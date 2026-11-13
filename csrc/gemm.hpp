// Shared by the matrix product sources (csrc/gemm.hip, csrc/gemm_s.hip): the writeback of
// cpu::gemm, and the interface of the S series' paths, which gemm.hip selects and launches.
#pragma once

#include <string>

#include "elem.hpp"
#include "kernel.hpp"

namespace hmr {
namespace {

// The type a dtype's products accumulate in on the generic path (and are staged in): the
// computation type for floats, 64 bits for integers.
template <int DT>
struct Acc {
    using A = int64_t;
};
template <>
struct Acc<HMR_F32> {
    using A = float;
};
template <>
struct Acc<HMR_F64> {
    using A = double;
};

// The output element of the product `p`, with the input element at `c` (read only if needed),
// in cpu::gemm's steps: alpha * p (skipped for alpha == 1), then beta * c + that (beta * c
// skipped for beta == 1), each rounded; integers wrap.
template <int DT>
__device__ inline typename Elem<DT>::T write(const hmr_gemm_desc &d, typename Acc<DT>::A p,
                                             const typename Elem<DT>::T *c) {
    if constexpr (DT == HMR_F32 || DT == HMR_F64) {
        using C = typename Comp<DT>::C;
        if (!d.scaled) return down<DT>(p);
        C alpha = static_cast<C>(d.alpha), beta = static_cast<C>(d.beta);
        C a = alpha == C(1) ? p : p * alpha;
        if (!d.reads_input) return down<DT>(a);
        C v = up<DT>(*c);
        return down<DT>(beta == C(1) ? v + a : v * beta + a);
    } else {
        using U = uint64_t;
        U r = static_cast<U>(p);
        if (d.scaled) {
            U alpha = static_cast<U>(d.alpha_int), beta = static_cast<U>(d.beta_int);
            U a = d.alpha_int == 1 ? r : r * alpha;
            if (d.reads_input) {
                U v = static_cast<U>(static_cast<int64_t>(*c));
                r = d.beta_int == 1 ? v + a : v * beta + a;
            } else {
                r = a;
            }
        }
        return from_i64<DT>(static_cast<int64_t>(r));
    }
}

// The output of a product: element (i, j) of output batch `ob`, written back as cpu::gemm does.
struct Direct {
    hmr_gemm_desc d;
    int64_t ob;
    __device__ void put(int64_t i, int64_t j, float v) const {
        const float *c = d.reads_input ? static_cast<const float *>(d.c) + ob * d.c_strides[0] +
                                             i * d.c_strides[1] + j * d.c_strides[2]
                                       : nullptr;
        static_cast<float *>(d.out)[ob * d.out_strides[0] + i * d.out_strides[1] +
                                    j * d.out_strides[2]] = write<HMR_F32>(d, v, c);
    }
};

// The layout of the factors (HMR_GEMM_NN ...): a unit stride along a dimension of size 0 or 1
// is any stride.
inline int layout_of(const hmr_gemm_desc &d) {
    bool an = d.a_strides[2] == 1 || d.k <= 1, at = d.a_strides[1] == 1 || d.m <= 1;
    bool bn = d.b_strides[2] == 1 || d.n <= 1, bt = d.b_strides[1] == 1 || d.k <= 1;
    if (an && bn) return HMR_GEMM_NN;
    if (an && bt) return HMR_GEMM_NT;
    if (at && bn) return HMR_GEMM_TN;
    if (at && bt) return HMR_GEMM_TT;
    return HMR_GEMM_STRIDED;
}

}  // namespace

// The S series (csrc/gemm_s.hip): F32 paths ported from rocm-exp's measured programs.
//
// s_refusal: why `path` doesn't take `d` of layout `layout` ("": it does).
// s_preferred: the path the measured rules prefer for `d` (they may still refuse it), or -1.
// s_workspace: the bytes of workspace `path` needs for `d`, or -1 if the count overflows.
// s_launch: queues `d` along `path` (which takes it) on `stream`, with that workspace.
std::string s_refusal(int32_t path, const hmr_gemm_desc &d, int layout);
int32_t s_preferred(const hmr_gemm_desc &d, int layout);
int64_t s_workspace(int32_t path, const hmr_gemm_desc &d);
void s_launch(int32_t path, const hmr_gemm_desc &d, void *workspace, hipStream_t stream);

}  // namespace hmr
