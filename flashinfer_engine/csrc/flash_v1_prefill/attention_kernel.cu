#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

using namespace nvcuda;
using namespace torch::indexing;

// ------------------------------------------------------------------ //
// Warp reductions (same as v0 — fp32 softmax stays fp32)
// ------------------------------------------------------------------ //

__inline__ __device__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

// ------------------------------------------------------------------ //
// FlashAttention v1 kernel — bf16 input + Tensor Cores (wmma)
//
// Template parameters:
//   B_r      : rows per Q tile   (must be 16 — wmma M dimension)
//   B_c      : cols per KV tile  (must be 16 — wmma N dimension)
//   HEAD_DIM : attention head dimension (64 or 128)
//
// wmma tile shape used: M=16, N=16, K=16
//   QK^T: A=Q(16×HEAD_DIM), B=K^T(HEAD_DIM×16) → C=scores(16×16)
//         loop over K in steps of 16: HEAD_DIM/16 iterations
//   PV:   A=P(16×16),       B=V(16×HEAD_DIM)   → C=O(16×HEAD_DIM)
//         loop over N in steps of 16: HEAD_DIM/16 output tiles
//
// Each thread block handles one (batch-head, row-tile) pair.
// One warp (32 threads) per wmma operation.
// ------------------------------------------------------------------ //

template <int B_r, int B_c, int HEAD_DIM>
__global__ void flashattention_v1_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float*               __restrict__ out,
    int  seq_len,
    int  head_dim,
    bool causal)
{
    // ---- shared memory tiles (bf16 for Q/K/V, fp32 for O/softmax state) ----
    __shared__ __nv_bfloat16 q_tile[B_r][HEAD_DIM + 8];  // +8: avoid bank conflicts
    __shared__ __nv_bfloat16 k_tile[B_c][HEAD_DIM + 8];
    __shared__ __nv_bfloat16 v_tile[B_c][HEAD_DIM + 8];
    __shared__ float          o_tile[B_r][HEAD_DIM + 1];

    // softmax running stats (fp32 for precision)
    __shared__ float m_tile[B_r];      // row max
    __shared__ float l_tile[B_r];      // row normalizer
    __shared__ float alpha_tile[B_r];  // rescale factor

    // scores and softmax probabilities (fp32)
    __shared__ float scores[B_r][B_c];
    __shared__ float p[B_r][B_c];

    // ---- thread/block indexing ----
    int tid           = threadIdx.x;
    int warp_id       = tid / 32;
    int lane_id       = tid % 32;
    int warps_per_block = blockDim.x / 32;

    int bh        = blockIdx.y;
    int row_block = blockIdx.x;
    int row_start = row_block * B_r;
    int row_end   = min(row_start + B_r, seq_len);

    int base  = bh * seq_len * head_dim;
    float scale = rsqrtf((float)HEAD_DIM);

    // ---- load Q tile (bf16) and initialize O tile (fp32) ----
    for (int idx = tid; idx < B_r * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;
        int global_r = row_start + r;
        q_tile[r][d] = (global_r < seq_len)
            ? q[base + global_r * head_dim + d]
            : __float2bfloat16(0.0f);
        o_tile[r][d] = 0.0f;
    }

    // ---- initialize softmax state ----
    if (tid < B_r) {
        m_tile[tid]     = -INFINITY;
        l_tile[tid]     = 0.0f;
        alpha_tile[tid] = 0.0f;
    }

    __syncthreads();

    // ==================================================================
    // Main loop over K/V column tiles
    // ==================================================================
    for (int col_start = 0; col_start < seq_len; col_start += B_c) {
        if (causal && col_start > row_end - 1) continue;

        int col_end   = min(col_start + B_c, seq_len);
        int Bc_actual = col_end - col_start;

        // ---- load K and V tiles (bf16) ----
        for (int idx = tid; idx < B_c * HEAD_DIM; idx += blockDim.x) {
            int c = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int global_c = col_start + c;
            bool valid = (c < Bc_actual) && (global_c < seq_len);
            k_tile[c][d] = valid ? k[base + global_c * head_dim + d]
                                 : __float2bfloat16(0.0f);
            v_tile[c][d] = valid ? v[base + global_c * head_dim + d]
                                 : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // ============================================================
        // TODO 1: Compute scores = Q_tile @ K_tile^T using wmma
        //
        // Target: scores[B_r][B_c]  (fp32, already in shared memory)
        //
        // Approach:
        //   - Each warp computes one 16×16 output tile of scores
        //   - wmma fragment shapes: matrix_a(16,16,16,bf16,row_major)
        //                           matrix_b(16,16,16,bf16,col_major)
        //                           accumulator(16,16,16,float)
        //   - Loop over K in steps of 16 (HEAD_DIM/16 iterations)
        //   - Accumulate into fp32 accumulator fragment
        //   - Apply scale factor after wmma
        //   - Apply causal mask (set masked positions to -INFINITY)
        //   - Store result into scores[r][c] shared memory
        //
        // Hint: use wmma::fill_fragment, load_matrix_sync,
        //       mma_sync, store_matrix_sync
        // ============================================================

        __syncthreads();

        // ---- online softmax update (same as v0 — fp32, no changes needed) ----
        for (int r = warp_id; r < B_r; r += warps_per_block) {
            int global_r = row_start + r;

            float local_max = -INFINITY;
            if (global_r < seq_len)
                for (int c = lane_id; c < B_c; c += 32)
                    local_max = fmaxf(local_max, scores[r][c]);

            float row_max = warp_reduce_max(local_max);
            row_max = __shfl_sync(0xffffffff, row_max, 0);

            float m_old = m_tile[r];
            float m_new = fmaxf(m_old, row_max);
            float alpha  = (m_old != -INFINITY) ? expf(m_old - m_new) : 0.0f;

            float local_sum = 0.0f;
            if (global_r < seq_len) {
                for (int c = lane_id; c < B_c; c += 32) {
                    float val = (scores[r][c] != -INFINITY) ? expf(scores[r][c] - m_new) : 0.0f;
                    p[r][c]    = val;
                    local_sum += val;
                }
            }

            float row_sum = warp_reduce_sum(local_sum);

            if (lane_id == 0 && global_r < seq_len) {
                l_tile[r]     = alpha * l_tile[r] + row_sum;
                m_tile[r]     = m_new;
                alpha_tile[r] = alpha;
            }
        }

        __syncthreads();

        // ============================================================
        // TODO 2: Update output accumulator O_tile += P_tile @ V_tile
        //         using wmma
        //
        // Target: o_tile[B_r][HEAD_DIM]  (fp32)
        //
        // Approach:
        //   - P_tile is bf16-converted from p[B_r][B_c] (fp32 → bf16)
        //   - V_tile is already bf16 in v_tile[B_c][HEAD_DIM]
        //   - Each warp computes a 16×16 output tile of O
        //   - Loop over HEAD_DIM in steps of 16 (HEAD_DIM/16 iterations)
        //   - Before accumulating: rescale existing o_tile by alpha_tile[r]
        //   - Store updated o_tile back to shared memory
        //
        // Hint: convert p[][] to bf16 before feeding into wmma;
        //       accumulator fragment is fp32; multiply by alpha after load
        // ============================================================

        __syncthreads();
    }

    // ---- normalize and write output ----
    for (int idx = tid; idx < B_r * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;
        int global_r = row_start + r;
        if (global_r < seq_len) {
            float denom = l_tile[r];
            out[base + global_r * head_dim + d] =
                denom > 0.0f ? o_tile[r][d] / denom : 0.0f;
        }
    }
}

// ------------------------------------------------------------------ //
// Host launcher
// ------------------------------------------------------------------ //

torch::Tensor custom_flash_attention_v1(
    torch::Tensor q,   // (batch, seq, hidden)  bf16
    torch::Tensor k,
    torch::Tensor v,
    int  num_heads,
    bool causal)
{
    TORCH_CHECK(q.is_cuda(),  "q must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "v1 kernel requires bf16 input");

    int batch      = q.size(0);
    int seq_len    = q.size(1);
    int hidden_dim = q.size(2);
    int head_dim   = hidden_dim / num_heads;

    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "v1 kernel supports head_dim 64 or 128");

    // reshape to (batch*heads, seq, head_dim)
    auto to_bh = [&](torch::Tensor t) {
        return t.view({batch, seq_len, num_heads, head_dim})
                .permute({0, 2, 1, 3}).contiguous()
                .view({batch * num_heads, seq_len, head_dim});
    };

    auto q_bh = to_bh(q);
    auto k_bh = to_bh(k);
    auto v_bh = to_bh(v);

    // output is fp32 (matches v0 interface)
    auto out_bh = torch::zeros(
        {batch * num_heads, seq_len, head_dim},
        q.options().dtype(torch::kFloat32));

    constexpr int B_r = 16;
    constexpr int B_c = 16;

    dim3 block(256);
    dim3 grid((seq_len + B_r - 1) / B_r, batch * num_heads);

    auto q_ptr = reinterpret_cast<const __nv_bfloat16*>(q_bh.data_ptr());
    auto k_ptr = reinterpret_cast<const __nv_bfloat16*>(k_bh.data_ptr());
    auto v_ptr = reinterpret_cast<const __nv_bfloat16*>(v_bh.data_ptr());
    auto o_ptr = out_bh.data_ptr<float>();

    if (head_dim == 64) {
        flashattention_v1_kernel<B_r, B_c, 64>
            <<<grid, block>>>(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, head_dim, causal);
    } else {
        flashattention_v1_kernel<B_r, B_c, 128>
            <<<grid, block>>>(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, head_dim, causal);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error: ", cudaGetErrorString(err));

    // reshape back to (batch, seq, hidden)
    auto out = out_bh.view({batch, num_heads, seq_len, head_dim})
                     .permute({0, 2, 1, 3}).contiguous()
                     .view({batch, seq_len, hidden_dim});
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention_v1", &custom_flash_attention_v1,
          "FlashAttention v1 — bf16 input + Tensor Cores (wmma)");
}
