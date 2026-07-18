#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

using namespace nvcuda;
using namespace torch::indexing;

torch::Tensor debug_pv_wmma(torch::Tensor p, torch::Tensor v);

// ------------------------------------------------------------------ //
// Warp reductions (same as v0 - fp32 softmax stays fp32)
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
// FlashAttention v1 kernel - bf16 input + Tensor Cores (wmma)
//
// Template parameters:
//   B_r      : rows per Q tile   (must be 16 - wmma M dimension)
//   B_c      : cols per KV tile  (must be 16 - wmma N dimension)
//   HEAD_DIM : attention head dimension (64 or 128)
//
// wmma tile shape used: M=16, N=16, K=16
//   QK^T: A=Q(16xHEAD_DIM), B=K^T(HEAD_DIMx16) -> C=scores(16x16)
//         loop over K in steps of 16: HEAD_DIM/16 iterations
//   PV:   A=P(16x16),       B=V(16xHEAD_DIM)   -> C=O(16xHEAD_DIM)
//         loop over N in steps of 16: HEAD_DIM/16 output tiles
//
// Each thread block handles one (batch-head, row-tile) pair.
// One warp computes the score tile; multiple warps update output tiles.
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
    __shared__ __nv_bfloat16 p_bf16[B_r][B_c + 8];
    __shared__ float         o_tile[B_r][HEAD_DIM + 1];

    // softmax running stats (fp32 for precision)
    __shared__ float m_tile[B_r];      // row max
    __shared__ float l_tile[B_r];      // row normalizer
    __shared__ float alpha_tile[B_r];  // rescale factor

    // scores and softmax probabilities (fp32)
    __shared__ float scores[B_r][B_c];
    __shared__ float p[B_r][B_c];
    __shared__ float o_partial[HEAD_DIM / 16][B_r][16];

    // ---- thread/block indexing ----
    int tid             = threadIdx.x;
    int warp_id         = tid / 32;
    int lane_id         = tid % 32;
    int warps_per_block = blockDim.x / 32;

    int bh        = blockIdx.y;
    int row_block = blockIdx.x;
    int row_start = row_block * B_r;
    int row_end   = min(row_start + B_r, seq_len);

    int base    = bh * seq_len * head_dim;
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

        // ---- compute scores = Q_tile @ K_tile^T with WMMA ----
        if (warp_id == 0) {
            wmma::fragment<wmma::matrix_a, B_r, B_c, 16, __nv_bfloat16, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, B_r, B_c, 16, __nv_bfloat16, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, B_r, B_c, 16, float> score_frag;

            wmma::fill_fragment(score_frag, 0.0f);

            #pragma unroll
            for (int k0 = 0; k0 < HEAD_DIM; k0 += 16) {
                wmma::load_matrix_sync(q_frag, &q_tile[0][k0], HEAD_DIM + 8);
                wmma::load_matrix_sync(k_frag, &k_tile[0][k0], HEAD_DIM + 8);
                wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
            }

            wmma::store_matrix_sync(&scores[0][0], score_frag, B_c, wmma::mem_row_major);
        }

        __syncthreads();

        for (int idx = tid; idx < B_r * B_c; idx += blockDim.x) {
            int r = idx / B_c;
            int c = idx % B_c;
            int global_r = row_start + r;
            int global_c = col_start + c;

            bool valid = (global_r < seq_len) &&
                         (c < Bc_actual) &&
                         (global_c < seq_len) &&
                         (!causal || global_c <= global_r);

            scores[r][c] = valid ? scores[r][c] * scale : -INFINITY;
        }

        __syncthreads();

        // ---- online softmax update (same as v0 - fp32) ----
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
            float alpha = (m_old != -INFINITY) ? expf(m_old - m_new) : 0.0f;

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

        // ---- rescale running output and convert P tile to bf16 ----
        for (int idx = tid; idx < B_r * HEAD_DIM; idx += blockDim.x) {
            int r = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            o_tile[r][d] *= alpha_tile[r];
        }

        for (int idx = tid; idx < B_r * B_c; idx += blockDim.x) {
            int r = idx / B_c;
            int c = idx % B_c;
            p_bf16[r][c] = __float2bfloat16(p[r][c]);
        }

        __syncthreads();

        // P @ V uses Tensor Cores. Store each WMMA result into a compact
        // 16x16 tile first; WMMA store/load is fragile with padded fp32 strides.
        constexpr int OUTPUT_TILES = HEAD_DIM / 16;
        if (warp_id < OUTPUT_TILES) {
            int out_col = warp_id * 16;

            wmma::fragment<wmma::matrix_a, B_r, 16, B_c, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, B_r, 16, B_c, __nv_bfloat16, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, B_r, 16, B_c, float> pv_frag;

            wmma::fill_fragment(pv_frag, 0.0f);
            wmma::load_matrix_sync(p_frag, &p_bf16[0][0], B_c + 8);
            wmma::load_matrix_sync(v_frag, &v_tile[0][out_col], HEAD_DIM + 8);
            wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            wmma::store_matrix_sync(&o_partial[warp_id][0][0], pv_frag, 16, wmma::mem_row_major);

            __syncwarp();

            for (int idx = lane_id; idx < B_r * 16; idx += 32) {
                int r = idx / 16;
                int d = idx % 16;
                o_tile[r][out_col + d] += o_partial[warp_id][r][d];
            }
        }

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
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
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
          "FlashAttention v1 - bf16 input + Tensor Cores (wmma)");
    m.def("debug_pv_wmma", &debug_pv_wmma,
          "Debug 16x16 bf16 P @ V WMMA block");
}
