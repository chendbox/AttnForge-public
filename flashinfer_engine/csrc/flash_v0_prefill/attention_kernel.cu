#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>


using namespace torch::indexing;

//to optimize softmax row reduction
// max
__inline__ __device__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

// sum
__inline__ __device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}


// FlashAttention kernel
template <int B_r, int B_c, int HEAD_DIM>
__global__ void flashattention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq_len,
    int head_dim,
    bool causal)
{
    // Shared memory for tiles (padded by +1 to avoid bank conflicts)
    __shared__ float q_tile[B_r][HEAD_DIM + 1];
    __shared__ float k_tile[B_c][HEAD_DIM + 1];
    __shared__ float v_tile[B_c][HEAD_DIM + 1];
    __shared__ float o_tile[B_r][HEAD_DIM + 1];
    __shared__ float m_tile[B_r];
    __shared__ float l_tile[B_r];
    __shared__ float scores[B_r][B_c];
    __shared__ float p[B_r][B_c];
    __shared__ float alpha_tile[B_r];

    // ########################################################

    // TODO: Implement the FlashAttention operation 
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warps_per_block = blockDim.x / 32;

    int bh = blockIdx.y;              // batch-head index
    int row_block = blockIdx.x;       // Q row tile index
    int row_start = row_block * B_r;
    int row_end = row_start + B_r < seq_len ? row_start + B_r : seq_len;
    
    int base = bh * seq_len * head_dim;
    float scale = rsqrtf((float)HEAD_DIM);
    
    // Load Q tile and initialize O tile
    // for (int idx = tid; idx < B_r * head_dim; idx += blockDim.x) {
    //     int r = idx / head_dim;
    //     int d = idx % head_dim;
    
    //     int global_r = row_start + r;
    
    //     if (global_r < seq_len) {
    //         q_tile[r][d] = q[base + global_r * head_dim + d];
    //     } else {
    //         q_tile[r][d] = 0.0f;
    //     }
    
    //     o_tile[r][d] = 0.0f;
    // }

    
    constexpr int VEC = 4;
    constexpr int HEAD_DIM_VEC = HEAD_DIM / VEC;

    // Load Q tile and initialize O tile using float4
    for (int idx = tid; idx < B_r * HEAD_DIM_VEC; idx += blockDim.x) {
        int r = idx / HEAD_DIM_VEC;
        int d4 = idx % HEAD_DIM_VEC;
        int d = d4 * VEC;

        int global_r = row_start + r;

        float4 qv;

        if (global_r < seq_len) {
            const float4* q_ptr4 =
                reinterpret_cast<const float4*>(q + base + global_r * head_dim + d);
            qv = *q_ptr4;
        } else {
            qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        q_tile[r][d + 0] = qv.x;
        q_tile[r][d + 1] = qv.y;
        q_tile[r][d + 2] = qv.z;
        q_tile[r][d + 3] = qv.w;

        o_tile[r][d + 0] = 0.0f;
        o_tile[r][d + 1] = 0.0f;
        o_tile[r][d + 2] = 0.0f;
        o_tile[r][d + 3] = 0.0f;
    }
    
    // Initialize streaming softmax states
    if (tid < B_r) {
        m_tile[tid] = -INFINITY;
        l_tile[tid] = 0.0f;
        alpha_tile[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Loop over K/V column tiles
    for (int col_start = 0; col_start < seq_len; col_start += B_c) {
        int col_end = col_start + B_c < seq_len ? col_start + B_c : seq_len;
        int Bc_actual = col_end - col_start;
    
        // If causal and this whole K/V tile is in the future, skip it
        if (causal && col_start > row_end - 1) {
            continue;
        }
    
    //     // Load K and V tiles
    //     for (int idx = tid; idx < B_c * head_dim; idx += blockDim.x) {
    //         int c = idx / head_dim;
    //         int d = idx % head_dim;
    
    //         int global_c = col_start + c;
    
    //         if (c < Bc_actual && global_c < seq_len) {
    //             k_tile[c][d] = k[base + global_c * head_dim + d];
    //             v_tile[c][d] = v[base + global_c * head_dim + d];
    //         } else {
    //             k_tile[c][d] = 0.0f;
    //             v_tile[c][d] = 0.0f;
    //         }
    //     }
    
        // K/V tile vectorized load
        for (int idx = tid; idx < B_c * HEAD_DIM_VEC; idx += blockDim.x) {
            int c = idx / HEAD_DIM_VEC;
            int d4 = idx % HEAD_DIM_VEC;
            int d = d4 * VEC;
        
            int global_c = col_start + c;
        
            float4 kv;
            float4 vv;
        
            if (c < Bc_actual && global_c < seq_len) {
                const float4* k_ptr4 =
                    reinterpret_cast<const float4*>(k + base + global_c * head_dim + d);
                const float4* v_ptr4 =
                    reinterpret_cast<const float4*>(v + base + global_c * head_dim + d);
        
                kv = *k_ptr4;
                vv = *v_ptr4;
            } else {
                kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                vv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        
            k_tile[c][d + 0] = kv.x;
            k_tile[c][d + 1] = kv.y;
            k_tile[c][d + 2] = kv.z;
            k_tile[c][d + 3] = kv.w;
        
            v_tile[c][d + 0] = vv.x;
            v_tile[c][d + 1] = vv.y;
            v_tile[c][d + 2] = vv.z;
            v_tile[c][d + 3] = vv.w;
        }
            __syncthreads();
    
            // Compute scores: S_ij = Q_i K_j^T
            for (int idx = tid; idx < B_r * B_c; idx += blockDim.x) {
                int r = idx / B_c;
                int c = idx % B_c;
        
                int global_r = row_start + r;
                int global_c = col_start + c;
        
                bool valid = (global_r < seq_len) &&
                            (global_c < seq_len) &&
                            (!causal || global_c <= global_r);
        
                if (valid) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        sum += q_tile[r][d] * k_tile[c][d];
                    }
                    scores[r][c] = sum * scale;
                } else {
                    scores[r][c] = -INFINITY;
                }
            }
        
            __syncthreads();
    
        // Online softmax update per row -
        // Online softmax update per row using warp-level reduction.
        // Each warp processes rows: warp_id, warp_id + warps_per_block, ...
        for (int r = warp_id; r < B_r; r += warps_per_block) {
            int global_r = row_start + r;

            // 1. Each lane computes local max over part of the columns.
            float local_max = -INFINITY;

            if (global_r < seq_len) {
                for (int c = lane_id; c < B_c; c += 32) {
                    local_max = fmaxf(local_max, scores[r][c]);
                }
            }

            // 2. Warp reduction to get row max.
            // Important: warp_reduce_max only leaves the final result in lane 0.
            float row_max = warp_reduce_max(local_max);

            // Broadcast row_max from lane 0 to all lanes in the warp.
            row_max = __shfl_sync(0xffffffff, row_max, 0);

            float m_old = m_tile[r];
            float m_new = fmaxf(m_old, row_max);

            // Broadcast m_new too, so every lane uses the same softmax normalizer.
            // m_new = __shfl_sync(0xffffffff, m_new, 0);

            float alpha = 0.0f;
            if (m_old != -INFINITY && m_new != -INFINITY) {
                alpha = expf(m_old - m_new);
            }

            // 3. Each lane computes exp(score - m_new) for its columns.
            float local_sum = 0.0f;

            if (global_r < seq_len) {
                for (int c = lane_id; c < B_c; c += 32) {
                    float val = 0.0f;

                    if (scores[r][c] != -INFINITY && m_new != -INFINITY) {
                        val = expf(scores[r][c] - m_new);
                    }

                    p[r][c] = val;
                    local_sum += val;
                }
            }

            // 4. Warp reduction to get row sum.
            float row_sum = warp_reduce_sum(local_sum);

            // 5. Lane 0 updates row state.
            if (lane_id == 0 && global_r < seq_len) {
                float l_new = alpha * l_tile[r] + row_sum;

                m_tile[r] = m_new;
                l_tile[r] = l_new;
                alpha_tile[r] = alpha;
            }
        }
            
        __syncthreads();
    
        // Update output accumulator:
        // O_i = alpha * O_i + P_ij V_j
        for (int idx = tid; idx < B_r * head_dim; idx += blockDim.x) {
            int r = idx / head_dim;
            int d = idx % head_dim;
    
            int global_r = row_start + r;
    
            if (global_r < seq_len) {
                float pv = 0.0f;
    
                for (int c = 0; c < B_c; ++c) {
                    pv += p[r][c] * v_tile[c][d];
                }
    
                o_tile[r][d] = alpha_tile[r] * o_tile[r][d] + pv;
            }
        }
    
        __syncthreads();
    }
    
    // Normalize and write output
    // for (int idx = tid; idx < B_r * head_dim; idx += blockDim.x) {
    //     int r = idx / head_dim;
    //     int d = idx % head_dim;
    
    //     int global_r = row_start + r;
    
    //     if (global_r < seq_len) {
    //         float denom = l_tile[r];
    //         out[base + global_r * head_dim + d] =
    //             denom > 0.0f ? o_tile[r][d] / denom : 0.0f;
    //     }
    // }

    for (int idx = tid; idx < B_r * HEAD_DIM_VEC; idx += blockDim.x) {
        int r = idx / HEAD_DIM_VEC;
        int d4 = idx % HEAD_DIM_VEC;
        int d = d4 * VEC;
    
        int global_r = row_start + r;
    
        if (global_r < seq_len) {
            float denom = l_tile[r];
            float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    
            float4 ov;
            ov.x = o_tile[r][d + 0] * inv;
            ov.y = o_tile[r][d + 1] * inv;
            ov.z = o_tile[r][d + 2] * inv;
            ov.w = o_tile[r][d + 3] * inv;
    
            float4* out_ptr4 =
                reinterpret_cast<float4*>(out + base + global_r * head_dim + d);
            *out_ptr4 = ov;
        }
    }
}
    // ########################################################

torch::Tensor custom_flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal) {
    auto options = q.options();
    
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    
    // Reshape to (batch_size * num_heads, seq_len, head_dim)
    auto q_bh = q.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    auto k_bh = k.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    auto v_bh = v.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    
    auto out_bh = torch::zeros_like(q_bh, options);
    
    // ########################################################
    
    // TODO: launch the kernel
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");

    TORCH_CHECK(head_dim <= 128, "This simple kernel only supports head_dim <= 128");

    // 8 x 8 too small for small seq_len, each tile only computes 64 scores, kernel launch/sync/shared memory overhead dominates.
    constexpr int B_r = 16;
    constexpr int B_c = 16; // 32 is a good trade-off between shared memory and parallelism: 16*32 excceed shared memory limit

    dim3 block(256);
    dim3 grid((seq_len + B_r - 1) / B_r, batch_size * num_heads);

    // flashattention_kernel<B_r, B_c><<<grid, block>>>(
    //     q_bh.data_ptr<float>(),
    //     k_bh.data_ptr<float>(),
    //     v_bh.data_ptr<float>(),
    //     out_bh.data_ptr<float>(),
    //     seq_len,
    //     head_dim,
    //     causal
    // );

    if (head_dim == 64) {
        flashattention_kernel<B_r, B_c, 64><<<grid, block>>>(
            q_bh.data_ptr<float>(),
            k_bh.data_ptr<float>(),
            v_bh.data_ptr<float>(),
            out_bh.data_ptr<float>(),
            seq_len,
            head_dim,
            causal
        );
    }

    else if (head_dim == 128) {
        flashattention_kernel<B_r, B_c, 128><<<grid, block>>>(
            q_bh.data_ptr<float>(),
            k_bh.data_ptr<float>(),
            v_bh.data_ptr<float>(),
            out_bh.data_ptr<float>(),
            seq_len,
            head_dim,
            causal
        );
    }

    else {
        TORCH_CHECK(false, "Only head_dim 64/128 supported");
    }





    // ########################################################

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Reshape back to (batch_size, seq_len, hidden_dim)
    auto out = out_bh.view({batch_size, num_heads, seq_len, head_dim})
                     .permute({0, 2, 1, 3})
                     .contiguous()
                     .view({batch_size, seq_len, hidden_dim});
    
    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention", &custom_flash_attention, "Custom FlashAttention in CUDA");
}
