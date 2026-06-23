#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

using namespace torch::indexing;

// FlashAttention kernel
template <int B>
__global__ void flash_attention_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq_len,
    int head_dim,
    bool causal)
{
    // Shared memory for tiles (padded by +1 to avoid bank conflicts)
    __shared__ float q_tile[128];  // Max head_dim = 128
    __shared__ float k_tile[B][128 + 1];
    __shared__ float v_tile[B][128 + 1];
    __shared__ float o_tile[128];
    __shared__ float m_tile;
    __shared__ float l_tile;
    __shared__ float scores[B];
    __shared__ float p[B];
    __shared__ float red[256];

    // ########################################################

    // TODO: Implement the FlashAttention operation 
    int tid = threadIdx.x;
    int bh = blockIdx.x;  // batch-head index

    int q_base = bh * head_dim;              // q shape: (B*H, 1, D)
    int kv_base = bh * seq_len * head_dim;   // k/v shape: (B*H, seq_len, D)

    float scale = rsqrtf((float)head_dim);

    // Load single query vector and initialize output accumulator
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_tile[d] = q[q_base + d];
        o_tile[d] = 0.0f;
    }

    if (tid == 0) {
        m_tile = -INFINITY;
        l_tile = 0.0f;
    }

    __syncthreads();

    // Sweep over cached K/V in column tiles
    for (int col_start = 0; col_start < seq_len; col_start += B) {
        int col_end = (col_start + B < seq_len) ? (col_start + B) : seq_len;
        int B_actual = col_end - col_start;

        // Load K/V tile
        for (int idx = tid; idx < B * head_dim; idx += blockDim.x) {
            int c = idx / head_dim;
            int d = idx % head_dim;

            int global_c = col_start + c;

            if (c < B_actual) {
                k_tile[c][d] = k[kv_base + global_c * head_dim + d];
                v_tile[c][d] = v[kv_base + global_c * head_dim + d];
            } else {
                k_tile[c][d] = 0.0f;
                v_tile[c][d] = 0.0f;
            }
        }

        __syncthreads();

        // Compute scores[c] = q dot k[c]
        if (tid < B) {
            int c = tid;

            if (c < B_actual) {
                float sum = 0.0f;

                for (int d = 0; d < head_dim; ++d) {
                    sum += q_tile[d] * k_tile[c][d];
                }

                scores[c] = sum * scale;
            } else {
                scores[c] = -INFINITY;
            }
        }

        __syncthreads();

        // Parallel reduction: tile max
        red[tid] = (tid < B) ? scores[tid] : -INFINITY;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] = fmaxf(red[tid], red[tid + stride]);
            }
            __syncthreads();
        }

        float tile_max = red[0];

        float alpha = 0.0f;

        if (tid == 0) {
            float m_old = m_tile;
            float m_new = fmaxf(m_old, tile_max);

            if (m_old != -INFINITY && m_new != -INFINITY) {
                alpha = expf(m_old - m_new);
            }

            m_tile = m_new;
            red[0] = alpha;  // reuse red[0] to broadcast alpha
        }

        __syncthreads();

        alpha = red[0];

        // Compute p[c] = exp(scores[c] - m_tile)
        float local_sum = 0.0f;

        if (tid < B) {
            float val = 0.0f;

            if (scores[tid] != -INFINITY && m_tile != -INFINITY) {
                val = expf(scores[tid] - m_tile);
            }

            p[tid] = val;
            local_sum = val;
        }

        __syncthreads();

        // Parallel reduction: tile sum
        red[tid] = local_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            __syncthreads();
        }

        float tile_sum = red[0];

        if (tid == 0) {
            l_tile = alpha * l_tile + tile_sum;
        }

        __syncthreads();

        // Update output accumulator:
        // o = alpha * o + p @ V
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float pv = 0.0f;

            for (int c = 0; c < B; ++c) {
                pv += p[c] * v_tile[c][d];
            }

            o_tile[d] = alpha * o_tile[d] + pv;
        }

        __syncthreads();
    }

    // Normalize and write output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float denom = l_tile;
        out[q_base + d] = denom > 0.0f ? o_tile[d] / denom : 0.0f;
    }

    // ########################################################
}

torch::Tensor custom_flash_attention_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal) {
    auto options = q.options();
    
    int batch_size = q.size(0);
    int seq_len = k.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    
    // Reshape to (batch_size * num_heads, 1, head_dim)
    auto q_bh = q.view({batch_size, 1, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, 1, head_dim});
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

    TORCH_CHECK(head_dim <= 128, "This kernel only supports head_dim <= 128");

    constexpr int B = 32;

    dim3 block(256);
    dim3 grid(batch_size * num_heads);

    flash_attention_decode_kernel<B><<<grid, block>>>(
        q_bh.data_ptr<float>(),
        k_bh.data_ptr<float>(),
        v_bh.data_ptr<float>(),
        out_bh.data_ptr<float>(),
        seq_len,
        head_dim,
        causal
    );
    
    // ########################################################

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Reshape back to (batch_size, 1, hidden_dim)
    auto out = out_bh.view({batch_size, num_heads, 1, head_dim})
                     .permute({0, 2, 1, 3})
                     .contiguous()
                     .view({batch_size, 1, hidden_dim});
    
    return out;
}

__global__ void update_cache_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int batch_size,
    int current_pos,
    int max_seq_len,
    int hidden_dim
) {
    // ########################################################

    // TODO: Implement the update kv cache operation 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * hidden_dim;

    if (idx >= total) {
        return;
    }

    int b = idx / hidden_dim;
    int d = idx % hidden_dim;

    // k_cache/v_cache shape: (batch_size, max_seq_len, hidden_dim)
    // k/v shape: effectively (batch_size, hidden_dim), because seq_len = 1
    int cache_idx = (b * max_seq_len + current_pos) * hidden_dim + d;
    int new_idx = b * hidden_dim + d;

    k_cache[cache_idx] = k[new_idx];
    v_cache[cache_idx] = v[new_idx];

    // ########################################################
}

void update_kv_cache(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k, torch::Tensor v, int current_pos) {
    auto options = k_cache.options();

    int batch_size = k_cache.size(0);
    int max_seq_len = k_cache.size(1);
    int hidden_dim = k_cache.size(2);
    
    int total_threads = batch_size * hidden_dim;

    int threads_per_block = 256;
    dim3 threads(threads_per_block);
    dim3 blocks((total_threads + threads_per_block - 1) / threads_per_block);

    update_cache_kernel<<<blocks, threads>>>(
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        batch_size,
        current_pos,
        max_seq_len,
        hidden_dim
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention_decode", &custom_flash_attention_decode, "Custom FlashAttention in CUDA");
    m.def("update_kv_cache", &update_kv_cache, "Update KV cache in CUDA");
}
