#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kTile = 32;
constexpr int kMaxHeadDim = 128;
constexpr int kThreads = 256;

__global__ void gqa_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int context_len,
    int max_seq_len,
    int head_dim
) {
    __shared__ float q_tile[kMaxHeadDim];
    __shared__ float k_tile[kTile][kMaxHeadDim + 1];
    __shared__ float v_tile[kTile][kMaxHeadDim + 1];
    __shared__ float o_tile[kMaxHeadDim];
    __shared__ float scores[kTile];
    __shared__ float probs[kTile];
    __shared__ float red[kThreads];
    __shared__ float m_shared;
    __shared__ float l_shared;

    int tid = threadIdx.x;
    int bh = blockIdx.x;
    int b = bh / num_q_heads;
    int qh = bh % num_q_heads;
    int groups = num_q_heads / num_kv_heads;
    int kvh = qh / groups;

    float scale = rsqrtf(static_cast<float>(head_dim));

    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_tile[d] = q[((b * num_q_heads + qh) * head_dim) + d];
        o_tile[d] = 0.0f;
    }
    if (tid == 0) {
        m_shared = -INFINITY;
        l_shared = 0.0f;
    }
    __syncthreads();

    for (int start = 0; start < context_len; start += kTile) {
        int tile_len = min(kTile, context_len - start);

        for (int idx = tid; idx < kTile * head_dim; idx += blockDim.x) {
            int c = idx / head_dim;
            int d = idx % head_dim;
            int pos = start + c;
            if (c < tile_len) {
                int cache_idx = (((b * num_kv_heads + kvh) * max_seq_len + pos) * head_dim) + d;
                k_tile[c][d] = k_cache[cache_idx];
                v_tile[c][d] = v_cache[cache_idx];
            } else {
                k_tile[c][d] = 0.0f;
                v_tile[c][d] = 0.0f;
            }
        }
        __syncthreads();

        if (tid < kTile) {
            if (tid < tile_len) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    sum += q_tile[d] * k_tile[tid][d];
                }
                scores[tid] = sum * scale;
            } else {
                scores[tid] = -INFINITY;
            }
        }
        __syncthreads();

        red[tid] = (tid < kTile) ? scores[tid] : -INFINITY;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] = fmaxf(red[tid], red[tid + stride]);
            }
            __syncthreads();
        }
        float tile_max = red[0];

        if (tid == 0) {
            float m_old = m_shared;
            float m_new = fmaxf(m_old, tile_max);
            float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : expf(m_old - m_new);
            m_shared = m_new;
            red[0] = alpha;
        }
        __syncthreads();
        float alpha = red[0];

        float local_sum = 0.0f;
        if (tid < kTile) {
            float val = 0.0f;
            if (scores[tid] != -INFINITY && m_shared != -INFINITY) {
                val = expf(scores[tid] - m_shared);
            }
            probs[tid] = val;
            local_sum = val;
        }
        __syncthreads();

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
            l_shared = alpha * l_shared + tile_sum;
        }
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x) {
            float pv = 0.0f;
            for (int c = 0; c < tile_len; ++c) {
                pv += probs[c] * v_tile[c][d];
            }
            o_tile[d] = alpha * o_tile[d] + pv;
        }
        __syncthreads();
    }

    for (int d = tid; d < head_dim; d += blockDim.x) {
        int out_idx = (((b * num_q_heads + qh) * 1) * head_dim) + d;
        out[out_idx] = l_shared > 0.0f ? o_tile[d] / l_shared : 0.0f;
    }
}

__global__ void update_gqa_cache_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int batch_size,
    int num_kv_heads,
    int max_seq_len,
    int head_dim,
    int pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_kv_heads * head_dim;
    if (idx >= total) {
        return;
    }

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int kvh = tmp % num_kv_heads;
    int b = tmp / num_kv_heads;

    int cache_idx = (((b * num_kv_heads + kvh) * max_seq_len + pos) * head_dim) + d;
    k_cache[cache_idx] = k[idx];
    v_cache[cache_idx] = v[idx];
}

}  // namespace

torch::Tensor custom_flash_attention_v2_decode(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int context_len
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
    TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k_cache.scalar_type() == torch::kFloat32, "k_cache must be float32");
    TORCH_CHECK(v_cache.scalar_type() == torch::kFloat32, "v_cache must be float32");
    TORCH_CHECK(q.dim() == 4, "q must have shape (batch, q_heads, 1, head_dim)");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape (batch, kv_heads, max_seq_len, head_dim)");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache shape must match k_cache");

    int batch_size = q.size(0);
    int num_q_heads = q.size(1);
    int q_len = q.size(2);
    int head_dim = q.size(3);
    int num_kv_heads = k_cache.size(1);
    int max_seq_len = k_cache.size(2);

    TORCH_CHECK(q_len == 1, "v2 decode only supports single-token q");
    TORCH_CHECK(k_cache.size(0) == batch_size, "batch mismatch");
    TORCH_CHECK(k_cache.size(3) == head_dim, "head_dim mismatch");
    TORCH_CHECK(head_dim <= kMaxHeadDim, "head_dim must be <= 128");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(context_len > 0 && context_len <= max_seq_len, "invalid context_len");

    auto q_c = q.contiguous();
    auto k_c = k_cache.contiguous();
    auto v_c = v_cache.contiguous();
    auto out = torch::empty_like(q_c);
    dim3 block(kThreads);
    dim3 grid(batch_size * num_q_heads);
    gqa_decode_kernel<<<grid, block>>>(
        q_c.data_ptr<float>(),
        k_c.data_ptr<float>(),
        v_c.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_q_heads,
        num_kv_heads,
        context_len,
        max_seq_len,
        head_dim
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    return out;
}

void update_gqa_kv_cache(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k,
    torch::Tensor v,
    int pos
) {
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
    TORCH_CHECK(v_cache.is_cuda(), "v_cache must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(k_cache.scalar_type() == torch::kFloat32, "k_cache must be float32");
    TORCH_CHECK(v_cache.scalar_type() == torch::kFloat32, "v_cache must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be (batch, kv_heads, max_seq_len, head_dim)");
    TORCH_CHECK(k.dim() == 4, "k must be (batch, kv_heads, 1, head_dim)");
    TORCH_CHECK(k.size(0) == k_cache.size(0) && k.size(1) == k_cache.size(1)
                && k.size(2) == 1 && k.size(3) == k_cache.size(3),
                "k must be (batch, kv_heads, 1, head_dim)");
    TORCH_CHECK(v.sizes() == k.sizes(), "v shape must match k shape");
    TORCH_CHECK(pos >= 0 && pos < k_cache.size(2), "invalid cache position");

    int batch_size = k_cache.size(0);
    int num_kv_heads = k_cache.size(1);
    int max_seq_len = k_cache.size(2);
    int head_dim = k_cache.size(3);
    int total = batch_size * num_kv_heads * head_dim;
    int blocks = (total + kThreads - 1) / kThreads;

    update_gqa_cache_kernel<<<blocks, kThreads>>>(
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        k.contiguous().data_ptr<float>(),
        v.contiguous().data_ptr<float>(),
        batch_size,
        num_kv_heads,
        max_seq_len,
        head_dim,
        pos
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention_v2_decode", &custom_flash_attention_v2_decode,
          "GQA-native single-token decode");
    m.def("update_gqa_kv_cache", &update_gqa_kv_cache,
          "Update GQA-native KV cache at one position");
}
