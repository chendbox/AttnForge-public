#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

__global__ void pv_wmma_debug_kernel(
    const __nv_bfloat16* __restrict__ p,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ out)
{
    __shared__ __nv_bfloat16 p_tile[16][16 + 8];
    __shared__ __nv_bfloat16 v_tile[16][16 + 8];
    __shared__ float out_tile[16][16];

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int c = idx % 16;
        p_tile[r][c] = p[idx];
        v_tile[r][c] = v[idx];
        out_tile[r][c] = 0.0f;
    }

    __syncthreads();

    if (warp_id == 0) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frag;

        wmma::fill_fragment(out_frag, 0.0f);
        wmma::load_matrix_sync(p_frag, &p_tile[0][0], 16 + 8);
        wmma::load_matrix_sync(v_frag, &v_tile[0][0], 16 + 8);
        wmma::mma_sync(out_frag, p_frag, v_frag, out_frag);
        wmma::store_matrix_sync(&out_tile[0][0], out_frag, 16, wmma::mem_row_major);

        __syncwarp();

        for (int idx = lane_id; idx < 16 * 16; idx += 32) {
            int r = idx / 16;
            int c = idx % 16;
            out[idx] = out_tile[r][c];
        }
    }
}

torch::Tensor debug_pv_wmma(torch::Tensor p, torch::Tensor v)
{
    TORCH_CHECK(p.is_cuda(), "p must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(p.scalar_type() == torch::kBFloat16, "p must be bf16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bf16");
    TORCH_CHECK(p.sizes() == torch::IntArrayRef({16, 16}), "p must have shape [16, 16]");
    TORCH_CHECK(v.sizes() == torch::IntArrayRef({16, 16}), "v must have shape [16, 16]");

    auto p_contig = p.contiguous();
    auto v_contig = v.contiguous();
    auto out = torch::empty({16, 16}, p.options().dtype(torch::kFloat32));

    auto p_ptr = reinterpret_cast<const __nv_bfloat16*>(p_contig.data_ptr());
    auto v_ptr = reinterpret_cast<const __nv_bfloat16*>(v_contig.data_ptr());
    auto out_ptr = out.data_ptr<float>();

    pv_wmma_debug_kernel<<<1, 32>>>(p_ptr, v_ptr, out_ptr);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return out;
}
