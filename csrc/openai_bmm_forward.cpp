#include "torch/extension.h"

at::Tensor openai_bmm_32_64_32(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor B,
    int M,
    int K,
    int N,
    int batch_size,
    int block_nnz);

at::Tensor openai_bmm_32_64_32_condense(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor B,
    int M,
    int K,
    int N,
    int block_h,
    int block_w,
    int batch_size,
    int block_nnz);

at::Tensor openai_bmm_32_64_32_condense_dim_m(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor B,
    int M,
    int K,
    int N,
    int block_h,
    int block_w,
    int batchsize,
    int block_nnz);

at::Tensor openai_bmm_32_64_32_condense_dim_m_v2(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor B,
    int M,
    int K,
    int N,
    int block_h,
    int block_w,
    int batchsize,
    int block_nnz);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &openai_bmm_32_64_32, "openai_bmm_32_64_32 sparse forward");
    m.def("forward_condense", &openai_bmm_32_64_32_condense, "openai_bmm_32_64_32_condense forward");
    m.def("forward_condense_m", &openai_bmm_32_64_32_condense_dim_m, "openai_bmm_32_64_32_condense_dim_m forward");
    m.def("forward_condense_m_v2", &openai_bmm_32_64_32_condense_dim_m_v2, "openai_bmm_32_64_32_condense_dim_m forward");
}