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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &openai_bmm_32_64_32, "openai_bmm_32_64_32 sparse forward");
}