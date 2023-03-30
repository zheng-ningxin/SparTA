#include "torch/extension.h"
#include <vector>

std::vector<at::Tensor> convert_forward(
    torch::Tensor sparse_pattern,
    torch::Tensor k_count,
    torch::Tensor k_index, 
    int block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &convert_forward, "Convert block csr format");
}