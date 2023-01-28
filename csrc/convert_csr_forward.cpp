#include "torch/extension.h"
#include <vector>

std::vector<at::Tensor> convert_csr_forward(
    torch::Tensor sparse_pattern,
    torch::Tensor dense_values);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &convert_csr_forward, "Convert block csr format");
}