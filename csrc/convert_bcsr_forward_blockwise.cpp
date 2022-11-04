#include "torch/extension.h"
#include <vector>

std::vector<at::Tensor> convert_bcsr_forward_blockwise(
    torch::Tensor sparse_pattern,
    int transpose
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &convert_bcsr_forward_blockwise, "Convert block csr format");
}