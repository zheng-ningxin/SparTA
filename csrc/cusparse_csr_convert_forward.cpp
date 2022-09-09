#include <vector>
#include "torch/extension.h"

std::vector<at::Tensor> cusparse_convert_forward(
    torch::Tensor dense_values);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &cusparse_convert_forward, "cuSparse sparse csr convert forward");
}