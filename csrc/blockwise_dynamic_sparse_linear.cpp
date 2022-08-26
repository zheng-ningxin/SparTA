#include <vector>
#include "torch/extension.h"

at::Tensor blockwise_dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor blockwise_mask
);

std::vector<at::Tensor> blockwise_dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor grad_c,
    torch::Tensor blockwise_mask
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &blockwise_dynamic_sparse_linear_forward, "dynamic sparse linear forward");
    m.def("backward", &blockwise_dynamic_sparse_linear_backward, "dynamic sparse linear backward");
}
