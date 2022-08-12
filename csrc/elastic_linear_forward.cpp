
#include <vector>
#include "torch/extension.h"

at::Tensor elastic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_features,
    int out_features
);

std::vector<at::Tensor> elastic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor grad_c,
    int in_features,
    int out_features
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &elastic_sparse_linear_forward, "elastic sparse linear forward");
    m.def("backward", &elastic_sparse_linear_backward, "elastic sparse linear backward");
}
