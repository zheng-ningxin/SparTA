#include <vector>
#include "torch/extension.h"

at::Tensor outdim_dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor index,
    torch::Tensor bias
);
at::Tensor indim_dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor index,
    torch::Tensor bias
);

std::vector<at::Tensor> outdim_dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor grad_c,
    torch::Tensor index
);
std::vector<at::Tensor> indim_dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor grad_c,
    torch::Tensor index
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("outdim_forward", &outdim_dynamic_sparse_linear_forward, "dynamic sparse linear forward");
    m.def("outdim_backward", &outdim_dynamic_sparse_linear_backward, "dynamic sparse linear backward");
    m.def("indim_forward", &indim_dynamic_sparse_linear_forward, "in dim sparse dynamic linear");
    m.def("indim_backward", &indim_dynamic_sparse_linear_backward, "in dim sparse dynamic linear backward");
}
