
#include <vector>
#include "torch/extension.h"

at::Tensor seqlen_dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor seqlens
);

std::vector<at::Tensor> seqlen_dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor seqlens,
    torch::Tensor grad_c
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &seqlen_dynamic_sparse_linear_forward, "dynamic sparse linear forward");
    m.def("backward", &seqlen_dynamic_sparse_linear_backward, "dynamic sparse linear backward");
}
