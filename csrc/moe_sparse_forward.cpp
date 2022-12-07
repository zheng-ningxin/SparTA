
#include <vector>
#include "torch/extension.h"

at::Tensor moe_sparse_forward(
    torch::Tensor tokens,
    torch::Tensor weight,
    torch::Tensor router_index, // given by the router function
    torch::Tensor sparse_index
);

void moe_sparse_convert_index(
    torch::Tensor tokens,
    torch::Tensor weight,
    torch::Tensor router_index,
    torch::Tensor sparse_index,
    torch::Tensor expert_count
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &moe_sparse_forward, "dynamic sparse forward function of MOE");
    m.def("convert_index", &moe_sparse_convert_index, "dynamic sparse index construction function of MOE");

}
