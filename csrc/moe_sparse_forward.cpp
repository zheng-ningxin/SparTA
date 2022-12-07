
#include <vector>
#include "torch/extension.h"

at::Tensor moe_sparse_forward(
    torch::Tensor tokens,
    torch::Tensor weight,
    torch::Tensor router_index, // given by the router function
    torch::Tensor sparse_index, 
    int n_expert,
    int in_hidden,
    int out_hidden
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &moe_sparse_forward, "dynamic sparse forward function of MOE");
}
