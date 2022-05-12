
#include <vector>
#include "torch/extension.h"

at::Tensor dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor row_ptr,
    torch::Tensor col_indx,
    torch::Tensor val,
    torch::Tensor bias,
    int M, int K, int N
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &dynamic_sparse_linear_forward, "our sparse attention forward");
}
