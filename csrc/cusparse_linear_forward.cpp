#include <vector>
#include "torch/extension.h"

at::Tensor cusparse_linear_forward(
    torch::Tensor input,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor values,
    std::vector<int> weight_shape,
    int nnz);

std::vector<at::Tensor> cusparse_linear_backward(
    torch::Tensor data,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor values,
    torch::Tensor grad_out,
    int M, int K, int N,
    int nnz);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &cusparse_linear_forward, "cuSparse sparse forward");
    m.def("backward", &cusparse_linear_backward, "Cusparse sparse linear backward");
}