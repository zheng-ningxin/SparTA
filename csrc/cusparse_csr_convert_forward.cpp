#include <vector>
#include "torch/extension.h"

std::vector<at::Tensor> cusparse_convert_forward(
    torch::Tensor dense_values);


at::Tensor cusparse_convert_backward(
    torch::Tensor csr_row,
    torch::Tensor csr_col,
    torch::Tensor csr_val,
    int n_row,
    int n_col,
    int nnz
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &cusparse_convert_forward, "cuSparse sparse csr convert forward");
    m.def("backward", &cusparse_convert_backward, "cuSparse sparse csr convert backward");
}