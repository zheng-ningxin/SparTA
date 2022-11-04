
#include <vector>
#include "torch/extension.h"

at::Tensor dynamic_sparse_linear_condense_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor row_ptr,
    torch::Tensor col_indx,
    torch::Tensor bias,
    int M, int K, int N, int block_h, int block_w,
    int batch_size, int seq_len
);

std::vector<at::Tensor> dynamic_sparse_linear_condense_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor grad_a_row_ptr,
    torch::Tensor grad_a_col_index,
    torch::Tensor grad_c,
    int M, int K, int N, int block_h, int block_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &dynamic_sparse_linear_condense_forward, "dynamic sparse linear forward");
    m.def("backward", &dynamic_sparse_linear_condense_backward, "dynamic sparse linear backward");
}
