
#include <vector>
#include "torch/extension.h"

at::Tensor dynamic_sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_pos,
    torch::Tensor val_mask,
    int block_nnz,
    int head_num
);
std::vector<at::Tensor> dynamic_sparse_attention_backward(
    torch::Tensor grad,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_pos,
    torch::Tensor val_mask,
    torch::Tensor block_index,
    torch::Tensor grad_row_idx,
    torch::Tensor grad_col_idx,
    int block_nnz
    );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &dynamic_sparse_attention_forward, "our sparse attention forward");
    m.def("backward", &dynamic_sparse_attention_backward, "our sparse attention backward");

}
