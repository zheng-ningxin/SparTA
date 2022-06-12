#include <vector>
#include "torch/extension.h"

at::Tensor our_sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor d_m_index,
    torch::Tensor d_n_index,
    torch::Tensor d_block_index,
    torch::Tensor val,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor val_mask,
    torch::Tensor col_range_index
);

std::vector<at::Tensor> our_sparse_attention_backward(
    torch::Tensor grad,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor gradv_row_idx,
    torch::Tensor gradv_col_idx,
    torch::Tensor gradv_subblock_idx,
    torch::Tensor val,
    torch::Tensor m_index,
    torch::Tensor n_index,
    torch::Tensor block_index,
    torch::Tensor col_range_index,
    torch::Tensor row_ptr,
    torch::Tensor col_idx
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &our_sparse_attention_forward, "our sparse attention forward");
    m.def("backward", &our_sparse_attention_backward, "our sparse attention backward");

}