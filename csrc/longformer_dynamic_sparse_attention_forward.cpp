
#include <vector>
#include "torch/extension.h"

at::Tensor batch_matmul_block_sparse_out(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor row_pos,
    torch::Tensor col,
    torch::Tensor output,
    int block_h, int block_w, int block_nnz
);

at::Tensor longformer_mixed_softmax(
    torch::Tensor A,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor val_mask,
    torch::Tensor global_attention,
    torch::Tensor extra_buffer,
    int block_h, int block_w, int block_nnz

);

at::Tensor batch_matmul_block_sparse(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    int block_h,
    int block_n

);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batch_matmul_sdd", &batch_matmul_block_sparse_out, "dynamic sparse attention forward");
    m.def("longformer_softmax", &longformer_mixed_softmax, "sparse softmax for the longformer pattern");
    m.def("batch_matmul_dsd", &batch_matmul_block_sparse, "attention score x V");
}
