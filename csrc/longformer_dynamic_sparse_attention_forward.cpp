
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batch_matmul_sdd", &batch_matmul_block_sparse_out, "dynamic sparse linear forward");
}
