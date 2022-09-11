#include <vector>
#include "torch/extension.h"


at::Tensor cusparse_sddmm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor csr_row,
    torch::Tensor csr_col,
    torch::Tensor csr_val
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sddmm_forward", &cusparse_convert_forward, "cuSparse sparse csr sddmm forward");
}