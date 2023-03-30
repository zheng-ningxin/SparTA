#include "torch/extension.h"
#include <vector>

std::vector<at::Tensor> convert_bcsr_forward(
    torch::Tensor sparse_pattern,
    torch::Tensor dense_values,
    int block_h, 
    int block_w);

void convert_bcsr_forward_v2(
    torch::Tensor sparse_act,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor ext_buffer,
    int h, int w,
    int block_h, 
    int block_w);


void convert_bcsr_forward_v3(
    torch::Tensor sparse_act,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor ext_buffer,
    torch::Tensor seq_lens,
    int h, int w,
    int block_h, 
    int block_w,
    int batch_size);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &convert_bcsr_forward, "Convert block csr format");
    m.def("forward_v2", &convert_bcsr_forward_v2, "Convert block csr format"); // build the sparse index directly according to the values
    m.def("forward_v3", &convert_bcsr_forward_v3, "Convert block csr format"); // build the sparse index directly according to the values
    
}