
#include <vector>
#include "torch/extension.h"

at::Tensor dynamic_sparse_cache_attention_forward(
    torch::Tensor Q, // new Q 
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor pad_len,
    int max_token_len
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &dynamic_sparse_cache_attention_forward, "our sparse cache attention forward");
}
