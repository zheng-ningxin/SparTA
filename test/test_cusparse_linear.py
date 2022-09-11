import torch
import sparta
from sparta.opset.cusparse_dynamic_linear import CusparseDynamicLinear

if __name__ == '__main__':
    pass
    M = 1024
    K = 1024
    N = 1024
    ori_linear = torch.nn.Linear(K, N, bias=True).cuda()
    c_linear = CusparseDynamicLinear(ori_linear)
    for sparsity_ratio in [0.5]:
        data = torch.rand(M, K).cuda()
        mask_weight = torch.randn(N, K).cuda()
        mask = (mask_weight > sparsity_ratio).to(torch.int32)
        c_linear.update_mask(mask)
        out = c_linear(data)
        ref_out = c_linear.ref_forward(data)
        import ipdb; ipdb.set_trace()