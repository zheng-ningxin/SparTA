import torch
import sparta
from sparta.opset.cusparse_dynamic_linear import CusparseDynamicLinear

if __name__ == '__main__':
    M = 1024
    K = 2048
    N = 4096
    ori_linear = torch.nn.Linear(K, N, bias=True).cuda()
    c_linear = CusparseDynamicLinear(ori_linear)
    for sparsity_ratio in [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print('Sparsity ratio:', sparsity_ratio)
        data = torch.rand(M, K).cuda()
        mask_weight = torch.randn(N, K).cuda()
        mask = (mask_weight > sparsity_ratio).to(torch.int32)
        c_linear.update_mask(mask)
        out = c_linear(data)
        ref_out = c_linear.ref_forward(data)
        # import ipdb; ipdb.set_trace()
        torch.allclose(out, ref_out, rtol=1e-08, atol=1e-04)
        print('correctness passed')