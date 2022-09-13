import torch
import sparta
from sparta.opset.cusparse_dynamic_linear import CusparseDynamicLinear

if __name__ == '__main__':
    M = 1024
    K = 2048
    N = 4096
    for sparsity_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ori_linear = torch.nn.Linear(K, N, bias=True).cuda()
        ori_linear.weight.data.uniform_(0.1, 1)
        c_linear = CusparseDynamicLinear(ori_linear)

        print('Sparsity ratio:', sparsity_ratio)
        data = torch.rand(M, K).cuda()
        data_1 = data.clone().detach()
        data_1.requires_grad_()
        data_2 = data.clone().detach()
        data_2.requires_grad_()
        mask_weight = torch.randn(N, K).cuda()
        mask = (mask_weight > sparsity_ratio).to(torch.int32)
        c_linear.update_mask(mask)
        ori_linear.weight.data *= mask
        out = c_linear(data_1)
        ref_out = ori_linear(data_2)
        tmp_grad = torch.rand_like(out)
        out.backward(tmp_grad)
        ref_out.backward(tmp_grad)
        # import ipdb; ipdb.set_trace()
        flag = True
        flag = flag and torch.allclose(out, ref_out, rtol=1e-08, atol=1e-03)
        # import ipdb; ipdb.set_trace()
        flag = flag and torch.allclose(data_1.grad, data_2.grad, rtol=1e-08, atol=1e-03)
        # import ipdb; ipdb.set_trace()
        flag = flag and torch.allclose(ori_linear.weight.grad * mask, c_linear.weight.grad, rtol=1e-08, atol=1e-03)
        if not flag:
            import ipdb; ipdb.set_trace()
        print('correctness passed')