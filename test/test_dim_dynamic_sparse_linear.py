import torch
import time
import copy
import random
import sparta
from sparta.opset.dim_dynamic_sparse_linear import DimDynamicLinear

def test_correctness(d_linear, data, mask):
    data_1 = data.clone().detach()
    data_2 = data.clone().detach()
    data_1.requires_grad_()
    data_2.requires_grad_()
    d_linear2 = copy.deepcopy(d_linear)
    ref_out = d_linear2.ref_forward(data_2, mask)
    out = d_linear(data_1, mask)
    tmp_grad = torch.rand_like(ref_out)
    # tmp_grad = torch.ones_like(ref_out)
    # import ipdb; ipdb.set_trace()
    ref_out.backward(tmp_grad)
    out.backward(tmp_grad)
    flag = torch.allclose(ref_out, out, rtol=1e-08, atol=1e-03)
    # import ipdb; ipdb.set_trace()
    flag = flag and torch.allclose(d_linear.weight.grad, d_linear2.weight.grad, rtol=1e-08, atol=1e-03)
    flag = flag and torch.allclose(data_1.grad, data_2.grad, rtol=1e-08, atol=1e-03)
    # import ipdb; ipdb.set_trace()
    if not flag:
        import ipdb; ipdb.set_trace()

def test_speed(d_linear, data, mask):
    run_time = 10000
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(run_time):
        re = d_linear(data, mask)
    torch.cuda.synchronize()
    t_end = time.time()
    print('Sparse per batch(ms):' , (t_end-t_start)*1000/run_time)
    

def dense_speed(d_linear, data, mask):
    run_time = 10000
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(run_time):
        re = d_linear.ref_forward(data, mask)
    torch.cuda.synchronize()
    t_end = time.time()
    print('Dense per batch(ms):' , (t_end-t_start)*1000/run_time)


def random_mask_64(total_c, sparsity_ratio):
    mask = torch.ones(total_c, dtype=torch.int32).cuda()
    num = int((total_c * sparsity_ratio) // 64 *64)
    pos = random.sample(list(range(total_c)), num)
    mask[pos] = 0
    return mask



if __name__ == '__main__':
    batch_size = 8
    seq_len = 128
    K = 1024
    N = 1024
    ori_linear = torch.nn.Linear(K, N).cuda()
    # ori_linear.weight.data[:] = 1
    # ori_linear.bias.data[:] = 0
    # d_linear = DimDynamicLinear(ori_linear, 0)
    
    # for sparsity_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    # # for sparsity_ratio in [0]:
    #     print('Sparsity ratio:', sparsity_ratio)
    #     data = torch.rand(batch_size, seq_len, K).cuda()
    #     # mask_w = torch.rand(N).cuda()
    #     # c_mask = (mask_w > sparsity_ratio).to(torch.int32)
    #     c_mask = random_mask_64(N, sparsity_ratio)
    #     # print(torch.sum(c_mask)%64)
    #     test_correctness(d_linear, data, c_mask)
    #     # test_speed(d_linear, data, c_mask)
    #     # dense_speed(d_linear, data, c_mask)
    
    d_linear = DimDynamicLinear(ori_linear, 1)
    
    for sparsity_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    # for sparsity_ratio in [0]:
        print('Sparsity ratio:', sparsity_ratio)
        c_mask = random_mask_64(N, sparsity_ratio)
        data = torch.rand(batch_size, seq_len, torch.sum(c_mask)).cuda()

        test_correctness(d_linear, data, c_mask)
        # test_speed(d_linear, data, c_mask)
        # dense_speed(d_linear, data, c_mask)