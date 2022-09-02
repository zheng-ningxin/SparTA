import torch
import time
import sparta
from sparta.opset.dim_dynamic_sparse_linear import DimDynamicLinear

def test_correctness(d_linear, data, mask):
    ref_out = d_linear.ref_forward(data, mask)
    out = d_linear(data, mask)
    # import ipdb; ipdb.set_trace()
    flag = torch.allclose(ref_out, out, rtol=1e-08, atol=1e-04)
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


if __name__ == '__main__':
    batch_size = 8
    seq_len = 128
    K = 1024
    N = 1024
    ori_linear = torch.nn.Linear(K, N).cuda()
    # ori_linear.weight.data[:] = 1
    # ori_linear.bias.data[:] = 0
    d_linear = DimDynamicLinear(ori_linear, 0)
    
    for sparsity_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    # for sparsity_ratio in [ 0.5]:
        print('Sparsity ratio:', sparsity_ratio)
        data = torch.rand(batch_size, seq_len, K).cuda()
        mask_w = torch.rand(N).cuda()
        c_mask = (mask_w > sparsity_ratio).to(torch.int32)
        test_correctness(d_linear, data, c_mask)
        test_speed(d_linear, data, c_mask)
        dense_speed(d_linear, data, c_mask)