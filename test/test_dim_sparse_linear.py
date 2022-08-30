from audioop import avg
import torch
import sparta
import time
from sparta.opset.dim_dynamic_sparse_linear import DimDynamicLinear

def measure_time(module, data):
    runtime = 5000
    in_data = data.clone().detach()
    in_data.requires_grad_()
    #warmup
    for i in range(100):
        re = module(in_data)
    _grad = torch.rand_like(re)
    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(runtime):
        re = module(in_data)
        re.backward(_grad)
    torch.cuda.synchronize()
    t_end = time.time()
    avg_t = (t_end-t_start)*1000/runtime
    print('Dense Time(ms):', avg_t)
    
def measure_time_sparse_profile(module, data, mask):
    runtime = 15000
    in_data = data.clone().detach()
    in_data.requires_grad_()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=20),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/dim_sparse_linear_v2'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        #warmup
        for i in range(100):
            re = module(in_data, mask)
        _grad = torch.rand_like(re)
        torch.cuda.synchronize()
        t_start = time.time()
        for i in range(runtime):
            re = module(in_data, mask)
            re.backward(_grad)
            prof.step()
        torch.cuda.synchronize()
        t_end = time.time()
        avg_t = (t_end-t_start)*1000/runtime
        print('Sparse Time(ms):', avg_t)

def measure_time_sparse(module, data, mask):
    runtime = 15000
    in_data = data.clone().detach()
    in_data.requires_grad_()
    #warmup
    for i in range(100):
        re = module(in_data, mask)
    _grad = torch.rand_like(re)
    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(runtime):
        re = module(in_data, mask)
        re.backward(_grad)
    torch.cuda.synchronize()
    t_end = time.time()
    avg_t = (t_end-t_start)*1000/runtime
    print('Sparse Time(ms):', avg_t)
    

if __name__ == '__main__':
    M = 1024
    K = 1024
    N = 1024
    
    # for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    for sparsity in [ 0.95]:
        ori_linear = torch.nn.Linear(K, N).cuda()
        d_linear = DimDynamicLinear(ori_linear, 0)
        out_c_w = torch.rand(N).cuda()
        c_mask = (out_c_w > sparsity).to(torch.int32)
        print('Sparsity Ratio:', torch.sum(c_mask)/c_mask.numel())
        data = torch.rand(M, K).cuda()
        # measure_time(ori_linear, data)
        measure_time_sparse_profile(d_linear, data, c_mask)
        
