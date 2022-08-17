import torch
import time

if __name__ == '__main__':
    M = N = 1024
    data = torch.rand(M, N).cuda()
    runtime = 10000
    torch.cuda.synchronize()
    t_start = time.time()
    for rid in range(runtime):
        data = data.t().contiguous()
    torch.cuda.synchronize()
    t_end = time.time()
    print("time cost(ms):", (t_end-t_start)*1000/runtime)