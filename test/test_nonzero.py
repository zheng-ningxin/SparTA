import torch
import time
if __name__ == '__main__':
    runtime = 10000
    M = 1024
    mask = torch.ones(M, dtype=torch.int32).cuda()
    mask[:512]=0
    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(runtime):
        _pos = mask.nonzero(as_tuple=True)[0]
    torch.cuda.synchronize()
    t_end = time.time()
    print("Time Cost(ms)", (t_end-t_start)*1000/runtime)
    import ipdb; ipdb.set_trace()
    pass
