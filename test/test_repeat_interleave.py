import torch
import time

def expand(ori_t, h_dim, w_dim):
    return ori_t.repeat_interleave(h_dim, 0).repeat_interleave(w_dim, 1)


if __name__ == '__main__':
    h = w = 32
    runtime = 1000000
    tmp_mask = torch.ones(h, w).cuda()
    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(runtime):
        expand(tmp_mask, 32, 32)
    torch.cuda.synchronize()
    t_end = time.time()
    print("Time cost(ms): ", (t_end-t_start)*1000/runtime)
    