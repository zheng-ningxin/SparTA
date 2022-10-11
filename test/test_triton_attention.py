
import torch
import time
import random
import sparta
from sparta.opset.triton_dynamic_sparse_attention import TritonDynamicAttention

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size, dim=0)
    full_mask = full_mask.repeat_interleave(block_size, dim=1)
    return full_mask

if __name__ == "__main__":
    device = torch.device('cuda:0')
    HEAD_NUM = 1
    seqlen = 1024
    hidden_dim = 64
    bs = 16
    block_size = 32
    block_wise_weight = torch.zeros(seqlen//block_size, seqlen//block_size, dtype=torch.float32, device=device)
    
    for sparsity_ratio in [0.1, 0.2, 0.3, 0.5 , 0.7, 0.9, 0.95]:
        run_times = 1000
        triton_time = []
        dynamic_time = []
        tda = TritonDynamicAttention(block_size, block_size, HEAD_NUM, global_model=True)
        q = torch.rand((bs, HEAD_NUM, seqlen, hidden_dim), dtype = torch.float32, device = device)
        k = torch.rand((bs, HEAD_NUM, seqlen, hidden_dim), dtype = torch.float32, device = device)
        v = torch.rand((bs, HEAD_NUM, seqlen, hidden_dim), dtype = torch.float32, device = device)
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        block_wise_weight = torch.rand(seqlen//block_size, seqlen//block_size, dtype=torch.float32, device=device)
        block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
        print('Sparsity ratio: ', torch.sum(block_mask)/block_mask.numel())
        full_mask = convert_to_full_mask(block_mask, block_size)
        head_full_mask = full_mask.repeat(HEAD_NUM, 1).view(HEAD_NUM, seqlen, seqlen)
        tda.set_global_mask(head_full_mask, True, 32, 32, HEAD_NUM)
        torch.cuda.synchronize()
        t_start = time.time()
        for i in range(run_times):
            out = tda(q, k, v, head_full_mask)    
        torch.cuda.synchronize()
        t_end = time.time()
        print("Time: {}ms".format((t_end-t_start)*1000/run_times))