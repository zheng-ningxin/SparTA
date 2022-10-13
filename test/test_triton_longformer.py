import torch
import time
import random
import sparta
import joblib
from sparta.opset.triton_dynamic_sparse_attention import TritonDynamicAttention

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size, dim=0)
    full_mask = full_mask.repeat_interleave(block_size, dim=1)
    return full_mask

if __name__ == "__main__":
    device = torch.device('cuda:0')
    HEAD_NUM = 12
    seqlen = 2048
    hidden_dim = 64
    bs = 1
    block_size = 32
    mask = joblib.load('mask_pattern.pkl')
    dynamic_global_attention = mask['dynamic']
    static_local_attention = mask['attention_mask'][0]
    static_local_attention[:,dynamic_global_attention.to(torch.long)] = 1
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
    head_full_mask = static_local_attention.repeat(HEAD_NUM, 1).view(HEAD_NUM, seqlen, seqlen)
    tda.set_global_mask(head_full_mask, True, 32, 32, HEAD_NUM)
    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(run_times):
        out = tda(q, k, v, head_full_mask)    
    torch.cuda.synchronize()
    t_end = time.time()
    print("Time: {}ms".format((t_end-t_start)*1000/run_times))