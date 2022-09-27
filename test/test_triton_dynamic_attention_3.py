import torch
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
    
    sparsity_ratio =  0.1
    run_times = 100
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
    full_mask = convert_to_full_mask(block_mask, block_size)
    head_full_mask = full_mask.repeat(HEAD_NUM, 1).view(HEAD_NUM, seqlen, seqlen).cuda()
    # import ipdb; ipdb.set_trace()
    tda.set_global_mask(head_full_mask, True, block_size, block_size, HEAD_NUM)
    out = tda(q, k, v, None)    
    tmp_grad = torch.rand_like(out)
    out.backward(tmp_grad)
    import ipdb; ipdb.set_trace()