import torch
import time
import sparta
from sparta.opset.bcsr_converter import BcsrConverter
import openai_bmm_cpp

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


if __name__ == '__main__':
    batchsize = 1
    M = 1024
    K = 1024
    N = 1024
    block_h = 32
    block_w = 64
    sparsity_ratio = 0.0
    RUNTIME = 10000
    A = torch.rand(batchsize, M, K).cuda()
    A_copy = A.clone().detach()
    B = torch.rand(batchsize, K, N).cuda()
    mask = torch.ones(M, K).cuda()
    block_wise_weight = torch.rand(M//block_h, K//block_w, dtype=torch.float32).cuda()
    block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
    print('Block-wise sparsity ratio:', torch.sum(block_mask)/block_mask.numel())
    full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
    A *= full_mask
    ref_out = torch.einsum('bmk,bkn->bmn',A, B)
    converter_1 = BcsrConverter()
    row_ptr, col_idx, row_pos, vals = converter_1(full_mask, A, block_h, block_w)
    block_nnz = row_ptr[M//block_h]
    out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
    if not torch.allclose(out, ref_out, rtol=1e-08, atol=1e-03):
        import ipdb; ipdb.set_trace()
    assert torch.allclose(out, ref_out, rtol=1e-08, atol=1e-03)
    # measure the latency of the original openai kernel
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(RUNTIME):
        out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
    t_end = time.time()
    print('Original openai bmm latency(ms):', (t_end-t_start)*1000/RUNTIME)
    new_block_h = block_h
    new_block_w = 1
    converter_2 = BcsrConverter(True)
    t_block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
    t_block_mask = (t_block_wise_weight > sparsity_ratio).to(torch.int32)
    print("Block-wise sparsity ratio:", torch.sum(t_block_mask)/t_block_mask.numel())
    t_full_mask = convert_to_full_mask(t_block_mask, (new_block_h, new_block_w))
    # print(torch.squeeze(A).size())
    t_row_ptr, t_col_idx, t_row_pos, t_vals = converter_2(t_full_mask, torch.squeeze(A), new_block_h, new_block_w)
    t_block_nnz = t_row_ptr[M//new_block_h]
    condense_out = openai_bmm_cpp.forward_condense(t_row_ptr, t_col_idx, t_vals, B, M, K, N, new_block_h, new_block_w, batchsize, t_block_nnz)
    import ipdb; ipdb.set_trace()