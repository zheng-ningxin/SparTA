import torch
import time
import sparta
from sparta.opset.bcsr_converter import BcsrConverter
from sparta.common.utils import convert_bcsr
import openai_bmm_cpp
import sys

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


if __name__ == '__main__':
    batchsize = 1
    sparsity_ratio, M, K, N, block_h, block_w = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    # for M, K, N in [(4096, 4096, 4096), (4096, 768, 4096)]:

    RUNTIME = 2000
    tile_block_h = 32
    tile_block_w = 64 
    n_block_h = M // block_h
    n_block_w = K // block_w
    block_wise_weight = torch.rand(n_block_h, n_block_w, dtype=torch.float32).cuda()
    block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
    full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
    device = torch.device('cuda')
   
    conv=torch.nn.Conv2d(1, 1, (tile_block_h, tile_block_w), (tile_block_h, tile_block_w), bias=False).cuda()
    conv.eval()
    conv.weight.data[:] = 1
    mask = full_mask.view(1, 1, full_mask.size(0), full_mask.size(1)).to(torch.float32)
    block_mask = conv(mask)
    layout = (block_mask.view(M//tile_block_h, N//tile_block_w)>0).to(torch.int32)
    #######################################################################
    # original openai sparse kernel
    A = torch.rand(batchsize, M, K).cuda()
    A_copy = A.clone().detach()
    B = torch.rand(batchsize, K, N).cuda()
    mask = torch.ones(M, K).cuda()
    block_mask = layout
    # print('Block-wise sparsity ratio:', torch.sum(block_mask)/block_mask.numel())
    full_mask = convert_to_full_mask(block_mask, (tile_block_h, tile_block_w))
    print('real sparsity ratio: {}\n'.format(torch.sum(full_mask)/full_mask.numel()))
    A *= full_mask
    ref_out = torch.einsum('bmk,bkn->bmn',A, B)
    converter_1 = BcsrConverter()
    row_ptr, col_idx, row_pos, vals = converter_1(full_mask, A, tile_block_h, tile_block_w)
    block_nnz = row_ptr[M//block_h]
    out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
    if not torch.allclose(out, ref_out, rtol=1e-04, atol=1e-03):
        import ipdb; ipdb.set_trace()
    assert torch.allclose(out, ref_out, rtol=1e-04, atol=1e-03)
    # measure the latency of the original openai kernel
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(RUNTIME):
        out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
    torch.cuda.synchronize()
    t_end = time.time()
    t_re = (t_end-t_start)*1000/RUNTIME
    print(f"Time= {t_re} ms\n")