import triton
import torch
import argparse
import time
import sys

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


def measure_triton(sparsity, M, K, N, block_h, block_w):
    n_block_h = K // block_h
    n_block_w = N // block_w
    block_wise_weight = torch.rand(n_block_h, n_block_w, dtype=torch.float32).cuda()
    block_mask = (block_wise_weight > sparsity).to(torch.int32)
    full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
    device = torch.device('cuda')
    block_size = 32
    conv=torch.nn.Conv2d(1, 1, (block_size, block_size), (block_size, block_size), bias=False).cuda()
    conv.eval()
    conv.weight.data[:] = 1
    mask = full_mask.view(1, 1, full_mask.size(0), full_mask.size(1)).to(torch.float32)
    block_mask = conv(mask)
    layout = (block_mask.view(1, K//block_size, N//block_size)>0).to(torch.int32)
    matmul = triton.ops.blocksparse.matmul(layout, block_size, "dds", trans_a=False, trans_b=False, device=device)
    block_nnz = torch.sum(layout)
    sparse_weight = torch.rand(1, block_nnz, block_size, block_size)
    data = torch.rand(1, 1, M, K)
    out = matmul(data, sparse_weight)
    n_iter = 2000
    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(n_iter):
        out = matmul(data, sparse_weight)
    torch.cuda.synchronize()
    t_end = time.time()
    avg_time = (t_end-t_start)*1000/n_iter
    print(f"Time= {avg_time} ms")

if __name__ == '__main__':
    measure_triton(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))