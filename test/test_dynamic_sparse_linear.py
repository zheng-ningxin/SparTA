# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import torch
import time
import random
import numpy as np
from sparta.opset.dynamic_sparse_linear import *
from sparta.common.utils import verify_bcsr
from sparta.opset.bcsr_converter import BcsrConverter



def test_speed(spl, input, mask):
    runtime = 2000
    tmp_grad = torch.rand_like(input)
    input.requires_grad_()
    torch.cuda.synchronize()
    st = time.time()
    for rid in range(runtime):
        spl.update_mask(mask)
        out = spl(input)
        out.backward(tmp_grad)
    torch.cuda.synchronize()
    end = time.time()
    total_block_nnz = spl.ori_linear.weight.numel() / spl.block_h / spl.block_w 
    print('Forward Sparsity:{} Backward Sparsity:{}'.format(spl.csr_row[-1]/total_block_nnz, spl.grad_csr_row[-1]/total_block_nnz))
    print('csr_row_size:{} grad_csr_size:{}'.format(spl.csr_row.size(), spl.grad_csr_row.size()))
    print("Sparse speed: ", (end-st)/runtime*1000)
        
def dense_speed(linear, input):
    runtime = 1000
    tmp_grad = torch.rand_like(input)
    input.requires_grad_()

    torch.cuda.synchronize()
    st = time.time()
    for rid in range(runtime):
        out = linear(input)
        out.backward(tmp_grad)
    torch.cuda.synchronize()
    end = time.time()
    print("Dense speed: ", (end-st)/runtime*1000)
    
def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size, dim=0)
    full_mask = full_mask.repeat_interleave(block_size, dim=1)
    return full_mask

def random_sparse_pattern_block(M, N, sparsity, block_h, block_w):
    pattern = torch.zeros(M, N, dtype=torch.int32)
    b_nnz = int(M * N //block_h //block_w*sparsity)
    print("Block_nnz: ", b_nnz)
    block_pos = [(i, j) for i in range(0, N//block_h) for j in range(0, N//block_w)]
    random.shuffle(block_pos)
    # import ipdb; ipdb.set_trace()
    remain_pos = block_pos[:b_nnz] 
    debug = []
    for i, j in remain_pos:
        i_start = i * block_h
        i_end = i_start+ block_h
        j_start = j * block_w
        j_end = j_start + block_w
        pattern[i_start:i_end, j_start:j_end] = 1
        debug.append((i_start,i_end, j_start,j_end))
    remaining_ratio =  torch.sum(pattern)/pattern.numel()
    if abs(remaining_ratio.item()-sparsity) > 0.01:
        import ipdb; ipdb.set_trace()
    print('Remaining ratio: ', torch.sum(pattern)/pattern.numel())
    return pattern

def test_correctness():
    linear = torch.nn.Linear(2048, 1024, bias=True).cuda()
    d_linear = DynamicSparseLinear(linear)
    ori_linear = copy.deepcopy(linear)

    data_1 = torch.rand(1024, 2048).cuda()
    data_1.requires_grad_()
    data_2 = copy.deepcopy(data_1)
    data_2.requires_grad_()

    out1 = d_linear(data_1)
    grad = torch.ones_like(out1)
    print(grad.size())
    out1.backward(grad)
    out2 = ori_linear(data_2)
    out2.backward(grad)
    assert torch.allclose(out1, out2, rtol=1e-08, atol=1e-04)
    # import ipdb; ipdb.set_trace()
    verify_bcsr(d_linear.mask, d_linear.ori_linear.weight, d_linear.csr_row, d_linear.csr_col, d_linear.csr_val, d_linear.block_h, d_linear.block_w)
    verify_bcsr(d_linear.mask.t(), d_linear.ori_linear.weight.t(), d_linear.grad_csr_row, d_linear.grad_csr_col, d_linear.grad_csr_val, d_linear.block_h, d_linear.block_w)
    
    assert torch.allclose(d_linear.ori_linear.weight.grad, ori_linear.weight.grad, rtol=1e-08, atol=1e-04)
    assert torch.allclose(data_1.grad, data_2.grad,  rtol=1e-08, atol=1e-04)
    print("test correctness passed")
    
if __name__ == '__main__':
    in_dim = 1024
    out_dim = 1024
    batch_size = 1024
    block_h = 64
    block_w = 32
    ori_linear = torch.nn.Linear(in_dim, out_dim, bias=True).cuda()
    d_linear = DynamicSparseLinear(ori_linear)
    data = torch.rand(batch_size, in_dim).cuda()
    # for sparsity_ratio in np.arange(0.1, 1, 0.1):
    dense_speed(ori_linear, data) # warm up
    for sparsity_ratio in [0.05]:
        sp_pattern = random_sparse_pattern_block(out_dim, in_dim, sparsity_ratio, block_h, block_w).cuda()
        print('Sparsity ratio:', sparsity_ratio)
        test_speed(d_linear, data, sp_pattern)
        # dense_speed(ori_linear, data)
        # test_correctness()
        # dense_speed(ori_linear, data)