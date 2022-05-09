# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import torch
import random
from SparTA.OPs import *

from SparTA.OPs.DynamicSparseAttention import DynamicSparseAttention


def random_sparse_pattern(seq_len, sparsity):
    pattern = torch.zeros(seq_len, seq_len, dtype=torch.int32)
    nnz = int(seq_len * seq_len * sparsity)
    print("NNZ: ", nnz)
    for _ in range(nnz):
        i, j = random.randint(0, seq_len-1), random.randint(0, seq_len-1)
        pattern[i][j] = 1
    return pattern

def random_sparse_pattern_v2(seq_len, sparsity):
    pattern = torch.zeros(seq_len, seq_len, dtype=torch.int32)
    pattern[:900, 0] = 1
    return pattern

def test_speed(sparse_attention, head_num, seq_len, hidden_n, device):
    # warmup
    q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    out = sparse_attention(q, k, v)
    out_grad = torch.rand_like(out)

    torch.cuda.synchronize()
    st = time.time()
    for _ in range(50):
        q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        sparse_attention(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    print('Sparse Forward Implementation', end-st)



def dense_speed(sparse_attention, head_num, seq_len, hidden_n, device):
    # warmup
    q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    out = sparse_attention.reference_forward(q, k, v)
    out_grad = torch.rand_like(out)

    torch.cuda.synchronize()
    st = time.time()
    for _ in range(50):
        q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        out = sparse_attention.reference_forward(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    print('Dense Forward Implementation', end-st)



def test_correctness(sparse_attention, HEAD_NUM, seq_len, hidden_n, device):
    q, k, v = torch.randn(batch_size, HEAD_NUM, seq_len, hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, HEAD_NUM, seq_len,
                                                                                                                    hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, HEAD_NUM, seq_len, hidden_n, dtype=torch.float32, device=device)

    # test the correctness of the backward function
    q1 = q.clone().detach()
    q2 = q.clone().detach()
    k1 = k.clone().detach()
    k2 = k.clone().detach()
    v1 = v.clone().detach()
    v2 = v.clone().detach()
    q1.requires_grad_()
    q2.requires_grad_()
    k1.requires_grad_()
    k2.requires_grad_()
    v1.requires_grad_()
    v2.requires_grad_()
    out_2 = sparse_attention.reference_forward(q2, k2, v2)
    in_grad = torch.rand_like(out_2)
    out = sparse_attention(q1, k1, v1)
    # tmp_attn.retain_grad()
    # out_2.backward(in_grad)
    # out.backward(in_grad)
    
    if not torch.allclose(out, out_2, rtol=1e-08, atol=1e-04):
        import pdb
        pdb.set_trace()
    assert torch.allclose(out, out_2, rtol=1e-08, atol=1e-04)
    
    print('Correctness test passed')



def test_random(HEAD_NUM, seq_len, hidden_dim, sparsity):
    print(HEAD_NUM, seq_len, hidden_dim, sparsity)
    sp_pattern = random_sparse_pattern(seq_len, sparsity)
    # sp_pattern = random_sparse_pattern_v2(seq_len, sparsity)
    M, N = sp_pattern.size()
    K = hidden_dim
    device = torch.device('cuda:0')
    sp_pattern = sp_pattern.cuda()
    spa = DynamicSparseAttention(True)
    DynamicSparseAttention.set_global_sparse_pattern(sp_pattern)
    # test the speed
    test_correctness(spa, HEAD_NUM, M, K, device)
    # test_speed(spa, HEAD_NUM, M, K, device)


if __name__ == '__main__':
    batch_size = 2
    test_random(20, 1024, 128, 0.999)
    # test_random(20, 1024, 64, 0.00001)
