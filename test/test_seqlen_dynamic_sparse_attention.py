# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import torch
import random
import numpy as np
from cmath import inf
from sparta.opset import *
from sparta.opset.seqlen_dynamic_sparse_attention import SeqlenDynamicSparseAttention
import joblib

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
    pattern[:512, 0] = 1
    return pattern

def test_speed(sparse_attention, sparse_pattern, head_num, seq_len, hidden_n, device):
    # warmup
    q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    # import ipdb; ipdb.set_trace()

    out = sparse_attention(q, k, v)
    out_grad = torch.rand_like(out)

    torch.cuda.synchronize()
    st = time.time()
    for _ in range(50):
        sparse_attention.set_global_seqlens(sparse_pattern)
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

def convert_to_attention_mask(seq_lens, max_seq_len):
    print("Convert the sequence length pattern according to the max length: ", max_seq_len)
    batch_size = seq_lens.size(0)
    attention_mask = torch.zeros(batch_size, 1, 1, max_seq_len, dtype=torch.int32)
    for bid in range(seq_lens.size(0)):
        cur_len = seq_lens[bid]
        attention_mask.data[bid][0][0][:cur_len] = 1
    return attention_mask


def dense_speed(sparse_attention, seq_len_pattern, head_num, max_seq_len, hidden_n, device):
    # warmup
    attention_mask = convert_to_attention_mask(seq_len_pattern, max_seq_len).to(device)
    q = torch.rand(batch_size, head_num, max_seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    k = torch.rand(batch_size, head_num, max_seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    v = torch.rand(batch_size, head_num, max_seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    out = sparse_attention.reference_forward(q, k, v, attention_mask)
    out_grad = torch.rand_like(out)

    torch.cuda.synchronize()
    st = time.time()
    for _ in range(50):
        q = torch.rand(batch_size, head_num, max_seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        k = torch.rand(batch_size, head_num, max_seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        v = torch.rand(batch_size, head_num, max_seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        out = sparse_attention.reference_forward(q, k, v, attention_mask)
    torch.cuda.synchronize()
    end = time.time()
    print('Dense Forward Implementation', end-st)

def debug_reference_forward(Q, K, V, attention_mask):
    add_mask = torch.zeros(attention_mask.size()).to(Q.device)
    add_mask[attention_mask == 0] = float(-inf)
    dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
    # return dots
    added = torch.add(dots, add_mask)
    # import ipdb; ipdb.set_trace()
    attn = added.softmax(dim=-1)
    nan_pos = torch.isnan(attn)
    attn[nan_pos] = 0.0
    # return attn
    ref_out = torch.einsum('b h m n, b h n k -> b h m k', attn, V)
    return ref_out

def test_correctness(sparse_attention, seq_len_pattern, HEAD_NUM, max_seq_len, hidden_n, device):
    q, k, v = torch.randn(batch_size, HEAD_NUM, max_seq_len, hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, HEAD_NUM, max_seq_len,
                                                                                                           hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, HEAD_NUM, max_seq_len, hidden_n, dtype=torch.float32, device=device)
    # q, k, v = joblib.load('qkv.pkl')
    # q = torch.load('q.pth').to(device)
    # k = torch.load('k.pth').to(device)
    # v = torch.load('v.pth').to(device)
    attention_mask = convert_to_attention_mask(seq_len_pattern, max_seq_len).to(device)
    sparse_attention.set_global_seqlens(seq_len_pattern)
    # test the correctness of the backward function
    q1 = q.clone().detach().to(device)
    q2 = q.clone().detach().to(device)
    k1 = k.clone().detach().to(device)
    k2 = k.clone().detach().to(device)
    v1 = v.clone().detach().to(device)
    v2 = v.clone().detach().to(device)
    q1.requires_grad_()
    q2.requires_grad_()
    k1.requires_grad_()
    k2.requires_grad_()
    v1.requires_grad_()
    v2.requires_grad_()
    out_2 = debug_reference_forward(q2, k2, v2, attention_mask)
    

    out = sparse_attention(q1, k1, v1)

    # import ipdb; ipdb.set_trace()
    # mask the useless token manually here
    for bid in range(seq_len_pattern.size(0)):
        cur_len = seq_len_pattern[bid]
        out_2.data[bid,:,cur_len:,:] = 0
    # print(out.isnan().sum())
    # import ipdb; ipdb.set_trace()
    if not torch.allclose(out, out_2, rtol=1e-08, atol=1e-04):
        import ipdb
        ipdb.set_trace()
    assert torch.allclose(out, out_2, rtol=1e-08, atol=1e-04)
    
    print('Correctness test passed')

def random_seqlen(batchsize, max_seqlen):
    torch.manual_seed(1)
    seqlens = torch.randint(1, max_seqlen, (batchsize,), dtype=torch.int32)
    return seqlens

if __name__ == '__main__':
    batch_size = 8
    max_seq_len = 128
    HEAD_NUM = 12
    hidden_n = 64
    device = torch.device('cuda:0')
    seqlens = random_seqlen(batch_size, max_seq_len).to(device)
    print('Sequence length:', seqlens)
        
    spa = SeqlenDynamicSparseAttention(True)
    SeqlenDynamicSparseAttention.set_global_seqlens(seqlens)
    test_speed(spa, seqlens, HEAD_NUM, max_seq_len, hidden_n, device)
    dense_speed(spa, seqlens, HEAD_NUM, max_seq_len, hidden_n, device)
    test_correctness(spa, seqlens, HEAD_NUM, max_seq_len, hidden_n, device)
