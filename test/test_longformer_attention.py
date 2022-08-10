# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import torch
import random
import copy
import numpy as np
import joblib
from sparta.opset import *
from sparta.opset.longformer_sparse_attention import LongformerSparseAttention

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

def random_sparse_pattern_block(seq_len, sparsity, block_h, block_w):
    pattern = torch.zeros(seq_len, seq_len, dtype=torch.int32)
    # b_map = torch.zeros(seq_len//block_h, seq_len//block_w, dtype=torch.int32)
    b_nnz = int(seq_len * seq_len //block_h //block_w*sparsity)
    print("Block_nnz: ", b_nnz)
    block_pos = [(i, j) for i in range(0, seq_len//block_h) for j in range(0, seq_len//block_w)]
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

# def test_speed(sparse_attention, sparse_pattern, head_num, seq_len, hidden_n, device):
#     # warmup
#     q = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                    dtype=torch.float32, device=device)
#     k = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                    dtype=torch.float32, device=device)
#     v = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                    dtype=torch.float32, device=device)
#     # import ipdb; ipdb.set_trace()

#     out = sparse_attention(q, k, v)
#     out_grad = torch.rand_like(out)

#     torch.cuda.synchronize()
#     st = time.time()
#     for _ in range(50):
#         sparse_attention.set_global_sparse_pattern(sparse_pattern)
#         q = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                        dtype=torch.float32, device=device)
#         k = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                        dtype=torch.float32, device=device)
#         v = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                        dtype=torch.float32, device=device)
#         sparse_attention(q, k, v)
#     torch.cuda.synchronize()
#     end = time.time()
#     print('Sparse Forward Implementation', (end-st)/50*1000)



# def dense_speed(sparse_attention, head_num, seq_len, hidden_n, device):
#     # warmup
#     q = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                    dtype=torch.float32, device=device)
#     k = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                    dtype=torch.float32, device=device)
#     v = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                    dtype=torch.float32, device=device)
#     out = sparse_attention.reference_forward(q, k, v)
#     out_grad = torch.rand_like(out)

#     torch.cuda.synchronize()
#     st = time.time()
#     for _ in range(50):
#         q = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                        dtype=torch.float32, device=device)
#         k = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                        dtype=torch.float32, device=device)
#         v = torch.rand(batch_size, head_num, seq_len, hidden_n,
#                        dtype=torch.float32, device=device)
#         out = sparse_attention.reference_forward(q, k, v)
#     torch.cuda.synchronize()
#     end = time.time()
#     print('Dense Forward Implementation', (end-st)/50*1000)



def test_correctness(sparse_attention, static_local_attention, dynamic_global_attention, HEAD_NUM, seq_len, hidden_n, device):
    q, k, v = torch.randn(batch_size, HEAD_NUM, seq_len, hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, HEAD_NUM, seq_len,
                        hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, HEAD_NUM, seq_len, hidden_n, dtype=torch.float32, device=device)

    # test the correctness of the backward function
    q1 = q.clone().detach()
    q2 = q.clone().detach()
    k1 = k.clone().detach()
    k2 = k.clone().detach()
    v1 = v.clone().detach()
    v2 = v.clone().detach()

    full_attention_pattern = copy.deepcopy(static_local_attention)
    # import ipdb; ipdb.set_trace()
    full_attention_pattern[:, dynamic_global_attention.to(torch.long)] = 1
    out_2 = sparse_attention.reference_forward(q2, k2, v2, full_attention_pattern)
    # in_grad = torch.rand_like(out_2)
    sparse_attention.set_global_static_attention(static_local_attention)
    sparse_attention.set_global_dynamic_attention(dynamic_global_attention)
    out = sparse_attention(q1, k1, v1)
    

    if not torch.allclose(out, out_2, rtol=1e-08, atol=1e-04):
        import ipdb
        ipdb.set_trace()
    assert torch.allclose(out, out_2, rtol=1e-08, atol=1e-04)
    
    print('Correctness test passed')



def test_random(HEAD_NUM, seq_len, hidden_dim, sparsity):
    print(HEAD_NUM, seq_len, hidden_dim, sparsity)
    # sp_pattern = random_sparse_pattern(seq_len, sparsity)
    sp_pattern = random_sparse_pattern_v2(seq_len, sparsity)
    M, N = sp_pattern.size()
    K = hidden_dim
    device = torch.device('cuda:0')
    sp_pattern = sp_pattern.cuda()
    spa = DynamicSparseAttention(True)
    DynamicSparseAttention.set_global_sparse_pattern(sp_pattern)
    # test the speed
    test_correctness(spa, HEAD_NUM, M, K, device)
    # test_speed(spa, HEAD_NUM, M, K, device)

def test_longformer():
    head_num =  12
    hidden_n = 64
    seq_len =  4096
    batch_size = 1
    device = torch.device('cuda:0')
    data = joblib.load('longformer_inputs.pkl')
    dynamic_global_attention = data['dynamic']
    static_local_attention = data['attention_mask'][0]
    q, k, v = torch.randn(batch_size, head_num, seq_len, hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, head_num, seq_len,
                        hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, head_num, seq_len, hidden_n, dtype=torch.float32, device=device)
    spa = LongformerSparseAttention(True)
    spa.set_global_static_attention(static_local_attention)
    # test the correctness of the backward function
    for run_id in range(1000):
        spa.set_global_dynamic_attention(dynamic_global_attention)
        spa(q, k, v)

if __name__ == '__main__':
    test_longformer()
# if __name__ == '__main__':
#     batch_size = 2
    
#     # test_random(20, 1024, 128, 0.999)
#     # exit()
#     torch.manual_seed(1)
#     random.seed(1)
#     seq_len = 2048
#     HEAD_NUM = 20
#     block_h, block_w = 32, 32
#     hidden_n = 128
#     device = torch.device('cuda:0')
#     for sparsity in [0.05, 0.1, 0.15]:
        
#         static_local_pattern =  random_sparse_pattern_block(seq_len, sparsity, block_h, block_w).cuda()
#         dynamic_global_attention = torch.tensor(random.sample(list(range(seq_len)), int(seq_len*sparsity*0.1))).to(torch.int32).cuda()
#         spa = LongformerSparseAttention(True)
#         # import ipdb; ipdb.set_trace()
#         # test_speed(spa, sp_pattern, HEAD_NUM, seq_len, hidden_n, device)
#         # dense_speed(spa, HEAD_NUM, seq_len, hidden_n, device)
#         test_correctness(spa, static_local_pattern, dynamic_global_attention, HEAD_NUM, seq_len, hidden_n, device)

