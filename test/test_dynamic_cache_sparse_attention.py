import torch
import time
import random
import numpy as np
from sparta.opset import *
from sparta.opset.dynamic_sparse_cache_attention import DynamicSparseCacheAttention


def test_correctness(cache_atten, new_q, k, v, max_token_len, paddings):
    out = cache_atten(new_q, k, v, max_token_len)
    ref_out = cache_atten.ref_forward(new_q, k, v, max_token_len)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    HEAD_NUM = 20
    hidden_dim = 64
    paddings = torch.tensor([20, 120, 50, 68, 70, 38, 94, 72], dtype=torch.int32).cuda() # batchsize = 8
    batch_size = paddings.size(0)
    max_token_len = torch.max(paddings)+16
    cache_atten = DynamicSparseCacheAttention(paddings)
    new_q = torch.rand(batch_size, HEAD_NUM, 1, hidden_dim, dtype=torch.float16).cuda()
    k = torch.rand(batch_size, HEAD_NUM, max_token_len+64, hidden_dim, dtype=torch.float16).cuda()
    v = torch.rand_like(k)
    # new_q[:] = 1
    # k[:] = 1
    # v[:] = 1
    test_correctness(cache_atten, new_q, k, v, max_token_len, paddings)