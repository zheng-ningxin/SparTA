import time
import torch
import random
import numpy as np
from cmath import inf
from sparta.opset import *
from sparta.opset.seqlen_dynamic_sparse_linear import SeqlenDynamicSparseLinear
import joblib

def random_seqlen(batchsize, max_seqlen):
    torch.manual_seed(1)
    seqlens = torch.randint(1, max_seqlen, (batchsize,), dtype=torch.int32)
    return seqlens


def test_correstness(spl, seqlens, batch_size, max_seqlen, hidden_n, dtype):
    activation = torch.rand(batch_size, max_seqlen, hidden_n, dtype=dtype)
    a1 = activation.clone().detach().cuda()
    a2 = activation.clone().detach().cuda()
    o1 = spl(a1)
    o2 = spl.reference_forward(a2)
    import ipdb; ipdb.set_trace()
    for bid in range(batch_size):
        cur_seq_len = seqlens[bid]
        assert torch.allclose(o1[bid][:cur_seq_len], o2[bid][:cur_seq_len], rtol=1e-08, atol=1e-04)
    print('correctness passed')

def sparse_speed(spl, seqlens, batch_size, max_seqlen, hidden_n):
    activation = torch.rand(batch_size, max_seqlen, hidden_n).cuda()
    runtimes = 100
    torch.cuda.synchronize()
    st = time.time()
    for i in range(runtimes):
        spl.set_global_seqlens(seqlens)
        o1 = spl(activation)
    torch.cuda.synchronize()
    end = time.time()
    print('Sparse Forward Implementation', end-st)

def dense_speed(spl, seqlens, batch_size, max_seqlen, hidden_n):
    activation = torch.rand(batch_size, max_seqlen, hidden_n).cuda()
    runtimes = 100
    torch.cuda.synchronize()
    st = time.time()
    for i in range(runtimes):
        o2 = spl.reference_forward(activation)
    torch.cuda.synchronize()
    end = time.time()
    print('Dense Forward Implementation', end-st)

if __name__ == '__main__':
    batch_size = 8
    max_seqlen = 128
    hidden_n = 768
    test_type = torch.float16
    seqlens = random_seqlen(batch_size, max_seqlen).cuda()

    ori_linear = torch.nn.Linear(hidden_n, hidden_n, bias=True).cuda().to(test_type)
    spl = SeqlenDynamicSparseLinear(ori_linear, True)
    SeqlenDynamicSparseLinear.set_global_seqlens(seqlens)
    print(seqlens)
    # sparse_speed(spl, seqlens, batch_size, max_seqlen, hidden_n)
    # dense_speed(spl, seqlens, batch_size, max_seqlen, hidden_n)
    test_correstness(spl, seqlens, batch_size, max_seqlen, hidden_n, dtype=test_type)

