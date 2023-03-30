import torch
import time
from sparta.opset.input_sparse_linear import InSparseLinear

for sparsity in [0.98, 0.985, 0.99, 0.995, 0.999]:
    linear = torch.nn.Linear(3072, 768, bias=False).cuda()
    in_linear = InSparseLinear(linear)
    # data = torch.ones(32, 128, 3072).cuda()
    data = torch.rand(32, 128, 3072).cuda()
    seqlens = torch.zeros(32, dtype=torch.int32, device=data.device)
    seqlens[:]=64
    mask = data < sparsity
    data[mask] = 0
    # RUNTIME = 10000
    # torch.cuda.synchronize()
    # t_start = time.time()
    # for _ in range(RUNTIME):
    #     ref_out = linear(data)
    # torch.cuda.synchronize()
    # t_end = time.time()
    # print('Ori Dense time: ', (t_end-t_start)*1000/RUNTIME)

    # RUNTIME = 10000
    # torch.cuda.synchronize()
    # t_start = time.time()
    # for _ in range(RUNTIME):
    #     tmp = data.view(32*128, 3072).t().contiguous()
    # torch.cuda.synchronize()
    # t_end = time.time()
    # print('Transpose time: ', (t_end-t_start)*1000/RUNTIME)

    # in_linear.set_global_seqlens(seqlens)
    print('Sparsity', sparsity)
    RUNTIME = 1000
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(RUNTIME):
        in_linear(data)
    torch.cuda.synchronize()
    t_end = time.time()
    print(in_linear.row[4096//32]/(4096*96))
    print('Convert time: ', (t_end-t_start)*1000/RUNTIME)
# import ipdb; ipdb.set_trace()