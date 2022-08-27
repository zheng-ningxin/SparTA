import sparta
import torch
import time
import random
from sparta.opset.blockwise_dynamic_sparse_linear import BlockwiseSparseLinear

    
def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
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

def test_corressness(data, block_mask, ori_linear, b_linear, block_h=32, block_w=64):
    full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
    ori_linear.weight.data *= full_mask.data
    ref_out = ori_linear(data)
    out = b_linear(data, block_mask)
    # import ipdb; ipdb.set_trace()
    assert torch.allclose(out, ref_out, rtol=1e-08, atol=1e-03)
    

if __name__ == '__main__':
    B = 16
    S = 256
    K = 2048
    N = 2048
    block_h = 32
    block_w = 64
    for sparsity_ratio in [0, 0.1, 0.5, 0.8, 0.9]:
        block_wise_weight = torch.rand(N//block_h, K//block_w, dtype=torch.float32).cuda()
        block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
        print("Sparsity ratio:", torch.sum(block_mask)/block_mask.numel())
        data =  torch.rand(B, S, K).cuda()
        # data =  torch.ones(B, S, K).cuda()
        ori_linear = torch.nn.Linear(K, N).cuda()
        # ori_linear.weight.data[:] = 1
        # ori_linear.bias.data[:] = 0
        b_linear = BlockwiseSparseLinear(ori_linear)
        test_corressness(data, block_mask, ori_linear, b_linear)
        