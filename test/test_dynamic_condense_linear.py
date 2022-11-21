import sparta
import torch
import time
import random
from sparta.opset.blockwise_dynamic_sparse_linear_condense import BlockwiseSparseLinearCondense

    
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

# def export_tensort(t, fname):
    

def test_corressness(data, block_mask, ori_linear, b_linear, block_h=1, block_w=32):
    data_1 = data.clone().detach()
    data_1.requires_grad_()
    data_2 = data.clone().detach()
    data_2.requires_grad_()
    
    full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
    # print("Block mask:", block_mask.size())
    # print("Full mask:", full_mask.size())
    ori_linear.weight.data *= full_mask.data.t()
    b_linear.weight.data *= full_mask.data
    ref_out = ori_linear(data_1)
    out = b_linear(data_2, block_mask)
    # assert torch.allclose(out, ref_out, rtol=1e-08, atol=1e-03)
    # import ipdb; ipdb.set_trace()
    tmp_grad = torch.rand_like(ref_out)
    tmp_grad.data[:] = 1
    ref_out.backward(tmp_grad)
    out.backward(tmp_grad)
    w_grad_1 = b_linear.weight.grad
    w_grad_2 = ori_linear.weight.grad.t() * full_mask
    # import ipdb; ipdb.set_trace()
    # # import ipdb; ipdb.set_trace()
    # # import ipdb; ipdb.set_trace()
    flag = True
    flag = flag and torch.allclose(out, ref_out, rtol=1e-04, atol=1e-03)
    flag = flag and torch.allclose(data_1.grad, data_2.grad, rtol=1e-04, atol=1e-03)
    flag = flag and torch.allclose(w_grad_2, w_grad_1, rtol=1e-04, atol=1e-03)
    if not flag:
        import ipdb;
        ipdb.set_trace()

def dense_speed(linear, data):
    run_time = 2000
    data.requires_grad_()
    _tmp = linear(data)
    grad = torch.rand_like(_tmp)

    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(run_time):
        re = linear(data)
        re.backward(grad)
    torch.cuda.synchronize()
    t_end = time.time()
    assert data.grad is not None
    assert linear.weight.grad is not None
    print('Dense per batch(ms):' , (t_end-t_start)*1000/run_time)

def test_speed(b_linear, block_mask, data):
    run_time = 2000
    data.requires_grad_()
    _tmp = b_linear(data, block_mask)
    grad = torch.rand_like(_tmp)
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(run_time):
        re = b_linear(data, block_mask)
        re.backward(grad)
    torch.cuda.synchronize()
    t_end = time.time()
    assert data.grad is not None
    assert b_linear.weight.grad is not None
    print('Sparse per batch(ms):' , (t_end-t_start)*1000/run_time)
    

if __name__ == '__main__':
    B = 32
    S = 128
    K = 768
    N = 768
    block_h = 1
    block_w = 32
    
    for sparsity_ratio in [0.6]:
    # for sparsity_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        block_wise_weight = torch.rand(K//block_h, N//block_w, dtype=torch.float32).cuda()
        block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
        print("Sparsity ratio:", torch.sum(block_mask)/block_mask.numel())
        data =  torch.rand(B, S, K).cuda()
        ori_linear = torch.nn.Linear(K, N).cuda()
        ori_linear.bias.data[:] = 0
        ori_linear.weight.data[:] = 1
        # data.data[:] = 1
        b_linear = BlockwiseSparseLinearCondense(ori_linear, block_h, block_w)
        # test_corressness(data, block_mask, ori_linear, b_linear)
        # dense_speed(ori_linear, data)
        test_speed(b_linear, block_mask, data)