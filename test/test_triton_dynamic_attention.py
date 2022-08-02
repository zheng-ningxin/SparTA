
import torch
import time
import triton
from sparta.opset.dynamic_sparse_attention import DynamicSparseAttention

# @pytest.mark.parametrize("MODE", ["sdd", "dds", "dsd"])
# @pytest.mark.parametrize("TRANS_A", [False, True])
# @pytest.mark.parametrize("TRANS_B", [False, True])
# @pytest.mark.parametrize("BLOCK", [16, 32, 64])
# @pytest.mark.parametrize("DTYPE", [torch.float16])
global_sparse_dot_sdd_nt=None
global_sparse_dot_dsd_nn=None
global_sparse_softmax=None

# @pytest.mark.parametrize("block", [16, 32, 64])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_attention_fwd_bwd(
    block,
    dtype,
    input_scale=1.0,
    scale=1 / 8.0,
    n_ctx=256,
    batch_size=2,
    n_heads=2,
):
    # inputs
    print(f'batchsize: {batch_size} heads:{n_heads}\n')
    qkv_shape = (batch_size, n_heads, n_ctx, 64)
    qkvs = [
        torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True).to(dtype).cuda() for _ in range(3)
    ]

    # Triton:
    n_blocks = n_ctx // block
    layout = torch.tril(torch.ones([n_heads, n_blocks, n_blocks], dtype=torch.long))
    query, key, value = [x.clone() for x in qkvs]
    query.retain_grad()
    key.retain_grad()
    value.retain_grad()

    torch.cuda.synchronize()
    st = time.time()
    for i in range(50):
        attn_out = triton_attention(layout, block, query=query, key=key, value=value, scale=scale)
    torch.cuda.synchronize()
    end = time.time()
    print('Sparse Forward Implementation', (end-st)/50*1000)



    torch.cuda.synchronize()
    st = time.time()
    for i in range(50):
        attn_out = triton_attention(layout, block, query=query, key=key, value=value, scale=scale)
        # ad hoc loss
        loss = (attn_out ** 2).mean()
        loss.backward()
        grads = [query.grad, key.grad, value.grad]
    torch.cuda.synchronize()
    end = time.time()
    print('Sparse Backward Implementation', (end-st)/50*1000)


# @pytest.mark.parametrize("block", [16, 32, 64])
def triton_attention_init(
    layout,
    block: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
):
    global global_sparse_dot_sdd_nt, global_sparse_dot_dsd_nn, global_sparse_softmax
    global_sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True, device=value.device)
    global_sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False, device=value.device)
    global_sparse_softmax = triton.ops.blocksparse.softmax(layout, block, device=value.device)

def triton_attention(
    layout,
    block: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
):
    
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True, device=value.device)
    sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False, device=value.device)
    sparse_softmax = triton.ops.blocksparse.softmax(layout, block, device=value.device)

    w = sparse_dot_sdd_nt(query, key)
    w = sparse_softmax(w, scale=scale, is_causal=True)
    a = sparse_dot_dsd_nn(w, value)
    return a

def measure_triton_dynamic(q, k, v, layout, block):
    torch.cuda.synchronize()
    st_time = time.time()
    triton_attention(layout, block, q, k, v, 1.0)
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - st_time


def measure_dynamic_attention(sparse_attention, q, k, v, full_mask):
    torch.cuda.synchronize()
    st_time = time.time()
    sparse_attention.set_global_sparse_pattern(full_mask)
    sparse_attention(q, k, v)    
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - st_time
    
def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size, dim=0)
    full_mask = full_mask.repeat_interleave(block_size, dim=1)
    return full_mask

if __name__ == "__main__":
    device = torch.device('cuda:0')
    HEAD_NUM = 20
    seqlen = 1024
    hidden_dim = 64
    bs = 16
    block_size = 32
    block_wise_weight = torch.zeros(seqlen//block_size, seqlen//block_size, dtype=torch.float32, device=device)
    
    sparsity_ratio = 0.1
    run_times = 100
    triton_time = []
    dynamic_time = []
    spa = DynamicSparseAttention(True)
    q = torch.rand((bs, HEAD_NUM, seqlen, hidden_dim), dtype = torch.float32, device = device)
    k = torch.rand((bs, HEAD_NUM, seqlen, hidden_dim), dtype = torch.float32, device = device)
    v = torch.rand((bs, HEAD_NUM, seqlen, hidden_dim), dtype = torch.float32, device = device)
    for rid in range(run_times):
        block_wise_weight = torch.rand(seqlen//block_size, seqlen//block_size, dtype=torch.float32, device=device)
        block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
        full_mask = convert_to_full_mask(block_mask, block_size)
        # import ipdb; ipdb.set_trace()
        t_time = measure_triton_dynamic(q, k, v, block_mask.reshape(1, seqlen//block_size, seqlen//block_size).repeat(HEAD_NUM, 1, 1), block_size)
        d_time = measure_dynamic_attention(spa, q, k, v, full_mask)
        triton_time.append(t_time)
        dynamic_time.append(d_time)
        
    print('Triton Speed: ', sum(triton_time)/len(triton_time)*1000)
    print('Dynamic Speed: ', sum(dynamic_time)/len(dynamic_time)*1000)