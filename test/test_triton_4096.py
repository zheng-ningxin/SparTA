import triton
import torch

def triton_attention(
    layout,
    block: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
):
    # global global_sparse_dot_sdd_nt, global_sparse_dot_dsd_nn, global_sparse_softmax
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True, device=value.device)
    sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False, device=value.device)
    sparse_softmax = triton.ops.blocksparse.softmax(layout, block, device=value.device)

    w = sparse_dot_sdd_nt(query, key)
    w = sparse_softmax(w, scale=scale, is_causal=True)
    a = sparse_dot_dsd_nn(w, value)
    return a
if __name__ == '__main__':
    B = 12
    H = 12
    S = 4096
    hidden = 64
    block = 32
    mask = torch.zeros(S//block, S//block, dtype=torch.int32).cuda()
    mask[10:16] = 1
    query = torch.rand(B, H, S, hidden)
    key = torch.rand(B, H, S, hidden)
    value = torch.rand(B, H, S, hidden)
    out = triton_attention(mask, block, query, key, value, 1.0)
    