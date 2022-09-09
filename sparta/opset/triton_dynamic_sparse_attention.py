import torch
import triton

class TritonDynamicAttention(torch.nn.Module):
    def __init__(self, block_h, block_w, head_num, full_mask=True):
        super(TritonDynamicAttention, self).__init__()

        self.block_h, self.block_w = block_h, block_w
        self.conv = None
        self.full_mask = full_mask
        self.head_num = head_num
        if self.full_mask:
            self.conv=torch.nn.Conv2d(self.head_num, self.head_num,(self.block_h, self.block_w), (self.block_h, self.block_w), groups=self.head_num).cuda()
            self.conv.eval()
            self.conv.weight.data[:] = 1

    def forward(self, query, key, value, mask):
        block_mask = mask
        scale = 1.0
        if self.full_mask:
            ori_mask_size = mask.size()
            mask = mask.view(1, self.head_num, ori_mask_size[-2], ori_mask_size[-1]).to(torch.float32)
            block_mask = self.conv(mask)
            block_mask = (block_mask.view(self.head_num, ori_mask_size[-2]//self.block_h, ori_mask_size[-1]//self.block_w)>0).to(torch.int32)
    
        sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(block_mask, self.block_h, "sdd", trans_a=False, trans_b=True, device=value.device)
        sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(block_mask, self.block_h, "dsd", trans_a=False, trans_b=False, device=value.device)
        sparse_softmax = triton.ops.blocksparse.softmax(block_mask, self.block_h, device=value.device)

        w = sparse_dot_sdd_nt(query, key)
        w = sparse_softmax(w, scale=scale, is_causal=True)
        a = sparse_dot_dsd_nn(w, value)
        return a