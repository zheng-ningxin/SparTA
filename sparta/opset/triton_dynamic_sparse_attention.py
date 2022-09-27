import torch
import triton
import time
class TritonDynamicAttention(torch.nn.Module):
    global_sparse_dot_sdd_nt = None
    global_sparse_dot_dsd_nn = None
    global_sparse_softmax = None
    def __init__(self, block_h, block_w, head_num, full_mask=True, profile=False, global_model=False):
        super(TritonDynamicAttention, self).__init__()

        self.block_h, self.block_w = block_h, block_w
        self.conv = None
        self.full_mask = full_mask
        self.head_num = head_num
        if self.full_mask:
            self.conv=torch.nn.Conv2d(self.head_num, self.head_num,(self.block_h, self.block_w), (self.block_h, self.block_w), groups=self.head_num, bias=False).cuda()
            self.conv.eval()
            self.conv.weight.data[:] = 1
        self.profile = profile
        self.convert_overhead = []
        self.global_model = global_model

    def set_global_mask(self, mask, is_full_mask, block_h, block_w, head_num):
        if not is_full_mask:
            block_mask = mask
        else:
            conv = torch.nn.Conv2d(head_num, head_num, (block_h, block_w), (block_h, block_w), groups=head_num, bias=False).cuda()
            ori_mask_size = mask.size()
            mask = mask.view(1, head_num, ori_mask_size[-2], ori_mask_size[-1]).to(torch.float32)
            block_mask = conv(mask)
            block_mask = (block_mask.view(head_num, ori_mask_size[-2]//block_h, ori_mask_size[-1]//block_w)>0).to(torch.int32)
        TritonDynamicAttention.global_sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(block_mask, block_h, "sdd", trans_a=False, trans_b=True, device=mask.device)
        TritonDynamicAttention.global_sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(block_mask, block_h, "dsd", trans_a=False, trans_b=False, device=mask.device)
        TritonDynamicAttention.global_sparse_softmax = triton.ops.blocksparse.softmax(block_mask, block_h, device=mask.device)

    def forward(self, query, key, value, mask=None):
        scale = 1.0
        if not self.global_model:
            assert mask is not None
            if self.profile:
                torch.cuda.synchronize()
                t_start = time.time()
            block_mask = mask
            if self.full_mask:
                ori_mask_size = mask.size()
                mask = mask.view(1, self.head_num, ori_mask_size[-2], ori_mask_size[-1]).to(torch.float32)
                block_mask = self.conv(mask)
                block_mask = (block_mask.view(self.head_num, ori_mask_size[-2]//self.block_h, ori_mask_size[-1]//self.block_w)>0).to(torch.int32)
            # import ipdb; ipdb.set_trace()
            sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(block_mask, self.block_h, "sdd", trans_a=False, trans_b=True, device=value.device)
            sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(block_mask, self.block_h, "dsd", trans_a=False, trans_b=False, device=value.device)
            sparse_softmax = triton.ops.blocksparse.softmax(block_mask, self.block_h, device=value.device)
            if self.profile:
                torch.cuda.synchronize()
                t_end = time.time()
                self.convert_overhead.append((t_end-t_start)*1000)
        else:
            sparse_dot_sdd_nt = TritonDynamicAttention.global_sparse_dot_sdd_nt
            sparse_dot_dsd_nn = TritonDynamicAttention.global_sparse_dot_dsd_nn
            sparse_softmax = TritonDynamicAttention.global_sparse_softmax
        w = sparse_dot_sdd_nt(query, key)
        w = sparse_softmax(w, scale=scale, is_causal=True)
        a = sparse_dot_dsd_nn(w, value)
        return a