import torch
import triton
import functools
import time
from functools import reduce
# class TritonDynamicLinearFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         activation,
#         weight,
#         block_mask,
#         block_h,
#         block_w,
#         bias
#     ):
#         ctx.save_for_backward(
#             activation,
#             weight,
#             block_mask,
#             block_h,
#             block_w
#         )
#         assert(block_h == block_w)
#         # import ipdb; ipdb.set_trace()
#         dot_dds_nt = triton.ops.blocksparse.matmul(block_mask, block_h, "dds", trans_a=False, trans_b=True, device=activation.device)
#         import ipdb; ipdb.set_trace()
#         out = dot_dds_nt(activation, weight)
#         if bias is not None:
#             out += bias
#         return out
    
#     @staticmethod
#     def backward(
#         ctx, *grad_out
#     ):
#         activation, weight, block_mask, block_h, block_w = ctx.saved_tensors
#         # a_grad(MxK) = grad_c(MxN) * weight(NxK)
#         dot_dds_nn = triton.ops.blocksparse.matmul(block_mask, block_h, "dds", trans_a=False, trans_b=False, device=activation.device)
#         a_grad = dot_dds_nn(grad_out[0], weight)
#         # w_grad(N*K) = grad_c^T(NxM) * activation(MxK)
#         dot_sdd_tn = triton.ops.blocksparse.matmul(block_mask, block_h, "sdd", trans_a=True, trans_b=False, device=activation.device)
#         w_grad = dot_sdd_tn(grad_out[0], activation)
#         import ipdb; ipdb.set_trace()
#         return a_grad, w_grad, None, None, None, None

class TritonDynamicLinear(torch.nn.Module):
    def __init__(self, ori_linear, block_h, block_w, full_mask=True, profile=False):
        super(TritonDynamicLinear, self).__init__()
        assert isinstance(ori_linear, torch.nn.Linear)
        self.ori_linear = ori_linear
        self.weight = ori_linear.weight.clone().detach()
        self.bias = None
        if ori_linear.bias is not None:
            self.bias = ori_linear.bias.clone().detach()
        self.block_h, self.block_w = block_h, block_w
        self.weight = torch.zeros(self.weight.size(0)//self.block_h, self.weight.size(1)//self.block_w, self.block_h, self.block_w, device=self.weight.device)
        for i in range(self.weight.size(0)):
            for j in range(self.weight.size(1)):
                self.weight.data[i,j] = ori_linear.weight.data[i*self.block_h:(i+1)*self.block_h, j*self.block_w:(j+1)*self.block_w]
        self.conv = None
        self.full_mask = full_mask
        if self.full_mask:
            self.conv=torch.nn.Conv2d(1,1,(self.block_h, self.block_w), (self.block_h, self.block_w)).cuda()
            self.conv.eval()
            self.conv.weight.data[:] = 1
        self.weight.requires_grad_()
        self.profile = profile
        self.convert_overhead = []

    def forward(self, data, mask):
        ori_data_size = list(data.size())
        K = ori_data_size[-1]
        M = reduce(lambda x, y : x*y, ori_data_size) // K
        data = data.view(1, 1, M, K)
        if self.profile:
            torch.cuda.synchronize()
            t_start = time.time()    
        block_mask = mask
        if self.full_mask:
            ori_mask_size = mask.size()
            mask = mask.view(1, 1, mask.size(0), mask.size(1)).to(torch.float32)
            block_mask = self.conv(mask)
            block_mask = (block_mask.view(1, ori_mask_size[0]//self.block_h, ori_mask_size[1]//self.block_w)>0).to(torch.int32)
        dot_dds_nt = triton.ops.blocksparse.matmul(block_mask, self.block_h, "dds", trans_a=False, trans_b=True, device=data.device)
        # convert the weight online
        block_nnz = torch.sum(block_mask)
        # import ipdb; ipdb.set_trace()
        # print('triton sparse linear: block nnz', block_nnz)
        
        sparse_weight = self.weight[block_mask[0]>0].view(1, block_nnz, self.block_h, self.block_w)
        if self.profile:
            torch.cuda.synchronize()
            t_end = time.time()
            self.convert_overhead.append((t_end-t_start)*1000)
        # return TritonDynamicLinearFunction.apply(data, self.weight, block_mask, self.block_h, self.block_w, self.bias)
        out = dot_dds_nt(data, sparse_weight)
        if len(ori_data_size)==2:
            return out.view(M,K) if self.bias is None else out.view(M, K) + self.bias
        else:
            # import ipdb; ipdb.set_trace()
            return out.view(ori_data_size[:-1] + [out.size(-1)]) if self.bias is None else out.view(ori_data_size[:-1] + [out.size(-1)]) + self.bias
            