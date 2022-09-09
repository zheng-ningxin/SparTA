import torch
import triton

class TritonDynamicLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        activation,
        weight,
        block_mask,
        block_h,
        block_w,
        bias
    ):
        ctx.save_for_backward(
            activation,
            weight,
            block_mask,
            block_h,
            block_w
        )
        assert(block_h == block_w)
        # import ipdb; ipdb.set_trace()
        dot_dds_nt = triton.ops.blocksparse.matmul(block_mask, block_h, "dds", trans_a=False, trans_b=True, device=activation.device)
        import ipdb; ipdb.set_trace()
        out = dot_dds_nt(activation, weight)
        if bias is not None:
            out += bias
        return out
    
    @staticmethod
    def backward(
        ctx, *grad_out
    ):
        activation, weight, block_mask, block_h, block_w = ctx.saved_tensors
        # a_grad(MxK) = grad_c(MxN) * weight(NxK)
        dot_dds_nn = triton.ops.blocksparse.matmul(block_mask, block_h, "dds", trans_a=False, trans_b=False, device=activation.device)
        a_grad = dot_dds_nn(grad_out[0], weight)
        # w_grad(N*K) = grad_c^T(NxM) * activation(MxK)
        dot_sdd_tn = triton.ops.blocksparse.matmul(block_mask, block_h, "sdd", trans_a=True, trans_b=False, device=activation.device)
        w_grad = dot_sdd_tn(grad_out[0], activation)
        import ipdb; ipdb.set_trace()
        return a_grad, w_grad, None, None, None, None

class TritonDynamicLinear(torch.nn.Module):
    def __init__(self, ori_linear, block_h, block_w, full_mask=True):
        super(TritonDynamicLinear, self).__init__()
        assert isinstance(ori_linear, torch.nn.Linear)
        self.weight = ori_linear.weight.clone().detach()
        self.bias = None
        if ori_linear.bias is not None:
            self.bias = ori_linear.bias.clone().detach()
        self.weight.requires_grad_()
        self.block_h, self.block_w = block_h, block_w
        self.conv = None
        self.full_mask = full_mask
        if self.full_mask:
            self.conv=torch.nn.Conv2d(1,1,(self.block_h, self.block_w), (self.block_h, self.block_w)).cuda()
            self.conv.eval()
            self.conv.weight.data[:] = 1

    def forward(self, data, mask):
        block_mask = mask
        if self.full_mask:
            ori_mask_size = mask.size()
            mask = mask.view(1, 1, mask.size(0), mask.size(1)).to(torch.float32)
            block_mask = self.conv(mask)
            # import ipdb; ipdb.set_trace()
            block_mask = (block_mask.view(1, ori_mask_size[0]//self.block_h, ori_mask_size[1]//self.block_w)>0).to(torch.int32)

        return TritonDynamicLinearFunction.apply(data, self.weight, block_mask, self.block_h, self.block_w, self.bias)
    
        
            