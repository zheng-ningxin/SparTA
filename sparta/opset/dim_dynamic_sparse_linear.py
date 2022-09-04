import torch
import torch.nn.functional as F
import dim_dynamic_sparse_linear_cpp

class OutDimDynamicLinearSparseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                data,
                weight,
                bias,
                index):
        ctx.save_for_backward(
            data,
            weight,
            index
        )
        return dim_dynamic_sparse_linear_cpp.outdim_forward(data, weight, index, bias)
    @staticmethod
    def backward(ctx, *grad_out):
        data, weight, index = ctx.saved_tensors
        a_grad, w_grad = dim_dynamic_sparse_linear_cpp.outdim_backward(data, weight, grad_out[0], index)
        return a_grad, w_grad, None, None

class DimDynamicLinear(torch.nn.Module):
    def __init__(self, ori_linear, in_or_out):
        super(DimDynamicLinear, self).__init__()
        assert isinstance(ori_linear, torch.nn.Linear)
        self.weight = ori_linear.weight.clone().detach()
        self.weight.requires_grad_()
        self.bias = ori_linear.bias.clone().detach()
        self.in_or_out = in_or_out
        

    def forward(self, data, mask):
        _pos = (mask.nonzero(as_tuple=True)[0]).to(torch.int32)
        if self.in_or_out == 0:
            # out dimension sparse
            return OutDimDynamicLinearSparseFunction.apply(data, self.weight, self.bias, _pos)
        else:
            # input dimensions sparse
            pass
        
    def ref_forward(self, data, mask):
        assert mask.is_cuda
        # import ipdb; ipdb.set_trace()
        _pos = mask.nonzero(as_tuple=True)[0]
        
        if self.in_or_out == 0:
            # out
            re = F.linear(data,
                     torch.index_select(self.weight, 0, _pos),
                     None if self.bias is None else torch.index_select(self.bias, 0, _pos))
        else:
            re = F.linear(data, torch.index_select(self.weight, 1, _pos),
                     self.bias)
        return re
