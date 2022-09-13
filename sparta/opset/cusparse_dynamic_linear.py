import torch
import cusparse_linear
import cusparse_csr_cpp


class CusparseDynamicLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                activation,
                weight,
                mask,
                bias):
        csr_row, csr_col, csr_val = cusparse_csr_cpp.forward(weight*mask)
        out_features = torch.tensor(weight.size(0))
        in_features = torch.tensor(weight.size(1))
        nnz = csr_row[-1]
        ctx.save_for_backward(activation,
                              csr_row,
                              csr_col,
                              csr_val,
                              bias,
                              nnz,
                              out_features,
                              in_features)
        out = cusparse_linear.forward(activation, csr_row, csr_col, csr_val, (out_features, in_features), nnz) 
        if bias is not None:
            out += bias
        return out
    @staticmethod
    def backward(ctx, *grads_out):
        activation, csr_row, csr_col, csr_val, bias, nnz, out_features, in_features = ctx.saved_tensors
        N, K = out_features, in_features
        M = activation.numel() / K
        a_grad, w_grad = cusparse_linear.backward(activation, csr_row, csr_col, csr_val, grads_out[0], M, K, N, nnz)
        full_w_grad = cusparse_csr_cpp.backward(csr_row, csr_col, w_grad, out_features, in_features, nnz)
        return a_grad, full_w_grad, None, None

class CusparseDynamicLinear(torch.nn.Module):
    def __init__(self, ori_linear):
        super(CusparseDynamicLinear, self).__init__()
        # self.mask = torch.ones_like(ori_linear.weight)
        self.weight = ori_linear.weight.clone().detach()
        self.bias = ori_linear.bias
        # self.update_mask(self.mask)
        self.weight.requires_grad_()

    # def update_mask(self, mask):
    #     self.mask = mask
        # self.csr_row, self.csr_col, self.csr_val = cusparse_csr_cpp.forward((mask*self.weight).t().contiguous())
        # self.csr_row, self.csr_col, self.csr_val = cusparse_csr_cpp.forward((mask*self.weight).contiguous())
    
    def forward(self, data, w_mask):
        # out = cusparse_linear.forward(data, self.csr_row, self.csr_col, self.csr_val, self.weight.size(), self.csr_row[-1]) 
        # if self.bias is not None:
        #     out += self.bias
        # return out
        # nnz = self.csr_row[-1].clone().detach()
        # out_features = torch.tensor(self.weight.size(0), dtype=torch.int32)
        # in_features = torch.tensor(self.weight.size(1), dtype=torch.int32)
        return CusparseDynamicLinearFunction.apply(data, self.weight, w_mask, self.bias)
    
    def ref_forward(self, data):
        return torch.nn.functional.linear(data, self.mask*self.weight, self.bias)