import torch
import cusparse_linear
import cusparse_csr_cpp


class CusparseDynamicLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                activation,
                csr_row,
                csr_col,
                csr_val,
                bias,
                nnz,
                out_features,
                in_features):
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
        return a_grad, None, None, None, None, None, None, None

class CusparseDynamicLinear(torch.nn.Module):
    def __init__(self, ori_linear):
        super(CusparseDynamicLinear, self).__init__()
        self.mask = torch.ones_like(ori_linear.weight)
        self.weight = ori_linear.weight.clone().detach()
        self.bias = ori_linear.bias
        self.update_mask(self.mask)

    def update_mask(self, mask):
        self.mask = mask
        # self.csr_row, self.csr_col, self.csr_val = cusparse_csr_cpp.forward((mask*self.weight).t().contiguous())
        self.csr_row, self.csr_col, self.csr_val = cusparse_csr_cpp.forward((mask*self.weight).contiguous())
    
    def forward(self, data):
        # out = cusparse_linear.forward(data, self.csr_row, self.csr_col, self.csr_val, self.weight.size(), self.csr_row[-1]) 
        # if self.bias is not None:
        #     out += self.bias
        # return out
        nnz = self.csr_row[-1].clone().detach()
        out_features = torch.tensor(self.weight.size(0), dtype=torch.int32)
        in_features = torch.tensor(self.weight.size(1), dtype=torch.int32)
        return CusparseDynamicLinearFunction.apply(data, self.csr_row, self.csr_col, self.csr_val, self.bias, nnz, out_features, in_features)
    
    def ref_forward(self, data):
        return torch.nn.functional.linear(data, self.mask*self.weight, self.bias)