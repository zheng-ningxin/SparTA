import torch
import cusparse_linear
import cusparse_csr_cpp

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
        # import ipdb; ipdb.set_trace()
        # out = cusparse_linear.forward(data, self.csr_row, self.csr_col, self.csr_val, self.weight.t().size(), self.csr_row[-1]) 
        out = cusparse_linear.forward(data, self.csr_row, self.csr_col, self.csr_val, self.weight.size(), self.csr_row[-1]) 
        if self.bias is not None:
            out += self.bias
        return out
    
    def ref_forward(self, data):
        return torch.nn.functional.linear(data, self.mask*self.weight, self.bias)