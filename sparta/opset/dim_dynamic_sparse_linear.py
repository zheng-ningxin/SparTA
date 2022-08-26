import torch
import torch.nn.functional as F

class DimDynamicLinear(torch.nn.Module):
    def __init__(self, ori_linear, in_or_out):
        assert isinstance(ori_linear, torch.nn.Linear)
        self.weight = ori_linear.weight
        self.bias = ori_linear.bias
        self.in_or_out = in_or_out

    def forward(self, data, mask):
        b_mask = mask.to(torch.bool)
        if self.in_or_out == 0:
            re = F.linear(data,
                     self.weight[b_mask],
                     None if self.bias is None else self.bias[b_mask])
        else:
            re = F.linear(data,
                     self.weight[:, b_mask],
                     self.bias)
        return re