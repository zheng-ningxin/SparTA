import torch
import torch.nn.functional as F

class DimDynamicLinear(torch.nn.Module):
    def __init__(self, ori_linear, in_or_out):
        super(DimDynamicLinear, self).__init__()
        assert isinstance(ori_linear, torch.nn.Linear)
        self.weight = ori_linear.weight
        self.bias = ori_linear.bias
        self.in_or_out = in_or_out

    def forward(self, data, mask):
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
    # def forward(self, data, mask):
    #     assert mask.is_cuda
    #     b_mask = mask.to(torch.bool)
    #     if self.in_or_out == 0:
    #         # out
    #         re = F.linear(data,
    #                  self.weight[b_mask],
    #                  None if self.bias is None else self.bias[b_mask])
    #     else:
    #         re = F.linear(data,
    #                  self.weight[:, b_mask],
    #                  self.bias)
    #     return re