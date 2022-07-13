# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import torch
import random
from sparta.opset.dynamic_sparse_linear import *
from sparta.common.utils import verify_bcsr
from sparta.opset.bcsr_converter import BcsrConverter



if __name__ == '__main__':

    linear = torch.nn.Linear(2048, 1024, bias=True).cuda()
    d_linear = DynamicSparseLinear(linear)
    ori_linear = copy.deepcopy(linear)

    data_1 = torch.rand(1024, 2048).cuda()
    data_1.requires_grad_()
    data_2 = copy.deepcopy(data_1)
    data_2.requires_grad_()

    out1 = d_linear(data_1)
    grad = torch.ones_like(out1)
    print(grad.size())
    out1.backward(grad)
    out2 = ori_linear(data_2)
    out2.backward(grad)
    assert torch.allclose(out1, out2, rtol=1e-08, atol=1e-04)
    # import ipdb; ipdb.set_trace()
    verify_bcsr(d_linear.mask, d_linear.ori_linear.weight, d_linear.csr_row, d_linear.csr_col, d_linear.csr_val, d_linear.block_h, d_linear.block_w)
    verify_bcsr(d_linear.mask.t(), d_linear.ori_linear.weight.t(), d_linear.grad_csr_row, d_linear.grad_csr_col, d_linear.grad_csr_val, d_linear.block_h, d_linear.block_w)
    
    assert torch.allclose(d_linear.ori_linear.weight.grad, ori_linear.weight.grad, rtol=1e-08, atol=1e-04)
    assert torch.allclose(data_1.grad, data_2.grad,  rtol=1e-08, atol=1e-04)