# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import torch
import random
from SparTA.OPs import *

if __name__ == '__main__':
    ori_linear = torch.nn.Linear(1024, 1024, bias=True).cuda()
    d_linear = DynamicSparseLinear(ori_linear)
    dummy_input =  torch.rand(1024, 1024).cuda()
    out1 = d_linear(dummy_input)
    out2 = ori_linear(dummy_input)
    assert torch.allclose(out1, out2, rtol=1e-08, atol=1e-04)