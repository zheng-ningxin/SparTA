# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
from cmath import inf
import torch
import types
import logging

from .sparse_opbase import SparseOPBase
from sparta.codegen.template.sparse_attention import *
from sparta.common.utils import *

import blockwise_sparse_linear_cpp
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class BlockwiseSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                activation,
                weight,
                bias,
                blockwise_mask):
        ctx.save_for_backward(
            activation,
            weight,
            blockwise_mask
        )
        return blockwise_sparse_linear_cpp.forward(activation, weight, bias, blockwise_mask)

    @staticmethod
    def backward(ctx, *grad_out):
        activation, weight, blockwise_mask = ctx.saved_tensors
        a_grad, w_grad = blockwise_sparse_linear_cpp.backward(activation, weight, grad_out[0], blockwise_mask)
        return a_grad, w_grad, None, None
class BlockwiseSparseLinear(SparseOPBase):
    def __init__(self, ori_linear):
        super(BlockwiseSparseLinear, self).__init__()
        assert isinstance(ori_linear, torch.nn.Linear)
        self.weight = copy.deepcopy(ori_linear.weight)
        self.bias = copy.deepcopy(ori_linear.bias)

    def forward(self, activation, block_mask):
        return BlockwiseSparseLinearFunction.apply(activation, self.weight, self.bias, block_mask)
