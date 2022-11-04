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
from sparta.opset.bcsr_converter_blockwise import BcsrConverterBlockwise
import condense_sparse_linear_cpp
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class BlockwiseSparseLinearCondenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                activation,
                weight,
                row_ptr,
                col_idx,
                bias,
                M, K, N, block_h, block_w):
        ctx.save_for_backward(
            activation,
            weight,
            row_ptr,
            col_idx,
            M, K, N,
            block_h,
            block_w
        )
        return condense_sparse_linear_cpp.forward(activation, weight, row_ptr, col_idx, M, K, N, block_h, block_w)

    @staticmethod
    def backward(ctx, *grad_out):
        # activation, weight, blockwise_mask = ctx.saved_tensors
        # a_grad, w_grad = blockwise_sparse_linear_cpp.backward(activation, weight, grad_out[0], blockwise_mask)
        # return a_grad, w_grad, None, None
        return None, None, None, None

class BlockwiseSparseLinearCondense(SparseOPBase):
    def __init__(self, ori_linear, block_h, block_w):
        super(BlockwiseSparseLinearCondense, self).__init__()
        assert isinstance(ori_linear, torch.nn.Linear)
        self.weight = copy.deepcopy(ori_linear.weight).t()
        self.bias = copy.deepcopy(ori_linear.bias)
        # for memory usage saving
        del ori_linear
        self.convert = BcsrConverterBlockwise(True)
        self.block_h = block_h
        self.block_w = block_w
        self.K = self.weight.size(0)
        self.N = self.weight.size(1)
    def forward(self, activation, block_mask):  
        csr_row, csr_col = self.convert(block_mask)
        M = activation.numel() / self.K
        K = self.K
        N = self.N        
        return BlockwiseSparseLinearCondenseFunction.apply(activation, self.weight, csr_row, csr_col, self.bias, M, K, N, self.block_h, self.block_w)
