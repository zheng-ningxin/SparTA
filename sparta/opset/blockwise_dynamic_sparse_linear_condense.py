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
                M, K, N, block_h, block_w,
                batch_size, seq_len):
        ctx.save_for_backward(
            activation,
            weight,
            row_ptr,
            col_idx
        )
        ctx.M, ctx.K, ctx.N, ctx.block_h, ctx.block_w = M, K, N, block_h, block_w
        return condense_sparse_linear_cpp.forward(activation, weight, row_ptr, col_idx, bias, M, K, N, block_h, block_w, batch_size, seq_len)

    @staticmethod
    def backward(ctx, *grad_out):
        activation, weight, row_ptr, col_idx= ctx.saved_tensors
        M, K, N, block_h, block_w = ctx.M, ctx.K, ctx.N, ctx.block_h, ctx.block_w
        a_grad, w_grad = condense_sparse_linear_cpp.backward(activation, weight, row_ptr, col_idx, grad_out[0], M, K, N, block_h, block_w)
        # import ipdb; ipdb.set_trace()
        return a_grad, w_grad, None, None, None, None, None, None, None, None, None, None

class BlockwiseSparseLinearCondense(SparseOPBase):
    def __init__(self, ori_linear, block_h, block_w):
        super(BlockwiseSparseLinearCondense, self).__init__()
        assert isinstance(ori_linear, torch.nn.Linear)
        tmp_t = ori_linear.weight.clone().detach().t().contiguous()
        # self.register_parameter('weight', tmp_t)
        self.weight = torch.nn.Parameter(tmp_t)
        # self.weight.require_grad_(ori_linear.weight.require_grad)
        # self.weight = copy.deepcopy(ori_linear.weight)
        # with torch.no_grad():
        #     self.weight = self.weight.t().contiguous()
        self.bias = copy.deepcopy(ori_linear.bias)
        # for memory usage saving
        del ori_linear
        self.convert = BcsrConverterBlockwise(True)
        self.block_h = block_h
        self.block_w = block_w
        self.K = self.weight.size(0)
        self.N = self.weight.size(1)

    def forward(self, activation, block_mask):
        self.csr_row, self.csr_col = self.convert(block_mask)
        batch_size, seq_len, in_hidden = activation.size()
        M = batch_size * seq_len
        K = self.K
        N = self.N
        activation = activation.view(M, K).t().contiguous()
        # activation = activation.view(M, K).contiguous()
        return BlockwiseSparseLinearCondenseFunction.apply(activation, self.weight, self.csr_row, self.csr_col, self.bias, M, K, N, self.block_h, self.block_w, batch_size, seq_len)
