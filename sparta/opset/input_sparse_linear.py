# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging
import convert_bcsr_cpp
import openai_bmm_cpp
from .sparse_opbase import SparseOPBase



class InSparseLinear(SparseOPBase):
    global_seqlen = None

    @staticmethod
    def set_global_seqlens(seqlens):
        # seqlens is an one-dimension tensor with size of [Batchsize]
        # each element in the tensor represents the effective sequence
        # length of current instance
        assert isinstance(seqlens, torch.Tensor)
        assert seqlens.is_cuda
        assert seqlens.dtype == torch.int32, "only support int32 type"
        InSparseLinear.global_seqlen = seqlens


    def __init__(self, ori_linear):
        super(InSparseLinear, self).__init__()
        self.weight = ori_linear.weight
        self.bias = ori_linear.bias
        self.ext_buffer = None
        self.row = None
        self.col = None
        self.block_w = 32
        self.out_shape = None

    # def forward(self, activation):
    #     if self.ext_buffer is None:
    #         act_size = activation.size()
    #         if(len(act_size)==3):
    #             self.h, self.w = act_size[0] * act_size[1], act_size[2]
    #             self.out_shape = (act_size[0], act_size[1], self.weight.size(0))
    #         else:
    #             self.h, self.w = act_size
    #             self.out_shape = (act_size[0], self.weight.size(0))
    #         row_size = self.w//self.block_w
    #         self.ext_buffer = torch.empty(2 * self.w + row_size * self.h, dtype=torch.int32, device=activation.device)
    #         self.row = torch.empty(row_size+1, dtype=torch.int32, device=activation.device)
    #         self.col = torch.empty(row_size*self.h, dtype=torch.int32, device=activation.device)
    #     convert_bcsr_cpp.forward_v2(activation, self.row, self.col, self.ext_buffer, self.h, self.w, 1, self.block_w)
    #     # should use the m-dim condense
    #     out = torch.empty(self.out_shape, device=activation.device)
    #     return out


    def forward(self, activation):
        if self.ext_buffer is None:
            act_size = activation.size()
            if(len(act_size)==3):
                self.h, self.w = act_size[0] * act_size[1], act_size[2]
                self.out_shape = (act_size[0], act_size[1], self.weight.size(0))
            else:
                self.h, self.w = act_size
                self.out_shape = (act_size[0], self.weight.size(0))
            row_size = self.h//self.block_w
            self.ext_buffer = torch.empty(2 * self.h + row_size * self.w, dtype=torch.int32, device=activation.device)
            self.row = torch.empty(row_size+1, dtype=torch.int32, device=activation.device)
            self.col = torch.empty(row_size*self.w, dtype=torch.int32, device=activation.device)
        if InSparseLinear.global_seqlen is None:
            convert_bcsr_cpp.forward_v2(activation, self.row, self.col, self.ext_buffer, self.w, self.h, 1, self.block_w)
        else:
            assert( len(activation.size()) == 3)
            convert_bcsr_cpp.forward_v3(activation, self.row, self.col, self.ext_buffer, InSparseLinear.global_seqlen, self.w, self.h, 1, self.block_w, activation.size(0))
        out = torch.empty(self.out_shape, device=activation.device)
        # current the result is not correct just to simulate the speed
        openai_bmm_cpp.forward_condense(self.row, self.col, activation, self.weight, self.h, self.w, self.weight.size(0), self.block_w, 1 , 1, 1)
        return out
