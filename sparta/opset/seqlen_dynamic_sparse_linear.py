# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging

from .sparse_opbase import SparseOPBase
from sparta.common.utils import *

import seqlen_dynamic_sparse_linear_cpp
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)



class SeqlenDynamicSparseLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        activation,
        weight,
        bias,
        seqlens
    ):
        # ctx.save_for_backward(
        # )
        if bias is not None:
            return seqlen_dynamic_sparse_linear_cpp.forward(
                activation,
                weight,
                bias,
                seqlens
            )
        else:
            return seqlen_dynamic_sparse_linear_cpp.forward2(
                activation,
                weight,
                seqlens
            )
    @staticmethod
    def backward(ctx, *grad_outputs):
        # Not implemented yet
        pass


class SeqlenDynamicSparseLinear(SparseOPBase):
    """
    The Sparse Attention module that support the dynamic sparse pattern.
    """
    # if all the sparse attention share the same sparse pattern, then
    # no need to convert the sparse pattern for each module. Set the global_mode
    # to be true when initialize the module
    global_seqlen = None

    @staticmethod
    def set_global_seqlens(seqlens):
        # seqlens is an one-dimension tensor with size of [Batchsize]
        # each element in the tensor represents the effective sequence
        # length of current instance
        assert isinstance(seqlens, torch.Tensor)
        assert seqlens.is_cuda
        assert seqlens.dtype == torch.int32, "only support int32 type"
        SeqlenDynamicSparseLinear.global_seqlen = seqlens


    def __init__(self, ori_linear, global_mode=True):
        """
        Parameters
        ----------
        global_mode: bool
            If use the global sparse pattern, if true, then all the sparse_attention
            instance share the same sparse pattern to get the better performance
        """
        super(SeqlenDynamicSparseLinear, self).__init__()
        self.global_mode = global_mode
        # currently only support 32 x 64
        self.inter_result = None  # tensor to store the internal results
        self.weight = ori_linear.weight
        self.bias = ori_linear.bias
        self.ori_linear = ori_linear

    def forward(self, activation, seqlens=None):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        sparse_
        """
        if not activation.is_contiguous():
            activation = activation.contiguous()
        if self.global_mode is not True:
            assert isinstance(seqlens, torch.Tensor)
            assert seqlens.size(0) == activation.size(0)
        else:
            seqlens = SeqlenDynamicSparseLinear.global_seqlen.to(activation.device)
        # need create val each time
        assert isinstance(activation, torch.Tensor)
        result = SeqlenDynamicSparseLinearFunction.apply(activation,
                                                         self.weight,
                                                         self.bias,
                                                         seqlens)
        return result

    def reference_forward(self, activation):
        """
        Calculate the reference result the sparse attention to test the correctness.
        """
        ref_out = self.ori_linear(activation)
        return ref_out
