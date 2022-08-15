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

import in_out_elastic_linear_cpp
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)



class InOutElasticLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        activation,
        weight,
        bias,
        in_features,
        out_features
    ):
        ctx.save_for_backward(
            activation,
            weight
        )
        ctx.in_features = in_features
        ctx.out_features = out_features
        return in_out_elastic_linear_cpp.forward(
            activation,
            weight,
            bias,
            in_features,
            out_features
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Not implemented yet
        activation, weight = ctx.saved_tensors
        a_grad, w_grad = in_out_elastic_linear_cpp.backward(
            activation,
            weight,
            grad_outputs[0],
            ctx.in_features,
            ctx.out_features
        )
        # import ipdb; ipdb.set_trace()
        return a_grad, w_grad, None, None, None


class InOutElasticLinear(SparseOPBase):
    """
    
    """
    # if all the sparse attention share the same sparse pattern, then
    # no need to convert the sparse pattern for each module. Set the global_mode
    # to be true when initialize the module


    def __init__(self, ori_linear):
        """
        Parameters
        ----------
        global_mode: bool
            If use the global sparse pattern, if true, then all the sparse_attention
            instance share the same sparse pattern to get the better performance
        """
        super(InOutElasticLinear, self).__init__()

        self.weight = ori_linear.weight
        self.bias = ori_linear.bias
        self.ori_linear = ori_linear

    def forward(self, activation, in_features, out_features):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        sparse_
        """
        if not activation.is_contiguous():
            activation = activation.contiguous()
        assert activation.size(-1) == in_features
        # need create val each time
        assert isinstance(activation, torch.Tensor)
        result = InOutElasticLinearFunction.apply(activation,
                                                self.weight,
                                                self.bias,
                                                in_features,
                                                out_features)
        return result

    def reference_forward(self, activation, in_features, out_features):
        """
        Calculate the reference result the sparse attention to test the correctness.
        """
        return torch.nn.functional.linear(
            activation,
            self.ori_linear.weight[:out_features, :in_features],
            None if self.ori_linear.bias is None else self.ori_linear.bias[:out_features],
        )
