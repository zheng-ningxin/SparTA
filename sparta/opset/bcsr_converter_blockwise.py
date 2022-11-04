# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import torch
import types
import logging
from .sparse_opbase import SparseOPBase
from sparta.common.utils import *
import convert_bcsr_blockwise_cpp
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class BcsrConverterBlockwise(SparseOPBase):
    """
    The Sparse Attention module.
    """

    def __init__(self, transpose=False):
        super(BcsrConverterBlockwise, self).__init__()
        self.csr_row = None
        self.csr_col = None
        self.transpose = transpose

    def forward(self, sparse_pattern):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        """
        assert(isinstance(sparse_pattern, torch.Tensor))
        assert(sparse_pattern.is_cuda)
        if not self.transpose:
            self.csr_row, self.csr_col = convert_bcsr_blockwise_cpp.forward(sparse_pattern, 0)
        else:
            # import ipdb; ipdb.set_trace()
            self.csr_row, self.csr_col = convert_bcsr_blockwise_cpp.forward(sparse_pattern, 1)
        return self.csr_row, self.csr_col