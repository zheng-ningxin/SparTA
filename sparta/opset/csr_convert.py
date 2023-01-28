# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
import torch
import types
import logging
from .sparse_opbase import SparseOPBase
from sparta.common.utils import *
import convert_csr_cpp

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class CsrConverter(SparseOPBase):
    """
    The Sparse Attention module.
    """

    def __init__(self, transpose=False):
        super(CsrConverter, self).__init__()
        self.csr_row = None
        self.csr_col = None
        self.csr_value = None
        self.transpose = transpose

    def forward(self, sparse_pattern, dense_values):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        """
        assert(isinstance(sparse_pattern, torch.Tensor))
        assert(isinstance(dense_values, torch.Tensor))
        # currently only support on the cuda devices
        assert(sparse_pattern.is_cuda)
        assert(dense_values.is_cuda)
        self.csr_row, self.csr_col, self.csr_value = convert_csr_cpp.forward(sparse_pattern, dense_values)
        return self.csr_row, self.csr_col, self.csr_value
