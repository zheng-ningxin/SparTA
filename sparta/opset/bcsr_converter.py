# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
import torch
import types
import logging
from torch.utils.cpp_extension import load as module_load
from .sparse_opbase import SparseOPBase
from sparta.common.utils import *
import convert_bcsr_cpp
import convert_bcsr_transpose_cpp
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class BcsrConverter(SparseOPBase):
    """
    The Sparse Attention module.
    """

    def __init__(self, transpose=False):
        super(BcsrConverter, self).__init__()
        self.csr_row = None
        self.csr_col = None
        self.csr_value = None
        self.csr_row_pos = None
        self.block_index = None
        self.transpose = transpose

    def forward(self, sparse_pattern, dense_values, block_size_h, block_size_w, need_block_index = False):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        """
        assert(isinstance(sparse_pattern, torch.Tensor))
        assert(isinstance(dense_values, torch.Tensor))
        # currently only support on the cuda devices
        assert(sparse_pattern.is_cuda)
        assert(dense_values.is_cuda)
        if not self.transpose:
            self.csr_row, self.csr_col, self.csr_row_pos, self.csr_value, self.block_index = convert_bcsr_cpp.forward(sparse_pattern, dense_values, block_size_h, block_size_w)
        else:
            self.csr_row, self.csr_col, self.csr_row_pos, self.csr_value, self.block_index = convert_bcsr_transpose_cpp.forward(sparse_pattern, dense_values, block_size_h, block_size_w)
        if not need_block_index:
            return self.csr_row, self.csr_col, self.csr_row_pos, self.csr_value
        else:
            return self.csr_row, self.csr_col, self.csr_row_pos, self.csr_value, self.block_index