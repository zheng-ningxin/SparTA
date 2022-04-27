# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging

from .SparseOPBase import SparseOPBase
from .Template.SparseAttention import *
from SparTA.Common.Utils import *
from .BcsrConverter import BcsrConverter
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
our_sparse_attention = None


class DynamicSparseAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        row_ptr,
        col_idx,
        val_mask
    ):
        ctx.save_for_backward(
            Q,
            K,
            V,
            row_ptr,
            col_idx
        )

        return dynamic_sparse_attention.forward(
            Q,
            K,
            V,
            row_ptr,
            col_idx,
            val_mask
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass


class DynamicSparseAttention(SparseOPBase):
    """
    The Sparse Attention module that support the dynamic sparse pattern.
    """
    # if all the sparse attention share the same sparse pattern, then
    # no need to convert the sparse pattern for each module. Set the global_mode
    # to be true when initialize the module
    global_sparse_pattern = None
    global_bcsr_row = None
    global_bcsr_col = None
    global_bcsr_val_mask = None
    global_converter = BcsrConverter()

    @staticmethod
    def set_global_sparse_pattern(sparse_pattern):
        assert isinstance(sparse_pattern, torch.Tensor)
        assert sparse_pattern.dtype == torch.int32, "only support int32 type"
        DynamicSparseAttention.global_sparse_pattern = sparse_pattern
        DynamicSparseAttention.global_bcsr_row, DynamicSparseAttention.global_bcsr_col, \
            DynamicSparseAttention.global_bcsr_val_mask = DynamicSparseAttention.global_converter(
                DynamicSparseAttention.global_sparse_pattern, DynamicSparseAttention.global_sparse_pattern.to(torch.float))

    def __init__(self, HEAD_NUM, max_seq_len, global_mode=True):
        super(DynamicSparseAttention, self).__init__()
        self.HEAD_NUM = HEAD_NUM
        self.max_seq_len = max_seq_len
        self.global_mode = global_mode
        err_msg = 'Currently, seq_len and hidden_dim should be divisible by 32'
        assert max_seq_len % 32 == 0, err_msg
        # currently only support 32 x 64
        self.block_size_h = 32
        self.block_size_w = 32
        self.target_device = None
        self.converter = BcsrConverter() 

    def forward(self, Q, K, V, sparse_mask=None):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        sparse_
        """
        if self.global_mode is not True:
            assert isinstance(sparse_mask, torch.Tensor)
            csr_row, csr_col, csr_value_mask = self.converter(sparse_mask, sparse_mask.to(torch.float), self.block_size_h, self.block_size_w)
        else:
            csr_row, csr_col, csr_value_mask = DynamicSparseAttention.global_bcsr_row, DynamicSparseAttention.global_bcsr_col, DynamicSparseAttention.global_bcsr_val_mask 
            sparse_mask = DynamicSparseAttention.global_sparse_pattern
        # need create val each time
        assert isinstance(Q, torch.Tensor)
        assert isinstance(K, torch.Tensor)
        assert isinstance(V, torch.Tensor)
        # Shape of tensor Q should be {Batchsize, sequence length, hidden dim}
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        hidden_dim = Q.size(2)
        assert seq_len % 32 == 0
        assert hidden_dim % 32 == 0
        assert seq_len < self.max_seq_len
        sparse_val_size = (csr_row[sparse_mask.size(0)//self.block_size_h] + 1) * self.block_size_h * self.block_size_w
        val = torch.zeros(batch_size * self.HEAD_NUM * sparse_val_size,
                          dtype=torch.float32, device=self.target_device)
        result = DynamicSparseAttentionFunction.apply(Q, K, V,
                                                val,
                                                csr_row,
                                                csr_col,
                                                csr_value_mask)

        return result

    def reference_forward(self, Q, K, V):
        """
        Calculate the reference result the sparse attention to test the correctness.
        """
        add_mask = torch.zeros(self.out_mask.size()).to(self.target_device)
        add_mask[self.out_mask == 0] = float(-inf)
        dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
        added = torch.add(dots, add_mask)
        attn = added.softmax(dim=-1)
        ref_out = torch.einsum('b h m n, b h n k -> b h m k', attn, V)

        return ref_out

    def _move_index(self, target_device):
        with torch.no_grad():
            # move the index tensors to the target device
            self.row_ptr, self.col_index, self.val_mask, self.m_index, self.n_index, self.block_index,\
                self.col_range_index, self.gradv_row_idx, self.gradv_col_idx, self.gradv_subblock_idx =\
                self.row_ptr.to(target_device), self.col_index.to(target_device), self.val_mask.to(target_device),\
                self.m_index.to(target_device), self.n_index.to(target_device), self.block_index.to(target_device),\
                self.col_range_index.to(target_device), self.gradv_row_idx.to(target_device), self.gradv_col_idx.to(target_device),\
                self.gradv_subblock_idx.to(target_device)

    def _build_forward_index(self):
        block_idx = []
        m_idx = []
        n_idx = []
        large_block_cnt = 0
        # dummy code copy from Quanlu
        M, N = self.out_mask.size()
        for m in range(M//self.block_size_h):
            for n in range(N//self.block_size_w):
                m_start = m * self.block_size_h
                m_end = m_start + self.block_size_h
                n_start = n * self.block_size_w
                n_end = n_start + self.block_size_w
                n_mid = n_start + self.block_size_w//2

                if torch.sum(self.out_mask[m_start:m_end, n_start:n_end]) > 0:
                    if torch.sum(self.out_mask[m_start:m_end, n_start:n_mid]) > 0:
                        m_idx.append(m)
                        n_idx.append(2*n)
                        block_idx.append(2*large_block_cnt)
                    if torch.sum(self.out_mask[m_start:m_end, n_mid:n_end]) > 0:
                        m_idx.append(m)
                        n_idx.append(2*n+1)
                        block_idx.append(2*large_block_cnt+1)
                    large_block_cnt += 1
        # TODO fix me, bug here when there is a blank line
        col_range_index = [0] * (M // self.block_size_h + 1)
        for i in range(1, len(block_idx)):
            if m_idx[i] != m_idx[i-1]:
                for row_id in range(m_idx[i-1]+1, m_idx[i]+1):
                    col_range_index[row_id] = i
        col_range_index[M//self.block_size_h] = len(block_idx)
        return torch.tensor(m_idx, dtype=torch.int32), torch.tensor(n_idx, dtype=torch.int32),  \
            torch.tensor(block_idx, dtype=torch.int32), torch.tensor(
                col_range_index, dtype=torch.int32)

    def _build_backward_index(self):
        """
        Build the index used to calculate the gradient of V.
        """
        # append a zero block at the end of the vals
        zero_idx = self.col_index.size(
            0) * self.block_size_h * self.block_size_w  # TODO point to the zeros at the end of the val
        self.zero_idx = zero_idx
        t_mask = self.out_mask.data.t()
        gradv_row_idx, gradv_col_idx, _ = convert_bcsr(
            t_mask, t_mask, block_h=self.block_size_h, block_w=self.block_size_w)
        subblock_idx = []
        for row_id in range(gradv_row_idx.size(0)-1):
            index_start = gradv_row_idx[row_id]
            index_end = gradv_row_idx[row_id+1]
            for _pos in range(index_start, index_end):
                # Note: there must be a subblock of zeros at the end of vals
                # the
                col_id = gradv_col_idx[_pos].item()
                i_start = row_id * self.block_size_h
                i_end = i_start + self.block_size_h
                j_start = col_id * self.block_size_w
                j_mid = j_start + self.block_size_w // 2
                j_end = j_start + self.block_size_w
                # print(i_start, i_end, j_start, j_mid, j_end)
                if torch.sum(t_mask[i_start:i_end, j_start:j_mid]) > 0:
                    # left subblock has values to be computed
                    # left upper corner(i_start, j_start)
                    subblock_i, subblock_j = j_start//self.block_size_h, i_start//self.block_size_w
                    subblock_pos = self.csr_index[subblock_i][subblock_j] * self.block_size_h * self.block_size_w + 32 * (
                        i_start % self.block_size_w != 0)
                    subblock_idx.append(subblock_pos)
                else:
                    subblock_idx.append(zero_idx)

                if torch.sum(t_mask[i_start:i_end, j_mid:j_end]) > 0:
                    # right subblock has values to be computed
                    subblock_i, subblock_j = j_mid//self.block_size_h, i_start//self.block_size_w
                    subblock_pos = self.csr_index[subblock_i][subblock_j] * self.block_size_h * self.block_size_w + 32 * (
                        i_start % self.block_size_w != 0)
                    subblock_idx.append(subblock_pos)
                else:
                    subblock_idx.append(zero_idx)
        gradv_subblock_idx = torch.tensor(subblock_idx, dtype=torch.int32)

        return gradv_row_idx, gradv_col_idx, gradv_subblock_idx

