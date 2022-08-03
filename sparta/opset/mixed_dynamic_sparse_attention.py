# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from distutils.bcppcompiler import BCPPCompiler
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging
from .bcsr_converter import BcsrConverter
from .sparse_opbase import SparseOPBase
from sparta.codegen.template.sparse_attention import *
from sparta.common.utils import *
import mixed_dynamic_sparse_attention_cpp

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)



class MixedDynamicSparseAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        inter_result,
        static_bcsr_row,
        static_bcsr_col,
        static_bcsr_row_pos,
        static_bcsr_val_mask,
        global_atten,
        block_h, block_w, block_nnz
    ):
        # Q, K, V have the same shape: [Batchsize, Headnum, Sequence length, hidden_n]
        static_qxk = mixed_dynamic_sparse_attention_cpp.batch_matmul_block_sparse_out(Q, K, static_bcsr_row_pos, static_bcsr_col, inter_result, block_h, block_w, block_nnz)
        dynamic_cols = K.index_select(global_atten, dim=2)
        dynamic_qxk = torch.einsum('bcxd,bcyd->bcxy', (Q, dynamic_cols))
        # directly rewrite the global attention parts
        static_qxk[:,:,:,global_atten] = dynamic_qxk
        

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Not implemented yet
        pass


class MixedDynamicSparseAttention(SparseOPBase):
    """
    The Sparse Attention module that support the dynamic sparse pattern.
    """
    # if all the sparse attention share the same sparse pattern, then
    # no need to convert the sparse pattern for each module. Set the global_mode
    # to be true when initialize the module
    global_dynamic_attention = None
    global_static_attention = None
    global_static_bcsr_row = None
    global_static_bcsr_col = None
    global_static_bcsr_row_pos = None
    global_static_bcsr_val_mask = None
    global_static_block_nnz = None
    global_static_block_h = 32
    global_static_block_w = 32
    global_bcsr_converter = BcsrConverter()
    @staticmethod
    def set_global_dynamic_attention(dynamic_pattern):
        # seqlens is an one-dimension tensor with size of [Batchsize]
        # each element in the tensor represents the effective sequence
        # length of current instance
        assert isinstance(dynamic_pattern, torch.Tensor)
        assert dynamic_pattern.is_cuda
        assert dynamic_pattern.dtype == torch.int32, "only support int32 type"
        MixedDynamicSparseAttention.global_dynamic_attention = dynamic_pattern
        
    @staticmethod
    def set_global_static_attention(static_pattern):
        # seqlens is an one-dimension tensor with size of [Batchsize]
        # each element in the tensor represents the effective sequence
        # length of current instance
        assert isinstance(static_pattern, torch.Tensor)
        assert static_pattern.is_cuda
        assert static_pattern.dtype == torch.int32, "only support int32 type"
        MixedDynamicSparseAttention.global_static_pattern = static_pattern
        MixedDynamicSparseAttention.global_static_bcsr_row, MixedDynamicSparseAttention.global_static_bcsr_col, \
        MixedDynamicSparseAttention.global_static_bcsr_row_pos, MixedDynamicSparseAttention.global_static_bcsr_val_mask = \
            MixedDynamicSparseAttention.global_bcsr_converter(MixedDynamicSparseAttention.global_static_attention, \
            MixedDynamicSparseAttention.global_static_attention.to(torch.float32), MixedDynamicSparseAttention.global_static_block_h,\
            MixedDynamicSparseAttention.global_static_block_w)
        n_row = MixedDynamicSparseAttention.global_static_sparse_pattern.size(0) // MixedDynamicSparseAttention.global_static_block_h
        MixedDynamicSparseAttention.global_static_block_nnz = MixedDynamicSparseAttention.global_static_bcsr_row[n_row].item()

    def __init__(self, global_mode=True, static_pattern=None):
        """
        Parameters
        ----------
        HEAD_NUM: int
            The number of heads of the sparse attention
        max_seq_length: int
            The maximum length of the input sequence
        global_mode: bool
            If use the global sparse pattern, if true, then all the sparse_attention
            instance share the same sparse pattern to get the better performance
        """
        super(MixedDynamicSparseAttention, self).__init__()
        self.global_mode = global_mode
        # currently only support 32 x 64
        self.inter_result = None  # tensor to store the internal results
        self.static_bcsr_row = None
        self.static_bcsr_col = None
        self.static_bcsr_row_pos = None
        self.static_bcsr_val_mask = None
        self.static_block_nnz = None
        
        
        if not self.global_model:
            assert isinstance(static_pattern, torch.Tensor)
            assert static_pattern.is_cuda
            self.static_bcsr_row, self.static_bcsr_col, self.static_bcsr_row_pos, self.static_bcsr_val_mask = \
                MixedDynamicSparseAttention.global_bcsr_converter(static_pattern)
            self.static_block_nnz = self.static_bcsr_row[static_pattern.size(0)//MixedDynamicSparseAttention.global_block_h]
        else:
            self.static_bcsr_row, self.static_bcsr_col, self.static_bcsr_row_pos, self.static_bcsr_val_mask = \
                MixedDynamicSparseAttention.global_static_bcsr_row, MixedDynamicSparseAttention.global_static_bcsr_col, \
                MixedDynamicSparseAttention.global_static_bcsr_row_pos, MixedDynamicSparseAttention.global_static_bcsr_val_mask
            self.static_block_nnz = MixedDynamicSparseAttention.global_static_block_nnz

    def forward(self, Q, K, V, dynamic_attention=None):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        sparse_
        """
        if not Q.is_contiguous():
            Q = Q.contiguous()
        if not K.is_contiguous():
            K = K.contiguous()
        if not V.is_contiguous():
            V = V.contiguous()
        if self.global_mode is not True:
            assert isinstance(dynamic_attention, torch.Tensor)
        else:
            dynamic_attention = MixedDynamicSparseAttention.global_dynamic_attention
        # need create val each time
        assert isinstance(Q, torch.Tensor)
        assert isinstance(K, torch.Tensor)
        assert isinstance(V, torch.Tensor)
        # Shape of tensor Q should be {Batchsize, sequence length, hidden dim}
        batch_size = Q.size(0)
        head_num =  Q.size(1)
        max_seq_len = Q.size(2)
        hidden_dim = Q.size(3)
        err_msg = 'Currently, seq_len and hidden_dim should be divisible by 32'
        assert max_seq_len % 32 == 0, err_msg
        assert hidden_dim % 32 == 0
        if self.inter_result is None or self.inter_result.numel() < batch_size * head_num * max_seq_len * max_seq_len:
            self.inter_result = torch.zeros(batch_size * head_num * max_seq_len * max_seq_len,
                          dtype=torch.float32, device=Q.device)
        result = MixedDynamicSparseAttentionFunction.apply(Q, K, V,
                                                      self.inter_result,
                                                      self.static_bcsr_row,
                                                      self.static_bcsr_col,
                                                      self.static_bcsr_row_pos,
                                                      self.static_bcsr_val_mask,
                                                      dynamic_attention,
                                                      MixedDynamicSparseAttention.global_block_h,
                                                      MixedDynamicSparseAttention.global_block_w,
                                                      self.static_block_nnz)

        return result

    def reference_forward(self, Q, K, V, dynamic_attention):
        """
        Calculate the reference result the sparse attention to test the correctness.
        """
        pass
        # add_mask = torch.zeros(attention_mask.size()).to(Q.device)
        # add_mask[attention_mask == 0] = float(-inf)
        # dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
        # added = torch.add(dots, add_mask)
        # attn = added.softmax(dim=-1)
        # nan_pos = torch.isnan(attn)
        # attn[nan_pos] = 0.0
        # ref_out = torch.einsum('b h m n, b h n k -> b h m k', attn, V)

        # return ref_out
