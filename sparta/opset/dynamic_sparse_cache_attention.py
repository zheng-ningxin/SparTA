import torch
import logging
import os
import copy
from .sparse_opbase import SparseOPBase
import sparse_cache_atten

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class DynamicSparseCacheAttention(SparseOPBase):
    def __init__(self, padding_lens ):
        super(DynamicSparseCacheAttention, self).__init__()
        self.update_padding_lens(padding_lens)
        self.batch_size = padding_lens.size(0)
        self.inter_result = None
        
    def forward(self, Q: torch.Tensor, K:torch.Tensor, V:torch.Tensor, max_token_length:int):
        """
        max_token_length: the number of the regressive iterations
        """
        head_num = Q.size(1)
        batch_size = Q.size(0)
        if self.inter_result is None:
            self.inter_result = torch.empty(batch_size, head_num, 1, max_token_length + 64, dtype=Q.dtype, device=Q.device)
        elif self.inter_result.size(1) < max_token_length+64:
            self.inter_result = torch.empty(batch_size, head_num, 1, max_token_length+64, dtype=Q.dtype, device=Q.device)
        out = sparse_cache_atten.forward(Q, K, V, self.inter_result, self.padding_lens, max_token_length)
        return out

    def update_padding_lens(self, padding_lens):
        self.padding_lens = padding_lens.cuda()
        self.cum_paddings = torch.cumsum(self.padding_lens, dim=0)

    def ref_forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, max_token_length:int):
        dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
        return dots