import torch
import logging
import os
import copy
from .sparse_opbase import SparseOPBase
import sparse_cache_atten

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class DynamicSparseCacheAttention(SparseOPBase):
    global_padding_len = None
    global_min_padding_len = None
    global_padding_updated = False
    @staticmethod
    def update_global_padding_lens(padding_len):
        DynamicSparseCacheAttention.global_padding_updated = True
        DynamicSparseCacheAttention.global_padding_len = padding_len.cuda()
        DynamicSparseCacheAttention.global_min_padding_len = torch.min(DynamicSparseCacheAttention.global_padding_len).item()
    
    def __init__(self, padding_lens, global_mode=False):
        super(DynamicSparseCacheAttention, self).__init__()
        self.global_mode = global_mode
        if not global_mode:
            self.update_padding_lens(padding_lens)
        self.batch_size = padding_lens.size(0)
        self.inter_result = None
        
    def forward(self, Q: torch.Tensor, K:torch.Tensor, V:torch.Tensor, max_token_length:int):
        """
        max_token_length: the number of the regressive iterations
        """
        if self.global_mode and DynamicSparseCacheAttention.global_padding_updated:
            self.padding_lens = DynamicSparseCacheAttention.global_padding_len.to(Q.device)
            self.min_padding_len = DynamicSparseCacheAttention.global_min_padding_len
            DynamicSparseCacheAttention.global_padding_updated = False
        head_num = Q.size(1)
        batch_size = Q.size(0)
        if self.inter_result is None:
            self.inter_result = torch.empty(batch_size, head_num, 1, max_token_length + 64, dtype=Q.dtype, device=Q.device)
        elif self.inter_result.size(1) < max_token_length+64:
            self.inter_result = torch.empty(batch_size, head_num, 1, max_token_length+64, dtype=Q.dtype, device=Q.device)
        out = sparse_cache_atten.forward(Q, K, V, self.inter_result, self.padding_lens, max_token_length, self.min_padding_len)
        return out

    def update_padding_lens(self, padding_lens):
        self.padding_lens = padding_lens.cuda()
        self.min_padding_len = torch.min(self.padding_lens).item()
        # self.cum_paddings = torch.cumsum(self.padding_lens, dim=0)
        

    def ref_forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, max_token_length:int):
        if self.global_mode and DynamicSparseCacheAttention.global_padding_updated:
            self.padding_lens = DynamicSparseCacheAttention.global_padding_len.to(Q.device)
            self.min_padding_len = DynamicSparseCacheAttention.global_min_padding_len
            DynamicSparseCacheAttention.global_padding_updated = False
        dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
        # FILL_VAL = 0
        FILL_VAL = -10000.0
        dots[:,:,:,max_token_length:] = FILL_VAL
        for i in range(dots.size(0)):
            dots[i,:,:,:self.padding_lens[i]] = FILL_VAL
        dots = dots.to(torch.float32)
        score = torch.softmax(dots, dim=-1).to(torch.float16)
        ref_out = torch.einsum('b h m n, b h n k -> b h m k', score, V)
        return ref_out