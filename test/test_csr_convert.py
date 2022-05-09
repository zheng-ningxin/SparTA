# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import random
from SparTA.OPs import *
from SparTA.Common.Utils import convert_bcsr
# def verify(mask, dense_value, row, col, value, block_h, block_w):
#     n_row =  mask.size(0)//block_h
#     for rid in range(n_row):
#         _start = row[rid]
#         _end = row[rid+1]
#         for pos in range(_start, _end):
#             cid =  col[pos]
#             for i in range(block_h):
#                 for j in range(block_w):
#                     if torch.abs(dense_value)

if __name__ == '__main__':
    h = 1024
    w = 1024
    block_h = 32
    block_w = 32
    sparsity = 0.99999
    device = torch.device('cuda')
    dense_value = torch.rand(h, w).to(device)
    k = int(dense_value.numel() * sparsity)
    threshold = torch.topk(dense_value.view(-1), k, largest=False)[0].max()
    mask = (dense_value > threshold).to(torch.int32).to(device)
    dense_value *= mask
    print(dense_value)
    converter = BcsrConverter()
    row, col, row_pos, value = converter(mask, dense_value, block_h, block_w)
    row_ref, col_ref, val_ref = convert_bcsr(mask, dense_value, block_h, block_w)
    import ipdb; ipdb.set_trace()
    # if verify(mask, dense_value, row, col, value, block_h, block_w):
    #     import ipdb; ipdb.set_trace()