# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import random
from sparta.opset import *
from sparta.common.utils import convert_bcsr, verify_bcsr


def test_dense():
    h = 1024
    w = 2048
    block_h = 64
    block_w = 32
    sparsity = 0.0
    device = torch.device('cuda')
    dense_value = torch.rand(h, w).to(device)
    k = int(dense_value.numel() * sparsity)
    # threshold = torch.topk(dense_value.view(-1), k, largest=False)[0].max()
    # mask = (dense_value > threshold).to(torch.int32).to(device)
    mask = torch.ones_like(dense_value, dtype=torch.int32)
    dense_value *= mask
    print(dense_value)
    converter = BcsrConverter()
    row, col, row_pos, value = converter(mask, dense_value, block_h, block_w)
    row_ref, col_ref, val_ref = convert_bcsr(mask, dense_value, block_h, block_w)
    # import ipdb; ipdb.set_trace()
    verify_bcsr(mask, dense_value, row, col, value, block_h, block_w)

def test_empty_line():
    h = 1024
    w = 2048
    block_h = 64
    block_w = 32
    sparsity = 0.0
    device = torch.device('cuda')
    dense_value = torch.rand(h, w).to(device)
    k = int(dense_value.numel() * sparsity)
    # threshold = torch.topk(dense_value.view(-1), k, largest=False)[0].max()
    # mask = (dense_value > threshold).to(torch.int32).to(device)
    mask = torch.zeros_like(dense_value, dtype=torch.int32)
    mask[:512]= 1
    dense_value *= mask
    print(dense_value)
    converter = BcsrConverter()
    row, col, row_pos, value = converter(mask, dense_value, block_h, block_w)
    row_ref, col_ref, val_ref = convert_bcsr(mask, dense_value, block_h, block_w)
    import ipdb; ipdb.set_trace()
    verify_bcsr(mask, dense_value, row, col, value, block_h, block_w)
        

if __name__ == '__main__':
    # test_dense()
    test_empty_line()