# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import random
from sparta.opset import *
from sparta.common.utils import convert_bcsr, verify_bcsr

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


def test(h, w, block_h, block_w, sparsity):

    device = torch.device('cuda')
    block_weight = torch.rand(h//block_h, w//block_w).to(device)
    block_mask = (block_weight > sparsity).to(torch.int32)
    mask = convert_to_full_mask(block_mask, (block_h, block_w))
    print('real sparsity: ', 1 - torch.sum(block_mask)/block_mask.numel())

    dense_value = torch.rand(h, w).to(device)
    dense_value *= mask
    converter = BcsrConverter()
    RUNTIME = 1000
    torch.cuda.synchronize()    
    t_start = time.time()
    for i in range(RUNTIME):
        row, col, row_pos, value = converter(mask, dense_value, block_h, block_w)
    torch.cuda.synchronize()
    t_end = time.time()
    print(f"H:{h} W:{w} Block_h:{block_h} Block_w:{block_w} Sparsity:{sparsity} Time: ",(t_end-t_start)*1000/RUNTIME)
    
if __name__ == '__main__':
    for h, w in [(4096, 4096)]:
        for block_h, block_w in [(4,4), (8,8), (16,16), (32,32)]:
            for sparsity in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
                test(h, w, block_h, block_w, sparsity)
