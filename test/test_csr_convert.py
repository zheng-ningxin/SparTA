# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from sqlite3 import converters
import time
import torch
import random
from SparTA.OPs import *


if __name__ == '__main__':
    h = 1024
    w = 1024
    block_h = 4
    block_w = 4
    sparsity = 0.99
    device = torch.device('cuda')
    dense_value = torch.rand(h, w).to(device)
    k = int(dense_value.numel() * sparsity)
    threshold = torch.topk(dense_value.view(-1), k, largest=False)[0].max()
    mask = (dense_value > threshold).to(torch.int32).to(device)
    dense_value *= mask
    print(dense_value)
    converter = BcsrConverter()
    row, col, value = converter(mask, dense_value, block_h, block_w)