# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch

def convert_bcsr(weight_m, weight_v, block_h=1, block_w=1):
    assert len(weight_m.size()) == 2, "Only support two dimension"
    weight_m = torch.abs(weight_m)
    size_h, size_w = weight_m.size()
    if size_h % block_h != 0 or size_w % block_w != 0:
        return None, None, None
    rows = []
    cols = []
    values = []
    for _i in range(size_h//block_h):
        rows.append(len(cols))
        for _j in range(size_w//block_w):
            i_start = _i * block_h
            i_end = (_i+1) * block_h
            j_start = _j * block_w
            j_end = (_j+1) * block_w
            if torch.sum(weight_m[i_start:i_end, j_start:j_end]) > 0:
                cols.append(_j)
                values.extend(weight_v[i_start:i_end,j_start:j_end].flatten().tolist())
    rows.append(len(cols))
    t_rows = torch.tensor(rows).to(torch.int32)
    t_cols = torch.tensor(cols).to(torch.int32)
    t_values = torch.tensor(values)
    return t_rows, t_cols, t_values