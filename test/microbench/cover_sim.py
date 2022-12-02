import torch
import argparse
import sys
import csv
from sparta.common.utils import convert_bcsr

shape = [(4096, 4096)]
sparsity_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
ori_blocks = [(2,1), (3, 1), (2, 2), (4, 4)]
data_tiles = [(4,1), (8,1), (16,1), (32,1)]
def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


def sim(sparsity, h, w, ori_block_h, ori_block_w, new_block_h, new_block_w):

    full_mask = torch.zeros(h, w, dtype=torch.int32)
    ori_block_weight = torch.rand(h//ori_block_h, w//ori_block_w)
    ori_block_mask = (ori_block_weight > sparsity).to(torch.int32)
    ori_full_mask = convert_to_full_mask(ori_block_mask, (ori_block_h, ori_block_w))
    tmp_h, tmp_w = h//ori_block_h*ori_block_h, w//ori_block_w*ori_block_w
    assert ori_full_mask.size(0) == tmp_h
    assert ori_full_mask.size(1) == tmp_w
    full_mask[:tmp_h, :tmp_w] = ori_full_mask
    conv=torch.nn.Conv2d(1, 1,(new_block_h, new_block_w), (new_block_h, new_block_w), bias=False)
    conv.eval()
    conv.weight.data[:] = 1
    # row, col, val = convert_bcsr(full_mask, full_mask.to(torch.float32), new_block_h, new_block_w)
    # bnnz = row[h//new_block_h]
    # new_sparsity_ratio = 1.0 - bnnz / (h*w/new_block_h/new_block_w)    
    tmp_full = full_mask.view(1, 1, h, w).to(torch.float32)
    new_block_mask = conv(tmp_full) > 0
    new_sparsity_ratio = 1-torch.sum(new_block_mask)/ new_block_mask.numel()    
    return new_sparsity_ratio


if __name__ == '__main__':
    with open('cover_sim.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for h, w in shape:
            for ori_block_h, ori_block_w in ori_blocks:
                for new_block_h, new_block_w in data_tiles:
                    for sparsity in sparsity_ratios:
                        sim_sparsity = sim(sparsity, h, w, ori_block_h, ori_block_w, new_block_h, new_block_w)
                        print(f'h:{h} w:{w} {ori_block_h}, {ori_block_w}-> {new_block_h}, {new_block_w} , {sparsity}->{sim_sparsity}')
                        writer.writerow(str(c) for c in [h, w, ori_block_h, ori_block_w, new_block_h, new_block_w, sparsity, sim_sparsity.item()])