
import re
import os
import sys
import csv

#sparsity_ratio=(0.5, 0.75, 0.9)
sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
shape = [(4096, 4096, 4096), (4096, 768, 3072)]
blocksize = [(1, 32), (64, 1), (64, 32)]

# result = []
# for s in sparsity_ratio:
#     for b in baseline:
#         for kn in KN:
#             k, n = kn
#             for m in M:
#                 fpath = './log/{}_{}_{}_{}_{}.log'.format(b, m, k, n, s)
#                 if not os.path.exists(fpath):
#                     continue
#                 with open(fpath) as f:
#                     lines = f.readlines()
#                     lines = [line for line in lines if 'Time=' in line]
#                     if len(lines) == 0:
#                         continue
#                     tmp = re.split(' ', lines[0])
#                     time = float(tmp[1])
#                     print(s, m, k, n, b, time)
#                     result.append((s, m, k, n, b, time))
# with open('baseline.csv', 'w') as f:
#     writer = csv.writer(f, delimiter=',')
#     for row in result:
#         writer.writerow([str(v) for v in row])

def parse_result(fpath):
    if not os.path.exists(fpath):
        return 0   
    with open(fpath) as f:
        lines = f.readlines()
        lines = [line for line in lines if 'Time=' in line]
        if len(lines) == 0:
            return 0
        tmp = re.split(' ', lines[0])
        _time = float(tmp[1])
        return _time
    
def summary_cusparse():
    with open('cusparse.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for m,k,n in shape:
            for block_h, block_w in blocksize:
                for s in sparsity:
                    f_path = f'./log/cusparse_{s}_{m}_{k}_{n}_{block_h}_{block_w}.log'
                    lat = parse_result(f_path)
                    writer.writerow(str(c) for c in [m, k, n, block_h, block_w, s, lat])

def summary_sputnik():
    with open('sputnik.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for m,k,n in shape:
            for block_h, block_w in blocksize:
                for s in sparsity:
                    f_path = f'./log/sputnik_{s}_{m}_{k}_{n}_{block_h}_{block_w}.log'
                    lat = parse_result(f_path)
                    writer.writerow(str(c) for c in [m, k, n, block_h, block_w, s, lat])

def summary_triton():
    with open('triton.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for m,k,n in shape:
            for block_h, block_w in blocksize:
                for s in sparsity:
                    f_path = f'./log/triton_{s}_{m}_{k}_{n}_{block_h}_{block_w}.log'
                    lat = parse_result(f_path)
                    writer.writerow(str(c) for c in [m, k, n, block_h, block_w, s, lat])


if __name__ == '__main__':
    summary_cusparse()
    summary_sputnik()
    summary_triton()