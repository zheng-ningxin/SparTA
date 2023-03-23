import os
import numpy as np
import re
import csv
os.makedirs('log', exist_ok=True)
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
    
if __name__ == '__main__':
    
    with open('stile_offline.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for sparsity in np.arange(0.5, 1.0, 0.02):
            sparsity = round(sparsity, 2)
            print(sparsity)
            fpath = f'log/stile_{sparsity}.log'
            os.system(f'./stile {sparsity} > {fpath}')
            t_ = parse_result(fpath)
            print(sparsity, t_)
            writer.writerow(str(c) for c in [sparsity, t_])
        