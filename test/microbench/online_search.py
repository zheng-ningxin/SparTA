import csv
import re
import sys
from cover_sim import sim

def load_stile_db():
    stiles = {}
    with open('stile_db.csv', 'r') as f:
        reader =  csv.reader(f, delimiter=',')
        for row in reader:
            block_h, block_w, sparsity, lat = int(row[0]), int(row[1]), float(row[2]), float(row[3])
            if ((block_h, block_w)) not in stiles:
                stiles[(block_h, block_w)] = {sparsity:lat}
            else:
                stiles[(block_h, block_w)][sparsity] = lat
    return stiles
if __name__ == '__main__':
    H, W = 4096, 4096
    dense_time = 10.22
    BLOCKS = [(2, 1), (3, 1), (4, 1), (2,2), (4,4),(8,1),(7,1)]
    stiles = load_stile_db()
    import ipdb; ipdb.set_trace()
    sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    with open('online_search.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for block_h, block_w in BLOCKS:
            for s in sparsity:
                t_min = dense_time
                best = "Dense"
                corres_sparsity = 0
                for d_h, d_w in stiles:
                    new_s = sim(s, H, W, block_h, block_w, d_h, d_w)
                    for _s in stiles[(d_h, d_w)]:
                        if _s < new_s:
                            lat = stiles[(d_h, d_w)][_s]
                            if lat < t_min:
                                best = (d_h, d_w)                        
                                corres_sparsity = _s
                                t_min = lat
                writer.writerow(str(c) for c in [block_h, block_w, s, best, corres_sparsity, t_min])
                print(block_h, block_w, s, best, corres_sparsity, t_min)