import os

os.makedirs('log', exist_ok=True)

sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
shape = [(4096, 4096, 4096), (4096, 768, 3072)]
blocksize = [(1, 32), (64, 1), (64, 32)]

def run_cusparse():
    for s in sparsity:
        for m,k,n in shape:
            for block_h, block_w in blocksize:
                try:
                    print(f'cusparse {s} {m} {k} {n} {block_h} {block_w}')
                    os.system(f'./cusparse {s} {m} {k} {n} {block_h} {block_w} > log/cusparse_{s}_{m}_{k}_{n}_{block_h}_{block_w}.log')
                except Exception as err:
                    print(err)

def run_sputnik():
    for s in sparsity:
        for m,k,n in shape:
            for block_h, block_w in blocksize:
                try:
                    print(f'sputnik {s} {m} {k} {n} {block_h} {block_w}')
                    os.system(f'./sputnik {s} {m} {k} {n} {block_h} {block_w} > log/sputnik_{s}_{m}_{k}_{n}_{block_h}_{block_w}.log')
                except Exception as err:
                    print(err)


def run_triton():
    for s in sparsity:
        for m,k,n in shape:
            for block_h, block_w in blocksize:
                try:
                    print(f'triton {s} {m} {k} {n} {block_h} {block_w}')
                    os.system(f'python test_openai_bmm.py {s} {m} {k} {n} {block_h} {block_w} > log/triton_{s}_{m}_{k}_{n}_{block_h}_{block_w}.log')
                except Exception as err:
                    print(err)


if __name__ == '__main__':
    # run_cusparse()
    # run_sputnik()
    run_triton()

