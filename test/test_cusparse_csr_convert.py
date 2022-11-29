import time
import torch
import cusparse_csr_cpp
sparsity_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
shape = [(4096, 4096, 4096), (4096, 768, 3072)]

if __name__ == '__main__':
    M = 4096
    K = 4096
    RUNTIME = 1000
    for sparsity in sparsity_ratios:    
        mask = torch.rand(M, K).cuda()
        mask = (mask > sparsity).to(torch.float32)
        print('Sparsity ratio: ', torch.sum(mask)/mask.numel())
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(RUNTIME):
            csr_row, csr_col, csr_val = cusparse_csr_cpp.forward(mask)
        torch.cuda.synchronize()
        t_end = time.time()
        print("Time: ", (t_end-t_start)*1000/RUNTIME)