import torch
import cusparse_csr_cpp

if __name__ == '__main__':
    M = 1024
    K = 1024
    t = torch.rand(M, K).cuda()
    t[10] = 0
    csr_row, csr_col, csr_val = cusparse_csr_cpp.forward(t)
    import ipdb; ipdb.set_trace()
    print('sss')