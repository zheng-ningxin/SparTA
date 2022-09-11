import torch
import cusparse_csr_cpp

if __name__ == '__main__':
    M = 1024
    K = 1024
    t = torch.rand(M, K).cuda()
    t[10] = 0
    csr_row, csr_col, csr_val = cusparse_csr_cpp.forward(t)
    nnz = csr_row[M]
    n_row = M
    n_col = K
    dense_t = cusparse_csr_cpp.backward(csr_row, csr_col, csr_val, n_row, n_col, nnz)
    assert torch.allclose(t, dense_t, rtol=1e-08, atol=1e-03)
    import ipdb; ipdb.set_trace()
    print('sss')