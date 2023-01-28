import torch
from sparta.opset.csr_convert import CsrConverter

if __name__ == '__main__':
    M = 1024
    K = 1024
    N = 1024
    sparsity_ratio = 0.5
    val = torch.rand(M,K).cuda()
    mask = (val > sparsity_ratio).to(torch.int32)
    converter = CsrConverter()
    csr_row, csr_col, csr_val = converter(mask, val)
    import ipdb; ipdb.set_trace()
    pass