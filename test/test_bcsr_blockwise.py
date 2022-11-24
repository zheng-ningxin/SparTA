import torch
from sparta.opset.bcsr_converter_blockwise import BcsrConverterBlockwise

if __name__ == '__main__':
    convert = BcsrConverterBlockwise(False)
    sparse_pattern = torch.zeros(768, 3072, dtype=torch.int32).cuda()
    # sparse_pattern[1,:] = 1
    # sparse_pattern[5,:] = 1
    sparse_pattern[:,:] = 1
    csr_row, csr_col = convert(sparse_pattern)
    import ipdb; ipdb.set_trace()

    print(csr_row)
    print(csr_col)
