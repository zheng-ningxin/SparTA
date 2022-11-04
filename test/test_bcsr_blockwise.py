import torch
from sparta.opset.bcsr_converter_blockwise import BcsrConverterBlockwise

if __name__ == '__main__':
    convert = BcsrConverterBlockwise(True)
    sparse_pattern = torch.zeros(10, 10, dtype=torch.int32).cuda()
    sparse_pattern[1,:] = 1
    sparse_pattern[5,:] = 1
    csr_row, csr_col = convert(sparse_pattern)
    
    print(csr_row)
    print(csr_col)