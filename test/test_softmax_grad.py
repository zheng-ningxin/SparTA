from curses.ascii import SO
import torch
import torch.nn.functional as F

def get_softmax_grad(input, out_grad):
    _input = input.clone().detach()
    _input.requires_grad_()
    out = F.softmax(_input)
    out.backward(out_grad)
    return _input.grad

def my_func(input, out_grad):
    output = F.softmax(input, dim=-1)
    O = output * out_grad
    re = torch.zeros_like(output)
    _sum = torch.sum(O, dim=1)
    for i in range(output.size(0)):
        for j in range(input.size(1)):
            re[i][j]=output[i][j] * out_grad[i][j] - _sum[i] * output[i][j]
            
    return re

if __name__ == '__main__':
    M = 10
    N = 100
    input = torch.rand(M, N)
    out_grad = torch.rand(M, N)
    ref_grad = get_softmax_grad(input, out_grad)
    grad = my_func(input, out_grad)
    import ipdb; ipdb.set_trace()