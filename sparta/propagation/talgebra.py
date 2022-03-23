import torch
from sparta.common.sparsity import TeSA

class Algebra:
    """
    TeSA algebra for pruning
    True means non-pruned
    False means pruned
    """
    def __init__(self, val):
        self.val = bool(val)

    def __add__(self, ele):
        return self.val or ele.val

    def __mul__(self, ele):
        return self.val and ele.val


def algebra_matmul(tesa_in1: TeSA, tesa_in2: TeSA, tesa_out: TeSA):
    """
    tesa_out = tesa_in1 * tesa_in2
    (m,n) = (m,k) * (k,n)
    """
    m, n = tesa_out.tesa.size()
    k = tesa_in1.tesa.size()[1]
    for i in range(m):
        for j in range(n):
            tmp = Algebra(tesa_out.tesa[i][j])
            for x in range(k):
                tmp += Algebra(tesa_in1.tesa[i][x]) * Algebra(tesa_in1.tesa[j][x])
            tesa_out.tesa[m][n] = tmp.val


def transpose(tesa: TeSA):
    return TeSA(torch.transpose(tesa.tesa))


def propagate_matmul(tesa_in1: TeSA, tesa_in2: TeSA, tesa_out: TeSA) -> tuple(TeSA, TeSA, TeSA):
    """
    sparsity propagation on matmul, both forward and backward
    """
    # forward propagation
    algebra_matmul(tesa_in1, tesa_in2, tesa_out)
    # backward propagation
    algebra_matmul(tesa_out, transpose(tesa_in2), tesa_in1)
    algebra_matmul(transpose(tesa_in1), tesa_out, tesa_in2)