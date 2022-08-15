import torch
import numpy as np
import time
from sparta.opset.in_out_elastic_linear import InOutElasticLinear

def test_correctness(e_linear, in_features, out_features):
    data = torch.rand(32, 128, in_features)
    data_1 = data.clone().detach().cuda()
    data_2 = data.clone().detach().cuda()
    data_1.requires_grad_()
    data_2.requires_grad_()
    
    re = e_linear(data_1, in_features, out_features)
    re2 = e_linear.reference_forward(data_2, in_features, out_features)
    # import ipdb; ipdb.set_trace()
    tmp_grad = torch.rand_like(re)
    re.backward(tmp_grad)
    re2.backward(tmp_grad)
    import ipdb; ipdb.set_trace()
    
    
if __name__ == '__main__':
    M = 4096
    K = 4096
    N = 4096
    ori_linear = torch.nn.Linear(K, N, bias=True).cuda()
    elastic_linear = InOutElasticLinear(ori_linear)
    test_correctness(elastic_linear, 1024, 512)