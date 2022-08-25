import torch
import numpy as np
import time
import copy
from sparta.opset.in_out_elastic_linear import InOutElasticLinear


def reference_forward(ori_linear, activation, in_features, out_features):
    """
    Calculate the reference result the sparse attention to test the correctness.
    """
    return torch.nn.functional.linear(
        activation,
        ori_linear.weight[:out_features, :in_features],
        None if ori_linear.bias is None else ori_linear.bias[:out_features],
    )


def test_correctness(e_linear, in_features, out_features):
    data = torch.rand(8, 128, in_features)
    data_1 = data.clone().detach().cuda()
    data_2 = data.clone().detach().cuda()
    data_1.requires_grad_()
    data_2.requires_grad_()
    ori_linear = copy.deepcopy(e_linear.ori_linear)
    re = e_linear(data_1, in_features, out_features)
    re2 = reference_forward(ori_linear, data_2, in_features, out_features)
    # import ipdb; ipdb.set_trace()
    tmp_grad = torch.rand_like(re)
    re.backward(tmp_grad)
    re2.backward(tmp_grad)
    import ipdb; ipdb.set_trace()
    
def test_speed(e_linear, in_features, out_features):
    data = torch.rand(batch, seq_len, in_features).cuda()
    data.requires_grad_()
    re = e_linear(data, in_features, out_features)
    tmp_grad = torch.rand_like(re)
    run_time = 1000
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(run_time):
        re = e_linear(data, in_features, out_features)
        re.backward(tmp_grad)
    torch.cuda.synchronize()
    t_end = time.time()
    print('Sparse per batch(ms):' , (t_end-t_start)*1000/run_time)
    
def dense_profile(e_linear, in_features, out_features):
    data = torch.rand(batch, seq_len, in_features).cuda()
    data.requires_grad_()
    re = e_linear.reference_forward(data, in_features, out_features)
    tmp_grad = torch.rand_like(re)
    run_time = 1000
    # torch.cuda.synchronize()
    t_start = time.time()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=100),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/IO_Elastic_Linear_v2'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(run_time):
            re = e_linear.reference_forward(data, in_features, out_features)
            re.backward(tmp_grad)
            prof.step()
    # torch.cuda.synchronize()
    t_end = time.time()
    print('Dense per batch(ms):' , (t_end-t_start)*1000/run_time)
    
def dense_speed(e_linear, in_features, out_features):
    data = torch.rand(batch, seq_len, in_features).cuda()
    data.requires_grad_()
    re = e_linear.reference_forward(data, in_features, out_features)
    tmp_grad = torch.rand_like(re)
    run_time = 10000
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(run_time):
        re = e_linear.reference_forward(data, in_features, out_features)
        # re.backward(tmp_grad)
        # torch.sum(e_linear.ori_linear.weight.grad)
    torch.cuda.synchronize()
    t_end = time.time()
    print('Dense per batch(ms):' , (t_end-t_start)*1000/run_time)
    # import ipdb; ipdb.set_trace()

def dense_speed_v2(in_features, out_features):
    data = torch.rand(batch, seq_len, in_features).cuda()
    data.requires_grad_()
    dense_linear = torch.nn.Linear(in_features, out_features).cuda()
    re = dense_linear(data)
    tmp_grad = torch.rand_like(re)
    run_time = 10000
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(run_time):
        re = dense_linear(data)
        # re.backward(tmp_grad)
        # torch.sum(dense_linear.weight.grad)
    torch.cuda.synchronize()
    t_end = time.time()
    print('Dense_v2 per batch(ms):' , (t_end-t_start)*1000/run_time)

if __name__ == '__main__':
    batch = 1
    seq_len = 256
    hidden = 768
    # M = 64 *256 #batch . seq , head* hidden
    # K = 768
    # N = 768
    ori_linear = torch.nn.Linear(hidden, hidden, bias=True).cuda()
    elastic_linear = InOutElasticLinear(ori_linear)
    # only works on small size, WTF
    # test_correctness(elastic_linear, 1024, 512)
    in_feature = 512
    out_feature = 512
    # test_speed(elastic_linear, in_feature, out_feature)
    
    dense_speed(elastic_linear, in_feature, out_feature)
    dense_speed_v2(in_feature, out_feature)
    # dense_profile(elastic_linear, in_feature, out_feature)