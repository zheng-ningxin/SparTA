import torch
import sparta
from sparta.opset.sparse_moe import DynamicSparseMoE


def calculate_ref(data, exps, exp_ids, out_hidden):
    n_exp = len(exps)
    out = torch.zeros(data.size(0), out_hidden).to(data.device)
    for eid in range(n_exp):
        out[exp_ids == eid]=exps[eid](data[exp_ids == eid])
    return out

if __name__ == '__main__':
    B = 32
    S = 128
    N_exp = 16
    in_hidden = 768
    out_hidden = 3072
    exps = []
    for i in range(N_exp):
        exps.append(torch.nn.Linear(in_hidden, out_hidden, bias=False).cuda())
    # for i in range(N_exp):
    #     exps[i].weight.data[:]=1
    moe = DynamicSparseMoE(N_exp, exps)
    data = torch.rand(B*S, in_hidden).cuda()
    exp_ids = torch.randint(0, N_exp, (B*S,)).to(torch.int32).cuda()
    # import ipdb; ipdb.set_trace()
    out = moe(data, exp_ids)
    ref_out =  calculate_ref(data, exps, exp_ids, out_hidden)
    import ipdb; ipdb.set_trace()
    
    assert torch.allclose(out, ref_out, rtol=1e-08, atol=1e-04)
    # pass