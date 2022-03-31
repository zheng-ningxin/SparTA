
from mobilenet_utils import *

def get_mobile_coarse():
    import torch
    from nni.compression.pytorch.speedup import ModelSpeedup
    model = create_model('mobilenet_v1')
    dummy_input = torch.rand(1,3,224,224)
    new_model = align_speedup(model, dummy_input, 'checkpoints/coarsegrained/mobilenet_0.6_align_run1/mask_temp.pth')
    state = torch.load('checkpoints/coarsegrained/mobilenet_0.6_align_run1/finetune_weights.pth')
    new_model.load_state_dict(state)
    return new_model

m = get_mobile_coarse()
device = torch.device('cuda')
test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
import ipdb; ipdb.set_trace()
print(run_eval(m.cuda(), test_dataloader, device))
print('Propagation done')
