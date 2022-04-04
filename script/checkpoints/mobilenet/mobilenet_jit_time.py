from torch import device
from mobilenet_utils import *
import torch
from sparta.common.utils import measure_time

device = torch.device('cuda')

model = create_model('mobilenet_v1').to(device)
data = torch.rand(32, 3, 224, 224).to(device)
jit_model = torch.jit.trace(model, data)
time_mean, time_std = measure_time(jit_model, [data])
print(time_mean)