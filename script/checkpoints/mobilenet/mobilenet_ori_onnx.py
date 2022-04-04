from torch import device
from mobilenet_utils import *
import torch


device = torch.device('cuda')

model = create_model('mobilenet_v1').to(device)
data = torch.rand(32, 3, 224, 224).to(device)
torch.onnx.export(model, data, 'mobilenet_ori_no_tesa.onnx')