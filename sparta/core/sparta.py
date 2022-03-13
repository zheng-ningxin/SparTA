import torch
import torch.nn as nn

from sparta.propagation import propagate_sparsity
from sparta.transformation import optimize_and_rebuild

class SpartaModel(nn.Module):
    def __init__(self, model_cls: nn.Module, *args, **kwargs):
        self.model_cls = model_cls
        self.model_args = args
        self.model_kwargs = kwargs
        self.model = model_cls(*args, **kwargs)
        self.opt_modules = None
        self.opt_model = None
        # optimize the model based on its sparsity
        self.opt_modules = self._optimize_sparsity()

    def _optimize_sparsity(self):
        # sparsity attributes are inplace updated in self.model
        propagate_sparsity(self.model)
        # transformation, specialization, and rebuild model
        opt_model = optimize_and_rebuild(self.model, backend='pytorch', device_info=None)
        return opt_model

    def forward(self, *inputs):
        return self.opt_model(*inputs)