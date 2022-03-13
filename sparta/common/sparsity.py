import torch

__all__ = ['SparseModuleInfo', 'ModelSparsityInfo']

TesaAttr = {-1: 'constant',
            0: 'pruned',
            7: 'uint8',
            8: 'int8',
            31: 'int32',
            32: 'float32',
            33: 'nonpruned'}

class SparseModuleInfo:
    """
    Attributes
    ----------
    ...
    """
    def __init__(self, module_name: str,
                 module_type: str,
                 weight_tesa: torch.Tensor,
                 input_tesa: torch.Tensor,
                 output_tesa: torch.Tensor):
        self.module_name = module_name
        self.module_type = module_type
        self.weight_tesa = weight_tesa
        self.input_tesa = input_tesa
        self.output_tesa = output_tesa

class ModelSparsityInfo:
    """
    Attributes
    ----------
    ...
    """
    def __init__(self):
        self.modules_info: dict = {}

    def update(self, info: SparseModuleInfo):
        if info.module_name in self.modules_info:
            # merge sparsity
            ...
        else:
            self.modules_info[info.module_name] = info

class ModelDataLayouts:
    def __init__(self):
        ...