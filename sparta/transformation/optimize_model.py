import torch
import torch.nn as nn

from sparta.propagation import extract_sparsity
from sparta.common import ModelSparsityInfo
from sparta.specialization import specialize_kernel
from sparta.transformation.transform_policy import TransformPolicy

__all__ = ['optimize_and_rebuild']

class TransformedModule:
    def __init__(self, module_name, kernels, aggregate_op):
        self.module_name: str = module_name
        self.kernels: list = kernels
        self.aggregate_op: str = aggregate_op

def transform_module(module_sparsity: ModelSparsityInfo, tpolicy: TransformPolicy):
    """
    Enumerate possible tensor decomposition options

    Returns
    -------
    TransformedModule
    """
    ...
    kernel = specialize_kernel()
    ...

def rebuild_sparse_model_pytorch(model: nn.Module, opt_modules: dict):
    # first support pytorch
    ...

def rebuild_sparse_model_nnfusion(model: nn.Module, opt_modules: dict):
    ...

def optimize_and_rebuild(model: nn.Module, backend = 'pytorch', device_info = None):
    post_sparsity = extract_sparsity(model)
    opt_modules = {}
    # init a transformation policy
    tpolicy = TransformPolicy()
    for module_name, module_sparsity in post_sparsity.items():
        transformed = transform_module(module_sparsity, tpolicy)
        opt_modules[module_name] = transformed
    # rebuild the sparse model: module replacement or nnfusion
    if backend == 'pytorch':
        opt_model = rebuild_sparse_model_pytorch(model, opt_modules)
    elif backend == 'nnfusion':
        opt_model = rebuild_sparse_model_nnfusion(model, opt_modules)
    else:
        raise
    return opt_model