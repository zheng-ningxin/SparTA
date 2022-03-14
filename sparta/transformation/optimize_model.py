import torch
import torch.nn as nn

from sparta.common import SparseModuleInfo, ModelSparsityInfo
from sparta.specialization import specialize_kernel
from sparta.transformation.transform_policy import TransformPolicy

__all__ = ['optimize_and_rebuild']

def rebuild_sparse_model_pytorch(model: nn.Module, opt_modules: dict):
    """
    Important: users can directly use our optimized module,
    and also they can use the meta/wrapper module, which will be replace
    with the optimized module here.
    """
    # first support pytorch
    # return mocked for now
    return model

def rebuild_sparse_model_nnfusion(model: nn.Module, opt_modules: dict):
    ...

def optimize_and_rebuild(model: nn.Module,
                         post_sparsity: ModelSparsityInfo,
                         backend = 'pytorch',
                         device_info = None):
    opt_modules = {}
    # init a transformation policy
    tpolicy = TransformPolicy(device_info)
    for module_name, module_sparsity in post_sparsity.items():
        transformed = tpolicy.transform_module(module_sparsity)
        opt_modules[module_name] = transformed
    # rebuild the sparse model: module replacement or nnfusion
    if backend == 'pytorch':
        opt_model = rebuild_sparse_model_pytorch(model, opt_modules)
    elif backend == 'nnfusion':
        opt_model = rebuild_sparse_model_nnfusion(model, opt_modules)
    else:
        raise
    return opt_model