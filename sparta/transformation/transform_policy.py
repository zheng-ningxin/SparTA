import abc
import torch.nn as nn
from itertools import product
from typing import List
from sparta.common import SparseModuleInfo
from sparta.specialization import specialize_matmul

class TransformedModule:
    def __init__(self, module_info, kernels, aggregate_type=None):
        self.module_info: SparseModuleInfo = module_info
        self.kernels: list = kernels
        self.aggregate_type: str = aggregate_type

class TransformPolicyBase(abc.ABC):
    @abc.abstractmethod
    def transform_module(self, module_sparsity: SparseModuleInfo):
        pass

class TransformPolicy(TransformPolicyBase):
    def __init__(self, device_info: str):
        self.device_info = device_info

    def transform_module(self, module_sparsity: SparseModuleInfo):
        """
        Enumerate possible tensor decomposition options.
        In the current implementation, we do not decompose activation tensor into
        multiple tensors. Only do this for weight tensors.

        Returns
        -------
        TransformedModule
        """
        kernels = None
        if isinstance(module_sparsity.module_obj, nn.Linear):
            kernels, aggr_type = self.transform_matmul(module_sparsity.input_tesa,
                                            module_sparsity.weight_tesa,
                                            module_sparsity.output_tesa)
        elif isinstance(module_sparsity.module_obj, nn.Conv2d):
            ...
        # support our sparse attention
        #elif isinstance(module_sparsity.module_obj, SparseAttention):
        #    ...
        else:
            ...
        return TransformedModule(module_sparsity, kernels, aggr_type)
    
    def transform_tesa(self, tesa, n_candidates: int = 1, decompose: bool = True) -> List[tuple]:
        """
        Sparsity pattern matcher here for decomposing/splitting and covering

        Parameters
        ----------
        tesa : torch.Tensor
            the sparsity attribute
        n_candidates : int
            the number of generated transformation options
        decompose : bool
            whether decompose a tensor into multiple tensors

        Returns
        -------
        list[tuple]
            a list of candidates, each candidate is a tuple of sub-tensors
        """
        # mocked
        return [(tesa)]

    def transform_matmul(self, in_tesa, weight_tesa, out_tesa):
        in_tesas = self.transform_tesa(in_tesa, decompose=False)
        weight_tesas = self.transform_tesa(weight_tesa)
        out_tesas = self.transform_tesa(out_tesa, decompose=False)
        best_latency = float('inf')
        best_kernels = None
        best_aggr_type = None
        for in_t, weight_t, out_t in product(in_tesas, weight_tesas, out_tesas):
            latency, kernels, aggr_type = specialize_matmul(in_t, weight_t, out_t)
            if latency < best_latency:
                best_latency = latency
                best_kernels = kernels
                best_aggr_type = aggr_type
        return best_kernels, best_aggr_type

    def transform_conv(self, in_tesa, weight_tesa, out_tesa):
        ...
