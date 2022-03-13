import abc
from sparta.common import SparseModuleInfo

class TransformedModule:
    def __init__(self, module_info, kernels, aggregate_op=None):
        self.module_info: SparseModuleInfo = module_info
        self.kernels: list = kernels
        self.aggregate_op: str = aggregate_op

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
        if module_sparsity.module_type == 'Linear':
            kernels = self.transform_matmul(module_sparsity.input_tesa,
                                            module_sparsity.weight_tesa,
                                            module_sparsity.output_tesa)
        elif module_sparsity.module_type == 'Conv2d':
            ...
        elif module_sparsity.module_type == 'SparseAttention':
            ...
        else:
            ...
        return TransformedModule(module_sparsity, kernels)
    
    def transform_tesa(self, tesa, decompose: bool = True):
        ...

    def transform_matmul(self, in_tesa, weight_tesa, out_tesa):
        ...

    def transform_conv(self, in_tesa, weight_tesa, out_tesa):
        ...
