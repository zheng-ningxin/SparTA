# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import logging
import torch.nn as nn
from typing import Dict, List
from sparta.common.utils import convert_bcsr
from sparta.codegen.emitter import EmitterBase
from nni.common.graph_utils import build_module_graph
__all__ = ['TeSA', 'SparseModuleInfo', 'ModelSparsityInfo', 'BcsrSparseLayout']

_logger = logging.Logger(__name__)
_logger.setLevel(logging.INFO)

# TODO: more elegant definition for TesaAttr
TesaAttr = {-1: 'constant',
            0: 'pruned',
            4: 'int4',
            7: 'uint8',
            8: 'int8',
            16: 'float16',
            31: 'int32',
            32: 'float32',
            33: 'nonpruned'}


class SparseLayout:
    def __init__(self):
        pass

    def _build_index(self):
        raise NotImplementedError


class BcsrSparseLayout(SparseLayout):
    def __init__(self, block_size, dense_mask):
        self.block_size = block_size
        self.dense_mask = dense_mask
        self._build_index()

    def _build_index(self):
        self.block_row_idx, self.block_col_idx, self.fine_graind_mask = convert_bcsr(
            self.dense_mask, self.dense_mask)


class TeSA:
    """
    Tensor with sparse attribute.
    """

    def __init__(self, tesaattr_tensor: torch.Tensor):
        self.tesa: torch.Tensor = tesaattr_tensor
        # NOTE: can be refactored here to support balanced sparsity pattern
        self.block_size: tuple = None
        # number of different bits in this tesa
        self.n_bits: int = None

    def set_transform_meta(self, block_size: tuple, n_bits: int):
        # this meta information is for guiding kernel specialization
        self.block_size = block_size
        self.n_bits = n_bits


class SparseModuleInfo:
    """
    Attributes
    ----------
    ...
    """

    def __init__(self, module_name: str,
                 module_obj: nn.Module,
                 dummy_input: List[torch.Tensor] = None,
                 weight_tesa: Dict[str, torch.Tensor] = None,
                 input_tesa: List[torch.Tensor] = None,
                 output_tesa: List[torch.Tensor] = None):
        self.module_name = module_name
        self.module_obj = module_obj
        self.dummy_input = dummy_input
        self.weight_tesa = {name: TeSA(
            t) for name, t in weight_tesa.items()} if weight_tesa is not None else {}
        self.input_tesa = [
            TeSA(t) for t in input_tesa] if input_tesa is not None else []
        self.output_tesa = [
            TeSA(t) for t in output_tesa] if output_tesa is not None else []
        self.graph = None
        if self.dummy_input is not None:
            self.construct_graph(self.module_obj, self.dummy_input)

    def construct_graph(self, module: torch.nn.Module, dummy_input):
        _logger.info('Constructing the graph')
        self.graph = build_module_graph(module, dummy_input)


class Task:
    def __init__(self, task_name):
        self.task_name = task_name
    
    def execute_command(self):
        raise NotImplementedError
    
    def get_result(self):
        raise NotImplemented


class KernelTuningTask(Task):
    def __init__(self, task_name:str, code:str, fpath:str, arch_gen:str):
        super(KernelTuningTask, self).__init__(task_name)
        self.code = code
        self.fpath = fpath
        self.arch_gen = arch_gen

    def execute_command(self):
        with open(self.fpath, 'w') as f:
            f.write(self.code)
        os.system(f'nvcc {self.fpath} ')
                
    def get_result(self):
        pass
    
class TemplateTuningTaskPool:
    def __init__(self, template_path:str, emitter:EmitterBase):
        self.template_path = template_path
        self.emitter = emitter
        self.task_pools = self.construct_task_pools()
    
    def construct_task_pools(self):
        pass

    def fetch_one_task(self):
        pass
        