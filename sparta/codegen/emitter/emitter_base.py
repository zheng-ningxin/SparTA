# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import abc
import copy
import logging
from abc import abstractmethod
from typing import Dict, List

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

class EmitterBase(abc.ABC):

    @abstractmethod        
    def emit_function_call():
        """
        Emit the funtion call
        """

    @abstractmethod
    def emit_function_body():
        """
        Emit the body of the function
        """

    @abstractmethod
    def emit_dependency():
        """
        Emit the dependent headers
        """


    @abstractmethod
    def emit_test_main():
        """
        Emit the main function used to test the speedup/memory footprint
        """ 

class TunningKernelEmitter(EmitterBase):
    @abstractmethod
    def tunning_kernel_cfg(self, *args, **kwargs):
        """
        Tuning the kernel within the specified search space.
        """
    
    @abstractmethod
    def measure_trail_latency(self, *args, **kwargs):
        """
        Measure the latency of a specific kernel of the template.
        """

    @abstractmethod
    def init_search_space(self, *args, **kwargs):
        """
        Perform the initialization for the search space.
        """


class GridSearchEmitter(TunningKernelEmitter):

    def _dfs_space(self, cfgid, search_space:List):
        if cfgid == len(self.space):
            search_space.append(copy.deepcopy(self.choices))
            return
        cur_key = self.keys[cfgid]
        for value in self.space[cur_key]:
            self.choices[cur_key] = value
        self._dfs_space(cfgid+1, search_space)

    def generate_all_cfgs(self):
        all_cfgs = []
        self._dfs_space(0, all_cfgs)
        return all_cfgs

    def init_search_space(self, space:Dict[str, list]):
        self.keys = list(space.keys())
        self.choices = {}
        self.space = space

    def tunning_kernel_cfg(self, search_space: Dict[str, list]):
        self.init_search_space(search_space)
        cfg_space = self.generate_all_cfgs()
        best_cfg = None
        best_latency = float('inf')
        for _cfg in cfg_space:
            try:
                latency = self.measure_trail_latency(_cfg)
                if latency < best_latency:
                    best_cfg = _cfg
                    best_latency = latency
            except Exception as exp:
                _logger.info('Config %s failed with exception: %s', str(_cfg), str(exp))
        return best_cfg
        