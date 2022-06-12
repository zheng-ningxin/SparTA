# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from random import shuffle
import shutil
from typing import Dict, List
from sparta.common.utils import cuda_detect
from sparta.common.abstract import SparseModuleInfo, KernelTask

class CodeGenEngine:
    def __init__(self, tmp_dir='/tmp'):
        assert os.path.exists(tmp_dir)
        self.gpu_devices = None
        self.init_measurement_env()

    def init_measurement_env(self):
        """
        Initialize the measurement envrionment of code generation:
        including check if there are available devices, and the architecture
        of the corresponding devices.
        """
        self.gpu_devices = cuda_detect()       
        errmsg = "Cannot find NVCC, please set the CUDA environment correctly" 
        assert shutil.which('nvcc') is not None, errmsg
    
    def parse_tasks(self, sparse_module:SparseModuleInfo) -> List[KernelTask]:
        pass
    
    def generate_code(self, sparse_module: SparseModuleInfo):
        tasks = self.parse_tasks(sparse_module)
        