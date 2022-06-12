# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import re
import json
import copy
import subprocess
from sparta.common.utils import cuda_detect
from .emitter_base import GridSearchEmitter

class GPUBlockSparseEmitter(GridSearchEmitter):
    def __init__(self, M:int, K:int, N:int, tmp_dir:str):
        dir_path = os.path.join(os.path.split(__file__)[0], '../template')
        cfg_path = os.path.join(dir_path, 'block_sparse_linear.json')
        assert os.path.exists(cfg_path)
        with open(cfg_path, 'r') as f:
            self.template_cfg = json.load(f)   
        code_path = os.path.join(dir_path, self.template_cfg['code'])
        assert os.path.exists(code_path)
        with open(code_path, 'r') as f:
            code_template = f.read()
        self.space = self.template_cfg["space"] if "space" in self.template_cfg else None
        self.function_body, self.function_call = self.parse_code_template(code_template)
        self.tmp_dir = tmp_dir # directory used to compile and test the program
        kv = {"GLOBAL_M_VALUE":M, "GLOBAL_K_VALUE":K, "GLOBAL_N_VALUE":N}
        for key, value in kv.items():
            self.function_body =  self.function_body.replace(key, str(value))
        self.gpu_devices = cuda_detect()
        assert len(self.gpu_devices) > 0
        self.gpu_code = self.gpu_devices[0][1]
    
    def parse_code_template(self, code_template:str):
        function_body = code_template
        function_call = code_template.split('{')[0]
        return function_body, function_call
        
    def emit_function_body(self, stringstream, kw_args=None):
        function_body = copy.deepcopy(self.function_body)        
        for key, value in kw_args.items():
            function_body = function_body.replace(key, str(value))
        stringstream.write(function_body)
    
    def emit_function_call(self, stringstream, kw_args=None):
        funciton_call = copy.deepcopy(self.function_call)
        for key, value in kw_args.items():
            function_call = function_call.replace(key, value)
        stringstream.write(funciton_call)
    
    def emit_dependency(self, stringstream):
        header = """
        #include <cuda.h>
        #include <stdio.h>
        """
        stringstream.write(header)

    def emit_test_main(self, stringstream):
        # self.emit_dependency()
        main_body = """
        int main()
        {
            printf("0.5");
            return 0;
        }
        """
        stringstream.write(main_body)
        
    def measure_trail_latency(self, cfg):
        kernel_f_path = os.path.join(self.tmp_dir, 'block_sparse_kernel.cu')
        exec_path = os.path.join(self.tmp_dir, 'block_sparse')
        log_path = os.path.join(self.tmp_dir, 'block_sparse_run.log')
        kernel_f = open(kernel_f_path, 'w')
        self.emit_dependency(kernel_f)
        self.emit_function_body(kernel_f, cfg)
        self.emit_test_main(kernel_f)
        kernel_f.close()
        latency = float('inf')
        import pdb; pdb.set_trace()
        
        subprocess.run(f"nvcc -gencode arch=compute_{self.gpu_code},code=sm_{self.gpu_code} {kernel_f_path} -o {exec_path}", shell=True, check=True)
        subprocess.run(f"{exec_path} > {log_path}", shell=True, check=True)
        with open(log_path, 'r') as f:
            latency = float(f.readline())
        return latency