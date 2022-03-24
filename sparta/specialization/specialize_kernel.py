import math
from sqlite3 import paramstyle
from pytest import param
from sparta.common.sparsity import TeSA, TesaAttr
from typing import Dict, List, Optional, Tuple

__all__ = ['specialize_matmul']

def specialize_matmul(in_tesa: tuple, weight_tesa: tuple, out_t_tesa: tuple):
    """
    Generate the kernels and profile the combined latency
    """
    # mocked for now
    assert(len(in_tesa) == 1 and len(out_t_tesa) == 1 and len(weight_tesa) >= 1)

    i_tesa, o_tesa = in_tesa[0], out_t_tesa[0]
    
    latency = 0
    kernels = []
    # for matmul, aggr_type should be add
    aggr_type = 'add'
    for idx, w_tesa in enumerate(weight_tesa):
        if idx == 0:
            kernel, params = matmul_kernel_init(i_tesa, w_tesa, o_tesa)
        else:
            kernel, params = matmul_kernel_init(i_tesa, w_tesa, o_tesa, bias=True)
        kernel, exec_time = matmul_kernel_tune(kernel, params)
        kernels.append(kernel)
        latency += exec_time

    return latency, kernels, aggr_type

def matmul_kernel_codegen(w_tesa, shape: List[int], dismantle: int, block_size: tuple, n_bits: int, bias: bool) -> Tuple[str, Dict[str, List[int]]]:
    """
    Choose kernel template according to dismantle primitive
    """
    if dismantle == -1:
        kernel, params = dense_matmul_template(shape, n_bits, bias)
    elif dismantle == 8:
        kernel, params = finegrained_matmul_template(w_tesa, shape, n_bits, bias)
    elif dismantle == 2:
        kernel, params = blocksparse_matmul_template(shape, n_bits, block_size, bias)
    return kernel, params

def matmul_kernel_init(i_tesa: TeSA, w_tesa: TeSA, o_tesa: TeSA, bias=False) -> Tuple[str, Dict[str, List[int]]]:
    """
    Generate basic kernel code based on matmul shape, dismantle primitive, block size and bits.
    i, j, k -> i0, j0, k0, k1, i1, j1, k2, i2, j2
    dense: dismantle = -1, it will not be applied to any loop.
    finegrained sparse: dismantle = 8, it will be applied to the inner most loop.
    block sparse: dismantle = 2, it will be applied to k0.

    """
    matmul_shape = get_matmul_shape(i_tesa, w_tesa, o_tesa)
    block_size = w_tesa.block_size
    n_bits = w_tesa.n_bits
    if block_size == None:
        dismantle = -1
    elif block_size == 1:
        # not apply dismantle, kernel execute in dense way.
        dismantle = 8
    else:
        dismantle = 2

    kernel, params = matmul_kernel_codegen(w_tesa, matmul_shape, dismantle, block_size, n_bits, bias)
    return kernel, params

def matmul_kernel_tune(kernel, params):
    """
    Kernel tuning process
    """
    search_space = generate_grid_search_space(params)
    least_exec_time = math.inf
    best_kernel = None
    for param_dict in search_space:
        kernel, exec_time = kernel_execution(kernel, param_dict)
        if exec_time < least_exec_time:
            best_kernel = kernel
            least_exec_time = exec_time
    return best_kernel, least_exec_time

def dense_matmul_template(shape, n_bits, bias):
    assert(n_bits == 8 or n_bits == 32, "only support two bit types currently")
    assert(len(shape) == 3, "shape should contain m, k, n")
    m, k, n = shape[0], shape[1], shape[2]
    if n_bits == 8:
        template_name = "quantize_dot_template_bias.cu"
        f = open(template_name)
        kernel = f.read()
        sub_param = {"M_GLOBAL_VALUE": m, "K_GLOBAL_VALUE": k, "N_GLOBAL_VALUE": n}
        tuning_param = {"CHUNK_K_VALUE": [1, 2, 4, 6, 8], "BLOCK_ROW_WARPS": [1, 2, 3, 4], "BLOCK_COL_WARPS": [1, 2, 3, 4], "WARP_ROW_TILES": [1, 2, 3, 4], "WARP_COL_TILES": [1, 2, 3, 4]}
        for key, value in sub_param.items():
            kernel = kernel.replace(key, str(value))
    else:
        # n_bits == 32, should use cublas directly
        pass
    return kernel, tuning_param

def finegrained_matmul_template(w_tesa, shape, n_bits, bias) -> Tuple[str, Dict[str, List[int]]]:
    """
    Parameters
    ----------
    w_tesa
        TeSA of weight
    shape
        List, contains [m, k, n]
    n_bits
        bit width of weight, Choice[8, 32]
    bias
        bool, true represent using add fusion, false represent using non-fusion
    
    Returns
    -------
    kernel: str
        kernel string, should have embedded m, k, n into itself
    tuning_param: dict
        dict of tuning parameters

    Return Parameters Usage Example
    -------
        kernel_tmp = kernel
        tuning_param = {"BLOCK_SIZE_M": [32, 64, 128], "BLOCK_SIZE_N": [32, 64, 128]}
        search_space = grid_search_space_gen(tuning_param)

        best_kernel = ""
        least_latency = math.inf

        for case in search_space:
            for key, val in case.items():
                kernel_tmp = kernel.replace(key, val)
            latency = exec_kernel(kernel_tmp)
            if latency < least_latency:
                least_latency = latency
                best_kernel = kernel_tmp
    """
    kernel = ""
    tuning_param = {}
    return kernel, tuning_param

def blocksparse_matmul_template(shape, n_bits, block_size, bias):
    assert(n_bits == 8 or n_bits == 32, "only support two bit types currently")
    assert(len(shape) == 3, "shape should contain m, k, n")

    m, k, n = shape[0], shape[1], shape[2]
    # block_size should in formation [block_size_k, block_size_n]
    block_size_k = block_size[0]
    block_size_n = block_size[1]

    if n_bits == 32:
        template_name = "block_sparse_template_bias_row.cu"
        f = open(template_name)
        kernel = f.read()
        sub_param = {"M_VALUE": m , "K_VALUE": k, "N_VALUE": n, \
            "BLOCK_SIZE_K_VALUE": block_size_k, "BLOCK_SIZE_N_VALUE": block_size_n}
        tuning_param = {"BLOCK_SIZE_M_sub": [32, 64, 128], "THREAD_SIZE_M_sub": [2, 4, 8, 16],\
            "THREAD_SIZE_K_sub": [1, 2, 4, 8], "THREAD_SIZE_N_sub": [2, 4, 8, 16]}
        for key, value in sub_param.items():
            kernel = kernel.replace(key, str(value))
    else:
        template_name = "block_quantize_template_bias.cu"
        f = open(template_name)
        kernel = f.read()
        chunk_k = block_size_k / 16
        warp_row_tiles = 1 if block_size_n <= 16 else 2
        block_row_warps = (block_size_n / 16) / warp_row_tiles
        
        sub_param = sub_param = {"M_GLOBAL_VALUE": m, "K_GLOBAL_VALUE": k, "N_GLOBAL_VALUE": n, \
            "CHUNK_K_VALUE": chunk_k, "BLOCK_ROW_WARPS_VALUE": block_row_warps, \
                "WARP_ROW_TILES_VALUE": warp_row_tiles}
        tuning_param = {"BLOCK_COL_WARPS": [1, 2, 3, 4], "WARP_COL_TILES": [1, 2, 3, 4]}
        for key, value in sub_param.items():
            kernel = kernel.replace(key, str(value))
    
    return kernel, tuning_param