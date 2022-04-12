#include <sstream>
#include <stdexcept>
#include <cudnn.h>
#include <cublas_v2.h>
#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include <cusparse.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <fstream>
#include "nnfusion_rt.h"
#include <limits>
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
extern "C" int kernel_entry(int m, int k, int n, int nnz, int*d_row_swizzle, float*d_values, int*d_row_idx, int*d_col_idx, float* dB, float * dC, int beta){
    CHECK_CUDA(sputnik::CudaSpmm(m, k, n, nnz, d_row_swizzle, d_values, d_row_idx, d_col_idx, dB, dC, 0);)
}