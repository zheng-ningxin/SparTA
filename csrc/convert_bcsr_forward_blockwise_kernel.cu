#include "common.h"
#include "cusparse.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

#include "common.h"

using namespace std;

#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024

#define CUBLAS_SAFE_CALL(func)                                                                  \
    do                                                                                          \
    {                                                                                           \
        cublasStatus_t e = (func);                                                              \
        if (e != CUBLAS_STATUS_SUCCESS)                                                         \
        {                                                                                       \
            std::stringstream safe_call_ss;                                                     \
            safe_call_ss << "\nerror: " #func " failed with error"                              \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e; \
            throw std::runtime_error(safe_call_ss.str());                                       \
        }                                                                                       \
    } while (0)

#define CUDA_SAFE_CALL(x)                                                                         \
    do                                                                                            \
    {                                                                                             \
        cudaError_t result = (x);                                                                 \
        if (result != cudaSuccess)                                                                \
        {                                                                                         \
            const char *msg = cudaGetErrorString(result);                                         \
            std::stringstream safe_call_ss;                                                       \
            safe_call_ss << "\nerror: " #x " failed with error"                                   \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
            throw std::runtime_error(safe_call_ss.str());                                         \
        }                                                                                         \
    } while (0)

__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid + 8]; 
    sdata[tid] += sdata[tid + 4]; 
    sdata[tid] += sdata[tid + 2]; 
    sdata[tid] += sdata[tid + 1]; 
}

__device__ __forceinline__ const int* add_ptr_u(const int* src, int offset)      \
{                                                                            \
    const int* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}

__device__ __forceinline__ const float* add_ptr_f(const float* src, int offset)      \
{                                                                            \
    const float* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}



__global__ void convert_bcsr_kernel_transpose_1(const int * __restrict__  mask, int* __restrict__ row_ptr, int* __restrict__ col_idx, int * extra_buffer, int n_block_h, int n_block_w)
{

    
    // initial the shared flag
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;
    uint step = blockDim.x;

    int pos_id;
    int flag;
    int global_offset;
    for(int by=tid; by<n_block_h;by+=step){
        global_offset =  (by * n_block_w) + bx;
        flag = mask[global_offset];
        if(flag>0){
            pos_id= atomicAdd(&extra_buffer[bx], 1);
            atomicAdd(&extra_buffer[bx+n_block_w], 1);
            atomicAdd(&row_ptr[n_block_w], 1);
            extra_buffer[2*n_block_w + n_block_h * bx + pos_id] = by;
        }
    
    }


}
__global__ void convert_bcsr_kernel_transpose_2(const int * __restrict__  mask, int* __restrict__ row_ptr, int* __restrict__ col_idx, int * extra_buffer, int n_block_h, int n_block_w)
{
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;
    int pos_id, ori_bx, ori_by;
    __shared__ int prefix_count;
    __shared__ int remain_count;
    if(tid==0){
        prefix_count = 0;
        for(int i=0; i<bx;i++){
            prefix_count +=  extra_buffer[n_block_w+i];
        }
        remain_count = extra_buffer[n_block_w+bx];
        row_ptr[bx] = prefix_count;
    }
    __syncthreads();
    for(int tmp=tid; tmp<remain_count;tmp+=blockDim.x){
        pos_id = atomicSub(&extra_buffer[bx], 1);
        pos_id-=1;
        if(pos_id>=0){
            ori_bx = bx;
            ori_by = extra_buffer[ 2*n_block_w + bx * n_block_h + pos_id];       
            row_ptr[bx] = prefix_count;
            col_idx[prefix_count + pos_id] = ori_by;
        }
    }


}


__global__ void convert_bcsr_kernel_1(const int * __restrict__  mask, int* __restrict__ row_ptr, int* __restrict__ col_idx, int * extra_buffer, int n_block_h, int n_block_w)
{

    
    // initial the shared flag
    uint by = blockIdx.x;
    uint tid = threadIdx.x;
    uint step = blockDim.x;

    int pos_id;
    int flag;
    int global_offset;
    for(int bx=tid; bx<n_block_w;bx+=step){
        global_offset =  (by * n_block_w) + bx;
        flag = mask[global_offset];
        if(flag>0){
            pos_id= atomicAdd(&extra_buffer[by], 1);
            atomicAdd(&extra_buffer[by+n_block_h], 1);
            atomicAdd(&row_ptr[n_block_h], 1);
            extra_buffer[2*n_block_h + n_block_w * by + pos_id] = bx;
        }
    
    }


}
__global__ void convert_bcsr_kernel_2(const int * __restrict__  mask, int* __restrict__ row_ptr, int* __restrict__ col_idx, int * extra_buffer, int n_block_h, int n_block_w)
{
    uint by = blockIdx.x;
    uint tid = threadIdx.x;
    int pos_id, ori_bx, ori_by;
    __shared__ int prefix_count;
    __shared__ int remain_count;
    if(tid==0){
        prefix_count = 0;
        for(int i=0; i<by;i++){
            prefix_count +=  extra_buffer[n_block_h+i];
        }
        remain_count = extra_buffer[n_block_h+by];
        row_ptr[by] = prefix_count;
    }
    __syncthreads();
    for(int tmp=tid; tmp<remain_count;tmp+=blockDim.x){
        pos_id = atomicSub(&extra_buffer[by], 1);
        pos_id-=1;
        if(pos_id>=0){
            ori_by = by;
            ori_bx = extra_buffer[ 2*n_block_h + by * n_block_w + pos_id];       
            row_ptr[by] = prefix_count;
            col_idx[prefix_count + pos_id] = ori_bx;
        }
    }


}
void convert_bcsr_transpose(int * mask, int * row_ptr, int * col_idx, int * ext_buffer, 
                            int n_block_h, int n_block_w)
{
    // the mask is a binary matrix with shape of n_block_h x n_block_w
    // build the csr index along the n_block_w
    // need reset the extra buffer here
    CUDA_SAFE_CALL(cudaMemset((void*)ext_buffer, 0, sizeof(int)*(2*n_block_w+n_block_h*n_block_w)) );
    CUDA_SAFE_CALL(cudaMemset((void*)row_ptr, 0, sizeof(int)*(1+(n_block_w))) );
    dim3 block_dim(128);
    dim3 grid_dim(n_block_w);

    convert_bcsr_kernel_transpose_1<<<grid_dim, block_dim>>>(mask, row_ptr, col_idx, ext_buffer, n_block_h, n_block_w);
    convert_bcsr_kernel_transpose_2<<<grid_dim, block_dim>>>(mask, row_ptr, col_idx, ext_buffer, n_block_h, n_block_w);
}


void convert_bcsr(int * mask, int * row_ptr, int * col_idx, int * ext_buffer, 
                            int n_block_h, int n_block_w)
{

    CUDA_SAFE_CALL(cudaMemset((void*)ext_buffer, 0, sizeof(int)*(2*n_block_h+n_block_h*n_block_w)) );
    CUDA_SAFE_CALL(cudaMemset((void*)row_ptr, 0, sizeof(int)*(1+(n_block_h))) );
    dim3 block_dim(128);
    dim3 grid_dim(n_block_h);

    convert_bcsr_kernel_1<<<grid_dim, block_dim>>>(mask, row_ptr, col_idx, ext_buffer, n_block_h, n_block_w);
    convert_bcsr_kernel_2<<<grid_dim, block_dim>>>(mask, row_ptr, col_idx, ext_buffer, n_block_h, n_block_w);
}

void convert_bcsr_blockwise(int * mask, int *row_ptr, int * col_idx, int* ext_buffer, int n_block_h, int n_block_w, int transpose)
{
    if(transpose){
        convert_bcsr_transpose(mask, row_ptr, col_idx, ext_buffer, n_block_h, n_block_w);
    } else{
        // TO BE DONE
        convert_bcsr(mask, row_ptr, col_idx, ext_buffer, n_block_h, n_block_w);
    }

}
std::vector<at::Tensor> convert_bcsr_forward_blockwise(
    torch::Tensor sparse_pattern,
    int transpose)
{
    int n_block_h = sparse_pattern.size(0);
    int n_block_w = sparse_pattern.size(1);
    int row_size = transpose? n_block_w: n_block_h;
    int n_total_block = n_block_h * n_block_w;
    torch::Tensor csr_row = torch::zeros({row_size+1}, sparse_pattern.options());
    torch::Tensor csr_col = torch::zeros({n_total_block}, sparse_pattern.options());
    torch::Tensor ext_buffer = torch::zeros({2*row_size+n_total_block}, sparse_pattern.options());
    AT_DISPATCH_INTEGRAL_TYPES(sparse_pattern.type(), "convert_bcsr_blockwise", ([&]
        { convert_bcsr_blockwise(
                sparse_pattern.data_ptr<int>(),
                csr_row.data_ptr<int>(),
                csr_col.data_ptr<int>(),
                ext_buffer.data_ptr<int>(),
                n_block_h,
                n_block_w,
                transpose
            ); }));
    std::vector<torch::Tensor> bcsr({csr_row, csr_col});
    return bcsr;
}
