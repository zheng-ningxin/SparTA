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
__global__ void convert_bcsr_kernel_fine_1(int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, float * values, int * extra_buffer)
{
    __shared__ int reduce[MAX_BLOCK_THREAD_COUNT];
    uint by = blockIdx.x; // row
    uint tid = threadIdx.x;
    int sum =0;
    int4 flag;
    for(int _pos = tid; _pos<w/4; _pos +=blockDim.x){
        flag = FETCH_INT4(mask[by * w + _pos*4]);
        sum += flag.x + flag.y + flag.z + flag.w;
    }
    reduce[tid] = sum;
    __syncthreads();
    // fast tree reduce accross the block
    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid<s)
            reduce[tid] += reduce[tid+s];
        __syncthreads();
    }
    if(tid<32)
        warpReduce(reduce, tid);
    __syncthreads();
    if(tid==0){
        extra_buffer[by] = reduce[0];
        extra_buffer[by+h] = reduce[0];
        atomicAdd(&row[h], reduce[0]);
    }

}
__global__ void convert_bcsr_kernel_fine_2(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, float * values, int * extra_buffer)
{
    uint tid = threadIdx.x;
    uint by = blockIdx.x;
    __shared__ int prefix_sum;
    if (tid==0){
        prefix_sum = 0;
        for(int i=0; i<by; i++)
            prefix_sum += extra_buffer[i];
        row[by] = prefix_sum;
    }
    __syncthreads();
    for(int _pos=tid; _pos<w; _pos+=blockDim.x){
        if(mask[by*w+_pos]>0){
            int tmp = atomicSub(&extra_buffer[by+h], 1);
            tmp-=1;
            col[prefix_sum+tmp] = _pos;
            values[prefix_sum+tmp] = dense[by*w+_pos];
        }
    }
}
void convert_csr_fine(int * mask, float* dense, int h, int w,
                        int * row, int * col,
                        float * values, int * extra_buffer)
{
    const int block_h =1, block_w=1;
    // assert(block_w==1);
    // assert(block_h==1);
    CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*(2*h+(h/block_h)*(w/block_w))) );
    CUDA_SAFE_CALL(cudaMemset((void*)row, 0, sizeof(int)*(1+(h/block_h))) );
    dim3 block_dim(256);
    dim3 grid_dim(h/block_h);
    convert_bcsr_kernel_fine_1<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, values, extra_buffer);
    convert_bcsr_kernel_fine_2<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, values, extra_buffer);
}

void convert_csr(int * mask, float * dense, int h, int w,
    int * row, int *col, float*values, int * extra_buffer)
{
    
    convert_csr_fine(mask, dense, h, w, row, col, values, extra_buffer);
}

std::vector<at::Tensor> convert_csr_forward(
    torch::Tensor sparse_pattern,
    torch::Tensor dense_values)
{
    int h = sparse_pattern.size(0);
    int w = sparse_pattern.size(1);
    // allocate enough memory for the sparse values
    torch::Tensor csr_values = torch::empty_like(dense_values);
    torch::Tensor csr_row = torch::zeros({h+1}, sparse_pattern.options());
    int n_total = h * w;
    torch::Tensor csr_col = torch::zeros({n_total}, sparse_pattern.options());
    torch::Tensor ext_buffer = torch::zeros({2*h+n_total}, sparse_pattern.options());
    AT_DISPATCH_FLOATING_TYPES(dense_values.type(), "convert_csr", ([&]
        { convert_csr(
                sparse_pattern.data_ptr<int>(),
                dense_values.data_ptr<float>(),
                h, w,
                csr_row.data_ptr<int>(),
                csr_col.data_ptr<int>(),
                csr_values.data_ptr<float>(),
                ext_buffer.data_ptr<int>()
            ); }));
    std::vector<torch::Tensor> csr({csr_row, csr_col, csr_values});
    return csr;
}
