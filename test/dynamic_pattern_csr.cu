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


#include "utils.hpp"

using namespace std;

#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_ROW_COUNT 2048

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

__global__ void convert_csr_kernel_1(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
                                int * row, int *col, float * values, int * extra_buffer)
{

    __shared__ int reduce[MAX_ROW_COUNT];
    assert(blockDim.x<=MAX_ROW_COUNT);
    // initial the shared flag
    uint row_id = blockIdx.x;
    uint tid = threadIdx.x;
    int global_offset =  row_id * w + tid * 4;
    
    assert(w % 4 == 0);

    int flag = 0;
    for(int _pos = tid; _pos< w / 4; _pos+=blockDim.x){     
        int4 data = __ldg((const int4*)(add_ptr_u(mask, global_offset)));
        flag += data.x + data.y + data.z + data.w;
    }
    reduce[tid] = flag;

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

    if(tid==0 && reduce[0]>0){
        extra_buffer[row_id] = reduce[0];
        extra_buffer[h+row_id] = reduce[0];
    }

}
__global__ void convert_csr_kernel_2(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int * row, int *col, float * values, int * extra_buffer)
{
    // __shared__ int prefix_sum[MAX_ROW_COUNT];
    __shared__ int prefix_count;
    uint row_id = blockIdx.x;
    uint tid = threadIdx.x;
    uint posid;
    if(extra_buffer[row_id]>0){
        if (tid==0){

            prefix_count = 0;
            for(int i=0; i<row_id;i++){
                prefix_count +=  extra_buffer[i];
            }
            row[row_id] = prefix_count;
            if (row_id==h-1){
                row[h] = prefix_count + extra_buffer[row_id];
            }
        }
        __syncthreads();
        int global_offset =  row_id * w + tid;
        if(mask[global_offset]){
            posid = atomicSub(&(extra_buffer[h+row_id]), 1);
            col[prefix_count+posid-1] = tid;
            values[prefix_count+posid-1] = dense[global_offset];
        }
    }
}

void convert_csr(int * mask, float * dense, int h, int w,
    int * row, int *col, float*values, int * extra_buffer)
{
    // need reset the extra buffer here
    CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*h*2) );
    dim3 block_dim(w/4);
    dim3 block_dim2(w);
    dim3 grid_dim(h);
    // std::cout<<"grid_dim "<< w/block_w << ", " <<h/block_h << std::endl;
    convert_csr_kernel_1<<<grid_dim, block_dim>>>(mask, dense, h, w, row, col, values, extra_buffer);
    convert_csr_kernel_2<<<grid_dim, block_dim2>>>(mask, dense, h, w, row, col, values, extra_buffer);


}
bool verify_csr(int * mask, float * data, int h, int w, int* row, int * col, float* values)
{
    for(int row_id = 0; row_id<h; row_id++){
        int _start = row[row_id];
        int _end = row[row_id+1];
        for(int _pos= _start; _pos<_end; _pos++){
            int col_id = col[_pos];
            int global_offset = row_id * w + col_id;
            assert(values[_pos] == data[global_offset]);
        }
    }
    int sum = 0;
    for(int i=0; i<h; i++)
        for(int j=0; j<w; j++)
            sum+=mask[i*w+j];
    assert(sum == row[h]);
    printf("nnz: %d, sparsity: %f\n", sum, 1.0*sum/h/w);
    return true;
}

int main()
{
    cudaEvent_t time_start, time_end;
    float msecTotal=0;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    int M=1024, K=1024;
    
    // float sparsiy=0.9999;
    float sparsity=0.95;
    float * data = (float*) malloc(sizeof(float)*M*K);
    int * mask = (int*) malloc(sizeof(int)*M*K);
    int * row = (int*) malloc(sizeof(int)*(M+1));
    int * col = (int*) malloc(sizeof(int)*M*K);
    float * values = (float*)malloc(sizeof(float)*M*K);


    init_mask(mask, M*K , sparsity);
    init(data, M*K, 0);
    
    int * d_mask, *d_row, *d_col, *extra_buffer;
    float * d_data, *d_val;
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int)*(M+1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&extra_buffer, sizeof(int)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&d_data, sizeof(float)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float)*M*K));
    CUDA_SAFE_CALL(cudaMemcpy(d_data, data, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int)*M*K, cudaMemcpyHostToDevice));
    
    convert_csr(d_mask, d_data, M, K, d_row, d_col, d_val, extra_buffer);

    CUDA_SAFE_CALL(cudaMemcpy(row, d_row, sizeof(int)*(M+1), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(col, d_col, sizeof(int)*M*K, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(values, d_val, sizeof(int)*M*K, cudaMemcpyDeviceToHost));

    verify_csr(mask, data, M, K, row, col, values);
    printf("Verification passed!\n");
    int n_iter = 100;

    CUDA_SAFE_CALL(cudaEventRecord(time_start));
    for(int i=0;i<n_iter;i++){
        convert_csr(d_mask, d_data, M, K, d_row, d_col, d_val, extra_buffer);    
    }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    printf("Dynamic convert time cost= %f msec\n", msecTotal/n_iter);
    return 0;
}