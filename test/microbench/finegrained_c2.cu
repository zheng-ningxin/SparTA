
#include <assert.h>
// CUDA runtime
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
#include <time.h>
// #include <math>
#include <algorithm>
#include <assert.h>
#include "iostream"
#include "sstream"
#include "time.h"
#include "memory"
#include "vector"
using namespace std;

// #include "utils.hpp"
using namespace std;
#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024
#define FULL_MASK 0xffffffff

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

__device__ __forceinline__ float2  _add(float2 x, float2 y) { float2 res; res.x = x.x + y.x; res.y = x.y + y.y; return res; }

void init(float * ptr, size_t length, float sparsity)
{
    for (int i = 0; i < length; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (pro < sparsity)
        {
            ptr[i] = 0.0;
        }
        else
        {
            ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}
int convert_csr(float * ptr, int32_t row, int32_t col, int32_t * row_idx, int32_t * col_idx, float * values)
{
    auto v_row_idx = std::make_shared<vector<int32_t>>();
    auto v_col_idx = std::make_shared<vector<int32_t>>();
    auto v_values = std::make_shared<vector<float>>();

    for (int i = 0; i < row; i++)
    {
        v_row_idx->push_back(v_values->size());
        for (int j = 0; j < col; j++)
        {
            size_t pos = i * col + j;
            if (ptr[pos] < 1e-8)
            {
                // sparsity
                continue;
            }
            else
            {
                v_values->push_back(ptr[pos]);
                v_col_idx->push_back(j);
            }
        }
    }
    v_row_idx->push_back(v_values->size());
    int row_idx_size = sizeof(int32_t)*v_row_idx->size();
    int col_idx_size = sizeof(int32_t)*v_col_idx->size();
    int values_size = sizeof(float)*v_values->size();
    printf("values_size: %d\n", values_size);

    memcpy(row_idx, v_row_idx->data(), row_idx_size);
    memcpy(col_idx, v_col_idx->data(), col_idx_size);
    memcpy(values, v_values->data(), values_size);
    return v_values->size();
}

void calculate_reference(int m, int k, int n, float * A, float *B, float * C) 
{
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            float sum = 0.0;
            for(int tmp=0; tmp<k; tmp++){
                sum += A[i * k + tmp] * B[tmp * n + j];
            }
            C[i*n+j] = sum;
        }
    }
}
#define MAX_SM_SIZE 256
__global__ void
//__launch_bounds__(256, 5)
fused_gather_mul(const int sz_m,
                 const int sz_k,
                 const int sz_n,
                 const int *__restrict__ dst_idx,
                 const int *__restrict__ src_idx,
                 const float *__restrict__ edge_val,
                 const float *__restrict__ src_feature,
                 float *__restrict__ accum)
{
    __shared__ int col_src_s[MAX_SM_SIZE];
    __shared__ float val_src_s[MAX_SM_SIZE];
    int row = blockIdx.x;
    int st = dst_idx[row];
    int off = st;
    int ed = dst_idx[row + 1];
    int n_item = ed - st;
    for (int _ = 0; _ < sz_n; _ += blockDim.x)
    {
        accum[row * sz_n + _ + threadIdx.x] = 0;
    }
    for (int i = st + threadIdx.x;; i += blockDim.x)
    {
        if (i < ed)
        {
            col_src_s[i - off] = __ldg(src_idx + i);
            val_src_s[i - off] = __ldg(edge_val + i);
        }
        n_item = min(blockDim.x, ed - off);
        off += n_item;
        __syncthreads();
#pragma unroll 1
        for (int _ = threadIdx.x; _ < sz_n; _ += blockDim.x)
        {
            float val = 0;
            for (int t = 0; t < n_item; t++)
            {
                val += val_src_s[t] * src_feature[col_src_s[t] * sz_n + _];
            }
            accum[row * sz_n + _] += val;
        }
        __syncthreads();
        if (off >= ed)
            break;
    }
}
void SpmmFunc(
    int sz_m,
    int sz_k,
    int sz_n,
    int nnz,
    const int *__restrict__ dst_idx,
    const int *__restrict__ src_idx,
    const float *__restrict__ edge_val,
    const float *__restrict__ src_feature,
    float *__restrict__ accum)
{
    int threads = min(MAX_SM_SIZE, sz_n); //   int threads = min(1024, sz_n);
    fused_gather_mul<<<sz_m, threads, 0, 0>>>(sz_m, sz_k, sz_n, dst_idx, src_idx, edge_val, src_feature, accum);
}



int main()
{
    int M, K, N;
    M = 4096;
    K = 4096;
    N = 4096;
    const int n_iter = 100;
    float sparsity_ratio = 0.6;

    cudaEvent_t time_start, time_end;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    float msecTotal = 0;
    float * A, *B, *C, *val, *refC;
    float * dA, *dB, *dC, *d_val;

    int * mask, *d_mask, *row, *d_row, *row_pos, *d_row_pos, *col, *d_col, *d_extra_buffer;
    A = (float*) malloc(sizeof(float) * M * K);
    B = (float*) malloc(sizeof(float) * K * N);
    C = (float*) malloc(sizeof(float) * M * N);
    refC = (float*) malloc(sizeof(float) * M * N);

    row = (int*) malloc(sizeof(int) * (M+1));
    col = (int*) malloc(sizeof(int) *  M * K);
    val = (float*) malloc(sizeof(float) * M * K);
    init(A, M*K, sparsity_ratio);
    init(B, N*K, sparsity_ratio);
    // apply mask

    convert_csr(A, M, K, row, col, val);
    int nnz = row[M];
    
    printf("NNZ: %d\n", nnz);
    printf("Sparsity ratio: %f\n", nnz*1.0/M/K);
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int) * (M + 1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int) * M * K));

    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(float) * N * K));
    CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_SAFE_CALL(cudaMemset(dC, 0, sizeof(float)* M * N));
    
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row, row, sizeof(int)*(M+1), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col, col, sizeof(int)* M * K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_val, val, sizeof(float) * M * K, cudaMemcpyHostToDevice));

    

    // KxM = KxN * (MxN)^T
    CUDA_SAFE_CALL(cudaEventRecord(time_start));

    for(int run=0; run<n_iter; run++){
        SpmmFunc(M, K, N, nnz, d_row, d_col, d_val, dB, dC);
    }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    printf("Time Cost: %.3fms\n", msecTotal/n_iter);
    CUDA_SAFE_CALL(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    // calculate_reference(M, K, N, A, B, refC);
    // for(int i=0;i<100;i++){
    //     printf("%f %f\n", C[i], refC[i]);
    // }

    return 0;

}