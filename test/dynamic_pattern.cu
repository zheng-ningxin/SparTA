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

__global__ void convert_bcsr_kernel_1(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
                                int block_h, int block_w, int * row, int *col,
                                float * values, int * extra_buffer)
{

    __shared__ int reduce[MAX_BLOCK_THREAD_COUNT];
    assert(blockDim.x<=MAX_BLOCK_THREAD_COUNT);
    // initial the shared flag
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tid = threadIdx.x;
    int global_offset =  (by * block_h) * w + bx * block_w;
    int block_size =  block_h * block_w;
    assert(block_w % 4 == 0);
    // cannot handle the misalignment for now
    assert((block_size / 4) % blockDim.x==0);
    int flag = 0;
    for(int _pos = tid; _pos< block_size / 4; _pos+=blockDim.x){
        uint block_offset = _pos / (block_w / 4) * w + _pos % (block_w / 4) * 4;        
        int4 data = __ldg((const int4*)(add_ptr_u(mask, global_offset+block_offset)));
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
    int pos_id;
    if(tid==0 && reduce[0]>0){
        pos_id= atomicAdd(&extra_buffer[by], 1);
        atomicAdd(&extra_buffer[by+h], 1);
        atomicAdd(&row[h/block_h], 1);
        extra_buffer[2*h + gridDim.x * by + pos_id] = bx;
    }

}
__global__ void convert_bcsr_kernel_2(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col,
    float * values, int * extra_buffer)
{
    __shared__ int pos_id, prefix_count, ori_bx, ori_by;
    __shared__ int prefix_sum[MAX_BLOCK_THREAD_COUNT];
    uint by = blockIdx.y;
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;

    if (tid==0){
        pos_id = -1;
        prefix_count = 0;
        // contend for the block

        pos_id = atomicSub(&extra_buffer[by], 1);
        pos_id-=1;
        if (pos_id>=0){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }
            ori_by = by;
            ori_bx = extra_buffer[ 2*h + by * gridDim.x + pos_id];       
            
            row[by] = prefix_count;
            col[prefix_count+pos_id] = ori_bx;
        }
    }
    __syncthreads();
    if(pos_id>=0){
        int global_offset =  (ori_by * block_h) * w + ori_bx * block_w;
        int block_size = block_h * block_w;
        int write_global_offset = (prefix_count + pos_id) * block_size;

        for(int _pos=tid; _pos<block_size/4; _pos+=blockDim.x){
            uint block_offset = _pos / (block_w / 4) * w + _pos % (block_w / 4) * 4;
            float4 data = __ldg((const float4*)(add_ptr_f(dense, global_offset + block_offset)));
            *(float4*)&values[write_global_offset+_pos*4] = data;
        }
        
    }

}

void convert_bcsr(int * mask, float * dense, int h, int w,
    int block_h, int block_w, int * row, int *col,
    float*values, int * extra_buffer)
{
    // need reset the extra buffer here
    assert(block_w % 4 == 0);
    CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*(h+(h/block_h)*(w/block_w))) );
    dim3 block_dim(block_h*block_w/4);
    dim3 grid_dim(w/block_w, h/block_h);
    // std::cout<<"grid_dim "<< w/block_w << ", " <<h/block_h << std::endl;
    convert_bcsr_kernel_1<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, values, extra_buffer);
    convert_bcsr_kernel_2<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, values, extra_buffer);


}
bool verify_bcsr(int * mask, float * data, int h, int w, int block_h, int block_w, int* row, int * col, float* values)
{
    for(int rid=0; rid<h/block_h; rid++){
        // printf("row-%d: %d row-%d : %d\n", rid, row[rid], rid+1, row[rid+1]);
        int _start = row[rid];
        int _end = row[rid+1];
        for(int _pos=_start; _pos<_end; _pos++){
            int cid = col[_pos];
            for(int i=0;i<block_h;i++){
                for(int j=0;j<block_w;j++){
                    int offset = (rid * block_h+i) * w + cid * block_w + j;
                    int csr_offset = _pos * block_h * block_w + i * block_w + j;
                    if (mask[offset]>0){
                        // printf("%f %f\n", data[offset], values[csr_offset]);
                        if(abs(data[offset]-values[csr_offset])>1e-8)
                        {
                            return false;
                        }
                        mask[offset]= 0;
                    }
                }
            }
        }
    }
    printf("%d blocks remained\n", row[h/block_h]);
    printf("Blockwise sparsity %f \n", 1.0-1.0*row[h/block_h]/(h/block_w)/(w/block_w));
    for(int i=0;i<block_h*block_w;i++)
        if(mask[i])
            return false;
    return true;
}

int main()
{
    cudaEvent_t time_start, time_end;
    float msecTotal=0;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    int M=1024, K=1024;
    int block_h=32, block_w = 32;
    // float sparsiy=0.9999;
    float sparsity=0.99;
    float * data = (float*) malloc(sizeof(float)*M*K);
    int * mask = (int*) malloc(sizeof(int)*M*K);
    int * row = (int*) malloc(sizeof(int)*(M+1));
    int * col = (int*) malloc(sizeof(int)*M*K);
    float * values = (float*)malloc(sizeof(float)*M*K);

    // FILE * mask_f;
    // mask_f = fopen("mask.bin", "rb");
    // if (mask_f==NULL){
    //     printf("Load failed!\n");
    //     return -1;
    // }
    // printf("load succeed\n");
    // fread(mask, sizeof(int), M*K, mask_f);
    // fclose(mask_f);
    // int vcount = 0;
    // for(int i=0;i<M;i++)
    //     for(int j=0;j<K;j++)
    //         if (mask[i*K+j]){
    //             vcount +=1;
    //             // printf("mask:%d\n", mask[i*K+j]);
    // }
    init_mask_blockwise(mask, M, K, block_h, block_w, sparsity);
    // init_mask(mask, M*K , sparsity);
    init(data, M*K, 0);
    // FILE * mask_f;
    // mask_f = fopen("mask.bin", "wb");
    // if (mask_f==NULL){
    //     printf("Dump failed!\n");
    //     return -1;
    // }
    // fwrite(mask, sizeof(int), M*K, mask_f);
    // fclose(mask_f);
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

    convert_bcsr(d_mask, d_data, M, K, block_h, block_w, d_row, d_col, d_val, extra_buffer);

    CUDA_SAFE_CALL(cudaMemcpy(row, d_row, sizeof(int)*(M+1), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(col, d_col, sizeof(int)*M*K, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(values, d_val, sizeof(int)*M*K, cudaMemcpyDeviceToHost));

    std::cout<<"Correctness check: "<< verify_bcsr(mask, data, M, K, block_h, block_w, row, col, values) << std::endl;
    int n_iter = 100;

    CUDA_SAFE_CALL(cudaEventRecord(time_start));
    for(int i=0;i<n_iter;i++){
        convert_bcsr(d_mask, d_data, M, K, block_h, block_w, d_row, d_col, d_val, extra_buffer);    
    }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    printf("Dynamic convert time cost= %f msec\n", msecTotal/n_iter);
    return 0;
}