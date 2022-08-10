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
#include <random>
// #include <math>
#include <algorithm>
#include <assert.h>
// CUDA runtime
#include <cuda.h>
#include "utils.hpp"

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

__global__ void convert_bcsr_kernel_1(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
                                int block_h, int block_w, int * row, int *col, int * row_pos,
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
    int block_h, int block_w, int * row, int *col, int * row_pos,
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
            row_pos[prefix_count+pos_id] = by;
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
    int block_h, int block_w, int * row, int *col, int * row_pos,
    float*values, int * extra_buffer)
{
    // need reset the extra buffer here
    assert(block_w % 4 == 0);
    CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*(2*h+(h/block_h)*(w/block_w))) );
    CUDA_SAFE_CALL(cudaMemset((void*)row, 0, sizeof(int)*(1+(h/block_h))) );
    dim3 block_dim(block_h*block_w/4);
    dim3 grid_dim(w/block_w, h/block_h);
    // std::cout<<"grid_dim "<< w/block_w << ", " <<h/block_h << std::endl;
    convert_bcsr_kernel_1<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer);
    convert_bcsr_kernel_2<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer);


}
__global__ void longformer_mixed_softmax_kernel(float * A,
                                     int * row,
                                     int *col,
                                     float* val_mask,
                                     int * global_attention,
                                     float* extra_buffer,
                                     int block_h,
                                     int block_w,
                                     int block_nnz,
                                     int row_tile,
                                     int M,
                                     int N,
                                     int global_attention_size)
{
    /*
    description:
    each row of blocks is dealt with a thread group
    each block is 32x32
    */
    A += M * N * blockIdx.y;
    extra_buffer += M * global_attention_size * blockIdx.y;
    uint blk_row_idx = blockIdx.x / (block_h/row_tile) ;
    int block_inter_row = (blockIdx.x % (block_h/row_tile)) * row_tile;
    uint bm = threadIdx.x / block_w;
    uint bn = threadIdx.x % block_w;
    assert(block_w % 32==0);
    float regC = 0.0f;
    float regSum = 0.0f;
    float regMax = -100000.0;
    int block_seq_start = row[blk_row_idx];
    int block_seq_end = row[blk_row_idx+1];
    uint A_index, col_idx, mask_index;
    for(int i=bn; i<global_attention_size; i+=32){
        A_index = (blockIdx.x * row_tile + bm) * N + global_attention[i];
        regMax = max(regMax, A[A_index]);
    }
    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        mask_index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        A_index = (blockIdx.x * row_tile + bm) * N + (col[block_seq] * block_w + bn);
        regMax = max(regMax, A[A_index] * val_mask[mask_index]);

    }
    for (int offset = 16; offset > 0; offset /= 2) {
        regMax = max(regMax, __shfl_down_sync(FULL_MASK, regMax, offset));
    }
    regMax = __shfl_sync(FULL_MASK, regMax, 0);

    for(int i=bn; i<global_attention_size; i+=32){
        A_index = (blockIdx.x * row_tile + bm) * N + global_attention[i];
        regC = expf(A[A_index]-regMax);
        regSum += regC;
        A[A_index] = -10000.0;
        extra_buffer[(blockIdx.x * row_tile + bm)*global_attention_size+ i] = regC; 
    }
    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        mask_index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        A_index = (blockIdx.x * row_tile + bm) * N + (col[block_seq] * block_w + bn);
        if (val_mask[mask_index] != 0) {
            regC = expf(A[A_index]-regMax);
            regSum += regC;
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        regSum += __shfl_down_sync(FULL_MASK, regSum, offset);
    }
    regSum = __shfl_sync(FULL_MASK, regSum, 0);
    // if(threadIdx.x%32==1)
    //     printf("Row %d Regsum %f  \n", block_inter_row + bm + blk_row_idx * block_h, regSum);
    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        mask_index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        A_index = (blockIdx.x * row_tile + bm) * N + (col[block_seq] * block_w + bn);
        regC = 0.0f;
        if (val_mask[mask_index] > 0) {
            A[A_index] = expf(A[A_index]-regMax)/regSum;
        }
        else{
            A[A_index] = 0;
        }

    }
    for(int i=bn; i<global_attention_size; i+=32){
        A_index = (blockIdx.x * row_tile + bm) * N + global_attention[i];
        A[A_index] = extra_buffer[(blockIdx.x * row_tile + bm)*global_attention_size+ i]/regSum; 
    }


}
__global__ void longformer_mixed_softmax_kernel_v2(float * A,
                                     int * row,
                                     int *col,
                                     float* val_mask,
                                     int * global_attention,
                                     float* extra_buffer,
                                     int block_h,
                                     int block_w,
                                     int block_nnz,
                                     int row_tile,
                                     int M,
                                     int N,
                                     int global_attention_size)
{
    /*
    description:
    each row of blocks is dealt with a thread group
    each block is 32x32
    */
    A += M * N * blockIdx.y;
    extra_buffer += M * global_attention_size * blockIdx.y;
    uint blk_row_idx = blockIdx.x / (block_h/row_tile) ;
    int block_inter_row = (blockIdx.x % (block_h/row_tile)) * row_tile;
    uint bm = threadIdx.x / block_w;
    uint bn = threadIdx.x % block_w;
    assert(block_w % 32==0);
    float regC = 0.0f;
    float regSum = 0.0f;
    float regMax = -100000.0;
    int block_seq_start = row[blk_row_idx];
    int block_seq_end = row[blk_row_idx+1];
    uint A_index, col_idx, mask_index;
    for(int i=bn; i<global_attention_size; i+=32){
        A_index = (blockIdx.x * row_tile + bm) * N + global_attention[i];
        regMax = max(regMax, A[A_index]);
    }
    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        mask_index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        A_index = (blockIdx.x * row_tile + bm) * N + (col[block_seq] * block_w + bn);
        regMax = max(regMax, A[A_index] * val_mask[mask_index]);

    }
    for (int offset = 16; offset > 0; offset /= 2) {
        regMax = max(regMax, __shfl_down_sync(FULL_MASK, regMax, offset));
    }
    regMax = __shfl_sync(FULL_MASK, regMax, 0);

    for(int i=bn; i<global_attention_size; i+=32){
        A_index = (blockIdx.x * row_tile + bm) * N + global_attention[i];
        regC = expf(A[A_index]-regMax);
        regSum += regC;
        A[A_index] = -10000.0;
        extra_buffer[(blockIdx.x * row_tile + bm)*global_attention_size+ i] = regC; 
    }
    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        mask_index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        A_index = (blockIdx.x * row_tile + bm) * N + (col[block_seq] * block_w + bn);
        if (val_mask[mask_index] != 0) {
            regC = expf(A[A_index]-regMax);
            regSum += regC;
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        regSum += __shfl_down_sync(FULL_MASK, regSum, offset);
    }
    regSum = __shfl_sync(FULL_MASK, regSum, 0);
    // if(threadIdx.x%32==1)
    //     printf("Row %d Regsum %f  \n", block_inter_row + bm + blk_row_idx * block_h, regSum);
    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        mask_index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        A_index = (blockIdx.x * row_tile + bm) * N + (col[block_seq] * block_w + bn);
        regC = 0.0f;
        if (val_mask[mask_index] > 0) {
            A[A_index] = expf(A[A_index]-regMax)/regSum;
        }
        else{
            A[A_index] = 0;
        }

    }
    for(int i=bn; i<global_attention_size; i+=32){
        A_index = (blockIdx.x * row_tile + bm) * N + global_attention[i];
        A[A_index] = extra_buffer[(blockIdx.x * row_tile + bm)*global_attention_size+ i]/regSum; 
    }


}
void longformer_mixed_softmax_launch(float * A,
                                     int * row,
                                     int *col,
                                     float* val_mask,
                                     int * global_attention,
                                     float* extra_buffer,
                                     int block_h,
                                     int block_w,
                                     int block_nnz,
                                     int M,
                                     int N,
                                     int head_num,
                                     int batch_size,
                                     int global_attention_size
)
{
    const int row_tile=8;
    const dim3 blockDim(row_tile*32);
    const dim3 gridDim(M/row_tile, head_num*batch_size);
    longformer_mixed_softmax_kernel<<<gridDim, blockDim>>>(A,
                                                           row,
                                                           col,
                                                           val_mask,
                                                           global_attention,
                                                           extra_buffer,
                                                           block_h,
                                                           block_w,
                                                           block_nnz,
                                                           row_tile,
                                                           M, N,
                                                           global_attention_size
                                                           );
}
void generate_global_attention(int * g, int count, int max_seq_len)
{
    std::vector<int> vec;
    for(int i=0; i<max_seq_len; i++)
        vec.push_back(i);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(vec.begin(), vec.end(), rng);
    for(int i=0;i<count;i++)
        g[i] = vec[i];
    for(int i=0;i<count;i++)
        printf("%d\n", g[i]);
}
int main()
{
    cudaEvent_t time_start, time_end;
    float msecTotal=0;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    int head_num = 12;
    int batch_size = 1;
    int M=4096, N=4096;
    int block_h=32, block_w = 32;
    int global_attention_size = 80;
    // float sparsiy=0.9999;
    float sparsiy=0.85;
    float * data = (float*) malloc(sizeof(float)*M*N*batch_size*head_num);
    int * mask = (int*) malloc(sizeof(int)*M*N);
    int * row = (int*) malloc(sizeof(int)*(M+1));
    int * col = (int*) malloc(sizeof(int)*M*N);
    int * global_atten = (int *) malloc(sizeof(int) * global_attention_size);
    float * values = (float*)malloc(sizeof(float)*M*N*batch_size*head_num);
    float * A = (float*)malloc(sizeof(float)*M*N*batch_size*head_num);
    
    float *dA;
    generate_global_attention(global_atten, global_attention_size, M);
    init_mask_blockwise(mask, M, N , block_h, block_w, sparsiy);
    int mask_nnz = 0;
    for(int i=0;i<M*N; i++){
        data[i] = float(mask[i]);
        mask_nnz += mask[i];
    }
    printf("Mask NNZ: %d \n", mask_nnz);
    // init(data, M*N, 0);
    init(A, M*N*batch_size*head_num, 0);
    

    int * d_mask, *d_row, *d_col, *extra_buffer, *d_row_pos, *d_global_atten;
    float * d_data, *d_val, *d_extra_buffer;
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float)*M*N*batch_size*head_num));
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int)*M*N));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int)*(M+1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int)*M*N));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_pos, sizeof(int)*M*N));
    CUDA_SAFE_CALL(cudaMalloc(&d_extra_buffer, sizeof(float)*M*N*batch_size*head_num));
    CUDA_SAFE_CALL(cudaMalloc(&extra_buffer, sizeof(int)*M*N*batch_size*head_num));
    CUDA_SAFE_CALL(cudaMalloc(&d_data, sizeof(float)*M*N*batch_size*head_num));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float)*M*N*batch_size*head_num));
    CUDA_SAFE_CALL(cudaMalloc(&d_global_atten, sizeof(int)*global_attention_size));
    CUDA_SAFE_CALL(cudaMemcpy(d_data, data, sizeof(float)*M*N*batch_size*head_num, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int)*M*N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*N*batch_size*head_num, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_global_atten, global_atten, sizeof(int)*global_attention_size, cudaMemcpyHostToDevice));
    convert_bcsr(d_mask, d_data, M, N, block_h, block_w, d_row, d_col, d_row_pos, d_val, extra_buffer);

    CUDA_SAFE_CALL(cudaEventRecord(time_start));
    for(int runtime=0; runtime<10; runtime++){
        longformer_mixed_softmax_launch(dA,
                                    d_row,
                                    d_col,
                                    d_val,
                                    d_global_atten,
                                    d_extra_buffer,
                                    block_h,
                                    block_w,
                                    0,
                                    M,
                                    N,
                                    head_num,
                                    batch_size,
                                    global_attention_size);
    }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));

    
    printf("Longformer Softmax Kernel Time: %f\n", msecTotal/10);

    return 0;
}