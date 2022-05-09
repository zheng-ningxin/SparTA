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
    for(int i=0;i<block_h*block_w;i++)
        if(mask[i])
            return false;
    return true;
}



__device__ __forceinline__ float2  _add(float2 x, float2 y) { float2 res; res.x = x.x + y.x; res.y = x.y + y.y; return res; }



__global__ void BLOCK_SPARSE_MATMUL_OUT_32_64_32(
    float* A,
    float* B,
    float* C_val,
    int * row_index,
    int * col_index,
    int GLOBAL_M,
    int GLOBAL_K,
    int GLOBAL_N,
    int SPARSE_VAL_SIZE){
    /*
    description:
    tiling k dimension
    smm_dd_s_nn: sparse matmul, dense (MxK, along K) x dense (KxN, along N) -> sparse (MxN, along N)
    the output sparse is block size 32x32, the blocks will be written to bcsr 32x64
    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;

    A += M*K*blockIdx.y;
    B += K*N*blockIdx.y;
    C_val += SPARSE_VAL_SIZE*blockIdx.y;

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = col_index[blockIdx.x]; // N
    uint by = row_index[blockIdx.x]; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;
    uint ori_offsetA00 = (by * 32 + ty) * K + k;
    uint ori_offsetA16 = ori_offsetA00 + K * 16;
    uint ori_offsetB00 = (bx * 32 + ty) * K + k;
    uint ori_offsetB16 = ori_offsetB00 + K * 16;

    uint tid224 = tid & 224;
    uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;
    loadA += (tid224 * 32) + (tid224 / 2);
    loadB += (tid224 * 32) + (tid224 / 2);

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storAB) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    for (int k_seq = 0; k_seq < (int)(K/64); k_seq++)
    {
        uint offsetA00 = ori_offsetA00 + 64 * k_seq;
        uint offsetA16 = ori_offsetA16 + 64 * k_seq;
        uint offsetB00 = ori_offsetB00 + 64 * k_seq;
        uint offsetB16 = ori_offsetB16 + 64 * k_seq;

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        a00 = __ldg((const float4*)(add_ptr_f(A, offsetA00)));
        a16 = __ldg((const float4*)(add_ptr_f(A, offsetA16)));
        b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
        b16 = __ldg((const float4*)(add_ptr_f(B, offsetB16)));

        __syncthreads();

        *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        *(float*)&bShare[storAB + (0*32 +  0 + 1*65*32)*4] = b00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 1*65*32)*4] = b00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 1*65*32)*4] = b00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 1*65*32)*4] = b00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 1*65*32)*4] = b16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 1*65*32)*4] = b16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 1*65*32)*4] = b16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 1*65*32)*4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
            *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
            *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        #pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
            *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
            *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
            *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
            *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
            *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
    }

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bx)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(by) :);

    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    // uint blk_index = block_index[blockIdx.x] / 2;
    uint blk_index = blockIdx.x;
    // uint intra_blk_index = block_index[blockIdx.x] % 2;
    C_val += 32 * 32 * blk_index;
    C_val += ty * 32 + tx * 2;

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    //-> store((bhalf2*)C, c2[0]);
    *(float2*)C_val = c2[0];

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    C_val += 16 * 32;
    *(float2*)C_val = c2[0];

}

void matmul_sparse_out_32_64_32(float * dA, float * dB, float *dC, int *row_index, int * col_index, int * row_pos,int M, int K, int N, int block_h, int block_w, int SPARSE_VAL_SIZE, int SMALL_BLOCK_NUM)
{
    const dim3 dimBlock(256);
    int n_row = M/block_h;
    // int SMALL_BLOCK_NUM;
    // Note: need copy from the device to get the number of the number of the small block
    // CUDA_SAFE_CALL(cudaMemcpy(&SMALL_BLOCK_NUM, row_index+n_row, sizeof(int), cudaMemcpyDeviceToHost));
    const dim3 dimGrid(SMALL_BLOCK_NUM, 1);
    printf("Small block number: %d \n", SMALL_BLOCK_NUM);
    // const int SPARSE_VAL_SIZE = SMALL_BLOCK_NUM * block_h * block_w;
    BLOCK_SPARSE_MATMUL_OUT_32_64_32<<<dimGrid, dimBlock>>>(dA, dB, dC, row_pos, col_index, M, K, N, SPARSE_VAL_SIZE);
}

void calculate_ref_nt(float * A, float *B, float * C, int M, int K, int N)
{
    for(int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            float sum = 0;
            for(int k=0;k<K;k++){
                // sum += A[m][k] * B[n][k];
                sum += A[m*K+k] * B[n*K+k];
            }
            // C[m][n] = sum
            C[m*N+n] = sum;
        }

    }
}


void calculate_ref_nn(float * A, float *B, float * C, int M, int K, int N)
{
    for(int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            float sum = 0;
            for(int k=0;k<K;k++){
                // sum += A[m][k] * B[k][n];
                sum += A[m*K+k] * B[k*N+n];
            }
            // C[m][n] = sum
            C[m*N+n] = sum;
        }

    }
}


bool verify_matmul_sparse_out(float * ref_C, float * C_val, int * row_index, int * col_index, int h, int w, int block_h, int block_w)
{
    bool flag = true;
    for(int rowid=0; rowid<h/block_h; rowid++){
        int start = row_index[rowid];
        int end = row_index[rowid+1];
        for(int pos=start; pos<end; pos++){
            int colid = col_index[pos];
            int offset_ref = rowid* block_h * w + colid * block_w;
            int offset_c = pos * block_h * block_w;
            for(int i=0;i<block_h;i++){
                for(int j=0;j<block_w;j++){
                    float ref_value =  ref_C[offset_ref+ i*w+j];
                    float our_value =  C_val[offset_c+i*block_w+j];
                    if(fabs(ref_value-our_value)>1e-5){
                        printf("%f %f\n", ref_value, our_value);
                        flag = false;
                    }
                }
            }
        }
    }
    return flag;
}



void sparse_softmax_v2(float * C_val, float * C_val_mask, int * row_ptr, int * col_index, int * row_pos, float * ext_buffer, int h, int w, int block_h, int block_w)
{
    CUDA_SAFE_CALL(cudaMemset(ext_buffer, 0, sizeof(float)*h));
    int batchsize = 1;
    int HEAD_NUM = 1;
    CUDA_SAFE_CALL(cudaMemset(ext_buffer, 0, sizeof(float) * (h/block_h)+1 * batchsize * HEAD_NUM));
    const dim3 softmax_dimBlock(32*32);
    const dim3 softmax_dimGrid(h/block_h, HEAD_NUM * batchsize);

    
}

__global__ void SPARSE_SOFTMAX_STAGE1(float * C_val, float*C_val_mask, int*row_ptr, int*col_index, int * row_pos, float * ext_buffer,int h, int w, int block_h, int block_w, int SPARSE_VAL_SIZE)
{
    C_val += SPARSE_VAL_SIZE * blockIdx.y;

    uint bm = threadIdx.x / 32;
    uint bn = threadIdx.x % 32;
    float regC = 0.0f;
    float regSum = 0.0f;
    int by = row_pos[blockIdx.x];
    int bx = col_index[blockIdx.x];
    uint index=32*32*blockIdx.x + bm*32 + bn;
    regC = expf(C_val[index]) * C_val_mask[index];
    for (int offset = 16; offset > 0; offset /= 2) {
        regC += __shfl_down_sync(FULL_MASK, regC, offset);
    }
    regC = __shfl_sync(FULL_MASK, regC, 0);
    atomicAdd(&ext_buffer[by*32+bm], regC);
}

__global__ void SPARSE_SOFTMAX_STAGE2(float * C_val, float*C_val_mask, int*row_ptr, int*col_index, int * row_pos, float * ext_buffer,int h, int w, int block_h, int block_w, int SPARSE_VAL_SIZE)
{

}
__global__ void SPARSE_SOFTMAX(
    float* C_val,
    float* C_val_mask,
    int* row_index,
    int block_h, int block_w, int SPARSE_VAL_SIZE, int row_tile){
    /*
    description:
    each row of blocks is dealt with a thread group
    each block is 32x32
    */
    C_val += SPARSE_VAL_SIZE*blockIdx.y;

    uint blk_row_idx = blockIdx.x / (block_h/row_tile) ;
    int block_inter_row = (blockIdx.x % (block_h/row_tile)) * row_tile;
    uint bm = threadIdx.x / block_w;
    uint bn = threadIdx.x % block_w;
    assert(block_w % 32==0);
    float regC = 0.0f;
    float regSum = 0.0f;
    int block_seq_start = row_index[blk_row_idx];
    int block_seq_end = row_index[blk_row_idx+1];

    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        uint index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        // regC = (float)C_val_mask[index]*C_val[index];
        // if (C_val_mask[index] != 0) {
            regC = expf(C_val[index]) * C_val_mask[index];
        // }
        regSum += regC;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        regSum += __shfl_down_sync(FULL_MASK, regSum, offset);
    }
    regSum = __shfl_sync(FULL_MASK, regSum, 0);
    // if(threadIdx.x%32==1)
    //     printf("Row %d Regsum %f  \n", block_inter_row + bm + blk_row_idx * block_h, regSum);
    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        uint index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        regC = 0.0f;
        if (C_val_mask[index] > 0) {
            C_val[index] = expf(C_val[index])/regSum;
        }
        else{
            C_val[index] = 0;
        }

    }
}

void sparse_softmax_v1(float * C_val, float * C_val_mask, int * row_ptr, int * col_index, int * row_pos, int h, int w, int block_h, int block_w, int SPARSE_VAL_SIZE)
{
    int batchsize = 1;
    int HEAD_NUM = 1;
    const int row_tile = 4;
    const dim3 softmax_dimBlock(row_tile*32);
    const dim3 softmax_dimGrid(h/row_tile, HEAD_NUM * batchsize);
    SPARSE_SOFTMAX<<<softmax_dimGrid, softmax_dimBlock>>>(C_val, C_val_mask, row_ptr, block_h, block_w, SPARSE_VAL_SIZE, row_tile);
}

void calculate_softmax_ref(float* tin, int* mask, int M, int N)
{
    for(int i=0; i<M; i++){
        float sum = 0;
        for(int j=0; j<N; j++){
            int index =  i *N + j;
            // printf("index: %d mask[index]: %d\n",index, mask[index]);
            if(mask[index]>0){
                sum+=expf(tin[index]);
                // printf("#### i:%d %f \n", mask[index], expf(tin[index]));
            }
        }
        // printf("Row: %d sum: %f\n", i, sum);
        for(int j=0; j<N; j++){
            int index =  i *N + j;
            if(mask[index])
                tin[index]=expf(tin[index]) / sum;
            else
                tin[index] = 0;
        }
    }
}
bool verify_softmax(float * ref_C, float * C_val, int * row_index, int * col_index, int h, int w, int block_h, int block_w)
{
    bool flag = true;
    for(int rowid=0; rowid<h/block_h; rowid++){
        int start = row_index[rowid];
        int end = row_index[rowid+1];
        for(int pos=start; pos<end; pos++){
            int colid = col_index[pos];
            int offset_ref = rowid* block_h * w + colid * block_w;
            int offset_c = pos * block_h * block_w;
            for(int i=0;i<block_h;i++){
                for(int j=0;j<block_w;j++){
                    float ref_value =  ref_C[offset_ref+ i*w+j];
                    float our_value =  C_val[offset_c+i*block_w+j];
                    if(fabs(ref_value-our_value)>1e-5){
                        printf("%f %f\n", ref_value, our_value);
                        flag = false;
                    }
                }
            }
        }
    }
    return flag;
}
template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_MATMUL_SDD(int* csr_row, int * csr_col, float* csr_val, float * B, float* C,  int M, int K, int N, int block_h, int block_w){
    // const int BLOCK_SIZE_M = 32;
    // const int BLOCK_SIZE_K = 32;
    // const int BLOCK_SIZE_N = 64;
    // const int THREAD_SIZE_M = 4;
    // const int THREAD_SIZE_K = 4;
    // const int THREAD_SIZE_N = 4;
    int by = blockIdx.y; // M
    int bx = blockIdx.x; // N
    int ty = threadIdx.y; 
    int tx = threadIdx.x;
    const int padding = 1;
    __shared__ float As[BLOCK_SIZE_M * (padding+BLOCK_SIZE_K)];
    __shared__ float Bs[BLOCK_SIZE_N * (padding+BLOCK_SIZE_K)];

    float accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    float a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    float b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int index_start = csr_row[by], index_end = csr_row[by+1];

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;
    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;

    for(int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1){
        int col_pos = csr_col[tile_block_idx] * BLOCK_SIZE_K;
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                FETCH_FLOAT4(csr_val[tile_block_idx * BLOCK_SIZE_M * BLOCK_SIZE_K + OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]);
        }

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
            FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = 
                FETCH_FLOAT4(B[OFFSET(col_pos+k+B_BLOCK_ROW_START, bx*BLOCK_SIZE_N + B_BLOCK_COL_START, N)]);
                // FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]);
                // FETCH_FLOAT4(B[OFFSET(tile_idx+k+B_BLOCK_ROW_START, bx*BLOCK_SIZE_N+B_BLOCK_COL_START, N)]);
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += THREAD_SIZE_K){
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j += 1){
                    a_frag[j][i] = As[OFFSET(ty + vBLOCK_SIZE_M * j, k+i, BLOCK_SIZE_K)];
                    //a_frag[j][i] = As[OFFSET(k+i, ty + vBLOCK_SIZE_M * j, BLOCK_SIZE_M)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_N; j += 1){
                    b_frag[j][i] = Bs[OFFSET(k+i, tx + vBLOCK_SIZE_N * j, BLOCK_SIZE_N)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_N; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j++){
                    #pragma unroll
                    for(int k_in = 0; k_in < THREAD_SIZE_K; k_in++){
                        // accum[i][j] = fma(a_frag[j][k_in], b_frag[i][k_in], accum[i][j]);
                        accum[i][j] += a_frag[j][k_in] * b_frag[i][k_in];
                    }
                }
            }
        }

        __syncthreads();
    }


    #pragma unroll
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){
            C[OFFSET(
                BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N,
                N
            )] = (accum[thread_x][thread_y]);
        }
    }
}

void matmul_sdd_32_32(int * csr_row, int *csr_col, float * csr_value, float *B, float *C, int M, int K, int N, int block_h,int block_w)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_K = 4;
    const int THREAD_SIZE_N = 4;
    assert(BLOCK_SIZE_M==block_h);
    assert(BLOCK_SIZE_K==block_w);
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);
    BLOCK_SPARSE_MATMUL_SDD<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<gridDim, blockDim>>>(csr_row, csr_col, csr_value, B, C, M, K, N, block_h, block_w);
}

bool verify_matmul_sdd(float * A, float * B, int M, int N)
{
    bool flag = true;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            int index = i * N + j;
            if(fabs(A[index]-B[index])>0.001){
                printf("%d %d : %f %f\n", i, j, A[index], B[index]);
                flag=false;
            }
        }
    }
    return flag;
}
int main()
{
    cudaEvent_t time_start, time_end;
    float msecTotal=0;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    int M=1024, K=1024, N=1024;
    int block_h=32, block_w = 32;
    // float sparsiy=0.9999;
    float sparsiy=0.95;
    float * data = (float*) malloc(sizeof(float)*M*K);
    int * mask = (int*) malloc(sizeof(int)*M*N);
    int * row = (int*) malloc(sizeof(int)*(M+1));
    int * col = (int*) malloc(sizeof(int)*M*K);
    float * values = (float*)malloc(sizeof(float)*M*N);
    float * A = (float*)malloc(sizeof(float)*M*K);
    float * B = (float*)malloc(sizeof(float)*K*N);
    float * C = (float*)malloc(sizeof(float)*M*N);
    float * ref_C = (float*) malloc(sizeof(float)*M*N);
    float * out = (float*) malloc(sizeof(float)*M*N);
    float * ref_out = (float*) malloc(sizeof(float)*M*N);
    float *dA, *dB, *dC;
    // init(data, M*N, 0);
    init_mask(mask, M*K, 0.95);
    init(A, M*K, 0);
    init(B, N*K, 0);
    // memset(A,0,sizeof(float)*M*K);
    // memset(B,0,sizeof(float)*N*K);
    // for(int i=0;i<K;i++)
    //     A[i] = 1, B[i*N]=1;
    for(int i=0;i<M*K;i++){
        if (mask[i] == 0)
            A[i] = 0.0;    // B[i*N] = 1;
    }

    int * d_mask, *d_row, *d_col, *extra_buffer, *d_row_pos;
    float * d_data, *d_val, *d_out;
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(float)*N*K));
    CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(float)*M*N));
    CUDA_SAFE_CALL(cudaMalloc(&d_out, sizeof(float)*M*N));
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int)*(M+1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_pos, sizeof(int)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&extra_buffer, sizeof(int)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&d_data, sizeof(float)*M*K));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float)*M*K));
    CUDA_SAFE_CALL(cudaMemcpy(d_data, data, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaEventRecord(time_start));
    for(int runtime=0; runtime<10; runtime++){
        convert_bcsr(d_mask, dA, M, K, block_h, block_w, d_row, d_col, d_row_pos, d_val, extra_buffer);
        int n_row = M / block_h;
        int SMALL_BLOCK_NUM;
        CUDA_SAFE_CALL(cudaMemcpy(&SMALL_BLOCK_NUM, d_row + n_row, sizeof(int), cudaMemcpyDeviceToHost));
        const int SPARSE_VAL_SIZE = SMALL_BLOCK_NUM * block_h * block_w;
        matmul_sdd_32_32(d_row, d_col, d_val, dB, dC, M, K, N, block_h, block_w);
    }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));

    CUDA_SAFE_CALL(cudaMemcpy(row, d_row, sizeof(int)*(M+1), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(col, d_col, sizeof(int)*M*K, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(values, d_val, sizeof(int)*M*K, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(C, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    if (!verify_bcsr(mask, A, M, N, block_h, block_w, row, col, values)){
        printf("Convert check failed!!!\n");
        return -1;
    }
    calculate_ref_nn(A, B, ref_C, M ,K, N);

    if(!verify_matmul_sdd(ref_C, C, M, N)){
        printf("value error!\n");
    }
    return 0;
}
// int main()
// {
//     cudaEvent_t time_start, time_end;
//     float msecTotal=0;
//     CUDA_SAFE_CALL(cudaEventCreate(&time_start));
//     CUDA_SAFE_CALL(cudaEventCreate(&time_end));
//     int M=1024, K=1024, N=1024;
//     int block_h=32, block_w = 32;
//     // float sparsiy=0.9999;
//     float sparsiy=0.95;
//     float * data = (float*) malloc(sizeof(float)*M*K);
//     int * mask = (int*) malloc(sizeof(int)*M*N);
//     int * row = (int*) malloc(sizeof(int)*(M+1));
//     int * col = (int*) malloc(sizeof(int)*M*K);
//     float * values = (float*)malloc(sizeof(float)*M*N);
//     float * A = (float*)malloc(sizeof(float)*M*K);
//     float * B = (float*)malloc(sizeof(float)*K*N);
//     float * C = (float*)malloc(sizeof(float)*M*N);
//     float * ref_C = (float*) malloc(sizeof(float)*M*N);
//     float * out = (float*) malloc(sizeof(float)*M*N);
//     float * ref_out = (float*) malloc(sizeof(float)*M*N);
//     float *dA, *dB, *dC;
//     init_mask(mask, M*N , sparsiy);
//     for(int i=0;i<M*N; i++)
//         data[i] = float(mask[i]);
//     // init(data, M*N, 0);
//     init(A, M*K, 0.95);
//     init(B, N*K, 0.95);
//     // for(int i=0;i<K;i++){
//     //     A[i] = 1;
//     //     B[i*N] = 1;
//     // }
//     int * d_mask, *d_row, *d_col, *extra_buffer, *d_row_pos;
//     float * d_data, *d_val, *d_out;
//     CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float)*M*K));
//     CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(float)*N*K));
//     CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(float)*M*N));
//     CUDA_SAFE_CALL(cudaMalloc(&d_out, sizeof(float)*M*N));
//     CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int)*M*K));
//     CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int)*(M+1)));
//     CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int)*M*K));
//     CUDA_SAFE_CALL(cudaMalloc(&d_row_pos, sizeof(int)*M*K));
//     CUDA_SAFE_CALL(cudaMalloc(&extra_buffer, sizeof(int)*M*K));
//     CUDA_SAFE_CALL(cudaMalloc(&d_data, sizeof(float)*M*K));
//     CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float)*M*K));
//     CUDA_SAFE_CALL(cudaMemcpy(d_data, data, sizeof(float)*M*K, cudaMemcpyHostToDevice));
//     CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int)*M*K, cudaMemcpyHostToDevice));
//     CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice));
//     CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice));

//     CUDA_SAFE_CALL(cudaEventRecord(time_start));
//     for(int runtime=0; runtime<10; runtime++){
//         convert_bcsr(d_mask, d_data, M, N, block_h, block_w, d_row, d_col, d_row_pos, d_val, extra_buffer);
//         int n_row = M / block_h;
//         int SMALL_BLOCK_NUM;
//         CUDA_SAFE_CALL(cudaMemcpy(&SMALL_BLOCK_NUM, d_row + n_row, sizeof(int), cudaMemcpyDeviceToHost));
//         const int SPARSE_VAL_SIZE = SMALL_BLOCK_NUM * block_h * block_w;
//         matmul_sparse_out_32_64_32(dA, dB, dC, d_row, d_col, d_row_pos, M, K, N, block_h, block_w, SPARSE_VAL_SIZE, SMALL_BLOCK_NUM);
//         sparse_softmax_v1(dC, d_val, d_row, d_col, d_row_pos, M, N, block_h, block_w, SPARSE_VAL_SIZE);
//         matmul_sdd_32_32(d_row, d_col, dC, dB, d_out, M, K, N, block_h, block_w);
//     }
//     CUDA_SAFE_CALL(cudaEventRecord(time_end));
//     CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
//     CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));

//     CUDA_SAFE_CALL(cudaMemcpy(row, d_row, sizeof(int)*(M+1), cudaMemcpyDeviceToHost));
//     CUDA_SAFE_CALL(cudaMemcpy(col, d_col, sizeof(int)*M*K, cudaMemcpyDeviceToHost));
//     CUDA_SAFE_CALL(cudaMemcpy(values, d_val, sizeof(int)*M*K, cudaMemcpyDeviceToHost));
//     CUDA_SAFE_CALL(cudaMemcpy(out, d_out, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
//     // if (!verify_bcsr(mask, data, M, N, block_h, block_w, row, col, values)){
//     //     printf("Convert check failed!!!\n");
//     //     return -1;
//     // }
//     calculate_ref(A, B, ref_C,M ,K, N);

//     calculate_softmax_ref(ref_C, mask, M, N);

//     calculate_ref(ref_C, B, ref_out, M, K, N);
    
//     if(!verify_matmul_sdd(ref_out, out, M,N)){
//         printf("value error!\n");
//     }
//     return 0;
// }