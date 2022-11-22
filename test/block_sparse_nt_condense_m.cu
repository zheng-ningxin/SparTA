
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
// CUDA runtime
#include <cuda.h>
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


template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_NT_CONDENSE(float* A, float * weight, int * csr_row, int * csr_col, float* C, int M, int K, int N){
    
   
    // A and Weight tensor are stored in the dense format
    // bx-> K by->M
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    const int padding = 1;
    __shared__ float As[BLOCK_SIZE_M * (BLOCK_SIZE_K + padding)];
    __shared__ float Bs[BLOCK_SIZE_N * (BLOCK_SIZE_K + padding)];
    __shared__ int m_index[BLOCK_SIZE_M];
    float accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    float a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    float b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_THREAD_PER_ROW = BLOCK_SIZE_K / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;
    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;

    // int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4 + BLOCK_SIZE_K * bx;
    // int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4 + BLOCK_SIZE_K * bx;

    int index_start = csr_row[bx] + by * BLOCK_SIZE_M; 
    int index_end = min(csr_row[bx+1], index_start + BLOCK_SIZE_M);
    
    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;
    float4 tmp_float4;
    float4 const0 = {0,0,0,0};
    if(index_start < index_end){
        // __syncthreads();

        if(tid<index_end-index_start)
            m_index[tid] = csr_col[tid+index_start];
        __syncthreads();
        for(int block_n_id=0; block_n_id < N/BLOCK_SIZE_N; block_n_id++){
            #pragma unroll
            for(int i=0;i<THREAD_SIZE_N;i++)
                for(int j=0;j<THREAD_SIZE_M;j++)
                accum[i][j] = 0;
            #pragma unroll
            for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
                // FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                if(k + A_BLOCK_ROW_START < index_end-index_start){
                    tmp_float4 = FETCH_FLOAT4(A[OFFSET(m_index[k+A_BLOCK_ROW_START], A_BLOCK_COL_START + BLOCK_SIZE_K * bx, K)]);
                }
                else{
                    tmp_float4 = const0;
                }
                FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) = tmp_float4;
            }

            #pragma unroll
            for(int k=0; k < BLOCK_SIZE_N; k+= B_TILE_ROW_STRIDE){
                // transpose here
                tmp_float4 = FETCH_FLOAT4(weight[OFFSET(block_n_id * BLOCK_SIZE_N + k + B_BLOCK_ROW_START, B_BLOCK_COL_START + BLOCK_SIZE_K * bx, K)]);
                // tmp_float4 =  FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_K + B_BLOCK_COL_START]);
                Bs[OFFSET(B_BLOCK_COL_START, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.x;
                Bs[OFFSET(B_BLOCK_COL_START+1, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.y;
                Bs[OFFSET(B_BLOCK_COL_START+2, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.z;
                Bs[OFFSET(B_BLOCK_COL_START+3, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.w;
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
            // Write back to the correponding position
            // Write in the normal way
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
                #pragma unroll
                for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){
                    // C[OFFSET(
                    //     BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                    //     BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N,
                    //     N
                    // )] = (accum[thread_x][thread_y]);
                    // if(by==0 && bx==0 && tid==0){
                    //     printf("bx:%d by:%d block_n_id:%d accum[tx, ty]:%f C_OFFSET:%d\n", bx, by, block_n_id, accum[thread_x][thread_y], OFFSET(
                    //     BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                    //     BLOCK_SIZE_N * block_n_id + tx + thread_x * vBLOCK_SIZE_N,
                    //     N));
                    // }
                    // if(ty + thread_y * vBLOCK_SIZE_M>=index_end-index_start){
                    //     printf("m_index offset:%d index_end-index_start:%d \n", ty + thread_y * vBLOCK_SIZE_M, index_end-index_start);
                    // }
                    // atomicAdd(C+OFFSET(
                    //     BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                    //     BLOCK_SIZE_N * block_n_id + tx + thread_x * vBLOCK_SIZE_N,
                    //     N),
                    //     accum[thread_x][thread_y]);
                    if(ty + thread_y * vBLOCK_SIZE_M < index_end-index_start)
                        atomicAdd(C+OFFSET(
                            m_index[ty + thread_y * vBLOCK_SIZE_M],
                            BLOCK_SIZE_N * block_n_id + tx + thread_x * vBLOCK_SIZE_N,
                            N),
                            accum[thread_x][thread_y]);
                        
                }
            }
            // #pragma unroll
            // for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){        
            //     #pragma unroll
            //     for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){

            //         if(ty + thread_y * vBLOCK_SIZE_M < index_end-index_start)
            //             atomicAdd(C+OFFSET(
            //                 BLOCK_SIZE_N * block_n_id + tx + thread_x * vBLOCK_SIZE_N,
            //                 m_index[ty + thread_y * vBLOCK_SIZE_M],
            //                 M),
            //                 accum[thread_x][thread_y]);
                        
            //     }
            // }
        }
    }

}


void init_mask_blockwise(int * ptr, size_t M, size_t N, int block_h, int block_w, float sparsity)
{
    int m_block_n = M / block_h;
    int n_block_n = N / block_w;
    int block_nnz = 0;
    for (int i = 0; i < m_block_n; i++)
    {
        for(int j=0; j < n_block_n; j++){
            float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            int pos = i*block_h*N+j*block_w;
            if (pro < sparsity)
            {
                ptr[pos] = 0;
            }
            else
            {
                ptr[pos] = 1;
                block_nnz++;
            }
        }
        
    }
    printf("random %d blocks in init_mask_blockwise\n", block_nnz);

}


void convert_bcsr_condense_m(int * mask, float* dense_val, int M, int N, int block_h, int block_w, int *row_ptr, int* col_idx, float * val )
{
    vector<int> vr(N/block_w+1, 0);
    vector<vector<int>> vc(M/block_h, vector<int>());
    int block_nnz = 0;
    assert(M%block_h==0);
    assert(N%block_w==0);
    for(int cid=0; cid<N/block_w; cid++){
        for(int rid=0; rid<M/block_h; rid++){
            int flag = 0;
            for(int i=0; i<block_h; i++){
                for(int j=0; j<block_w; j++){
                    int _pos = (rid * block_h + i) * N + cid * block_w + j;
                    if(mask[_pos]>0)
                        flag = 1;
                }
            }
            if(flag){
                vc[cid].push_back(rid);
            }
        }
    }
    row_ptr[0]=0;
    for(int i=0;i<N/block_w;i++){
        row_ptr[i+1] = row_ptr[i] + vc[i].size();
        for(int j =0; j<vc[i].size(); j++){
            int _block_idx = row_ptr[i]+j;
            col_idx[_block_idx] = vc[i][j];
            for(int b_i=0; b_i<block_h; b_i++){
                for(int b_j=0; b_j<block_w; b_j++){
                    int pos_1 = _block_idx * block_h *block_w + b_i * block_w + b_j;
                    int pos_2 = (b_i + vc[i][j] * block_h) * N + (b_j + block_w * i);
                    val[pos_1] = dense_val[pos_2];
                }
            }
        }
    }

}

int main()
{
    int M, K, N;
    M = 3072;
    K = 768;
    N = 4096;
    const int n_iter = 1000;
    float sparsity_ratio = 0.6;
    const int A_BLOCK_SIZE_M = 16;
    const int A_BLOCK_SIZE_K = 32;
    const int A_BLOCK_SIZE_N = 128;
    const int A_THREAD_SIZE_M = 4;
    const int A_THREAD_SIZE_K = 4;
    const int A_THREAD_SIZE_N = 4;
    const int BLOCK_H = 1;
    const int BLOCK_W = 32;
    // const int BLOCK_W = 1;
    cudaEvent_t time_start, time_end;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    float msecTotal = 0;
    float * A, *B, *C, *val;
    float * dA, *dB, *dC, *d_val;
    int * mask, *d_mask, *row, *d_row, *row_pos, *d_row_pos, *col, *d_col, *d_extra_buffer;
    A = (float*) malloc(sizeof(float) * M * K);
    B = (float*) malloc(sizeof(float) * K * N);
    C = (float*) malloc(sizeof(float) * M * N);
    mask = (int*) malloc(sizeof(int) * M * K);
    row = (int*) malloc(sizeof(int) * (K+1));
    col = (int*) malloc(sizeof(int) *  M * K / BLOCK_H / BLOCK_W);
    val = (float*) malloc(sizeof(float) * M * K);
    init_mask_blockwise(mask, M, K, BLOCK_H, BLOCK_W, sparsity_ratio);
    // apply mask
    for(int i=0; i< M*K; i++){
        A[i] = A[i] * mask[i];
    }
    convert_bcsr_condense_m(mask, A, M, K, BLOCK_H, BLOCK_W, row, col, val);
    int block_nnz = row[K/BLOCK_W];
    int sparse_val_size = (block_nnz * BLOCK_H * BLOCK_W);
    printf("Block NNZ: %d\n", block_nnz);
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int) * (K + 1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_pos, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(float) * N * K));
    CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_SAFE_CALL(cudaMemset(dC, 0, sizeof(float)*M*N));
    CUDA_SAFE_CALL(cudaMalloc(&d_extra_buffer, sizeof(float) * M * K));
    
    CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int) * M * K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row, row, sizeof(int)*(K+1), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col, col, sizeof(int)*M * K / BLOCK_H / BLOCK_W, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_val, val, sizeof(float) * M * K, cudaMemcpyHostToDevice));

    

    // KxM = KxN * (MxN)^T
    dim3 a_grad_grid_dim(K/A_BLOCK_SIZE_K, M/A_BLOCK_SIZE_M);
    dim3 a_grad_block_dim(A_BLOCK_SIZE_N/A_THREAD_SIZE_N, A_BLOCK_SIZE_M/A_THREAD_SIZE_M);
    printf("Test Condense-M on our block sparse template\n");
    CUDA_SAFE_CALL(cudaEventRecord(time_start));

    for(int run=0; run<n_iter; run++){
        BLOCK_SPARSE_NT_CONDENSE<A_BLOCK_SIZE_M, A_BLOCK_SIZE_K, A_BLOCK_SIZE_N, A_THREAD_SIZE_M, A_THREAD_SIZE_K, A_THREAD_SIZE_N><<<a_grad_grid_dim, a_grad_block_dim>>>(dA, dB, d_row, d_col, dC, M, K, N);
        // openai_bmm_32_64_32_condense_dim_m_launch(d_val, d_row, d_col, dB, dC, M, K, N, BLOCK_H, BLOCK_W, sparse_val_size, 1);
    }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    printf("Time Cost: %.3fms\n", msecTotal/n_iter);


    return 0;

}
