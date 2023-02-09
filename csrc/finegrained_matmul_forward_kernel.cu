#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
using namespace std;
// Macro definition for the cuda and cusparse

#include <assert.h>
// CUDA runtime
#include <cuda.h>
#define OFFSET(row, col, ld) ((row)*ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int *>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4 *>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int *>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024
#define FULL_MASK 0xffffffff
#define SOFTMAX_ROW_TILE 4
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

__device__ void warpReduce(volatile int *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__device__ __forceinline__ const int *add_ptr_u(const int *src, int offset)
{
    const int *dst;
    asm("{                       \n\t"
        ".reg .u32 lo,hi,of;     \n\t"
        "mul.lo.u32 of, %2, %3;  \n\t"
        "mov.b64    {lo,hi}, %1; \n\t"
        "add.cc.u32  lo,lo,  of; \n\t"
        "addc.u32    hi,hi,  0;  \n\t"
        "mov.b64 %0, {lo,hi};    \n\t"
        "}"
        : "=l"(dst)
        : "l"(src), "r"(offset), "r"((int)sizeof(*src)));
    return dst;
}

__device__ __forceinline__ const float *add_ptr_f(const float *src, int offset)
{
    const float *dst;
    asm("{                       \n\t"
        ".reg .u32 lo,hi,of;     \n\t"
        "mul.lo.u32 of, %2, %3;  \n\t"
        "mov.b64    {lo,hi}, %1; \n\t"
        "add.cc.u32  lo,lo,  of; \n\t"
        "addc.u32    hi,hi,  0;  \n\t"
        "mov.b64 %0, {lo,hi};    \n\t"
        "}"
        : "=l"(dst)
        : "l"(src), "r"(offset), "r"((int)sizeof(*src)));
    return dst;
}

__device__ __forceinline__ float2 _add(float2 x, float2 y)
{
    float2 res;
    res.x = x.x + y.x;
    res.y = x.y + y.y;
    return res;
}

template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void FINEGRAINED_CONDENSE_KERNEL_V3(const int* __restrict__  csr_row, const int* __restrict__  csr_col, const float* __restrict__  csr_val,  float* __restrict__  B, float* __restrict__  C, const int M, const int K, const int N){
    

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    const int padding = 1;

    __shared__ int Is[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Vs[BLOCK_SIZE_M * BLOCK_SIZE_K];

    int tx = tid % BLOCK_SIZE_N;
    const int n_thread_per_row = BLOCK_SIZE_N;
    // for(int n_round=0; n_round<N_TILE_SIZE/BLOCK_SIZE_N; n_round++){
    float sum = 0;
    int n_start = bx * BLOCK_SIZE_N;
    int m_stride = blockDim.x / BLOCK_SIZE_N;
    #pragma unroll
    for(int ty=tid/BLOCK_SIZE_N; ty<BLOCK_SIZE_M; ty+=m_stride){
        sum = 0;
        int row_id = by * BLOCK_SIZE_M + ty;
        int index_start = csr_row[row_id];
        int index_end = csr_row[row_id+1];

        #pragma unroll
        for(int k_round=0; k_round< (index_end-index_start-1+BLOCK_SIZE_K)/BLOCK_SIZE_K; k_round++){
            // load the A to the shared memory
            int k_start = index_start + k_round * BLOCK_SIZE_K;
            // int k_end = min(k_start+ BLOCK_SIZE_K, index_end);
            int k_end = k_start + BLOCK_SIZE_K;
            for(int _pos=tx+k_start; _pos<k_end; _pos+=n_thread_per_row){
                if(_pos<index_end){
                    Is[ty*BLOCK_SIZE_K + _pos-k_start] = csr_col[_pos];
                    Vs[ty*BLOCK_SIZE_K + _pos-k_start] = csr_val[_pos];
                }else{
                    Vs[ty*BLOCK_SIZE_K + _pos-k_start] = 0;
                }
            }
            __syncthreads();

            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i++){
                int k_offset = Is[ty*BLOCK_SIZE_K+i];
                sum += Vs[ty*BLOCK_SIZE_K + i]*B[OFFSET(k_offset, n_start + tx, N)];
            }

        }
        // write backto C
        C[OFFSET(row_id, n_start+tx, N)] = sum;
    }
    // }

}
void FINEGRAINED_CONDESE_V3(int *csr_row, int * csr_col, float* csr_val, float * B, float* C, int M, int K, int N)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 64;
    const int BLOCK_SIZE_K = 32;
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(512);
    FINEGRAINED_CONDENSE_KERNEL_V3<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>(csr_row, csr_col, csr_val, B, C, M, K, N);

}
