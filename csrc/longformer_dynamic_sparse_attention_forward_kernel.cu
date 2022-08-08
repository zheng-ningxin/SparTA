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

__global__ void BLOCK_SPARSE_MATMUL_OUT_32_64_32(
    float *A,
    float *B,
    float *C_val,
    int *row_index,
    int *col_index,
    int GLOBAL_M,
    int GLOBAL_K,
    int GLOBAL_N,
    int SPARSE_VAL_SIZE)
{
    /*
    description:
    tiling k dimension
    smm_dd_s_nn: sparse matmul, dense (MxK, along K) x dense (KxN, along N) -> sparse (MxN, along N)
    the output sparse is block size 32x32, the blocks will be written to bcsr 32x64
    */
    
    const int BLOCK_SIZE_M = 32; // 64
    const int BLOCK_SIZE_K = 64; // 8
    const int BLOCK_SIZE_N = 32; // 128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;

    A += M * K * blockIdx.y;
    B += K * N * blockIdx.y;
    C_val += SPARSE_VAL_SIZE * blockIdx.y;
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("blockIdx.x:%d blockIdx.y:%d\n", blockIdx.x, blockIdx.y);
    // }
    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char *bShare = (char *)fShare;

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
    asm("mov.b32 %0, %0;"
        : "+r"(storAB)
        :);
    asm("mov.b32 %0, %0;"
        : "+r"(loadA)
        :);
    asm("mov.b32 %0, %0;"
        : "+r"(loadB)
        :);

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    for (int k_seq = 0; k_seq < (int)(K / 64); k_seq++)
    {
        uint offsetA00 = ori_offsetA00 + 64 * k_seq;
        uint offsetA16 = ori_offsetA16 + 64 * k_seq;
        uint offsetB00 = ori_offsetB00 + 64 * k_seq;
        uint offsetB16 = ori_offsetB16 + 64 * k_seq;

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        a00 = __ldg((const float4 *)(add_ptr_f(A, offsetA00)));
        a16 = __ldg((const float4 *)(add_ptr_f(A, offsetA16)));
        b00 = __ldg((const float4 *)(add_ptr_f(B, offsetB00)));
        b16 = __ldg((const float4 *)(add_ptr_f(B, offsetB16)));

        __syncthreads();

        *(float *)&bShare[storAB + (0 * 32 + 0 + 0 * 65 * 32) * 4] = a00.x;
        *(float *)&bShare[storAB + (1 * 32 + 0 + 0 * 65 * 32) * 4] = a00.y;
        *(float *)&bShare[storAB + (2 * 32 + 0 + 0 * 65 * 32) * 4] = a00.z;
        *(float *)&bShare[storAB + (3 * 32 + 0 + 0 * 65 * 32) * 4] = a00.w;
        *(float *)&bShare[storAB + (0 * 32 + 16 + 0 * 65 * 32) * 4] = a16.x;
        *(float *)&bShare[storAB + (1 * 32 + 16 + 0 * 65 * 32) * 4] = a16.y;
        *(float *)&bShare[storAB + (2 * 32 + 16 + 0 * 65 * 32) * 4] = a16.z;
        *(float *)&bShare[storAB + (3 * 32 + 16 + 0 * 65 * 32) * 4] = a16.w;

        *(float *)&bShare[storAB + (0 * 32 + 0 + 1 * 65 * 32) * 4] = b00.x;
        *(float *)&bShare[storAB + (1 * 32 + 0 + 1 * 65 * 32) * 4] = b00.y;
        *(float *)&bShare[storAB + (2 * 32 + 0 + 1 * 65 * 32) * 4] = b00.z;
        *(float *)&bShare[storAB + (3 * 32 + 0 + 1 * 65 * 32) * 4] = b00.w;
        *(float *)&bShare[storAB + (0 * 32 + 16 + 1 * 65 * 32) * 4] = b16.x;
        *(float *)&bShare[storAB + (1 * 32 + 16 + 1 * 65 * 32) * 4] = b16.y;
        *(float *)&bShare[storAB + (2 * 32 + 16 + 1 * 65 * 32) * 4] = b16.z;
        *(float *)&bShare[storAB + (3 * 32 + 16 + 1 * 65 * 32) * 4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4 *)&regA[0] = *(float4 *)&bShare[loadA + (32 * j + 0) * 4];
            *(float4 *)&regA[4] = *(float4 *)&bShare[loadA + (32 * j + 16) * 4];
            *(float4 *)&regB[0] = *(float4 *)&bShare[loadB + (32 * j + 65 * 32) * 4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
#pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2 *)&regA[0] = *(float2 *)&bShare[loadA + (32 * j + 0 + (j / 4) * 2) * 4];
            *(float2 *)&regA[2] = *(float2 *)&bShare[loadA + (32 * j + 2 + (j / 4) * 2) * 4];
            *(float2 *)&regA[4] = *(float2 *)&bShare[loadA + (32 * j + 16 + (j / 4) * 2) * 4];
            *(float2 *)&regA[6] = *(float2 *)&bShare[loadA + (32 * j + 18 + (j / 4) * 2) * 4];
            *(float2 *)&regB[0] = *(float2 *)&bShare[loadB + (32 * j + 0 + (j / 4) * 2 + 65 * 32) * 4];
            *(float2 *)&regB[2] = *(float2 *)&bShare[loadB + (32 * j + 2 + (j / 4) * 2 + 65 * 32) * 4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
    }

    asm volatile("mov.u32 %0, %tid.x;"
                 : "=r"(tid)
                 :);
    asm volatile("mov.u32 %0, %ctaid.x;"
                 : "=r"(bx)
                 :);
    asm volatile("mov.u32 %0, %ctaid.y;"
                 : "=r"(by)
                 :);

    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty * 32 * 8 * 4 + tx * 4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty * 32 * 8 + tx * 2 + ((tid & 192) >> 2);

    // uint blk_index = block_index[blockIdx.x] / 2;
    uint blk_index = blockIdx.x;
    // uint intra_blk_index = block_index[blockIdx.x] % 2;
    // C_val += 32 * 32 * blk_index;
    // C_val += ty * 32 + tx * 2;
    // C_val += ((blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_M + ty) * GLOBAL_N + (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_N + tx * 2;
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("blockIdx.x:%d blockIdx.y:%d\n", blockIdx.x, blockIdx.y);
    // }
    C_val += (row_index[blockIdx.x]  * BLOCK_SIZE_M + ty) * GLOBAL_N + col_index[blockIdx.x] * BLOCK_SIZE_N + tx * 2;


    __syncthreads();
    *(float4 *)&fShare[storC + 0 * 32 * 8] = *(float4 *)regC[0];
    *(float4 *)&fShare[storC + 1 * 32 * 8] = *(float4 *)regC[1];
    *(float4 *)&fShare[storC + 2 * 32 * 8] = *(float4 *)regC[2];
    *(float4 *)&fShare[storC + 3 * 32 * 8] = *(float4 *)regC[3];
    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2 *)&fShare[readC + i * 32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i + j]);

    //-> store((bhalf2*)C, c2[0]);
    *(float2 *)C_val = c2[0];

    __syncthreads();
    *(float4 *)&fShare[storC + 0 * 32 * 8] = *(float4 *)regC[4];
    *(float4 *)&fShare[storC + 1 * 32 * 8] = *(float4 *)regC[5];
    *(float4 *)&fShare[storC + 2 * 32 * 8] = *(float4 *)regC[6];
    *(float4 *)&fShare[storC + 3 * 32 * 8] = *(float4 *)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2 *)&fShare[readC + i * 32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i + j]);

    // C_val += 16 * 32;
    C_val += 16 * GLOBAL_N;

    *(float2 *)C_val = c2[0];
}

void batch_matmul_block_sparse_out_kernel_launch(
    float *A,
    float *W,
    int *row_pos,
    int *col,
    float *output,
    int M, int K, int N,
    int block_h,
    int block_w,
    int block_nnz,
    int batch_size,
    int head_num)
{
    const dim3 dimBlock(256);
    const dim3 dimGrid(block_nnz, head_num * batch_size);
    BLOCK_SPARSE_MATMUL_OUT_32_64_32<<<dimGrid, dimBlock>>>(A, W, output, row_pos, col, M, K, N,  M * N);
}

at::Tensor batch_matmul_block_sparse_out(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor row_pos,
    torch::Tensor col,
    torch::Tensor output,
    int block_h, int block_w, int block_nnz)
{
    cudaSetDevice(A.get_device());
    int batch_size = A.size(0);
    int head_num = A.size(1);
    int max_seq_length = A.size(2);
    int hidden_dim = A.size(3);
    // printf("block nnz: %d \n", block_nnz);
    AT_DISPATCH_FLOATING_TYPES(A.type(), "longformer_batch_matmul", ([&]
            { batch_matmul_block_sparse_out_kernel_launch(
                    A.data_ptr<float>(),
                    W.data_ptr<float>(),
                    row_pos.data_ptr<int>(),
                    col.data_ptr<int>(),
                    output.data_ptr<float>(),
                    max_seq_length,
                    hidden_dim,
                    max_seq_length,
                    block_h,
                    block_w,
                    block_nnz,
                    batch_size,
                    head_num); }));
    return output;
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
at::Tensor longformer_mixed_softmax(
    torch::Tensor A,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor val_mask,
    torch::Tensor global_attention,
    torch::Tensor extra_buffer,
    int block_h, int block_w, int block_nnz

)
{
    cudaSetDevice(A.get_device());
    int batch_size = A.size(0);
    int head_num = A.size(1);
    int M = A.size(2);
    int N = A.size(3);
    AT_DISPATCH_FLOATING_TYPES(A.type(), "longformer_mixed_softmax", ([&]
            { longformer_mixed_softmax_launch(
                    A.data_ptr<float>(),
                    row.data_ptr<int>(),
                    col.data_ptr<int>(),
                    val_mask.data_ptr<float>(),
                    global_attention.data_ptr<int>(),
                    extra_buffer.data_ptr<float>(),
                    block_h,
                    block_w,
                    block_nnz,
                    M,
                    N,
                    head_num,
                    batch_size,
                    global_attention.size(0)
                    ); }));
    return A;
}


template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_MATMUL_DSD(int* csr_row, int * csr_col, float* csr_val, float * B, float* C,  int M, int K, int N, int sparse_val_size){
    // const int BLOCK_SIZE_M = 32;
    // const int BLOCK_SIZE_K = 32;
    // const int BLOCK_SIZE_N = 64;
    // const int THREAD_SIZE_M = 4;
    // const int THREAD_SIZE_K = 4;
    // const int THREAD_SIZE_N = 4;
    int by = blockIdx.y; // M
    int bx = blockIdx.x; // N
    int bz = blockIdx.z;
    int ty = threadIdx.y; 
    int tx = threadIdx.x;
    csr_val = csr_val + sparse_val_size * bz;
    B = B + K * N * bz;
    C = C + M * N * bz;

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
                FETCH_FLOAT4(csr_val[OFFSET(k + A_BLOCK_ROW_START + by*BLOCK_SIZE_M, A_BLOCK_COL_START + col_pos, K)]);
                // FETCH_FLOAT4(csr_val[tile_block_idx * BLOCK_SIZE_M * BLOCK_SIZE_K + OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]);
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


void batch_matmul_block_sparse_kernel_launch(
    float * A,
    float * B,
    float * C,
    int * row_ptr,
    int * col_idx,
    int M,
    int K,
    int N,
    int head_num,
    int batch_size,
    int block_h,
    int block_w

)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_K = 4;
    const int THREAD_SIZE_N = 4;
    assert(block_h==BLOCK_SIZE_M);
    assert(block_w==BLOCK_SIZE_K);
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, head_num * batch_size);
    dim3 blockDim(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);
    BLOCK_SPARSE_MATMUL_DSD<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<gridDim, blockDim>>>(
        row_ptr,
        col_idx,
        A,
        B,
        C,
        M,
        K,
        N,
        M*K);

}

at::Tensor batch_matmul_block_sparse(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    int block_h,
    int block_w
){
    cudaSetDevice(A.get_device());
    int batch_size = A.size(0);
    int head_num = A.size(1);
    int M = A.size(2);
    int K = A.size(3);
    int N = B.size(3);
    // printf("M:%d N:%d\n", M, N);
    torch::Tensor output = torch::zeros({batch_size, head_num, M, N}, A.options());
    AT_DISPATCH_FLOATING_TYPES(A.type(), "longformer_batch_matmul_attenxV", ([&]
            { batch_matmul_block_sparse_kernel_launch(
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                output.data_ptr<float>(),
                row_ptr.data_ptr<int>(),
                col_idx.data_ptr<int>(),
                M,
                K,
                N,
                head_num,
                batch_size,
                block_h,
                block_w); }));
    return output;
}
