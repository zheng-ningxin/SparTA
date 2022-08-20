
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
using namespace std;
// Macro definition for the cuda and cusparse

#include <assert.h>
// CUDA runtime
#include <cuda.h>
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
    float regMax = -100000.0;
    int block_seq_start = row_index[blk_row_idx];
    int block_seq_end = row_index[blk_row_idx+1];

    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        uint index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        regMax = max(regMax, C_val[index] * C_val_mask[index]);
        // regMax = max(regMax, C_val[index]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        regMax = max(regMax, __shfl_down_sync(FULL_MASK, regMax, offset));
    }
    regMax = __shfl_sync(FULL_MASK, regMax, 0);
    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        uint index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        // regC = (float)C_val_mask[index]*C_val[index];
        if (C_val_mask[index] != 0) {
            regC = expf(C_val[index]-regMax);
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
        uint index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + bn;
        regC = 0.0f;
        // if(blk_row_idx==0 && bm == 0 && block_inter_row==0 && blockIdx.y==0){
        //     printf("expf: %f regMax:%f regSum:%f\n", C_val[index], regMax, regSum);
        // }
        if (C_val_mask[index] > 0) {
            C_val[index] = expf(C_val[index]-regMax)/regSum;
        }
        else{
            C_val[index] = 0;
        }

    }


}
template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_MATMUL_SDD(int* csr_row, int * csr_col, float* csr_val, float * B, float* C,  int M, int K, int N, int block_h, int block_w, int sparse_val_size){
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


void dynamic_forward_function(float* Q, float* K, float* V,
                    float * inter_result, int * row_ptr, int * col_idx, int * row_pos, float * val_mask,
                    int batch_size, int head_num, int q_seq_length, int k_seq_length, int hidden_dim, const int block_nnz, float* output)
{
    const int sparse_val_size =  block_nnz * 32* 32 ; //block_nnz * block_h * block_w
    CUDA_SAFE_CALL(cudaMemset(inter_result, 0, sizeof(float) * sparse_val_size * batch_size * head_num));
    // already set to zero outside, no need to memset here
    //cudaMemset((void*)val, 0, (SPARSE_VAL_SIZE * HEAD_NUM) * batch_size);
    const dim3 dimBlock(256);
    const dim3 dimGrid(block_nnz, head_num * batch_size);
    BLOCK_SPARSE_MATMUL_OUT_32_64_32<<<dimGrid, dimBlock>>>(
        Q,
        K,
        inter_result,
        row_pos,
        col_idx,
        q_seq_length, // M
        hidden_dim, // K
        k_seq_length, // N
        sparse_val_size
        
    );

    const int row_tile = 4;
    const dim3 softmax_dimBlock(row_tile*32);
    const dim3 softmax_dimGrid(q_seq_length/row_tile, head_num * batch_size);
    SPARSE_SOFTMAX<<<softmax_dimGrid, softmax_dimBlock>>>(
        inter_result,
        val_mask,
        row_ptr,
        32,
        32,
        sparse_val_size,
        row_tile);

    // sparse x dense
    // M: seq_length K: seq_length N:hidden dim
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_K = 4;
    const int THREAD_SIZE_N = 4;

    dim3 sdd_gridDim(hidden_dim/BLOCK_SIZE_N, q_seq_length/BLOCK_SIZE_M, head_num * batch_size);
    dim3 sdd_blockDim(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);
    BLOCK_SPARSE_MATMUL_SDD<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<sdd_gridDim, sdd_blockDim>>>(
        row_ptr,
        col_idx,
        inter_result,
        V,
        output,
        q_seq_length,
        k_seq_length,
        hidden_dim,
        32,
        32,
        sparse_val_size);

    
}



at::Tensor dynamic_sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_pos,
    torch::Tensor val_mask,
    int block_nnz,
    int head_num
)
{
    cudaSetDevice(Q.get_device());
    // Q, K, V should have the same shape which is {batchsize, seq_length, hidden_dim}
    int batch_size = Q.size(0);
    // int head_num = Q.size(1);
    int q_seq_length = Q.size(2);
    int k_seq_length = K.size(2); // the sequence length of query and key may be different
    int hidden_dim = Q.size(3);
    torch::Tensor output = torch::empty({batch_size, head_num, q_seq_length, hidden_dim}, Q.options());
    
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "dynamic_sparse_attention", ([&]
                            { dynamic_forward_function(
                                    Q.data_ptr<float>(),
                                    K.data_ptr<float>(),
                                    V.data_ptr<float>(),
                                    inter_result.data_ptr<float>(),
                                    row_ptr.data_ptr<int>(),
                                    col_idx.data_ptr<int>(),
                                    row_pos.data_ptr<int>(),
                                    val_mask.data_ptr<float>(),
                                    batch_size,
                                    head_num,
                                    q_seq_length,
                                    k_seq_length,
                                    hidden_dim,
                                    block_nnz,
                                    output.data_ptr<float>()
                                ); }));
    return output;
}
template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BATCH_MATMUL_SDD_TN(int* csr_row, int * csr_col, int * block_index, float* csr_val, float * B, float* C,  int M, int K, int N, int block_h, int block_w, int sparse_val_size)
{

    // block_index store the index of the block in the csr_val
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
        int block_shift = block_index[csr_col[tile_block_idx] * gridDim.y + by];
        if(threadIdx.x==0 && threadIdx.y==0 && bz==0){
            printf("bx:%d by:%d col_pos:%d block_shift:%d\n gridDim:(%d, %d) tile_block_idx:%d", bx, by, csr_col[tile_block_idx], block_shift, gridDim.x, gridDim.y, tile_block_idx);
        }
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                FETCH_FLOAT4(csr_val[block_shift * BLOCK_SIZE_M * BLOCK_SIZE_K + OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]);
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
                    // a_frag[j][i] = As[OFFSET(ty + vBLOCK_SIZE_M * j, k+i, BLOCK_SIZE_K)];
                    a_frag[j][i] = As[OFFSET(k+i, ty + vBLOCK_SIZE_M * j, BLOCK_SIZE_M)];
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

void dynamic_backward_function( float* grad_out,
                                float* Q,
                                float* K,
                                float* V,
                                float* inter_result,
                                int * row_ptr,
                                int * col_idx,
                                int * row_pos,
                                float * val_mask,
                                int * block_index,
                                int * grad_row_ptr,
                                int * grad_col_idx,

                                float* Q_grad,
                                float* K_grad,
                                float* V_grad,
                                float* Attn_grad,
                                float* Score_grad,
                                int batchsize,
                                int head_num,
                                int hidden_dim,
                                int q_seq_len,
                                int k_seq_len,
                                int block_h,
                                int block_w,
                                int sparse_val_size
                                )
{
    // grad_v: Batch x Head x k_seq_len, hidden_dim
    // Grad_V = Attn^T x grad_out
    // Attn: q_seq_len, k_seq_len
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_K = 4;
    const int THREAD_SIZE_N = 4;

    dim3 gradv_dimGrid(hidden_dim/BLOCK_SIZE_N, k_seq_len/BLOCK_SIZE_M, head_num * batchsize);
    dim3 gradv_dimBlock(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);
    // const dim3 gradv_dimBlock(256);
    // const dim3 gradv_dimGrid(hidden_dim/32, k_seq_len/32, head_num*batchsize);
    BATCH_MATMUL_SDD_TN<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<gradv_dimGrid, gradv_dimBlock>>>(grad_row_ptr, grad_col_idx, block_index, inter_result, grad_out, V_grad, k_seq_len, q_seq_len, hidden_dim, 32, 32, sparse_val_size);


}

std::vector<at::Tensor> dynamic_sparse_attention_backward(
    torch::Tensor grad,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor row_pos,
    torch::Tensor val_mask,
    torch::Tensor block_index,
    torch::Tensor grad_row_idx,
    torch::Tensor grad_col_idx,
    int block_nnz

    )
{
    /*
        Compute the gradient of the Q, K, V.
        A * B = C
            |  backward()
            V
        Grad_A = Grad_C * B^T
        Grad_B = A^T * Grad_C

        Score = Q x K^T
        Attn = softmax(Attn)
        Out = Attn x V
    The gradient calculation formulars are as following:
    
    Grad_V = Attn^T x grad_out
    Grad_Attn = grad_out x V^T
    Grad_Score = grad_soft(Grad_Attn)
    Grad_Q = Grad_Score x K^T
    Grad_K = Q^T * Grad_Score

    // Grad_V = Sparse*dense kernel: (row_ptr, col_idx, val)^T * in_grad 
    // Grad_val = output sparse kernel(sparse index row_ptr, col_idx, val): in_grad * V^T 
    // Grad_Q = Grad_softmax * K^T
    // Grad_K = Q^T * Grad_softmax
    */
    const int sparse_val_size =  block_nnz * 32* 32;
    cudaSetDevice(Q.get_device());
    // Q, K, V should have the same shape which is {batchsize, seq_length, hidden_dim}
    int batch_size = Q.size(0);
    int head_num = Q.size(1);
    int q_seq_length = Q.size(2);
    int k_seq_length = K.size(2); // the sequence length of query and key may be different
    // K, V should have the same sequence length
    int hidden_dim = Q.size(3);
    torch::Tensor Q_grad = torch::empty_like(Q);
    torch::Tensor K_grad = torch::empty_like(K);
    torch::Tensor V_grad = torch::empty_like(V);
    torch::Tensor Attn_grad = torch::empty_like(inter_result);
    torch::Tensor Score_grad = torch::empty_like(inter_result);
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "our_sparse_attention", ([&]
    { dynamic_backward_function(
            grad.data_ptr<float>(),
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            inter_result.data_ptr<float>(),
            row_ptr.data_ptr<int>(),
            col_idx.data_ptr<int>(),
            row_pos.data_ptr<int>(),
            val_mask.data_ptr<float>(),
            block_index.data_ptr<int>(),
            grad_row_idx.data_ptr<int>(),
            grad_col_idx.data_ptr<int>(),
            
            Q_grad.data_ptr<float>(),
            K_grad.data_ptr<float>(),
            V_grad.data_ptr<float>(),
            Attn_grad.data_ptr<float>(),
            Score_grad.data_ptr<float>(),
            batch_size,
            head_num,
            hidden_dim,
            q_seq_length,
            k_seq_length,
            32,
            32,
            sparse_val_size
        ); }));
    return vector<at::Tensor>({Q_grad, K_grad, V_grad, Attn_grad, Score_grad});
}
