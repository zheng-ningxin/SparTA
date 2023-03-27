#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
// Macro definition for the cuda and cusparse

#include <assert.h>
// CUDA runtime
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
using namespace std;
using namespace nvcuda;
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
__global__ void BLOCK_SPARSE_MATMUL_BIAS(float* A, float* W_val, int* W_row, int* W_col, float* C, float *bias, int M, int K, int N){
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_N * BLOCK_SIZE_K];

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

    int index_start = W_row[bx], index_end = W_row[bx+1];

    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;
    float4 tmp_float4;
    for(int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1){
        int tile_idx = W_col[tile_block_idx] * BLOCK_SIZE_K;
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                FETCH_FLOAT4(A[OFFSET(by*BLOCK_SIZE_M+k+A_BLOCK_ROW_START, tile_idx+A_BLOCK_COL_START, K)]);
        }
        /*
        for(int k = 0; k < BLOCK_SIZE_K; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_M)]) = 
                FETCH_FLOAT4(A[OFFSET(tile_idx+k+A_BLOCK_ROW_START, by*BLOCK_SIZE_M+A_BLOCK_COL_START, M)]);
        }
        */

        // #pragma unroll
        // for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
        //     FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = 
        //         FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]);
        //         // FETCH_FLOAT4(B[OFFSET(tile_idx+k+B_BLOCK_ROW_START, bx*BLOCK_SIZE_N+B_BLOCK_COL_START, N)]);
        // }

        #pragma unroll
        for(int k=0; k < BLOCK_SIZE_N; k+= B_TILE_ROW_STRIDE){
            // transpose here
            tmp_float4 =  FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_K + B_BLOCK_COL_START]);
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

        __syncthreads();
    }

    float bias_local[THREAD_SIZE_N];
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        bias_local[thread_x] = bias[BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N];
    }

    #pragma unroll
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){
            C[OFFSET(
                BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N,
                N
            )] = (accum[thread_x][thread_y]) + bias_local[thread_x];
        }
    }
}



__global__ void BLOCK_SPARSE_MATMUL_BIAS_OPENAI(
    float* A,
    float* B,
    float* bias,
    int * seqlens,
    int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int batchsize, float* output){
    /*
    description:
    tiling k dimension
    smm_dd_s_nt: sparse matmul, dense (MxK, along K) x dense (NxK, along k) -> sparse (MxN, along N)
    the output sparse is block size 32x32, the blocks will be written to bcsr 32x64
    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;

    A += M * K * blockIdx.z;
    // B += K * N * blockIdx.z;
    output += M * N * blockIdx.z;
    int batchid = blockIdx.z;
    int cur_seq_len = seqlens[batchid];
    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    __shared__ float bias_share[BLOCK_SIZE_N];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    if(by * BLOCK_SIZE_M < cur_seq_len){
        // if(threadIdx.x==0 && blockIdx.z==1 && by==0 && bx==0){
        //     printf("by:%d bx:%d bz:%d\n", by, bx, blockIdx.z);
        // }
        // uint bx = n_index[blockIdx.x]; // N
        // uint by = m_index[blockIdx.x]; // M
        if(tid<BLOCK_SIZE_N){
            bias_share[tid] = bias[bx * BLOCK_SIZE_N + tid %32]; 
        }
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
        // uint intra_blk_index = block_index[blockIdx.x] % 2;
        // C_val += 32 * 64 * blk_index + intra_blk_index * 32;
        // C_val += ty * 64 + tx * 2;
        // TODO double check here!
        // if(threadIdx.x==0 && blockIdx.z==1 && by==0 && bx==0){
        //     printf("output offset: %d\n", (blockIdx.y * BLOCK_SIZE_M + ty) * N + blockIdx.x * BLOCK_SIZE_N + tx *2);
        // }

        output += (blockIdx.y * BLOCK_SIZE_M + ty) * N + blockIdx.x * BLOCK_SIZE_N + tx *2;
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
        // *(float2*)C_val = c2[0];
        *(float2*)output = _add(c2[0], *(float2*)(bias_share+tx*2));
        // if(threadIdx.x==0 && blockIdx.z==1 && by==0 && bx==0){
        //     printf("output value: %f\n", *output);
        // }

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

        output += 16 * N;
        // *(float2*)C_val = c2[0];
        *(float2*)output = _add(c2[0], *(float2*)(bias_share+tx*2));

    }
}


__global__ void BLOCK_SPARSE_MATMUL_OPENAI(
    float* A,
    float* B,
    int * seqlens,
    int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int batchsize, float* output){
    /*
    description:
    tiling k dimension
    smm_dd_s_nt: sparse matmul, dense (MxK, along K) x dense (NxK, along k) -> sparse (MxN, along N)
    the output sparse is block size 32x32, the blocks will be written to bcsr 32x64
    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;

    A += M * K * blockIdx.z;
    // B += K * N * blockIdx.z;
    output += M * N * blockIdx.z;
    int batchid = blockIdx.z;
    int cur_seq_len = seqlens[batchid];
    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    if(by * BLOCK_SIZE_M < cur_seq_len){
        // if(threadIdx.x==0 && blockIdx.z==1 && by==0 && bx==0){
        //     printf("by:%d bx:%d bz:%d\n", by, bx, blockIdx.z);
        // }
        // uint bx = n_index[blockIdx.x]; // N
        // uint by = m_index[blockIdx.x]; // M
        // if(tid<BLOCK_SIZE_N){
        //     bias_share[tid] = bias[bx * BLOCK_SIZE_N + tid %32]; 
        // }
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
        // uint intra_blk_index = block_index[blockIdx.x] % 2;
        // C_val += 32 * 64 * blk_index + intra_blk_index * 32;
        // C_val += ty * 64 + tx * 2;
        // TODO double check here!
        // if(threadIdx.x==0 && blockIdx.z==1 && by==0 && bx==0){
        //     printf("output offset: %d\n", (blockIdx.y * BLOCK_SIZE_M + ty) * N + blockIdx.x * BLOCK_SIZE_N + tx *2);
        // }

        output += (blockIdx.y * BLOCK_SIZE_M + ty) * N + blockIdx.x * BLOCK_SIZE_N + tx *2;
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
        *(float2*)output = c2[0];
        // *(float2*)output = _add(c2[0], *(float2*)(bias_share+tx*2));
        // if(threadIdx.x==0 && blockIdx.z==1 && by==0 && bx==0){
        //     printf("output value: %f\n", *output);
        // }

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

        output += 16 * N;
        *(float2*)output = c2[0];
        // *(float2*)output = _add(c2[0], *(float2*)(bias_share+tx*2));

    }
}


void seqlen_dynamic_forward_function(float* activation, float* weight,
                    float * bias, int * seqlens, int M, int K, int N, int batchsize, float*output)
{

    // dense x dense^T -> sparse output
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;
    
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
    dim3 blockDim(256);
    // printf("gridDim: %d %d %d")
    if(bias!=nullptr)
        BLOCK_SPARSE_MATMUL_BIAS_OPENAI<<<gridDim, blockDim>>>(activation, weight, bias, seqlens, M,K,N, batchsize, output);
    else
        BLOCK_SPARSE_MATMUL_OPENAI<<<gridDim, blockDim>>>(activation, weight, seqlens, M,K,N, batchsize, output);

}



template<
    const int GLOBAL_M,
    const int GLOBAL_K,
    const int GLOBAL_N,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int N_WARP
>
__global__ void BLOCK_SPARSE_MATMUL_BIAS_FP16(
    half* __restrict__ A,
    half* __restrict__ B,
    half* __restrict__ C_val,
    int * seqlens
)
{
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    const int CPAD = 8;
    // const int N_WARP = 8;
    const int WARP_PER_ROW = 4;
    assert(N_WARP * 32 == blockDim.x); // thread num: 256
    const int WARP_COUNT_N = BLOCK_SIZE_N / 16;
    const int WARP_COUNT_M = BLOCK_SIZE_M / 16;
    
    const int WARP_N_ROWS = N_WARP / WARP_PER_ROW; // 4
    const int WARP_ROW_STRIDE = WARP_COUNT_M / WARP_N_ROWS;
    const int WARP_COL_STRIDE = WARP_COUNT_N / WARP_PER_ROW;
    int batch_idx = blockIdx.z;
    A += GLOBAL_K * GLOBAL_M * batch_idx;
    C_val += GLOBAL_M * GLOBAL_N * batch_idx;
    uint cur_seq_len = seqlens[batch_idx];
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    int wy = wid / WARP_PER_ROW;
    int wx = wid % WARP_PER_ROW;
    __shared__ half As[2 * BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[2 * BLOCK_SIZE_N][BLOCK_SIZE_K + BPAD];
    // __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N + CPAD];
    int As_base_addr = __cvta_generic_to_shared(&As[0][0]);
    int Bs_base_addr = __cvta_generic_to_shared(&Bs[0][0]);
    const int LD_AS = BLOCK_SIZE_K + APAD;
    const int LD_BS = BLOCK_SIZE_K + BPAD;
    const int LD_CS = BLOCK_SIZE_N + CPAD;
    if (by * BLOCK_SIZE_M < cur_seq_len){
        // perform the computation
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_K / 8;
        const int C_THREAD_PER_ROW = BLOCK_SIZE_N / 8;

        const int A_TILE_ROW_STRIDE = (32 * N_WARP) / A_THREAD_PER_ROW;
        const int B_TILE_ROW_STRIDE = (32 * N_WARP) / B_THREAD_PER_ROW;
        const int C_TILE_ROW_STRIDE = (32 * N_WARP) / C_THREAD_PER_ROW;
    
        const int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
        const int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;
        const int C_BLOCK_ROW_START = tid / C_THREAD_PER_ROW;

        const int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 8;
        const int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 8;
        const int C_BLOCK_COL_START = tid % C_THREAD_PER_ROW * 8;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[WARP_ROW_STRIDE];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[WARP_COL_STRIDE];
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[WARP_ROW_STRIDE][WARP_COL_STRIDE];
        // reset to zero for all the accumulators
        #pragma unroll
        for(int i=0; i<WARP_ROW_STRIDE; i++){
            #pragma unroll
            for(int j=0; j<WARP_COL_STRIDE; j++){
                wmma::fill_fragment(frag_c[i][j], 0.0);
            }
        }
        // double buffer initialization
        const int k_seq=0;
        #pragma unroll
        for(int k=A_BLOCK_ROW_START; k<BLOCK_SIZE_M; k+=A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(A[(by*BLOCK_SIZE_M+k)*K + k_seq*BLOCK_SIZE_K + A_BLOCK_COL_START]);
        }
        #pragma unroll
        for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_N; k+=B_TILE_ROW_STRIDE){
            FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(bx*BLOCK_SIZE_N+k)*K + k_seq*BLOCK_SIZE_K + B_BLOCK_COL_START]);
        }
        #pragma unroll
        for(int k_seq=1; k_seq<K/BLOCK_SIZE_K; k_seq++){
            int smem_select = (k_seq & 1) ^ 1;
            int smem_next = smem_select ^ 1;
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<BLOCK_SIZE_M; k+=A_TILE_ROW_STRIDE){
                int load_a_s_addr = As_base_addr + sizeof(half) * OFFSET(k + smem_next * BLOCK_SIZE_M, A_BLOCK_COL_START, LD_AS); 
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_a_s_addr), "l"(&A[(by * BLOCK_SIZE_M+k)*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]));
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_N; k+=B_TILE_ROW_STRIDE){
                int load_b_s_addr = Bs_base_addr + sizeof(half) * OFFSET(k + smem_next * BLOCK_SIZE_N, B_BLOCK_COL_START, LD_BS);
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_b_s_addr), "l"(&B[(bx * BLOCK_SIZE_N+k)*K + k_seq * BLOCK_SIZE_K +  B_BLOCK_COL_START]));
                
            }
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    int y = wy * WARP_ROW_STRIDE + frag_y;
                    wmma::load_matrix_sync(frag_a[frag_y], &As[y*16+smem_select*BLOCK_SIZE_M][k_step*16], LD_AS);                   
                }
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    int x = wx * WARP_COL_STRIDE + frag_x;
                    // wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16+][x*16], LD_BS);               
                    wmma::load_matrix_sync(frag_b[frag_x], &Bs[x*16+smem_select*BLOCK_SIZE_N][k_step*16], LD_BS);               
                }
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    #pragma unroll
                    for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                        wmma::mma_sync(frag_c[frag_y][frag_x], frag_a[frag_y], frag_b[frag_x], frag_c[frag_y][frag_x]);                        
                    }
                }
            }

            asm ("cp.async.commit_group;\n" ::);
            asm ("cp.async.wait_group 0;\n" ::);
            __syncthreads();

        }
        int smem_select = ((K/BLOCK_SIZE_K) & 1) ^ 1;
        #pragma unroll
        for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
            #pragma unroll
            for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                int y = wy * WARP_ROW_STRIDE + frag_y;
                wmma::load_matrix_sync(frag_a[frag_y], &As[y*16+smem_select*BLOCK_SIZE_M][k_step*16], LD_AS);                   
            }
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int x = wx * WARP_COL_STRIDE + frag_x;
                // wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16+][x*16], LD_BS);               
                wmma::load_matrix_sync(frag_b[frag_x], &Bs[x*16+smem_select*BLOCK_SIZE_N][k_step*16], LD_BS);               
            }
            #pragma unroll
            for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    wmma::mma_sync(frag_c[frag_y][frag_x], frag_a[frag_y], frag_b[frag_x], frag_c[frag_y][frag_x]);                        
                }
            }
        }
        #pragma unroll
        for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int y = wy * WARP_ROW_STRIDE + frag_y;
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::store_matrix_sync(&C_val[OFFSET(by * BLOCK_SIZE_M + y * 16, bx * BLOCK_SIZE_N + x * 16, N)], frag_c[frag_y][frag_x], N, wmma::mem_row_major);                        
            }
        }
    }

}



void seqlen_dynamic_forward_function(c10::Half* activation, c10::Half* weight,
                    c10::Half * bias, int * seqlens, int M, int K, int N, int batchsize, c10::Half*output)
{    
        // dense x dense^T -> sparse output

    if(M==128 && K==768 && N==768){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<128, 768, 768, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else if(M==128 && K==768 && N==3072){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<128, 768, 3072, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else if(M==128 && K==3072 && N==768){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<128, 3072, 768, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else if(M==256 && K==768 && N==768){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<256, 768, 768, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else if(M==256 && K==768 && N==3072){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<256, 768, 3072, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else if(M==256 && K==3072 && N==768){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<256, 3072, 768, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else if(M==4096 && K==768 && N==768){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<4096, 768, 768, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else if(M==4096 && K==768 && N==3072){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<4096, 768, 3072, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else if(M==4096 && K==3072 && N==768){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M, batchsize);
        dim3 blockDim(N_WARP*32);
        BLOCK_SPARSE_MATMUL_BIAS_FP16<4096, 3072, 768, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<gridDim, blockDim>>>((half*)activation, (half*) weight, (half*)output, seqlens);
    }else{
        printf("Please extend the shape accordingly for the seqlens linear\n");
        assert(false);
    }

}

void seqlen_dynamic_forward_function(double* activation, double* weight,
                    double * bias, int * seqlens, int M, int K, int N, int batchsize, double*output)
{    
}

at::Tensor seqlen_dynamic_sparse_linear_forward_2(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor seqlens
){
    cudaSetDevice(activation.get_device());
    int batch_size = activation.size(0);
    int max_seq_len = activation.size(1);
    int in_hidden = weight.size(1);
    int out_hidden = weight.size(0);
    int M = max_seq_len;
    int K = in_hidden;
    int N = out_hidden;
    // Q, K, V should have the same shape which is {batchsize, seq_length, hidden_dim}
    torch::Tensor output = torch::zeros({batch_size, max_seq_len, out_hidden}, activation.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(activation.type(), "seqlen_dynamic_sparse_linear", ([&]
                            {       seqlen_dynamic_forward_function(
                                    activation.data_ptr<scalar_t>(),
                                    weight.data_ptr<scalar_t>(),
                                    nullptr,
                                    seqlens.data_ptr<int>(),
                                    M, K, N, batch_size,
                                    output.data_ptr<scalar_t>()
                                ); }));
    return output;
}


at::Tensor seqlen_dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor seqlens
)
{
    cudaSetDevice(activation.get_device());
    int batch_size = activation.size(0);
    int max_seq_len = activation.size(1);
    int in_hidden = weight.size(1);
    int out_hidden = weight.size(0);
    int M = max_seq_len;
    int K = in_hidden;
    int N = out_hidden;
    // Q, K, V should have the same shape which is {batchsize, seq_length, hidden_dim}
    torch::Tensor output = torch::zeros({batch_size, max_seq_len, out_hidden}, activation.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(activation.type(), "seqlen_dynamic_sparse_linear", ([&]
                            {       seqlen_dynamic_forward_function(
                                    activation.data_ptr<scalar_t>(),
                                    weight.data_ptr<scalar_t>(),
                                    bias.data_ptr<scalar_t>(),
                                    seqlens.data_ptr<int>(),
                                    M, K, N, batch_size,
                                    output.data_ptr<scalar_t>()
                                ); }));
    // AT_DISPATCH_FLOATING_TYPES(activation.type(), "seqlen_dynamic_sparse_linear", ([&]
    //                         {       seqlen_dynamic_forward_function(
    //                                 activation.data_ptr<float>(),
    //                                 weight.data_ptr<float>(),
    //                                 bias.data_ptr<float>(),
    //                                 seqlens.data_ptr<int>(),
    //                                 M, K, N, batch_size,
    //                                 output.data_ptr<float>()
    //                             ); }));
    return output;
}

vector<at::Tensor> seqlen_dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor seqlens,
    torch::Tensor grad_c)
{
    cudaSetDevice(activation.get_device());
    // TODO: support backward in the future
    torch::Tensor a_grad = torch::zeros_like(activation);
    torch::Tensor w_grad = at::matmul(grad_c.t(), activation);
    vector<torch::Tensor> grads({a_grad, w_grad});
    return grads;
}