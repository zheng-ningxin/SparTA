
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
#define FETCH_HALF2(pointer) (reinterpret_cast<half2*>(&pointer))[0]
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


__global__ void BATCH_BLOCK_SPARSE_MATMUL(float* __restrict__ tokens, int* __restrict__ sparse_index, int* __restrict__ expert_count, float* __restrict__ B, float* __restrict__ C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, const int TMAX)
{
    /*
    description:
    tiling k dimension
    tile size: 32x64x32
    smm_sd_d_nt: sparse matmul, sparse (MxK, along K, K major bcsr) x dense (KxN, along N, need transpose) -> dense (MxN, along N)
    block sparse matrix (block size: 32x64) X dense matrix -> dense matrix

    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;

    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    __shared__ int m_index[BLOCK_SIZE_M];
    char* bShare = (char*)fShare;
    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // N
    uint by = blockIdx.y; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offsetB16 = ori_offsetB00 + N * 32;
    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4

    // B is stored in sparse format, thus, should be dealt with differently


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
    #pragma unroll
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id*TMAX+n_token);

    if(index_start<index_end){
        // load the m-dimension index to the shared memory
        if(tid<index_end-index_start ){
            m_index[tid] = sparse_index[index_start+tid];
        }
        __syncthreads();
        uint ori_offsetA00 = m_index[ty] * K + k;
        uint ori_offsetA16 = m_index[ty+16] * K + k;
        for(int k_seq=0; k_seq < (int) ( K / 64) ; k_seq++){
            uint offsetB00 = ori_offsetB00 + 64 * k_seq * N;
            uint offsetB16 = ori_offsetB16 + 64 * k_seq * N;
            uint offsetA00 = ori_offsetA00 + 64 * k_seq;
            uint offsetA16 = ori_offsetA16 + 64 * k_seq;
            float4 a00 = {0}, a16 = {0};
            float4 b00 = {0}, b16 = {0};
            if(ty<index_end-index_start)
                a00 = __ldg((const float4*)(add_ptr_f(tokens, offsetA00)));
            if(ty+16<index_end-index_start)
                a16 = __ldg((const float4*)(add_ptr_f(tokens, offsetA16)));
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

            *(float*)&bShare[storB + (1*65*32)*4] = b00.x;
            *(float*)&bShare[storB + (1*65*32 + 1)*4] = b00.y;
            *(float*)&bShare[storB + (1*65*32 + 2)*4] = b00.z;
            *(float*)&bShare[storB + (1*65*32 + 3)*4] = b00.w;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32)*4] = b16.x;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 1)*4] = b16.y;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 2)*4] = b16.z;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 3)*4] = b16.w;
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

        // C should be row major
        // C += (bx * BLOCK_SIZE_M + ty) * N + (by * BLOCK_SIZE_N + tx * 2);

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
        float * WC;
        WC = C + m_index[ty] * N  + blockIdx.x * BLOCK_SIZE_N + tx *2;
        if(ty < index_end-index_start)
            *(float2*)WC = c2[0];

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

        WC = C + m_index[ty+16] * N  + blockIdx.x * BLOCK_SIZE_N + tx *2;
        if(ty + 16 < index_end - index_start){
            *(float2*)WC = c2[0];
        }

    }
    
}

template<
    const int GLOBAL_K,
    const int GLOBAL_N,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE(half* __restrict__ tokens, int* __restrict__ sparse_index, int* __restrict__ expert_count, half* __restrict__ B, half* __restrict__ C, const int TMAX)
{
    // const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    const int CPAD = 8;
    const int N_WARP = 8;
    const int WARP_PER_ROW = 4;
    assert(N_WARP * 32 == blockDim.x); // thread num: 256
    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    const int WARP_COUNT_N = BLOCK_SIZE_N / 16;
    const int WARP_COUNT_M = BLOCK_SIZE_M / 16;
    
    const int WARP_N_ROWS = N_WARP / WARP_PER_ROW; // 4
    const int WARP_ROW_STRIDE = WARP_COUNT_M / WARP_N_ROWS;
    const int WARP_COL_STRIDE = WARP_COUNT_N / WARP_PER_ROW;

    assert(WARP_COUNT_N % WARP_PER_ROW==0);
    assert(WARP_COUNT_M % WARP_N_ROWS==0);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / WARP_PER_ROW;
    int wx = wid % WARP_PER_ROW;

    __shared__ half As[BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N + CPAD];

    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
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
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[WARP_COL_STRIDE];
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[WARP_ROW_STRIDE][WARP_COL_STRIDE];
        // reset to zero for all the accumulators
        #pragma unroll
        for(int i=0; i<WARP_ROW_STRIDE; i++){
            #pragma unroll
            for(int j=0; j<WARP_COL_STRIDE; j++){
                wmma::fill_fragment(frag_c[i][j], 0.0);
            }
        }
        // load to the m_index to shared memory
        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];

        __syncthreads();
        #pragma unroll
        for(int k_seq=0; k_seq<K/BLOCK_SIZE_K; k_seq++){
            // Fetch shared memory
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            __syncthreads();
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    int y = wy * WARP_ROW_STRIDE + frag_y;
                    wmma::load_matrix_sync(frag_a[frag_y], &As[y*16][k_step*16], BLOCK_SIZE_K + APAD);                   
                }
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    int x = wx * WARP_COL_STRIDE + frag_x;
                    wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16][x*16], BLOCK_SIZE_N + BPAD);               
                }
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    #pragma unroll
                    for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                        wmma::mma_sync(frag_c[frag_y][frag_x], frag_a[frag_y], frag_b[frag_x], frag_c[frag_y][frag_x]);                        
                    }
                }
            }

            __syncthreads();

        }

        //write back to the shared memory
        #pragma unroll
        for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int y = wy * WARP_ROW_STRIDE + frag_y;
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::store_matrix_sync(&Cs[y*16][x*16], frag_c[frag_y][frag_x], BLOCK_SIZE_N+CPAD, wmma::mem_row_major);                        
            }
        }
        __syncthreads();
        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
        }
    }
    
}


template<
    const int GLOBAL_K,
    const int GLOBAL_N,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int SPLITK
>
__global__ void BATCH_BLOCK_SPARSE_MATMUL_FP16_SPLITK_TEMPLATE(half* __restrict__ tokens, int* __restrict__ sparse_index, int* __restrict__ expert_count, half* __restrict__ B, half* __restrict__ C, const int TMAX)
{
    // const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    const int CPAD = 8;
    const int N_WARP = 8;
    const int WARP_PER_ROW = 4;
    assert(N_WARP * 32 == blockDim.x); // thread num: 256
    assert(GLOBAL_K%(SPLITK*BLOCK_SIZE_K) == 0);
    int exp_id = blockIdx.z/SPLITK;
    B += K * N * exp_id;
    const int SPLIT_K_ID = blockIdx.z % SPLITK;
    const int WARP_COUNT_N = BLOCK_SIZE_N / 16;
    const int WARP_COUNT_M = BLOCK_SIZE_M / 16;
    const int GLOBAL_K_STRIDE = K/SPLITK;

    const int WARP_N_ROWS = N_WARP / WARP_PER_ROW; // 4
    const int WARP_ROW_STRIDE = WARP_COUNT_M / WARP_N_ROWS;
    const int WARP_COL_STRIDE = WARP_COUNT_N / WARP_PER_ROW;

    assert(WARP_COUNT_N % WARP_PER_ROW==0);
    assert(WARP_COUNT_M % WARP_N_ROWS==0);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / WARP_PER_ROW;
    int wx = wid % WARP_PER_ROW;

    __shared__ half As[BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N + CPAD];

    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
        const int C_THREAD_PER_ROW = BLOCK_SIZE_N / 2; // the granularity of C write is half2

        const int A_TILE_ROW_STRIDE = (32 * N_WARP) / A_THREAD_PER_ROW;
        const int B_TILE_ROW_STRIDE = (32 * N_WARP) / B_THREAD_PER_ROW;
        const int C_TILE_ROW_STRIDE = (32 * N_WARP) / C_THREAD_PER_ROW;
    
        const int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
        const int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;
        const int C_BLOCK_ROW_START = tid / C_THREAD_PER_ROW;

        const int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 8;
        const int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 8;
        const int C_BLOCK_COL_START = tid % C_THREAD_PER_ROW * 2;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[WARP_ROW_STRIDE];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[WARP_COL_STRIDE];
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[WARP_ROW_STRIDE][WARP_COL_STRIDE];
        // reset to zero for all the accumulators
        #pragma unroll
        for(int i=0; i<WARP_ROW_STRIDE; i++){
            #pragma unroll
            for(int j=0; j<WARP_COL_STRIDE; j++){
                wmma::fill_fragment(frag_c[i][j], 0.0);
            }
        }
        // load to the m_index to shared memory
        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];

        __syncthreads();
        const int k_offset_start = (SPLIT_K_ID * GLOBAL_K_STRIDE)/BLOCK_SIZE_K;
        const int k_offset_end = k_offset_start + GLOBAL_K_STRIDE/BLOCK_SIZE_K;
        // if(tid==0 && by==0 && bx == 0 && blockIdx.z < 4){
        //     printf("bx:%d by:%d bz:%d expid:%d k_off_start:%d k_off_end:%d \n", bx, by, blockIdx.z, exp_id, k_offset_start, k_offset_end);
        // }
        #pragma unroll 8
        for(int k_seq=k_offset_start; k_seq<k_offset_end; k_seq++){
            // Fetch shared memory
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            __syncthreads();
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    int y = wy * WARP_ROW_STRIDE + frag_y;
                    wmma::load_matrix_sync(frag_a[frag_y], &As[y*16][k_step*16], BLOCK_SIZE_K + APAD);                   
                }
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    int x = wx * WARP_COL_STRIDE + frag_x;
                    wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16][x*16], BLOCK_SIZE_N + BPAD);               
                }
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    #pragma unroll
                    for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                        wmma::mma_sync(frag_c[frag_y][frag_x], frag_a[frag_y], frag_b[frag_x], frag_c[frag_y][frag_x]);                        
                    }
                }
            }

            __syncthreads();

        }

        //write back to the shared memory
        #pragma unroll
        for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int y = wy * WARP_ROW_STRIDE + frag_y;
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::store_matrix_sync(&Cs[y*16][x*16], frag_c[frag_y][frag_x], BLOCK_SIZE_N+CPAD, wmma::mem_row_major);                        
            }
        }
        __syncthreads();
         
        // write back with atomic add
        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            atomicAdd((half2*)&C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START], FETCH_HALF2(Cs[k][C_BLOCK_COL_START]));
            // FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
        }
    }
    
}


template<
    const int GLOBAL_K,
    const int GLOBAL_N,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int BLOCK_GROUP_SIZE
>
__global__ void BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE_V2(half* __restrict__ tokens, int* __restrict__ sparse_index, int* __restrict__  expert_count, half* __restrict__ B, half* __restrict__ C, const int TMAX)
{
    // const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    const int CPAD = 8;
    const int N_WARP = 8;
    const int WARP_PER_ROW = 4;
    assert(N_WARP * 32 == blockDim.x); // thread num: 256
    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    const int WARP_COUNT_N = BLOCK_SIZE_N / 16;
    const int WARP_COUNT_M = BLOCK_SIZE_M / 16;
    
    const int WARP_N_ROWS = N_WARP / WARP_PER_ROW; // 4
    const int WARP_ROW_STRIDE = WARP_COUNT_M / WARP_N_ROWS;
    const int WARP_COL_STRIDE = WARP_COUNT_N / WARP_PER_ROW;

    assert(WARP_COUNT_N % WARP_PER_ROW==0);
    assert(WARP_COUNT_M % WARP_N_ROWS==0);

    int obx = blockIdx.x;
    int oby = blockIdx.y;
    // const int BLOCK_GROUP_SIZE = 8;
    assert(gridDim.x%BLOCK_GROUP_SIZE==0);

    int obid = oby * gridDim.x + obx;
    int bgid = obid / BLOCK_GROUP_SIZE;
    int by = bgid % gridDim.y;
    int bx = bgid / gridDim.y * BLOCK_GROUP_SIZE + obid % BLOCK_GROUP_SIZE; 

    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / WARP_PER_ROW;
    int wx = wid % WARP_PER_ROW;

    __shared__ half As[BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N + CPAD];

    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
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
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[WARP_COL_STRIDE];
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[WARP_ROW_STRIDE][WARP_COL_STRIDE];
        // reset to zero for all the accumulators
        #pragma unroll
        for(int i=0; i<WARP_ROW_STRIDE; i++){
            #pragma unroll
            for(int j=0; j<WARP_COL_STRIDE; j++){
                wmma::fill_fragment(frag_c[i][j], 0.0);
            }
        }
        // load to the m_index to shared memory
        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];

        __syncthreads();
        #pragma unroll
        for(int k_seq=0; k_seq<K/BLOCK_SIZE_K; k_seq++){
            // Fetch shared memory
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            __syncthreads();
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    int y = wy * WARP_ROW_STRIDE + frag_y;
                    wmma::load_matrix_sync(frag_a[frag_y], &As[y*16][k_step*16], BLOCK_SIZE_K + APAD);                   
                }
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    int x = wx * WARP_COL_STRIDE + frag_x;
                    wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16][x*16], BLOCK_SIZE_N + BPAD);               
                }
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    #pragma unroll
                    for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                        wmma::mma_sync(frag_c[frag_y][frag_x], frag_a[frag_y], frag_b[frag_x], frag_c[frag_y][frag_x]);                        
                    }
                }
            }

            __syncthreads();

        }

        //write back to the shared memory
        #pragma unroll
        for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int y = wy * WARP_ROW_STRIDE + frag_y;
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::store_matrix_sync(&Cs[y*16][x*16], frag_c[frag_y][frag_x], BLOCK_SIZE_N+CPAD, wmma::mem_row_major);                        
            }
        }
        __syncthreads();
        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
        }
    }
    
}



template<
    const int GLOBAL_K,
    const int GLOBAL_N,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE_V3(half* __restrict__  tokens, int* __restrict__  sparse_index, int* __restrict__  expert_count, half* __restrict__ B, half* __restrict__ C, const int TMAX)
{
    // const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    const int CPAD = 8;
    const int N_WARP = 8;
    const int WARP_PER_ROW = 4;
    assert(N_WARP * 32 == blockDim.x); // thread num: 256
    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    const int WARP_COUNT_N = BLOCK_SIZE_N / 16;
    const int WARP_COUNT_M = BLOCK_SIZE_M / 16;
    
    const int WARP_N_ROWS = N_WARP / WARP_PER_ROW; // 4
    const int WARP_ROW_STRIDE = WARP_COUNT_M / WARP_N_ROWS;
    const int WARP_COL_STRIDE = WARP_COUNT_N / WARP_PER_ROW;

    assert(WARP_COUNT_N % WARP_PER_ROW==0);
    assert(WARP_COUNT_M % WARP_N_ROWS==0);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / WARP_PER_ROW;
    int wx = wid % WARP_PER_ROW;

    __shared__ half As[2 * BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[2 * BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N + CPAD];
    
    int As_base_addr = __cvta_generic_to_shared(&As[0][0]);
    int Bs_base_addr = __cvta_generic_to_shared(&Bs[0][0]);
    const int LD_AS = BLOCK_SIZE_K + APAD;
    const int LD_BS = BLOCK_SIZE_N + BPAD;
    const int LD_CS = BLOCK_SIZE_N + CPAD;

    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
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
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[WARP_COL_STRIDE];
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[WARP_ROW_STRIDE][WARP_COL_STRIDE];
        // reset to zero for all the accumulators
        #pragma unroll
        for(int i=0; i<WARP_ROW_STRIDE; i++){
            #pragma unroll
            for(int j=0; j<WARP_COL_STRIDE; j++){
                wmma::fill_fragment(frag_c[i][j], 0.0);
            }
        }
        // load to the m_index to shared memory
        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];

        __syncthreads();

        {
            const int k_seq=0;
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                //     : "r"(As_base_addr + OFFSET(k, A_BLOCK_COL_START, LD_AS)), "l"(&tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]));
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                //     : "r"(Bs_base_addr + OFFSET(k, B_BLOCK_COL_START, LD_BS)), "l"(&B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]));
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            // asm ("cp.async.commit_group;\n" ::);
            // asm ("cp.async.wait_group 0;\n" ::);
        }

        #pragma unroll
        for(int k_seq=1; k_seq<K/BLOCK_SIZE_K; k_seq++){
            // Fetch shared memory
            int smem_select = (k_seq & 1) ^ 1;
            int smem_next = smem_select ^ 1;
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                int load_a_s_addr = As_base_addr + sizeof(half) * OFFSET(k + smem_next * BLOCK_SIZE_M, A_BLOCK_COL_START, LD_AS); 
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_a_s_addr), "l"(&tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]));
                //FETCH_FLOAT4(As[k + smem_next * BLOCK_SIZE_M][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                int load_b_s_addr = Bs_base_addr + sizeof(half) * OFFSET(k + smem_next * BLOCK_SIZE_K, B_BLOCK_COL_START, LD_BS);
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_b_s_addr), "l"(&B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]));
                //FETCH_FLOAT4(Bs[k + smem_next * BLOCK_SIZE_K][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            // __syncthreads();
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    int y = wy * WARP_ROW_STRIDE + frag_y;
                    wmma::load_matrix_sync(frag_a[frag_y], &As[y*16+smem_select*BLOCK_SIZE_M][k_step*16], BLOCK_SIZE_K + APAD);                   
                }
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    int x = wx * WARP_COL_STRIDE + frag_x;
                    wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16+smem_select*BLOCK_SIZE_K][x*16], BLOCK_SIZE_N + BPAD);               
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
                wmma::load_matrix_sync(frag_a[frag_y], &As[y*16+smem_select*BLOCK_SIZE_M][k_step*16], BLOCK_SIZE_K + APAD);                   
            }
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16+smem_select*BLOCK_SIZE_K][x*16], BLOCK_SIZE_N + BPAD);               
            }
            #pragma unroll
            for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    wmma::mma_sync(frag_c[frag_y][frag_x], frag_a[frag_y], frag_b[frag_x], frag_c[frag_y][frag_x]);                        
                }
            }
        }

        //write back to the shared memory
        #pragma unroll
        for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int y = wy * WARP_ROW_STRIDE + frag_y;
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::store_matrix_sync(&Cs[y*16][x*16], frag_c[frag_y][frag_x], BLOCK_SIZE_N+CPAD, wmma::mem_row_major);                        
            }
        }
        __syncthreads();
        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
        }
    }
    
}


template<
    const int GLOBAL_K,
    const int GLOBAL_N,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void BATCH_BLOCK_SPARSE_MATMUL_RELU_FP16_TEMPLATE_V3(half* __restrict__  tokens, int* __restrict__  sparse_index, int* __restrict__  expert_count, half* __restrict__ B, half* __restrict__ C, const int TMAX)
{
    // const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    const int CPAD = 8;
    const int N_WARP = 8;
    const int WARP_PER_ROW = 4;
    assert(N_WARP * 32 == blockDim.x); // thread num: 256
    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    const int WARP_COUNT_N = BLOCK_SIZE_N / 16;
    const int WARP_COUNT_M = BLOCK_SIZE_M / 16;
    
    const int WARP_N_ROWS = N_WARP / WARP_PER_ROW; // 4
    const int WARP_ROW_STRIDE = WARP_COUNT_M / WARP_N_ROWS;
    const int WARP_COL_STRIDE = WARP_COUNT_N / WARP_PER_ROW;

    assert(WARP_COUNT_N % WARP_PER_ROW==0);
    assert(WARP_COUNT_M % WARP_N_ROWS==0);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / WARP_PER_ROW;
    int wx = wid % WARP_PER_ROW;

    __shared__ half As[2 * BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[2 * BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N + CPAD];
    half tmp_reg[8];
    int As_base_addr = __cvta_generic_to_shared(&As[0][0]);
    int Bs_base_addr = __cvta_generic_to_shared(&Bs[0][0]);
    const int LD_AS = BLOCK_SIZE_K + APAD;
    const int LD_BS = BLOCK_SIZE_N + BPAD;
    const int LD_CS = BLOCK_SIZE_N + CPAD;

    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
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
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[WARP_COL_STRIDE];
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[WARP_ROW_STRIDE][WARP_COL_STRIDE];
        // reset to zero for all the accumulators
        #pragma unroll
        for(int i=0; i<WARP_ROW_STRIDE; i++){
            #pragma unroll
            for(int j=0; j<WARP_COL_STRIDE; j++){
                wmma::fill_fragment(frag_c[i][j], 0.0);
            }
        }
        // load to the m_index to shared memory
        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];

        __syncthreads();

        {
            const int k_seq=0;
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                //     : "r"(As_base_addr + OFFSET(k, A_BLOCK_COL_START, LD_AS)), "l"(&tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]));
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                //     : "r"(Bs_base_addr + OFFSET(k, B_BLOCK_COL_START, LD_BS)), "l"(&B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]));
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            // asm ("cp.async.commit_group;\n" ::);
            // asm ("cp.async.wait_group 0;\n" ::);
        }

        #pragma unroll
        for(int k_seq=1; k_seq<K/BLOCK_SIZE_K; k_seq++){
            // Fetch shared memory
            int smem_select = (k_seq & 1) ^ 1;
            int smem_next = smem_select ^ 1;
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                int load_a_s_addr = As_base_addr + sizeof(half) * OFFSET(k + smem_next * BLOCK_SIZE_M, A_BLOCK_COL_START, LD_AS); 
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_a_s_addr), "l"(&tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]));
                //FETCH_FLOAT4(As[k + smem_next * BLOCK_SIZE_M][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                int load_b_s_addr = Bs_base_addr + sizeof(half) * OFFSET(k + smem_next * BLOCK_SIZE_K, B_BLOCK_COL_START, LD_BS);
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_b_s_addr), "l"(&B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]));
                //FETCH_FLOAT4(Bs[k + smem_next * BLOCK_SIZE_K][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);

            }
            // __syncthreads();
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    int y = wy * WARP_ROW_STRIDE + frag_y;
                    wmma::load_matrix_sync(frag_a[frag_y], &As[y*16+smem_select*BLOCK_SIZE_M][k_step*16], BLOCK_SIZE_K + APAD);                   
                }
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    int x = wx * WARP_COL_STRIDE + frag_x;
                    wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16+smem_select*BLOCK_SIZE_K][x*16], BLOCK_SIZE_N + BPAD);               
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
                wmma::load_matrix_sync(frag_a[frag_y], &As[y*16+smem_select*BLOCK_SIZE_M][k_step*16], BLOCK_SIZE_K + APAD);                   
            }
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16+smem_select*BLOCK_SIZE_K][x*16], BLOCK_SIZE_N + BPAD);               
            }
            #pragma unroll
            for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    wmma::mma_sync(frag_c[frag_y][frag_x], frag_a[frag_y], frag_b[frag_x], frag_c[frag_y][frag_x]);                        
                }
            }
        }

        //write back to the shared memory
        #pragma unroll
        for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int y = wy * WARP_ROW_STRIDE + frag_y;
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::store_matrix_sync(&Cs[y*16][x*16], frag_c[frag_y][frag_x], BLOCK_SIZE_N+CPAD, wmma::mem_row_major);                        
            }
        }
        __syncthreads();
        // for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
        //     FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
        // }
        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            FETCH_FLOAT4(tmp_reg[0]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
            #pragma unroll
            for(int i=0;i<8;i++){
                tmp_reg[i] = max(tmp_reg[i], 0.0);
            }
            FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(tmp_reg[0]);
        }
    }
    
}


__global__ void BATCH_BLOCK_SPARSE_MATMUL_FP16(half* __restrict__  tokens, int* __restrict__  sparse_index, int* __restrict__  expert_count, half* __restrict__ B, half* __restrict__ C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, const int TMAX)
{
    /*
    description:
    tiling k dimension
    tile size: 32x64x32
    smm_sd_d_nt: sparse matmul, sparse (MxK, along K, K major bcsr) x dense (KxN, along N, need transpose) -> dense (MxN, along N)
    block sparse matrix (block size: 32x64) X dense matrix -> dense matrix

    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    // const int CPAD=8;
    assert(blockDim.x % 32 == 0);
    const int n_warp = blockDim.x / 32;
    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    const int w_per_row = BLOCK_SIZE_N / 16;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / w_per_row;
    int wx = wid % w_per_row;

    __shared__ half As[BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N];
    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){        
        uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
        uint ori_offsetB16 = ori_offsetB00 + N * 32;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;
        wmma::fill_fragment(frag_c, 0.0);

        // load to the shared memory
        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
        const int C_THREAD_PER_ROW = BLOCK_SIZE_N / 8;

        const int A_TILE_ROW_STRIDE = blockDim.x / A_THREAD_PER_ROW;
        const int B_TILE_ROW_STRIDE = blockDim.x / B_THREAD_PER_ROW;
        const int C_TILE_ROW_STRIDE = blockDim.x / C_THREAD_PER_ROW;
    
        const int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
        const int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;
        const int C_BLOCK_ROW_START = tid / C_THREAD_PER_ROW;

        const int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 8;
        const int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 8;
        const int C_BLOCK_COL_START = tid % C_THREAD_PER_ROW * 8;
        // uint ori_offsetB00 = B_BLOCK_ROW_START * N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START;

        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];
        __syncthreads();
        // uint ori_offsetA00 = m_index[ty] * K + k;
        for(int k_seq=0; k_seq<K/BLOCK_SIZE_K; k_seq++){
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            __syncthreads();
            // if(tid==0)
            //     printf("load value: %f\n", __half2float(Bs[0][0]));
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                wmma::load_matrix_sync(frag_a, &As[wy*16][k_step*16], BLOCK_SIZE_K + APAD);
                wmma::load_matrix_sync(frag_b, &Bs[k_step*16][wx*16], BLOCK_SIZE_N + BPAD);
                wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
            }

            __syncthreads();

        }
        wmma::store_matrix_sync(&Cs[wy*16][wx*16], frag_c, BLOCK_SIZE_N, wmma::mem_row_major);
        __syncthreads();

        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            // if(tid==0){
            //     printf("bx:%d by:%d k:%d C_BLOCK_COL_START:%d m_index:%d offset:%d result %f\n", bx, by, k, C_BLOCK_COL_START, m_index[k], m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START,__half2float(Cs[k][C_BLOCK_COL_START]));
            // }
            FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
        }
    }


    
}

__global__ void BATCH_BLOCK_SPARSE_MATMUL_FP16_V2(half* __restrict__  tokens, int* __restrict__  sparse_index, int* __restrict__  expert_count, half* __restrict__ B, half* __restrict__ C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, const int TMAX)
{
    /*
    description:
    tiling k dimension
    tile size: 32x64x32
    smm_sd_d_nt: sparse matmul, sparse (MxK, along K, K major bcsr) x dense (KxN, along N, need transpose) -> dense (MxN, along N)
    block sparse matrix (block size: 32x64) X dense matrix -> dense matrix

    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 32;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    assert(blockDim.x % 32 == 0);
    const int n_warp = blockDim.x / 32;
    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    const int w_per_row = BLOCK_SIZE_N / 16;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / w_per_row;
    int wx = wid % w_per_row;

    __shared__ half As[2 * BLOCK_SIZE_M][(BLOCK_SIZE_K + APAD)];
    __shared__ half Bs[2 * BLOCK_SIZE_K][(BLOCK_SIZE_N + BPAD)];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N];
    // const int s_a_offset = BLOCK_SIZE_M * (BLOCK_SIZE_K+APAD);
    // const int s_b_offset = BLOCK_SIZE_K * (BLOCK_SIZE_N+BPAD);
    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){
        uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
        uint ori_offsetB16 = ori_offsetB00 + N * 32;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;
        wmma::fill_fragment(frag_c, 0.0);

        // load to the shared memory
        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
        const int C_THREAD_PER_ROW = BLOCK_SIZE_N / 8;

        const int A_TILE_ROW_STRIDE = blockDim.x / A_THREAD_PER_ROW;
        const int B_TILE_ROW_STRIDE = blockDim.x / B_THREAD_PER_ROW;
        const int C_TILE_ROW_STRIDE = blockDim.x / C_THREAD_PER_ROW;
    
        const int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
        const int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;
        const int C_BLOCK_ROW_START = tid / C_THREAD_PER_ROW;

        const int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 8;
        const int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 8;
        const int C_BLOCK_COL_START = tid % C_THREAD_PER_ROW * 8;
        // uint ori_offsetB00 = B_BLOCK_ROW_START * N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START;

        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];
        __syncthreads();
        // load the first time
        {
            const int k_seq = 0;
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
                // FETCH_FLOAT4(As[OFFSET(k, A_BLOCK_COL_START, BLOCK_SIZE_K+APAD)]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(0*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
                // FETCH_FLOAT4(Bs[OFFSET(k, B_BLOCK_COL_START, BLOCK_SIZE_N+BPAD)]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
        }
        __syncthreads();
        // uint ori_offsetA00 = m_index[ty] * K + k;
        #pragma unroll
        for(int k_seq=1; k_seq<K/BLOCK_SIZE_K; k_seq++){
            int smem_select = (k_seq & 1) ^ 1;
            int smem_next = smem_select ^ 1;
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[k+smem_next*BLOCK_SIZE_M][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
                // FETCH_FLOAT4(As[OFFSET(k, A_BLOCK_COL_START, BLOCK_SIZE_K+APAD)+smem_next*s_a_offset]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);

            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[k+smem_next*BLOCK_SIZE_K][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
                // FETCH_FLOAT4(Bs[OFFSET(k, B_BLOCK_COL_START, BLOCK_SIZE_N+BPAD)+smem_next*s_b_offset]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            // __syncthreads();
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                wmma::load_matrix_sync(frag_a, &As[wy*16+smem_select*BLOCK_SIZE_M][k_step*16], BLOCK_SIZE_K + APAD);
                wmma::load_matrix_sync(frag_b, &Bs[k_step*16+smem_select*BLOCK_SIZE_K][wx*16], BLOCK_SIZE_N + BPAD);
                // wmma::load_matrix_sync(frag_a, &As[OFFSET(wy*16, k_step*16, BLOCK_SIZE_K+APAD)+smem_select*s_a_offset], BLOCK_SIZE_K + APAD);
                // wmma::load_matrix_sync(frag_b, &Bs[OFFSET(k_step*16, wx*16, BLOCK_SIZE_N+BPAD)+smem_select*s_b_offset], BLOCK_SIZE_N + BPAD);
                wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
            }

            __syncthreads();

        }
        int smem_select = ((K/BLOCK_SIZE_K) & 1) ^ 1;
        #pragma unroll
        for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
            wmma::load_matrix_sync(frag_a, &As[wy*16+smem_select*BLOCK_SIZE_M][k_step*16], BLOCK_SIZE_K + APAD);
            wmma::load_matrix_sync(frag_b, &Bs[k_step*16+smem_select*BLOCK_SIZE_K][wx*16], BLOCK_SIZE_N + BPAD);
            // wmma::load_matrix_sync(frag_a, &As[OFFSET(wy*16, k_step*16, BLOCK_SIZE_K+APAD)+smem_select*s_a_offset], BLOCK_SIZE_K + APAD);
            // wmma::load_matrix_sync(frag_b, &Bs[OFFSET(k_step*16, wx*16, BLOCK_SIZE_N+BPAD)+smem_select*s_b_offset], BLOCK_SIZE_N + BPAD);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
        // __syncthreads();
        wmma::store_matrix_sync(&Cs[wy*16][wx*16], frag_c, BLOCK_SIZE_N, wmma::mem_row_major);
        __syncthreads();

        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
        }
    }
    
}



__global__ void BATCH_BLOCK_SPARSE_MATMUL_FP16_V3(half* __restrict__  tokens, int* __restrict__  sparse_index, int* __restrict__  expert_count, half* __restrict__ B, half* __restrict__ C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, const int TMAX)
{
    /*
    description:
    tiling k dimension
    tile size: 32x64x32
    smm_sd_d_nt: sparse matmul, sparse (MxK, along K, K major bcsr) x dense (KxN, along N, need transpose) -> dense (MxN, along N)
    block sparse matrix (block size: 32x64) X dense matrix -> dense matrix

    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    assert(blockDim.x % 32 == 0);
    const int n_warp = blockDim.x / 32;
    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    const int w_per_row = BLOCK_SIZE_N / 16;

    int obx = blockIdx.x;
    int oby = blockIdx.y;
    const int BLOCK_GROUP_M = 2;
    const int BLOCK_GROUP_N = 2;
    assert(gridDim.x%BLOCK_GROUP_N==0);
    assert(gridDim.y%BLOCK_GROUP_M==0);
    const int BLOCK_PER_GROUP = BLOCK_GROUP_M * BLOCK_GROUP_N;
    const int GROUP_PER_ROW = gridDim.x / BLOCK_GROUP_N;
    int obid = oby * gridDim.x + obx;
    int bgid = obid / BLOCK_PER_GROUP;
    int bgy = bgid / GROUP_PER_ROW;
    int bgx = bgid % GROUP_PER_ROW;
    int bid_within_group = obid % BLOCK_PER_GROUP;
    int y_within_group = bid_within_group / BLOCK_GROUP_N;
    int x_within_group = bid_within_group % BLOCK_GROUP_N;
    int bx = x_within_group + BLOCK_GROUP_N * bgx;
    int by = y_within_group + BLOCK_GROUP_M * bgy;


    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / w_per_row;
    int wx = wid % w_per_row;

    __shared__ half As[BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N];
    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){        
        uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
        uint ori_offsetB16 = ori_offsetB00 + N * 32;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;
        wmma::fill_fragment(frag_c, 0.0);

        // load to the shared memory
        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
        const int C_THREAD_PER_ROW = BLOCK_SIZE_N / 8;

        const int A_TILE_ROW_STRIDE = blockDim.x / A_THREAD_PER_ROW;
        const int B_TILE_ROW_STRIDE = blockDim.x / B_THREAD_PER_ROW;
        const int C_TILE_ROW_STRIDE = blockDim.x / C_THREAD_PER_ROW;
    
        const int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
        const int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;
        const int C_BLOCK_ROW_START = tid / C_THREAD_PER_ROW;

        const int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 8;
        const int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 8;
        const int C_BLOCK_COL_START = tid % C_THREAD_PER_ROW * 8;
        // uint ori_offsetB00 = B_BLOCK_ROW_START * N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START;

        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];
        __syncthreads();
        // uint ori_offsetA00 = m_index[ty] * K + k;
        for(int k_seq=0; k_seq<K/BLOCK_SIZE_K; k_seq++){
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            __syncthreads();
            // if(tid==0)
            //     printf("load value: %f\n", __half2float(Bs[0][0]));
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                wmma::load_matrix_sync(frag_a, &As[wy*16][k_step*16], BLOCK_SIZE_K + APAD);
                wmma::load_matrix_sync(frag_b, &Bs[k_step*16][wx*16], BLOCK_SIZE_N + BPAD);
                wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
            }

            __syncthreads();

        }
        wmma::store_matrix_sync(&Cs[wy*16][wx*16], frag_c, BLOCK_SIZE_N, wmma::mem_row_major);
        __syncthreads();

        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            // if(tid==0){
            //     printf("bx:%d by:%d k:%d C_BLOCK_COL_START:%d m_index:%d offset:%d result %f\n", bx, by, k, C_BLOCK_COL_START, m_index[k], m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START,__half2float(Cs[k][C_BLOCK_COL_START]));
            // }
            FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
        }
    }


    
}
 __global__ void convert_sparse_index(int * router_index, int total_token, int * expert_count, int *sparse_index, const int TMAX)
{
    int bx =  blockIdx.x;
    int tid = threadIdx.x;
    int offset = bx * blockDim.x + tid;
    int exp_id = router_index[offset];
    int pos_id = atomicAdd(&expert_count[exp_id], 1);
    sparse_index[exp_id*TMAX+pos_id] = offset;

}


void convert_index(
    int * router_index,
    int * sparse_index,
    int * expert_count,
    int total_token,
    int n_expert,
    const int TMAX
    )
{
    // convert the sparse index on the fly
    dim3 blockDim(256);
    dim3 gridDim(total_token/256);
    CUDA_SAFE_CALL(cudaMemset((void*)expert_count, 0, sizeof(int)*n_expert));
    convert_sparse_index<<<gridDim, blockDim>>>(router_index, total_token, expert_count, sparse_index, TMAX);
    // compuate the sparse forward with stile

}


void forward_function(
    int * router_index,
    int * sparse_index,
    float* tokens,
    float* weight,
    float* output,
    int * expert_count,
    int total_token,
    int in_hidden,
    int out_hidden,
    int n_expert,
    const int GLOBAL_M,
    const int TMAX
    )
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;
    const int max_block = (GLOBAL_M -1 + BLOCK_SIZE_M)/BLOCK_SIZE_M;
    dim3 blockDim(256);
    dim3 gridDim(out_hidden/BLOCK_SIZE_N, max_block, n_expert);
    BATCH_BLOCK_SPARSE_MATMUL<<<gridDim, blockDim>>>(tokens, sparse_index, expert_count, weight, output, GLOBAL_M, in_hidden, out_hidden, TMAX);
}


__global__ void HGEMM(
    int * csr_row, int * csr_col, half *__restrict__ csr_val,
    half *__restrict__ B, half *__restrict__ C,
    const int M, const int N, const int K, const int BLOCK_H, const int BLOCK_W)
{

    const int BM = 32;
    const int BN = 32;
    const int BK = 64;
    const int w_per_row = BN/16;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    int wy = wid / w_per_row;
    int wx = wid % w_per_row;
    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half As[BM][BK + APAD];
    __shared__ half Bs[BK][BN + BPAD];



    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;
    wmma::fill_fragment(frag_c, 0.0);
    // load to the shared memory
    const int A_THREAD_PER_ROW = BK / 8; // 1 float4 = 8 half
    const int B_THREAD_PER_ROW = BN / 8;
    const int A_TILE_ROW_STRIDE = blockDim.x / A_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = blockDim.x / B_THREAD_PER_ROW;
    const int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    const int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;
    const int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 8;
    const int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 8;
    
    int index_start = csr_row[by];
    int index_end = csr_row[by+1];
    for(int tile_block_idx = index_start; tile_block_idx< index_end; tile_block_idx++){
        int col_pos = csr_col[tile_block_idx] * BK;
        #pragma unroll
        for(int k = 0; k < BM; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[k+A_BLOCK_ROW_START][A_BLOCK_COL_START]) = FETCH_FLOAT4(csr_val[tile_block_idx * BM * BK + OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BK)]);
        }

        #pragma unroll
        for(int k = 0; k < BK; k += B_TILE_ROW_STRIDE){
            // FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BN+BPAD)]) = FETCH_FLOAT4(B[OFFSET(col_pos+k+B_BLOCK_ROW_START, bx*BN + B_BLOCK_COL_START, N)]);
            FETCH_FLOAT4(Bs[k+B_BLOCK_ROW_START][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[OFFSET(col_pos+k+B_BLOCK_ROW_START, bx*BN + B_BLOCK_COL_START, N)]);
        }

        __syncthreads();
        #pragma unroll
        for(int k_step=0; k_step<BK/16; k_step++){
            wmma::load_matrix_sync(frag_a[k_step], &As[wy*16][k_step*16], BK + APAD);
            wmma::load_matrix_sync(frag_b[k_step], &Bs[k_step*16][wx*16], BN + BPAD);
        }
        #pragma unroll
        for(int k_step=0; k_step<BK/16; k_step++){
            wmma::mma_sync(frag_c, frag_a[k_step], frag_b[k_step], frag_c);
        }
        __syncthreads();

    }
    int write_offset = (by * BM + wy * 16) * N + bx * BN + wx * 16;
    wmma::store_matrix_sync(&C[write_offset], frag_c, N, wmma::mem_row_major);
}

// void forward_function(
//     int * router_index,
//     int * sparse_index,
//     c10::Half* __restrict__ tokens,
//     c10::Half* __restrict__ weight,
//     c10::Half* output,
//     int * expert_count,
//     int total_token,
//     int in_hidden,
//     int out_hidden,
//     int n_expert,
//     const int GLOBAL_M,
//     const int TMAX
//     )
// {
//     const int BLOCK_SIZE_M = 32;
//     const int BLOCK_SIZE_K = 64;
//     const int BLOCK_SIZE_N = 32;
//     const int max_block = (GLOBAL_M -1 + BLOCK_SIZE_M)/BLOCK_SIZE_M;
//     dim3 blockDim(32*BLOCK_SIZE_M*BLOCK_SIZE_N/16/16);
//     dim3 gridDim(out_hidden/BLOCK_SIZE_N, max_block, n_expert);
//     BATCH_BLOCK_SPARSE_MATMUL_FP16<<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, GLOBAL_M, in_hidden, out_hidden, TMAX);
//     // BATCH_BLOCK_SPARSE_MATMUL_FP16_V2<<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, GLOBAL_M, in_hidden, out_hidden, TMAX);
//     // BATCH_BLOCK_SPARSE_MATMUL_FP16_V3<<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, GLOBAL_M, in_hidden, out_hidden, TMAX);
// }


void forward_function(
    int * router_index,
    int * sparse_index,
    c10::Half* __restrict__ tokens,
    c10::Half* __restrict__ weight,
    c10::Half* output,
    int * expert_count,
    int total_token,
    int in_hidden,
    int out_hidden,
    int n_expert,
    const int GLOBAL_M,
    const int TMAX
    )
{
    if(in_hidden==3072 && out_hidden==768){
        const int GLOBAL_K = 3072;
        const int GLOBAL_N = 768;
        assert(in_hidden==GLOBAL_K);
        assert(out_hidden==GLOBAL_N);
        const int BLOCK_SIZE_M = 64;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int max_block = (GLOBAL_M -1 + BLOCK_SIZE_M)/BLOCK_SIZE_M;
        dim3 blockDim(256);
        dim3 gridDim(out_hidden/BLOCK_SIZE_N, max_block, n_expert);
        // BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
        // BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE_V2<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, 4><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
        //cudaFuncSetAttribute(BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE_V3<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
	    BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE_V3<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
        //BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
        /*
        //////////Split Line for SPLIT-K
        */
        // const int SPLITK=2;
        // dim3 gridDim2(out_hidden/BLOCK_SIZE_N, max_block, n_expert*SPLITK);
        // BATCH_BLOCK_SPARSE_MATMUL_FP16_SPLITK_TEMPLATE<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, SPLITK><<<gridDim2, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);

    }else if(in_hidden==768 && out_hidden==3072){
        const int GLOBAL_K = 768;
        const int GLOBAL_N = 3072;
        assert(in_hidden==GLOBAL_K);
        assert(out_hidden==GLOBAL_N);
        const int BLOCK_SIZE_M = 64;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int max_block = (GLOBAL_M -1 + BLOCK_SIZE_M)/BLOCK_SIZE_M;
        dim3 blockDim(256);
        dim3 gridDim(out_hidden/BLOCK_SIZE_N, max_block, n_expert);
        // BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
	    BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE_V3<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
        //BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
        // V2 does not work for the small or middel shapes
        // BATCH_BLOCK_SPARSE_MATMUL_FP16_TEMPLATE_V2<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, 8><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
    }else{
        assert(0); // not implemented
    }

}


void forward_function(
    int * router_index,
    int * sparse_index,
    double* __restrict__ tokens,
    double* __restrict__ weight,
    double* output,
    int * expert_count,
    int total_token,
    int in_hidden,
    int out_hidden,
    int n_expert,
    const int GLOBAL_M,
    const int TMAX
    )
{
    // Not Implemented
}


void moe_sparse_convert_index(
    torch::Tensor router_index,
    torch::Tensor sparse_index,
    torch::Tensor expert_count
)
{
    // tokens: [total_token, in_hidden]
    cudaSetDevice(router_index.get_device());
    int total_token = router_index.size(0);
    int n_expert = expert_count.size(0);
    const int TMAX = sparse_index.size(1); // the max token each expert can take
    assert(router_index.size(0) == total_token);
    AT_DISPATCH_INTEGRAL_TYPES(router_index.type(), "seqlen_dynamic_sparse_attention", ([&]
                            { convert_index(
                                    router_index.data_ptr<int>(),
                                    sparse_index.data_ptr<int>(),
                                    expert_count.data_ptr<int>(),
                                    total_token,
                                    n_expert,
                                    TMAX
                                ); }));



}

at::Tensor moe_sparse_forward(
    torch::Tensor tokens,
    torch::Tensor weight,
    torch::Tensor router_index,
    torch::Tensor sparse_index,
    torch::Tensor expert_count,
    const int GLOBAL_M
)
{
    cudaSetDevice(router_index.get_device());
    // tokens: [total_token, in_hidden]
    int total_token = tokens.size(0);
    int in_hidden = tokens.size(1);
    // weight : [n_expert, out_hidden, in_hidden]
    int n_expert = weight.size(0);
    int out_hidden = weight.size(2);
    const int TMAX = sparse_index.size(1); // the max token each expert can take
    assert(router_index.size(0) == total_token);
    torch::Tensor output = torch::zeros({total_token, out_hidden}, tokens.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "seqlen_dynamic_sparse_attention", ([&]
                            { forward_function(
                                    router_index.data_ptr<int>(),
                                    sparse_index.data_ptr<int>(),
                                    tokens.data_ptr<scalar_t>(),
                                    weight.data_ptr<scalar_t>(),
                                    output.data_ptr<scalar_t>(),
                                    expert_count.data_ptr<int>(),
                                    total_token,
                                    in_hidden,
                                    out_hidden,
                                    n_expert,
                                    GLOBAL_M,
                                    TMAX
                                ); }));
    // AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, tokens.type(), "seqlen_dynamic_sparse_attention", ([&]
    //                         { forward_function(
    //                                 router_index.data_ptr<int>(),
    //                                 sparse_index.data_ptr<int>(),
    //                                 tokens.data_ptr<scalar_t>(),
    //                                 weight.data_ptr<scalar_t>(),
    //                                 output.data_ptr<scalar_t>(),
    //                                 expert_count.data_ptr<int>(),
    //                                 total_token,
    //                                 in_hidden,
    //                                 out_hidden,
    //                                 n_expert,
    //                                 GLOBAL_M,
    //                                 TMAX
    //                             ); }));

    return output;


}

/////////////////////////////////////////////////////////////////////////////////////////////
// Fused version
/////////////////////////////////////////////////////////////////////////////////////////////

template<
    const int GLOBAL_K,
    const int GLOBAL_N,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void BATCH_BLOCK_SPARSE_MATMUL_RELU_FP16_TEMPLATE(half* __restrict__  tokens, int* __restrict__  sparse_index, int* __restrict__  expert_count, half* __restrict__ B, half* __restrict__ C, const int TMAX)
{
    // const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    const int CPAD = 8;
    const int N_WARP = 8;
    const int WARP_PER_ROW = 4;
    assert(N_WARP * 32 == blockDim.x); // thread num: 256
    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    const int WARP_COUNT_N = BLOCK_SIZE_N / 16;
    const int WARP_COUNT_M = BLOCK_SIZE_M / 16;
    
    const int WARP_N_ROWS = N_WARP / WARP_PER_ROW; // 4
    const int WARP_ROW_STRIDE = WARP_COUNT_M / WARP_N_ROWS;
    const int WARP_COL_STRIDE = WARP_COUNT_N / WARP_PER_ROW;

    assert(WARP_COUNT_N % WARP_PER_ROW==0);
    assert(WARP_COUNT_M % WARP_N_ROWS==0);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id

    int wy = wid / WARP_PER_ROW;
    int wx = wid % WARP_PER_ROW;

    __shared__ half As[BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    __shared__ int m_index[BLOCK_SIZE_M];
    __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N + CPAD];
    half tmp_reg[8];
    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id * TMAX + n_token);
    if(index_start<index_end){        
        const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 8; // 1 float4 = 8 half
        const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 8;
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
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[WARP_COL_STRIDE];
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[WARP_ROW_STRIDE][WARP_COL_STRIDE];
        // reset to zero for all the accumulators
        #pragma unroll
        for(int i=0; i<WARP_ROW_STRIDE; i++){
            #pragma unroll
            for(int j=0; j<WARP_COL_STRIDE; j++){
                wmma::fill_fragment(frag_c[i][j], 0.0);
            }
        }
        // load to the m_index to shared memory
        if(tid<index_end-index_start)
            m_index[tid] = sparse_index[index_start+tid];

        __syncthreads();
        #pragma unroll
        for(int k_seq=0; k_seq<K/BLOCK_SIZE_K; k_seq++){
            // Fetch shared memory
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<index_end-index_start; k+=A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(tokens[m_index[k]*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]);
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq*BLOCK_SIZE_K+k)*N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
            }
            __syncthreads();
            #pragma unroll
            for(int k_step=0; k_step<BLOCK_SIZE_K/16; k_step++){
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    int y = wy * WARP_ROW_STRIDE + frag_y;
                    wmma::load_matrix_sync(frag_a[frag_y], &As[y*16][k_step*16], BLOCK_SIZE_K + APAD);                   
                }
                #pragma unroll
                for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                    int x = wx * WARP_COL_STRIDE + frag_x;
                    wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16][x*16], BLOCK_SIZE_N + BPAD);               
                }
                #pragma unroll
                for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
                    #pragma unroll
                    for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                        wmma::mma_sync(frag_c[frag_y][frag_x], frag_a[frag_y], frag_b[frag_x], frag_c[frag_y][frag_x]);                        
                    }
                }
            }

            __syncthreads();

        }

        //write back to the shared memory
        #pragma unroll
        for(int frag_y=0; frag_y<WARP_ROW_STRIDE; frag_y++){
            #pragma unroll
            for(int frag_x=0; frag_x<WARP_COL_STRIDE; frag_x++){
                int y = wy * WARP_ROW_STRIDE + frag_y;
                int x = wx * WARP_COL_STRIDE + frag_x;
                wmma::store_matrix_sync(&Cs[y*16][x*16], frag_c[frag_y][frag_x], BLOCK_SIZE_N+CPAD, wmma::mem_row_major);                        
            }
        }
        __syncthreads();
        for(int k=C_BLOCK_ROW_START; k<index_end-index_start; k+=C_TILE_ROW_STRIDE){
            FETCH_FLOAT4(tmp_reg[0]) = FETCH_FLOAT4(Cs[k][C_BLOCK_COL_START]);
            #pragma unroll
            for(int i=0;i<8;i++){
                tmp_reg[i] = max(tmp_reg[i], 0.0);
            }
            FETCH_FLOAT4(C[m_index[k] * N + bx * BLOCK_SIZE_N + C_BLOCK_COL_START]) = FETCH_FLOAT4(tmp_reg[0]);
        }
    }
    
}




void forward_function_with_relu(
    int * router_index,
    int * sparse_index,
    c10::Half* __restrict__ tokens,
    c10::Half* __restrict__ weight,
    c10::Half* output,
    int * expert_count,
    int total_token,
    int in_hidden,
    int out_hidden,
    int n_expert,
    const int GLOBAL_M,
    const int TMAX
    )
{   // onle the first layer of ffn can be fused with relu
    if(in_hidden==3072 && out_hidden==768){
        const int GLOBAL_K = 3072;
        const int GLOBAL_N = 768;
        assert(in_hidden==GLOBAL_K);
        assert(out_hidden==GLOBAL_N);
        const int BLOCK_SIZE_M = 64;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int max_block = (GLOBAL_M -1 + BLOCK_SIZE_M)/BLOCK_SIZE_M;
        dim3 blockDim(256);
        dim3 gridDim(out_hidden/BLOCK_SIZE_N, max_block, n_expert);
        BATCH_BLOCK_SPARSE_MATMUL_RELU_FP16_TEMPLATE_V3<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
        //BATCH_BLOCK_SPARSE_MATMUL_RELU_FP16_TEMPLATE<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
    }else if(in_hidden==768 && out_hidden==3072){
        const int GLOBAL_K = 768;
        const int GLOBAL_N = 3072;
        assert(in_hidden==GLOBAL_K);
        assert(out_hidden==GLOBAL_N);
        const int BLOCK_SIZE_M = 64;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 64;
        const int max_block = (GLOBAL_M -1 + BLOCK_SIZE_M)/BLOCK_SIZE_M;
        dim3 blockDim(256);
        dim3 gridDim(out_hidden/BLOCK_SIZE_N, max_block, n_expert);
        BATCH_BLOCK_SPARSE_MATMUL_RELU_FP16_TEMPLATE_V3<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);
        // BATCH_BLOCK_SPARSE_MATMUL_RELU_FP16_TEMPLATE<GLOBAL_K, GLOBAL_N, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>((half *)tokens, sparse_index, expert_count, (half *)weight, (half *)output, TMAX);

    }
    else{
        assert(0); // not implemented
    }

}
void forward_function_with_relu(
    int * router_index,
    int * sparse_index,
    double* __restrict__ tokens,
    double* __restrict__ weight,
    double* output,
    int * expert_count,
    int total_token,
    int in_hidden,
    int out_hidden,
    int n_expert,
    const int GLOBAL_M,
    const int TMAX
    )
{
    // Not Implemented
}

void forward_function_with_relu(
    int * router_index,
    int * sparse_index,
    float* tokens,
    float* weight,
    float* output,
    int * expert_count,
    int total_token,
    int in_hidden,
    int out_hidden,
    int n_expert,
    const int GLOBAL_M,
    const int TMAX
    )
{
    // Not implemented
}
at::Tensor moe_sparse_forward_with_relu(
    torch::Tensor tokens,
    torch::Tensor weight,
    torch::Tensor router_index,
    torch::Tensor sparse_index,
    torch::Tensor expert_count,
    const int GLOBAL_M
)
{
    cudaSetDevice(router_index.get_device());
    // tokens: [total_token, in_hidden]
    int total_token = tokens.size(0);
    int in_hidden = tokens.size(1);
    // weight : [n_expert, out_hidden, in_hidden]
    int n_expert = weight.size(0);
    int out_hidden = weight.size(2);
    const int TMAX = sparse_index.size(1); // the max token each expert can take
    assert(router_index.size(0) == total_token);
    torch::Tensor output = torch::zeros({total_token, out_hidden}, tokens.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "seqlen_dynamic_sparse_attention", ([&]
                            { forward_function_with_relu(
                                    router_index.data_ptr<int>(),
                                    sparse_index.data_ptr<int>(),
                                    tokens.data_ptr<scalar_t>(),
                                    weight.data_ptr<scalar_t>(),
                                    output.data_ptr<scalar_t>(),
                                    expert_count.data_ptr<int>(),
                                    total_token,
                                    in_hidden,
                                    out_hidden,
                                    n_expert,
                                    GLOBAL_M,
                                    TMAX
                                ); }));

    return output;


}
