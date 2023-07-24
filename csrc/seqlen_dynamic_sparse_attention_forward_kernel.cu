
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
using namespace std;
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
#define FETCH_HALF2(pointer) (reinterpret_cast<half2*>(&pointer))[0]
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

// #if (__CUDA_ARCH__ == 800)

template<
    const int GLOBAL_M,
    const int GLOBAL_K,
    const int GLOBAL_N,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int N_WARP
>
__global__ void BLOCK_SPARSE_MATMUL_OUT_FP16(
    half* __restrict__ A,
    half* __restrict__ B,
    half* __restrict__ C_val,
    int * seqlens,
    bool triangle
)
{
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    const int APAD = 8;
    const int BPAD = 8;
    const int CPAD = 8;
    // const int N_WARP = 8;
    const int WARP_PER_ROW = 2;
    assert(N_WARP * 32 == blockDim.x); // thread num: 256
    const int WARP_COUNT_N = BLOCK_SIZE_N / 16;
    const int WARP_COUNT_M = BLOCK_SIZE_M / 16;
    
    const int WARP_N_ROWS = N_WARP / WARP_PER_ROW; // 4
    const int WARP_ROW_STRIDE = WARP_COUNT_M / WARP_N_ROWS;
    const int WARP_COL_STRIDE = WARP_COUNT_N / WARP_PER_ROW;
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y + gridDim.y * blockIdx.z;
    A += GLOBAL_K * GLOBAL_M * head_idx;
    B += GLOBAL_K * GLOBAL_N * head_idx;
    C_val += GLOBAL_M * GLOBAL_N * head_idx;
    uint cur_seq_len = seqlens[batch_idx];
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id
    uint bx = (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N));
    uint by = (blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N));
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
    if(triangle && by<bx){
        // only compute the left triangle of the attention region
        return;
    }
    if (bx * BLOCK_SIZE_N < cur_seq_len && by * BLOCK_SIZE_M < cur_seq_len){
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
// #endif

__global__ void BLOCK_SPARSE_MATMUL_OUT_32_64_32(
    float* A,
    float* B,
    float* C_val,
    int * seqlens,
    int GLOBAL_M,
    int GLOBAL_K,
    int GLOBAL_N){
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
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y + gridDim.y * blockIdx.z; // launch config: block_idx, head_idx, batch_idx
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("hid:%d, bY:%d, bZ:%d , gdim:%d \n", head_idx, blockIdx.y, blockIdx.z, gridDim.y);
    // }
    A += M * K * head_idx;
    B += K * N * head_idx;
    C_val += GLOBAL_M * GLOBAL_N * head_idx;
    uint cur_seq_len = seqlens[batch_idx];

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N));
    uint by = (blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N));
    // uint bx = col_index[blockIdx.x]; // N
    // uint by = row_index[blockIdx.x]; // M

    if (bx * BLOCK_SIZE_N < cur_seq_len && by * BLOCK_SIZE_M < cur_seq_len){
        // if(threadIdx.x==0 ){
        //     printf("## bid:%d blockIdx.y:%d bx:%d by:%d seqlen:%d headid:%d\n", batch_idx, blockIdx.y, bx, by, cur_seq_len, head_idx);
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
        // uint blk_index = blockIdx.x;
        // uint intra_blk_index = block_index[blockIdx.x] % 2;
        // C_val += 32 * 32 * blk_index;
        // if(threadIdx.x==0 ){
        //     printf("#&& bid:%d blockIdx.y:%d bx:%d by:%d seqlen:%d headid:%d\n", batch_idx, blockIdx.y, (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N)), (blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N)), cur_seq_len, head_idx);
        // }
        C_val += ((blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_M + ty) * GLOBAL_N + (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_N + tx * 2;
        // C_val += ty * 32 + tx * 2;

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

        // C_val += 16 * 32;
        C_val += 16 * GLOBAL_N;
        *(float2*)C_val = c2[0];


    }
}

__global__ void BLOCK_SPARSE_MATMUL_OUT_NN_32_64_32(
    float* A,
    float* B,
    float* C_val,
    int * seqlens,
    int GLOBAL_M,
    int GLOBAL_K,
    int GLOBAL_N){
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
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y + gridDim.y * blockIdx.z; // launch config: block_idx, head_idx, batch_idx
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("hid:%d, bY:%d, bZ:%d , gdim:%d \n", head_idx, blockIdx.y, blockIdx.z, gridDim.y);
    // }
    A += M * K * head_idx;
    B += K * N * head_idx;
    C_val += GLOBAL_M * GLOBAL_N * head_idx;
    uint cur_seq_len = seqlens[batch_idx];

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N));
    uint by = (blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N));
    // uint bx = col_index[blockIdx.x]; // N
    // uint by = row_index[blockIdx.x]; // M

    if (by * BLOCK_SIZE_M < cur_seq_len){
        // if(threadIdx.x==0 ){
        //     printf("## bid:%d blockIdx.y:%d bx:%d by:%d seqlen:%d headid:%d\n", batch_idx, blockIdx.y, bx, by, cur_seq_len, head_idx);
        // }
        uint tx = tid % 16;
        uint ty = tid / 16;
        assert(THREAD_SIZE_K % 16 == 0);
        uint k = tx * 4;
        uint ori_offsetA00 = (by * 32 + ty) * K + k;
        uint ori_offsetA16 = ori_offsetA00 + K * 16;
        // uint ori_offsetB00 = (bx * 32 + ty) * K + k;
        // uint ori_offsetB16 = ori_offsetB00 + K * 16;
        uint ori_offsetB00 = bx * BLOCK_SIZE_N + tid / (BLOCK_SIZE_N/4) * N + (tid % (BLOCK_SIZE_N/4)) * 4;
        uint ori_offsetB16 = ori_offsetB00 + N * 32;
        uint tid224 = tid & 224;
        uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
        uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4
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

        for (int k_seq = 0; k_seq < (int)((cur_seq_len+63)/64); k_seq++)
        {
            uint offsetA00 = ori_offsetA00 + 64 * k_seq;
            uint offsetA16 = ori_offsetA16 + 64 * k_seq;
            // uint offsetB00 = ori_offsetB00 + 64 * k_seq;
            // uint offsetB16 = ori_offsetB16 + 64 * k_seq;
            uint offsetB00 = ori_offsetB00 + 64 * k_seq * N;
            uint offsetB16 = ori_offsetB16 + 64 * k_seq * N;
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

            // *(float*)&bShare[storAB + (0*32 +  0 + 1*65*32)*4] = b00.x;
            // *(float*)&bShare[storAB + (1*32 +  0 + 1*65*32)*4] = b00.y;
            // *(float*)&bShare[storAB + (2*32 +  0 + 1*65*32)*4] = b00.z;
            // *(float*)&bShare[storAB + (3*32 +  0 + 1*65*32)*4] = b00.w;
            // *(float*)&bShare[storAB + (0*32 + 16 + 1*65*32)*4] = b16.x;
            // *(float*)&bShare[storAB + (1*32 + 16 + 1*65*32)*4] = b16.y;
            // *(float*)&bShare[storAB + (2*32 + 16 + 1*65*32)*4] = b16.z;
            // *(float*)&bShare[storAB + (3*32 + 16 + 1*65*32)*4] = b16.w;
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

        // uint blk_index = block_index[blockIdx.x] / 2;
        // uint blk_index = blockIdx.x;
        // uint intra_blk_index = block_index[blockIdx.x] % 2;
        // C_val += 32 * 32 * blk_index;
        // if(threadIdx.x==0 ){
        //     printf("#&& bid:%d blockIdx.y:%d bx:%d by:%d seqlen:%d headid:%d\n", batch_idx, blockIdx.y, (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N)), (blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N)), cur_seq_len, head_idx);
        // }
        C_val += ((blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_M + ty) * GLOBAL_N + (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_N + tx * 2;
        // C_val += ty * 32 + tx * 2;

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

        // C_val += 16 * 32;
        C_val += 16 * GLOBAL_N;
        *(float2*)C_val = c2[0];


    }
}

template<
    const int GLOBAL_M,
    const int GLOBAL_N,
    const int MAX_LEN,
    const int ROW_TILE
>
__global__ void SPARSE_SOFTMAX_FP16(
    half* C_val,
    int* seqlens,
    bool triangle
){
    const int M = GLOBAL_M;
    const int N = GLOBAL_N;
    assert(MAX_LEN>=N);
    assert(ROW_TILE*32==blockDim.x);
    __shared__ half Cs[ROW_TILE][MAX_LEN];
    uint cur_seq_len = seqlens[blockIdx.z];
    uint tmp_seq_len = int((cur_seq_len+31)/32)*32;
    assert(M%32==0 && N%32==0);
    uint row_idx = blockIdx.x * ROW_TILE;
    uint bm = threadIdx.x / 32;
    uint bn = threadIdx.x % 32;
    uint head_idx = blockIdx.y + gridDim.y * blockIdx.z;
    C_val += M * N * head_idx;
    // half2 regC = {0, 0};
    half regSum = 0;
    half regMax = -1000;
    half tmp[2];
    int COL_START = bn * 8;
    int COL_STRIDE = 32 * 8;
    int line_end = cur_seq_len;
    if(triangle){
        line_end = min(line_end, row_idx + bm);
    }
    if(row_idx + bm < cur_seq_len){
        // need to perform the softmax
        #pragma unroll
        for(int pos=COL_START; pos<line_end; pos+=COL_STRIDE){
            FETCH_FLOAT4(Cs[bm][pos]) = FETCH_FLOAT4(C_val[OFFSET(row_idx + bm, pos, N)]);
        }
        __syncthreads();
        // scan once for the max value
        #pragma unroll
        for(int pos=bn; pos*2 < line_end; pos += 32){
            FETCH_HALF2(tmp) = FETCH_HALF2(Cs[bm][pos*2]);
            regMax = __hmax(regMax, tmp[0]);
            if(pos*2+1< line_end){
                regMax=__hmax(regMax, tmp[1]);
            }
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            regMax = max(regMax, __shfl_down_sync(FULL_MASK, regMax, offset));
        }
        regMax = __shfl_sync(FULL_MASK, regMax, 0);
        #pragma unroll
        for(int pos=bn; pos * 2 < line_end; pos += 32){
            half2 tmp = FETCH_HALF2(Cs[bm][pos*2]);
            regSum += hexp(tmp.x-regMax);
            if(pos*2+1< line_end){
                regSum+=hexp(tmp.y-regMax);
            }
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            regSum += __shfl_down_sync(FULL_MASK, regSum, offset);
        }
        regSum = __shfl_sync(FULL_MASK, regSum, 0);
        for (int index = bn; index < tmp_seq_len; index+=32) {
            int pos = (row_idx+bm) * N + index;
            if(index<line_end){
                C_val[pos] = hexp(C_val[pos]-regMax) / regSum;
            }else{
                C_val[pos] = 0;
            }

        }

    }


}
__global__ void SPARSE_SOFTMAX(
    float* C_val,
    int* seqlens,
    int M,
    int N,
    int row_tile
){
    /*
    description:
    each row of blocks is dealt with a thread group
    each block is 32x32
    */

    uint cur_seq_len = seqlens[blockIdx.z];
    uint tmp_seq_len = int((cur_seq_len+31)/32)*32;
    assert(M%32==0 && N%32==0);
    uint row_idx = blockIdx.x * row_tile;
    uint bm = threadIdx.x / 32;
    uint bn = threadIdx.x % 32;
    uint head_idx = blockIdx.y + gridDim.y * blockIdx.z;
    C_val += M * N * head_idx;
    float regC = 0.0f;
    float regSum = 0.0f;
    float regMax = -100000000.0;
    uint pos;

    if(row_idx + bm<cur_seq_len){
        for (int index = bn; index < cur_seq_len; index+=32) {
            pos = (row_idx+bm) * N + index;
            regMax = max(regMax, C_val[pos]);
        }        
        for (int offset = 16; offset > 0; offset /= 2) {
            regMax = max(regMax, __shfl_down_sync(FULL_MASK, regMax, offset));
        }
        regMax = __shfl_sync(FULL_MASK, regMax, 0);
        for (int index = bn; index < cur_seq_len; index+=32) {
            pos = (row_idx+bm) * N + index;
            regSum += expf(C_val[pos]-regMax);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            regSum += __shfl_down_sync(FULL_MASK, regSum, offset);
        }
        regSum = __shfl_sync(FULL_MASK, regSum, 0);
        // if(head_idx==0 && threadIdx.x==0 && blockIdx.x==0){
        //     printf("regSum: %f \n", regSum);
        // }
        for (int index = bn; index < tmp_seq_len; index+=32) {
            pos = (row_idx+bm) * N + index;
            // if(head_idx==3 && row_idx+bm == 22 && index ==23){
            //     printf("regSum: %f exp:%f regMax:%f\n", regSum, expf(C_val[pos]), regMax);
            // }
            if(index<cur_seq_len){
                // if(head_idx==0 && threadIdx.x==0 && blockIdx.x==0){
                //     printf("cur_seq_len: %d   tmp_seq_len:%d \n", cur_seq_len, tmp_seq_len);
                //     printf("tid:%d index:%d regSum:%f expf(val):%f \n", threadIdx.x, index, regSum, expf(C_val[pos]));
                // }
                C_val[pos] = expf(C_val[pos]-regMax) / regSum;
            }else{
                C_val[pos] = 0;
            }

        }

    }
    else if(row_idx + bm < tmp_seq_len && row_idx + bm < M){
        for (int index = bn; index < tmp_seq_len; index+=32) {
            pos = (row_idx+bm) * N + index;
            C_val[pos] = 0;
        }
    }

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
__global__ void BLOCK_SPARSE_MATMUL_SDD_FP16(
    half* __restrict__ A,
    half* __restrict__ B,
    half* __restrict__ C,
    int * seqlens,
    int HEAD_NUM,
    bool triangle
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
    int batch_idx = blockIdx.z/HEAD_NUM;
    // int head_idx = blockIdx.y + gridDim.y * blockIdx.z;
    int head_idx = blockIdx.z;
    A += GLOBAL_K * GLOBAL_M * head_idx;
    B += GLOBAL_K * GLOBAL_N * head_idx;
    C += GLOBAL_M * GLOBAL_N * head_idx;
    uint cur_seq_len = seqlens[batch_idx];
    int tid = threadIdx.x;
    int wid = tid >> 5; // warp id
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    int wy = wid / WARP_PER_ROW;
    int wx = wid % WARP_PER_ROW;
    __shared__ half As[2 * BLOCK_SIZE_M][BLOCK_SIZE_K + APAD];
    __shared__ half Bs[2 * BLOCK_SIZE_K][BLOCK_SIZE_N + BPAD];
    // __shared__ half Cs[BLOCK_SIZE_M][BLOCK_SIZE_N + CPAD];
    int As_base_addr = __cvta_generic_to_shared(&As[0][0]);
    int Bs_base_addr = __cvta_generic_to_shared(&Bs[0][0]);
    const int LD_AS = BLOCK_SIZE_K + APAD;
    const int LD_BS = BLOCK_SIZE_N + BPAD;
    const int LD_CS = BLOCK_SIZE_N + CPAD;
    int line_end = cur_seq_len;
    if(triangle){
        line_end = min(cur_seq_len, by * BLOCK_SIZE_M);
    }
    if (by * BLOCK_SIZE_M < cur_seq_len){
        // perform the computation
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
        // double buffer initialization
        const int k_seq=0;
        #pragma unroll
        for(int k=A_BLOCK_ROW_START; k<BLOCK_SIZE_M; k+=A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[k][A_BLOCK_COL_START]) = FETCH_FLOAT4(A[(by*BLOCK_SIZE_M+k)*K + k_seq*BLOCK_SIZE_K + A_BLOCK_COL_START]);
        }
        #pragma unroll
        for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
            // FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(bx*BLOCK_SIZE_N+k)*K + k_seq*BLOCK_SIZE_K + B_BLOCK_COL_START]);
            FETCH_FLOAT4(Bs[k][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[(k_seq * BLOCK_SIZE_K + k) * N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]);
        }
        #pragma unroll
        for(int k_seq=1; k_seq<(line_end + BLOCK_SIZE_K-1) / BLOCK_SIZE_K; k_seq++){
            int smem_select = (k_seq & 1) ^ 1;
            int smem_next = smem_select ^ 1;
            #pragma unroll
            for(int k=A_BLOCK_ROW_START; k<BLOCK_SIZE_M; k+=A_TILE_ROW_STRIDE){
                int load_a_s_addr = As_base_addr + sizeof(half) * OFFSET(k + smem_next * BLOCK_SIZE_M, A_BLOCK_COL_START, LD_AS); 
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_a_s_addr), "l"(&A[(by * BLOCK_SIZE_M+k)*K + k_seq * BLOCK_SIZE_K + A_BLOCK_COL_START]));
            }
            #pragma unroll
            for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
                int load_b_s_addr = Bs_base_addr + sizeof(half) * OFFSET(k + smem_next * BLOCK_SIZE_K, B_BLOCK_COL_START, LD_BS);
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_b_s_addr), "l"(&B[(k_seq * BLOCK_SIZE_K + k) * N + bx * BLOCK_SIZE_N + B_BLOCK_COL_START]));
                
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
                    wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16+smem_select*BLOCK_SIZE_K][x*16], LD_BS);               

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
        int smem_select = (((line_end + BLOCK_SIZE_K-1) / BLOCK_SIZE_K) & 1) ^ 1;
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
                wmma::load_matrix_sync(frag_b[frag_x], &Bs[k_step*16+smem_select*BLOCK_SIZE_K][x*16], LD_BS);               
            
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
                wmma::store_matrix_sync(&C[OFFSET(by * BLOCK_SIZE_M + y * 16, bx * BLOCK_SIZE_N + x * 16, N)], frag_c[frag_y][frag_x], N, wmma::mem_row_major);                        
            }
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
__global__ void BLOCK_SPARSE_MATMUL_SDD(float* A, float * B, float* C, int* seqlens,  int M, int K, int N, int HEAD_NUM){

    int by = blockIdx.y; // M
    int bx = blockIdx.x; // N
    int bz = blockIdx.z;
    int ty = threadIdx.y; 
    int tx = threadIdx.x;
    int head_idx = bz % HEAD_NUM;
    int batch_idx = bz / HEAD_NUM;
    int cur_seq_len = seqlens[batch_idx];
    // int tmp_seq_len = ()
    A = A + M * K * bz;
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

    // int index_start = csr_row[by], index_end = csr_row[by+1];
    if(by * BLOCK_SIZE_M < cur_seq_len){
        int index_start = 0, index_end = (cur_seq_len + BLOCK_SIZE_K-1) / BLOCK_SIZE_K;
        int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
        int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

        int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
        int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;
        const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
        const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;

        for(int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1){
            int col_pos = tile_block_idx * BLOCK_SIZE_K;
            #pragma unroll
            for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
                FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                    FETCH_FLOAT4(A[OFFSET(k + A_BLOCK_ROW_START + by*BLOCK_SIZE_M, A_BLOCK_COL_START + col_pos, K)]);
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

}

void seqlen_dynamic_forward_function(c10::Half* Q, c10::Half* K, c10::Half* V,
                    c10::Half * inter_result, int * seqlens,
                    int batch_size, int head_num, int max_seq_length, int hidden_dim, c10::Half* output, bool triangle=false)
{
// #if (__CUDA_ARCH__ == 800)

    CUDA_SAFE_CALL(cudaMemset(inter_result, 0, sizeof(half) * max_seq_length * max_seq_length * batch_size * head_num));
    if(max_seq_length==128 && hidden_dim==64){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 32;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        int block_nnz = max_seq_length * max_seq_length / BLOCK_SIZE_M / BLOCK_SIZE_N;
        const dim3 dimBlock(32*N_WARP);
        const dim3 dimGrid(block_nnz, head_num, batch_size);
        BLOCK_SPARSE_MATMUL_OUT_FP16<128, 64, 128, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<dimGrid, dimBlock>>>((half*)Q, (half*)K, (half*)inter_result, seqlens, triangle);
        const int ROWTILE = 8;
        const dim3 softBlock(32*ROWTILE);
        const dim3 softGrid(128/ROWTILE, head_num, batch_size);
        SPARSE_SOFTMAX_FP16<128, 128, 128, ROWTILE><<<softGrid, softBlock>>>((half*)inter_result, seqlens, triangle);
        const int BLOCK_SIZE_M_2 = 32;
        const int BLOCK_SIZE_K_2 = 32;
        const int BLOCK_SIZE_N_2 = 64;
        const int N_WARP_2 = (BLOCK_SIZE_M_2/16) * (BLOCK_SIZE_N_2/16);
        const dim3 dimBlock_2(32*N_WARP_2);
        const dim3 dimGrid_2(hidden_dim/BLOCK_SIZE_N_2, max_seq_length/BLOCK_SIZE_M_2, head_num*batch_size);
        BLOCK_SPARSE_MATMUL_SDD_FP16<128, 128, 64, BLOCK_SIZE_M_2, BLOCK_SIZE_K_2, BLOCK_SIZE_N_2, N_WARP_2><<<dimGrid_2, dimBlock_2>>>((half*)inter_result, (half*)V, (half*)output, seqlens, head_num, triangle);
    
    
    }else if(max_seq_length==256 && hidden_dim==64){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 32;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        int block_nnz = max_seq_length * max_seq_length / BLOCK_SIZE_M / BLOCK_SIZE_N;
        const dim3 dimBlock(32*N_WARP);
        const dim3 dimGrid(block_nnz, head_num, batch_size);
        BLOCK_SPARSE_MATMUL_OUT_FP16<256, 64, 256, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<dimGrid, dimBlock>>>((half*)Q, (half*)K, (half*)inter_result, seqlens, triangle);
        const int ROWTILE = 8;
        const dim3 softBlock(32*ROWTILE);
        const dim3 softGrid(256/ROWTILE, head_num, batch_size);
        SPARSE_SOFTMAX_FP16<256, 256, 256, ROWTILE><<<softGrid, softBlock>>>((half*)inter_result, seqlens, triangle);
        const int BLOCK_SIZE_M_2 = 32;
        const int BLOCK_SIZE_K_2 = 32;
        const int BLOCK_SIZE_N_2 = 64;
        const int N_WARP_2 = (BLOCK_SIZE_M_2/16) * (BLOCK_SIZE_N_2/16);
        const dim3 dimBlock_2(32*N_WARP_2);
        const dim3 dimGrid_2(hidden_dim/BLOCK_SIZE_N_2, max_seq_length/BLOCK_SIZE_M_2, head_num*batch_size);
        BLOCK_SPARSE_MATMUL_SDD_FP16<256, 256, 64, BLOCK_SIZE_M_2, BLOCK_SIZE_K_2, BLOCK_SIZE_N_2, N_WARP_2><<<dimGrid_2, dimBlock_2>>>((half*)inter_result, (half*)V, (half*)output, seqlens, head_num, triangle);


    }else if(max_seq_length==512 && hidden_dim==64){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 32;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        int block_nnz = max_seq_length * max_seq_length / BLOCK_SIZE_M / BLOCK_SIZE_N;
        const dim3 dimBlock(32*N_WARP);
        const dim3 dimGrid(block_nnz, head_num, batch_size);
        BLOCK_SPARSE_MATMUL_OUT_FP16<512, 64, 512, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<dimGrid, dimBlock>>>((half*)Q, (half*)K, (half*)inter_result, seqlens, triangle);
        const int ROWTILE = 8;
        const dim3 softBlock(32*ROWTILE);
        const dim3 softGrid(512/ROWTILE, head_num, batch_size);
        SPARSE_SOFTMAX_FP16<512, 512, 512, ROWTILE><<<softGrid, softBlock>>>((half*)inter_result, seqlens, triangle);
        const int BLOCK_SIZE_M_2 = 32;
        const int BLOCK_SIZE_K_2 = 32;
        const int BLOCK_SIZE_N_2 = 64;
        const int N_WARP_2 = (BLOCK_SIZE_M_2/16) * (BLOCK_SIZE_N_2/16);
        const dim3 dimBlock_2(32*N_WARP_2);
        const dim3 dimGrid_2(hidden_dim/BLOCK_SIZE_N_2, max_seq_length/BLOCK_SIZE_M_2, head_num*batch_size);
        BLOCK_SPARSE_MATMUL_SDD_FP16<512, 512, 64, BLOCK_SIZE_M_2, BLOCK_SIZE_K_2, BLOCK_SIZE_N_2, N_WARP_2><<<dimGrid_2, dimBlock_2>>>((half*)inter_result, (half*)V, (half*)output, seqlens, head_num, triangle);

    
    }else if(max_seq_length==1024 && hidden_dim==64){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 32;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        int block_nnz = max_seq_length * max_seq_length / BLOCK_SIZE_M / BLOCK_SIZE_N;
        const dim3 dimBlock(32*N_WARP);
        const dim3 dimGrid(block_nnz, head_num, batch_size);
        BLOCK_SPARSE_MATMUL_OUT_FP16<1024, 64, 1024, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<dimGrid, dimBlock>>>((half*)Q, (half*)K, (half*)inter_result, seqlens, triangle);
        const int ROWTILE = 8;
        const dim3 softBlock(32*ROWTILE);
        const dim3 softGrid(1024/ROWTILE, head_num, batch_size);
        SPARSE_SOFTMAX_FP16<1024, 1024, 1024, ROWTILE><<<softGrid, softBlock>>>((half*)inter_result, seqlens, triangle);
        const int BLOCK_SIZE_M_2 = 32;
        const int BLOCK_SIZE_K_2 = 32;
        const int BLOCK_SIZE_N_2 = 64;
        const int N_WARP_2 = (BLOCK_SIZE_M_2/16) * (BLOCK_SIZE_N_2/16);
        const dim3 dimBlock_2(32*N_WARP_2);
        const dim3 dimGrid_2(hidden_dim/BLOCK_SIZE_N_2, max_seq_length/BLOCK_SIZE_M_2, head_num*batch_size);
        BLOCK_SPARSE_MATMUL_SDD_FP16<1024, 1024, 64, BLOCK_SIZE_M_2, BLOCK_SIZE_K_2, BLOCK_SIZE_N_2, N_WARP_2><<<dimGrid_2, dimBlock_2>>>((half*)inter_result, (half*)V, (half*)output, seqlens, head_num, triangle);



    }else if(max_seq_length==4096 && hidden_dim==64){
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_K = 64;
        const int BLOCK_SIZE_N = 32;
        const int N_WARP = (BLOCK_SIZE_M/16) * (BLOCK_SIZE_N/16);
        int block_nnz = max_seq_length * max_seq_length / BLOCK_SIZE_M / BLOCK_SIZE_N;
        const dim3 dimBlock(32*N_WARP);
        const dim3 dimGrid(block_nnz, head_num, batch_size);
        BLOCK_SPARSE_MATMUL_OUT_FP16<4096, 64, 4096, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, N_WARP><<<dimGrid, dimBlock>>>((half*)Q, (half*)K, (half*)inter_result, seqlens, triangle);
        const int ROWTILE = 4;
        const dim3 softBlock(32*ROWTILE);
        const dim3 softGrid(4096/ROWTILE, head_num, batch_size);
        SPARSE_SOFTMAX_FP16<4096, 4096, 4096, ROWTILE><<<softGrid, softBlock>>>((half*)inter_result, seqlens, triangle);
        const int BLOCK_SIZE_M_2 = 32;
        const int BLOCK_SIZE_K_2 = 32;
        const int BLOCK_SIZE_N_2 = 64;
        const int N_WARP_2 = (BLOCK_SIZE_M_2/16) * (BLOCK_SIZE_N_2/16);
        const dim3 dimBlock_2(32*N_WARP_2);
        const dim3 dimGrid_2(hidden_dim/BLOCK_SIZE_N_2, max_seq_length/BLOCK_SIZE_M_2, head_num*batch_size);
        BLOCK_SPARSE_MATMUL_SDD_FP16<4096, 4096, 64, BLOCK_SIZE_M_2, BLOCK_SIZE_K_2, BLOCK_SIZE_N_2, N_WARP_2><<<dimGrid_2, dimBlock_2>>>((half*)inter_result, (half*)V, (half*)output, seqlens, head_num, triangle);

    
    }else{
        // please add more shape here
        assert(false);
    }
    // BLOCK_SPARSE_MATMUL_OUT_FP16<>
// #endif
}
void seqlen_dynamic_forward_function(double* Q, double* K, double* V,
                    double * inter_result, int * seqlens,
                    int batch_size, int head_num, int max_seq_length, int hidden_dim, double* output, bool triangle=false)
{

}

void seqlen_dynamic_forward_function(float* Q, float* K, float* V,
                    float * inter_result, int * seqlens,
                    int batch_size, int head_num, int max_seq_length, int hidden_dim, float* output, bool triangle=false)
{
    int block_nnz = max_seq_length * max_seq_length / 32 / 32;
    CUDA_SAFE_CALL(cudaMemset(inter_result, 0, sizeof(float) * max_seq_length * max_seq_length * batch_size * head_num));
    // already set to zero outside, no need to memset here
    //cudaMemset((void*)val, 0, (SPARSE_VAL_SIZE * HEAD_NUM) * batch_size);
    const dim3 dimBlock(256);
    const dim3 dimGrid(block_nnz, head_num, batch_size);
    BLOCK_SPARSE_MATMUL_OUT_32_64_32<<<dimGrid, dimBlock>>>(
        Q,
        K,
        inter_result,
        seqlens,
        max_seq_length, // M
        hidden_dim, // K
        max_seq_length // N
    );
    // printf("debug point 1\n");
    const int row_tile = 4;
    const dim3 softmax_dimBlock(row_tile*32);
    const dim3 softmax_dimGrid(max_seq_length/row_tile, head_num, batch_size);
    SPARSE_SOFTMAX<<<softmax_dimGrid, softmax_dimBlock>>>(
        inter_result,
        seqlens,
        max_seq_length,
        max_seq_length,
        row_tile
    );
    // printf("debug point 2\n");

    // // // sparse x dense
    // // // M: seq_length K: seq_length N:hidden dim
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_K = 4;
    const int THREAD_SIZE_N = 4;

    dim3 sdd_gridDim(hidden_dim/BLOCK_SIZE_N, max_seq_length/BLOCK_SIZE_M, head_num * batch_size);
    dim3 sdd_blockDim(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);
    BLOCK_SPARSE_MATMUL_SDD<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<sdd_gridDim, sdd_blockDim>>>(
        inter_result,
        V,
        output,
        seqlens,
        max_seq_length,
        max_seq_length,
        hidden_dim,
        head_num);
    // // printf("debug point 3\n");
    

}


at::Tensor seqlen_dynamic_sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor seqlens,
    int head_num,
    bool triangle=false
)
{
    cudaSetDevice(Q.get_device());
    // Q, K, V should have the same shape which is {batchsize, seq_length, hidden_dim}
    int batch_size = Q.size(0);
    // int head_num = Q.size(1);
    int max_seq_length = Q.size(2);
    int hidden_dim = Q.size(3);
    torch::Tensor output = torch::zeros({batch_size, head_num, max_seq_length, hidden_dim}, Q.options());
    // printf("bs:%d head_num:%d seq_len:%d hidden_dim:%d\n", batch_size, head_num, max_seq_length, hidden_dim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.type(), "seqlen_dynamic_sparse_attention", ([&]
                            { seqlen_dynamic_forward_function(
                                    Q.data_ptr<scalar_t>(),
                                    K.data_ptr<scalar_t>(),
                                    V.data_ptr<scalar_t>(),
                                    inter_result.data_ptr<scalar_t>(),
                                    seqlens.data_ptr<int>(),
                                    batch_size,
                                    head_num,
                                    max_seq_length,
                                    hidden_dim,
                                    output.data_ptr<scalar_t>(),
                                    triangle
                                ); }));

    // AT_DISPATCH_FLOATING_TYPES(Q.type(), "seqlen_dynamic_sparse_attention", ([&]
    //                         { seqlen_dynamic_forward_function(
    //                                 Q.data_ptr<float>(),
    //                                 K.data_ptr<float>(),
    //                                 V.data_ptr<float>(),
    //                                 inter_result.data_ptr<float>(),
    //                                 seqlens.data_ptr<int>(),
    //                                 batch_size,
    //                                 head_num,
    //                                 max_seq_length,
    //                                 hidden_dim,
    //                                 output.data_ptr<float>()
    //                             ); }));
    return output;
}




__global__ void BLOCK_SPARSE_MATMUL_TN_OUT_32_64_32(
    float* A,
    float* B,
    float* C_val,
    int * seqlens,
    int GLOBAL_M,
    int GLOBAL_K,
    int GLOBAL_N){
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
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y + gridDim.y * blockIdx.z; // launch config: block_idx, head_idx, batch_idx
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("hid:%d, bY:%d, bZ:%d , gdim:%d \n", head_idx, blockIdx.y, blockIdx.z, gridDim.y);
    // }
    A += M * K * head_idx;
    B += K * N * head_idx;
    C_val += GLOBAL_M * GLOBAL_N * head_idx;
    uint cur_seq_len = seqlens[batch_idx];

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N));
    uint by = (blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N));
    // uint bx = col_index[blockIdx.x]; // N
    // uint by = row_index[blockIdx.x]; // M

    if (by * BLOCK_SIZE_M < cur_seq_len ){
        // if(threadIdx.x==0 ){
        //     printf("## bid:%d blockIdx.y:%d bx:%d by:%d seqlen:%d headid:%d\n", batch_idx, blockIdx.y, bx, by, cur_seq_len, head_idx);
        // }
        uint tx = tid % 16;
        uint ty = tid / 16;
        assert(THREAD_SIZE_K % 16 == 0);
        uint k = tx * 4;
        uint ori_offsetA00 = by * BLOCK_SIZE_M + tid / (BLOCK_SIZE_M/4) * M + (tid % (BLOCK_SIZE_M/4)) * 4;;
        uint ori_offsetA16 = ori_offsetA00 + M * 32;
        uint ori_offsetB00 = bx * BLOCK_SIZE_N + tid / (BLOCK_SIZE_N/4) * N + (tid % (BLOCK_SIZE_N/4)) * 4;
        uint ori_offsetB16 = ori_offsetB00 + N * 32;

        uint tid224 = tid & 224;
        uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
        uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4
        uint storA = (tid * 4 + tid / (BLOCK_SIZE_M/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4
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

        for (int k_seq = 0; k_seq < (int)((cur_seq_len+63)/64); k_seq++)
        {
            uint offsetA00 = ori_offsetA00 + 64 * k_seq * M;
            uint offsetA16 = ori_offsetA16 + 64 * k_seq * M;
            uint offsetB00 = ori_offsetB00 + 64 * k_seq * N;
            uint offsetB16 = ori_offsetB16 + 64 * k_seq * N;

            float4 a00 = {0}, a16 = {0};
            float4 b00 = {0}, b16 = {0};
            a00 = __ldg((const float4*)(add_ptr_f(A, offsetA00)));
            a16 = __ldg((const float4*)(add_ptr_f(A, offsetA16)));
            b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
            b16 = __ldg((const float4*)(add_ptr_f(B, offsetB16)));
            // if(by==0 && bx== 0 && tid%8==0 && head_idx==0)
            //     printf("cur_seqlen:%d bx:%d by:%d tid:%d offsetB00:%d offsetB16:%d %f %f %f %f\n",cur_seq_len, bx, by, tid, offsetB00, offsetB16, b16.x, b16.y, b16.z, b16.w);
            __syncthreads();

            *(float*)&bShare[storA + (0*65*32)*4] = a00.x;
            *(float*)&bShare[storA + (0*65*32 + 1)*4] = a00.y;
            *(float*)&bShare[storA + (0*65*32 + 2)*4] = a00.z;
            *(float*)&bShare[storA + (0*65*32 + 3)*4] = a00.w;
            *(float*)&bShare[storA + (32*32 + 8*2 + 0*65*32)*4] = a16.x;
            *(float*)&bShare[storA + (32*32 + 8*2 + 0*65*32 + 1)*4] = a16.y;
            *(float*)&bShare[storA + (32*32 + 8*2 + 0*65*32 + 2)*4] = a16.z;
            *(float*)&bShare[storA + (32*32 + 8*2 + 0*65*32 + 3)*4] = a16.w;

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

        // uint blk_index = block_index[blockIdx.x] / 2;
        // uint blk_index = blockIdx.x;
        // uint intra_blk_index = block_index[blockIdx.x] % 2;
        // C_val += 32 * 32 * blk_index;
        // if(threadIdx.x==0 ){
        //     printf("#&& bid:%d blockIdx.y:%d bx:%d by:%d seqlen:%d headid:%d\n", batch_idx, blockIdx.y, (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N)), (blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N)), cur_seq_len, head_idx);
        // }
        C_val += ((blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_M + ty) * GLOBAL_N + (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_N + tx * 2;
        // C_val += ty * 32 + tx * 2;

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
        // if(by==0 && bx== 0 && tid==0 && head_idx==0)
        //     printf("write: %f %f\n", c2[0].x, c2[0].y);
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

        // C_val += 16 * 32;
        C_val += 16 * GLOBAL_N;
        *(float2*)C_val = c2[0];


    }
}

template<
    const int SOFTMAX_ROW_TILE,
    const int ROW_MAX_SIZE
>
__global__ void SPARSE_SOFTMAX_BACKWARD(float * inter_result, float *attn_grad, float* score_grad, int * seqlens, int M, int N, int head_num)
{
    // grad[i] = sum(-grad[j]*out[j]*out[i] if i!=j else grad[j]*(1-out[j])*out[j])
    
    // const int ROW_MAX_SIZE = 4096;
    const int WARP_SIZE = 32;
    __shared__ float shared_v[SOFTMAX_ROW_TILE*ROW_MAX_SIZE];
    __shared__ float shared_g[SOFTMAX_ROW_TILE*ROW_MAX_SIZE];
    
    float * vs = shared_v;
    float * gs = shared_g;
    const int bid = blockIdx.y / head_num;
    const int cur_seq_len = seqlens[bid];

    inter_result += blockIdx.y * M * N;
    attn_grad += blockIdx.y * M * N;
    score_grad += blockIdx.y * M * N;

    uint bm = threadIdx.x / WARP_SIZE + blockIdx.x * SOFTMAX_ROW_TILE; // lineid
    uint bn = threadIdx.x % WARP_SIZE;
    assert(cur_seq_len<ROW_MAX_SIZE);
    if(bm>cur_seq_len){
        return;
    }

    float regC = 0.0f;
    float regSum = 0.0f;
    // const int wtid = threadIdx.x % WARP_SIZE;
    vs += ROW_MAX_SIZE * int(threadIdx.x/WARP_SIZE);
    gs += ROW_MAX_SIZE * int(threadIdx.x/WARP_SIZE);
    // load from the global memory to the shared memory
    #pragma unroll
    for (int pos = bn; pos < cur_seq_len; pos+=WARP_SIZE) {
        // FIXME: there need one more for loop to handle the WARP_SIZE -> BLOCK_W
        int _index =  bm * N + pos;
        gs[pos] = attn_grad[_index];
        vs[pos] = inter_result[_index];
    }

    // calculate the regSum
    #pragma unroll
    for (int pos = bn; pos < cur_seq_len; pos+=WARP_SIZE) {
        regSum += gs[pos] * vs[pos];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        regSum += __shfl_down_sync(FULL_MASK, regSum, offset);
    }
    regSum = __shfl_sync(FULL_MASK, regSum, 0);
    // calculate the corresponding value for each element

    #pragma unroll
    for (int pos = bn; pos < cur_seq_len; pos+=WARP_SIZE) {
        vs[pos] = vs[pos] * gs[pos] - regSum* vs[pos];
    }

    // Write the results back to the global memory
    #pragma unroll
    for (int pos = bn; pos < cur_seq_len; pos+=WARP_SIZE) {
        // FIXME: there need one more for loop to handle the WARP_SIZE -> BLOCK_W
        int _index =  bm * N + pos;
        score_grad[_index] = vs[pos];
    }
}


void seqlen_backward_function(float * grad_in, float * Q, float* K, float* V, float* inter_result, int * seqlens, float* Q_grad, float*K_grad, float* V_grad, float * Attn_grad, float* Score_grad, int batchsize, int head_num, int hidden_dim, int q_seq_len, int k_seq_len)
{

    // grad_v: Batch x Head x k_seq_len, hidden_dim
    // Grad_V = Attn^T x grad_out
    // Attn: q_seq_len, k_seq_len
    const int grad_v_M = k_seq_len;
    const int grad_v_K = q_seq_len;
    const int grad_v_N = hidden_dim;
    dim3 gradv_dimGrid(grad_v_N/32 * grad_v_M/32, head_num, batchsize);
    dim3 gradv_dimBlock(256);
    BLOCK_SPARSE_MATMUL_TN_OUT_32_64_32<<<gradv_dimGrid, gradv_dimBlock>>>(inter_result, grad_in, V_grad, seqlens, grad_v_M, grad_v_K, grad_v_N);
    
    // Calculate the Attn_grad
    // Grad_Attn = grad_out x V^T
    // batch head q_seq k_seq = batch head q_seq hidden x (batch head k_seq hiddeh) 
    const int attn_M = q_seq_len;
    const int attn_K = hidden_dim;
    const int attn_N = k_seq_len;
    dim3 attn_dimGrid(attn_M /32 * attn_N/32, head_num, batchsize);
    dim3 attn_dimBlock(256);
    BLOCK_SPARSE_MATMUL_OUT_32_64_32<<<attn_dimGrid, attn_dimBlock>>>(grad_in, V, Attn_grad, seqlens, attn_M, attn_K, attn_N);
    if(k_seq_len<512){
        const int ROW_MAX = 512;
        const int ROW_TILE = 8;
        dim3 soft_dimBlock(32*ROW_TILE);
        dim3 soft_dimGrid(q_seq_len/ROW_TILE, head_num*batchsize);
        SPARSE_SOFTMAX_BACKWARD<ROW_TILE, ROW_MAX><<<soft_dimGrid, soft_dimBlock>>>(inter_result, Attn_grad, Score_grad, seqlens, q_seq_len, k_seq_len, head_num);
    }else if(k_seq_len < 1024){
        const int ROW_MAX = 1024;
        const int ROW_TILE = 4;
        dim3 soft_dimBlock(32*ROW_TILE);
        dim3 soft_dimGrid(q_seq_len/ROW_TILE, head_num*batchsize);
        SPARSE_SOFTMAX_BACKWARD<ROW_TILE, ROW_MAX><<<soft_dimGrid, soft_dimBlock>>>(inter_result, Attn_grad, Score_grad, seqlens, q_seq_len, k_seq_len, head_num);
    }else if(k_seq_len < 2048){
        const int ROW_MAX = 2048;
        const int ROW_TILE = 2;
        dim3 soft_dimBlock(32*ROW_TILE);
        dim3 soft_dimGrid(q_seq_len/ROW_TILE, head_num*batchsize);
        SPARSE_SOFTMAX_BACKWARD<ROW_TILE, ROW_MAX><<<soft_dimGrid, soft_dimBlock>>>(inter_result, Attn_grad, Score_grad, seqlens, q_seq_len, k_seq_len, head_num);
    }else if(k_seq_len<4096){
        const int ROW_MAX = 4096;
        const int ROW_TILE = 1;
        dim3 soft_dimBlock(32*ROW_TILE);
        dim3 soft_dimGrid(q_seq_len/ROW_TILE, head_num*batchsize);
        SPARSE_SOFTMAX_BACKWARD<ROW_TILE, ROW_MAX><<<soft_dimGrid, soft_dimBlock>>>(inter_result, Attn_grad, Score_grad, seqlens, q_seq_len, k_seq_len, head_num);
    }else{
        printf("Too long please use the extremly long kernel in the dynamic sparse attention!\n");
        assert(false);
    }
    // Calculate the Q_grad
    // Grad_Q = Grad_softmax * K
    // (q_seq_len x k_seq_len) x (k_seq_len x hidden_dim)
    const int grad_q_M = q_seq_len;
    const int grad_q_K = k_seq_len;
    const int grad_q_N = hidden_dim;
    dim3 gradq_dimGrid(grad_q_N/32 * grad_q_M/32, head_num, batchsize);
    dim3 gradq_dimBlock(256);
    BLOCK_SPARSE_MATMUL_OUT_NN_32_64_32<<<gradq_dimGrid, gradq_dimBlock>>>(Score_grad, K, Q_grad, seqlens, grad_q_M, grad_q_K, grad_q_N);

    // Calculate the K_grad
    // Grad_K = Grad_Score^T X Q
    const int grad_k_M = k_seq_len;
    const int grad_k_K = q_seq_len;
    const int grad_k_N = hidden_dim;
    dim3 gradk_dimGrid(grad_k_N/32 * grad_k_M/32, head_num, batchsize);
    dim3 gradk_dimBlock(256);
    BLOCK_SPARSE_MATMUL_TN_OUT_32_64_32<<<gradk_dimGrid, gradk_dimBlock>>>(Score_grad, Q, K_grad, seqlens, grad_k_M, grad_k_K, grad_k_N);
}

std::vector<at::Tensor> seqlen_dynamic_sparse_attention_backward(
    torch::Tensor grad,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor seqlens
    )
{
    // TODO: support backward for the seqlen_dynamic_sparse_attention
    cudaSetDevice(Q.get_device());
    int batch_size = Q.size(0);
    int head_num = Q.size(1);
    int q_seq_length = Q.size(2);
    int hidden_dim = Q.size(3);
    int k_seq_length = K.size(2);
    torch::Tensor Q_grad = torch::empty_like(Q);
    torch::Tensor K_grad = torch::empty_like(K);
    torch::Tensor V_grad = torch::empty_like(V);
    torch::Tensor Attn_grad = torch::empty_like(inter_result);
    torch::Tensor Score_grad = torch::zeros_like(inter_result);
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "our_sparse_attention", ([&]
        { seqlen_backward_function(
                grad.data_ptr<float>(),
                Q.data_ptr<float>(),
                K.data_ptr<float>(),
                V.data_ptr<float>(),
                inter_result.data_ptr<float>(),
                seqlens.data_ptr<int>(),
                Q_grad.data_ptr<float>(),
                K_grad.data_ptr<float>(),
                V_grad.data_ptr<float>(),
                Attn_grad.data_ptr<float>(),
                Score_grad.data_ptr<float>(),
                batch_size,
                head_num,
                hidden_dim,
                q_seq_length,
                k_seq_length
        ); }));

    return vector<at::Tensor>({Q_grad, K_grad, V_grad, Attn_grad, Score_grad});
}
