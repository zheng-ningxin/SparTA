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

__global__ void BLOCK_SPARSE_MATMUL_BIAS_OPENAI(
    float* A,
    float* B,
    float* bias,
    int * index,
    int GLOBAL_M,
    int GLOBAL_K,
    int GLOBAL_N,
    float* output){
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

    // A += M * K * blockIdx.z;
    // B += K * N * blockIdx.z;
    // output += M * N * blockIdx.z;
    // int batchid = blockIdx.z;
    // int cur_seq_len = seqlens[batchid];
    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    __shared__ int n_index[BLOCK_SIZE_N];
    __shared__ float bias_share[BLOCK_SIZE_N];
    char* bShare = (char*)fShare;
    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // N
    uint by = blockIdx.y; // M
    if(tid < BLOCK_SIZE_N && bx * BLOCK_SIZE_N + tid < N){
        n_index[tid] = index[bx * BLOCK_SIZE_N + tid];
    }
    uint n_pos;
    if(tid<BLOCK_SIZE_N && bx * BLOCK_SIZE_N + tid < N){
        bias_share[tid] = bias[n_index[tid]]; 
    }
    __syncthreads();

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offsetA00 = (by * 32 + ty) * K + k;
    uint ori_offsetA16 = ori_offsetA00 + K * 16;
    // uint ori_offsetB00 = (bx * 32 + ty) * K + k;
    // uint ori_offsetB16 = ori_offsetB00 + K * 16;
    // n_pos = bx * 32 + ty;
    uint ori_offsetB00 = n_index[ty] * K + k;
    uint ori_offsetB16 = n_index[ty + 16] * K + k;;

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
    // if(threadIdx.x==0){
    //     printf("bx:%d by:%d\n", bx, by);
    // }
    for (int k_seq = 0; k_seq < (int)(K/64); k_seq++)
    {
        // if(k_dim_mask[k_seq]!=1 && tid==0){
        //     printf("k_dim_mask:%d tid:%d bx:%d by:%d\n", k_dim_mask[tid], tid, bx, by);
        // }

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        uint offsetA00 = ori_offsetA00 + 64 * k_seq;
        uint offsetA16 = ori_offsetA16 + 64 * k_seq;
        a00 = __ldg((const float4*)(add_ptr_f(A, offsetA00)));
        a16 = __ldg((const float4*)(add_ptr_f(A, offsetA16)));
        uint offsetB00, offsetB16;
        if(bx * 32 + ty < N){
            offsetB00 = ori_offsetB00 + 64 * k_seq;
            b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
        }
        if(bx*32 + ty+16 < N){
            offsetB16 = ori_offsetB16 + 64 * k_seq;
            b16 = __ldg((const float4*)(add_ptr_f(B, offsetB16)));
        }
        // if(threadIdx.x== 0 && bx == 0 && by == 0){
        //     printf("offsetA00:%d offsetA16:%d offsetB00:%d offsetB16:%d a00:(%f, %f, %f, %f)\n", offsetA00, offsetA16, offsetB00, ori_offsetB16, a00.x, a00.y, a00.z, a00.w);
        // }
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
    float2 re1, re2;
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

    re1 = _add(c2[0], *(float2*)(bias_share+tx*2));
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

    // output += 16 * N;
    // *(float2*)C_val = c2[0];
    re2 = _add(c2[0], *(float2*)(bias_share+tx*2));
    if(blockIdx.x * BLOCK_SIZE_N + tx *2 < N){
        *output = re1.x;
        *(output+16*N) = re2.x;
    }
    if(blockIdx.x * BLOCK_SIZE_N + tx *2 +1 < N){
        *(output+1) = re1.y;
        *(output+16*N+1) = re2.y;
    }
}

__global__ void grad_w_kernel(
    float* A,
    float* B,
    float* C,
    int GLOBAL_M,
    int GLOBAL_K,
    int GLOBAL_N,
    int * index
)
{
    // ori_out_features is on the M dim
    // ori_in_features is on the N dim
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

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // N
    uint by = blockIdx.y; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;
    
        // uint ori_offsetA00 = (by * 32 + ty) * K + k;
        // uint ori_offsetA16 = ori_offsetA00 + K * 16;
        // uint ori_offsetB00 = (bx * 32 + ty) * K + k;
        // uint ori_offsetB16 = ori_offsetB00 + K * 16;
        uint ori_offsetA00 = tid / (BLOCK_SIZE_M/4) * M + by * BLOCK_SIZE_M + (tid % (BLOCK_SIZE_M/4)) * 4;
        uint ori_offsetA16 = ori_offsetA00 + M * 32;
        uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
        uint ori_offsetB16 = ori_offsetB00 + N * 32;

        uint tid224 = tid & 224;
        // uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
        uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
        uint loadB = ((tid >> 1) & 7) << 4;
        uint storA = (tid * 4 + tid / (BLOCK_SIZE_M/4) / 4 *2) * 4;;
        uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4

        loadA += (tid224 * 32) + (tid224 / 2);
        loadB += (tid224 * 32) + (tid224 / 2);

        // This keeps all prior logic outside of the loops.
        asm("mov.b32 %0, %0;" : "+r"(storA) : );
        // asm("mov.b32 %0, %0;" : "+r"(storB) : );
        asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
        asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

        float regC[8][4];
        for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
        regC[i][j] = 0.0f;

        for (int k_seq = 0; k_seq < (int)(K/64); k_seq++)
        {
        // uint offsetA00 = ori_offsetA00 + 64 * k_seq;
        // uint offsetA16 = ori_offsetA16 + 64 * k_seq;
        // uint offsetB00 = ori_offsetB00 + 64 * k_seq;
        // uint offsetB16 = ori_offsetB16 + 64 * k_seq;
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

        __syncthreads();

        // *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        // *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        // *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        // *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        // *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        // *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        // *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        // *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        // *(float*)&bShare[storAB + (0*32 +  0 + 1*65*32)*4] = b00.x;
        // *(float*)&bShare[storAB + (1*32 +  0 + 1*65*32)*4] = b00.y;
        // *(float*)&bShare[storAB + (2*32 +  0 + 1*65*32)*4] = b00.z;
        // *(float*)&bShare[storAB + (3*32 +  0 + 1*65*32)*4] = b00.w;
        // *(float*)&bShare[storAB + (0*32 + 16 + 1*65*32)*4] = b16.x;
        // *(float*)&bShare[storAB + (1*32 + 16 + 1*65*32)*4] = b16.y;
        // *(float*)&bShare[storAB + (2*32 + 16 + 1*65*32)*4] = b16.z;
        // *(float*)&bShare[storAB + (3*32 + 16 + 1*65*32)*4] = b16.w;
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
        // C_val += ((blockIdx.x / (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_M + ty) * GLOBAL_N + (blockIdx.x % (GLOBAL_N / BLOCK_SIZE_N)) * BLOCK_SIZE_N + tx * 2;
        // // C_val += ty * 32 + tx * 2;
        float * WC;
        WC = C + index[blockIdx.y * BLOCK_SIZE_M + ty] * N + blockIdx.x  * BLOCK_SIZE_N + tx * 2;
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

        // C_val += 16 * 32;
        WC = C + index[blockIdx.y * BLOCK_SIZE_M + ty +16 ] * N + blockIdx.x  * BLOCK_SIZE_N + tx * 2;
        // if(bx==0 && by==0){
        //     printf("tid:%d tx:%d ty:%d M-index:%d c2[0]:(%f %f)\n", threadIdx.x, tx, ty, index[blockIdx.y * BLOCK_SIZE_M + ty +16], c2[0].x, c2[0].y);
        // }
        *(float2*)WC = c2[0];
    

    
}

__global__ void BLOCK_SPARSE_MATMUL_NN_OPENAI(float* A,
                              float* B,
                              float* C,
                              int * index,
                              int GLOBAL_M,
                              int GLOBAL_K,
                              int GLOBAL_N
                              )
{
    /*
    grad_a = grad_c * weight 
            (M * N) * (N * K))
    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    // if(blockIdx.x == 0 && blockIdx.y ==0 && threadIdx.x ==0){
    //     printf("M:%d K:%d N:%d\n", M, K, N);
    // }
    // A += M * K * blockIdx.z;
    // B += K * N * blockIdx.z;
    // output += M * N * blockIdx.z;
    // int batchid = blockIdx.z;
    // int cur_seq_len = seqlens[batchid];
    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    __shared__ int k_dim_index[BLOCK_SIZE_K];
    // __shared__ float bias_share[BLOCK_SIZE_N];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    // if(by * BLOCK_SIZE_M < cur_seq_len){
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
        // uint ori_offsetB00 = (bx * 32 + ty) * K + k;
        // uint ori_offsetB16 = ori_offsetB00 + K * 16;
        // K x N -> ori_out_features, ori_in_features
        // uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * ori_in_features + bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
        // uint ori_offsetB16 = ori_offsetB00 + ori_in_features * 32;
        uint ori_offsetB00 =  bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
        uint ori_B_k_offset = tid / (BLOCK_SIZE_N/4);

        uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4


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
            if (tid<BLOCK_SIZE_K){
                k_dim_index[tid] = index[tid + 64 * k_seq];
            }
            __syncthreads();
            uint offsetA00 = ori_offsetA00 + 64 * k_seq;
            uint offsetA16 = ori_offsetA16 + 64 * k_seq;
            // uint offsetB00 = ori_offsetB00 + 64 * k_seq;
            // uint offsetB16 = ori_offsetB16 + 64 * k_seq;
            // uint offsetB00 = ori_offsetB00 + 64 * k_seq * ori_in_features;
            // uint offsetB16 = ori_offsetB16 + 64 * k_seq * ori_in_features;
            uint offsetB00 = ori_offsetB00 + k_dim_index[ori_B_k_offset] * N;
            uint offsetB16 = ori_offsetB00 + k_dim_index[ori_B_k_offset + 32] * N;
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


            *(float*)&bShare[storB + (1*65*32)*4] = b00.x;
            *(float*)&bShare[storB + (1*65*32 + 1)*4] = b00.y;
            *(float*)&bShare[storB + (1*65*32 + 2)*4] = b00.z;
            *(float*)&bShare[storB + (1*65*32 + 3)*4] = b00.w;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32)*4] = b16.x;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 1)*4] = b16.y;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 2)*4] = b16.z;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 3)*4] = b16.w;            __syncthreads();

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

        C += (blockIdx.y * BLOCK_SIZE_M + ty) * N + blockIdx.x * BLOCK_SIZE_N + tx *2;
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
        *(float2*)C = c2[0];
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

        C += 16 * N;
        // *(float2*)C_val = c2[0];
        *(float2*)C = c2[0];

    // }
    
}

__global__ void BLOCK_SPARSE_MATMUL_NN_BIAS_OPENAI(float* A,
                              float* B,
                              float* bias,
                              int * index,
                              int GLOBAL_M,
                              int GLOBAL_K,
                              int GLOBAL_N,
                              float* C
                              )
{

    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;
    // if(blockIdx.x == 0 && blockIdx.y ==0 && threadIdx.x ==0){
    //     printf("M:%d K:%d N:%d\n", M, K, N);
    // }
    // A += M * K * blockIdx.z;
    // B += K * N * blockIdx.z;
    // output += M * N * blockIdx.z;
    // int batchid = blockIdx.z;
    // int cur_seq_len = seqlens[batchid];
    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    __shared__ int k_dim_index[BLOCK_SIZE_K];
    __shared__ float bias_share[BLOCK_SIZE_N];
    // __shared__ float bias_share[BLOCK_SIZE_N];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    // if(by * BLOCK_SIZE_M < cur_seq_len){
        // if(threadIdx.x==0 && blockIdx.z==1 && by==0 && bx==0){
        //     printf("by:%d bx:%d bz:%d\n", by, bx, blockIdx.z);
        // }
        // uint bx = n_index[blockIdx.x]; // N
        // uint by = m_index[blockIdx.x]; // M
        if(tid<BLOCK_SIZE_N){
            bias_share[tid] = bias[bx * BLOCK_SIZE_N + tid]; 
        }
        uint tx = tid % 16;
        uint ty = tid / 16;
        assert(THREAD_SIZE_K % 16 == 0);
        uint k = tx * 4;

        uint ori_offsetA00 = (by * 32 + ty) * K + k;
        uint ori_offsetA16 = ori_offsetA00 + K * 16;
        // uint ori_offsetB00 = (bx * 32 + ty) * K + k;
        // uint ori_offsetB16 = ori_offsetB00 + K * 16;
        // K x N -> ori_out_features, ori_in_features
        // uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * ori_in_features + bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
        // uint ori_offsetB16 = ori_offsetB00 + ori_in_features * 32;
        uint ori_offsetB00 =  bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
        uint ori_B_k_offset = tid / (BLOCK_SIZE_N/4);

        uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4


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
            if (tid<BLOCK_SIZE_K){
                k_dim_index[tid] = index[tid + 64 * k_seq];
            }
            __syncthreads();
            uint offsetA00 = ori_offsetA00 + 64 * k_seq;
            uint offsetA16 = ori_offsetA16 + 64 * k_seq;
            // uint offsetB00 = ori_offsetB00 + 64 * k_seq;
            // uint offsetB16 = ori_offsetB16 + 64 * k_seq;
            // uint offsetB00 = ori_offsetB00 + 64 * k_seq * ori_in_features;
            // uint offsetB16 = ori_offsetB16 + 64 * k_seq * ori_in_features;
            uint offsetB00 = ori_offsetB00 + k_dim_index[ori_B_k_offset] * N;
            uint offsetB16 = ori_offsetB00 + k_dim_index[ori_B_k_offset + 32] * N;
            float4 a00 = {0}, a16 = {0};
            float4 b00 = {0}, b16 = {0};
            a00 = __ldg((const float4*)(add_ptr_f(A, offsetA00)));
            a16 = __ldg((const float4*)(add_ptr_f(A, offsetA16)));
            b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
            b16 = __ldg((const float4*)(add_ptr_f(B, offsetB16)));
            // if(tid==0 && bx==0 && by==0){
            //     printf("tid:%d offsetB00:%d b00:(%f %f %f %f)\n", tid, offsetB00, b00.x, b00.y, b00.z, b00.w);
            // }
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
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 3)*4] = b16.w;            __syncthreads();

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

        C += (blockIdx.y * BLOCK_SIZE_M + ty) * N + blockIdx.x * BLOCK_SIZE_N + tx *2;
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
        // *(float2*)C = c2[0];
        *(float2*)C = _add(c2[0], *(float2*)(bias_share+tx*2));
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

        C += 16 * N;
        // *(float2*)C_val = c2[0];
        // *(float2*)C = c2[0];
        *(float2*)C = _add(c2[0], *(float2*)(bias_share+tx*2));

    // }
    
}


void backward_function( float * activation,
                        float * weight,
                        float * grad_out,
                        int * index,
                        int M,
                        int K,
                        int N,
                        int ori_out_features,
                        float * a_grad,
                        float * w_grad
                        )
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;
    //w_grad(N*K) = grad_c^T(N*M) X activation(M*K)
    dim3 w_block_dim(256);
    dim3 w_grid_dim(K/BLOCK_SIZE_N, N/BLOCK_SIZE_M);
    grad_w_kernel<<<w_grid_dim, w_block_dim>>>(grad_out, activation, w_grad, N, M, K, index);
    //a_grad(M*K) = grad_c(MxN) * weight(NxK)
    dim3 a_grid_dim(K/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    BLOCK_SPARSE_MATMUL_NN_OPENAI<<<a_grid_dim, w_block_dim>>>(grad_out, weight, a_grad, index, M, N, K);
}


void forward_function(  float * activation,
    float* weight,
    float* bias,
    int * index,
    int M,
    int K,
    int N,
    float* output
)
{
    // dense x dense^T -> sparse output
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;

    dim3 gridDim((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(256);

    BLOCK_SPARSE_MATMUL_BIAS_OPENAI<<<gridDim, blockDim>>>(activation, weight, bias, index, M, K, N, output);

}
void indim_forward_function(  float * activation,
    float* weight,
    float* bias,
    int * index,
    int M,
    int K,
    int N,
    float* output
)
{
    // dense x dense^T -> sparse output
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;

    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(256);

    BLOCK_SPARSE_MATMUL_NN_BIAS_OPENAI<<<gridDim, blockDim>>>(activation, weight, bias, index, M, K, N, output);

}
at::Tensor outdim_dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor index,
    torch::Tensor bias
)
{
    cudaSetDevice(activation.get_device());   
    int batch_size = activation.size(0);
    int seq_len = activation.size(1);
    int in_hidden = activation.size(2);
    assert(in_hidden==weight.size(1));
    int out_hidden = index.size(0); // NOTE:
    torch::Tensor output = torch::empty({batch_size, seq_len, out_hidden}, activation.options());
    AT_DISPATCH_FLOATING_TYPES(activation.type(), "seqlen_dynamic_sparse_linear", ([&]
        {       forward_function(
                activation.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                index.data_ptr<int>(),
                batch_size * seq_len,
                in_hidden,
                out_hidden,
                output.data_ptr<float>()
            ); }));
    return output;

}

std::vector<at::Tensor> outdim_dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor grad_c,
    torch::Tensor index
)
{
    cudaSetDevice(activation.get_device());   
    int batch_size = activation.size(0);
    int seq_len = activation.size(1);
    int in_hidden = activation.size(2);
    assert(in_hidden==weight.size(1));
    int out_hidden = index.size(0); // NOTE:
    int ori_out_hidden = weight.size(0);
    torch::Tensor w_grad = torch::zeros_like(weight);
    torch::Tensor a_grad = torch::empty_like(activation);
    AT_DISPATCH_FLOATING_TYPES(activation.type(), "seqlen_dynamic_sparse_linear", ([&]
    {       backward_function(
            activation.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_c.data_ptr<float>(),
            index.data_ptr<int>(),
            batch_size * seq_len,
            in_hidden,
            out_hidden,
            ori_out_hidden,
            a_grad.data_ptr<float>(),
            w_grad.data_ptr<float>()
        ); }));
    std::vector<at::Tensor> grads({a_grad, w_grad});
    return grads;
}

at::Tensor indim_dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor index,
    torch::Tensor bias
)
{
    cudaSetDevice(activation.get_device());   
    int batch_size = activation.size(0);
    int seq_len = activation.size(1);
    int in_hidden = activation.size(2);
    assert(in_hidden==index.size(0));
    int out_hidden = weight.size(1); // NOTE: the weight has been transposed
    torch::Tensor output = torch::empty({batch_size, seq_len, out_hidden}, activation.options());
    AT_DISPATCH_FLOATING_TYPES(activation.type(), "seqlen_dynamic_sparse_linear", ([&]
        {       indim_forward_function(
                activation.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                index.data_ptr<int>(),
                batch_size * seq_len,
                in_hidden,
                out_hidden,
                output.data_ptr<float>()
            ); }));
    return output;

}

void indim_backward_function( float * activation,
                        float * weight,
                        float * grad_out,
                        int * index,
                        int M,
                        int K,
                        int N,
                        int ori_in_features,
                        float * a_grad,
                        float * w_grad
                        )
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;
    //w_grad(K*N) = activation^T(K*M) * grad_c(M*N)
    dim3 w_block_dim(256);
    // the size of w is K x N
    dim3 w_grid_dim(N/BLOCK_SIZE_N, K/BLOCK_SIZE_M);
    grad_w_kernel<<<w_grid_dim, w_block_dim>>>(activation, grad_out, w_grad, K, M, N, index);
    // //a_grad(M*K) = grad_c(MxN) * weight(NxK)
    // dim3 a_grid_dim(K/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    // BLOCK_SPARSE_MATMUL_NN_OPENAI<<<a_grid_dim, w_block_dim>>>(grad_out, weight, a_grad, index, M, N, K);
}

std::vector<at::Tensor> indim_dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor grad_c,
    torch::Tensor index
)
{
    cudaSetDevice(activation.get_device());   
    int batch_size = activation.size(0);
    int seq_len = activation.size(1);
    int in_hidden = activation.size(2);
    assert(in_hidden==index.size(0));
    int out_hidden = weight.size(1); // NOTE: the weight has been transposed
    int ori_in_hidden = weight.size(0);
    torch::Tensor w_grad = torch::zeros_like(weight);
    torch::Tensor a_grad = torch::empty_like(activation);
    AT_DISPATCH_FLOATING_TYPES(activation.type(), "seqlen_dynamic_sparse_linear", ([&]
    {       indim_backward_function(
            activation.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_c.data_ptr<float>(),
            index.data_ptr<int>(),
            batch_size * seq_len,
            in_hidden,
            out_hidden,
            ori_in_hidden,
            a_grad.data_ptr<float>(),
            w_grad.data_ptr<float>()
        ); }));
    std::vector<at::Tensor> grads({a_grad, w_grad});
    return grads;

}