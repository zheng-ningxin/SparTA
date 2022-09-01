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

// __device__ __forceinline__ void atomic_add(float* x, float y) {atomicAdd(&x->x, y.x); atomicAdd(&x.y, y.y)};

__global__ void BATCH_BLOCK_SPARSE_MATMUL(float* A_val, int* A_row, int* A_col, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int SPARSE_VAL_SIZE)
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

    A_val += SPARSE_VAL_SIZE * blockIdx.z;
    B += K * N * blockIdx.z;
    C += M * N * blockIdx.z;

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

    uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + by * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offsetB16 = ori_offsetB00 + N * 32;
    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4

    // B is stored in sparse format, thus, should be dealt with differently
    uint offsetA00 = A_row[bx] * BLOCK_SIZE_M * BLOCK_SIZE_K + ty * BLOCK_SIZE_K + k;
    uint offsetA16 = offsetA00 + BLOCK_SIZE_K * 16;

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

    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    int index_start = A_row[bx], index_end = A_row[bx+1];
    // if(blockIdx.z == 1 && blockIdx.x==0 && blockIdx.y == 0){
    //     printf("bx:%d by%d bz:%d index_start:%d index_end:%d Sparse_VAL_SIZE:%d\n", blockIdx.x, blockIdx.y, blockIdx.z, index_start, index_end, SPARSE_VAL_SIZE);
    // }
    for(int bcsr_col_idx = index_start; bcsr_col_idx < index_end; bcsr_col_idx += 1)
    {
        uint offsetB00 = ori_offsetB00 + 64 * A_col[bcsr_col_idx] * N;
        uint offsetB16 = ori_offsetB16 + 64 * A_col[bcsr_col_idx] * N;

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        a00 = __ldg((const float4*)(add_ptr_f(A_val, offsetA00)));
        a16 = __ldg((const float4*)(add_ptr_f(A_val, offsetA16)));
        b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
        b16 = __ldg((const float4*)(add_ptr_f(B, offsetB16)));

        offsetA00 += BLOCK_SIZE_M * BLOCK_SIZE_K;
        offsetA16 += BLOCK_SIZE_M * BLOCK_SIZE_K;

        __syncthreads();
        // if(blockIdx.z == 1 && blockIdx.x==0 && blockIdx.y == 0){
        //     printf("a00.x:%f a00.y:%f, a16.x:%f a16.y:%f\n", a00.x, a00.y, a16.x, a16.y);
        //     printf("b00.x:%f b00.y:%f, b16.x:%f b16.y:%f\n", b00.x, b00.y, b16.x, b16.y);
        // }
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
    C += (bx * BLOCK_SIZE_M + ty) * N + (by * BLOCK_SIZE_N + tx * 2);

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

    *(float2*)C = c2[0];
    // if(blockIdx.z == 1 && blockIdx.x==0 && blockIdx.y == 0){
    //     printf("c2:%f \n", *(float*)(&c2[0]));
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
    *(float2*)C = c2[0];

    
}

__global__ void BATCH_BLOCK_SPARSE_MATMUL_CONDENSE(float* A_val, int* A_row, int* A_col, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int BLOCK_H, int BLOCK_W, int SPARSE_VAL_SIZE)
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

    A_val += SPARSE_VAL_SIZE * blockIdx.z;
    B += K * N * blockIdx.z;
    C += M * N * blockIdx.z;

    assert(blockDim.x % 32 == 0);
    assert(BLOCK_SIZE_K % BLOCK_W==0);
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

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    // uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + by * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    // uint ori_offsetB16 = ori_offsetB00 + N * 32;
    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4

    // B is stored in sparse format, thus, should be dealt with differently
    // uint offsetA00 = A_row[bx] * BLOCK_SIZE_M * BLOCK_SIZE_K + ty * BLOCK_SIZE_K + k;
    // uint offsetA16 = offsetA00 + BLOCK_SIZE_K * 16;
    uint offsetA00 = A_row[bx] * BLOCK_SIZE_M * BLOCK_W + tid % (BLOCK_SIZE_M/4) * 4 + BLOCK_SIZE_M * (tid/(BLOCK_SIZE_M/4));
    uint offsetA32 = offsetA00 + 32 * BLOCK_SIZE_M;
    // uint ori_offset_B00 = tid / (BLOCK_SIZE_N/4) * N + by * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offset_B00 = by * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
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

    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    int index_start = A_row[bx], index_end = A_row[bx+1];
    float4 const0 = {0};
    int round = (index_end - index_start-1+BLOCK_SIZE_K/BLOCK_W)/(BLOCK_SIZE_K/BLOCK_W);
    // if(threadIdx.x==0)
    //     printf("bx:%d by:%d round:%d\n", bx, by, round);
    for(int rid =0; rid<round;rid++)
    // for(int bcsr_col_idx = index_start; bcsr_col_idx < index_end; bcsr_col_idx += 1)
    {
        uint k_offset = tid / (BLOCK_SIZE_N/4) + rid * BLOCK_SIZE_K;
        uint k_offset32 = k_offset + 32;
        // offsetA00 = offsetA00 + BLOCK_SIZE_K * BLOCK_SIZE_M; 
        // offsetA32 = offsetA32 + BLOCK_SIZE_K * BLOCK_SIZE_M;
        uint _pos = (k_offset / BLOCK_W);
        uint _pos32 = (k_offset32/BLOCK_W);
        uint offsetB00 = (A_col[index_start+_pos]+k_offset%BLOCK_W) * N + ori_offset_B00;
        uint offsetB32 = (A_col[index_start+_pos32]+k_offset32%BLOCK_W) * N + ori_offset_B00;
        // uint offsetB00 = ori_offsetB00 + 64 * A_col[bcsr_col_idx] * N;
        // uint offsetB16 = ori_offsetB16 + 64 * A_col[bcsr_col_idx] * N;
        // if(bx==0 && by == 0 &&  threadIdx.x==0){
        //     printf("_pos:%d _pos32:%d index_end-index_start:%d\n", _pos, _pos32, index_end-index_start);
        // }
        float4 a00 = {0,0,0,0}, a16 = {0,0,0,0};
        float4 b00 = {0,0,0,0}, b16 = {0,0,0,0};
        if(_pos<index_end-index_start){
            a00 = __ldg((const float4*)(add_ptr_f(A_val, offsetA00)));
            b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
        }
        if(_pos32<index_end-index_start){
            a16 = __ldg((const float4*)(add_ptr_f(A_val, offsetA32)));
            b16 = __ldg((const float4*)(add_ptr_f(B, offsetB32)));
        }
        offsetA00 += BLOCK_SIZE_M * BLOCK_SIZE_K;
        offsetA32 += BLOCK_SIZE_M * BLOCK_SIZE_K;

        __syncthreads();

        // *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        // *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        // *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        // *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        // *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        // *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        // *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        // *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;
        *(float*)&bShare[storB + (0*65*32)*4] = a00.x;
        *(float*)&bShare[storB + (0*65*32 + 1)*4] = a00.y;
        *(float*)&bShare[storB + (0*65*32 + 2)*4] = a00.z;
        *(float*)&bShare[storB + (0*65*32 + 3)*4] = a00.w;
        *(float*)&bShare[storB + (32*32 + 8*2 + 0*65*32)*4] = a16.x;
        *(float*)&bShare[storB + (32*32 + 8*2 + 0*65*32 + 1)*4] = a16.y;
        *(float*)&bShare[storB + (32*32 + 8*2 + 0*65*32 + 2)*4] = a16.z;
        *(float*)&bShare[storB + (32*32 + 8*2 + 0*65*32 + 3)*4] = a16.w;

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
    C += (bx * BLOCK_SIZE_M + ty) * N + (by * BLOCK_SIZE_N + tx * 2);

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

    *(float2*)C = c2[0];

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
    *(float2*)C = c2[0];

    
}


void openai_bmm_32_64_32_launch(float* A_val, int* A_row, int* A_col, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int SPARSE_VAL_SIZE, int batchsize)
{
    const dim3 dimBlock(256);
    dim3 dimGrid(GLOBAL_M/32, GLOBAL_N/32, batchsize);
    // printf("Batchsize:%d\n", batchsize);
    BATCH_BLOCK_SPARSE_MATMUL<<<dimGrid, dimBlock>>>(A_val, A_row, A_col, B, C, GLOBAL_M, GLOBAL_K, GLOBAL_N, SPARSE_VAL_SIZE);
}

void openai_bmm_32_64_32_condense_launch(float* A_val, int* A_row, int* A_col, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int BLOCK_H, int BLOCK_W, int SPARSE_VAL_SIZE, int batchsize)
{
    const dim3 dimBlock(256);
    assert(BLOCK_H==32);
    dim3 dimGrid(GLOBAL_M/32, GLOBAL_N/32, batchsize);
    // printf("Batchsize:%d\n", batchsize);
    BATCH_BLOCK_SPARSE_MATMUL_CONDENSE<<<dimGrid, dimBlock>>>(A_val, A_row, A_col, B, C, GLOBAL_M, GLOBAL_K, GLOBAL_N, BLOCK_H, BLOCK_W, SPARSE_VAL_SIZE);
}

__global__ void BATCH_BLOCK_SPARSE_MATMUL_CONDENSE_DIM_M(float* A_val, int* A_row, int* A_col, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int BLOCK_H, int BLOCK_W, int SPARSE_VAL_SIZE)
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

    A_val += SPARSE_VAL_SIZE * blockIdx.z;
    B += K * N * blockIdx.z;
    C += M * N * blockIdx.z;

    assert(blockDim.x % 32 == 0);
    assert(BLOCK_SIZE_K % BLOCK_W==0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    __shared__ int m_index[BLOCK_SIZE_M];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // M
    uint by = blockIdx.y; // K

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; 

    uint ori_offset_A00 = A_row[by] * BLOCK_H * BLOCK_SIZE_K + k;
    uint ori_offset_B00 = (by * BLOCK_SIZE_K + tid / (BLOCK_SIZE_N/4)) * N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offset_B32 = ori_offset_B00 + 32 * N;
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


    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    float4 const0 = {0};
    float4 a00 = {0,0,0,0}, a16 = {0,0,0,0};


    int index_start = A_row[by] + bx * BLOCK_SIZE_M, index_end = min(A_row[by+1], index_start + BLOCK_SIZE_M);
    if (index_start < index_end){

        if(tid<index_end-index_start){
            m_index[tid] = A_col[tid+index_start];
        }
        // if(threadIdx.x == 0){
        //     printf("bx:%d by:%d index_start:%d index_end:%d\n", bx, by, index_start, index_end);
        // }

        // Load the A_global to A_shared
        __syncthreads();


        for(int block_n_id =0; block_n_id < N/BLOCK_SIZE_N; block_n_id++)
        {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] = 0.0f;
            uint offsetA00 = 0;
            uint offsetA16 = 0;
            if(ty<index_end-index_start){
                offsetA00 =(bx * BLOCK_SIZE_M + ty) * BLOCK_H * BLOCK_SIZE_K  + ori_offset_A00;
                a00 = __ldg((const float4*)(add_ptr_f(A_val, offsetA00)));
            }
            if(ty+16<index_end-index_start){
                offsetA16 = (bx * BLOCK_SIZE_M + ty + 16) * BLOCK_H * BLOCK_SIZE_K  + ori_offset_A00;
                a16 = __ldg((const float4*)(add_ptr_f(A_val, offsetA16)));
            }
            *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
            *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
            *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
            *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
            *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
            *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
            *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
            *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;
            uint offsetB00 = block_n_id * BLOCK_SIZE_N + ori_offset_B00;
            uint offsetB32 = block_n_id * BLOCK_SIZE_N + ori_offset_B32;;
            float4 b00 = {0,0,0,0}, b16 = {0,0,0,0};
            b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
            b16 = __ldg((const float4*)(add_ptr_f(B, offsetB32)));
            // if(offsetA00>1024*1024 || offsetA16 > 1024*1024 || offsetB00>1024*1024 || offsetB32>1024*1024){
            //         printf("bx:%d by:%d tid:%d tx:%d ty:%d offsetA00:%d offsetA16:%d offsetB00:%d offsetB32:%d m_index[ty]:%d m_index[ty+16]:%d ori_offset_A00:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetA00, offsetA16, offsetB00, offsetB32, m_index[ty], m_index[ty+16], ori_offset_A00, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);

            // }

            // if(bx==0 && ty==11 && by==0){
            //     printf("CK1:!!!bx:%d by:%d tid:%d tx:%d ty:%d offsetA00:%d offsetA16:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetA00, offsetA16, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);
            //     printf("CK2:!!!bx:%d by:%d tid:%d tx:%d ty:%d offsetB00:%d offsetB32:%d b00:(%f %f %f %f) b16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetB00, offsetB32, b00.x, b00.y, b00.z, b00.w, b16.x, b16.y, b16.z, b16.w);
                
            // }

            // if(bx==0){
            //     if(a00.x+a00.y+a00.z+a00.w<3.999 || a16.x+a16.y+a16.z+a16.w<3.99){
            //         printf("CK1:!!!bx:%d by:%d tid:%d tx:%d ty:%d offsetA00:%d offsetA16:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetA00, offsetA16, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);
            //     }
            //     if(b00.x+b00.y+b00.z+b00.w<3.999 || b16.x+b16.y+b16.z+b16.w<3.99){
            //         printf("CK2:!!!bx:%d by:%d tid:%d tx:%d ty:%d offsetB00:%d offsetB32:%d b00:(%f %f %f %f) b16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetB00, offsetB32, b00.x, b00.y, b00.z, b00.w, b16.x, b16.y, b16.z, b16.w);
            //     }
            // }
            // if(bx==31 && by==0 && block_n_id==0){
            //     printf("tid:%d tx:%d ty:%d offsetB00:%d a00:(%f %f %f %f) b00:(%f %f %f %f)\n", tid, tx, ty, offsetB00, a00.x, a00.y, a00.z, a00.w, b00.x, b00.y, b00.z, b00.w);
            // }
            // __syncthreads();
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
                    for (int j = 0; j < 4; j++){
                        regC[i][j] += regA[i] * regB[j];

                        // if(bx==31 && by==0 && block_n_id==0){
                        //     printf("tid:%d tx:%d ty:%d i:%d j:%d regc[i][j]:%f regA[i]%f regB[j]:%f\n", tid, tx, ty, i, j, regC[i][j], regA[i], regB[j]);
                        // }
                    }
            }
        
            ty = ((tid & 16) >> 3) + (tid & 1);
            tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

            uint storC = ty*32*8*4 + tx*4;

            tx = tid % 16;
            ty = tid / 16;

            uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

            // C should be row major
            float * wC;

            // printf("DEBUG: bx:%d by:%d tid:%d tx:%d ty:%d m_index[ty]:%d block_n_id:%d\n", bx, by, tid, tx, ty, m_index[ty], block_n_id);
            __syncthreads();
            *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
            *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
            *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
            *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
            __syncthreads();

            float2 c2[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                c2[i] = *(float2*)&fShare[readC + i*32];

            // Tree reduce
            #pragma unroll
            for (int j = 4; j > 0; j >>= 1)
                for (int i = 0; i < j; i++)
                    c2[i] = _add(c2[i], c2[i+j]);

            // if(bx==31 && by==0 && block_n_id==0){
            //     printf("tid:%d tx:%d ty:%d  m_index[ty]:%d c2:(%f, %f) regc: %f %f %f %f \n", tid, tx, ty, m_index[ty], c2[0].x, c2[0].y, regC[0][0], regC[0][1], regC[0][2], regC[0][3]);
            // }
            // *(float2*)C = c2[0];
            // if(bx==0 && block_n_id==31){
            //     if(c2[0].x <63.9999 || c2[0].y<=63.999){
            //         printf("bx:%d by:%d block_n_id:%d tid:%d tx:%d ty:%d c2[0].x:%f c2[0].y:%f\n", bx, by, block_n_id, tid, tx, ty, c2[0].x, c2[0].y);
            //     }
            // }
            // if(bx==0&& (c2[0].x<64||c2[0].y<64)){
            //     // if(a00.x+a00.y+a00.z+a00.w<3.999 || a16.x+a16.y+a16.z+a16.w<3.99){
            //         printf("CK3: bx:%d by:%d tid:%d tx:%d ty:%d c2[0].x:%f c2[0].y:%f offsetA00:%d offsetA16:%d offsetB00:%d offsetB32:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, c2[0].x, c2[0].y , offsetA00, offsetA16, offsetB00, offsetB32, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);
            //     // }
            //     // if(b00.x+b00.y+b00.z+b00.w<3.999 || b16.x+b16.y+b16.z+b16.w<3.99){
            //         // printf("bx:%d by:%d tid:%d tx:%d ty:%d offsetB00:%d offsetB32:%d b00:(%f %f %f %f) b16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetB00, offsetB32, b00.x, b00.y, b00.z, b00.w, b16.x, b16.y, b16.z, b16.w);
            //     // }
            // }
            if(ty < index_end-index_start){
                wC = C + m_index[ty] * BLOCK_H * N + (block_n_id * BLOCK_SIZE_N + tx * 2);
                atomicAdd(wC, c2[0].x);
                atomicAdd(wC+1, c2[0].y);
            }

            
            __syncthreads();
            *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
            *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
            *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
            *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
            __syncthreads();

            for (int i = 0; i < 8; i++)
                c2[i] = *(float2*)&fShare[readC + i*32];

            // Tree reduce
            #pragma unroll
            for (int j = 4; j > 0; j >>= 1)
                for (int i = 0; i < j; i++)
                    c2[i] = _add(c2[i], c2[i+j]);
            // if(bx==0 && (c2[0].x<64||c2[0].y<64)){
            //     // if(a00.x+a00.y+a00.z+a00.w<3.999 || a16.x+a16.y+a16.z+a16.w<3.99){
            //         printf("CK3: bx:%d by:%d tid:%d tx:%d ty:%d c2[0].x:%f c2[0].y:%f offsetA00:%d offsetA16:%d offsetB00:%d offsetB32:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, c2[0].x, c2[0].y , offsetA00, offsetA16, offsetB00, offsetB32, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);
            //     // }
            //     // if(b00.x+b00.y+b00.z+b00.w<3.999 || b16.x+b16.y+b16.z+b16.w<3.99){
            //         // printf("bx:%d by:%d tid:%d tx:%d ty:%d offsetB00:%d offsetB32:%d b00:(%f %f %f %f) b16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetB00, offsetB32, b00.x, b00.y, b00.z, b00.w, b16.x, b16.y, b16.z, b16.w);
            //     // }
            // }
            if(ty+16<index_end-index_start){
                wC = C + m_index[ty+16] * BLOCK_H * N + (block_n_id * BLOCK_SIZE_N + tx * 2);
                atomicAdd(wC, c2[0].x);
                atomicAdd(wC+1, c2[0].y);
            }
            __syncthreads();
        }

    }
}

void openai_bmm_32_64_32_condense_dim_m_launch(float* A_val, int* A_row, int* A_col, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int BLOCK_H, int BLOCK_W, int SPARSE_VAL_SIZE, int batchsize)
{
    const dim3 dimBlock(256);
    assert(BLOCK_W==64);
    dim3 dimGrid(GLOBAL_M/32, GLOBAL_K/64, batchsize);
    BATCH_BLOCK_SPARSE_MATMUL_CONDENSE_DIM_M<<<dimGrid, dimBlock>>>(A_val, A_row, A_col, B, C, GLOBAL_M, GLOBAL_K, GLOBAL_N, BLOCK_H, BLOCK_W, SPARSE_VAL_SIZE);
}


at::Tensor openai_bmm_32_64_32(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor B,
    int M,
    int K,
    int N,
    int batchsize,
    int block_nnz)
{
    int sparse_val_size = block_nnz * 32 * 64;
    torch::Tensor output= torch::empty({batchsize, M, N}, B.options());
    AT_DISPATCH_FLOATING_TYPES(B.type(), "openai_bmm_forward", ([&]
                            { 
                                openai_bmm_32_64_32_launch(
                                    values.data_ptr<float>(),
                                    row_ptr.data_ptr<int>(),
                                    col_idx.data_ptr<int>(),
                                    B.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    M,
                                    K,
                                    N,
                                    sparse_val_size,
                                    batchsize
                                );
                                }));
    return output;
    
}

at::Tensor openai_bmm_32_64_32_condense(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor B,
    int M,
    int K,
    int N,
    int block_h,
    int block_w,
    int batchsize,
    int block_nnz)
{
    int sparse_val_size = block_nnz * block_h * block_w;
    torch::Tensor output= torch::empty({batchsize, M, N}, B.options());
    AT_DISPATCH_FLOATING_TYPES(B.type(), "openai_bmm_forward", ([&]
                            { 
                                openai_bmm_32_64_32_condense_launch(
                                    values.data_ptr<float>(),
                                    row_ptr.data_ptr<int>(),
                                    col_idx.data_ptr<int>(),
                                    B.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    M,
                                    K,
                                    N,
                                    block_h,
                                    block_w,
                                    sparse_val_size,
                                    batchsize
                                );
                                }));
    return output;
}

at::Tensor openai_bmm_32_64_32_condense_dim_m(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor B,
    int M,
    int K,
    int N,
    int block_h,
    int block_w,
    int batchsize,
    int block_nnz)
{
    int sparse_val_size = block_nnz * block_h * block_w;
    printf("block_nnz:%d\n", block_nnz);
    torch::Tensor output= torch::zeros({batchsize, M, N}, B.options());
    AT_DISPATCH_FLOATING_TYPES(B.type(), "openai_bmm_forward_dim_m", ([&]
                            { 
                                openai_bmm_32_64_32_condense_dim_m_launch(
                                    values.data_ptr<float>(),
                                    row_ptr.data_ptr<int>(),
                                    col_idx.data_ptr<int>(),
                                    B.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    M,
                                    K,
                                    N,
                                    block_h,
                                    block_w,
                                    sparse_val_size,
                                    batchsize
                                );
                                }));
    return output;
}
