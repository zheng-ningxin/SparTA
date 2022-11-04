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




__global__ void BLOCK_SPARSE_MATMUL_TN_CONDENSE_OPENAI(float* A, float* B, float*output, int * row_ptr, int * col_idx, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int BLOCK_H, int BLOCK_W)
{

    /*
    A : dense matrix with the shape of KxM
    B : dense matrix with the shape of KxN
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

    __syncthreads();

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offset_B00 = bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offset_A00 = by * BLOCK_SIZE_M + (tid % (BLOCK_SIZE_M/4)) * 4;
    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4
    uint storA = (tid * 4 + tid / (BLOCK_SIZE_M/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4
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
    int index_start = row_ptr[bx], index_end = row_ptr[bx+1];
    // float4 const0 = {0};
    int round = (index_end - index_start - 1 + BLOCK_SIZE_K/BLOCK_H) / (BLOCK_SIZE_K/BLOCK_H);
    for (int rid = 0; rid<round; rid++)
    {
        uint k_offset = tid / (BLOCK_SIZE_N/4) + rid * BLOCK_SIZE_K;
        uint k_offset32 = k_offset + 32;
        // offsetA00 = offsetA00 + BLOCK_SIZE_K * BLOCK_SIZE_M; 
        // offsetA32 = offsetA32 + BLOCK_SIZE_K * BLOCK_SIZE_M;
        uint _pos = (k_offset / BLOCK_H);
        uint _pos32 = (k_offset32/BLOCK_H);
        uint offsetB00 = (col_idx[index_start+_pos]+k_offset%BLOCK_H) * N + ori_offset_B00;
        uint offsetB32 = (col_idx[index_start+_pos32]+k_offset32%BLOCK_H) * N + ori_offset_B00;
        uint offsetA00 = (col_idx[index_start+_pos]+k_offset%BLOCK_H) * M + ori_offset_A00;;
        uint offsetA32 = (col_idx[index_start+_pos32]+k_offset32%BLOCK_H) * M + ori_offset_A00;
        // uint offsetB00 = ori_offsetB00 + 64 * A_col[bcsr_col_idx] * N;
        // uint offsetB16 = ori_offsetB16 + 64 * A_col[bcsr_col_idx] * N;
        // if(bx==0 && by == 0 &&  threadIdx.x==0){
        //     printf("_pos:%d _pos32:%d index_end-index_start:%d\n", _pos, _pos32, index_end-index_start);
        // }
        float4 a00 = {0,0,0,0}, a16 = {0,0,0,0};
        float4 b00 = {0,0,0,0}, b16 = {0,0,0,0};
        if(_pos<index_end-index_start){
            a00 = __ldg((const float4*)(add_ptr_f(A, offsetA00)));
            b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
        }
        if(_pos32<index_end-index_start){
            a16 = __ldg((const float4*)(add_ptr_f(A, offsetA32)));
            b16 = __ldg((const float4*)(add_ptr_f(B, offsetB32)));
        }

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
    *(float2*)output = c2[0];
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
    *(float2*)output = c2[0];

}

void condense_dynamic_forward_function(float* activation, float* weight, int* row_ptr, int* col_idx,
                    float* bias, int M, int K, int N, int block_h, int block_w, float* output)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(256);
    BLOCK_SPARSE_MATMUL_TN_CONDENSE_OPENAI<<<gridDim, blockDim>>>(activation, weight, output, row_ptr, col_idx, M, K, N, block_h, block_w);
}

void dynamic_backward_function(float* grad_in, int * row_ptr, int *col_idx, float* val, int M, int K, int N, int block_h, int block_w, float* grad_out)
{

}

at::Tensor dynamic_sparse_linear_condense_forward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor row_ptr,
    torch::Tensor col_index,
    torch::Tensor bias,
    int M, int K, int N, int block_h, int block_w
)
{
    // The weight tensor here should be transposed and in the shape of K x N
    cudaSetDevice(activation.get_device());
    int batch_size = activation.size(0);
    int seq_len = activation.size(1);
    int in_hidden = activation.size(2);
    assert(in_hidden == weight.size(0));
    assert(M == batch_size* seq_len);
    int out_hidden = weight.size(1);
    torch::Tensor output = torch::empty({batch_size, seq_len, out_hidden}, activation.options());
    // Q, K, V should have the same shape which is {batchsize, seq_length, hidden_dim}

    
    AT_DISPATCH_FLOATING_TYPES(activation.type(), "dynamic_sparse_linear", ([&]
                            { condense_dynamic_forward_function(
                                    activation.data_ptr<float>(),
                                    weight.data_ptr<float>(),
                                    row_ptr.data_ptr<int>(),
                                    col_index.data_ptr<int>(),
                                    bias.data_ptr<float>(),
                                    M, K, N, block_h, block_w,
                                    output.data_ptr<float>()
                                ); }));
    return output;
}

vector<at::Tensor> dynamic_sparse_linear_condense_backward(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor grad_a_row_ptr,
    torch::Tensor grad_a_col_index,
    torch::Tensor grad_c,
    int M, int K, int N, int block_h, int block_w
)
{
    vector<at::Tensor> grads;
    return grads;
}