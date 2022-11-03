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






void condense_dynamic_forward_function(float* activation, float* weight, int* row_ptr, int* col_idx,
                    float* bias, int M, int K, int N, int block_h, int block_w, float* output)
{

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