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

__device__ void warpReduce(int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid + 8]; 
    sdata[tid] += sdata[tid + 4]; 
    sdata[tid] += sdata[tid + 2]; 
    sdata[tid] += sdata[tid + 1]; 
}
__device__ void warpReduce(half* sdata, int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid + 8]; 
    sdata[tid] += sdata[tid + 4]; 
    sdata[tid] += sdata[tid + 2]; 
    sdata[tid] += sdata[tid + 1]; 
}


__device__ void warpReduce(float* sdata, int tid) {
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



template<
    const int GLOBAL_M,
    const int GLOBAL_K,
    const int TILE_SIZE,
    const int THREAD_NUM
>
__global__ void BATCH_OUT_SPARSE_MV_NT_FP16(half * A, half * B, half * C,  int k_stride, int inter_result_stride, const int HEAD_NUM, int * pad_len, int current_seq_len)
{      
    /*
    A: Batchsize, Headnum, 1, Hidden dim
    B: Batchsize, Headnum, Tokennum, Hidden dim
    */
    // The N is contiously changing in the auto regressive inference
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    assert(M == 1); 
    assert(blockDim.x == THREAD_NUM);
    int batch_id = blockIdx.y / HEAD_NUM;
    half * ori_c =C;
    A += M * K * blockIdx.y;
    B += K * k_stride * blockIdx.y;
    C += M * inter_result_stride * blockIdx.y;
    int pads = pad_len[batch_id];
    const int THREAD_PER_ROW = K / 8;
    const int ROW_STRIDE = THREAD_NUM / THREAD_PER_ROW;
    __shared__ half reduction[THREAD_NUM];
    assert(ROW_STRIDE==TILE_SIZE);
    int tid =  threadIdx.x;
    assert(ROW_STRIDE>=1);
    
    if(blockIdx.x * ROW_STRIDE + ROW_STRIDE > pads){
        // if(threadIdx.x==0){
        //     printf("bx:%d by:%d batchid:%d pads:%d ROW_STRIDE:%d\n", blockIdx.x, blockIdx.y, batch_id, pads, ROW_STRIDE);
        // }

        // skip the computation of padding
        const int ROW_START = blockIdx.x * ROW_STRIDE + tid / THREAD_PER_ROW ;
        const int COL_START = tid % THREAD_PER_ROW * 8;
        float4 B_local = FETCH_FLOAT4(B[OFFSET(ROW_START,  COL_START, K)]);
        float4 A_local = FETCH_FLOAT4(A[COL_START]);
        // compute the reduction sum and write back to the corresponding positions
        half sum = 0;
        half* h_a = (half*) &A_local;
        half* h_b = (half*) &B_local;
        #pragma unroll
        for(int i=0;i<8;i++){
            sum += h_a[i] * h_b[i];
        }
        reduction[tid] = sum;
        __syncthreads();
        
        #pragma unroll
        for(int step=THREAD_PER_ROW/2; step>0; step>>=1){
            if(tid%THREAD_PER_ROW < step){
                reduction[tid] += reduction[tid+step];
            }
            __syncthreads();
        }
        // write back to the global memory
        // use async memory copy to optimize the performance
        if(COL_START==0 && ROW_START>pads && ROW_START<current_seq_len){
            C[ROW_START] = reduction[tid];
            // if(blockIdx.y == (HEAD_NUM+1) && COL_START==0){
            //     printf("bx:%d by:%d tid:%d batchid:%d pads:%d ROW_STRIDE:%d ROW_START:%d  COL_START:%d write_offset:%d write_value:%f inter_result_stride:%d C[ROW_START]:%f \n", blockIdx.x, blockIdx.y, tid,batch_id, pads, ROW_STRIDE, ROW_START, COL_START, &C[ROW_START]-ori_c, __half2float(reduction[tid]), inter_result_stride, __half2float(C[ROW_START]));
            // }
        }
    }

}


template<const int LINE_SIZE>
__global__ void BATCH_SOFTMAX_FP16(half * A, int inter_result_stride, const int HEAD_NUM, int * pad_len, int max_seq_len)
{
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int batch_id = blockIdx.x / HEAD_NUM;
    A += inter_result_stride * blockIdx.x;
    int pads = pad_len[batch_id];
    __shared__ half line[LINE_SIZE];
    __shared__ float reduce[256];
    half* hreduce = (half*) reduce;
    const int line_offset = tid * 8;
    const int global_offset_start = pads / 8 * 8;
    const int global_offset_end = (max_seq_len +7) / 8 * 8;
    assert(LINE_SIZE > global_offset_end-global_offset_start);

    int stride = blockDim.x * 8;
    // load the values to the shared memory
    for(int pos=line_offset; pos + global_offset_start<global_offset_end; pos+=stride){
        FETCH_FLOAT4(line[pos]) = FETCH_FLOAT4(A[pos+global_offset_start]);
    }
    __syncthreads();
    half regMax = 0.0;
    float regSum = 0.0;
    const int line_offset_start = pads % 8;
    const int line_offset_end = max_seq_len - global_offset_start;

    for(int i=line_offset_start + tid; i<line_offset_end; i += blockDim.x){

        regMax = max(regMax, line[i]);
    }
    hreduce[tid] = regMax;
    __syncthreads();
    // synchronze accross the thread block
    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid<s)
            hreduce[tid] = max(hreduce[tid+s], hreduce[tid]);
        __syncthreads();
    }
    if(tid<32){
        hreduce[tid] = max(hreduce[tid + 32], hreduce[tid]); 
        hreduce[tid] = max(hreduce[tid + 16], hreduce[tid]); 
        hreduce[tid] = max(hreduce[tid + 8], hreduce[tid]); 
        hreduce[tid] = max(hreduce[tid + 4], hreduce[tid]); 
        hreduce[tid] = max(hreduce[tid + 2], hreduce[tid]); 
        hreduce[tid] = max(hreduce[tid + 1], hreduce[tid]); 
    }
        // warpReduceMax(hreduce, tid);
    __syncthreads();
    regMax = hreduce[0];
    float fregMax = __half2float(regMax);

    // compuate the regSum
    for(int i=line_offset_start + tid; i<line_offset_end; i += blockDim.x){
        regSum += expf(__half2float(line[i])-fregMax);
    }

    reduce[tid] = regSum;
    __syncthreads();
    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid<s)
            reduce[tid] += reduce[tid+s];
        __syncthreads();
    }
    if(tid<32)
        warpReduce(reduce, tid);
    __syncthreads();
    regSum = reduce[0];

    // write the results back to the global memory
    for(int i=line_offset_start+tid; i<line_offset_end; i+=blockDim.x){
        // if(tid==10){
        //     // printf("tid:%d bx:%d line_start:%d line_end:%d RegMax:%f RegSum:%f \n", tid, bx, line_offset_start, line_offset_end, fregMax, regSum);
        //     printf("tid:%d bx:%d line_start:%d line_end:%d line[i]:%f line[i]-max:%f RegMax:%f RegSum:%f \n", tid, bx, line_offset_start, line_offset_end, __half2float(line[i]), __half2float(line[i])-fregMax, fregMax, regSum);
        //     // printf("tid:%d bx:%d line_start:%d line_end:%d line[i]:%f RegMax:%f RegSum:%f \n", tid, bx, line_offset_start, line_offset_end, __half2float(line[i]), __half2float(regMax), regSum);
        // }
        A[i+global_offset_start] = __float2half(expf(__half2float(line[i])-fregMax)/regSum);
    }   
}


void forward_function(
    float * Q,
    float * K,
    float * V,
    float * inter_result,
    int * pad_len,
    int max_token_len,
    int batch_size,
    int head_num,
    int q_seq_length,
    int hidden_dim,
    int k_stride,
    int inter_result_stride,
    float * output
)
{

}

void forward_function(
    double * Q,
    double * K,
    double * V,
    double * inter_result,
    int * pad_len,
    int max_token_len,
    int batch_size,
    int head_num,
    int q_seq_length,
    int hidden_dim,
    int k_stride,
    int inter_result_stride,
    double * output
)
{

}

void forward_function(
    c10::Half * Q,
    c10::Half * K,
    c10::Half * V,
    c10::Half * inter_result,
    int * pad_len,
    int max_token_len,
    int batch_size,
    int head_num,
    int q_seq_length,
    int hidden_dim,
    int k_stride,
    int inter_result_stride,
    c10::Half * output
)
{
    cudaMemset(inter_result, 0, sizeof(half) * batch_size * head_num * inter_result_stride);
    // newQ x K (cachedK:newK)
    if(hidden_dim==64){
        const int TOTAL_THREAD_NUM = 256;
        const int THREAD_PER_ROW = 64 / 8; // float4 = 8 halfs
        const int TILE_SIZE = TOTAL_THREAD_NUM / THREAD_PER_ROW;
        dim3 qk_gridDim((max_token_len + TILE_SIZE -1) / TILE_SIZE, head_num * batch_size);
        dim3 qk_blockDim(TOTAL_THREAD_NUM);
        BATCH_OUT_SPARSE_MV_NT_FP16<1, 64, TILE_SIZE, TOTAL_THREAD_NUM><<<qk_gridDim, qk_blockDim>>>((half *)Q, (half *)K, (half *)inter_result, k_stride, inter_result_stride, head_num, pad_len, max_token_len);
        dim3 soft_gridDim(head_num*batch_size);
        dim3 soft_blockDim(TOTAL_THREAD_NUM);
        BATCH_SOFTMAX_FP16<4096><<<soft_gridDim, soft_blockDim>>>((half*)inter_result, inter_result_stride, head_num, pad_len, max_token_len);
    }else{
        throw std::invalid_argument("Not Implemented\n");
    }
    // Sofmax

    //
}

at::Tensor dynamic_sparse_cache_attention_forward(
    torch::Tensor Q, // new Q 
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor inter_result,
    torch::Tensor pad_len,
    int max_token_len
)
{   
    // B: Batch size
    // H: the number of heads
    // T: the number of tokens, here T = padding length + prompt length + the number of iterations
    // h: the hidden dim
    // Q: the projection of current token(newQ), with shape of [B H 1 h]
    // K: the tensor of cachedQ and newQ, with shape of [B H T h]
    // V: the tensor of cachedV and newV, with shape of [B H T h]

    cudaSetDevice(Q.get_device());
    int batch_size = Q.size(0);
    int head_num = Q.size(1);
    int q_seq_length = Q.size(2);
    int k_stride = K.size(2); // B H k_stride, hidden_dim
    int inter_result_stride = inter_result.size(3); // B H 1 inter_result_stride
    int hidden_dim = Q.size(3);
    torch:: Tensor output=torch::empty({batch_size, head_num, q_seq_length, hidden_dim}, Q.options()); 
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.type(), "dynamic_sparse_cache_attention", ([&]
                                { forward_function(
                                        Q.data_ptr<scalar_t>(),
                                        K.data_ptr<scalar_t>(),
                                        V.data_ptr<scalar_t>(),
                                        inter_result.data_ptr<scalar_t>(),
                                        pad_len.data_ptr<int>(),
                                        max_token_len,
                                        batch_size,
                                        head_num,
                                        q_seq_length,
                                        hidden_dim,
                                        k_stride,
                                        inter_result_stride,
                                        output.data_ptr<scalar_t>()
                                    ); }));
    return output;
}
