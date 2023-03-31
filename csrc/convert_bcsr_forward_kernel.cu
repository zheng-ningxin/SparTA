#include "common.h"
#include "cusparse.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

#include "common.h"

using namespace std;

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

__global__ void convert_bcsr_kernel_1(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
                                int block_h, int block_w, int * row, int *col, int * row_pos,
                                float * values, int * extra_buffer)
{

    __shared__ int reduce[MAX_BLOCK_THREAD_COUNT];
    assert(blockDim.x<=MAX_BLOCK_THREAD_COUNT);
    // initial the shared flag
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tid = threadIdx.x;
    int global_offset =  (by * block_h) * w + bx * block_w;
    int block_size =  block_h * block_w;
    assert(block_w % 4 == 0);
    // cannot handle the misalignment for now
    assert((block_size / 4) % blockDim.x==0);
    int flag = 0;
    for(int _pos = tid; _pos< block_size / 4; _pos+=blockDim.x){
        uint block_offset = _pos / (block_w / 4) * w + _pos % (block_w / 4) * 4;        
        int4 data = __ldg((const int4*)(add_ptr_u(mask, global_offset+block_offset)));
        flag += data.x + data.y + data.z + data.w;
    }
    reduce[tid] = flag;
    __syncthreads();
    // fast tree reduce accross the block
    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid<s)
            reduce[tid] += reduce[tid+s];
        __syncthreads();
    }
    if(tid<32)
        warpReduce(reduce, tid);
    __syncthreads();
    int pos_id;
    if(tid==0 && reduce[0]>0){
        pos_id= atomicAdd(&extra_buffer[by], 1);
        atomicAdd(&extra_buffer[by+h], 1);
        atomicAdd(&row[h/block_h], 1);
        extra_buffer[2*h + gridDim.x * by + pos_id] = bx;
    }

}
__global__ void convert_bcsr_kernel_2(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, int * row_pos,
    float * values, int * extra_buffer, int * block_index)
{
    __shared__ int pos_id, prefix_count, ori_bx, ori_by;
    __shared__ int prefix_sum[MAX_BLOCK_THREAD_COUNT];
    uint by = blockIdx.y;
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;

    if (tid==0){
        pos_id = -1;
        prefix_count = 0;
        // contend for the block

        pos_id = atomicSub(&extra_buffer[by], 1);
        pos_id-=1;
        if (pos_id>=0){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }
            ori_by = by;
            ori_bx = extra_buffer[ 2*h + by * gridDim.x + pos_id];       
            
            row[by] = prefix_count;
            col[prefix_count+pos_id] = ori_bx;
            row_pos[prefix_count+pos_id] = by;
            block_index[by*gridDim.x+ori_bx] = prefix_count+pos_id;
        }
        else if(pos_id==-1){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }            
            row[by] = prefix_count;
        }
    }
    __syncthreads();
    if(pos_id>=0){
        int global_offset =  (ori_by * block_h) * w + ori_bx * block_w;
        int block_size = block_h * block_w;
        int write_global_offset = (prefix_count + pos_id) * block_size;

        for(int _pos=tid; _pos<block_size/4; _pos+=blockDim.x){
            uint block_offset = _pos / (block_w / 4) * w + _pos % (block_w / 4) * 4;
            float4 data = __ldg((const float4*)(add_ptr_f(dense, global_offset + block_offset)));
            *(float4*)&values[write_global_offset+_pos*4] = data;
        }
        
    }

}


__global__ void convert_bcsr_kernel_fine_1(int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, float * values, int * extra_buffer)
{
    __shared__ int reduce[MAX_BLOCK_THREAD_COUNT];
    uint by = blockIdx.x; // row
    uint tid = threadIdx.x;
    int sum =0;
    int4 flag;
    for(int _pos = tid; _pos<w/4; _pos +=blockDim.x){
        flag = FETCH_INT4(mask[by * w + _pos*4]);
        sum += flag.x + flag.y + flag.z + flag.w;
    }
    reduce[tid] = sum;
    __syncthreads();
    // fast tree reduce accross the block
    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid<s)
            reduce[tid] += reduce[tid+s];
        __syncthreads();
    }
    if(tid<32)
        warpReduce(reduce, tid);
    __syncthreads();
    if(tid==0){
        extra_buffer[by] = reduce[0];
        extra_buffer[by+h] = reduce[0];
        atomicAdd(&row[h], reduce[0]);
    }

}
__global__ void convert_bcsr_kernel_fine_2(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, float * values, int * extra_buffer)
{
    uint tid = threadIdx.x;
    uint by = blockIdx.x;
    __shared__ int prefix_sum;
    if (tid==0){
        prefix_sum = 0;
        for(int i=0; i<by; i++)
            prefix_sum += extra_buffer[i];
        row[by] = prefix_sum;
    }
    __syncthreads();
    for(int _pos=tid; _pos<w; _pos+=blockDim.x){
        if(mask[by*w+_pos]>0){
            int tmp = atomicSub(&extra_buffer[by+h], 1);
            tmp-=1;
            col[prefix_sum+tmp] = _pos;
            values[prefix_sum+tmp] = dense[by*w+_pos];
        }
    }
}
void convert_bcsr_fine(int * mask, float* dense, int h, int w, int block_h,
                        int block_w, int * row, int * col,
                        float * values, int * extra_buffer)
{
    assert(block_w==1);
    assert(block_h==1);
    CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*(2*h+(h/block_h)*(w/block_w))) );
    CUDA_SAFE_CALL(cudaMemset((void*)row, 0, sizeof(int)*(1+(h/block_h))) );
    dim3 block_dim(256);
    dim3 grid_dim(h/block_h);
    convert_bcsr_kernel_fine_1<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, values, extra_buffer);
    convert_bcsr_kernel_fine_2<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, values, extra_buffer);
}

void convert_bcsr(int * mask, float * dense, int h, int w,
    int block_h, int block_w, int * row, int *col, int * row_pos,
    float*values, int * extra_buffer, int *block_index)
{
    if(block_h==1&&block_w==1){
        convert_bcsr_fine(mask, dense, h, w, 1, 1, row, col, values, extra_buffer);
    }else{
    // need reset the extra buffer here
        assert(block_w % 4 == 0);
        CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*(2*h+(h/block_h)*(w/block_w))) );
        CUDA_SAFE_CALL(cudaMemset((void*)row, 0, sizeof(int)*(1+(h/block_h))) );
        dim3 block_dim(block_h*block_w/4);
        dim3 grid_dim(w/block_w, h/block_h);
        // std::cout<<"grid_dim "<< w/block_w << ", " <<h/block_h << std::endl;
        convert_bcsr_kernel_1<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer);
        convert_bcsr_kernel_2<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer, block_index);
    }
}

std::vector<at::Tensor> convert_bcsr_forward(
    torch::Tensor sparse_pattern,
    torch::Tensor dense_values,
    int block_h, 
    int block_w)
{
    int h = sparse_pattern.size(0);
    int w = sparse_pattern.size(1);
    assert(h % block_h==0);
    assert(w % block_w==0);
    // allocate enough memory for the sparse values
    torch::Tensor csr_values = torch::empty_like(dense_values);
    torch::Tensor csr_row = torch::zeros({h/block_h+1}, sparse_pattern.options());
    int n_total_block = h * w / block_h / block_w;
    torch::Tensor csr_row_pos = torch::zeros({n_total_block}, sparse_pattern.options());
    torch::Tensor csr_col = torch::zeros({n_total_block}, sparse_pattern.options());
    torch::Tensor ext_buffer = torch::zeros({2*h+n_total_block}, sparse_pattern.options());
    torch::Tensor block_index = torch::empty({h * w / block_h / block_w}, sparse_pattern.options());
    AT_DISPATCH_FLOATING_TYPES(dense_values.type(), "convert_bcsr", ([&]
        { convert_bcsr(
                sparse_pattern.data_ptr<int>(),
                dense_values.data_ptr<float>(),
                h, w, block_h, block_w,
                csr_row.data_ptr<int>(),
                csr_col.data_ptr<int>(),
                csr_row_pos.data_ptr<int>(),
                csr_values.data_ptr<float>(),
                ext_buffer.data_ptr<int>(),
                block_index.data_ptr<int>()
            ); }));
    std::vector<torch::Tensor> bcsr({csr_row, csr_col, csr_row_pos, csr_values, block_index});
    return bcsr;
}

template<
    const int BLOCK_SIZE_H,
    const int BLOCK_SIZE_W
>
__global__ void convert_bcsr_v2_kernel_1(float* activation, int h, int w, int * row, int * col, int * extra_buffer)
{
    const int N_WARP = BLOCK_SIZE_H;
    const int BLOCK_W = BLOCK_SIZE_W;
    assert(N_WARP*32==blockDim.x);
    assert(BLOCK_W == BLOCK_SIZE_W); // currently only support the same
    __shared__ float s_data[BLOCK_SIZE_H][BLOCK_SIZE_W];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    const int TOTAL_THREADS = N_WARP * 32;
    const int THREAD_PER_ROW = BLOCK_SIZE_W / 4; // float4 load
    int ROW_START = tid / THREAD_PER_ROW;
    int ROW_STRIDE = TOTAL_THREADS / THREAD_PER_ROW;
    int COL_START = tid % THREAD_PER_ROW;
    // Load to shared memory
    for(int rid=ROW_START; rid<BLOCK_SIZE_H; rid+=ROW_STRIDE){
        FETCH_FLOAT4(s_data[rid][COL_START*4]) = FETCH_FLOAT4(activation[OFFSET(by * BLOCK_SIZE_H + rid, bx * BLOCK_SIZE_W + COL_START * 4, w)]);
    }
    __syncthreads();
    // go through the values
    const int wid =  tid / 32;
    const int tid_ = tid % 32; // tid within the warp
    // each warp is responsible for a row whose size is BLOCK_W
    int have_val = 0;
    float reg;
    
    #pragma unroll
    for(int step=0; step<BLOCK_SIZE_W/32; step++){
        reg = s_data[wid][step*32+tid_];
        have_val += (reg!=0); // < threshold
    
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        have_val += __shfl_down_sync(FULL_MASK, have_val, offset);
    }
    have_val = __shfl_sync(FULL_MASK, have_val, 0);
    int pos_id;
    // if(tid_==0){
    //     printf("bx:%d by%d wid:%d have_val:%d w/block_w:%d row[w/BLOCK_W]:%d\n", bx, by, wid, have_val, w/BLOCK_W, row[w/BLOCK_W]);
    // }
    if(tid_==0 && have_val>0){
        pos_id= atomicAdd(&extra_buffer[bx], 1);
        atomicAdd(&extra_buffer[bx+w], 1);
        atomicAdd(&row[w/BLOCK_W], 1);
        extra_buffer[2*w + h * bx + pos_id] = wid + by * BLOCK_SIZE_H;
    }    

} 

template<
    const int BLOCK_SIZE_H,
    const int BLOCK_SIZE_W
>
__global__ void convert_bcsr_v2_kernel_2(float* activation, int h, int w, int * row, int * col, int * extra_buffer)
{
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;
    int pos_id, ori_bx, ori_by;
    __shared__ int prefix_count;
    __shared__ int remain_count;
    if(tid==0){
        prefix_count = 0;
        for(int i=0; i<bx;i++){
            prefix_count +=  extra_buffer[w+i];
        }
        remain_count = extra_buffer[w+bx];
        row[bx] = prefix_count;
    }
    __syncthreads();
    for(int tmp=tid; tmp<remain_count;tmp+=blockDim.x){
        pos_id = atomicSub(&extra_buffer[bx], 1);
        pos_id-=1;
        if(pos_id>=0){
            ori_bx = bx;
            ori_by = extra_buffer[ 2 * w + bx * h + pos_id];       
            row[bx] = prefix_count;
            col[prefix_count + pos_id] = ori_by;
        }
    }

} 

void convert_bcsr_v2(float * dense, int h, int w,
    int block_h, int block_w, int * row, int *col,
    int * extra_buffer, int ext_buffer_size)
{
    // printf("wtf??\n");
    CUDA_SAFE_CALL(cudaMemset(extra_buffer, 0, sizeof(float)*ext_buffer_size));
    CUDA_SAFE_CALL(cudaMemset(row+w/block_w, 0, sizeof(int)));
    if(block_w==32){
        const int BLOCK_SIZE_H = 16;
        const int BLOCK_SIZE_W = 32;
        dim3 blockDim1(BLOCK_SIZE_H * 32);
        dim3 gridDim1(w/BLOCK_SIZE_W, h/BLOCK_SIZE_H);
        convert_bcsr_v2_kernel_1<BLOCK_SIZE_H, BLOCK_SIZE_W><<<gridDim1, blockDim1>>>(dense, h, w, row, col, extra_buffer);
        dim3 gridDim2(w/BLOCK_SIZE_W);
        dim3 blockDim2(256);        
        convert_bcsr_v2_kernel_2<BLOCK_SIZE_H, BLOCK_SIZE_W><<<gridDim2, blockDim2>>>(dense, h, w, row, col, extra_buffer);
    }
    else{
        assert(false);
    }

}

void convert_bcsr_forward_v2(
    torch::Tensor sparse_act,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor ext_buffer,
    int h,
    int w,
    int block_h, 
    int block_w)
{
    cudaSetDevice(sparse_act.get_device());
    assert(h % block_h==0);
    assert(w % block_w==0);
    AT_DISPATCH_FLOATING_TYPES(sparse_act.type(), "convert_bcsr", ([&]
    { convert_bcsr_v2(
                sparse_act.data_ptr<float>(),
                h, w, block_h, block_w,
                row.data_ptr<int>(),
                col.data_ptr<int>(),
                ext_buffer.data_ptr<int>(),
                ext_buffer.numel()
            ); }));
}



template<
    const int BLOCK_SIZE_H,
    const int BLOCK_SIZE_W
>
__global__ void convert_bcsr_v3_kernel_1(float* activation, int h, int w, int * row, int * col, int * extra_buffer, const int max_seq_len, int * seqlens)
{
    const int N_WARP = BLOCK_SIZE_H;
    const int BLOCK_W = BLOCK_SIZE_W;
    assert(N_WARP*32==blockDim.x);
    assert(BLOCK_W == BLOCK_SIZE_W); // currently only support the same
    __shared__ float s_data[BLOCK_SIZE_H][BLOCK_SIZE_W];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int batch_id = bx / (max_seq_len/BLOCK_SIZE_W);
    int cur_seq_len =  seqlens[batch_id];
    int bx_within_batch = bx % (max_seq_len/BLOCK_SIZE_W);
    if(bx_within_batch * BLOCK_SIZE_W >= cur_seq_len)
        return;
    int REMAINED_LEN = min(BLOCK_SIZE_W, cur_seq_len - bx_within_batch * BLOCK_SIZE_W);
    int tid = threadIdx.x;
    const int TOTAL_THREADS = N_WARP * 32;
    const int THREAD_PER_ROW = BLOCK_SIZE_W / 4; // float4 load
    int ROW_START = tid / THREAD_PER_ROW;
    int ROW_STRIDE = TOTAL_THREADS / THREAD_PER_ROW;
    int COL_START = tid % THREAD_PER_ROW;
    // Load to shared memory
    for(int rid=ROW_START; rid<BLOCK_SIZE_H; rid+=ROW_STRIDE){
        FETCH_FLOAT4(s_data[rid][COL_START*4]) = FETCH_FLOAT4(activation[OFFSET(by * BLOCK_SIZE_H + rid, bx * BLOCK_SIZE_W + COL_START * 4, w)]);
    }
    __syncthreads();
    // go through the values
    const int wid =  tid / 32;
    const int tid_ = tid % 32; // tid within the warp
    // each warp is responsible for a row whose size is BLOCK_W
    int have_val = 0;
    float reg;
    
    #pragma unroll
    for(int step=0; step<BLOCK_SIZE_W/32; step++){
        if(step*32+tid_<REMAINED_LEN){
            reg = s_data[wid][step*32+tid_];
            have_val += (reg!=0); // < threshold
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        have_val += __shfl_down_sync(FULL_MASK, have_val, offset);
    }
    have_val = __shfl_sync(FULL_MASK, have_val, 0);
    int pos_id;
    // if(tid_==0){
    //     printf("bx:%d by%d wid:%d have_val:%d w/block_w:%d row[w/BLOCK_W]:%d\n", bx, by, wid, have_val, w/BLOCK_W, row[w/BLOCK_W]);
    // }
    if(tid_==0 && have_val>0){
        pos_id= atomicAdd(&extra_buffer[bx], 1);
        atomicAdd(&extra_buffer[bx+w], 1);
        atomicAdd(&row[w/BLOCK_W], 1);
        extra_buffer[2*w + h * bx + pos_id] = wid + by * BLOCK_SIZE_H;
    }    

} 

template<
    const int BLOCK_SIZE_H,
    const int BLOCK_SIZE_W
>
__global__ void convert_bcsr_v3_kernel_2(float* activation, int h, int w, int * row, int * col, int * extra_buffer)
{
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;
    int pos_id, ori_bx, ori_by;
    __shared__ int prefix_count;
    __shared__ int remain_count;
    if(tid==0){
        prefix_count = 0;
        for(int i=0; i<bx;i++){
            prefix_count +=  extra_buffer[w+i];
        }
        remain_count = extra_buffer[w+bx];
        row[bx] = prefix_count;
    }
    __syncthreads();
    for(int tmp=tid; tmp<remain_count;tmp+=blockDim.x){
        pos_id = atomicSub(&extra_buffer[bx], 1);
        pos_id-=1;
        if(pos_id>=0){
            ori_bx = bx;
            ori_by = extra_buffer[ 2 * w + bx * h + pos_id];       
            row[bx] = prefix_count;
            col[prefix_count + pos_id] = ori_by;
        }
    }

} 

void convert_bcsr_v3(float * dense, int h, int w,
    int block_h, int block_w, int * row, int *col,
    int * extra_buffer, int ext_buffer_size, int batch_size, int max_seq_len, int * seqlens)
{
    // printf("wtf??\n");
    CUDA_SAFE_CALL(cudaMemset(extra_buffer, 0, sizeof(float)*ext_buffer_size));
    CUDA_SAFE_CALL(cudaMemset(row+w/block_w, 0, sizeof(int)));
    if(block_w==32){
        const int BLOCK_SIZE_H = 16;
        const int BLOCK_SIZE_W = 32;
        dim3 blockDim1(BLOCK_SIZE_H * 32);
        dim3 gridDim1(w/BLOCK_SIZE_W, h/BLOCK_SIZE_H/batch_size, batch_size);
        convert_bcsr_v3_kernel_1<BLOCK_SIZE_H, BLOCK_SIZE_W><<<gridDim1, blockDim1>>>(dense, h, w, row, col, extra_buffer, max_seq_len, seqlens);
        dim3 gridDim2(w/BLOCK_SIZE_W);
        dim3 blockDim2(256);        
        convert_bcsr_v3_kernel_2<BLOCK_SIZE_H, BLOCK_SIZE_W><<<gridDim2, blockDim2>>>(dense, h, w, row, col, extra_buffer);
    }
    else{
        assert(false);
    }

}


void convert_bcsr_forward_v3(
    torch::Tensor sparse_act,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor ext_buffer,
    torch::Tensor seq_lens,
    int h, int w,
    int block_h, 
    int block_w,
    int batch_size)

{
    cudaSetDevice(sparse_act.get_device());
    assert(h%batch_size==0);
    assert(h % block_h==0);
    assert(w % block_w==0);
    AT_DISPATCH_FLOATING_TYPES(sparse_act.type(), "convert_bcsr", ([&]
    { convert_bcsr_v3(
                sparse_act.data_ptr<float>(),
                h, w, block_h, block_w,
                row.data_ptr<int>(),
                col.data_ptr<int>(),
                ext_buffer.data_ptr<int>(),
                ext_buffer.numel(),
                batch_size, h/batch_size, seq_lens.data_ptr<int>()
            ); }));
}
