#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
// CUDA runtime
#include <cuda.h>

#include "bcsr.hpp"
#include "utils.hpp"

#define OFFSET(row, col, ld) ((row) * ld + col)

#define CPU_DEBUG 1

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT32x4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

size_t load_from_file(char* ptr, size_t buff_size, string filepath){
    std::ifstream fin(filepath, ios::in | ios::binary);
    size_t loaded_size = fin.read(ptr, buff_size).gcount();
    return loaded_size;
}

template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_MATMUL(float* A, float* W_val, int* W_row, int* W_col, float* C, float *bias, int M, int K, int N){
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
    int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

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

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
            FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = 
                FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]);
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


void HostComputation(float* A, float* W, float* D, float* bias, int M, int K, int N){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            float cSub = 0;
            for(int k = 0; k < K; k++){
                cSub += A[i * K + k] * W[k * N + j];
            }
            D[i * N + j] = cSub + bias[j];
        }
    }
}

void HostComputation_sparse(float* A, int* row, int* col, float* val, float* D, float* bias, int M, int K, int N, int BLOCK_SIZE_K, int BLOCK_SIZE_N){
    size_t mem_size_B = sizeof(float) * K * N;
    float* B = (float*)malloc(mem_size_B);
    std::memset(B, 0, sizeof(B));
    int ROW_BLOCK_NUM = K / BLOCK_SIZE_K;
    for(int i = 0; i < ROW_BLOCK_NUM; i++){
        int index_start = row[i], index_end = row[i+1];
        for(int index = index_start; index < index_end; index += 1){
            int col_index = col[index] * BLOCK_SIZE_N;
            int row_index = i * BLOCK_SIZE_K;
            float* val_ptr = val + index * BLOCK_SIZE_K * BLOCK_SIZE_N;
            for(int k = row_index; k < (i+1) * BLOCK_SIZE_K; k += 1){
                for(int n = col_index; n < col_index+BLOCK_SIZE_N; n += 1){
                    B[OFFSET(k,n,N)] = *(val_ptr + k * BLOCK_SIZE_N + n);
                }
            }
        }
    }
    for(int i = 0; i < M; i += 1){
        for(int j = 0; j < N; j += 1){
            float cSub = 0;
            for(int k = 0; k < K; k += 1){
                cSub += A[i * K + k] * B[k * N + j];
            }
            D[i * N + j] = cSub + bias[j];
        }
    }
}

int matrixMultiply(int M, int N, int K){
    int size_A = M * K;
    int size_C = M * N;

    /*
    const int BLOCK_SIZE_M = 32; // 64
    const int BLOCK_SIZE_K = 32;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_M = 8;  //8
    const int THREAD_SIZE_K = 4;  //4
    const int THREAD_SIZE_N = 8;  //8
    */

    const int BLOCK_SIZE_M = BLOCK_SIZE_M_VALUE; // 64
    const int BLOCK_SIZE_K = BLOCK_SIZE_K_VALUE;  //8
    const int BLOCK_SIZE_N = BLOCK_SIZE_N_VALUE;  //128
    const int THREAD_SIZE_M = THREAD_SIZE_M_VALUE;  //8
    const int THREAD_SIZE_K = THREAD_SIZE_K_VALUE;  //4
    const int THREAD_SIZE_N = THREAD_SIZE_N_VALUE;  //8

    int mem_size_A = sizeof(float) * size_A;
    int mem_size_C = sizeof(float) * size_C;
    int mem_size_bias = sizeof(float) * N;

    // memory size of row, col, val
    int mem_size_row = sizeof(int) * M;
    int mem_size_col = sizeof(int) * M * N;
    int mem_size_val = sizeof(float) * M * N;

    float* h_A = (float*)malloc(mem_size_A);
    float* h_C = (float*)malloc(mem_size_C);
    float* h_bias = (float*)malloc(mem_size_bias);
    float* h_result = (float*)malloc(mem_size_C);

    // memory allocation of row, col, val
    int* h_row = (int*)malloc(mem_size_row);
    int* h_col = (int*)malloc(mem_size_col);
    float* h_val = (float*)malloc(mem_size_val);

    // load data
    std::string row_path = ROW_PATH_SUB;
    std::string col_path = COL_PATH_SUB;
    std::string val_path = VAL_PATH_SUB;

    load_from_file((char*)h_row, mem_size_row, row_path);
    load_from_file((char*)h_col, mem_size_col, col_path);
    load_from_file((char*)h_val, mem_size_val, val_path);

    float* d_A;
    float* d_C;
    float* d_bias;

    // device memory allocation
    int* d_row;
    int* d_col;
    float* d_val;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 10;

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            h_A[i * K + j] = rand()%5;
        }
    }

    for(int i = 0; i < N; i++){
        h_bias[i] = rand()%5;
    }

    printf("host init successfully!\n");
    printf("number of iteration: %d\n", nIter);

    checkCudaErrors(cudaMalloc(&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc(&d_C, mem_size_C));
    checkCudaErrors(cudaMalloc(&d_bias, mem_size_bias));

    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bias, h_bias, mem_size_bias, cudaMemcpyHostToDevice));

    // device csr memory copy
    checkCudaErrors(cudaMalloc(&d_row, mem_size_row));
    checkCudaErrors(cudaMalloc(&d_col, mem_size_col));
    checkCudaErrors(cudaMalloc(&d_val, mem_size_val));

    checkCudaErrors(cudaMemcpy(d_row, h_row, mem_size_row, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_col, h_col, mem_size_col, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, h_val, mem_size_val, cudaMemcpyHostToDevice));

    printf("Device init successfully!\n");

    printf("Begin to run MatrixMulCUDA_8bit() function....\n");
    dim3 dimBlock(float(BLOCK_SIZE_N / THREAD_SIZE_N), BLOCK_SIZE_M / THREAD_SIZE_M);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);

    // warm-up
    for(int run = 0; run < nIter; run++){
        BLOCK_SPARSE_MATMUL<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<dimGrid, dimBlock>>>(d_A, d_val, d_row, d_col, d_C, d_bias, M, K, N);
    }

    checkCudaErrors(cudaEventRecord(start));
    for(int run = 0; run < nIter; run++) {
        BLOCK_SPARSE_MATMUL<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<dimGrid, dimBlock>>>(d_A, d_val, d_row, d_col, d_C, d_bias, M, K, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_result, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    float msecPerMatrixMul = msecTotal / nIter;

    printf("float32 block sparse kernel gemm Time= %f msec\n", msecPerMatrixMul);

#if CPU_DEBUG
    HostComputation_sparse(h_A, h_row, h_col, h_val, h_C, h_bias, M, K, N, BLOCK_SIZE_K, BLOCK_SIZE_N);
    bool correct = true;
    double eps = 1.e-4;

    for(int i = 0; i < M * N; i++){
        double abs_err = abs(h_C[i] - h_result[i]);
        double dot_length = M;
        double abs_val = abs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (abs_err > eps) {
            printf("abs_val: %lf, rel_err: %lf, abs_val: %lf, dot_length: %lf \n", abs_val, rel_err, abs_val, dot_length);
            printf("Error! Matrix[%05d]=%lf, ref=%lf error term is %lf > %E\n",
                    i, h_result[i], h_C[i], rel_err, eps);
            correct = false;
            break;
        }
    }

    if(correct) printf("Result = Pass\n");
    else printf("Result = Fail\n");
#endif
    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);

    free(h_A);
    free(h_C);
    free(h_row);
    free(h_col);
    free(h_val);

    return EXIT_SUCCESS;
}

/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line

    int M = M_GLOBAL_VALUE, N = N_GLOBAL_VALUE, K = K_GLOBAL_VALUE;

    printf("MatrixA(%d, %d), MatrixB(%d, %d)\n", M, K, K, N);

    matrixMultiply(M, N, K);
}