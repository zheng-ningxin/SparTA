#include "common.h"
#include "cusparse.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;
// Macro definition for the cuda and cusparse
// cuSparse SPMM interface

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
#define CUSPARSE_SAFE_CALL(func)                                                                \
    do                                                                                          \
    {                                                                                           \
        cusparseStatus_t e = (func);                                                            \
        if (e != CUSPARSE_STATUS_SUCCESS)                                                       \
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

int cusparse_csr_convert(
    float* dense_value,
    int n_row,
    int n_col,
    int * csr_row,
    int * csr_col,
    float * csr_val)
{
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CUSPARSE_SAFE_CALL(cusparseCreate(&handle));

    CUSPARSE_SAFE_CALL(cusparseCreateDnMat(&matA, n_row, n_col, n_col, dense_value,
                                    CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CUSPARSE_SAFE_CALL( cusparseCreateCsr(&matB, n_row, n_col, 0,
                                    csr_row, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    CUSPARSE_SAFE_CALL( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) );
    CUDA_SAFE_CALL( cudaMalloc(&dBuffer, bufferSize) );
    CUSPARSE_SAFE_CALL( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) );
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CUSPARSE_SAFE_CALL( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                        &nnz) );
    // torch::Tensor csr_col = torch::empty_like({nnz}, csr_row);
    // torch::Tensor csr_values = torch::empty_like({nnz}, dense_values);
    CUSPARSE_SAFE_CALL( cusparseCsrSetPointers(matB, csr_row, csr_col, csr_val) );
    // execute Sparse to Dense conversion
    CUSPARSE_SAFE_CALL( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) );
    CUSPARSE_SAFE_CALL( cusparseDestroyDnMat(matA) );
    CUSPARSE_SAFE_CALL( cusparseDestroySpMat(matB) );
    CUSPARSE_SAFE_CALL( cusparseDestroy(handle) );
}

std::vector<at::Tensor> cusparse_convert_forward(
    torch::Tensor dense_values)
{   
    cudaSetDevice(dense_values.get_device());
    // the weight shape should be KxN
    int n_row = dense_values.size(0);
    int n_col = dense_values.size(1);
    auto csr_row_options =
    torch::TensorOptions().dtype(torch::kInt32).device(dense_values.options().device());
    torch::Tensor csr_row = torch::empty({n_row+1}, csr_row_options);
    torch::Tensor csr_col = torch::empty({n_row*n_col}, csr_row_options);
    torch::Tensor csr_val = torch::empty_like(dense_values);
    AT_DISPATCH_FLOATING_TYPES(dense_values.type(), "cusparse convert_bcsr", ([&]
    { cusparse_csr_convert(
            dense_values.data_ptr<float>(),
            n_row,
            n_col,
            csr_row.data_ptr<int>(),
            csr_col.data_ptr<int>(),
            csr_val.data_ptr<float>()
        ); }));
    std::vector<at::Tensor> csr({csr_row, csr_col, csr_val});
    return csr;
}

void cusparse_csr_sparse_to_dense(int num_rows,
                                  int num_cols,
                                  int nnz,
                                  int * d_csr_offsets,
                                  int * d_csr_columns,
                                  float * d_csr_values,
                                  float * d_dense
                                    )
{
    // // CUSPARSE APIs
    // int num_rows = dense_shape[0];
    // int num_cols = dense_shape[1];

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CUSPARSE_SAFE_CALL( cusparseCreate(&handle) );
    // Create sparse matrix A in CSR format
    CUSPARSE_SAFE_CALL( cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                                      d_csr_offsets, d_csr_columns,
                                      d_csr_values, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    // Create dense matrix B
    CUSPARSE_SAFE_CALL( cusparseCreateDnMat(&matB, num_rows, num_cols, num_cols, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) );
    // allocate an external buffer if needed
    CUSPARSE_SAFE_CALL( cusparseSparseToDense_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                        &bufferSize) );
    CUDA_SAFE_CALL( cudaMalloc(&dBuffer, bufferSize) );

    // execute Sparse to Dense conversion
    CUSPARSE_SAFE_CALL( cusparseSparseToDense(handle, matA, matB,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                          dBuffer) );
    // destroy matrix/vector descriptors
    CUDA_SAFE_CALL( cudaFree(dBuffer) );
    CUSPARSE_SAFE_CALL( cusparseDestroySpMat(matA) );
    CUSPARSE_SAFE_CALL( cusparseDestroyDnMat(matB) );
    CUSPARSE_SAFE_CALL( cusparseDestroy(handle) );

}

at::Tensor cusparse_convert_backward(
    torch::Tensor csr_row,
    torch::Tensor csr_col,
    torch::Tensor csr_val,
    int n_row,
    int n_col,
    int nnz
){
    cudaSetDevice(csr_row.get_device());
    // the weight shape should be KxN
    // assert( n_row == csr_row.size(0));
    torch::Tensor dense_out = torch::empty({n_row, n_col}, csr_val.options());
    AT_DISPATCH_FLOATING_TYPES(csr_val.type(), "cusparse convert_csr", ([&]
    { cusparse_csr_sparse_to_dense(
            n_row,
            n_col,
            nnz,
            csr_row.data_ptr<int>(),
            csr_col.data_ptr<int>(),
            csr_val.data_ptr<float>(),
            dense_out.data_ptr<float>()
        ); }));
    
    return dense_out;

}
