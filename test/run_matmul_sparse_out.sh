rm matmul_sparse_out
nvcc -ccbin g++ -I../../common/inc -m64    -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O0  -o matmul_sparse_out.o -c matmul_sparse_out_32_32.cu -g -G
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O0  -o matmul_sparse_out matmul_sparse_out.o -g -G
./matmul_sparse_out
