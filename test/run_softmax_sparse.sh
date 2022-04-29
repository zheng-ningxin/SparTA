rm softmax_sparse_32_32
nvcc -ccbin g++ -I../../common/inc -m64    -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O0  -o softmax_sparse_32_32.o -c softmax_sparse_32_32.cu -g -G
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O0  -o softmax_sparse_32_32  softmax_sparse_32_32.o -g -G
./softmax_sparse_32_32
