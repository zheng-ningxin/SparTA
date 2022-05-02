rm matmul_sdd_32_32.o matmul_sdd_32_32
nvcc -ccbin g++ -I../../common/inc -m64    -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O0  -o matmul_sdd_32_32.o -c matmul_sdd_32_32.cu  -g -G
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O0  -o  matmul_sdd_32_32  matmul_sdd_32_32.o  -g -G
./matmul_sdd_32_32
