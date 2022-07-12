rm _dynamic_pattern*
nvcc -ccbin g++ -I../../common/inc -m64    -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O3  -o _dynamic_pattern.o -c dynamic_pattern.cu
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O3  -o _dynamic_pattern _dynamic_pattern.o 


nvcc -ccbin g++ -I../../common/inc -m64    -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O3  -o _dynamic_pattern_csr.o -c dynamic_pattern_csr.cu
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O3  -o _dynamic_pattern_csr _dynamic_pattern_csr.o 

