rm _dynamic_seqlen*
nvcc -ccbin g++ -I../../common/inc -m64    -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O3  -o _dynamic_seqlen.o -c test_dynamic_seqlen.cu
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O3  -o _dynamic_seqlen _dynamic_seqlen.o 

