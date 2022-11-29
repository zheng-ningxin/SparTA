nvcc -lcublas -o cublas cublas.cu
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -lcusparse -o cusparse_convert cusparse_convert.cu
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -lcusparse -o cusparse cusparse.cu
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -O3  stile_finegrained.cu -o stile_finegrained
SPUTNIK_ROOT=/root/sputnik
#nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik  -lcusparse -lcudart -lspmm  --generate-code=arch=compute_70,code=sm_70 -std=c++14  sputnik.cu -o sputnik
