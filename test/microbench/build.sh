nvcc -lcublas -o cublas cublas.cu
nvcc -gencode arch=compute_70,code=sm_70 -lcusparse -o cusparse cusparse.cu

SPUTNIK_ROOT=/root/sputnik
nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik  -lcusparse -lcudart -lspmm  --generate-code=arch=compute_70,code=sm_70 -std=c++14  sputnik.cu -o sputnik