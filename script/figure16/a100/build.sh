SPUTNIK_ROOT=/home/quzha/sputnik
CUSPARSELT_ROOT=/home/quzha/libcusparse_lt
nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  sparta.cu -o sparta
nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  sputnik.cu -o sputnik
