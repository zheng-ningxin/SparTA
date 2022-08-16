rm _run_condense _run_condense.o
nvcc -ccbin g++ -I../../common/inc -m64  -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O0  -o _run_condense.o -c test_condense_sparse.cu
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O0  -o _run_condense  _run_condense.o
./_run_condense
