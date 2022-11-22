rm _m_condense _m_condense.o
# nvcc -ccbin g++ -I../../common/inc -m64  -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O0  -o _m_condense.o -c test_condense_m.cu
# nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O0  -o _m_condense  _m_condense.o
nvcc -ccbin g++ -I../../common/inc -m64  -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_70,code=sm_70 -O3  -o _m_condense.o -c block_sparse_nt_condense_m.cu
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_70,code=sm_70 -O3  -o _m_condense  _m_condense.o
#nvcc -ccbin g++ -I../../common/inc -m64  -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O0 -g -G  -o _m_condense.o -c block_sparse_nt_condense_m.cu
#nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O0 -G -g  -o _m_condense  _m_condense.o
./_m_condense
