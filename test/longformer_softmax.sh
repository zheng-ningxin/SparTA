rm longformer_softmax_perf longformer_softmax_perf.o
nvcc -ccbin g++ -I../../common/inc -m64  -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O0  -o longformer_softmax_perf.o -c longformer_softmax_perf.cu
nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O0  -o longformer_softmax_perf  longformer_softmax_perf.o
#nvcc -ccbin g++ -I../../common/inc -m64    -maxrregcount=255 -gencode arch=compute_75,code=sm_75 -O0  -o longformer_softmax_perf.o -c longformer_softmax_perf.cu -g -G
#nvcc -ccbin g++ -m64  -gencode arch=compute_75,code=sm_75 -O0  -o longformer_softmax_perf  longformer_softmax_perf.o -g -G
./longformer_softmax_perf
