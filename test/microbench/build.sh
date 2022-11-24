nvcc -lcublas -o cublas cublas.cu
nvcc -gencode arch=compute_70,code=sm_70 -lcusparse -o cusparse cusparse.cu
