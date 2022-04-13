source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact
nvcc -gencode arch=compute_75,code=sm_75 -lcublas -o cubls cublas.cu
nvcc -gencode arch=compute_75,code=sm_75 -lcusparse -o cusparse cusparse.cu
nvcc -forward-unknown-to-host-compiler -I/usr/local/cuda/include -I/root/sputnik  -L/usr/local/cuda/lib64  -L/usr/local/lib -lcudart -lspmm  --generate-code=arch=compute_75,code=sm_75 -std=c++14  sputnik.cu -o sputnik
mkdir log
for sparsity in 0.5 0.7 0.8 0.9 0.95 0.99
do
    echo $sparsity
    ./cusparse ${sparsity} > log/cusparse_${sparsity}.log
    ./sputnik ${sparsity} > log/sputnik_${sparsity}.log
done
# taco
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU50"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU70"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU80"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU90"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU50"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU50"