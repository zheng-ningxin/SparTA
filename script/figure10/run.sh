#!/bin/bash
arr=("jit" "tvm" "tvm-s" "trt" "rammer" "rammer-s" "sparta")
 
for framework in ${arr[@]}
do
    pushd $framework
    bash run.sh
    popd
done
