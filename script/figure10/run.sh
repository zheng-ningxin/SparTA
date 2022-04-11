#!/bin/bash
cur_dir=`pwd`
mkdir -p ${cur_dir}/log
arr=("jit" "tvm" "tvm-s" "trt" "rammer" "rammer-s" "sparta")
 
for framework in ${arr[@]}
do
    pushd $framework
    bash run.sh > ${cur_dir}/log/${framework}.log
    popd
done

python draw.py