# measure the jit time
cur_dir=`pwd`
source ~/.bashrc
echo "Curret directory ${cur_dir}"
mkdir ${cur_dir}/log

source ~/anaconda3/etc/profile.d/conda.sh
conda activate artifact

# bert coarse jit
pushd bert_coarse_jit
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_jit.log
popd

pushd bert_finegrained_jit
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_jit.log
popd

pushd bert_coarse_int8_jit
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_jit.log
popd

pushd mobilenet_coarse_jit
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_jit.log
popd

pushd mobilenet_finegrained_jit
/bin/bash run.sh > ${cur_dir}/log/mobilenet_finegrained_jit.log
popd

pushd mobilenet_coarse_int8_jit
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_int8_jit.log
popd
# change the python environment for hubert (which is only available in transformers 4.12)
conda activate hubert

pushd hubert_coarse_jit
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_jit.log
popd

pushd hubert_finegrained_jit
/bin/bash run.sh > ${cur_dir}/log/hubert_finegrained_jit.log
popd

pushd hubert_coarse_int8_jit
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_int8_jit.log
popd

# bert coarse sparta
pushd bert_coarse_sparta
# /bin/bash 
popd