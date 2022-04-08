source /root/anaconda/etc/profile.d/conda.sh
conda activate artifact

pushd checkpoints/bert
python bert_propagate_finegrained.py
python bert_propagate_coarsegrained.py
python bert_sota_coarse_onnx.py
python bert_sota_finegrained_onnx.py
python bert_original_onnx.py
popd

pushd checkpoints/mobilenet
# prepare the data
bash prepare_data.sh
python mobilenet_propagate_coarsegrained.py
python mobilenet_propagate_finegrained.py
python mobilenet_sota_finegrained_onnx.py
python mobilenet_sota_coarsegrained_onnx.py
python mobilenet_ori_onnx.py
popd

pip install transformers==4.12.3
pushd checkpoints/hubert
bash run_coarse.sh
bash run_finegrained.sh
bash run_ori_onnx.sh
bash run_finegrained_sota.sh
bash run_coarse_sota.sh
popd