cp ../../checkpoints/bert/artifact_bert_ori/bert_ori_no_tesa.onnx .
nnfusion bert_ori_no_tesa.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1
pushd nnfusion_rt/cuda_codegen
mkdir build
pushd build
cmake ..
make
./main_test
popd
popd