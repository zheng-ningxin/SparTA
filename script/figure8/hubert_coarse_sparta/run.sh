cp -r ../../checkpoints/hubert/artifact_hubert_coarse_onnx_with_tesa .
mkdir nnfusion_cfg
cp artifact_hubert_coarse_onnx_with_tesa/model_tesa.onnx nnfusion_cfg
pushd nnfusion_cfg
nnfusion model_tesa.onnx -f onnx
pushd nnfusion_rt/cuda_codegen
mkdir build
pushd build
cmake ..
make
ln -s ../Constant
./main_test
popd
popd
popd

