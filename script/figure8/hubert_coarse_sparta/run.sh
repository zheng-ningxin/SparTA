cp -r ../../checkpoints/hubert/artifact_hubert_coarse_onnx_with_tesa .
mkdir nnfusion_cfg
cp artifact_hubert_coarse_onnx_with_tesa/model_tesa.onnx nnfusion_cfg
pushd nnfusion_cfg
nnfusion model_tesa.onnx -f onnx
popd

