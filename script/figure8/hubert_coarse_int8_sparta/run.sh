rm ~/.cache/nnfusion/kernel_cache.db
cp -r ../../checkpoints/hubert/artifact_hubert_coarse_onnx_with_tesa .
python hubert_codegen_int8.py
python prepare_kernel_cfg.py --in_dir artifact_hubert_coarse_onnx_with_tesa --out_dir ./nnfusion_cfg
pushd ./nnfusion_cfg
nnfusion model_tesa.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1 -fspargen_cfg config -frun_step 1000
#nnfusion model.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1 -fblockfusion_level=0 -fcodegen_debug=true -fspargen_cfg config
pushd nnfusion_rt/cuda_codegen
cp nnfusion_rt.cu nnfusion_rt.back
printf "#include <mma.h>\\nusing namespace nvcuda;\\n#define MAX(a, b) ((a) > (b) ? (a) : (b))\\n" > nnfusion_rt.cu
cat nnfusion_rt.back >> nnfusion_rt.cu
mkdir build
pushd build 
cmake ..
make 
ln -s ../Constant
./main_test
popd
popd
popd
