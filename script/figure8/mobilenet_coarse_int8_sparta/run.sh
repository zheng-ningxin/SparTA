rm ~/.cache/nnfusion/kernel_cache.db
python mobilenet_codegen_int8.py
cp -r ../../checkpoints/mobilenet/artifact_mobilenet_coarse_onnx_with_tesa .
python prepare_kernel_cfg.py --in_dir artifact_mobilenet_coarse_onnx_with_tesa --out_dir nnfusion_cfg
pushd nnfusion_cfg
nnfusion model_tesa.onnx  -f onnx -fspargen_cfg config -frun_step 2000
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
