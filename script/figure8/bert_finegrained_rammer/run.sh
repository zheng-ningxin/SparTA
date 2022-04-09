cp ../../checkpoints/bert/bert_ori_no_tesa.onnx .
nnfusion bert_ori_no_tesa.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1
