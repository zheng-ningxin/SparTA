FINETUNE_EPOCH=1
python iterative_SA.py --model resnet50 --load-pretrained-model False --fine-tune-epochs $FINETUNE_EPOCH --cool-down-rate 0.2 --speed-up False
mv Iterative_SA.log Iterative_SA.log.baseline 
python iterative_SA.py --model resnet50 --load-pretrained-model False --fine-tune-epochs $FINETUNE_EPOCH --cool-down-rate 0.2 --speed-up True
python draw.py
