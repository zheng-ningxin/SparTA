python iterative_SA.py --model resnet50 --load-pretrained-model False --fine-tune-epochs 10 --cool-down-rate 0.2 --speed-up False
cp Iterative_SA.log Iterative_SA.log.baseline 
python iterative_SA.py --model resnet50 --load-pretrained-model False --fine-tune-epochs 10 --cool-down-rate 0.2 --speed-up True
