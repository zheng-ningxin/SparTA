
# Steps to run the experiments of 2080Ti
## Initialization
```
build the docker file
sudo docker run -it -v /data/znx/:/data --gpus all --shm-size 16G zhengningxin/artifact10
```
```
# get the checkpoints
wget xxxxxxxxxxxxxx
# install the sparta
git clone https://github.com/zheng-ningxin/SparTA.git
conda activate artifact
cd SparTA && python setup.py develop
cd script && bash init_env.sh
bash run_all_2080ti.sh
```

# for all
bash init_env.sh
# for table 4
bash init_checkpoints.sh
# for figure8
cd figure8 bash run.sh