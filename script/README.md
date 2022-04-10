Steps to run

git clone https://github.com/zheng-ningxin/SparTA.git
conda activate artifact 
cd SparTA && python setup.py develop

# for all
bash init_env.sh
# for table 4
bash init_checkpoints.sh
# for figure8
cd figure8 bash run.sh