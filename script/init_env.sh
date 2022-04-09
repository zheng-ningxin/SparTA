source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

HTTP_PORT=8881 BACKEND=c-cuda nohup antares rest-server &
mkdir ~/.cache/antares/codehub/
cp figure8/codehub/* ~/.cache/antares/codehub/