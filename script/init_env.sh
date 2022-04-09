source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

HTTP_PORT=8881 BACKEND=c-cuda nohup antares rest-server &
# wait the antares to be ready
sleep 5s
mkdir -p ~/.cache/antares/codehub/
cp figure8/codehub/* ~/.cache/antares/codehub/