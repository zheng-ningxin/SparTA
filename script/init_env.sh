source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

HTTP_PORT=8881 BACKEND=c-cuda nohup antares rest-server &
# wait the antares to be ready
sleep 5s
mkdir -p ~/.cache/antares/codehub/
cp figure8/codehub/* ~/.cache/antares/codehub/

# download the checkpoint
azcopy copy "https://nni.blob.core.windows.net/artifact/cks?sp=r&st=2022-04-13T11:08:06Z&se=2023-01-06T19:08:06Z&spr=https&sv=2020-08-04&sr=c&sig=Um4u6yyPjByAIqY1UEH%2BxyJCPRSgNDzA%2BeNNofyzgUg%3D" "." --recursive
rm -rf checkpoints/bert/checkpoints
rm -rf checkpoints/mobilenet/checkpoints
rm -rf checkpoints/hubert/checkpoints
mv cks/bert/checkpoints checkpoints/bert/
mv cks/mobilenet/checkpoints checkpoints/mobilenet/
mv cks/hubert/checkpoints checkpoints/hubert/ 