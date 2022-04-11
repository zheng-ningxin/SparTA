# init the environment
echo "init_env.sh"
bash init_env.sh
sleep 5s

echo "Get the propagated mask/cks"
bash init_checkpoints.sh


echo "reproducing figure8"
pushd figure8
bash run.sh
popd

echo "reproducing figure9"
pushd figure9
bash run.sh
popd

echo "reproducing figure10"
pushd figure10
bash run.sh
popd

echo "reproduceing figure13"
pushd figure13
bash run.sh
popd