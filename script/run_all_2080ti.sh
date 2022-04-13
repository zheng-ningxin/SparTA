# init the environment

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

echo "reproducing figure13"
pushd figure13
bash run.sh
popd

echo "reproducing figure15"
pushd figure15
bash run.sh
popd


echo "reproducing figure19"
pushd figure19
bash run.sh
popd
