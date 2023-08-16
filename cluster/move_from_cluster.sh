#!/bin/bash

scp -r gajdoma6@tarkil.metacentrum.cz:/storage/plzen1/home/gajdoma6/models ./cluster_models
scp -r gajdoma6@tarkil.metacentrum.cz:/storage/plzen1/home/gajdoma6/histories ./cluster_histories

#num_existing_models="$(ls ../models/3D | wc -l)"
#num_new_models="$(ls ./cluster_models | wc -l)"

#for (( i=0; i < $num_new_models; ++i ))
#do
#	name=$((num_existing_models+i))
#	mkdir ../models/3D/$name
#	mv ./cluster_models/$i ../models/3D/$name/model
#	mv ./cluster_histories/$i.json ../models/3D/$name/history.json
#done

cd ./cluster_models
for model in *;
do
	mkdir ../../models/3D/"$model"
	mv -vn "$model" ../../models/3D/"$model"/model
	mv -vn ../cluster_histories/"$model".json ../../models/3D/"$model"/history.json
done
cd ..
rm -fr ./cluster_models
rm -fr ./cluster_histories