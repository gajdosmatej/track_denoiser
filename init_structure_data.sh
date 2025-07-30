mkdir data
mkdir data/simulated
mkdir data/x17
mkdir data/simulated/clean
mkdir data/simulated/noisy
echo "#> Data directory structure created. Data will now be downloaded from MetaCentrum."
cd ./data/x17/
read -p "#> Enter your MetaCentrum username: " username
scp $username@tarkil.metacentrum.cz:/storage/projects/utefx17/matej/data/* .
for file in *.tar.gz; do tar -xzf $file; done
echo "#> Data downloaded and extracted. Downloading track recognition labels."
ln -s clean_5sigma clean
ln -s noisy_5sigma noisy
rm ./*.tar.gz
cd ../..
scp $username@tarkil.metacentrum.cz:/storage/projects/utefx17/matej/labels.txt .