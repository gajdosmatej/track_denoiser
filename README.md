# track_denoiser
This repository forms the codebase for the paper M. Gajdoš, H.N. da Luz, G.G.A. de Souza, M. Bregant, *TPC track denoising and recognition using convolutional neural networks*, CPC (2025), https://doi.org/10.1016/j.cpc.2025.109608.

The dependencies are Python with numpy, matplotlib and tensorflow with keras. Additionally to the files in this repository, the data directories are necessary, with the following structure:
- data/ (the *root data directory*)
  - data/simulated/
    - data/simulated/clean
      - data/simulated/clean/BATCHINDEX.npy
    - data/simulated/noisy
      - data/simulated/noisy/BATCHINDEX.npy
  - data/x17/
    - data/x17/clean
      - data/x17/clean/trackTRACKNUMBER.txt
    - data/x17/noisy
      - data/x17/noisy/trackTRACKNUMBER.txt
    - data/x17/gauge_backgrounds
      - data/x17/gauge_backgrounds/trackTRACKNUMBER.txt

## Setup on Linux
1. **Create Python virtual environment:** `python3 -m venv .venv` (or possibly just `python` instead of `python3`, depending on the Linux distribution)
2. **Activate the virtual environment:** `source .venv/bin/activate`. This step needs to be applied each time when working with this project.
3. **Install dependencies:** `pip install -r dependencies.txt`.
4. **Create data folders:** If you have access to UTEF-X17 MetaCentrum storage, use the bash script `init_structure_data.sh`, which will also download the measured cosmics data. Otherwise, use the script `init_structure.sh` to construct the correct data folders tree.

## Misc
The data for data/simulated/ are generated using track_generator.py, each .npy file contains 5000 event arrays. The experimental data are available upon request and each .txt file contains one event, describing the array through rows of the form $(x,y,z,E)$, where $E$ stands for the amplitude; coordinates not present in any row are thought as describing zero amplitude. data/x17/gauge_backgrounds contain measured events with tracks removed, so that only noisy patterns are present which is one way to generate noise in track_generator.py. It is, however, entirely possible to generate only synthetic noise by changing `USE_MEASURED_NOISE` in track_generator.py to `0`.

The script track_generator.py is used to generate clean and corresponding noisy events. It is used as 
`python track_generator.py -n DESIRED_NUMBER_OF_BATCHES -p PATH_TO_ROOT_DATA_DIRECTORY -l 0/1`.
Flag `-l 0` generates data for denoising (i.e. ground truth are 3D tensors of clean events), whereas `-l 1` generates data for labeling (ground truth are labels 0/1, former for track present in the noisy event, latter for track not present in the noisy event; this feature is currently not being used, but it is available). Each batch contains 5000 event pairs.

Classes and support functions are stored in classes/ and used for plotting, NN model training, convenient data IO, clusterisation and more. 

Jupyter notebook analysis.ipynb contains some of the more interesting parts of the whole denoising process and its analysis.
