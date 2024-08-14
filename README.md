# track_denoiser

Additionally to these files, the data directories are necessary, with the following structure:
- ./data/ (the *root data directory*)
  - ./data/simulated/
    - ./data/simulated/clean
      - ./data/simulated/clean/<BATCHINDEX>.npy
    - ./data/simulated/noisy
      - ./data/simulated/noisy/<BATCHINDEX>.npy
  - ./data/x17/
    - ./data/x17/clean
      - ./data/x17/clean/track<TRACKNUMBER>.txt
    - ./data/x17/noisy
      - ./data/x17/noisy/track<TRACKNUMBER>.txt

The script track_generator.py is used for clean and corresponding noisy events generation. It is called as 
`python track_generator.py -n DESIRED_NUMBER_OF_BATCHES -p PATH_TO_ROOT_DATA_DIRECTORY -l 0/1`.
Flag `-l 0` generates data for denoising (i.e. ground truth are 3D tensors of clean events), whereas `-l 1` generates data for labeling (ground truth are labels 0/1, former for track present in the noisy event, latter for track not present in the noisy event). Each batch contains 5000 event pairs.

File classes.py contains several important classes used for plotting, NN model training, convenient data IO, clusterisation and more. 

Jupyter notebook analysis.ipynb contains some of the more interesting parts of the whole denoising process and its analysis.
