# track_denoiser

Additionally to these files, the data directories are necessary, with the following structure:
- ./data/ (the *root data directory*)
  - ./data/simulated/
    - ./data/simulated/clean
    - ./data/simulated/noisy
  - ./data/x17/
    - ./data/x17/clean
      - ./data/x17/clean/goodtracks
      - ./data/x17/clean/othertracks
    - ./data/x17/noisy
      - ./data/x17/noisy/goodtracks
      - ./data/x17/noisy/othertracks

The script track_generator.py is used for clean and corresponding noisy events generation. It is called as 
`python track_generator.py -n DESIRED_NUMBER_OF_EVENTS -p PATH_TO_ROOT_DATA_DIRECTORY`.


File classes.py contains several important classes used for plotting, NN model training, convenient data IO, clusterisation and more. It is meant to be imported and used in the main script file, which is not included in this repository, because it is changing often the time.

The directory models/ contains models which have been already trained, along with histograms of noise supression and signal reconstruction, schemes of model architecture, various data, plot of training history and several example pictures of the denoising process. Most of these files are created by the script postprocess.py after the model is trained.

There are NN training scripts ready to be used on MetaCentrum cluster in the cluster/ directory. In architectures_list.py, one can specify several model architectures that will be trained on the cluster. Then, script move_to_cluster.sh has to be run, which moves all the important files to the cluster and connects the user there. cluster_script.sh has to be then submited on the cluster by qsub command. After the job is completed, the final step is to run move_from_cluster.sh locally - the trained models will be then moved to the ./models/3D/ dictionary.
