# track_denoiser

The file trace_bkgnd_gen.py contains methods for generating and visualising noisy and corresponding clean track events in 3D space.

There are several important classes used for plotting, NN model training and IO in file denoise_traces.py. It is meant to be imported and used in the main script file, which is not included in this repository, because it is changing all the time.

The directory models/ contains models which have been already trained, along with histograms of noise supression and signal reconstruction, schemes of model architecture, various data, plot of training history and several example pictures of the denoising process. Most of these files are created by the script postprocess.py after the model is trained.

There are NN training scripts ready to be used on MetaCentrum cluster in the cluster/ directory. In architectures_list.py, one can specify several model architectures that will be trained on the cluster. Then, script move_to_cluster.sh has to be run, which moves all the important files to the cluster and connects the user there. cluster_script.sh has to be then submited on the cluster by qsub command. After the job is completed, the final step is to run move_from_cluster.sh locally - the trained models will be then moved to the ./models/3D/ dictionary.

More TODO.
