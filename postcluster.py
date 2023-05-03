from denoise_traces import Model, Plotting, QualityEstimator
import numpy
from tensorflow import keras
import keras.utils
import pandas
import matplotlib.pyplot

zy_projections = numpy.load("./data/data_noise_zy.npy")
zx_projections = numpy.load("./data/data_noise_zx.npy")
yx_projections = numpy.load("./data/data_noise_yx.npy")
zy_projections_sgnl = numpy.load("./data/data_signal_zy.npy")
zx_projections_sgnl = numpy.load("./data/data_signal_zx.npy")
yx_projections_sgnl = numpy.load("./data/data_signal_yx.npy")

zx_projections = zx_projections.reshape( (*zx_projections.shape, 1) )
zx_projections_sgnl = zx_projections_sgnl.reshape( (*zx_projections_sgnl.shape, 1) )


name = input("Model name: ")
path = "./models/" + name + "/"

model_zx = Model.load(path + name, "zx")
keras.utils.plot_model(model_zx.model, to_file= path + "architecture.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)
data = QualityEstimator.reconstructionQuality(zx_projections_sgnl[15000:], model_zx.model(zx_projections[15000:]))
numpy.savetxt(path + "wrong_signals.txt", data["false_signal"])
numpy.savetxt(path + "num_signals.txt", data["signal"])
numpy.savetxt(path + "residue_noise.txt", data["noise"])

Plotting.createPlot(model_zx, zx_projections_sgnl[-1], zx_projections[-1])
matplotlib.pyplot.savefig(path + "example_reconstruction.png")

matplotlib.pyplot.clf()
matplotlib.pyplot.hist(data["false_signal"] / data["signal"], bins=50)
matplotlib.pyplot.xlabel("Ratio of false reconstructed signal and signal")
matplotlib.pyplot.ylabel("#")
matplotlib.pyplot.suptitle("Quality of Signal Reconstruction")
matplotlib.pyplot.savefig(path + "hist_signal.png")

matplotlib.pyplot.clf()
matplotlib.pyplot.hist(data["noise"], bins=50)
matplotlib.pyplot.xlabel("Unfiltered noise")
matplotlib.pyplot.ylabel("#")
matplotlib.pyplot.suptitle("Quality of Noise Filtering")
matplotlib.pyplot.savefig(path + "hist_noise.png")