from denoise_traces import Model, Plotting, QualityEstimator
import numpy
from tensorflow import keras
import keras.utils
import json
import copy
import matplotlib.pyplot

names_projections = 	{ "zy": "./data/data_noise_zy.npy",
						"zx": "./data/data_noise_zx.npy",
						"yx": "./data/data_noise_yx.npy" }

names_projections_sgnl = 	{ "zy": "./data/data_signal_zy.npy",
							"zx": "./data/data_signal_zx.npy",
							"yx": "./data/data_signal_yx.npy" }


name = input("Model name: ")
path = "./models/" + name + "/"
plane = input("Type (zx/zy/yx): ")

projections = numpy.load(names_projections[plane])
projections_sgnl = numpy.load(names_projections_sgnl[plane])

projections = numpy.reshape(projections, (*projections.shape, 1))
projections_sgnl = numpy.reshape(projections_sgnl, (*projections_sgnl.shape, 1))

print("> Loading model...")
model = Model.load(path + name, plane)

print("> Plotting model architecture...")
keras.utils.plot_model(model.model, to_file= path + "architecture.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)
with open(path + "architecture.txt", "w") as f:
	model.model.summary(print_fn = lambda x: print(x, file=f))


print("> Plotting examples of reconstruction...")
Plotting.createPlot(model, projections_sgnl[-1], projections[-1])
matplotlib.pyplot.savefig(path + "example_reconstruction1.png", bbox_inches='tight')

Plotting.createPlot(model, projections_sgnl[-10], projections[-10])
matplotlib.pyplot.savefig(path + "example_reconstruction2.png", bbox_inches='tight')

Plotting.createPlot(model, projections_sgnl[-100], projections[-100])
matplotlib.pyplot.savefig(path + "example_reconstruction3.png", bbox_inches='tight')


print("> Estimating model quality...")
projections_sgnl = projections_sgnl[:100000]
projections = projections[:100000]
predictions = model.model.predict(projections, batch_size=500)

data, _ = QualityEstimator.reconstructionQuality(projections_sgnl, predictions)

print("> Saving calculated data...")
numpy.savetxt(path + "wrong_signals.txt", data["false_signal"])
numpy.savetxt(path + "num_signals.txt", data["signal"])
numpy.savetxt(path + "residue_noise.txt", data["noise"])

print("> Plotting quality histograms...")
matplotlib.pyplot.clf()
matplotlib.pyplot.hist(data["false_signal"] / data["signal"], bins=200, log=True)
matplotlib.pyplot.xlabel("Ratio of false reconstructed signal and signal")
matplotlib.pyplot.ylabel("log(#)")
matplotlib.pyplot.ylim(None, 10**4)
matplotlib.pyplot.xlim(None, 1.2)
matplotlib.pyplot.suptitle("Quality of Signal Reconstruction\n Model " + name)
matplotlib.pyplot.savefig(path + "hist_signal_log.png", bbox_inches='tight')

matplotlib.pyplot.clf()
matplotlib.pyplot.hist(data["noise"], bins=200, log=True)
matplotlib.pyplot.xlabel("Unfiltered noise")
matplotlib.pyplot.ylabel("log(#)")
matplotlib.pyplot.ylim(None, 10**5)
matplotlib.pyplot.xlim(None, 2)
matplotlib.pyplot.suptitle("Quality of Noise Filtering\n Model " + name)
matplotlib.pyplot.savefig(path + "hist_noise_log.png", bbox_inches='tight')

matplotlib.pyplot.clf()
matplotlib.pyplot.hist(data["false_signal"] / data["signal"], bins=200)
matplotlib.pyplot.xlabel("Ratio of false reconstructed signal and signal")
matplotlib.pyplot.ylabel("#")
matplotlib.pyplot.ylim(None, 10000)
matplotlib.pyplot.xlim(None, 1.2)
matplotlib.pyplot.suptitle("Quality of Signal Reconstruction\n Model " + name)
matplotlib.pyplot.savefig(path + "hist_signal.png", bbox_inches='tight')

matplotlib.pyplot.clf()
matplotlib.pyplot.hist(data["noise"], bins=200)
matplotlib.pyplot.xlabel("Unfiltered noise")
matplotlib.pyplot.ylabel("#")
matplotlib.pyplot.ylim(None, 50000)
matplotlib.pyplot.xlim(None, 2)
matplotlib.pyplot.suptitle("Quality of Noise Filtering\n Model " + name)
matplotlib.pyplot.savefig(path + "hist_noise.png", bbox_inches='tight')


fixed_cmap = copy.copy(matplotlib.cm.get_cmap('magma'))
fixed_cmap.set_bad((0,0,0))	#fix pixels with zero occurrence - otherwise problem for LogNorm
matplotlib.pyplot.clf()
matplotlib.pyplot.hist2d(data["false_signal"] / data["signal"], data["noise"], [70,70], [[0,1.2], [0,2]], cmap=fixed_cmap, norm=matplotlib.colors.LogNorm())
matplotlib.pyplot.xlabel("Ratio of false reconstructed signal and signal")
matplotlib.pyplot.ylabel("Unfiltered noise")
matplotlib.pyplot.suptitle("2D Log Histogram Of Reconstruction Quality Metrics\n Model " + name + ", $N = 10^{5}$")
matplotlib.pyplot.savefig(path + "hist_2d.png", bbox_inches='tight')

print("> Plotting model training history...")
f = open(path + "history.json", "r")
history = json.load(f)
f.close()
n = len(history["loss"])
matplotlib.pyplot.clf()
#matplotlib.pyplot.yscale("log")
matplotlib.pyplot.plot([i for i in range(1,n)], history["loss"][1:], label="Loss fuction")
matplotlib.pyplot.plot([i for i in range(1,n)], history["val_loss"][1:], label="Validation loss")
matplotlib.pyplot.legend()
matplotlib.pyplot.xlabel("epoch")
matplotlib.pyplot.ylabel("binary cross entropy")
matplotlib.pyplot.suptitle("Training of Model " + name)
matplotlib.pyplot.savefig(path + "history.png", bbox_inches='tight')

