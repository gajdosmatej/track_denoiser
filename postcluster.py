from denoise_traces import Model, Plotting, QualityEstimator
import numpy
from tensorflow import keras
import keras.utils
import json
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

print("> Estimating model quality...")

data = QualityEstimator.reconstructionQuality(projections_sgnl[15000:], model.model.predict(projections[15000:], batch_size=500))

print("> Saving calculated data...")
numpy.savetxt(path + "wrong_signals.txt", data["false_signal"])
numpy.savetxt(path + "num_signals.txt", data["signal"])
numpy.savetxt(path + "residue_noise.txt", data["noise"])

print("> Plotting examples of reconstruction...")
Plotting.createPlot(model, projections_sgnl[-1], projections[-1])
matplotlib.pyplot.savefig(path + "example_reconstruction1.png")

Plotting.createPlot(model, projections_sgnl[-10], projections[-10])
matplotlib.pyplot.savefig(path + "example_reconstruction2.png")

Plotting.createPlot(model, projections_sgnl[-100], projections[-100])
matplotlib.pyplot.savefig(path + "example_reconstruction3.png")

print("> Plotting quality histograms...")
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

print("> Plotting model training history...")
f = open(path + "history.json", "r")
history = json.load(f)
f.close()
n = len(history["loss"])
matplotlib.pyplot.clf()
#matplotlib.pyplot.yscale("log")
matplotlib.pyplot.plot([i for i in range(3,n)], history["loss"][3:], label="Loss fuction")
matplotlib.pyplot.plot([i for i in range(3,n)], history["val_loss"][3:], label="Validation loss")
matplotlib.pyplot.legend()
matplotlib.pyplot.xlabel("epoch")
matplotlib.pyplot.ylabel("binary cross entropy")
matplotlib.pyplot.suptitle("Training of Model")
matplotlib.pyplot.savefig(path + "history.png")

