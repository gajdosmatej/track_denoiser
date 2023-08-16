from denoise_traces import Model, Plotting, QualityEstimator
import numpy
from tensorflow import keras
import keras.utils
import json
import os
import matplotlib.pyplot
import preprocess_data

names_projections = 	{ "zy": "./data/data_noise_zy.npy",
						"zx": "./data/data_noise_zx.npy",
						"yx": "./data/data_noise_yx.npy" }

names_projections_sgnl = 	{ "zy": "./data/data_signal_zy.npy",
							"zx": "./data/data_signal_zx.npy",
							"yx": "./data/data_signal_yx.npy" }

def postprocessModel(path):
	print("> Loading model...")
	model = keras.models.load_model(path + "model")

	print("> Plotting model architecture...")
	keras.utils.plot_model(model, to_file= path + "architecture.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)
	with open(path + "architecture.txt", "w") as f:
		model.summary(print_fn = lambda x: print(x, file=f))


	print("> Plotting examples of reconstruction...")

	x17 = [event for (_, event) in preprocess_data.loadX17Data("goodtracks")]
	example_indices = [0, 12, 21]
	for i in range(3):
		noisy = x17[example_indices[i]]
		noisy = noisy / numpy.max(noisy)
		noisy = numpy.reshape( noisy, (1,12,14,208,1))

		reconstr = numpy.reshape(model(noisy)[0], (12,14,208))
		noisy = numpy.reshape(noisy[0], (12,14,208))
		fig, ax = matplotlib.pyplot.subplots(3, 2)

		#ax[0][1].imshow(numpy.sum(reconstr, axis=0), cmap="gray", norm=matplotlib.colors.LogNorm() )
		ax[0][1].imshow(numpy.sum(reconstr, axis=0), cmap="gray", vmin=0, vmax=1)
		ax[0][1].set_title("Reconstruction, ZY Projection")
		ax[0][1].set_xlabel("z")
		ax[0][1].set_ylabel("y")

		ax[0][0].imshow(numpy.sum(noisy, axis=0), cmap="gray", vmin=0, vmax=1)
		ax[0][0].set_title("Noisy, ZY Projection")
		ax[0][0].set_xlabel("z")
		ax[0][0].set_ylabel("y")

		ax[1][1].imshow(numpy.sum(reconstr, axis=1), cmap="gray", vmin=0, vmax=1)
		ax[1][1].set_title("Reconstruction, ZX Projection")
		ax[1][1].set_xlabel("z")
		ax[1][1].set_ylabel("x")

		ax[1][0].imshow(numpy.sum(noisy, axis=1), cmap="gray", vmin=0, vmax=1)
		ax[1][0].set_title("Noisy, ZX Projection")
		ax[1][0].set_xlabel("z")
		ax[1][0].set_ylabel("x")

		ax[2][1].imshow(numpy.sum(reconstr, axis=2), cmap="gray", vmin=0, vmax=1)
		ax[2][1].set_title("Reconstruction, YX Projection")
		ax[2][1].set_xlabel("y")
		ax[2][1].set_ylabel("x")

		ax[2][0].imshow(numpy.sum(noisy, axis=2), cmap="gray", vmin=0, vmax=1)
		ax[2][0].set_title("Noisy, YX Projection")
		ax[2][0].set_xlabel("y")
		ax[2][0].set_ylabel("x")

		matplotlib.pyplot.savefig(path + "example_reconstruction" + str(i) + ".pdf", bbox_inches="tight")
		matplotlib.pyplot.clf()

	'''
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
	matplotlib.pyplot.savefig(path + "hist_2d.png", bbox_inches='tight')'''

	print("> Plotting model training history...")
	f = open(path + "history.json", "r")
	history = json.load(f)
	f.close()
	n = len(history["loss"])
	matplotlib.pyplot.clf()
	#matplotlib.pyplot.yscale("log")
	matplotlib.pyplot.scatter([i for i in range(0,n)], history["loss"][0:], label="Loss fuction")
	matplotlib.pyplot.scatter([i for i in range(0,n)], history["val_loss"][0:], label="Validation loss")
	matplotlib.pyplot.legend()
	matplotlib.pyplot.xlabel("epoch")
	matplotlib.pyplot.ylabel("binary cross entropy (log scale)")
	matplotlib.pyplot.yscale("log")
	matplotlib.pyplot.suptitle("Training of Model " + name)
	matplotlib.pyplot.savefig(path + "history.pdf", bbox_inches='tight')

	print("> Evaluating special inputs...")

	special_inputs = [numpy.zeros((12,14,208)), numpy.ones((12,14,208)), numpy.pad(numpy.ones((1,1,208)), [(5,6), (6,7), (0,0)])]
	titles = ["Output of Zero Tensor", "Output of Ones Tensor", "Output of Central Line"]
	filenames = ["zeros.pdf", "ones.pdf", "line.pdf"]
	for i in range(3):
		neural_input = numpy.reshape(special_inputs[i], (1,12,14,208,1))
		reconstruction = numpy.reshape( model(neural_input)[0], (12,14,208))
		fig, ax = matplotlib.pyplot.subplots(3, 2)
		ax[0,0].imshow(numpy.sum(special_inputs[i], axis=0), cmap="gray", vmin=0, vmax=1)
		ax[0,0].set_title("Input, ZY Projection")
		ax[0,0].set_xlabel("z")
		ax[0,0].set_ylabel("y")

		ax[0,1].imshow(numpy.sum(reconstruction, axis=0), cmap="gray", vmin=0, vmax=1)
		ax[0,1].set_title("Output, ZY Projection")
		ax[0,1].set_xlabel("z")
		ax[0,1].set_ylabel("y")

		ax[1,0].imshow(numpy.sum(special_inputs[i], axis=1), cmap="gray", vmin=0, vmax=1)
		ax[1,0].set_title("Input, ZX Projection")
		ax[1,0].set_xlabel("z")
		ax[1,0].set_ylabel("x")

		ax[1,1].imshow(numpy.sum(reconstruction, axis=1), cmap="gray", vmin=0, vmax=1)
		ax[1,1].set_title("Output, ZX Projection")
		ax[1,1].set_xlabel("z")
		ax[1,1].set_ylabel("x")

		ax[2,0].imshow(numpy.sum(special_inputs[i], axis=2), cmap="gray", vmin=0, vmax=1)
		ax[2,0].set_title("Input, YX Projection")
		ax[2,0].set_xlabel("y")
		ax[2,0].set_ylabel("x")

		ax[2,1].imshow(numpy.sum(reconstruction, axis=2), cmap="gray", vmin=0, vmax=1)
		ax[2,1].set_title("Output, YX Projection")
		ax[2,1].set_xlabel("y")
		ax[2,1].set_ylabel("x")

		fig.suptitle(titles[i])
		matplotlib.pyplot.savefig(path + filenames[i], bbox_inches="tight")
		matplotlib.pyplot.clf()
		matplotlib.pyplot.close()


name = input("Model name: ")
if name == "ALL":
	counter = 1
	num_all = len(os.listdir("./models/3D/"))
	for model_name in os.listdir("./models/3D/"):
		print("-------------------------------------")
		print("> PROCESSING MODEL " + model_name + " [" + str(counter) + "/" + str(num_all) + "]")
		print("-------------------------------------")
		postprocessModel("./models/3D/" + model_name + "/")
		counter += 1
else:
	postprocessModel("./models/3D/" + name + "/")
