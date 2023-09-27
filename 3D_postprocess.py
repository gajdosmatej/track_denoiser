import numpy
from tensorflow import keras
import keras.utils
import json
import os
import matplotlib.pyplot
import matplotlib.animation
import copy
from denoise_traces import DataLoader, Plotting, ModelWrapper


def plotHistory(out_path :str, model_name :str):
	'''
	Plot the model's training history.
	'''

	f = open(out_path + "history.json", "r")
	history = json.load(f)
	f.close()
	n = len(history["loss"])
	matplotlib.pyplot.plot([i for i in range(0,n)], history["loss"][0:], label="Loss fuction")
	matplotlib.pyplot.plot([i for i in range(0,n)], history["val_loss"][0:], label="Validation loss")
	matplotlib.pyplot.legend()
	matplotlib.pyplot.xlabel("epoch")
	matplotlib.pyplot.ylabel("binary cross entropy (log scale)")
	matplotlib.pyplot.yscale("log")
	matplotlib.pyplot.suptitle("Training of Model " + model_name)
	matplotlib.pyplot.savefig(out_path + "history.pdf", bbox_inches='tight')
	matplotlib.pyplot.close()


def specialInputs(modelAPI :ModelWrapper, out_path :str):
	'''
	Plot the reconstruction of several special inputs.
	'''
	
	special_inputs = [	numpy.zeros((12,14,208)), 
						numpy.ones((12,14,208)), 
						numpy.pad(numpy.ones((1,1,208)), [(5,6), (6,7), (0,0)]),
						numpy.pad(numpy.ones((1,14,1)), [(5,6), (0,0), (103,104)]) ]
	
	titles = [	"Output of Zero Tensor", 
				"Output of Ones Tensor", 
				"Output of Central z-Line",
				"Output of Central y-Line"]
	filenames = ["special_zeros.pdf", "special_ones.pdf", "special_line_z.pdf", "special_line_y.pdf"]
	for i in range( len(special_inputs) ):
		reconstruction = modelAPI.evaluateSingleEvent(special_inputs[i])
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
		matplotlib.pyplot.savefig(out_path + filenames[i], bbox_inches="tight")
		matplotlib.pyplot.close()


def plotExamples(modelAPI :ModelWrapper, out_path :str, datapath :str):
	'''
	Plot few examples of track reconstruction.
	'''

	data_loader = DataLoader(datapath)
	x17 = [event for (_, event) in data_loader.loadX17Data("goodtracks", True)]
	example_indices = [0, 12, 21, 33]
	for i in range(4):
		noisy = x17[example_indices[i]]

		reconstr = modelAPI.evaluateSingleEvent(noisy / numpy.max(noisy))
		classificated = modelAPI.classify(reconstr)
		Plotting.plotEvent(noisy / numpy.max(noisy), reconstr, classificated, True, modelAPI.name)
		matplotlib.pyplot.savefig(out_path + "example_reconstruction" + str(i) + ".pdf", bbox_inches="tight")
		matplotlib.pyplot.close()

		Plotting.animation3D(out_path + "example_reconstruction_3D_" + str(i) + ".gif", modelAPI, noisy, True)
		matplotlib.pyplot.close()


def plotModelArchitecture(modelAPI :ModelWrapper, out_path :str):
	'''
	Plot model architecture and write it in a textfile.
	'''

	keras.utils.plot_model(modelAPI.model, to_file= out_path + "architecture.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)
	with open(out_path + "architecture.txt", "w") as f:
		modelAPI.model.summary(print_fn = lambda x: print(x, file=f))


def getBatchReconstruction(modelAPI :ModelWrapper, index :int, datapath :str, experimental :bool = False):
	'''
	Load noisy events from @index datafile, classificate them and return tuple of values of signal reconstruction and noise filtering metrics.
	'''

	data_loader = DataLoader(datapath)
	signals = data_loader.getBatch(experimental, False, index)
	noises = data_loader.getBatch(experimental, True, index)

	signals_map = signals > 0.0001

	all_signal_tiles = numpy.sum( numpy.where(signals > 0.0001, 1, 0), axis=(1,2,3) )
	all_noise_tiles = numpy.sum( numpy.where(noises > 0.0001, 1, 0), axis=(1,2,3) ) - all_signal_tiles

	reconstructions = modelAPI.evaluateBatch(noises)
	classificated = modelAPI.classify(reconstructions)

	reconstructed_signal_tiles = numpy.sum( numpy.where( signals_map, classificated, 0), axis=(1,2,3) )
	reconstructed_noise_tiles = numpy.sum( classificated, axis=(1,2,3) ) - reconstructed_signal_tiles

	return reconstructed_signal_tiles / all_signal_tiles, reconstructed_noise_tiles / all_noise_tiles


def plotMetrics(modelAPI :ModelWrapper, out_path :str, datapath :str):
	'''
	Plot histograms of reconstruction metrics.
	'''

	title_rec = "Histogram of Event Reconstruction Metric\n by Model " + model_name
	title_noise = "Histogram of Event Noise Filtering Metric\n by Model " + model_name
	xlabel_rec = r"$\rho$ (Ratio of reconstructed track tiles)"
	xlabel_noise = r"$\sigma$ (Ratio of unfiltered noise tiles)"

	num = 10
	low = 11
	reconstructed_noise_metric, reconstructed_signal_metric = numpy.zeros(5000*num), numpy.zeros(5000*num)

	for i in range(0, num):
		print(i+1, "/", num)
		batch_reconstructed_signal_metric, batch_reconstructed_noise_metric = getBatchReconstruction(modelAPI, i+low, datapath, False)
		
		for j in range(5000):
				reconstructed_signal_metric[j+5000*i] = batch_reconstructed_signal_metric[j]
				reconstructed_noise_metric[j+5000*i] = batch_reconstructed_noise_metric[j]

		f = open(out_path + "signal_metric.txt", "w")
		for val in reconstructed_signal_metric:
			f.write(str(val) + "\n")
		f.close()

		f = open(out_path + "noise_metric.txt", "w")
		for val in reconstructed_noise_metric:
			f.write(str(val) + "\n")
		f.close()

		matplotlib.pyplot.hist(reconstructed_signal_metric, bins=70, range=(0,1), log=False)
		matplotlib.pyplot.title(title_rec)
		matplotlib.pyplot.xlabel(xlabel_rec)
		matplotlib.pyplot.ylabel(r"$\#$")
		matplotlib.pyplot.xlim(0, 1)
		matplotlib.pyplot.savefig(out_path + "hist_signal_metric.pdf")
		matplotlib.pyplot.close()
		matplotlib.pyplot.hist(reconstructed_noise_metric, bins=70, range=(0,1), log=False)
		matplotlib.pyplot.title(title_noise)
		matplotlib.pyplot.xlabel(xlabel_noise)
		matplotlib.pyplot.ylabel(r"$\#$")
		matplotlib.pyplot.xlim(0, 1)
		matplotlib.pyplot.savefig(out_path + "hist_noise_metric.pdf")
		matplotlib.pyplot.close()

	fixed_cmap = copy.copy(matplotlib.colormaps["gray"])
	fixed_cmap.set_bad((0,0,0))	#fix pixels with zero occurrence - otherwise problem for LogNorm
	
	matplotlib.pyplot.hist2d(reconstructed_signal_metric, reconstructed_noise_metric, [70,70], [[0,1], [0,1]], cmap=fixed_cmap, norm=matplotlib.colors.LogNorm())
	matplotlib.pyplot.xlabel(r"$\rho$ (Ratio of reconstructed signal)")
	matplotlib.pyplot.ylabel(r"$\sigma$ (Ratio of unfiltered noise)")
	matplotlib.pyplot.suptitle("2D Log Norm Histogram Of Reconstruction and Filtering Metrics\n Model " + model_name)
	matplotlib.pyplot.savefig(out_path + "hist_2d.pdf", bbox_inches='tight')
	matplotlib.pyplot.close()
	
	'''
	for i in range(data_num):
		signal, noise = signals[i], noises[i]
		num_signal_tiles = numpy.sum( numpy.where(signal > 0.0001, 1, 0) )
		num_noise_tiles = numpy.sum( numpy.where(noise > 0.0001, 1, 0) ) - num_signal_tiles
	'''


def findThreshold(modelAPI :ModelWrapper, optimisedFunc, datapath :str, experimental :bool = False):
	'''
	Find classification threshold for @model which maximises @optimisedFunc. Return the optimal threshold and dictionary of metrics for various thresholds.
	'''

	data_loader = DataLoader(datapath)
	signals = data_loader.getBatch(experimental, False, 11)
	noises = data_loader.getBatch(experimental, True, 11)
	data_num = signals.shape[0]

	if experimental:
		for i in range(data_num):	#normalisation
			M = numpy.max(noises[i])
			if M != 0:	noises[i] = noises[i] / M
	
	signals_map = signals>0.0001

	print(">> Estimating the real counts of signal and noise tiles...")
	num_sgn = numpy.zeros(shape=data_num)
	num_noise = numpy.zeros(shape=data_num)
	for i in range(data_num):
		num_sgn[i] = numpy.sum( numpy.where(signals[i] > 0.0001, 1, 0) )
		num_noise[i] = numpy.sum( numpy.where(noises[i] > 0.0001, 1, 0) ) - num_sgn[i]

	print(">> Reconstructing testing data...")
	reconstructions = modelAPI.evaluateBatch(noises)

	print(">> Trying several thresholds:")
	num_steps = 100
	mean_rhos = numpy.zeros(num_steps)
	mean_sigmas = numpy.zeros(num_steps)

	#threshold is usually in [0, 0.2], but for the plots we want to try thresholds in [0, 1]
	thresholds = numpy.concatenate(	[numpy.linspace(start=0, stop=0.2, num=num_steps//2), 
									numpy.linspace(start=0.2, stop=1, num=num_steps//2)])

	for i in range(num_steps):
		threshold = thresholds[i]
		print(">>> Threshold", threshold)
		binary_reconstructions = numpy.where(reconstructions > threshold, 1, 0)
		rhos = []
		sigmas = []

		#count the remaining signal and noise tiles in every event after classification with current threshold
		for j in range(data_num):
			binary_remaining_signal = numpy.sum(binary_reconstructions[j][signals_map[j]])
			binary_remaining_noise = numpy.sum(binary_reconstructions[j]) - binary_remaining_signal
			if num_sgn[j] != 0:	rhos.append( binary_remaining_signal / num_sgn[j] )
			if num_noise[j] != 0:	sigmas.append( binary_remaining_noise / num_noise[j] )
		
		#get the average ratios of remaining signals and noises
		mean_rho, mean_sigma = numpy.mean( numpy.array(rhos) ), numpy.mean( numpy.array(sigmas) )
		mean_rhos[i] = mean_rho
		mean_sigmas[i] = mean_sigma
		print(">>> Reconstructed signal ratio: " + str(mean_rho) + ", Remaining noise: " + str(mean_sigma) + ", Optimised function: " + str(optimisedFunc(mean_rho, mean_sigma)))
	
	optimised_func_vals = optimisedFunc(mean_rhos, mean_sigmas)
	best_threshold = thresholds[numpy.argmax(optimised_func_vals)]

	history = {	"thresholds": thresholds,
				"signal_metrics": mean_rhos,
				"noise_metrics": mean_sigmas,
				"optimised_func_values": optimised_func_vals}

	return best_threshold, history


def plotThresholdPlots(thresholds, signal_metrics, noise_metrics, optimised_func_values, out_path):
	#ROC
	matplotlib.pyplot.plot(noise_metrics, signal_metrics, "blue")
	matplotlib.pyplot.title("Threshold Classifier ROC for Model " + model_name)
	matplotlib.pyplot.xlabel(r"$\overline{\sigma(D)}$ (Ratio of unfiltered noise tiles)")
	matplotlib.pyplot.ylabel(r"$\overline{\rho(D)}$ (Ratio of reconstructed track tiles)")
	matplotlib.pyplot.xlim(-0.05,1)
	matplotlib.pyplot.ylim(0,1.05)
	matplotlib.pyplot.savefig(out_path + "ROC.pdf", bbox_inches='tight')
	matplotlib.pyplot.close()
	
	#Evolution of metrics based on threshold
	matplotlib.pyplot.plot(thresholds, signal_metrics, ".-", label="Reconstructed signal ratio")
	matplotlib.pyplot.plot(thresholds, noise_metrics, ".-", label="Remaining noise ratio")
	matplotlib.pyplot.plot(thresholds, optimised_func_values, ".-", label="Optimised function")
	matplotlib.pyplot.xlabel("Threshold")
	matplotlib.pyplot.ylabel("Value")
	matplotlib.pyplot.legend()
	matplotlib.pyplot.xlim(0,1)
	matplotlib.pyplot.ylim(0,1)
	matplotlib.pyplot.title("Metrics of Reconstruction Quality Based On Threshold by Model " + model_name)
	matplotlib.pyplot.savefig(out_path + "thr_evol.pdf", bbox_inches='tight')
	matplotlib.pyplot.close()


def postprocessModel(out_path, model_name, datapath):
	print("> Loading model...")
	modelAPI = ModelWrapper( keras.models.load_model(out_path + "model"), model_name )

	print("> Plotting model architecture...")
	plotModelArchitecture(modelAPI, out_path)

	threshold = None
	try:
		f = open(out_path + "threshold.txt", "r")
		threshold = float( f.readline() )
		print("> Found saved threshold", threshold)
	except:
		print("> Finding optimal classification threshold...")
		def optimisedFunc(signal_metric, noise_metric):	return signal_metric - noise_metric
		threshold, history = findThreshold(modelAPI, optimisedFunc, datapath, False)
		plotThresholdPlots(history["thresholds"], history["signal_metrics"], history["noise_metrics"], history["optimised_func_values"], out_path)

		print("> Best threshold is", threshold)
		f = open(out_path + "threshold.txt", "w")
		f.write( str(threshold) )
		f.close()

	modelAPI.threshold = threshold

	print("> Plotting reconstruction quality histograms...")
	plotMetrics(modelAPI, out_path, datapath)

	print("> Plotting examples of reconstruction...")
	plotExamples(modelAPI, out_path, datapath)


	print("> Plotting model training history...")
	plotHistory(out_path, modelAPI.name)

	print("> Evaluating special inputs...")
	specialInputs(modelAPI, out_path)


datapath = input("Path to root data directory: ")
if datapath[-1] != "/":	datapath += "/"

name = input("Model names (or 'ALL' models or all 'NEW' models): ")
if name == "ALL":
	counter = 1
	num_all = len(os.listdir("./models/3D/"))
	for model_name in os.listdir("./models/3D/"):
		print("-------------------------------------")
		print("> PROCESSING MODEL " + model_name + " [" + str(counter) + "/" + str(num_all) + "]")
		print("-------------------------------------")
		postprocessModel("./models/3D/" + model_name + "/", model_name, datapath)
		counter += 1
elif name == "NEW":
	for model_name in os.listdir("./models/3D/"):
		print("-------------------------------------")
		if "architecture.txt" in os.listdir("./models/3D/" + model_name):
			print("> SKIPPING MODEL " + model_name)
		else:
			print("> PROCESSING MODEL " + model_name)
			print("-------------------------------------")
			postprocessModel("./models/3D/" + model_name + "/", model_name, datapath)
else:
	for model_name in name.split():
		print("-------------------------------------")
		print("> PROCESSING MODEL " + model_name)
		print("-------------------------------------")
		postprocessModel("./models/3D/" + model_name + "/", model_name, datapath)
