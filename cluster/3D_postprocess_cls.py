from denoise_traces_cls import DataLoader, Plotting
import numpy
from tensorflow import keras
import keras.utils
import json
import os
import matplotlib.pyplot
import X17_data_load_cls
import tensorflow
import copy


DATAPATH = "/data/simulated/3D/"

def getBatchReconstruction(model, threshold, index):
	signals = numpy.load(DATAPATH + str(index) + "_signal_3d.npy")
	noises = numpy.load(DATAPATH + str(index) + "_noise_3d.npy")

	signals_map = signals > 0.0001

	all_signal_tiles = numpy.sum( numpy.where(signals > 0.0001, 1, 0), axis=(1,2,3) )
	all_noise_tiles = numpy.sum( numpy.where(noises > 0.0001, 1, 0), axis=(1,2,3) ) - all_signal_tiles

	reconstructions = numpy.reshape( model.predict( numpy.reshape(noises, (*noises.shape, 1)) ), noises.shape)
	classificated = numpy.where(reconstructions > threshold, 1, 0)

	reconstructed_signal_tiles = numpy.sum( numpy.where( signals_map, classificated, 0), axis=(1,2,3) )
	reconstructed_noise_tiles = numpy.sum( classificated, axis=(1,2,3) ) - reconstructed_signal_tiles

	return reconstructed_signal_tiles / all_signal_tiles, reconstructed_noise_tiles / all_noise_tiles



def getBatchReconstructionOld(model, index):
	signals = numpy.load(DATAPATH + str(index) + "_signal_3d.npy")
	noises = numpy.load(DATAPATH + str(index) + "_noise_3d.npy")
	reconstructions = numpy.reshape( model.predict( numpy.reshape(noises, (*noises.shape, 1)) ), noises.shape)

	signals_map = signals > 0.0001

	all_signals = numpy.sum(signals, axis=(1,2,3) )
	signal_reconstruction_metrics = numpy.sum(numpy.abs( numpy.where(signals_map, reconstructions, 0) - signals), axis=(1,2,3))
	remaining_noise_metrics = numpy.sum( numpy.where(signals_map, 0, reconstructions), axis=(1,2,3) )
	all_noises = numpy.sum(noises, axis=(1,2,3)) - all_signals

	return signal_reconstruction_metrics / all_signals, remaining_noise_metrics / all_noises
	

def plotMetrics(model, path, model_name, threshold):

	titles_rec = {	"new": "Histogram of Ratio of Reconstructed Signal Tiles\n by Model " + model_name,
					"old": "Histogram of Mean Absolute Error of Reconstructed Signal Tiles\n by Model " + model_name}
	titles_noise = {	"new": "Histogram of Ratio of Unfiltered Noise Tiles\n by Model " + model_name,
		 				"old": "Histogram of Remaining Noise Sum in Reconstructions\n by Model " + model_name}
	xlabels_rec = {	"new": "Ratio of Reconstructed Signal Tiles",
					"old": "MAE of Reconstruction from Signal"}
	xlabels_noise = {	"new": "Ratio of Unfiltered Noise Tiles",
						"old": "Sum of Remaining Noise"}
	
	num = 10
	low = 11
	reconstructed_noise_metric, reconstructed_signal_metric = numpy.zeros(5000*num), numpy.zeros(5000*num)

	for metric_type in ["old", "new"]:
		print(">>", metric_type)

		for i in range(0, num):
			print(i+1, "/", num)

			if metric_type == "new":
				batch_reconstructed_signal_metric, batch_reconstructed_noise_metric = getBatchReconstruction(model, threshold, i+low)
			else:
				batch_reconstructed_signal_metric, batch_reconstructed_noise_metric = getBatchReconstructionOld(model, i+low)

			for j in range(5000):
				reconstructed_signal_metric[j+5000*i] = batch_reconstructed_signal_metric[j]
				reconstructed_noise_metric[j+5000*i] = batch_reconstructed_noise_metric[j]

		f = open(path + "signal_metric_" + metric_type + ".txt", "w")
		for val in reconstructed_signal_metric:
			f.write(str(val) + "\n")
		f.close()

		f = open(path + "noise_metric_" + metric_type + ".txt", "w")
		for val in reconstructed_noise_metric:
			f.write(str(val) + "\n")
		f.close()

		matplotlib.pyplot.hist(reconstructed_signal_metric, bins=70, range=(0,1), log=False)
		matplotlib.pyplot.title(titles_rec[metric_type])
		matplotlib.pyplot.xlabel(xlabels_rec[metric_type])
		matplotlib.pyplot.ylabel(r"$\#$")
		matplotlib.pyplot.xlim(0, 1)
		matplotlib.pyplot.savefig(path + "hist_signal_metric_ " + metric_type + ".pdf")
		matplotlib.pyplot.clf()
		matplotlib.pyplot.hist(reconstructed_noise_metric, bins=70, range=(0,1), log=False)
		matplotlib.pyplot.title(titles_noise[metric_type])
		matplotlib.pyplot.xlabel(xlabels_noise[metric_type])
		matplotlib.pyplot.ylabel(r"$\#$")
		matplotlib.pyplot.xlim(0, 1)
		matplotlib.pyplot.savefig(path + "hist_noise_metric_" + metric_type + ".pdf")
		matplotlib.pyplot.clf()

	fixed_cmap = copy.copy(matplotlib.cm.get_cmap('magma'))
	fixed_cmap.set_bad((0,0,0))	#fix pixels with zero occurrence - otherwise problem for LogNorm
	
	matplotlib.pyplot.hist2d(reconstructed_signal_metric, reconstructed_noise_metric, [70,70], [[0,1], [0,1]], cmap=fixed_cmap, norm=matplotlib.colors.LogNorm())
	matplotlib.pyplot.xlabel("Ratio of reconstructed signal")
	matplotlib.pyplot.ylabel("Ratio of unfiltered noise")
	matplotlib.pyplot.suptitle("2D Log Norm Histogram Of Reconstruction Quality Metrics\n Model " + model_name)
	matplotlib.pyplot.savefig(path + "hist_2d.pdf", bbox_inches='tight')
	matplotlib.pyplot.clf()
	
	'''
	for i in range(data_num):
		signal, noise = signals[i], noises[i]
		num_signal_tiles = numpy.sum( numpy.where(signal > 0.0001, 1, 0) )
		num_noise_tiles = numpy.sum( numpy.where(noise > 0.0001, 1, 0) ) - num_signal_tiles
	'''

def findThreshold(model, path, model_name):
	signals = numpy.load(DATAPATH + "/10_signal_3d.npy")
	noises = numpy.load(DATAPATH + "/10_noise_3d.npy")
	data_num = signals.shape[0]
	signals_map = signals>0.0001

	print(">> Estimating the real counts of signal and noise tiles...")
	num_sgn = numpy.zeros(shape=data_num)
	num_noise = numpy.zeros(shape=data_num)
	for i in range(data_num):
		num_sgn[i] = numpy.sum( numpy.where(signals[i] > 0.0001, 1, 0) )
		num_noise[i] = numpy.sum( numpy.where(noises[i] > 0.0001, 1, 0) ) - num_sgn[i]

	print(">> Reconstructing testing data...")
	reconstructions = numpy.reshape( model.predict( numpy.reshape(noises, (*noises.shape, 1)) ), noises.shape)

	print(">> Trying several thresholds:")
	num_steps = 100
	mean_rhos = numpy.zeros(num_steps)
	mean_sigmas = numpy.zeros(num_steps)

	#threshold is usually in [0, 0.2], but for the plots we want to try thresholds in [0, 1]
	thresholds = numpy.concatenate(	[numpy.linspace(start=0, stop=0.2, num=num_steps//2), 
									numpy.linspace(start=0.2, stop=1, num=num_steps//2)])

	#the (a bit arbitrary) function that is being maximised by the threshold
	def optimised_func(rho, sigma):	return rho - sigma

	for i in range(num_steps):
		threshold = thresholds[i]
		print(">>> Threshold", threshold)
		binary_reconstructions = numpy.where(reconstructions > threshold, 1, 0)
		rhos = numpy.zeros(shape=data_num)
		sigmas = numpy.zeros(shape=data_num)

		#count the remaining signal and noise tiles in every event after classification with current threshold
		for j in range(data_num):
			binary_remaining_signal = numpy.sum(binary_reconstructions[j][signals_map[j]])
			binary_remaining_noise = numpy.sum(binary_reconstructions[j]) - binary_remaining_signal
			rhos[j] = binary_remaining_signal / num_sgn[j]
			sigmas[j] = binary_remaining_noise / num_noise[j]
		
		#get the average ratios of remaining signals and noises
		mean_rho, mean_sigma = numpy.mean(rhos), numpy.mean(sigmas)
		mean_rhos[i] = mean_rho
		mean_sigmas[i] = mean_sigma
		print(">>> Reconstructed signal ratio: " + str(mean_rho) + ", Remaining noise: " + str(mean_sigma) + ", Optimised function: " + str(optimised_func(mean_rho, mean_sigma)))
	
	optimised_func_vals = optimised_func(mean_rhos, mean_sigmas)
	best_threshold = thresholds[numpy.argmax(optimised_func_vals)]
	print("> Best threshold is", best_threshold)
	f = open(path + "threshold.txt", "w")
	f.write( str(best_threshold) )
	f.close()

	#ROC
	matplotlib.pyplot.plot(mean_sigmas, mean_rhos, ".-b")
	matplotlib.pyplot.title("Threshold Classifier ROC for Model " + model_name)
	matplotlib.pyplot.xlabel("Fraction of Unfiltered Background Tiles")
	matplotlib.pyplot.ylabel("Fraction of Reconstructed Signal Tiles")
	matplotlib.pyplot.xlim(0,1)
	matplotlib.pyplot.ylim(0,1)
	matplotlib.pyplot.savefig(path + "ROC.pdf", bbox_inches='tight')
	matplotlib.pyplot.clf()
	
	#Evolution of metrics based on threshold
	matplotlib.pyplot.plot(thresholds, mean_rhos, ".-", label="Reconstructed signal ratio")
	matplotlib.pyplot.plot(thresholds, mean_sigmas, ".-", label="Remaining noise ratio")
	matplotlib.pyplot.plot(thresholds, optimised_func_vals, ".-", label="Optimised function")
	matplotlib.pyplot.xlabel("Threshold")
	matplotlib.pyplot.ylabel("Value")
	matplotlib.pyplot.legend()
	matplotlib.pyplot.xlim(0,1)
	matplotlib.pyplot.ylim(0,1)
	matplotlib.pyplot.title("Metrics of Reconstruction Quality Based On Threshold by Model " + model_name)
	matplotlib.pyplot.savefig(path + "thr_evol.pdf", bbox_inches='tight')
	matplotlib.pyplot.clf()
	return best_threshold


def postprocessModel(path, model_name):
	print("> Loading model...")
	model = keras.models.load_model(path + "model")

	print("> Plotting model architecture...")
	keras.utils.plot_model(model, to_file= path + "architecture.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)
	with open(path + "architecture.txt", "w") as f:
		model.summary(print_fn = lambda x: print(x, file=f))

	threshold = None
	try:
		f = open(path + "threshold.txt", "r")
		threshold = float( f.readline() )
		print("> Found saved threshold", threshold)
	except:
		print("> Finding optimal classification threshold...")
		threshold = findThreshold(model, path, model_name)

	print("> Plotting reconstruction quality histograms...")
	plotMetrics(model, path, model_name, threshold)

	print("> Plotting examples of reconstruction...")

	x17 = [event for (_, event) in X17_data_load_cls.loadX17Data("goodtracks")]
	example_indices = [0, 12, 21, 33]
	for i in range(4):
		noisy = x17[example_indices[i]]
		noisy = noisy / numpy.max(noisy)
		noisy_input = numpy.reshape( noisy, (1,12,14,208,1))

		reconstr = numpy.reshape(model(noisy_input)[0], (12,14,208))
		classificated = numpy.where(reconstr > threshold, 1, 0)
		Plotting.plotEvent(noisy, reconstr, classificated, True, model_name)

		matplotlib.pyplot.savefig(path + "example_reconstruction" + str(i) + ".pdf", bbox_inches="tight")
		matplotlib.pyplot.clf()


	print("> Plotting model training history...")
	f = open(path + "history.json", "r")
	history = json.load(f)
	f.close()
	n = len(history["loss"])
	matplotlib.pyplot.clf()
	#matplotlib.pyplot.yscale("log")
	matplotlib.pyplot.plot([i for i in range(0,n)], history["loss"][0:], label="Loss fuction")
	matplotlib.pyplot.plot([i for i in range(0,n)], history["val_loss"][0:], label="Validation loss")
	matplotlib.pyplot.legend()
	matplotlib.pyplot.xlabel("epoch")
	matplotlib.pyplot.ylabel("binary cross entropy (log scale)")
	matplotlib.pyplot.yscale("log")
	matplotlib.pyplot.suptitle("Training of Model " + model_name)
	matplotlib.pyplot.savefig(path + "history.pdf", bbox_inches='tight')
	matplotlib.pyplot.clf()

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
		postprocessModel("./models/3D/" + model_name + "/", model_name)
		counter += 1
elif name == "NEW":
	counter = 1
	for model_name in os.listdir("./models/3D/"):
		print("-------------------------------------")
		if "architecture.txt" in os.listdir("./models/3D/" + model_name):
			print("> SKIPPING MODEL " + model_name)
		else:
			print("> PROCESSING MODEL " + model_name)
			print("-------------------------------------")
			postprocessModel("./models/3D/" + model_name + "/", model_name)
		counter += 1
else:
	for model_name in name.split():
		print("-------------------------------------")
		print("> PROCESSING MODEL " + model_name)
		print("-------------------------------------")
		postprocessModel("./models/3D/" + model_name + "/", model_name)
