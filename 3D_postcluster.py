from denoise_traces import Model, Plotting, QualityEstimator
import numpy
from tensorflow import keras
import keras.utils
import json
import os
import matplotlib.pyplot
import preprocess_data
import tensorflow


def findThreshold(model, path):
	signals = numpy.load("data/simulated/3D/0_signal_3d.npy")
	noises = numpy.load("data/simulated/3D/0_noise_3d.npy")
	data_num = signals.shape[0]
	signals_map = signals>0.0001

	print(">> Estimating the real counts of signal and noise tiles...")
	num_sgn = numpy.zeros(shape=data_num)
	num_noise = numpy.zeros(shape=data_num)
	for i in range(data_num):
		num_sgn[i] = numpy.sum( numpy.where(signals[i] > 0.0001, 1, 0) )
		num_noise[i] = numpy.sum( numpy.where(noises[i] > 0.0001, 1, 0) ) - num_sgn[i]

	print(">> Reconstructing testing data...")
	reconstructions = model.predict( numpy.reshape(noises, (*noises.shape, 1)) )

	print(">> Trying several thresholds:")
	'''num_steps = 5
	leading_coeff = 0.5
	best_effunc = -numpy.Infinity
	best_threshold = 0
	for exponent in range(num_steps):
		best_new_leading = leading_coeff
		best_curr_effunc = best_effunc
		for i in range(-5, 6):
			threshold = numpy.round( leading_coeff + i * 10**(-exponent-1), 2+exponent)
			print(">> Thr = " + str(threshold))
			binary_reconstructions = numpy.where(reconstructions > threshold, 1, 0)
			rhos = numpy.zeros(shape=data_num)
			sigmas = numpy.zeros(shape=data_num)
			for j in range(data_num):
				binary_remaining_signal = numpy.sum(binary_reconstructions[j][signals_map[j]])
				binary_remaining_noise = numpy.sum(binary_reconstructions[j]) - binary_remaining_signal
				rhos[j] = binary_remaining_signal / num_sgn[j]
				sigmas[j] = binary_remaining_noise / num_noise[j]
			mean_rho, mean_sigma = numpy.mean(rhos), numpy.mean(sigmas)
			effunc = mean_rho - mean_sigma
			print(">>> Reconstructed signal ratio: " + str(mean_rho) + ", Remaining noise: " + str(mean_sigma) + ", effuc: " + str(effunc))
			if effunc > best_curr_effunc:
				best_new_leading = threshold
				best_curr_effunc = effunc
		if best_curr_effunc > best_effunc:
			best_effunc = best_curr_effunc
			best_threshold = best_new_leading
		leading_coeff = best_new_leading'''
	
	num_steps = 100
	optimised_func_vals = numpy.zeros(num_steps)
	mean_rhos = []
	mean_sigmas = []

	for i in range(num_steps):
		threshold = i/(2*num_steps)
		print(">>> Threshold", threshold)
		binary_reconstructions = numpy.where(reconstructions > threshold, 1, 0)
		rhos = numpy.zeros(shape=data_num)
		sigmas = numpy.zeros(shape=data_num)
		for j in range(data_num):
			binary_remaining_signal = numpy.sum(binary_reconstructions[j][signals_map[j]])
			binary_remaining_noise = numpy.sum(binary_reconstructions[j]) - binary_remaining_signal
			rhos[j] = binary_remaining_signal / num_sgn[j]
			sigmas[j] = binary_remaining_noise / num_noise[j]
		mean_rho, mean_sigma = numpy.mean(rhos), numpy.mean(sigmas)
		mean_rhos.append(mean_rho)
		mean_sigmas.append(mean_sigma)
		optimised_func_vals[i] = mean_rho - mean_sigma
		print(">>> Reconstructed signal ratio: " + str(mean_rho) + ", Remaining noise: " + str(mean_sigma) + ", Optimised function: " + str(mean_rho - mean_sigma))

	best_threshold = numpy.argmax(optimised_func_vals) / (2*num_steps)
	print("> Best threshold is", best_threshold)
	f = open(path + "threshold.txt", "w")
	f.write( str(best_threshold) )
	f.close()

	matplotlib.pyplot.plot(mean_sigmas, mean_rhos, ".-b")
	matplotlib.pyplot.title("Threshold Classifier ROC")
	matplotlib.pyplot.xlabel("Fraction of Unfiltered Background Tiles")
	matplotlib.pyplot.ylabel("Fraction of Reconstructed Signal Tiles")
	matplotlib.pyplot.xlim(0,1)
	matplotlib.pyplot.ylim(0,1)
	matplotlib.pyplot.savefig(path + "ROC.pdf", bbox_inches='tight')
	matplotlib.pyplot.clf()
	return best_threshold


def postprocessModel(path):
	print("> Loading model...")
	model = keras.models.load_model(path + "model")

	print("> Plotting model architecture...")
	keras.utils.plot_model(model, to_file= path + "architecture.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)
	with open(path + "architecture.txt", "w") as f:
		model.summary(print_fn = lambda x: print(x, file=f))


	print("> Finding optimal classification threshold...")
	threshold = findThreshold(model, path)

	print("> Plotting examples of reconstruction...")

	x17 = [event for (_, event) in preprocess_data.loadX17Data("goodtracks")]
	example_indices = [0, 12, 21, 33]
	for i in range(4):
		noisy = x17[example_indices[i]]
		noisy = noisy / numpy.max(noisy)
		noisy_input = numpy.reshape( noisy, (1,12,14,208,1))

		reconstr = numpy.reshape(model(noisy_input)[0], (12,14,208))
		classificated = numpy.where(reconstr > threshold, 1, 0)
		fig, ax = matplotlib.pyplot.subplots(3, 3)

		#ax[0][1].imshow(numpy.sum(reconstr, axis=0), cmap="gray", norm=matplotlib.colors.LogNorm() )
		ax[0][0].imshow(numpy.sum(noisy, axis=0), cmap="gray", vmin=0, vmax=1)
		#ax[0][0].set_title("Noisy, ZY Projection")
		ax[0][0].set_title("Noisy")
		ax[0][0].set_xlabel("z")
		ax[0][0].set_ylabel("y")
		
		ax[0][1].imshow(numpy.sum(reconstr, axis=0), cmap="gray", vmin=0, vmax=1)
		#ax[0][1].set_title("Raw Reconstruction, ZY Projection")
		ax[0][1].set_title("Raw Reconstruction")
		ax[0][1].set_xlabel("z")
		ax[0][1].set_ylabel("y")

		ax[0][2].imshow(numpy.sum(classificated, axis=0), cmap="gray", vmin=0, vmax=1)
		#ax[0][2].set_title("Reconstruction after Treshold, ZY Projection")
		ax[0][2].set_title("Reconstruction + Thr")
		ax[0][2].set_xlabel("z")
		ax[0][2].set_ylabel("y")

		ax[1][0].imshow(numpy.sum(noisy, axis=1), cmap="gray", vmin=0, vmax=1)
		#ax[1][0].set_title("Noisy, ZX Projection")
		ax[1][0].set_xlabel("z")
		ax[1][0].set_ylabel("x")

		ax[1][1].imshow(numpy.sum(reconstr, axis=1), cmap="gray", vmin=0, vmax=1)
		#ax[1][1].set_title("Reconstruction, ZX Projection")
		ax[1][1].set_xlabel("z")
		ax[1][1].set_ylabel("x")

		ax[1][2].imshow(numpy.sum(classificated, axis=1), cmap="gray", vmin=0, vmax=1)
		#ax[1][2].set_title("Reconstruction after Treshold, ZX Projection")
		ax[1][2].set_xlabel("z")
		ax[1][2].set_ylabel("x")

		ax[2][0].imshow(numpy.sum(noisy, axis=2), cmap="gray", vmin=0, vmax=1)
		#ax[2][0].set_title("Noisy, YX Projection")
		ax[2][0].set_xlabel("y")
		ax[2][0].set_ylabel("x")

		ax[2][1].imshow(numpy.sum(reconstr, axis=2), cmap="gray", vmin=0, vmax=1)
		#ax[2][1].set_title("Reconstruction, YX Projection")
		ax[2][1].set_xlabel("y")
		ax[2][1].set_ylabel("x")

		ax[2][2].imshow(numpy.sum(classificated, axis=2), cmap="gray", vmin=0, vmax=1)
		#ax[2][2].set_title("Reconstruction after Treshold, YX Projection")
		ax[2][2].set_xlabel("y")
		ax[2][2].set_ylabel("x")

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
elif name == "NEW":
	counter = 1
	for model_name in os.listdir("./models/3D/"):
		print("-------------------------------------")
		if "architecture.txt" in os.listdir("./models/3D/" + model_name):
			print("> SKIPPING MODEL " + model_name)
		else:
			print("> PROCESSING MODEL " + model_name)
		print("-------------------------------------")
		postprocessModel("./models/3D/" + model_name + "/")
		counter += 1
else:
	postprocessModel("./models/3D/" + name + "/")
