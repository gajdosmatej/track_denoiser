import numpy
from tensorflow import keras
import tensorflow
import matplotlib.pyplot
import keras.utils
import preprocess_data

class DataLoader:
	'''
	Class for loading generated data in tensorflow datasets
	'''
	def __init__(self, path :str):
		self.path = path + ('/' if path[-1] != '/' else '')


	def dataPairLoad(self, low_id :int, high_id :int):
		'''
		Yield a pair of noisy and clean event tensors from numbered data files in between @low_id and @high_id
		'''
		while True:
			order = numpy.arange(low_id, high_id)
			numpy.random.shuffle(order)
			for id in order:
				signal_batch = numpy.load(self.path + str(id) + "_signal_3d.npy")
				noise_batch = numpy.load(self.path + str(id) + "_noise_3d.npy")
				for i in range(5000):
					yield ( numpy.reshape(noise_batch[i], (12,14,208,1)), numpy.reshape(signal_batch[i], (12,14,208,1)))


	def getDataset(self, low_id :int, high_id :int):
		'''
		Pack the method _dataPairLoad_(@low_id, @high_id) into tensorflow dataset.
		'''
		return tensorflow.data.Dataset.from_generator(lambda: self.dataPairLoad(low_id, high_id), output_signature =
					(	tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16),
						tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16))
					).batch(100).prefetch(20)

	def getNoisyBatch(self, experimental :bool = True, file_id :int = 0):
		'''
		Return a list of noisy data. If @experimental is True, return real data from X17 experiment, otherwise generated data are used from file specified by $file_id.
		'''
		if experimental:
			x17_data = numpy.array( [event for (_, event) in preprocess_data.loadX17Data("goodtracks")] )
			return x17_data / numpy.max(x17_data)	#normalisation to [0,1] interval
		else:
			return numpy.load(self.path + str(file_id) + "_noise_3d.npy")

class Plotting:
	@staticmethod
	def plotEvent(noisy, reconstruction, classificated = None, are_data_experimental = None, model_name = '', axes=[0,1,2]):
		if classificated is None:
			fig, ax = matplotlib.pyplot.subplots(len(axes), 2)
		else:
			fig, ax = matplotlib.pyplot.subplots(len(axes), 3)


		x_labels = ['y', 'x', 'x']
		y_labels = ['z', 'z', 'y']
		for i in range(len(axes)):
			axis = axes[i]
			ax[i][0].set_title("Noisy")
			ax[i][0].imshow(numpy.sum(noisy, axis=axis), cmap="gray", vmin=0, vmax=1 )
			ax[i][0].set_xlabel(x_labels[axis])
			ax[i][0].set_ylabel(y_labels[axis])
			ax[i][1].set_title("Raw Reconstruction")
			ax[i][1].imshow(numpy.sum(reconstruction, axis=axis), cmap="gray", vmin=0, vmax=1 )
			ax[i][1].set_xlabel(x_labels[axis])
			ax[i][1].set_ylabel(y_labels[axis])

		if classificated is not None:
			for i in range(len(axes)):
				axis = axes[i]
				ax[i][2].set_title("After Threshold")
				ax[i][2].imshow(numpy.sum(classificated, axis=axis), cmap="gray", vmin=0, vmax=1 )
				ax[i][2].set_xlabel(x_labels[axis])
				ax[i][2].set_ylabel(y_labels[axis])
			
		title = "Reconstruction of "
		if are_data_experimental:	title += "experimental "
		elif are_data_experimental is False:	title += "generated "
		title += "data by model " + model_name
		fig.suptitle(title)

	@staticmethod
	def plotRandomData(model :keras.Model, noise_data :numpy.ndarray, are_data_experimental :bool = None, model_name :str = "", threshold :float = None):
		'''
		Plot @model's reconstruction of random events from @noise_data. If @threshold is specified, plot also the final classification after applying @threshold to reconstruciton.
		'''
		while True:
			index = numpy.random.randint(0, len(noise_data))
			noisy = numpy.reshape( noise_data[index], (1,12,14,208,1))
			reconstr = numpy.reshape(model(noisy)[0], (12,14,208))
			noisy = numpy.reshape(noisy[0], (12,14,208))

			if threshold != None:
				classif = numpy.where(reconstr > threshold, 1, 0)
				Plotting.plotEvent(noisy, reconstr, classif, are_data_experimental, model_name, axes = [0,1,2])
			else:
				Plotting.plotEvent(noisy, reconstr, None, are_data_experimental, model_name, axes = [0,1,2])
			matplotlib.pyplot.show()
			if input("Enter 'q' to stop plotting (or anything else for another plot):") == "q":	break
			

#TODO ... Remake to 3D, compare with new accuracy metrics 
class QualityEstimator:
	@staticmethod
	def signalsMetric(signal :numpy.ndarray, reconstructed :numpy.ndarray):
		'''Calculates how well the signal was reconstructed for each pair of data in \'signal\', \'reconstructed\' arrays.'''
		shape = signal.shape
		metric_data = []

		def metric(sgn, rec):	return numpy.sum(numpy.abs(sgn - rec)) / numpy.sum(sgn)

		for n in range(shape[0]):
			if n % 100 == 0:	print(n, "/", shape[0])
			rec_matrix, sgn_matrix = reconstructed[n], signal[n]
			mask = sgn_matrix > 0.1
			mtr = metric(sgn_matrix[mask], rec_matrix[mask])
			metric_data.append(mtr)
			#print(metric(sgn_matrix[mask], rec_matrix[mask]))
			#Testing.test_QualityEstimator_reconstructedSignals(rec_matrix, sgn_matrix)
		return metric_data

	@staticmethod
	def reconstructionQuality(signal :numpy.ndarray, reconstructed :numpy.ndarray):
		'''Calculates how well the signal was reconstructed for each pair of data in \'signal\', \'reconstructed\' arrays.'''
		shape = signal.shape
		data_signal_tiles = numpy.array([])
		data_wrong_reconstruction_tiles = numpy.array([])
		data_residue_noise_intensity = numpy.array([])
		interesting_indices = []

		for k in range(shape[0]):
			rec_matrix, sgn_matrix = numpy.reshape(reconstructed[k], (shape[1], shape[2])), numpy.reshape(signal[k], (shape[1], shape[2]) )

			rec_matrix_dscr, sgn_matrix_dscr = rec_matrix, sgn_matrix
			#rec_matrix_dscr, sgn_matrix_dscr = numpy.where(rec_matrix > threshold, 1, 0), numpy.where(sgn_matrix > threshold, 1, 0)
			mask = sgn_matrix_dscr>0.1

			rows, cols = mask.nonzero()
			n = rows.size
			for i in range(n):	#add signal neighbours to mask
				row, col = rows[i], cols[i]
				if row > 0 and col > 0:	mask[row-1, col-1] = True
				if row > 0:	mask[row-1, col] = True
				if row > 0 and col < shape[2]-1:	mask[row-1, col+1] = True
				if col < shape[2]-1:	mask[row, col+1] = True
				if col < shape[2]-1 and row < shape[1]-1:	mask[row+1, col+1] = True
				if row < shape[1]-1:	mask[row+1, col] = True
				if row < shape[1]-1 and col > 0:	mask[row+1, col-1] = True
				if col > 0:	mask[row, col-1] = True

			num_sgn = numpy.sum(sgn_matrix_dscr)
			num_wrong_reconstr = numpy.sum( numpy.abs(rec_matrix_dscr[mask] - sgn_matrix_dscr[mask]) )
			data_signal_tiles = numpy.append(data_signal_tiles, num_sgn)
			if num_wrong_reconstr / num_sgn > 0.5:
				interesting_indices.append(k)
			data_wrong_reconstruction_tiles = numpy.append(data_wrong_reconstruction_tiles, num_wrong_reconstr)
			data_residue_noise_intensity = numpy.append(data_residue_noise_intensity, numpy.sum(rec_matrix[mask==False]) )

			if k % 1000 == 999:
				print(k+1, "/", shape[0])
			
			#visualisation for testing only
			'''if num_wrong_reconstr / num_sgn >= 1:
				fig, ax = matplotlib.pyplot.subplots(2,2)
				ax[0,0].imshow(sgn_matrix, cmap='gray')
				ax[0,0].set_title("signal")
				ax[1,0].imshow(rec_matrix, cmap='gray')
				ax[1,0].set_title("reconstructed")
				ax[1,1].imshow(rec_matrix_dscr, cmap='gray')
				ax[1,1].set_title("reconstructed discr")
				ax[0,1].imshow(sgn_matrix_dscr, cmap='gray')
				ax[0,1].set_title("sign discr")
				matplotlib.pyplot.show()
			'''
		return {"signal": data_signal_tiles, "false_signal": data_wrong_reconstruction_tiles, "noise": data_residue_noise_intensity}, interesting_indices


	def filteredNoise(signal :numpy.ndarray, reconstructed :numpy.ndarray):
		'''Calculates how well the noise was filtered for each pair of data in \'signal\', \'reconstructed\' arrays.'''
		shape = signal.shape
		metric_data = []

		def metric(sgn, rec):	return numpy.sum(numpy.abs(sgn - rec))

		for n in range(shape[0]):
			if n % 100 == 0:	print(n, "/", shape[0])
			rec_matrix, sgn_matrix = reconstructed[n], signal[n]
			mask = sgn_matrix < 0.1
			mtr = metric(sgn_matrix[mask], rec_matrix[mask])
			metric_data.append(mtr)
			#print(metric(sgn_matrix[mask], rec_matrix[mask]))
			#Testing.test_QualityEstimator_reconstructedSignals(rec_matrix, sgn_matrix)
		return metric_data
