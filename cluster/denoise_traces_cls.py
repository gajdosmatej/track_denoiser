import numpy
from tensorflow import keras
import tensorflow
import matplotlib.pyplot
import keras.utils
import X17_data_load_cls
import matplotlib.animation

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
				signal_batch = numpy.where(signal_batch > 0.001, 1, 0)	#CLASSIFICATION
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
			x17_data = numpy.array( [event for (_, event) in X17_data_load_cls.loadX17Data("goodtracks")] )
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
		if are_data_experimental:	title += "Experimental "
		elif are_data_experimental is False:	title += "Generated "
		title += "data by Model " + model_name
		fig.suptitle(title)

	@staticmethod
	def getPlot3D(model :keras.Model, noise_event :numpy.ndarray, are_data_experimental :bool = None, model_name :str = "", threshold :float = None, rotation=(0,0,0)):
		fig = matplotlib.pyplot.figure(figsize=matplotlib.pyplot.figaspect(0.5))
		
		ax1 = fig.add_subplot(1, 2, 1, projection='3d')
		xs, ys, zs = noise_event.nonzero()
		vals = numpy.array([noise_event[xs[i],ys[i],zs[i]] for i in range(len(xs))])
		sctr1 = ax1.scatter(xs, ys, zs, c=vals, cmap="plasma")
		ax1.set_xlim(0, 11)
		ax1.set_xlabel("$x$")
		ax1.set_ylim(0, 13)
		ax1.set_ylabel("$y$")
		ax1.set_zlim(0, 200)
		ax1.set_zlabel("$z$")
		title = title = "Noisy "
		if are_data_experimental:	title += "Experimental "
		elif are_data_experimental is False:	title += "Generated "
		title += "Data"
		ax1.set_title(title)
		ax1.view_init(*rotation)	#rotate the scatter plot, useful for animation

		reconstr_event = numpy.reshape( model( numpy.reshape(noise_event, (1,12,14,208,1)) )[0], (12,14,208) )
		classificated_event = numpy.where(reconstr_event > threshold, 1, 0)
		ax2 = fig.add_subplot(1, 2, 2, projection='3d')
		xs, ys, zs = classificated_event.nonzero()
		vals = numpy.array([classificated_event[xs[i],ys[i],zs[i]] for i in range(len(xs))])
		sctr2 = ax2.scatter(xs, ys, zs, c=vals)
		ax2.set_xlim(0, 11)
		ax2.set_xlabel("$x$")
		ax2.set_ylim(0, 13)
		ax2.set_ylabel("$y$")
		ax2.set_zlim(0, 200)
		ax2.set_zlabel("$z$")
		ax2.view_init(*rotation)
		title = "Reconstruction and Threshold Classification\n"
		title += "by Model " + model_name
		ax2.set_title(title)

		#fig.subplots_adjust(right=0.8)
		#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		cb = fig.colorbar(sctr1, ax=[ax1, ax2], orientation="horizontal")
		cb.set_label("$E$")

		return fig, ax1, ax2


	def animation3D(path :str, model :keras.Model, noise_event :numpy.ndarray, are_data_experimental :bool = None, model_name :str = "", threshold :float = None):
		fig, ax1, ax2 = Plotting.getPlot3D(model, noise_event, are_data_experimental, model_name, threshold)

		def run(i):	
			ax1.view_init(0,i,0)
			ax2.view_init(0,i,0)

		anim = matplotlib.animation.FuncAnimation(fig, func=run, frames=360, interval=20, blit=False)
		anim.save(path, fps=30, dpi=200, writer="pillow")


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
