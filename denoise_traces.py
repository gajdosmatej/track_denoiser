import numpy
from tensorflow import keras
import tensorflow
import matplotlib.pyplot
import keras.utils
import os
import copy

class DataLoader:
	def __init__(self, path :str, plane :str, start_file :int, end_file :int):
		self.path = path
		self.plane = plane
		self.start_file = start_file
		self.end_file = end_file
		self.batch_size = 500
		shapes_dict = {"zx": (12,208,1), "zy": (14,208,1), "yx": (14,12,1)}
		self.shape = shapes_dict[plane]

	def dataPairLoad(self):
		while True:
			order = numpy.arange(self.start_file, self.end_file)
			numpy.random.shuffle(order)
			for id in order:
				signal_batch = numpy.load("./data/simulated/" + str(id) + "_signal_" + self.plane + ".npy")
				noise_batch = numpy.load("./data/simulated/" + str(id) + "_noise_" + self.plane + ".npy")
				for i in range(20000):
					yield ( numpy.reshape(noise_batch[i], self.shape), numpy.reshape(signal_batch[i], self.shape))
	
	def getDataset(self):
		return tensorflow.data.Dataset.from_generator(self.dataPairLoad, output_signature = 
						(	tensorflow.TensorSpec(shape=self.shape, dtype=tensorflow.float32), 
							tensorflow.TensorSpec(shape=self.shape, dtype=tensorflow.float32))
						).shuffle(100, reshuffle_each_iteration=True).batch(50).prefetch(2)


class Model:
	def __init__(self, plane=None):
		if plane == "zx":	self.chooseModelZX()
		elif plane == "yx":	self.chooseModelYX()
		elif plane == "zy":	self.chooseModelZY()

	#TODO
	def chooseModelYX(self):
		self.type = "yx"
		self.model = keras.Sequential([keras.layers.Input(shape=(12,10,1)),
									#keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,4), filters=64, activation="relu"),
									#keras.layers.MaxPool2D(pool_size = (2,2)),
									#keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,3), filters=128, activation="relu"),
									#keras.layers.UpSampling2D(size = (2,2)),
									#keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,4), filters=64, activation="relu"),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,4), filters=200, activation="relu"),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(6,6), filters=10, activation="relu"),
									keras.layers.Dense(units=1, activation="sigmoid")])

	#TODO
	def chooseModelZY(self):
		self.type = "zy"
		self.model = keras.Sequential([	keras.layers.Input(shape=(14,208,1)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,8), filters=32, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,8), filters=64, activation="relu"),
									keras.layers.UpSampling2D(size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=1, kernel_size=(6,8), filters=32, activation="relu"),
									keras.layers.Dense(units=1, activation="sigmoid") ])

	def chooseModelZX(self):
		self.type = "zx"
		self.model = keras.Sequential([	keras.layers.Input(shape=(12,208,1)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,5), filters=32, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,4), filters=64, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,3), filters=128, activation="relu"),
									keras.layers.UpSampling2D(size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,4), filters=64, activation="relu"),
									keras.layers.UpSampling2D(size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,5), filters=32, activation="relu"),
									keras.layers.Dense(units=1, activation="sigmoid") ])

	def estimate(self, data_point :numpy.ndarray):
		'''Returns the output of the CNN call on input \'datapoint\'.'''
		data_point = numpy.reshape( self.model( numpy.reshape( data_point, (1, *data_point.shape)) ), data_point.shape)
		return data_point

	@staticmethod
	def load(path :str, model_type :str):
		'''Loads CNN model of type \'model_type\' (\"zy\" / \"zx\" / \"yx\") from tensorflow \'name\' directory.'''
		model = Model()
		model.type = model_type
		model.model = keras.models.load_model(path)
		return model

	def save(self, name :str = None, save_img = True):
		'''Saves this model to tensorflow \'name\' directory and also saves diagram of the model architecture, if \'save_img\' is True. 
		If no \'name\' is specified, the default name is the smallest available natural number.'''
		existing_models = os.listdir("./models")
		i = 1
		while name == None or name in existing_models:
			name = str(i) + "_" + self.type
			i += 1
		self.model.save("./models/" + name)
		if save_img:
			keras.utils.plot_model(self.model, to_file='./models/' + name + ".png", show_shapes=True, show_layer_names=False, show_layer_activations=True)

class Plotting:
	@staticmethod
	def plotRandomData(model :keras.Model, noise_data :numpy.ndarray, signal_data :numpy.ndarray = None, num_plots :int = 5, rng = (15000, 15500), plane :str = "zx"):
		for _ in range(num_plots):
			index = numpy.random.randint(*rng)
			if signal_data is not None:
				Plotting.createPlot(model, noise_data[index], signal_data[index], plane = plane)
			else:
				Plotting.createPlot(model, noise_data[index], plane = plane)
			matplotlib.pyplot.show()
	
	@staticmethod
	def createPlot(model :keras.Model, noise_entry :numpy.ndarray, signal_entry :numpy.ndarray = None, plane :str = "zx"):
		axes = {"zx": ("$z$", "$x$"), "zy": ("$z$", "$y$"), "yx": ("$x$", "$y$")}
		if signal_entry is not None:
			fig, ax = matplotlib.pyplot.subplots(3)
			ax[0].imshow( signal_entry, cmap="gray" )
			ax[0].set_title("Signal only")
			ax[1].imshow( noise_entry, cmap="gray" )
			ax[1].set_title("Signal + noise")
			ax[2].imshow( model.estimate(noise_entry), cmap="gray" )
			ax[2].set_title("Signal reconstruction")
			for i in range(3):
				ax[i].set_xlabel(axes[plane][0])
				ax[i].set_ylabel(axes[plane][1])
		else:
			fig, ax = matplotlib.pyplot.subplots(2)
			ax[0].imshow( noise_entry, cmap="gray")
			ax[0].set_title("Signal + noise")
			ax[1].imshow( model.estimate(noise_entry), cmap="gray" )
			ax[1].set_title("Signal reconstruction")
			for i in range(2):
				ax[i].set_xlabel(axes[plane][0])
				ax[i].set_ylabel(axes[plane][1])
		

	@staticmethod
	def plotRandomAllProjections(models: list[keras.Model], signal_data :list[numpy.ndarray], noise_data :list[numpy.ndarray], num_plots = 5):
		for _ in range(num_plots):
			index = numpy.random.randint(15000, 15500)
			fig, ax = matplotlib.pyplot.subplots(3, 3)
			for i in range(3):
				ax[0,i].imshow( signal_data[i][index], cmap="gray" )
				ax[0,i].set_title("Signal only")
				ax[1,i].imshow( noise_data[i][index], cmap="gray" )
				ax[1,i].set_title("Signal + noise")
				ax[2,i].imshow( models[i].estimate(noise_data[i][index]), cmap="gray" )
				ax[2,i].set_title("Signal reconstruction")
			matplotlib.pyplot.get_current_fig_manager().window.showMaximized()
			matplotlib.pyplot.show()


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
