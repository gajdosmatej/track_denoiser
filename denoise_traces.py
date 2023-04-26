import numpy
from tensorflow import keras
import matplotlib.pyplot
import keras.utils
import os

class Model:
	'''
	@staticmethod
	def getModelZY():
		return keras.Sequential([	keras.layers.Input(shape=(10,208,1)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=6, filters=24, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=4, filters=64, activation="relu"),
									keras.layers.UpSampling2D(size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=1, kernel_size=6, filters=24, activation="relu"),
									keras.layers.Dense(units=1, activation="sigmoid") ])
	'''
	def __init__(self):
		self.chooseModelZX()

	def chooseModelZX(self):
		self.type = "zx"
		self.model = keras.Sequential([	keras.layers.Input(shape=(12,208,1)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,8), filters=16, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,6), filters=32, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(2,4), filters=64, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(2,2), filters=128, activation="relu"),
									keras.layers.UpSampling2D(size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(2,4), filters=64, activation="relu"),
									keras.layers.UpSampling2D(size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,6), filters=32, activation="relu"),
									keras.layers.UpSampling2D(size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,8), filters=16, activation="relu"),
									keras.layers.Dense(units=1, activation="sigmoid") ])

	def estimate(self, data_point :numpy.ndarray):
		'''Returns the output of the CNN call on input \'datapoint\'.'''
		data_point = numpy.reshape( self.model( numpy.reshape( data_point, (1, *data_point.shape)) ), data_point.shape)
		return data_point

	@staticmethod
	def load(name :str, model_type :str):
		'''Loads CNN model of type \'model_type\' (\"zy\" / \"zx\" / \"yx\") from tensorflow \'name\' directory.'''

		model = Model()
		model.type = model_type
		model.model = keras.models.load_model("./models/" + name)
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

	def saveSignalMetricData(self, data :list, path :str):
		f = open(path, "w")
		for val in data:
			f.write(str(val) + "\n")
		f.close()

class Plotting:
	@staticmethod
	def plotRandomData(model :keras.Model, signal_data :numpy.ndarray, noise_data :numpy.ndarray, num_plots = 5):
		for _ in range(num_plots):
			index = numpy.random.randint(15000, 15500)
			fig, ax = matplotlib.pyplot.subplots(3)
			ax[0].imshow( signal_data[index], cmap="gray" )
			ax[1].imshow( noise_data[index], cmap="gray" )
			ax[2].imshow( model.estimate(noise_data[index]), cmap="gray" )
			matplotlib.pyplot.show()


class Testing:
	@staticmethod
	def test_QualityEstimator_reconstructedSignals(reconstructed :numpy.ndarray, signal :numpy.ndarray, is_good_reconstruction :bool):
		fig, ax = matplotlib.pyplot.subplots(2,1)
		ax[0].imshow(signal, cmap='gray')
		ax[0].set_title("signal")
		ax[1].imshow(reconstructed, cmap='gray')
		ax[1].set_title("reconstructed")
		fig.suptitle("OK" if is_good_reconstruction else "Bad")
		matplotlib.pyplot.show()

class QualityEstimator:
	@staticmethod
	def reconstructedSignals(signal :numpy.ndarray, reconstructed :numpy.ndarray):
		'''Estimates the number of well reconstructed signal tracks.'''
		rec_num = 0
		treshold = 0.14
		shape = signal.shape
		metric_data = []

		def metric(sgn, rec):	return numpy.sum(numpy.abs(sgn - rec)) / sgn.size

		for n in range(shape[0]):
			if n % 100 == 0:	print(n, "/", shape[0])
			rec_matrix, sgn_matrix = reconstructed[n], signal[n]
			mask = sgn_matrix > 0.1
			mtr = metric(sgn_matrix[mask], rec_matrix[mask])
			#is_good = False
			if mtr < treshold:
				#is_good = True
				rec_num += 1
			metric_data.append(mtr)
			#print(metric(sgn_matrix[mask], rec_matrix[mask]))
			#Testing.test_QualityEstimator_reconstructedSignals(rec_matrix, sgn_matrix, is_good)
		return (rec_num / shape[0], metric_data)
