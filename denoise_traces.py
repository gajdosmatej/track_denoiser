import numpy
from tensorflow import keras
import tensorflow
import matplotlib.pyplot
import matplotlib.animation
import keras.utils
import copy
import os


def normalise(event :numpy.ndarray):
	'''
	Linearly map @event to [0,1] interval.
	'''

	M = numpy.max(event)
	if M == 0:	return event
	return event / M

class Metric:
	epsilon = 0.0000001

	def reconstructionMetric(classified, ground_truth):
		all_signal = numpy.sum( numpy.where(ground_truth > Metric.epsilon, 1, 0) )
		if all_signal == 0:	return None
		return numpy.sum( numpy.where(ground_truth > Metric.epsilon, classified, 0) ) / all_signal
	
	def noiseMetric(noisy, classified, ground_truth):
		num_signal_tiles = numpy.sum( numpy.where(ground_truth > Metric.epsilon, classified, 0) )
		num_noise_tiles = numpy.sum(classified) - num_signal_tiles
		num_all_noise = numpy.sum( numpy.where(noisy > Metric.epsilon, 1, 0) ) - numpy.sum( numpy.where(ground_truth > Metric.epsilon, 1, 0) )
		if num_all_noise == 0:	return None
		return num_noise_tiles / num_all_noise


class RandomNoise(keras.layers.Layer):
	'''
	Custom Keras Layer applying random gaussian noise to the input.	
	'''
	
	def __init__(self, std=1):
		super().__init__()
		self.std = std

	def call(self, x):
		return keras.layers.add([x, tensorflow.random.normal(shape=tensorflow.shape(x), mean=0, stddev=self.std)])

	def build(self, input_shape):
		pass


class SEAttention(keras.layers.Layer):
	'''
	Custom Keras Layer applying channelwise attention to a 3D tensor. 
	'''

	def __init__(self):
		super().__init__()
	
	def call(self, x):
		return keras.layers.multiply([x, self.getChannelAttention(x)])
	
	def build(self, input_shape):
		channels = input_shape[-1]

		self.avging = keras.layers.GlobalAveragePooling3D()
		self.maxing = keras.layers.GlobalMaxPooling3D()

		self.MLP_1 = keras.layers.Dense(channels//2, "relu")
		self.MLP_2 = keras.layers.Dense(channels, "relu")

	def getChannelAttention(self, x):
		'''
		Squeeze-Excitation returning channel attention for 3D tensor @x.
		'''

		return keras.layers.Activation("sigmoid")( keras.layers.Add()([	self.MLP_2(self.MLP_1(self.avging(x))),
																		self.MLP_2(self.MLP_1(self.maxing(x)))]) )


class SpatialAttention(keras.layers.Layer):
	'''
	Custom Keras Layer applying spatial attention to a 3D tensor.
	'''

	def __init__(self):
		super().__init__()
	
	def call(self, x):
		return keras.layers.multiply([x, self.getSpatialAttention(x)])
	
	def build(self, input_shape):
		self.aggregation = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(1,1,1), filters=1, activation="relu")
		self.conv = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")

	def getSpatialAttention(self, x):
		return self.conv( self.aggregation(x) )


class ModelWrapper:
	def __init__(self, model :keras.Model, model_name :str = "", threshold :float = None):
		self.model = model
		self.name = model_name
		self.threshold = threshold


	@staticmethod
	def loadPostprocessed(path :str, model_name :str):
		'''
		Return new instance of ModelWrapper class initiated by the files in @path directory.
		'''

		if path[-1] != "/":	path += "/"
		threshold_f = open(path + "threshold.txt", "r")
		threshold = float( threshold_f.read() )
		return ModelWrapper(keras.models.load_model(path + "model", compile=False), model_name, threshold)


	def evaluateSingleEvent(self, event :numpy.ndarray):
		'''
		Return Model(@event) for one single event.
		'''

		reshaped = numpy.reshape(event, (1, *event.shape, 1))
		result = self.model(reshaped)
		result = result[0]
		return numpy.reshape(result, event.shape)


	def evaluateBatch(self, events :numpy.ndarray):
		'''
		Return Model(@events), where @events is a batch of inputs.
		'''

		reshaped = numpy.reshape(events, (*events.shape, 1))
		results = self.model.predict(reshaped)
		return numpy.reshape(results, events.shape)


	def save(self, path :str):
		'''
		Save this model to @path.
		'''

		keras.models.save_model(self.model, path)


	def hasThreshold(self):
		'''
		Check, whether this object has defined classification threshold.
		'''

		return self.threshold != None


	def classify(self, raw_reconstruction :numpy.ndarray):
		'''
		Classify a reconstructed event by classification threshold.
		@raw_reconstruction ... Event outputed by this model.
		'''
		
		return numpy.where(raw_reconstruction > self.threshold, 1, 0)


class DataLoader:
	'''
	Class for loading X17 data and generated data in tensorflow datasets.
	'''
	def __init__(self, path :str):
		'''
		Create a new instance of DataLoader class.
		@path ... Path to data root directory (it should contain directories "simulated/" and "x17/").
		'''

		self.path = path + ('/' if path[-1] != '/' else '')


	def loadX17Data(self, track_type :str, noisy :bool):
		'''
		Parse X17 data from txt file into list of tuples (event name, 3D event array).
		@track_type: "goodtracks" or "othertracks"
		'''

		def parseX17Line(line :str):
			x = line[:line.index(",")]
			x = int(x)
			line = line[line.index(",")+1:]
			y = line[:line.index(",")]
			y = int(y)
			line = line[line.index(",")+1:]
			line = line[line.index(",")+1:]
			z = line[:line.index(",")]
			z = int(z)
			line = line[line.index(",")+1:]
			E = int(line)
			return (x, y, z, E)

		path = self.path + "x17/" + ("noisy/" if noisy else "clean/") + track_type

		for f_name in os.listdir(path):
			file = open(path + "/" + f_name, 'r')
			space = numpy.zeros((12,14,208))
			for line in file:
				x, y, z, E = parseX17Line(line)
				space[x,y,z] = E
			yield (f_name[:-4], space)


	def dataPairLoad(self, low_id :int, high_id :int):
		'''
		Yield a pair of noisy and clean event tensors from numbered data files in between @low_id and @high_id
		'''

		while True:
			order = numpy.arange(low_id, high_id)
			numpy.random.shuffle(order)
			for id in order:
				signal_batch = numpy.load(self.path + "simulated/clean/" + str(id) + ".npy")
				#signal_batch = numpy.where(signal_batch > 0.001, 1, 0)	#CLASSIFICATION
				noise_batch = numpy.load(self.path + "simulated/noisy/" + str(id) + ".npy")
				for i in range(5000):
					yield ( numpy.reshape(noise_batch[i], (12,14,208,1)), numpy.reshape(signal_batch[i], (12,14,208,1)))


	def getDataset(self, low_id :int, high_id :int, batch_size :int):
		'''
		Pack the method _dataPairLoad_(@low_id, @high_id) into TensorFlow dataset.
		'''

		return tensorflow.data.Dataset.from_generator(lambda: self.dataPairLoad(low_id, high_id), output_signature =
					(	tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16),
						tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16))
					).batch(batch_size).prefetch(20)


	def getValidationData(self):
		noisy, clean = self.getBatch(True, True, track_type="goodtracks"), self.getBatch(True, False, track_type="goodtracks")
		noisy = numpy.reshape(noisy, (*noisy.shape, 1))
		clean = numpy.reshape(clean, (*clean.shape, 1))
		return (noisy, clean)


	def getBatch(self, experimental :bool = True, noisy :bool = True, file_id :int = 0, track_type :str = "alltracks"):
		'''
		Get one data batch as numpy array.
		@experimental: Whether simulated or X17 data should be used.
		@noisy: Whether noisy or clean data should be used.
		@file_id: File ID, from which the simulated batch should be loaded (useless for @experimental = True).
		@track_type: Important only for @experimental = True. Either "goodtracks" (only "data/x17/goodtracks/"), "othertracks" (only "data/x17/othertracks"), "alltracks" (both "data/x17/goodtracks"
		and "data/x17/othertracks") or "midtracks" (like "alltracks", but filtered those that do not contain a track).
		'''
		
		if experimental:
			if track_type == "goodtracks":
				x17_data = []
				for _, event in self.loadX17Data("goodtracks", noisy):
					x17_data.append( normalise(event) )
				return numpy.array(x17_data)
			
			elif track_type == "othertracks":
				x17_data = []
				for _, event in self.loadX17Data("othertracks", noisy):
					x17_data.append( normalise(event) )
				return numpy.array(x17_data)
			
			elif track_type == "alltracks":
				return numpy.concatenate( [self.getBatch(True, noisy, track_type="goodtracks"), self.getBatch(True, noisy, track_type="othertracks")], axis=0 )
			
			elif track_type == "midtracks":
				good_enough_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
										31, 32, 33, 35, 37, 38, 39, 40, 42, 43, 47, 56, 59, 61, 62, 65, 74, 78, 79, 82, 90, 91, 92, 102, 103, 104, 105, 
										106, 112, 123, 133, 138, 149, 155, 160, 162, 163, 172, 175, 188, 189, 193, 199, 209, 210, 216, 217, 227, 228, 230, 
										234, 235, 239, 240, 241, 244, 245, 246, 253, 262, 266, 270, 283, 285, 289, 295, 303, 308, 309, 314, 315, 317, 320, 
										324, 326, 329, 337, 344, 345, 354, 357, 367, 369, 371, 381, 383, 388, 393, 398, 402, 404, 405, 411, 418, 422, 423, 
										434, 436, 441, 456, 461, 463, 465, 466, 468, 484, 488, 492, 501, 511, 516, 517, 519, 521, 524, 525, 526, 527, 531, 
										537, 538, 542, 552, 561, 565, 567, 572, 581, 592, 596, 598, 600, 601, 607, 612, 614, 617, 625, 628, 633, 638, 639]
				return self.getBatch(True, noisy, track_type="alltracks")[good_enough_indices]
			
			else:
				raise ValueError
		else:
			return numpy.load(self.path + "simulated/" + ("noisy/" if noisy else "clean/") + str(file_id) + ".npy")
	
	def getX17Names(self):
		names = []
		for (name, _) in self.loadX17Data("goodtracks", False):	names.append(name)
		for (name, _) in self.loadX17Data("othertracks", False):	names.append(name)
		return names



class Plotting:
	@staticmethod
	def plotEvent(noisy, reconstruction, classificated = None, are_data_experimental = None, model_name = '', axes=[0,1,2], use_log :bool = False, event_name :str = None):
		'''
		Create plot of track reconstruction.
		@noisy ... Noisy event tensor.
		@reconstruction ... Tensor of event reconstructed by model.
		@classificated ... Event after threshold classification. Default is None, which skips this plot.
		@are_data_experimental ... False iff the event is from the generated dataset.
		@model_name ... Name of the model, which will be displayed in the plot.
		@axes ... List of plotted projection axes.
		@use_log ... If True, use log scale.	
		'''
		
		if classificated is None:
			fig, ax = matplotlib.pyplot.subplots(len(axes), 2)
		else:
			fig, ax = matplotlib.pyplot.subplots(len(axes), 3)

		x_labels = ['z', 'z', 'y']
		y_labels = ['y', 'x', 'x']

		fixed_cmap = copy.copy(matplotlib.cm.get_cmap('gray'))
		fixed_cmap.set_bad((0,0,0))	#fix pixels with zero occurrence - otherwise problem for LogNorm
		norm = matplotlib.colors.LogNorm(vmin=0, vmax=1) if use_log else matplotlib.colors.PowerNorm(1, vmin=0, vmax=1)

		for i in range(len(axes)):
			axis = axes[i]
			ax[i][0].set_title("Noisy")
			ax[i][0].imshow(numpy.sum(noisy, axis=axis), cmap=fixed_cmap, norm=norm )
			ax[i][0].set_xlabel(x_labels[axis])
			ax[i][0].set_ylabel(y_labels[axis])
			ax[i][1].set_title("Raw Reconstruction")
			ax[i][1].imshow(numpy.sum(reconstruction, axis=axis), cmap=fixed_cmap, norm=norm )
			ax[i][1].set_xlabel(x_labels[axis])
			ax[i][1].set_ylabel(y_labels[axis])

		if classificated is not None:
			for i in range(len(axes)):
				axis = axes[i]
				ax[i][2].set_title("After Threshold")
				ax[i][2].imshow(numpy.sum(classificated, axis=axis), cmap=fixed_cmap, norm=norm )
				ax[i][2].set_xlabel(x_labels[axis])
				ax[i][2].set_ylabel(y_labels[axis])
			
		title = "Reconstruction of "
		if are_data_experimental:	title += "Experimental "
		elif are_data_experimental is False:	title += "Generated "
		title += "Data "
		if event_name is not None:	title += "(" + event_name + ") "
		title += "by Model " + model_name
		fig.suptitle(title)
	

	@staticmethod
	def plotEventOneAxis(modelAPI :ModelWrapper, noise_data :numpy.ndarray, axis :int, are_data_experimental :bool = None, event_name :str = None):
		x_labels = ['z', 'z', 'y']
		y_labels = ['y', 'x', 'x']
		
		reconstructed = modelAPI.evaluateSingleEvent(noise_data)

		if modelAPI.threshold is not None:
			classified = modelAPI.classify(reconstructed)
			fig, ax = matplotlib.pyplot.subplots(3)
		else:
			fig, ax = matplotlib.pyplot.subplots(2)
		
		ax[0].set_title("Noisy")
		ax[0].imshow(numpy.sum(noise_data, axis=axis), cmap="gray")
		ax[0].set_xlabel(x_labels[axis])
		ax[0].set_ylabel(y_labels[axis])
		ax[1].set_title("Raw Reconstruction")
		ax[1].imshow(numpy.sum(reconstructed, axis=axis), cmap="gray")
		ax[1].set_xlabel(x_labels[axis])
		ax[1].set_ylabel(y_labels[axis])

		if modelAPI.threshold is not None:
			ax[2].set_title("After Threshold")
			ax[2].imshow(numpy.sum(classified, axis=axis), cmap="gray")
			ax[2].set_xlabel(x_labels[axis])
			ax[2].set_ylabel(y_labels[axis])
			
		title = "Reconstruction of "
		if are_data_experimental:	title += "Experimental "
		elif are_data_experimental is False:	title += "Generated "
		title += "Data "
		if event_name is not None:	title += "(" + event_name + ") "
		title += "by Model " + modelAPI.name
		fig.suptitle(title)


	@staticmethod
	def plotRandomData(modelAPI :ModelWrapper, noise_data :numpy.ndarray, are_data_experimental :bool = None, axes :list = [0,1,2], use_log :bool = False):
		'''
		Plot @model's reconstruction of random events from @noise_data. If @threshold is specified, plot also the final classification after applying @threshold to reconstruciton.
		'''
		while True:
			index = numpy.random.randint(0, len(noise_data))
			noisy = noise_data[index]
			reconstr = modelAPI.evaluateSingleEvent(noisy)

			if modelAPI.hasThreshold():
				classif = modelAPI.classify(reconstr)
				Plotting.plotEvent(noisy, reconstr, classif, are_data_experimental, modelAPI.name, axes = axes, use_log = use_log)
			else:
				Plotting.plotEvent(noisy, reconstr, None, are_data_experimental, modelAPI.name, axes = axes, use_log = use_log)
			matplotlib.pyplot.show()
			if input("Enter 'q' to stop plotting (or anything else for another plot):") == "q":	break
	
	@staticmethod
	def getPlot3D(modelAPI :ModelWrapper, noise_event :numpy.ndarray, are_data_experimental :bool = None, event_name :str = None):
		'''
		Return 3D plot of @noise_event and its reconstruction by @model.
		@model ... Keras model reconstructing track in this plot.
		@noise_event ... One noisy event tensor.
		@are_data_experimental ... False iff the event is from generated dataset.
		@model_name ... Name of the model which will be displayed in the plot.
		@threshold ... Classification threshold for the model.
		@rotation ... Float triple specifying the plot should be rotated.
		'''

		fig = matplotlib.pyplot.figure(figsize=matplotlib.pyplot.figaspect(0.5))
		ax1 = fig.add_subplot(1, 2, 1, projection='3d')
		title = ""
		if are_data_experimental:	title += "Experimental "
		elif are_data_experimental is False:	title += "Generated "
		if event_name is not None:	title += "(" + event_name + ") "
		title += "Data"
		sctr = Plotting.plot3DToAxis(noise_event, ax1, title)

		reconstr_event = modelAPI.evaluateSingleEvent( noise_event / (numpy.max(noise_event) if numpy.max(noise_event) != 0 else 1) )
		classificated_event = modelAPI.classify(reconstr_event)
		ax2 = fig.add_subplot(1, 2, 2, projection='3d')
		title = "Reconstruction and Threshold Classification\n"
		title += "by Model " + modelAPI.name
		Plotting.plot3DToAxis(classificated_event, ax2, title)

		cb = fig.colorbar(sctr, ax=[ax1, ax2], orientation="horizontal")
		cb.set_label("$E$")

		return fig, ax1, ax2


	def plot3DToAxis(event :numpy.ndarray, ax, title :str = ""):
		def scaleSize(val):	return val*150 + 50
		xs, ys, zs = event.nonzero()
		vals = numpy.array([event[xs[i],ys[i],zs[i]] for i in range(len(xs))])
		sctr = ax.scatter(xs, ys, zs, c=vals, cmap="plasma", marker="s", s=scaleSize(vals))
		ax.set_xlim(0, 11)
		ax.set_xlabel("$x$")
		ax.set_ylim(0, 13)
		ax.set_ylabel("$y$")
		ax.set_zlim(0, 200)
		ax.set_zlabel("$z$")
		ax.set_title(title)
		ax.set_box_aspect((12, 14, 50))
		return sctr

	def animation3D(path :str, modelAPI :ModelWrapper, noise_event :numpy.ndarray, are_data_experimental :bool = None):
		fig, ax1, ax2 = Plotting.getPlot3D(modelAPI, noise_event, are_data_experimental)

		def run(i):	
			ax1.view_init(0,i,0)
			ax2.view_init(0,i,0)

		anim = matplotlib.animation.FuncAnimation(fig, func=run, frames=360, interval=20, blit=False)
		anim.save(path, fps=30, dpi=200, writer="pillow")
