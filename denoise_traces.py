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
		return ModelWrapper(keras.models.load_model(path + "model"), model_name, threshold)


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
				good_enough_indices = [	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
										34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 50, 52, 56, 57, 58, 59, 60, 61, 62, 64, 65, 67, 74, 77, 78, 79, 80, 82, 83, 84, 
										89, 90, 91, 92, 93, 96, 99, 101, 102, 103, 104, 105, 106, 109, 112, 117, 118, 120, 123, 124, 125, 128, 131, 133, 138, 141, 144, 
										145, 147, 148, 149, 150, 153, 155, 160, 161, 162, 163, 164, 170, 172, 175, 178, 184, 185, 187, 188, 189, 190, 192, 193, 194, 195, 
										196, 197, 199, 209, 210, 213, 215, 216, 217, 222, 223, 227, 228, 230, 234, 235, 239, 240, 241, 244, 245, 246, 249, 250, 251, 253, 
										257, 261, 262, 266, 268, 270, 271, 283, 284, 285, 286, 289, 290, 295, 296, 298, 301, 303, 306, 307, 308, 309, 312, 314, 315, 317, 
										320, 323, 324, 325, 326, 327, 328, 329, 330, 333, 336, 337, 340, 343, 344, 345, 346, 350, 352, 353, 354, 355, 357, 361, 367, 368, 
										369, 371, 372, 373, 378, 379, 381, 383, 384, 388, 393, 395, 398, 399, 402, 403, 404, 405, 408, 411, 412, 414, 418, 419, 422, 423, 425, 
										426, 434, 436, 441, 444, 448, 450, 454, 456, 458, 459, 461, 463, 464, 465, 466, 468, 484, 485, 488, 491, 492, 496, 498, 501, 502, 510, 
										511, 512, 513, 516, 517, 519, 521, 522, 524, 525, 526, 527, 530, 531, 537, 538, 542, 551, 552, 554, 555, 559, 561, 563, 565, 567, 572, 
										573, 581, 592, 594, 595, 596, 598, 600, 601, 604, 607, 612, 614, 617, 619, 625, 626, 628, 631, 632, 633, 634, 637, 638, 639, 640]
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
	def getPlot3D(modelAPI :ModelWrapper, noise_event :numpy.ndarray, are_data_experimental :bool = None, rotation=(0,0,0), event_name :str = None):
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
		xs, ys, zs = noise_event.nonzero()
		vals = numpy.array([noise_event[xs[i],ys[i],zs[i]] for i in range(len(xs))])
		sctr1 = ax1.scatter(xs, ys, zs, c=vals, cmap="plasma", marker="s", s=80)
		ax1.set_xlim(0, 11)
		ax1.set_xlabel("$x$")
		ax1.set_ylim(0, 13)
		ax1.set_ylabel("$y$")
		ax1.set_zlim(0, 200)
		ax1.set_zlabel("$z$")
		title = title = "Noisy "
		if are_data_experimental:	title += "Experimental "
		elif are_data_experimental is False:	title += "Generated "
		if event_name is not None:	title += "(" + event_name + ") "
		title += "Data"
		ax1.set_title(title)
		ax1.view_init(*rotation)	#rotate the scatter plot, useful for animation

		reconstr_event = modelAPI.evaluateSingleEvent( noise_event / (numpy.max(noise_event) if numpy.max(noise_event) != 0 else 1) )
		classificated_event = modelAPI.classify(reconstr_event)
		ax2 = fig.add_subplot(1, 2, 2, projection='3d')
		xs, ys, zs = classificated_event.nonzero()
		vals = numpy.array([classificated_event[xs[i],ys[i],zs[i]] for i in range(len(xs))])
		sctr2 = ax2.scatter(xs, ys, zs, c=vals, marker="s", s=80)
		ax2.set_xlim(0, 11)
		ax2.set_xlabel("$x$")
		ax2.set_ylim(0, 13)
		ax2.set_ylabel("$y$")
		ax2.set_zlim(0, 200)
		ax2.set_zlabel("$z$")
		ax2.view_init(*rotation)
		title = "Reconstruction and Threshold Classification\n"
		title += "by Model " + modelAPI.name
		ax2.set_title(title)

		#fig.subplots_adjust(right=0.8)
		#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		cb = fig.colorbar(sctr1, ax=[ax1, ax2], orientation="horizontal")
		cb.set_label("$E$")

		return fig, ax1, ax2


	def animation3D(path :str, modelAPI :ModelWrapper, noise_event :numpy.ndarray, are_data_experimental :bool = None):
		fig, ax1, ax2 = Plotting.getPlot3D(modelAPI, noise_event, are_data_experimental)

		def run(i):	
			ax1.view_init(0,i,0)
			ax2.view_init(0,i,0)

		anim = matplotlib.animation.FuncAnimation(fig, func=run, frames=360, interval=20, blit=False)
		anim.save(path, fps=30, dpi=200, writer="pillow")
