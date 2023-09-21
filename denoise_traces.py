import numpy
from tensorflow import keras
import tensorflow
import matplotlib.pyplot
import matplotlib.animation
import keras.utils
import copy
import os


class ModelWrapper:
	def __init__(self, model :keras.Model, model_name :str = "", threshold :float = None):
		self.model = model
		self.name = model_name
		self.threshold = threshold


	@staticmethod
	def loadPostprocessed(path :str, model_name :str):
		if path[-1] != "/":	path += "/"
		threshold_f = open(path + "threshold.txt", "r")
		threshold = float( threshold_f.read() )
		return ModelWrapper(keras.models.load_model(path + "model"), model_name, threshold)


	def evaluateSingleEvent(self, event :numpy.ndarray):
		reshaped = numpy.reshape(event, (1, *event.shape, 1))
		result = self.model(reshaped)
		result = result[0]
		return numpy.reshape(result, event.shape)


	def evaluateBatch(self, events :numpy.ndarray):
		reshaped = numpy.reshape(events, (*events.shape, 1))
		results = self.model.predict(reshaped)
		return numpy.reshape(results, events.shape)


	def save(self, path :str):
		keras.models.save_model(self.model, path)


	def hasThreshold(self):
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
		Pack the method _dataPairLoad_(@low_id, @high_id) into tensorflow dataset.
		'''
		return tensorflow.data.Dataset.from_generator(lambda: self.dataPairLoad(low_id, high_id), output_signature =
					(	tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16),
						tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16))
					).batch(batch_size).prefetch(20)

	'''
	#OBSOLETE
	def getNoisyBatch(self, experimental :bool = True, file_id :int = 0):
		''
		Return a list of noisy data. If @experimental is True, return real data from X17 experiment, otherwise generated data are used from file specified by $file_id.
		''
		
		if experimental:
			x17_data = numpy.array( [event for (_, event) in self.loadX17Data("goodtracks")] )
			return x17_data / numpy.max(x17_data)	#normalisation to [0,1] interval
		else:
			return numpy.load(self.path + "simulated/noisy/" + str(file_id) + ".npy")
	'''

	def getBatch(self, experimental :bool = True, noisy :bool = True, file_id = 0):
		if experimental:
			x17_data = [event for (_, event) in self.loadX17Data("goodtracks", noisy)]
			for (_, event) in self.loadX17Data("othertracks", noisy):
				x17_data.append(event)
			x17_data = numpy.array(x17_data)
			return x17_data / numpy.max(x17_data)	#normalisation to [0,1] interval
		else:
			return numpy.load(self.path + "simulated/" + ("noisy/" if noisy else "clean/") + str(file_id) + ".npy")



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
