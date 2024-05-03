import numpy
from tensorflow import keras
import tensorflow
import matplotlib.pyplot
import matplotlib.animation
import keras.utils
import copy
import os

VERBOSE = False

def normalise(event :numpy.ndarray):
	'''
	Linearly map @event to [0,1] interval.
	'''

	M = numpy.max(event)
	if M == 0:	return event
	return event / M


class Vector:
	@staticmethod
	def add(v, w):
		if len(v) != len(w):
			print("WARNING [Vector.add]: Non-compatible vector shapes")
			return None
		return tuple(v[i] + w[i] for i in range(len(v)))

	@staticmethod
	def multiply(alpha, v):
		return tuple(alpha*v[i] for i in range(len(v)))

	@staticmethod
	def linComb(alpha, beta, v, w):
		return tuple(alpha*v[i] + beta*w[i] for i in range(len(v)))

	@staticmethod
	def dotProduct(v, w):
		if len(v) != len(w):
			print("WARNING [Vector.dotProduct]: Non-compatible vector shapes")
			return None
		return sum(v[i] * w[i] for i in range(len(v)))

	@staticmethod
	def norm(v):
		return numpy.sqrt( Vector.dotProduct(v, v) )
	
	@staticmethod
	def normalise(v):
		if v == (0,0,0):	return v
		return Vector.multiply( 1/Vector.norm(v), v)

class Cluster:
	'''Class for clusters and methods for clusterisation.'''

	active_zone_threshold = 0.7
	max_neighbour_coef = 5
	min_energy_density = 80
	min_length = 10
	max_nonlinearity = 1

	def __init__(self, coords):
		self.coords = coords
		self.length = len(coords)
		self.corners = []
		self.neighbour_coef = self.getNeighbourCoefficient()
		self.findCorners()
		self.tests = {}

	def __str__(self):
		result = "L = " + str(self.length)
		result += "\nneigh_coef = " + str(self.neighbour_coef)
		result += "\nE_density = " + str(self.energy_density)
		result += "\n#Passed tests = " + str(self.getPassedTestsNum())
		return result
	
	@staticmethod
	def union(c, d):
		return Cluster(c.coords + d.coords)

	def findCorners(self):
		def coordsFormClique(coords):
			n = len(coords)
			for i in range(n):
				for j in range(i+1,n):
					u, v = coords[i], coords[j]
					if max( abs(u[k]-v[k]) for k in [0,1,2] ) > 1:	return False
			return True

		cluster_tensor = self.getTensor()
		for coord in self.coords:
			neighbours = []
			for neighbour in self.neighbourhood(coord, 1, 1, 1):
				if cluster_tensor[neighbour] != 0:	neighbours.append(neighbour)
			if coordsFormClique(neighbours):	self.corners.append(coord)
	
	def linify(self):
		if len(self.corners) < 2:
			if VERBOSE:	print("WARNING [Cluster.linify]: Less that two corner tiles found, skipping")
			return
		corner_left, corner_right = self.corners[0], self.corners[0]
		for c in self.corners:
			if c[2] < corner_left[2]:	corner_left = c
			if c[2] > corner_right[2]:	corner_right = c

		return (corner_left, corner_right)

		'''sum_direction = (0,0,0)
		for coord in self.coords:
			component_direction = Vector.linComb(1, -1, coord, corner_left)
			component_direction = Vector.normalise(component_direction)
			sum_direction = Vector.add(sum_direction, component_direction)
		sum_direction = Vector.multiply(1/abs(sum_direction[2]), sum_direction)
		return( corner_left, Vector.linComb(1, corner_right[2] - corner_left[2], corner_left, sum_direction) )'''

	@staticmethod
	def clusterise(event :numpy.ndarray):
		'''Create list of Clusters from input @event.'''
		event = numpy.copy(event)
		clusters = []

		remaining_tracks = event.nonzero()
		while remaining_tracks[0].size != 0:
			corners = []	#all tiles where the clusterisation stops (DFS leaves)
			cluster = [(remaining_tracks[0][0], remaining_tracks[1][0], remaining_tracks[2][0])]
			event[cluster[0]] = 0
			stack = [cluster[0]]
			while stack != []:
				coord = stack.pop()
				for neighbour in Cluster.neighbourhood(coord, 1, 1, 2):
					if event[neighbour] != 0:
						cluster.append(neighbour)
						stack.append(neighbour)
						event[neighbour] = 0
			clusters.append( Cluster(cluster) )
			remaining_tracks = event.nonzero()
		return clusters

	@staticmethod
	def neighbourhood(coord, x_range, y_range, z_range):
		x,y,z = coord
		for i in range(-x_range, x_range+1):
			for j in range(-y_range, y_range+1):
				for k in range(-z_range, z_range+1):
					if 0 <= x+i < 12 and 0 <= y+j < 14 and 0 <= z+k < 208:
						yield (x+i, y+j, z+k)
	
	def testActiveZone(self):
		'''Check whether the main part of @cluster is located in active zone @zone.'''
		num_in_zone = 0

		for coord in self.coords:
			if Cluster.isInModeledZone(coord):	num_in_zone += 1
		
		return ( num_in_zone/self.length > self.active_zone_threshold )
	
	def runTests(self):
		'''Check conditions for good cluster.'''

		try:	self.energy
		except:	
			print("WARNING [Cluster.runTests]: Cluster.energy not set yet, aborting.")
			return

		self.tests["length"] = self.length > self.min_length
		self.tests["zone"] = self.testActiveZone()
		self.tests["neighbours"] = self.neighbour_coef <= self.max_neighbour_coef
		self.tests["energy_density"] = self.energy_density > self.min_energy_density
	
	def getPassedTestsNum(self):
		if len(self.tests) != 4:
			print("WARNING [Cluster.getPassedTestsNum]: Some tests were not run yet.")
		return sum(self.tests[key] for key in self.tests)
	
	def isGood(self):
		if len(self.tests) != 4:
			print("WARNING [Cluster.getPassedTestsNum]: Some tests were not run yet.")
		return sum(self.tests[key] for key in self.tests) == 4

	def getNeighbourCoefficient(self):
		'''
		Return average number of neighbours (including self) of cluster tiles.
		'''
		
		if self.coords == []:	return None
		tensor = self.getTensor()
		coefs = []
		for coord in self.coords:
			current = 0
			for neighbour in Cluster.neighbourhood(coord, 1, 1, 1):
				if tensor[neighbour] == 1:	current += 1
			coefs.append(current)
		return sum(coefs)/len(coefs)

	@staticmethod
	def getModeledZoneMask():
		zone = numpy.zeros((12,14,208))
		for x in range(12):
			for y in range(14):
				for z in range(98,144):
					if Cluster.isInModeledZone((x,y,z)):
						zone[(x,y,z)] = 1
		return zone

	@staticmethod
	def isInModeledZone(coord):
		x,y,z = coord
		return (98 <= z <=126 and 45/2*x-82 <= z) or (126 < z <= 143 and 3/17*(z-126) <= x <= 2/45*(z+82))

	#TODO OPTIMISE
	def getTensor(self, event=None):
		result = numpy.zeros((12,14,208))
		for coord in self.coords:
			result[coord] = (1 if event is None else event[coord])
		return result
	
	def setEnergy(self, event):
		self.energy = 0
		self.effective_length = 0
		for coord in self.coords:
			self.energy += event[coord]
			if event[coord] != 0:
				self.effective_length += 1	
		self.energy_density = self.energy / self.effective_length

	@staticmethod
	def getGoodFromDataset(dataset, energy_dataset):
		good_dataset = numpy.zeros(dataset.shape)
		for i in range(dataset.shape[0]):
			clusters = Cluster.clusterise(dataset[i])
			for cluster in clusters:
				cluster.setEnergy(energy_dataset[i])
				cluster.runTests()
				if cluster.isGood():	good_dataset[i] += cluster.getTensor()
		return good_dataset

	@staticmethod
	def crossconnectClusters(clusters1, clusters2):
		'''
		Returns a tuple (@subsets1of2, @subsets2of1) -- @subsets1of2 has the same length as @clusters1, 
		an i-th item contains the index of cluster from @clusters2 which is an essential superset of @clusters1[i]. If such a superset does not exist, @subsets1of2[i] = None.
		'''
		
		essential_subset_threshold = 0.7	#minimal |a & b| / |a| so that a is essential subset of b 
		n1, n2 = len(clusters1), len(clusters2)
		subsets1of2 = [None]*n1
		subsets2of1 = [None]*n2

		for i in range(n1):	#fix one cluster in @clusters1
			a = clusters1[i]
			for j in range(n2):	#go through @clusters2 and compare the relationship of each with @a
				b = clusters2[j]
				intersect_size = sum(1 for coord in a.coords if coord in b.coords)
				if intersect_size / len(a.coords) >= essential_subset_threshold:	# @a is essential subset of @b
					subsets1of2[i] = j
				if intersect_size / len(b.coords) >= essential_subset_threshold:
					subsets2of1[j] = i
		return (subsets1of2, subsets2of1)

	@staticmethod
	def defragment(clusters1, clusters2):
		'''
		Returns (@defragmented_clusters1, @deframented_clusters2, @are_original1, @are_original2), 
		where the original clusters with the same essential superset are connected together in the first two lists. In the third and fourth lists, 
		information whether cluster was formed from multiple clusters is stored.
		'''

		n1, n2 = len(clusters1), len(clusters2)
		subsets1of2, subsets2of1 = Cluster.crossconnectClusters(clusters1, clusters2)
		defragmented_clusters1, defragmented_clusters2 = [], []
		are_original1, are_original2 = [], []

		for i in range(n1):	#fix one cluster in @clusters1
			if subsets1of2[i] is None:	#add to @defragmented_clusters1 the clusters which do not have essential superset
				defragmented_clusters1.append(clusters1[i])
				are_original1.append(True)

			new_cluster = None
			is_original = True
			for j in range(n2):	#connect all @clusters2 whose essential superset is @i
				if subsets2of1[j] == i:
					if new_cluster is None:
						new_cluster = clusters2[j]
					else: 
						new_cluster = Cluster.union(clusters2[j], new_cluster)
						is_original = False
			if new_cluster is not None:
				defragmented_clusters2.append(new_cluster)
				are_original2.append(is_original)

		for j in range(n2):
			if subsets2of1[j] is None:
				defragmented_clusters2.append(clusters2[j])
				are_original2.append(True)

			new_cluster = None
			is_original = True
			for i in range(n1):
				if subsets1of2[i] == j:
					if new_cluster is None:
						new_cluster = clusters1[i]
					else: 
						new_cluster = Cluster.union(clusters1[i], new_cluster)
						is_original = False
			if new_cluster is not None:
				defragmented_clusters1.append(new_cluster)
				are_original1.append(is_original)

		return (defragmented_clusters1, defragmented_clusters2, are_original1, are_original2)


#OBSOLETE CLASS
class Metric:
	'''
	Class wrapping methods for metrics calculations.
	'''

	epsilon = 0.0000001

	@staticmethod
	def reconstructionMetric(classified, ground_truth):
		'''
		Return the relative number of reconstructed signal tiles.
		'''

		all_signal = numpy.sum( numpy.where(ground_truth > Metric.epsilon, 1, 0) )
		if all_signal == 0:	return None
		return numpy.sum( numpy.where(ground_truth > Metric.epsilon, classified, 0) ) / all_signal


	@staticmethod
	def noiseMetric(noisy, classified, ground_truth):
		'''
		Return the relative number of unfiltered noise tiles.
		'''

		num_signal_tiles = numpy.sum( numpy.where(ground_truth > Metric.epsilon, classified, 0) )
		num_noise_tiles = numpy.sum(classified) - num_signal_tiles
		num_all_noise = numpy.sum( numpy.where(noisy > Metric.epsilon, 1, 0) ) - numpy.sum( numpy.where(ground_truth > Metric.epsilon, 1, 0) )
		if num_all_noise == 0:	return None
		return num_noise_tiles / num_all_noise

	#NEEDS TO FIX
	@staticmethod
	def getRatioOfGoodClusters(data):
		'''
		Return the ratio of good clusters in @data. If no cluster is found in @data, return -1.
		'''

		good_counter = 0
		all_counter = 0
		for event in data:
			clusters = Cluster.clusterise(event)
			if clusters == []:	continue
			for cluster in clusters:
				all_counter += 1
				if cluster.isGood():
					good_counter += 1

		return (good_counter / all_counter if all_counter != 0 else -1)

	def getNumberOfGoodEvents(data, energy_data):
		'''
		Return the number of events in @data which contain only clusters that pass all the tests.
		'''

		good_counter = 0
		for i in range(data.shape[0]):
			clusters = Cluster.clusterise(data[i])
			if clusters == []:	continue
			for cluster in clusters:
				cluster.setEnergy(energy_data[i])
				cluster.runTests()
				if cluster.getPassedTestsNum() != 4:	break
			else:
				good_counter += 1
		return good_counter



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
	'''
	Wrapper for Keras Model class with additional convenient methods.
	'''

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
		self.name_index_dict = {}
		self.fillNameIndexDictionary()

		self.existing_experimental_datasets = {}

	def dumpData(self, data, names, dir_name = "dump/"):
		'''
		Export @data to (DataLoader.path + @dir_name). Each @data[i] event is saved in its own { @names[i] }.txt file, where each line corresponds to "x, y, z, E" nonzero energy element of the event.
		'''

		if dir_name[-1] != "/":	dir_name += "/"
		path = self.path + dir_name
		if not os.path.isdir(path):
			os.makedirs(path)
		else:
			print("WARNING [DataLoader.dumpData]: Dump directory already exists, aborting")
			return
		
		for event, name in zip(data, names):
			dump_str = ""
			xs, ys, zs = numpy.nonzero(event)
			for i in range(xs.shape[0]):
				x, y, z = xs[i], ys[i], zs[i]
				E = event[x,y,z]
				dump_str += str(x) + ", " + str(y) + ", " + str(z) + ", " + str(int(E)) + "\n"
			dump_str = dump_str[:-1]
			f = open(path + name + ".txt", "w")
			f.write(dump_str)
			f.close()

	def importData(self, dir_name = "dump/", names=None):
		'''
		Import data previously exported by DataLoader._dumpData_ to a directory (DataLoader.path + @dir_name).  
		'''

		def parseLine(line :str):
			x = line[:line.index(",")]
			x = int(x)
			line = line[line.index(",")+1:]
			y = line[:line.index(",")]
			y = int(y)
			line = line[line.index(",")+1:]
			z = line[:line.index(",")]
			z = int(z)
			line = line[line.index(",")+1:]
			E = int(line)
			return (x, y, z, E)
		
		if dir_name[-1] != "/":	dir_name += "/"
		path = self.path + dir_name
		data = numpy.zeros( (len(os.listdir(path)), 12, 14, 208) )

		if names is None:
			names = []
			i = 0
			for name in os.listdir(path):
				f = open(path + name, "r")
				for line in f.readlines():
					x, y, z, E = parseLine(line)
					data[i,x,y,z] = E
				f.close()
				names.append(name[:-4])
				i += 1
			return (data, names)
		else:
			i = 0
			for name in names:
				f = open(path + name + ".txt", "r")
				for line in f.readlines():
					x, y, z, E = parseLine(line)
					data[i,x,y,z] = E
				f.close()
				i += 1
			return (data, names)

	def fillNameIndexDictionary(self):
		names = self.getX17Names()
		for i in range(len(names)):
			self.name_index_dict[names[i]] = i

	def getEventFromName(self, name: str, noisy :bool, normalising :bool = True):
		'''
		Return X17 event from its @name.
		'''

		return self.getBatch(True, noisy, track_type="alltracks", normalising=normalising)[ self.name_index_dict[name] ]
		#names, events = self.getX17Names(), self.getBatch(True, noisy, track_type="alltracks", normalising=normalising)
		#return [event for (n, event) in zip(names, events) if n == name][0]

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
					).batch(batch_size).prefetch(10)

	def getValidationData(self):
		'''
		Return tuple (X17_noisy, X17_clean), which is convenient format for Keras Model.fit validation_data argument.
		'''

		noisy, clean = self.getBatch(True, True, track_type="goodtracks"), self.getBatch(True, False, track_type="goodtracks")
		noisy = numpy.reshape(noisy, (*noisy.shape, 1))
		clean = numpy.reshape(clean, (*clean.shape, 1))
		return (noisy, clean)

	def datasetExists(self, noisy :bool, track_type :str, normalising :bool):
		return (noisy, track_type, normalising) in self.existing_experimental_datasets
	
	def getDatasetFromDictionary(self, noisy :bool, track_type :str, normalising :bool):
		return self.existing_experimental_datasets[(noisy, track_type, normalising)]

	def addDatasetToDictionary(self, dataset :numpy.array, noisy :bool, track_type :str, normalising :bool):
		self.existing_experimental_datasets[(noisy, track_type, normalising)] = dataset

	def getBatch(self, experimental :bool = True, noisy :bool = True, file_id :int = 0, track_type :str = "alltracks", normalising :bool = True):
		'''
		Get one data batch as numpy array.
		@experimental: Whether simulated or X17 data should be used.
		@noisy: Whether noisy or clean data should be used.
		@file_id: File ID, from which the simulated batch should be loaded (useless for @experimental = True).
		@track_type: Important only for @experimental = True. Either "goodtracks" (only "data/x17/goodtracks/"), "othertracks" (only "data/x17/othertracks") or "alltracks" (both "data/x17/goodtracks"
		and "data/x17/othertracks").
		@normalising: Whether X17 data should be mapped to [0,1] first (default True, good for evaluating by models, but energy information is lost).
		'''
		
		if not experimental:
			return numpy.load(self.path + "simulated/" + ("noisy/" if noisy else "clean/") + str(file_id) + ".npy")
		
		if self.datasetExists(noisy, track_type, normalising):
			return self.getDatasetFromDictionary(noisy, track_type, normalising)
	
		if track_type == "goodtracks":
			x17_data = []
			for _, event in self.loadX17Data("goodtracks", noisy):
				x17_data.append( normalise(event) if normalising else event )
			dataset = numpy.array(x17_data)
			self.addDatasetToDictionary(dataset, noisy, track_type, normalising)
			return dataset
		elif track_type == "othertracks":
			x17_data = []
			for _, event in self.loadX17Data("othertracks", noisy):
				x17_data.append( normalise(event) if normalising else event )
			dataset = numpy.array(x17_data)
			self.addDatasetToDictionary(dataset, noisy, track_type, normalising)
			return dataset
		elif track_type == "alltracks":
			dataset = numpy.concatenate( [self.getBatch(True, noisy, track_type="goodtracks", normalising=normalising), self.getBatch(True, noisy, track_type="othertracks", normalising=normalising)], axis=0 )
			self.addDatasetToDictionary(dataset, noisy, track_type, normalising)
			return dataset
		else:
			raise ValueError
	
	def getX17Names(self, track_type="alltracks"):
		'''
		Return X17 track names in the same order as alltracks (i.e. DataLoader.getBatch(experimental=True, track_type='alltracks')). 
		'''

		names = []
		if track_type in ["goodtracks", "alltracks"]:	
			for (name, _) in self.loadX17Data("goodtracks", False):	names.append(name)
		if track_type in ["alltracks", "othertracks"]:
			for (name, _) in self.loadX17Data("othertracks", False):	names.append(name)
		return names



class Plotting:
	'''
	Class wrapping methods for plotting.
	'''

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
	def getPlotEventOneAxis(noise_event :numpy.ndarray, nonNN_event :numpy.ndarray, NN_event :numpy.ndarray, axis :int, event_name :str = None, cmap :str="Greys"):
		'''
		Plot projection of @noise_data, its NN and non-NN reconstruction (@NN_event and @nonNN_event, respectively) in specified @axis. 
		'''

		x_labels = ['z', 'z', 'y']
		y_labels = ['y', 'x', 'x']

		fig, ax = matplotlib.pyplot.subplots(3)
		
		cmap = matplotlib.pyplot.get_cmap(cmap)
		cmap.set_under('cyan')
		eps = 1e-8

		ax[0].set_title("Noisy " + event_name)
		ax[0].imshow(numpy.sum(noise_event, axis=axis), cmap=cmap, vmin=eps)
		ax[1].set_title("non-NN Reconstruction")
		ax[1].imshow(numpy.sum(nonNN_event, axis=axis), cmap=cmap, vmin=numpy.min([eps, numpy.max(nonNN_event)]))
		ax[2].set_title("NN Reconstruction")
		ax[2].imshow(numpy.sum(NN_event, axis=axis), cmap=cmap, vmin=numpy.min([eps, numpy.max(NN_event)]))
		for i in range(3):
			ax[i].set_xlabel(x_labels[axis])
			ax[i].set_ylabel(y_labels[axis])

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
		M = numpy.max(noise_event)
		if M == 0:	M = 1
		scaleSize = (lambda x: 100*x/M + 30) if are_data_experimental else (lambda x: 150*x + 50)
		sctr = Plotting.plot3DToAxis(noise_event, ax1, title, scaleSize)

		reconstr_event = modelAPI.evaluateSingleEvent( noise_event / (numpy.max(noise_event) if numpy.max(noise_event) != 0 else 1) )
		classificated_event = modelAPI.classify(reconstr_event)
		ax2 = fig.add_subplot(1, 2, 2, projection='3d')
		title = "Reconstruction and Threshold Classification\n"
		title += "by Model " + modelAPI.name
		Plotting.plot3DToAxis(classificated_event, ax2, title)

		cb = fig.colorbar(sctr, ax=[ax1, ax2], orientation="horizontal")
		cb.set_label("$E$")

		return fig, ax1, ax2

	def plot3DToAxis(event :numpy.ndarray, ax, title :str = "", scaleSize = lambda x: 150*x+50, z_cut = (0,200)):
		'''
		Create 3D plot of @event on specified matplotlib axis @ax.
		@scaleSize ... Function that scales scatter point size based on the corresponding value.
		@z_cut ... (z_low, z_max) limits of z axis.
		'''

		xs, ys, zs = event.nonzero()
		vals = numpy.array([event[xs[i],ys[i],zs[i]] for i in range(len(xs))])
		sctr = ax.scatter(xs, ys, zs, c=vals, cmap="plasma", marker="s", s=scaleSize(vals))
		ax.set_xlim(0, 11)
		ax.set_xlabel("$x$")
		ax.set_ylim(0, 13)
		ax.set_ylabel("$y$")
		ax.set_zlim(*z_cut)
		ax.set_zlabel("$z$")
		ax.set_title(title)
		ax.set_box_aspect((12, 14, 50))
		return sctr

	def animation3D(path :str, modelAPI :ModelWrapper, noise_event :numpy.ndarray, are_data_experimental :bool = None):
		'''
		Create and save gif of rotating 3D plot.
		'''

		fig, ax1, ax2 = Plotting.getPlot3D(modelAPI, noise_event, are_data_experimental)

		def run(i):	
			ax1.view_init(0,i,0)
			ax2.view_init(0,i,0)

		anim = matplotlib.animation.FuncAnimation(fig, func=run, frames=360, interval=20, blit=False)
		anim.save(path, fps=30, dpi=200, writer="pillow")

	def plotTileDistribution(data :numpy.ndarray, modelAPI :ModelWrapper):
		'''
		Create histogram of tile z coordinates distribution 
		'''

		classified = modelAPI.classify( modelAPI.evaluateBatch(data) )
		fig, ax = matplotlib.pyplot.subplots(1)
		counts_raw = numpy.sum(numpy.where(data>0.000001,1,0), axis=(0,1,2))
		counts_rec = numpy.sum(classified, axis=(0,1,2))
		ax.hist(x=[i for i in range(208)], bins=69, weights=counts_raw, label="Noisy", histtype="step")
		ax.hist(x=[i for i in range(208)], bins=69, weights=counts_rec, label="Reconstructed")
		ax.set_title("Distribution of X17 tile z coordinates after reconstruction by model " + modelAPI.name)
		ax.set_xlabel("z")
		ax.set_ylabel("#")
		ax.legend()
