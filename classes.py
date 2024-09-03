import numpy
from tensorflow import keras
import tensorflow
import matplotlib.pyplot
import matplotlib.animation
import keras.utils
import copy
import os

VERBOSE = False
SATURATION_THRESHOLD = 800

def normalise(event :numpy.ndarray):
	'''
	Linearly map @event to [0,1] interval.
	'''

	M = numpy.max(event)
	if M == 0:	return event
	return event / M

def fitLinePCA(cluster, in_temporal_space = True):
	'''
	Take a list of 3D coordinates and return PCA line fit in the form of (start, end, directionUnitVector, meanPoint).
	@in_temporal_space (bool) ... If False, convert @cluster from (x,y,t) space to (x,y,z)
	'''

	coords = cluster.coords
	if not in_temporal_space:
		coords = [(x,y,t/10) for (x,y,t) in coords]
	
	data = numpy.array(coords, dtype=numpy.float64)
	mean = [numpy.mean(data[:,i]) for i in range(3)]

	#scaled_covariance_matrix = numpy.transpose(data) @ data
	#U, D, V = numpy.linalg.svd(scaled_covariance_matrix)

	#From SVD it follows that for data = UDV*, we have Var = data* data = VD*U* UDV* = VD^2 V*, so the right factor of @data and @scaled_covariance_matrix is the same
	_, _, V = numpy.linalg.svd(data - mean)
	direction = V[0]

	a, b = numpy.min(data[:,2]), numpy.max(data[:,2])

	t1, t2 = (a-mean[2]) / direction[2], (b-mean[2]) / direction[2]
	return (mean + t1*direction, mean + t2*direction, mean, direction)

def getTotalNonlinearityResiduum(cluster, line_direction, line_mean):
	'''
	Take list of 3D coordinates @cluster and their line fit specified by @line_direction and @line_mean, 
	return the sum of perpendicular distances of @cluster points from the line fit.
	'''
	
	residuum = 0
	data = numpy.array(cluster.coords, dtype=numpy.float64) - line_mean
	for i in range(data.shape[0]):
		coord = data[i]
		residuum += numpy.sqrt( coord[0]**2 + coord[1]**2 + coord[2]**2 - (coord[0]*line_direction[0] + coord[1]*line_direction[1] + coord[2]*line_direction[2])**2 + 1e-8)
	return residuum

def removeChimneys(event):
	'''
	Remove noisy waveforms in cases where the waveform crosses the saturation threshold.
	'''
	xs, ys, zs = numpy.where(event > SATURATION_THRESHOLD)
	pads = {}
	for i in range(xs.shape[0]):
		x, y, z = xs[i], ys[i], zs[i]
		if (x,y) not in pads:	pads[(x,y)] = z
		else:	pads[(x,y)] = min(z, pads[(x,y)])	# store the smallest z coordinate where the crossing occured
	
	for (x,y) in pads:
		z = pads[(x,y)]
		#event[x,y,(z+1):] = 0
		waveform = event[x,y,z:]
		waveform = waveform[waveform>0]
		length = waveform.shape[0]
		if length < 20:	continue	# The chimney is small, we let it be
		baseline = numpy.min(waveform)
		variance = numpy.var(waveform)
		#threshold = variance/length
		threshold = numpy.sqrt(variance)
		#threshold=0
		#threshold=numpy.infty
		event[x,y,z:] = numpy.where(event[x,y,z:]-baseline > threshold, event[x,y,z:], 0)
		


AVGPOOL = keras.layers.AveragePooling3D((1,1,8))
def customBCE(target, pred):
	'''
	Custom loss function, returns BCE(@target, @pred) + BCE( avgDownSample(1,1,8)(@target), avgDownSample(1,1,8)(@pred) ).
	'''

	pred_pool = AVGPOOL(pred)
	target_pool = AVGPOOL(target)
	#return tensorflow.math.reduce_mean( keras.losses.binary_crossentropy(target,pred) )
	return ( tensorflow.math.reduce_mean( keras.losses.binary_crossentropy(target_pool, pred_pool) )
			+ tensorflow.math.reduce_mean( keras.losses.binary_crossentropy(target, pred) ) )

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
	min_length = 4
	max_nonlinearity = 1
	z_scale_factor = 0.1

	def __init__(self, coords):
		self.coords = coords
		self.num_tiles = len(coords)
		self.corners = []
		self.pad_length = self.getPadLength()
		#self.neighbour_coef = self.getNeighbourCoefficient()
		#self.findCorners()
		self.tests = {}

	def __str__(self):
		result = "L = " + str(self.pad_length)
		result += "\nneigh_coef = " + str(self.neighbour_coef)
		result += "\nE_density = " + str(self.energy_density)
		#result += "\n#Passed tests = " + str(self.getPassedTestsNum())
		return result
	

	def getPadLength(self):
		pad_len = 0
		visited_xy = {}
		for coord in self.coords:
			xy = (coord[0], coord[1])
			if xy not in visited_xy:
				pad_len += 1
				visited_xy[xy] = 1
			else:
				pad_len += self.z_scale_factor
		return pad_len

	@staticmethod
	def union(c, d):
		return Cluster(c.coords + d.coords)

	'''def findCorners(self):
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
			if coordsFormClique(neighbours):	self.corners.append(coord)'''
	
	#OSBOLETE
	'''def linify(self):
		if len(self.corners) < 2:
			if VERBOSE:	print("WARNING [Cluster.linify]: Less that two corner tiles found, skipping")
			return
		corner_left, corner_right = self.corners[0], self.corners[0]
		for c in self.corners:
			if c[2] < corner_left[2]:	corner_left = c
			if c[2] > corner_right[2]:	corner_right = c

		return (corner_left, corner_right)'''

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
			cluster = [(remaining_tracks[0][0], remaining_tracks[1][0], remaining_tracks[2][0])]
			event[cluster[0]] = 0
			stack = [cluster[0]]
			while stack != []:
				coord = stack.pop()
				for neighbour in Cluster.neighbourhood(coord, 1, 1, 8):
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
		
		return ( num_in_zone/self.num_tiles > self.active_zone_threshold )
	

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

	#OBSOLETE?
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
		return 100<=z<=160
		#return (98 <= z <=126 and 45/2*x-82 <= z) or (126 < z <= 143 and 3/17*(z-126) <= x <= 2/45*(z+82))

	#TODO OPTIMISE
	def getTensor(self, event=None):
		result = numpy.zeros((12,14,208))
		for coord in self.coords:
			result[coord] = (1 if event is None else event[coord])
		return result
	
	def setEnergy(self, event):
		'''self.energy = 0
		self.effective_length = 0
		already_visited = {}
		for coord in self.coords:
			self.energy += event[coord]
			if event[coord] != 0:
				if coord in already_visited:
					self.effective_length += 0.1
				else:
					self.effective_length += 1
					already_visited[coord] = True'''
		
		self.energy = 0
		self.effective_length = 0
		already_visited = {}
		for coord in self.coords:
			self.energy += event[coord]
			if event[coord] != 0:
				if coord in already_visited:
					self.effective_length += 0.1
				else:
					self.effective_length += 1
					already_visited[coord] = True
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


#PROBABLY OBSOLETE
class FinalProcedure:
	epsilon = 1e-2

	def reconstructByDefaultEnsemble(self, noisy_data, path_to_models = "./models"):
		if path_to_models[-1] != "/":	path_to_models += "/"

		model_1 = ModelWrapper( keras.models.load_model(path_to_models + "3D/M1", compile=False), "M1" )
		self.reconstruction = model_1.evaluateBatch(noisy_data)
		#self.reconstruction = model_1.evaluateBatch(noisy_data) * noisy_data
		#self.reconstruction = numpy.power( noisy_data, (1-model_1.evaluateBatch(noisy_data)) / (model_1.evaluateBatch(noisy_data)+1e-7) ) 

		self.classified = numpy.where(self.reconstruction > self.epsilon, self.reconstruction, 0)

		'''model_1 = ModelWrapper( keras.models.load_model(path_to_models + "3D/MEW_basic_conv_information_100epochs"), "1" )
		#model_2 = ModelWrapper( keras.models.load_model(path_to_models + "3D/small/model"), "Spatial" )
		model_2 = ModelWrapper( keras.models.load_model(path_to_models + "3D/NEWNEWNEW_waveform_longer"), "2" )
		self.reconstruction_1 = model_1.evaluateBatch(noisy_data)
		self.reconstruction_2 = model_2.evaluateBatch(noisy_data)
		#self.reconstruction = self.reconstruction_1 * self.reconstruction_2

		def getConfidencyCoefficients(recs):
			stability_eps = 1e-7
			#lambdas = [numpy.abs( numpy.log( (rec+stability_eps) /self.epsilon) )**2 + stability_eps for rec in recs]
			lambdas = [ ( numpy.log( (rec+stability_eps) / (1-rec+stability_eps)) - numpy.log(self.epsilon/(1-self.epsilon)) )**2 + stability_eps for rec in recs]	#invert sigmoid, measure square distance
			N = 1/numpy.sum(lambdas, axis=0)
			return N*lambdas

		#lambdas = getConfidencyCoefficients([self.reconstruction_1, self.reconstruction_2])
		self.reconstruction = self.reconstruction_1
		#self.reconstruction = lambdas[0]*self.reconstruction_1 + lambdas[1]*self.reconstruction_2
		self.classified = numpy.where(self.reconstruction > self.epsilon, self.reconstruction, 0)'''

	def selfDefragmentation(self):
		original_clusters = []
		defragmentation_coef = 0
		cluster_count = 0

		for i in range(self.classified.shape[0]):
			clusters_model_1 = Cluster.clusterise( numpy.where(self.reconstruction_1[i] > self.epsilon, 1, 0) )
			#clusters_model_2 = Cluster.clusterise( numpy.where(self.reconstruction_2[i] > self.epsilon, 1, 0) )
			clusters_product = Cluster.clusterise( self.classified[i] )
			
			cluster_count_before = len(clusters_product)
			clusters_product, _, _, _ = Cluster.defragment(clusters_product, clusters_model_1)
			#clusters_product, _, _, _ = Cluster.defragment(clusters_product, clusters_model_2)
   
			original_clusters.append(clusters_product)
			defragmentation_coef += cluster_count_before - len(clusters_product)
			cluster_count += len(clusters_product)

		print("Defragmentation occured", defragmentation_coef, "times (final number of clusters " + str(cluster_count) + ")")
		self.clusters = original_clusters

	def applyBasicCuts(self, noisy_with_energy_dataset):
		passing_clusters = []
		cluster_count = 0

		for i in range(self.classified.shape[0]):
			current_passing_clusters = []
			for cluster in self.clusters[i]:
				if cluster.pad_length >= Cluster.min_length and cluster.testActiveZone():
					cluster.setEnergy(noisy_with_energy_dataset[i])
					if cluster.energy_density >= Cluster.min_energy_density:
						current_passing_clusters.append(cluster)
			passing_clusters.append(current_passing_clusters)
			cluster_count += len(current_passing_clusters)

		self.clusters = passing_clusters
		print("Current number of clusters after zone, L and E/L cuts is", cluster_count)
	
	def applyLinearityCut(self):
		passing_clusters = []
		cluster_count = 0

		for i in range(self.classified.shape[0]):
			current_passing_clusters = []
			for cluster in self.clusters[i]:
				line_start, line_end, line_mean, line_direction = fitLinePCA(cluster)
				total_residuum = getTotalNonlinearityResiduum(cluster, line_direction, line_mean)
				cluster.nonlinearity = total_residuum / cluster.num_tiles

				if total_residuum / cluster.num_tiles < Cluster.max_nonlinearity:
						current_passing_clusters.append(cluster)
			passing_clusters.append(current_passing_clusters)
			cluster_count += len(current_passing_clusters)

		self.clusters = passing_clusters
		print("Final number of clusters after linearity cut is", cluster_count)

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

#PROBABLY OBSOLETE
class DataLoader_OLDDIRSTRUCTURE:
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

	def dumpData(self, data, names, dir_name = "dump/", round_digits=0):
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
				dump_str += str(x) + ", " + str(y) + ", " + str(z) + ", " + str(round(E, round_digits)) + "\n"
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


class DataLoader:

	nice_track_indices = [0, 2, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 20, 27, 29, 32, 34, 36, 43, 44, 46, 50, 57, 59, 60, 67, 72, 73, 74, 78, 83, 85, 87, 91, 92, 93, 95, 96, 97, 99, 101, 103, 104, 106, 108, 113, 114, 115, 118, 119, 124, 129, 135, 139, 141, 142, 150, 152, 154, 158, 159, 160, 162, 163, 174, 175, 177, 184, 194, 196, 200, 201, 202, 206, 207, 213, 214, 217, 223, 232, 234, 235, 238, 245, 248, 249, 250, 251, 257, 259, 268, 270, 272, 276, 277, 278, 282, 284, 287, 290, 297, 298, 299, 300, 302, 303, 306, 310, 311, 316, 321, 325, 326, 330, 332, 333, 336, 339, 340, 349, 358, 363, 369, 372, 373, 374, 383, 384, 389, 390, 394, 396, 397, 398, 400, 403, 404, 410, 414, 420, 423, 424, 425, 432, 439, 446, 454, 455, 459, 460, 461, 464, 466, 468, 469, 473, 486, 490, 491, 493, 499, 503, 505, 508, 511, 512, 513, 515, 517, 522, 523, 528, 531, 539, 540, 541, 554, 555, 557, 561, 565, 569, 571, 572, 576, 579, 580, 581, 584, 585, 587, 589, 590, 591, 596, 597, 600, 607, 608, 609, 612, 614, 615, 616, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 632, 634, 635, 636, 639, 640]

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

	def dumpData(self, data, names, dir_name = "dump/", round_digits=0):
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
				dump_str += str(x) + ", " + str(y) + ", " + str(z) + ", " + str(round(E, round_digits)) + "\n"
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
			E = float(line)
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

	def getEventFromName(self, name: str, noisy :bool, preprocessed :bool = True):
		'''
		Return X17 event from its @name.
		'''

		return self.getBatch(True, noisy, preprocessed=preprocessed)[ self.name_index_dict[name] ]
		#names, events = self.getX17Names(), self.getBatch(True, noisy, track_type="alltracks", normalising=normalising)
		#return [event for (n, event) in zip(names, events) if n == name][0]

	def loadX17Data(self, noisy :bool):
		'''
		Parse X17 data from txt file into list of tuples (event name, 3D event array).
		'''

		def parseX17Line(line :str):
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

		path = self.path + "x17/" + ("noisy/" if noisy else "clean/")

		for f_int in sorted( [int(val[5:-4]) for val in os.listdir(path)] ):
			f_name = "track" + str(f_int) + ".txt"
			file = open(path + f_name, 'r')
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

	#def getValidationData(self):
		'''
		Return tuple (X17_noisy, X17_clean), which is convenient format for Keras Model.fit validation_data argument.
		'''

		'''noisy, clean = self.getBatch(True, True, track_type="goodtracks"), self.getBatch(True, False, track_type="goodtracks")
		noisy = numpy.reshape(noisy, (*noisy.shape, 1))
		clean = numpy.reshape(clean, (*clean.shape, 1))
		return (noisy, clean)'''

	def datasetExists(self, noisy :bool, preprocessed :bool):
		return (noisy, preprocessed) in self.existing_experimental_datasets
	
	def getDatasetFromDictionary(self, noisy :bool, preprocessed :bool):
		return self.existing_experimental_datasets[(noisy, preprocessed)]

	def addDatasetToDictionary(self, dataset :numpy.array, noisy :bool, preprocessed :bool):
		self.existing_experimental_datasets[(noisy, preprocessed)] = dataset

	def getBatch(self, experimental :bool = True, noisy :bool = True, file_id :int = 0, preprocessed :bool = True):
		'''
		Get one data batch as numpy array.
		@experimental: Whether simulated or X17 data should be used.
		@noisy: Whether noisy or clean data should be used.
		@file_id: File ID, from which the simulated batch should be loaded (useless for @experimental = True).
		@preprocessed: Whether chimneys should be removed and X17 data should be mapped to [0,1] first (default True, good for evaluating by models, but energy information is lost).
		'''
		
		if not experimental:
			return numpy.load(self.path + "simulated/" + ("noisy/" if noisy else "clean/") + str(file_id) + ".npy")
		
		if self.datasetExists(noisy, preprocessed):
			return self.getDatasetFromDictionary(noisy, preprocessed)
	
		x17_data = []
		for _, event in self.loadX17Data(noisy):
			if preprocessed:
				if noisy:	removeChimneys(event)
				x17_data.append( normalise(event) )
			else:
				x17_data.append( event )
		dataset = numpy.array(x17_data)
		self.addDatasetToDictionary(dataset, noisy, preprocessed)
		return dataset

	
	def getX17Names(self):
		'''
		Return X17 track names. 
		'''

		names = []
		for (name, _) in self.loadX17Data(False):	names.append(name)
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
		ax.set_xlim(0, 110)
		ax.set_xlabel("$x$")
		ax.set_ylim(0, 130)
		ax.set_ylabel("$y$")
		ax.set_zlim(*z_cut)
		ax.set_zlabel("$z$")
		ax.set_title(title)
		#ax.set_box_aspect((12, 14, 50))
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
