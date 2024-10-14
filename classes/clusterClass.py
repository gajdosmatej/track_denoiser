import numpy

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
					already_visited[coord] = True
		self.energy_density = self.energy / self.effective_length'''

		self.energy = 0
		already_visited_xs, already_visited_ys, already_visited_ts = {}, {}, {}
		for coord in self.coords:
			self.energy += event[coord]
			if event[coord] != 0:
				already_visited_xs[coord[0]] = 1
				already_visited_ys[coord[1]] = 1
				already_visited_ts[coord[2]] = 1
		self.effective_length = numpy.sqrt( len(already_visited_xs)**2 + len(already_visited_ys)**2 + len(already_visited_ts)**2 / 10**2 )
		self.energy_density = self.energy / self.effective_length

		'''x1, y1, t1, x2, y2, t2 = None, None, None, None, None, None
		for coord in self.coords:
			self.energy += event[coord]
			if x1 is None or coord[0] < x1:	x1 = coord[0]
			if x2 is None or coord[0] > x2:	x2 = coord[0]
			if y1 is None or coord[1] < y1:	y1 = coord[1]
			if y2 is None or coord[1] > y2:	y2 = coord[1]
			if t1 is None or coord[2] < t1:	t1 = coord[2]
			if t2 is None or coord[2] > t2:	t2 = coord[2]
		self.effective_length = numpy.sqrt( (x2-x1+1)**2 + (y2-y1+1)**2 + (t2-t1+1)**2 / 10**2 )
		self.energy_density = self.energy / self.effective_length'''

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
