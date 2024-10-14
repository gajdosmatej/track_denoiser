import numpy
from classes import clusterClass

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


class LinearFit:
	outlier_threshold = 1

	def getResiduum(self, coord):
		transformed = (coord[0] - self.mean[0], coord[1] - self.mean[1], coord[2] - self.mean[2])
		residuum = numpy.sqrt( transformed[0]**2 + transformed[1]**2 + transformed[2]**2 - (transformed[0]*self.direction[0] + transformed[1]*self.direction[1] + transformed[2]*self.direction[2])**2 + 1e-8)
		return residuum

	def getTotalResiduum(self):
		total_residuum = 0
		for coord in self.coords:
			total_residuum += self.getResiduum(coord)
		return total_residuum

	def getMeanResiduum(self):
		return self.getTotalResiduum() / len(self.coords)

	def getOutliers(self):
		outliers = []
		for coord in self.coords:
			residuum = self.getResiduum(coord)
			if residuum > LinearFit.outlier_threshold:
				outliers.append(coord)
		return outliers

	def getMissingTiles(self, noisy_event, in_temporal_space = True):
		missing_tiles = []
		xs_noisy, ys_noisy, zs_noisy = numpy.nonzero(noisy_event)
		if not in_temporal_space:	zs_noisy = zs_noisy / 10
		coords_noisy = ( list(xs_noisy), list(ys_noisy), list(zs_noisy) )
		coords_noisy = zip(*coords_noisy)
		for coord in coords_noisy:
			if coord not in self.coords:
				residuum = self.getResiduum(coord)
				if residuum <= LinearFit.outlier_threshold:
					missing_tiles.append(coord)
		return missing_tiles

	def getColinearTiles(self):
		colin = []
		for coord in self.coords:
			residuum = self.getResiduum(coord)
			if residuum <= LinearFit.outlier_threshold:
				colin.append(coord)
		return colin

	def __init__(self, reconstructed_event, in_temporal_space = True):
		xs, ys, zs = numpy.nonzero(reconstructed_event)
		if not in_temporal_space:	zs = zs / 10
		self.coords = ( list(xs), list(ys), list(zs) )
		self.coords = list(zip(*self.coords))
		cluster = clusterClass.Cluster(self.coords)

		self.start, self.end, self.mean, self.direction = fitLinePCA(cluster, in_temporal_space=True)	# The conversion to (x,y,z) is already done
