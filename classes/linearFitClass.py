import numpy
from classes import clusterClass

def fitLinePCA(coords, weights=None):
	'''
	Take a list of 3D coordinates and return PCA line fit in the form of (start, end, directionUnitVector, meanPoint).
	'''

	data = numpy.array(coords, dtype=numpy.float64)

	mean, scaled_covariance_matrix = None, None
	if weights is None:
		mean = [numpy.mean(data[:,i]) for i in range(3)]
		scaled_covariance_matrix = numpy.transpose(data-mean) @ (data-mean)
	else:
		num_samples = len(coords)
		mean = [ sum(weights[j] * data[j,i] for j in range(num_samples)) / sum(w for w in weights) for i in range(3) ]
		scaled_covariance_matrix = numpy.transpose(data-mean) @ numpy.diag(weights) @ (data-mean)

	_, _, V = numpy.linalg.svd(scaled_covariance_matrix)
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
	
	def getRatioOutliers(self):
		return len(self.getOutliers()) / len(self.coords)

	def getMissingTiles(self, noisy_event, in_temporal_space = False):
		missing_tiles = []
		xs_noisy, ys_noisy, zs_noisy = numpy.nonzero(noisy_event)
		if not in_temporal_space:
			ys_noisy = 5/6 * ys_noisy
			zs_noisy = zs_noisy / 15
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
	
	'''def changeCoordinates(self, event, in_temporal_space = False):
		xs, ys, zs = numpy.nonzero(event)
		if not in_temporal_space:	
			ys = 5/6 * ys
			zs = zs / 15
		self.coords = ( list(xs), list(ys), list(zs) )
		self.coords = list(zip(*self.coords))'''
	
	@staticmethod
	def changeCoordinates(coords, input_in_temporal_space):
		new_coords = []
		for coord in coords:
			if input_in_temporal_space:
				x, y, t = coord
				new_coords.append( (x, 5/6*y, 1/15*t) )
			else:
				x, y, z = coord
				new_coords.append( (x, 6/5*y, 15*z) )
		return new_coords
	
	@staticmethod
	def eventToCoords(event, in_temporal_space = True):
		xs, ys, zs = numpy.nonzero(event)
		coords = ( list(xs), list(ys), list(zs) )
		coords = list(zip(*coords))
		if not in_temporal_space:
			coords = LinearFit.changeCoordinates(coords, True)
		return coords

	def __init__(self, event, in_temporal_space = False, use_weights = False):
		xs, ys, zs = numpy.nonzero(event)
		
		weights = None
		if use_weights:
			weights = event[xs, ys, zs]

		coords = ( list(xs), list(ys), list(zs) )
		coords = list(zip(*coords))
		if not in_temporal_space:
			coords = LinearFit.changeCoordinates(coords, True)
		self.coords = coords

		self.start, self.end, self.mean, self.direction = fitLinePCA(self.coords, weights=weights)	# The conversion to (x,y,z) is already done
