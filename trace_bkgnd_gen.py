import numpy
import matplotlib.pyplot
import time
import os

def getProjection(space :numpy.ndarray, axis :int):
	return numpy.sum(space, axis)

def normalise(arr :numpy.ndarray):
	'''Linearly maps values of input array to [0,1]'''
	return arr/numpy.max(arr)


class Generator:
	DIMENSIONS = (12, 14, 208)
	CURVATURE_SIGMA = 0.05	#sigma of normal distribution, which updateDirection samples R3 vector from
	DIRECTION_NORMS = numpy.array([0.5,0.5,1.0])	#factors which multiply each coordinate of direction unit vector (main aim is to increase velocity in Z direction in order to compensate dim_Z >> dim_X, dim_Y)
	SIGNAL_ENERGY = 1	#signal deposits this value in every point of lattice which it visits
	NOISE_ENERGY = 1.3	#the total sum of values that noise can deposit in visited points of lattice
	ENERGY_DEPOSITION_COEFFITIENT = 0.4	#the coefficient of the exponential distribution, which noise samples a random value from and deposits it in the visited point of lattice
	SIGNAL_CHANGE_DIRECTION_PROBABILITY = 0.1	#the signal probability that updateDirection is called in a step
	NOISE_CHANGE_DIRECTION_PROBABILITY = 0.8	#the noise probability that updateDirection is called in a step
	SIGNAL_STOP_PROBABILITY = 0.05	#the probability increment that the signal track stops
	NOISE_TRACKS_NUM_RANGE = (15,31)	#the minimal and maximal number of noise tracks
	DATA_DIR_PATH = "./data/simulated/"
	NOISE_MEAN_ENERGY = 0.5
	NOISE_SIGMA_ENERGY = 0.3

	def __init__(self):
		self.initialise()

	def initialise(self):
		self.space = numpy.zeros(self.DIMENSIONS)

	def updateDirection(self, direction :numpy.ndarray):
		'''Randomly updates the direction vector.'''
		direction += numpy.random.normal(loc=0, scale=self.CURVATURE_SIGMA, size=3)
		direction /= numpy.linalg.norm(direction)
		direction *= self.DIRECTION_NORMS

	def discretise(self, coord):
		return (round(coord[0]), round(coord[1]), round(coord[2]))

	def isCoordOutOfBounds(self, coord):
		return ( (0,0,0) > coord ).any() or (coord >= numpy.array(self.DIMENSIONS)-0.5 ).any()

	def getRandomBoundaryStart(self):
		side = numpy.random.randint(0,6)
		position, direction = None, None
		if side == 0:
			position = numpy.array( [0, numpy.random.randint(0, self.DIMENSIONS[1]), numpy.random.randint(0, self.DIMENSIONS[2])], dtype=float )
			direction = numpy.array([self.DIRECTION_NORMS[0],0.,0.])
		elif side == 1:
			position = numpy.array( [self.DIMENSIONS[0]-1, numpy.random.randint(0, self.DIMENSIONS[1]), numpy.random.randint(0, self.DIMENSIONS[2])], dtype=float )
			direction = numpy.array([-self.DIRECTION_NORMS[0],0.,0.])
		elif side == 2:
			position = numpy.array( [numpy.random.randint(0, self.DIMENSIONS[0]), 0, numpy.random.randint(0, self.DIMENSIONS[2])], dtype=float )
			direction = numpy.array([0.,self.DIRECTION_NORMS[1],0.])
		elif side == 3:
			position = numpy.array( [numpy.random.randint(0, self.DIMENSIONS[0]), self.DIMENSIONS[1]-1, numpy.random.randint(0, self.DIMENSIONS[2])], dtype=float )
			direction = numpy.array([0.,-self.DIRECTION_NORMS[1],0.])
		elif side == 4:
			position = numpy.array( [numpy.random.randint(0, self.DIMENSIONS[0]), numpy.random.randint(0, self.DIMENSIONS[1]), 0], dtype=float )
			direction = numpy.array([0.,0.,self.DIRECTION_NORMS[2]])
		else:
			position = numpy.array( [numpy.random.randint(0, self.DIMENSIONS[0]), numpy.random.randint(0, self.DIMENSIONS[1]), self.DIMENSIONS[2]-1], dtype=float )
			direction = numpy.array([0.,0.,-self.DIRECTION_NORMS[2]])
		return (position, direction)

	def sampleInitSignalDirection(self):
		'''Samples signal direction vector with bias in z axis.'''
		#Samples signal direction vector uniformly in spherical coordinates, so that the z axis (zenith) direction is biased.'''
		'''azimuthal_angle = numpy.random.random() * 2*numpy.pi
		polar_angle = numpy.random.random() * numpy.pi
		x = numpy.sin(polar_angle) * numpy.cos(azimuthal_angle)
		y = numpy.sin(polar_angle) * numpy.sin(azimuthal_angle)
		z = numpy.cos(polar_angle)
		return numpy.array( [x,y,z] )*self.DIRECTION_NORMS '''
		return normalise( numpy.array( [numpy.random.random()-0.5, numpy.random.random()-0.5, 1.0] ))

	def OLDaddSignal(self):
		'''Adds one signal track into the input 3D array space.'''
		position = numpy.array(self.DIMENSIONS) * numpy.random.normal(loc=0.5, scale=0.1, size=3)	#the track begins in the middle of the space
		direction = self.sampleInitSignalDirection()
		cumulation_stop_probability = 0
		while not self.isCoordOutOfBounds(position) and numpy.random.random() > cumulation_stop_probability:	#the track propagades until it moves out of the space boundaries
			self.space[self.discretise(position)] = 1
			if numpy.random.random() < self.SIGNAL_CHANGE_DIRECTION_PROBABILITY:	self.updateDirection(direction)
			position += direction
			cumulation_stop_probability += self.SIGNAL_STOP_PROBABILITY
		coord = numpy.nonzero(self.space)
		self.space[coord] = numpy.clip( numpy.random.normal(self.SIGNAL_ENERGY, self.SIGNAL_ENERGY/3, coord[0].shape), 0, None)

	def OLDaddNoise(self):
		'''Adds several noise tracks into the input 3D array space.'''
		num_of_traces = 20
		for _ in range(num_of_traces):
			position = (numpy.array(self.DIMENSIONS) - 1) * numpy.random.random(size=3)
			direction = numpy.zeros(3)
			self.updateDirection(direction)
			energy = self.NOISE_ENERGY
			while energy > 0:	#the particle moves until it loses all the energy or it moves out of the space
				lost_energy = numpy.clip( numpy.random.exponential(self.ENERGY_DEPOSITION_COEFFITIENT), 0, 1)
				self.space[self.discretise(position)] += lost_energy
				energy -= lost_energy
				if numpy.random.random() < self.NOISE_CHANGE_DIRECTION_PROBABILITY:	self.updateDirection(direction)
				position += direction
				if self.isCoordOutOfBounds(position):	break

	def addSignal(self):
		'''Adds one signal track into the input 3D array space.'''
		position = numpy.array(self.DIMENSIONS) * numpy.random.normal(loc=0.5, scale=0.1, size=3)	#the track begins in the middle of the space
		direction = self.sampleInitSignalDirection()
		num_steps = int(numpy.random.normal(loc=50, scale=1))

		for _ in range(num_steps):
			if self.isCoordOutOfBounds(position):	break
			self.space[self.discretise(position)] += 1
			if numpy.random.random() < self.SIGNAL_CHANGE_DIRECTION_PROBABILITY:	self.updateDirection(direction)
			position += direction
		coord = numpy.nonzero(self.space)
		self.space[coord] *= numpy.clip( numpy.random.normal(self.SIGNAL_ENERGY, self.SIGNAL_ENERGY/3, coord[0].shape), 0, None)

	def addNoise(self):
		'''Adds several noise tracks into the input 3D array space.'''
		num_of_traces = numpy.random.randint( *self.NOISE_TRACKS_NUM_RANGE )
		for _ in range(num_of_traces):
			position = (numpy.array(self.DIMENSIONS) - 1) * numpy.random.random(size=3)
			self.space[self.discretise(position)] += numpy.clip( numpy.random.normal(loc=self.NOISE_MEAN_ENERGY, scale=self.NOISE_SIGMA_ENERGY), 0, None)

	def genAndDumpData(self, iterations :int):
		'''Generates space 3D array with one signal and several noises in each iteration and saves the projections of the clean and noised data.'''
		noise_names = ["_noise_zy", "_noise_zx", "_noise_yx"]
		signal_names = ["_signal_zy", "_signal_zx", "_signal_yx"]
		file_size = 20000

		increment = len(os.listdir(self.DATA_DIR_PATH)) // 6

		file_num = iterations // file_size
		for file_i in range(file_num):
			data_noise, data_signal = [[], [], []], [[], [], []]
			print("Generating data...")
			for i in range(file_size):
				if i % 1000 == 0:	print("{:,}".format(file_i *file_size + i), "/", "{:,}".format(iterations))
				self.initialise()
				self.addSignal()
				for k in range(3):
					data_signal[k].append(getProjection(self.space, k))
				self.addNoise()

				for k in range(3):
					data_noise[k].append(getProjection(self.space, k))
			print("Saving batch...")
			for k in range(3):
				data_noise[k] = normalise(data_noise[k])
				data_signal[k] = normalise(data_signal[k])
				numpy.save(self.DATA_DIR_PATH + str(increment+file_i) + noise_names[k], data_noise[k])
				numpy.save(self.DATA_DIR_PATH + str(increment+file_i) + signal_names[k], data_signal[k])

class Support:
	@staticmethod
	def showProjection(space :numpy.ndarray, axis :int):
		fig, ax = matplotlib.pyplot.subplots(1)
		ax.imshow( getProjection(space, axis), cmap='gray', vmin=0)
		matplotlib.pyplot.show()

	@staticmethod
	def showProjections(space :numpy.ndarray, indices :int):
		'''Shows chosen plots (by indices) of the space projections into the xy, yz and zx planes.'''
		if len(indices) == 1:
			Support.showProjection(space, indices[0])
			return
		num_planes = len(indices)
		labels = [("z", "y"), ("z", "x"), ("y", "x")]
		fig, ax = matplotlib.pyplot.subplots(num_planes,1)
		for i in range(num_planes):
			ax[i].imshow(getProjection(space, indices[i]), cmap='gray', vmin=0)
			ax[i].set_xlabel(labels[i][0])
			ax[i].set_ylabel(labels[i][1])
		matplotlib.pyplot.show()

	@staticmethod
	def showRandomDataFromFile(num_pics : int):
		'''Shows some random data from datafile.'''
		global datapath
		zy_projections = numpy.load(datapath + "data_noise_zy.npy")
		zx_projections = numpy.load(datapath + "data_noise_zx.npy")
		yx_projections = numpy.load(datapath + "data_noise_yx.npy")
		zy_projections_sgnl = numpy.load(datapath + "data_signal_zy.npy")
		zx_projections_sgnl = numpy.load(datapath + "data_signal_zx.npy")
		yx_projections_sgnl = numpy.load(datapath + "data_signal_yx.npy")

		size = numpy.shape(zy_projections)[0]

		for _ in range(num_pics):
			index = numpy.random.randint(0, size-1)
			fig, ax = matplotlib.pyplot.subplots(3,2)
			ax[0,0].imshow(zy_projections_sgnl[index], cmap='gray')
			ax[0,0].set_xlabel("z")
			ax[0,0].set_ylabel("y")
			ax[1,0].imshow(zx_projections_sgnl[index], cmap='gray')
			ax[1,0].set_xlabel("z")
			ax[1,0].set_ylabel("x")
			ax[2,0].imshow(yx_projections_sgnl[index], cmap='gray')
			ax[2,0].set_xlabel("y")
			ax[2,0].set_ylabel("x")

			ax[0,1].imshow(zy_projections[index], cmap='gray')
			ax[0,1].set_xlabel("z")
			ax[0,1].set_ylabel("y")
			ax[1,1].imshow(zx_projections[index], cmap='gray')
			ax[1,1].set_xlabel("z")
			ax[1,1].set_ylabel("x")
			ax[2,1].imshow(yx_projections[index], cmap='gray')
			ax[2,1].set_xlabel("y")
			ax[2,1].set_ylabel("x")
			matplotlib.pyplot.show()

	@staticmethod
	def show3D(space :numpy.ndarray):
		'''Shows 3D scatter plot of the input space.'''
		xs, ys, zs = space.nonzero()
		vals = numpy.array([space[xs[i],ys[i],zs[i]] for i in range(len(xs))])
		fig = matplotlib.pyplot.figure()
		ax = fig.add_subplot(projection='3d')
		sctr = ax.scatter(xs, ys, zs, c=vals, cmap="plasma")
		ax.set_xlim(0, 11)
		ax.set_xlabel("$x$")
		ax.set_ylim(0, 13)
		ax.set_ylabel("$y$")
		ax.set_zlim(0, 200)
		ax.set_zlabel("$z$")
		cb = fig.colorbar(sctr, ax=ax)
		cb.set_label("$E$")
		matplotlib.pyplot.show()


#generator = Generator()
#generator.genAndDumpData(int(5e6))
#showRandomData()


generator = Generator()
while True:
	generator.addSignal()
	Support.show3D(generator.space)
	generator.addNoise()
	Support.show3D(generator.space)
	Support.showProjections(generator.space, [0,1,2])
	if input() == "q":	break
	else:	generator.initialise()
