import numpy
import matplotlib.pyplot
import time
import os
import sys

def getProjection(space :numpy.ndarray, axis :int):
	return numpy.sum(space, axis)

def normalise(arr :numpy.ndarray):
	'''Linearly maps values of input array to [0,1]'''
	M =  numpy.max(arr)
	return arr / (1 if M == 0 else M)


class Generator:
	DIMENSIONS = (12, 14, 208)
	CURVATURE_SIGMA = 0.05	#sigma of normal distribution, which updateDirection samples R3 vector from
	DIRECTION_NORMS = numpy.array([0.1,0.1,1.])	#factors which multiply each coordinate of direction unit vector (main aim is to increase velocity in Z direction in order to compensate dim_Z >> dim_X, dim_Y)
	SIGNAL_CHANGE_DIRECTION_PROBABILITY = 0.1	#the signal probability that updateDirection is called in a step
	NOISE_TRACKS_NUM_RANGE = (15,30)	#the minimal and maximal number of noise tracks
	DATA_DIR_PATH = "./data"
	NOISE_MEAN_ENERGY = 0.5
	NOISE_SIGMA_ENERGY = 1.
	MAX_NOISE_STEPS = 8

	def __init__(self):
		self.initialise()

	def initialise(self):
		self.space = numpy.zeros(self.DIMENSIONS, dtype=numpy.float16)

	def updateDirection(self, direction :numpy.ndarray):
		'''Randomly updates the direction vector.'''
		direction += numpy.random.normal(loc=0, scale=self.CURVATURE_SIGMA, size=3)
		direction /= numpy.linalg.norm(direction)
		direction *= self.DIRECTION_NORMS

	def discretise(self, coord):
		'''Find discretised space node corresponding to given spatial coordinates.'''
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

	def sampleInitDirection(self):
		'''Samples direction vector uniformly from a unit sphere.'''
		vect = numpy.random.normal(loc=0,scale=1,size=3)
		vect /= numpy.linalg.norm(vect)
		vect *= self.DIRECTION_NORMS
		return vect

	def addSignal(self):
		'''Adds one signal track into the input 3D array space.'''
		
		history = []
		last_coords = None
		position = numpy.array(self.DIMENSIONS) * numpy.random.uniform(0.2, 0.8, size=3)
		direction = self.sampleInitDirection()
		num_steps = int(numpy.random.normal(loc=60, scale=2))

		for _ in range(num_steps):
			if self.isCoordOutOfBounds(position):	break
			
			coords = self.discretise(position)
			if coords != last_coords:
				history.append(coords)
				last_coords = coords
				self.space[coords] = 1
			#if numpy.random.random() < self.SIGNAL_CHANGE_DIRECTION_PROBABILITY:	self.updateDirection(direction)
			position += direction
		
		if len(history) < 15:	#too short track
			self.initialise()
			return self.addSignal()
		return history

	def addNoise(self):
		'''Adds several noise tracks into the input 3D array space.'''
		num_of_traces = numpy.random.randint( *self.NOISE_TRACKS_NUM_RANGE )
		for _ in range(num_of_traces):
			position = (numpy.array(self.DIMENSIONS) - 1) * numpy.random.uniform(0, 1, size=3)
			steps = numpy.random.randint(1, self.MAX_NOISE_STEPS)
			direction = self.sampleInitDirection()
			for _ in range(steps):
				self.space[self.discretise(position)] += numpy.clip( numpy.random.normal(loc=self.NOISE_MEAN_ENERGY, scale=self.NOISE_SIGMA_ENERGY), 0, None)
				position += direction
				if self.isCoordOutOfBounds(position):	break
				self.updateDirection(direction)


	def energise(self, history :list):
		#E_prev = numpy.random.uniform(0.5,1.)
		#E = E_prev + numpy.random.normal(0,0.1)
		E = numpy.random.uniform(1., 1.5)
		for coord in history:
			if numpy.random.random() < 0.1:	E = numpy.random.uniform(0., 1.5)	#discontinuity
			self.space[coord] = E
			E += numpy.random.uniform(-0.4, 0.4)
			if E < 0:	E = (numpy.random.uniform(0, 0.1) if numpy.random.random() < 0.5 else 0)
			#E_prev, E = E, E + (E-E_prev) + numpy.random.normal(0,0.1)
			#if E < 0:	E = 0

	#OBSOLETE
	def genAndDumpData(self, iterations :int):
		'''Generates space 3D array with one signal and several noises in each iteration and saves the projections of the clean and noised data.'''
		noise_names = ["_noise_zy", "_noise_zx", "_noise_yx"]
		signal_names = ["_signal_zy", "_signal_zx", "_signal_yx"]
		file_size = 20000

		increment = len(os.listdir(self.DATA_DIR_PATH + "2D")) // 6	#some datafiles might be already in the directory, this ensures they will not be overwritten

		file_num = iterations // file_size
		for file_i in range(file_num):
			data_noise, data_signal = [[], [], []], [[], [], []]
			print("Generating data...")
			for i in range(file_size):
				if i % 1000 == 0:	print("{:,}".format(file_i *file_size + i), "/", "{:,}".format(iterations))
				self.initialise()
				self.addSignal()
				for k in range(3):
					data_signal[k].append( normalise(getProjection(self.space, k)) )
				self.addNoise()

				for k in range(3):
					data_noise[k].append( normalise(getProjection(self.space, k)) )
			print("Saving batch...")
			for k in range(3):
				numpy.save(self.DATA_DIR_PATH + "2D/" + str(increment+file_i) + noise_names[k], data_noise[k])
				numpy.save(self.DATA_DIR_PATH + "2D/" + str(increment+file_i) + signal_names[k], data_signal[k])

	def genAndDumpData3D(self, iterations :int):
		'''Generates space 3D array with one signal and several noises in each iteration and saves the clean and noised data.'''
		file_size = 5000

		increment = len(os.listdir(self.DATA_DIR_PATH + "simulated/noisy/"))	#some datafiles might be already in the directory, this ensures they will not be overwritten

		file_num = iterations // file_size
		for file_i in range(file_num):
			data_noise, data_signal = [], []
			print("Generating data...")
			for i in range(file_size):
				if i % 1000 == 0:	print("{:,}".format(file_i *file_size + i), "/", "{:,}".format(iterations))
				self.initialise()
				signal_history = self.addSignal()
				data_signal.append( numpy.copy(self.space) )
				
				#add energy to signal tiles
				self.energise(signal_history)
			
				self.addNoise()
				data_noise.append(normalise(self.space))

			print("Saving batch...")
			numpy.save(self.DATA_DIR_PATH + "simulated/noisy/" + str(increment+file_i) + ".npy", data_noise)
			numpy.save(self.DATA_DIR_PATH + "simulated/clean/" + str(increment+file_i) + ".npy", data_signal)

#OBSOLETE
'''
class Support:
	@staticmethod
	def showProjection(space :numpy.ndarray, axis :int):
		fig, ax = matplotlib.pyplot.subplots(1)
		ax.imshow( getProjection(space, axis), cmap='gray', vmin=0)
		matplotlib.pyplot.show()

	@staticmethod
	def showProjections(space :numpy.ndarray, indices :int):
		''Shows chosen plots (by indices) of the space projections into the xy, yz and zx planes.''
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
		''Shows some random data from datafile.''
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
		''Shows 3D scatter plot of the input space.''
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
'''

'''from denoise_traces import Plotting
generator = Generator()
while True:
	generator.initialise()
	history = generator.addSignal()
	generator.energise(history)
	fig = matplotlib.pyplot.figure(figsize=matplotlib.pyplot.figaspect(1))
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	Plotting.plot3DToAxis(generator.space, ax)
	fig, ax = matplotlib.pyplot.subplots(2)
	ax[0].imshow(numpy.sum(generator.space,0),"gray")
	ax[1].imshow(numpy.sum(generator.space,1),"gray")
	matplotlib.pyplot.show()'''

if __name__ == "__main__":
	num_gen, is_3D, path = None, None, None
	bool_n, bool_p = False, False

	for arg in sys.argv[1:]:
		if bool_n:
			num_gen = int(float(arg))
			bool_n = False
		elif bool_p:
			path = arg
			bool_p = False
		elif arg == "-h":
			print("-h ... list flags")
			print("-n <integer> ... number of generated events (at least 20000)")
			print("-p <path> ... path to data directory")
		elif arg == "-n":	bool_n = True
		elif arg == "-p":	bool_p = True

	if num_gen == None or path == None:
		print("Specify all the parameters")
	else:
		generator = Generator()
		generator.DATA_DIR_PATH = path + ("" if path[-1] == "/" else "/")
		generator.genAndDumpData3D(num_gen)
		
#generator = Generator()
#generator.genAndDumpData(int(1e6))
#generator.genAndDumpData3D(int(1e5))

'''
generator = Generator()
while True:
	generator.addSignal()
	Support.show3D(generator.space)
	Support.showProjections(generator.space, [0,1,2])

	coord = numpy.nonzero(generator.space)
	generator.space[coord] = numpy.clip( numpy.random.normal(generator.SIGNAL_ENERGY, 0.8*generator.SIGNAL_ENERGY, coord[0].shape), 0, None)

	generator.addNoise()
	Support.show3D(generator.space)
	Support.showProjections(generator.space, [0,1,2])
	if input() == "q":	break
	else:	generator.initialise()
'''