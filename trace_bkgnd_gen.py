import numpy
import matplotlib.pyplot

datapath = "./data/"
dim_X, dim_Y, dim_Z = 12, 10, 208	#dimensions of discretised space
direction_norms = numpy.array([0.6,0.6,0.9])	#factors which multiply each coordinate of direction unit vector (main aim is to increase velocity in Z direction in order to compensate dim_Z >> dim_X, dim_Y)
signal_energy = 1	#signal deposits this value in every point of lattice which it visits
noise_energy = 1.5	#the total sum of values that noise can deposit in visited points of lattice
energy_deposition_coeff = 0.2	#the coefficient of the exponential distribution, which noise samples a random value from and deposits it in the visited point of lattice
curvature_sigma = 0.1	#sigma of normal distribution, which updateDirection samples R3 vector from
signal_change_direction_probability = 0.8	#the signal probability that updateDirection is called in a step
noise_change_direction_probability = 0.8	#the noise probability that updateDirection is called in a step
noise_tracks_num_range = (20,21)	#the minimal and maximal number of noise tracks

def updateDirection(direction :numpy.ndarray):
	'''Randomly updates the direction vector.'''
	direction += numpy.random.normal(loc=0, scale=curvature_sigma)
	direction /= numpy.linalg.norm(direction)
	direction *= direction_norms

def discretise(coord):
	return (round(coord[0]), round(coord[1]), round(coord[2]))

def isCoordOutOfBounds(coord):
	return ( coord[0] < 0 or coord[1] < 0 or coord[2] < 0 or coord[0] >= dim_X-0.5 or coord[1] >= dim_Y-0.5 or coord[2] >= dim_Z-0.5)	#coord[i] >= dim_i - 0.5	iff 	round(coord[i]) >= dim_i 

def addSignal(space :numpy.ndarray):
	'''Adds one signal track into the input 3D array space.'''
	position = (dim_X/2, dim_Y/2, dim_Z/2)	#the track begins in the middle of the space
	direction = numpy.random.uniform( low = (-1)*direction_norms, high = direction_norms, size=3)	#random moving direction
	while not isCoordOutOfBounds(position):	#the track propagades until it moves out of the space boundaries
		space[discretise(position)] = signal_energy
		if numpy.random.random() < signal_change_direction_probability:	updateDirection(direction)
		position += direction


def addNoise(space :numpy.ndarray):
	'''Adds several noise tracks into the input 3D array space.'''
	num_of_traces = numpy.random.randint( *noise_tracks_num_range )
	for _ in range(num_of_traces):
		direction = None
		position = None
		side = numpy.random.randint(0,6)	#the noise always starts on the boundary of the space
		energy = noise_energy
		if side == 0:
			position = numpy.array( [0, numpy.random.randint(0, dim_Y), numpy.random.randint(0, dim_Z)], dtype=float )
			direction = numpy.array([direction_norms[0],0.,0.])
		elif side == 1:
			position = numpy.array( [dim_X-1, numpy.random.randint(0, dim_Y), numpy.random.randint(0, dim_Z)], dtype=float )
			direction = numpy.array([-direction_norms[0],0.,0.])
		elif side == 2:
			position = numpy.array( [numpy.random.randint(0, dim_X), 0, numpy.random.randint(0, dim_Z)], dtype=float )
			direction = numpy.array([0.,direction_norms[1],0.])
		elif side == 3:
			position = numpy.array( [numpy.random.randint(0, dim_X), dim_Y-1, numpy.random.randint(0, dim_Z)], dtype=float )
			direction = numpy.array([0.,-direction_norms[1],0.])
		elif side == 4:
			position = numpy.array( [numpy.random.randint(0, dim_X), numpy.random.randint(0, dim_Y), 0], dtype=float )
			direction = numpy.array([0.,0.,direction_norms[2]])
		else:
			position = numpy.array( [numpy.random.randint(0, dim_X), numpy.random.randint(0, dim_Y), dim_Z-1], dtype=float )
			direction = numpy.array([0.,0.,-direction_norms[2]])
		while energy > 0:	#the particle moves until it loses all the energy or it moves out of the space
			energy -= numpy.clip( numpy.random.exponential(energy_deposition_coeff), 0, 1)
			space[discretise(position)] += energy
			if numpy.random.random() < noise_change_direction_probability:	updateDirection(direction)
			position += direction
			if isCoordOutOfBounds(position):	break


def show3D(space :numpy.ndarray):
	'''Shows 3D scatter plot of the input space.'''
	xs, ys, zs = space.nonzero()
	fig = matplotlib.pyplot.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(xs, ys, zs)
	ax.set_xlim(0,dim_X)
	ax.set_ylim(0,dim_Y)
	ax.set_zlim(0,dim_Z)
	matplotlib.pyplot.show()

def getProjection(space :numpy.ndarray, axis :int):
	return numpy.sum(space, axis)

def showProjections(space :numpy.ndarray):
	'''Shows plots of the space projections into the xy, yz and zx planes.'''
	fig, ax = matplotlib.pyplot.subplots(1,3)
	ax[0].imshow(getProjection(space, 0), cmap='gray', vmin=0)
	ax[0].set_xlabel("z")
	ax[0].set_ylabel("y")
	ax[1].imshow(getProjection(space, 1), cmap='gray', vmin=0)
	ax[1].set_xlabel("z")
	ax[1].set_ylabel("x")
	ax[2].imshow(getProjection(space, 2), cmap='gray', vmin=0)
	ax[2].set_xlabel("y")
	ax[2].set_ylabel("x")
	matplotlib.pyplot.show()

def showProjection(space :numpy.ndarray, axis :int):
	fig, ax = matplotlib.pyplot.subplots(1)
	ax.imshow( getProjection(space, axis), cmap='gray', vmin=0)
	matplotlib.pyplot.show()

def normalise(arr :numpy.ndarray):
	'''Linearly maps values of input array to [0,1]'''
	min, max = numpy.min(arr), numpy.max(arr)
	f = lambda x:	(x - min) / (max - min)
	return f(arr)

def genAndDumpData(iterations :int):
	'''Generates space 3D array with one signal and several noises in each iteration and saves the projections of the clean and noised data.'''
	noise_names = ["data_noise_zy", "data_noise_zx", "data_noise_yx"]
	signal_names = ["data_signal_zy", "data_signal_zx", "data_signal_yx"]

	data_noise, data_signal = [[], [], []], [[], [], []]

	for i in range(iterations):
		space = numpy.zeros( (dim_X, dim_Y, dim_Z) )
		addSignal(space)
		for k in range(3):
			data_signal[k].append(getProjection(space, k))
			
		addNoise(space)
		for k in range(3):
			data_noise[k].append(getProjection(space, k))

		if i % 1000 == 0:	print(i, "/", iterations)

	for k in range(3):
		data_noise[k] = normalise(data_noise[k])
		data_signal[k] = normalise(data_signal[k])
		numpy.save(datapath + noise_names[k], data_noise[k])
		numpy.save(datapath + signal_names[k], data_signal[k])

def showRandomData():
	'''Shows some random data from datafile.'''
	zy_projections = numpy.load(datapath + "data_noise_zy.npy")
	zx_projections = numpy.load(datapath + "data_noise_zx.npy")
	yx_projections = numpy.load(datapath + "data_noise_yx.npy")
	zy_projections_sgnl = numpy.load(datapath + "data_signal_zy.npy")
	zx_projections_sgnl = numpy.load(datapath + "data_signal_zx.npy")
	yx_projections_sgnl = numpy.load(datapath + "data_signal_yx.npy")

	size = numpy.shape(zy_projections)[0]

	for _ in range(5):
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


#genAndDumpData(120000)
#showRandomData()

space = numpy.zeros( (dim_X, dim_Y, dim_Z) )
addSignal(space)
show3D(space)
addNoise(space)
show3D(space)
showProjection(space, 0)

