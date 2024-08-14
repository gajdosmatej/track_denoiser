import numpy
import matplotlib.pyplot
import time
import os
import sys
from classes import DataLoader


USE_MEASURED_NOISE = 2	# 0 => only generated, 1 => only measured, 2 => mixed 1:1
BATCH_SIZE = 5000
SHAPE = (12,14,208)
PROBABILITY_TILE_NOISY = 0.0005
MOVEMENT_FACTORS = numpy.array([0.1,0.1,1])
NOISE_LENGTH_LAMBDA = 6
Z_BOUNDS = (50,170)
TRACK_THRESHOLD = 3	# minimal number of pads for a valid track
DATA_LOADER = DataLoader("./data")
BACKGROUNDS, _ = DATA_LOADER.importData("x17/gauge_backgrounds")[:100]


def getWaveformNoiseTileCDF():
	'''
	Get cumulative distribution function for the random number of tiles in one pad which should start a noise waveform.
	Each tile has probability @PROBABILITY_TILE_NOISY for starting a noise waveform, so tile is a random variable 
	T_(x,y,z) ~ Bernoulli(@PROBABILITY_TILE_NOISY). Then the number of tiles in one pad starting a noise waveform is
	Sum_z T_(x,y,z) ~ Binomial(len(z), @PROBABILITY_TILE_NOISY).
	'''

	max_num = SHAPE[2]
	p = PROBABILITY_TILE_NOISY
	CDF = [0] * (max_num+1)
	
	probabilities = [0] * (max_num+1)
	probabilities[0] = (1-p)**max_num
	CDF[0] = probabilities[0]

	for i in range(1, max_num+1):
		probabilities[i] = p/(1-p) * (max_num-i+1) / i * probabilities[i-1]
		CDF[i] = CDF[i-1] + probabilities[i]
	return CDF

NOISE_TILE_CDF = getWaveformNoiseTileCDF()

def getBoundaryStart(direction):
		'''
		Get coordinate on a side of a cuboid, such that the direction points inside the cuboid.
		'''

		position = None
		m = numpy.max(direction)

		if direction[0] == m:
			position = numpy.array( [0, numpy.random.randint(0, SHAPE[1]), numpy.random.randint(*Z_BOUNDS)], dtype=float )
		elif direction[0] == -m:
			position = numpy.array( [SHAPE[0]-1, numpy.random.randint(0, SHAPE[1]), numpy.random.randint(*Z_BOUNDS)], dtype=float )
		elif direction[1] == m:
			position = numpy.array( [numpy.random.randint(0, SHAPE[0]), 0, numpy.random.randint(*Z_BOUNDS)], dtype=float )
		elif direction[1] == -m:
			position = numpy.array( [numpy.random.randint(0, SHAPE[0]), SHAPE[1]-1, numpy.random.randint(*Z_BOUNDS)], dtype=float )
		elif direction[2] == m:
			position = numpy.array( [numpy.random.randint(0, SHAPE[0]), numpy.random.randint(0, SHAPE[1]), Z_BOUNDS[0]], dtype=float )
		elif direction[2] == -m:
			position = numpy.array( [numpy.random.randint(0, SHAPE[0]), numpy.random.randint(0, SHAPE[1]), Z_BOUNDS[1]], dtype=float )
		return position


def sampleInitDirection():
		'''
		Samples direction vector uniformly from a unit sphere.
		'''

		vect = numpy.random.normal(loc=0,scale=1,size=3)
		vect /= numpy.linalg.norm(vect)
		return vect

def sampleLandau():
	'''
	Sample random value from Landau distribution.
	'''

	# Values are based on energy distribution of a subset of measured data (calibration dataset)
	mu = 0.2
	sigma = 0.05
	
	# Approximation of Landau(mu=0, sigma=1)
	def standardisedLandauPDF(x):
		return 1/numpy.sqrt(2*numpy.pi) * numpy.exp( -(x+numpy.exp(-x))/2 )
	
	while True:
		# Linearly transform X ~ Landau(0,1) so that aX+b ~ Landau(mu, sigma)
		offset = mu + 2*sigma*numpy.log(sigma) / numpy.pi
		x = numpy.random.uniform(-offset/sigma, (1-offset)/sigma)
		y = numpy.random.random()
		if y < standardisedLandauPDF(x):	# Rejection method for sampling from PDF
			return numpy.clip(sigma*x + offset, 0, 1)

def discretise(coord):
	'''
	Return closest tensor entry to continuous coordinate triple @coord.
	'''
	return (round(coord[0]), round(coord[1]), round(coord[2]))

def generateTrack(event_noise, event_clean):
	'''
	Add linear track to @event_noise (with Landau energies) and to @event_clean (every entry is value 1).
	'''

	direction = sampleInitDirection()
	position = getBoundaryStart(direction)
	coord = position
	visited_tiles = 0
	while True:
		new_position = position + direction*MOVEMENT_FACTORS
		new_coord = discretise(new_position)
		if numpy.any(new_coord != coord):	# The track moved to a new position in event tensors
			# Check whether the track is out of tensor boundaries
			if new_coord[0] < 0 or new_coord[0] >= SHAPE[0] or new_coord[1] < 0 or new_coord[1] >= SHAPE[1] or new_coord[2] < Z_BOUNDS[0] or new_coord[2] > Z_BOUNDS[1]:	break
			event_noise[new_coord] = sampleLandau()	# Add energy to noisy event
			if event_clean is not None:	event_clean[new_coord] = 1	# In case of labeling, @event_clean is left empty
		visited_tiles += 1 if (new_coord[0] != coord[0] or new_coord[1] != coord[1]) else 0
		visited_tiles += 0.05 if (new_coord[2] != coord[2]) else 0
		position = new_position
		coord = new_coord
	return visited_tiles >= TRACK_THRESHOLD

def sampleNoiseTilesNumber():
	'''
	Sample number of tiles in one pad which should start a noise waveform.
	'''
	p = numpy.random.random()
	for i in range(SHAPE[2]):
		if p < NOISE_TILE_CDF[i]:	break
	return i

def addNoiseWaveform(event, x, y, z0=None):
	'''
	Add noise waveform to @event, starting at position (@x, @y, @z0), propagating in z direction 
	with a length ~ Poisson( @NOISE_LENGTH_LAMBDA ).
	'''

	if z0 == None:	z0 = numpy.random.randint(0,SHAPE[2])
	length = numpy.random.poisson(NOISE_LENGTH_LAMBDA)
	z1 = min(z0+length+1, SHAPE[2])
	
	for z in range(z0, z1):
		E = numpy.clip( numpy.random.gamma(0.25297058, 1/1.88917480), 0, 1)	# Energy distribution approximation based on the calibration dataset (subset of measured events)
		event[x,y,z] = E

def addGeneratedNoise(event):
	'''
	Add synthetised noise waveforms to @event.
	'''
	for x in range(SHAPE[0]):
		for y in range(SHAPE[1]):
			noise_tiles_num = sampleNoiseTilesNumber()
			for _ in range(noise_tiles_num):
				addNoiseWaveform(event, x, y)

def addNoise(event, use_measured_noise):
	'''Add the correct noise type to @event based on @use_measured_noise (0 ... generated, 1 ... measured noisemaps, 2 ... both in 1:1 ratio).'''
	if use_measured_noise == 0 or (use_measured_noise == 2 and numpy.random.random() < 0.5):
		addGeneratedNoise(event)
		return

	#use measured noisemap
	num_swaps = 50
	background = BACKGROUNDS[numpy.random.randint(0,100)]
	# Permute the chosen noisemap
	for _ in range(num_swaps):
		ks, ls = [None, None], [None, None]
		for i in [0,1]:
			ks[i], ls[i] = numpy.random.randint(0, background.shape[0]), numpy.random.randint(0, background.shape[1])
		
		background[ks[0], ls[0]], background[ks[1], ls[1]] = background[ks[1], ls[1]].copy(), background[ks[0], ls[0]].copy()
	
	event += background

def generateEventDenoising(event_noise, event_clean, use_measured_noise):
	'''
	Add track and noise to input tensors used for denoising.
	'''
	while not generateTrack(event_noise, event_clean):
		event_noise[...], event_clean[...] = 0, 0
	addNoise(event_noise, use_measured_noise)

def generateEventLabeling(event_noise, use_measured_noise):
	'''
	Create noisy event either with or without a track for labeling usage.
	'''

	is_good = (numpy.random.random() < 0.5)
	if is_good:	# With a track
		while not generateTrack(event_noise, None):
			event_noise[...] = 0
	addNoise(event_noise, use_measured_noise)
	return int(is_good)

def generateBatch(just_labels, use_measured_noise):
	'''
	Generate batch of denoising/labeling data (based on @just_labels).
	'''

	batch_noise = numpy.zeros( (BATCH_SIZE, *SHAPE), dtype=numpy.float16 )
	
	if just_labels:
		batch_clean = numpy.zeros(BATCH_SIZE, dtype=numpy.float16)
	else:
		batch_clean = numpy.zeros( (BATCH_SIZE, *SHAPE), dtype=numpy.float16 )

	for i in range(BATCH_SIZE):
		if just_labels:
			batch_clean[i] = generateEventLabeling(batch_noise[i], use_measured_noise)
		else:
			generateEventDenoising(batch_noise[i], batch_clean[i], use_measured_noise)
		if i % 250 == 0:	print(i, "generated")
	return batch_noise, batch_clean


def generateAndDump(root_path :str, batch_number :int, just_labels :bool, use_measured_noise :int):
	'''
	Generate data and save them to .npy files.
	'''

	increment = len(os.listdir(root_path + "noisy/"))	#some datafiles might be already in the directory, this ensures they will not be overwritten
	for i in range(batch_number):
		batch_noise, batch_clean = generateBatch(just_labels, use_measured_noise)
		print("Saving " + str(i) + ". batch from " + str(batch_number))
		numpy.save(root_path + "noisy/" + str(increment+i) + ".npy", batch_noise)
		numpy.save(root_path + "clean/" + str(increment+i) + ".npy", batch_clean)


def showExample(event_noise, event_clean):
	'''
	Plot and show a generated event.
	'''

	x_labels = ["z","z","y"]
	y_labels = ["y","x","x"]
	cmap = matplotlib.pyplot.get_cmap("Greys")
	cmap.set_under('cyan')

	event_noise = numpy.repeat( numpy.repeat(event_noise, 10, axis=0), 10, axis=1 )
	event_clean = numpy.repeat( numpy.repeat(event_clean, 10, axis=0), 10, axis=1 )

	#aspects = [(10/14) / (8/200), (10/12) / (8/200), (10/12)/ (10/14)]

	fig, ax = matplotlib.pyplot.subplots(3,2)
	for i in range(3):
		ax[i,0].imshow(numpy.sum(event_noise, i), cmap=cmap ,vmin=0.00001)
		ax[i,0].set_title("Noisy")
		ax[i,1].imshow(numpy.sum(event_clean, i), cmap=cmap ,vmin=0.00001)
		ax[i,1].set_title("Clean")
		for is_noisy in [0,1]:
			ax[i,is_noisy].set_xlabel(x_labels[i])
			ax[i,is_noisy].set_ylabel(y_labels[i])
	matplotlib.pyplot.show()


if __name__ == "__main__":
	# Flags
	batch_num, path, labeling_only  = None, None, None
	bool_n, bool_p, bool_l = False, False, False
	only_show = False

	for arg in sys.argv[1:]:
		if bool_n:
			batch_num = int(float(arg))
			bool_n = False
		elif bool_p:
			path = arg
			bool_p = False
		elif bool_l:
			labeling_only = bool(int(arg))
			bool_l = False
		elif arg == "-h":
			print("-h ... list flags")
			print("-n <integer> ... number of generated batches (" + str(BATCH_SIZE) + " events/batch)")
			print("-p <path> ... path to data directory")
			print("-l <0/1> ... whether the generated data are used for denoising (0) or labeling (1)")
			print("-s ... only show a generated example")
			exit()
		elif arg == "-n":	bool_n = True
		elif arg == "-p":	bool_p = True
		elif arg == "-s":	only_show = True
		elif arg == "-l":	bool_l = True

	# Just plotting if flag -s is present
	if only_show:
		noise, clean = numpy.zeros((12,14,208), dtype=numpy.float16), numpy.zeros((12,14,208), dtype=numpy.float16)
		generateEventDenoising(noise, clean, USE_MEASURED_NOISE)
		showExample(noise, clean)

	elif batch_num == None or path == None or labeling_only == None:
		print("Specify all the parameters (flag -h shows their list)")
	else:
		# Generate and save data
		path += ("" if path[-1] == "/" else "/")
		path += ("labeling/" if labeling_only else "simulated/")
		generateAndDump(path, batch_num, labeling_only, USE_MEASURED_NOISE)