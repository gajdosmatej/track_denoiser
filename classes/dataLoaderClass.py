import numpy
import os
import tensorflow
from classes import supportFunctions

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
			with open(path + name + ".txt", "w") as f:
				f.write(dump_str)

	def importData(self, dir_name = "dump/", names=None, order=True):
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

		# @names are not specified, load the whole directory
		if names is None:
			data = numpy.zeros( (len(os.listdir(path)), 12, 14, 208) )
			names = []
			i = 0
			file_names = []
			if order:
				for num in sorted( [val[5:-4] for val in os.listdir(path)] ):	# Order the data
					file_names.append("track" + num + ".txt")
			else:
				file_names = os.listdir(path)
			for file_name in file_names:
				with open(path + file_name, "r") as f:
					for line in f.readlines():
						x, y, z, E = parseLine(line)
						data[i,x,y,z] = E
				names.append(file_name[:-4])
				i += 1
			return (names, data)
		
		# load only @names in that order
		else:
			data = numpy.zeros( (len(names), 12, 14, 208) )
			i = 0
			for name in names:
				with open(path + name + ".txt", "r") as f:
					for line in f.readlines():
						x, y, z, E = parseLine(line)
						data[i,x,y,z] = E
				i += 1
			return (names, data)

	def fillNameIndexDictionary(self):
		names = self.getX17Names()
		for i in range(len(names)):
			self.name_index_dict[names[i]] = i

	def getEventFromName(self, name: str, noisy :bool, preprocessed :bool = True):
		'''
		Return X17 event from its @name.
		'''
		return self.getBatch(True, noisy, preprocessed=preprocessed)[ self.name_index_dict[name] ]

	def loadX17Data(self, noisy :bool):
		'''
		Parse X17 data from txt file into iterator of tuples (name, event).
		'''
		path = "x17/" + ("noisy/" if noisy else "clean/")
		for (name, event) in zip( *self.importData(path, names=None, order=True) ):
			yield (name, event)

	def dataPairLoad(self, low_id :int, high_id :int):
		'''
		Yield a pair of noisy and clean event tensors from numbered data files in between @low_id and @high_id
		'''

		while True:
			order = numpy.arange(low_id, high_id)
			numpy.random.shuffle(order)
			for id in order:
				signal_batch = numpy.load(self.path + "simulated/clean/" + str(id) + ".npy")
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

	def datasetExists(self, noisy :bool, preprocessed :bool):
		return (noisy, preprocessed) in self.existing_experimental_datasets
	
	def getDatasetFromDictionary(self, noisy :bool, preprocessed :bool):
		return self.existing_experimental_datasets[(noisy, preprocessed)]

	def addDatasetToDictionary(self, dataset :numpy.ndarray, noisy :bool, preprocessed :bool):
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
				if noisy:	supportFunctions.removeChimneys(event)
				x17_data.append( supportFunctions.normalise(event) )
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
		for (name, _) in self.loadX17Data(False):	
			names.append(name)
		return names