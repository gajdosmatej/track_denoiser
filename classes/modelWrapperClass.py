from tensorflow import keras
import numpy

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
