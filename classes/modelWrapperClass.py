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


def getPaperModel():
	'''
	Return the model (untrained, only architecture) described in the paper.
	'''

	inputs = keras.layers.Input((12,14,208,1))

	support0 = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=2, activation="relu")(inputs)
	support0 = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=2, activation="relu")(support0)

	pool1 = keras.layers.AveragePooling3D((1,1,2))(inputs)
	support1 = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=2, activation="relu")(pool1)
	support1 = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=2, activation="relu")(support1)

	pool2 = keras.layers.AveragePooling3D((1,1,2))(pool1)
	support2 = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=2, activation="relu")(pool2)
	support2 = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=2, activation="relu")(support2)

	x = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=2, activation="relu")(inputs)
	x = keras.layers.AveragePooling3D((1,1,2))(x)
	x = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=3, activation="relu")(x)
	x = keras.layers.AveragePooling3D((1,1,2))(x)
	x = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=4, activation="relu")(x)
	x = keras.layers.AveragePooling3D((1,1,2))(x)

	x = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=4, activation="relu")(x)
	x = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=4, activation="relu")(x)

	x = keras.layers.UpSampling3D((1,1,2))(x)
	x = keras.layers.Concatenate()([x,support2])
	x = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=3, activation="relu")(x)
	x = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=3, activation="relu")(x)

	x = keras.layers.UpSampling3D((1,1,2))(x)
	x = keras.layers.Concatenate()([x,support1])
	x = keras.layers.Conv3D(padding="same", strides=1, kernel_size=(3,3,3), filters=3, activation="relu")(x)
	x = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=3, activation="relu")(x)

	x = keras.layers.UpSampling3D((1,1,2))(x)
	x = keras.layers.Concatenate()([x,support0])
	x = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=3, activation="relu")(x)
	x = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=3, activation="relu")(x)
	x = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=3, activation="relu")(x)
	outputs = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")(x)
	return keras.Model(inputs=inputs, outputs=outputs)