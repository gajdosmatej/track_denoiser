import tensorflow
from tensorflow import keras

class RandomNoise(keras.layers.Layer):
	'''
	Custom Keras Layer applying random gaussian noise to the input.	
	'''
	
	def __init__(self, std=1):
		super().__init__()
		self.std = std

	def call(self, x):
		return keras.layers.add([x, tensorflow.random.normal(shape=tensorflow.shape(x), mean=0, stddev=self.std)])

	def build(self, input_shape):
		pass


class SEAttention(keras.layers.Layer):
	'''
	Custom Keras Layer applying channelwise attention to a 3D tensor. 
	'''

	def __init__(self):
		super().__init__()
	
	def call(self, x):
		return keras.layers.multiply([x, self.getChannelAttention(x)])
	
	def build(self, input_shape):
		channels = input_shape[-1]

		self.avging = keras.layers.GlobalAveragePooling3D()
		self.maxing = keras.layers.GlobalMaxPooling3D()

		self.MLP_1 = keras.layers.Dense(channels//2, "relu")
		self.MLP_2 = keras.layers.Dense(channels, "relu")

	def getChannelAttention(self, x):
		'''
		Squeeze-Excitation returning channel attention for 3D tensor @x.
		'''

		return keras.layers.Activation("sigmoid")( keras.layers.Add()([	self.MLP_2(self.MLP_1(self.avging(x))),
																		self.MLP_2(self.MLP_1(self.maxing(x)))]) )


class SpatialAttention(keras.layers.Layer):
	'''
	Custom Keras Layer applying spatial attention to a 3D tensor.
	'''

	def __init__(self):
		super().__init__()
	
	def call(self, x):
		return keras.layers.multiply([x, self.getSpatialAttention(x)])
	
	def build(self, input_shape):
		self.aggregation = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(1,1,1), filters=1, activation="relu")
		self.conv = keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")

	def getSpatialAttention(self, x):
		return self.conv( self.aggregation(x) )

