import numpy
from tensorflow import keras
import keras.utils
import os

class Model:
	def __init__(self):
		self.chooseModelZX()

	def chooseModelZX(self):
		self.type = "zx"
		self.model = keras.Sequential([	keras.layers.Input(shape=(12,208,1)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(6,14), filters=16, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,9), filters=32, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,4), filters=128, activation="relu"),
									keras.layers.MaxPool2D(pool_size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(2,2), filters=128, activation="relu"),
									keras.layers.UpSampling2D(size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(3,4), filters=128, activation="relu"),
									keras.layers.UpSampling2D(size = (1,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(4,9), filters=32, activation="relu"),
									keras.layers.UpSampling2D(size = (2,2)),
									keras.layers.Conv2D(padding="same", strides=(1,1), kernel_size=(6,14), filters=16, activation="relu"),
									keras.layers.Dense(units=1, activation="sigmoid") ])

	def save(self):
		self.model.save("/scratchdir/CLS_MODEL")