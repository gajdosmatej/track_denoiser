from tensorflow import keras

ARCHITECTURES = [
	keras.Sequential([	keras.layers.Input(shape=(12,14,208,1)),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(4,4,5), filters=10, activation="relu"),
						keras.layers.MaxPooling3D(pool_size=(1,1,2)),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
						keras.layers.MaxPooling3D(pool_size=(2,2,2)),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
						keras.layers.UpSampling3D(size=(2,2,2)),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
						keras.layers.UpSampling3D(size=(1,1,2)),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,5), filters=10, activation="relu"),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")]),
	keras.Sequential([	keras.layers.Input(shape=(12,14,208,1)),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(4,4,10), filters=20, activation="relu"),
						keras.layers.MaxPooling3D(pool_size=(1,1,2)),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,6), filters=20, activation="relu"),
						keras.layers.UpSampling3D(size=(1,1,2)),
						keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")])
]