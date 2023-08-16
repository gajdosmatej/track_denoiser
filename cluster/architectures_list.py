from tensorflow import keras

ARCHITECTURES = [
	{	"name":	"larger_pool",
		"mode":	"classification",
		"model":	keras.Sequential([	keras.layers.Input(shape=(12,14,208,1)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,8), filters=50, activation="relu"),
										keras.layers.AveragePooling3D(pool_size=(1,1,4)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,5), filters=30, activation="relu"),
										keras.layers.MaxPooling3D(pool_size=(2,2,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,6), filters=30, activation="relu"),
										keras.layers.UpSampling3D(size=(2,2,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,6), filters=30, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,4), filters=30, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,4), filters=30, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")])
	},

	{	"name":	"larger_pool",
		"mode":	"regression",
		"model":	keras.Sequential([	keras.layers.Input(shape=(12,14,208,1)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,8), filters=50, activation="relu"),
										keras.layers.AveragePooling3D(pool_size=(1,1,4)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,5), filters=30, activation="relu"),
										keras.layers.MaxPooling3D(pool_size=(2,2,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,6), filters=30, activation="relu"),
										keras.layers.UpSampling3D(size=(2,2,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,6), filters=30, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,4), filters=30, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,4), filters=30, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")])
	}
]
