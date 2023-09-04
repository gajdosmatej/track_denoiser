from tensorflow import keras

ARCHITECTURES = [
	{	"name":	"stonozka_some_pooling",
		"mode":	"classification",
		"epochs": 200,
		"model": keras.Sequential([	keras.layers.Input(shape=(12,14,208,1)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,8), filters=30, activation="relu"),
										keras.layers.MaxPooling3D((1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=8, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,2), kernel_size=(3,3,3), filters=8, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=8, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,2), kernel_size=(3,3,3), filters=12, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,4), filters=12, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=12, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=8, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=8, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=8, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=8, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,6), filters=8, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")])
	}	
]

'''
{"name":	"stonozka_no_pooling",
		"mode":	"classification",
		"epochs": 200,
		"model":	keras.Sequential([	keras.layers.Input(shape=(12,14,208,1)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,8), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,2), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,2), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,2), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,4), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,6), filters=10, activation="relu"),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")])
	}
'''

'''{	"name":	"stonozka",
		"mode":	"classification",
		"epochs": 200,
		"model":	keras.Sequential([	keras.layers.Input(shape=(12,14,208,1)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,8), filters=10),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.MaxPooling3D(pool_size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=10),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.MaxPooling3D(pool_size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=10),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.MaxPooling3D(pool_size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,4), filters=10),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=10),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(2,2,5), filters=10),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=10, activation="relu"),
										keras.layers.UpSampling3D(size=(1,1,2)),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,6), filters=10),
										keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=(3,3,3), filters=1, activation="sigmoid")])
	}'''