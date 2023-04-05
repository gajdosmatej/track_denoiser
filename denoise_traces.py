import numpy
from tensorflow import keras
import matplotlib.pyplot

datapath = "./data/"

def estimate(model, data_point):
	data_point = numpy.reshape( model( numpy.reshape( data_point, (1, *data_point.shape)) ), data_point.shape)
	return data_point


zy_projections = numpy.load(datapath + "data_noise_zy.npy")
zx_projections = numpy.load(datapath + "data_noise_zx.npy")
yx_projections = numpy.load(datapath + "data_noise_yx.npy")
zy_projections_sgnl = numpy.load(datapath + "data_signal_zy.npy")
zx_projections_sgnl = numpy.load(datapath + "data_signal_zx.npy")
yx_projections_sgnl = numpy.load(datapath + "data_signal_yx.npy")


zy_projections = numpy.pad( zy_projections, ((0,0), (2,0), (0,0)), 'constant', constant_values=0 )
zy_projections_sgnl = numpy.pad( zy_projections_sgnl, ((0,0), (2,0), (0,0)), 'constant', constant_values=0 )


model = keras.Sequential([	keras.layers.Conv2D(input_shape = (*zy_projections[0].shape, 1), padding="same", strides=1, kernel_size=6, filters=16, activation="relu"),
							keras.layers.MaxPool2D(pool_size = (2,2)),
							#keras.layers.Conv2D(padding="same", strides=2, kernel_size=3, filters=4, activation="relu"),
							keras.layers.Conv2D(padding="same", strides=1, kernel_size=4, filters=32, activation="relu"),
							#keras.layers.MaxPool2D(pool_size = (2,2)),
							#keras.layers.Conv2D(padding="same", strides=1, kernel_size=3, filters=16, activation="relu"),
							#keras.layers.UpSampling2D(size = (2,2)),
							#keras.layers.Conv2D(padding="same", strides=1, kernel_size=4, filters=8, activation="relu"),
							keras.layers.UpSampling2D(size = (2,2)),
							keras.layers.Conv2D(padding="same", strides=1, kernel_size=6, filters=16, activation="relu"),
							keras.layers.Dense(units=1, activation="sigmoid") ])

model.compile(optimizer="adam", loss="binary_crossentropy")
model.summary()
model.fit(x = zy_projections[:5000], y = zy_projections_sgnl[:5000], shuffle=True, epochs=15, validation_data=(zy_projections[9000:10000], zy_projections_sgnl[9000:10000]))

model.save("./model_denoise")

#model = keras.models.load_model("./model_denoise")

for _ in range(10):
	j = numpy.random.randint(5000,5500)
	fig, ax = matplotlib.pyplot.subplots(3)
	ax[0].imshow(zy_projections_sgnl[j], cmap="gray")
	ax[1].imshow(zy_projections[j], cmap="gray")
	ax[2].imshow( estimate(model, zy_projections[j]), cmap="gray")

	#print(zy_projections_sgnl[j])
	#print(zy_projections[j])
	#print( numpy.reshape( model( numpy.reshape( zy_projections[j], (1, *zy_projections[j].shape)) ), zy_projections[j].shape))
	matplotlib.pyplot.show()