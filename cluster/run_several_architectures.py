import numpy
from tensorflow import keras
import tensorflow
import architectures_list
import json

def dataPairLoad(low_id :int, high_id :int, is_regression :bool):
		while True:
			order = numpy.arange(low_id, high_id)
			numpy.random.shuffle(order)
			for id in order:
				signal_batch = numpy.load("/data/simulated/3D/" + str(id) + "_signal_3d.npy")
				noise_batch = numpy.load("/data/simulated/3D/" + str(id) + "_noise_3d.npy")
				if not is_regression:
					signal_batch = numpy.where(signal_batch>0.01, 1, 0)
				for i in range(5000):
					yield ( numpy.reshape(noise_batch[i], (12,14,208,1)), numpy.reshape(signal_batch[i], (12,14,208,1)))
	
def getDataset(low :int, high :int, is_regression :bool):
	return tensorflow.data.Dataset.from_generator(lambda: dataPairLoad(low, high, is_regression), output_signature =
					(	tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16),
						tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16))
					).batch(100).prefetch(20)


for architecture in architectures_list.ARCHITECTURES:
	is_regression = (architecture["mode"] == "regression")
	training_dataset = getDataset(0,10, is_regression)
	validation_dataset = getDataset(11,13, is_regression)

	model = architecture["model"]
	model.compile(optimizer="Adam", loss="binary_crossentropy")
	history = model.fit(x = training_dataset, epochs=100, steps_per_epoch=10, validation_data=validation_dataset, validation_steps=10,
						callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
	name = architecture["name"] + ("_R" if architecture["mode"] == "regression" else "_C")
	architecture["model"].save("/scratchdir/models/" + name)
	json_file = open("/scratchdir/histories/" + name + ".json", "w")
	json.dump(history.history, json_file)