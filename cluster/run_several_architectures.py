import numpy
from tensorflow import keras
import tensorflow
import architectures_list
import json

def dataPairLoad(low_id :int, high_id :int):
		while True:
			order = numpy.arange(low_id, high_id)
			numpy.random.shuffle(order)
			for id in order:
				signal_batch = numpy.load("/data/simulated/3D/" + str(id) + "_signal_3d.npy")
				noise_batch = numpy.load("/data/simulated/3D/" + str(id) + "_noise_3d.npy")
				for i in range(5000):
					yield ( numpy.reshape(noise_batch[i], (12,14,208,1)), numpy.reshape(signal_batch[i], (12,14,208,1)))
	
def getDataset(low, high):
	return tensorflow.data.Dataset.from_generator(lambda: dataPairLoad(low, high), output_signature =
					(	tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16),
						tensorflow.TensorSpec(shape=(12,14,208,1), dtype=tensorflow.float16))
					).batch(50).prefetch(20)


training_dataset = getDataset(0,10)
validation_dataset = getDataset(11,13)
counter = 0
for model in architectures_list.ARCHITECTURES:
	model.compile(optimizer="Adam", loss="binary_crossentropy")
	history = model.fit(x = training_dataset, epochs=50, steps_per_epoch=10, validation_data=validation_dataset, validation_steps=10)
	model.save("/scratchdir/models/" + str(counter))
	json_file = open("/scratchdir/histories/" + str(counter) + ".json", "w")
	json.dump(history.history, json_file)
	counter += 1