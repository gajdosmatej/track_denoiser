import numpy
from tensorflow import keras
import tensorflow
import architectures_list
import json
from denoise_traces import DataLoader

for architecture in architectures_list.ARCHITECTURES:
	data_loader = DataLoader("/data/")
	training_dataset = data_loader.getDataset(0,10)
	validation_dataset = data_loader.getDataset(11,13)

	model = architecture["model"]
	model.compile(optimizer="Adam", loss="binary_crossentropy")
	num_epochs = architecture["epochs"] if "epochs" in architecture else 100
	history = model.fit(x = training_dataset, epochs=num_epochs, steps_per_epoch=10, validation_data=validation_dataset, 
						validation_steps=10)
	name = architecture["name"]
	architecture["model"].save("/scratchdir/models/" + name)
	json_file = open("/scratchdir/histories/" + name + ".json", "w")
	json.dump(history.history, json_file)