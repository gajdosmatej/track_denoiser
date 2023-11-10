import numpy
from tensorflow import keras
import tensorflow
import architectures_list
import json
from classes import DataLoader

for architecture in architectures_list.ARCHITECTURES:
	data_loader = DataLoader("/data/")
	training_dataset = data_loader.getDataset(0,20, 100)

	model = architecture["model"]
	model.compile(optimizer="Adam", loss="binary_crossentropy")
	num_epochs = architecture["epochs"] if "epochs" in architecture else 100
	history = model.fit(x = training_dataset, epochs=num_epochs, steps_per_epoch=50, validation_data=data_loader.getValidationData())
	name = architecture["name"]
	architecture["model"].save("/scratchdir/raw_models/" + name)
	json_file = open("/scratchdir/histories/" + name + ".json", "w")
	json.dump(history.history, json_file)