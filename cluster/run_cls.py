from denoise_traces_cls import Model
import numpy
from tensorflow import keras
import json


zy_projections = numpy.load("/scratchdir/data/data_noise_zy.npy")
zy_projections_sgnl = numpy.load("/scratchdir/data/data_signal_zy.npy")

zy_projections = zy_projections.reshape( (*zy_projections.shape, 1) )
zy_projections_sgnl = zy_projections_sgnl.reshape( (*zy_projections_sgnl.shape, 1) )

model_zy = Model("zy")
model_zy.model.compile(optimizer="adam", loss="binary_crossentropy")

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)
history = model_zy.model.fit(x = zy_projections[:15000], y = zy_projections_sgnl[:15000], shuffle=True, epochs=100, 
			     validation_data=(zy_projections[15000:17000], zy_projections_sgnl[15000:17000]), callbacks=[callback])
model_zy.save()

hist_json_file = 'history.json' 
with open("/scratchdir/history.json", mode='w') as f:
	json.dump(history.history, f)