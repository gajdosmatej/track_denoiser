from denoise_traces_cls import Model
import numpy
from tensorflow import keras
import json


zx_projections = numpy.load("/scratchdir/data/data_noise_zx.npy")
zx_projections_sgnl = numpy.load("/scratchdir/data/data_signal_zx.npy")

zx_projections = zx_projections.reshape( (*zx_projections.shape, 1) )
zx_projections_sgnl = zx_projections_sgnl.reshape( (*zx_projections_sgnl.shape, 1) )

model_zx = Model()
model_zx.model.compile(optimizer="adam", loss="binary_crossentropy")
history = model_zx.model.fit(x = zx_projections[:15000], y = zx_projections_sgnl[:15000], shuffle=True, epochs=30, validation_data=(zx_projections[15000:17000], zx_projections_sgnl[15000:17000]))
model_zx.save()

hist_json_file = 'history.json' 
with open("/scratchdir/history.json", mode='w') as f:
	json.dump(history.history, f)