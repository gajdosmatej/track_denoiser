from denoise_traces import Model, Plotting, QualityEstimator
import numpy
from tensorflow import keras
import keras.utils
import pandas

zy_projections = numpy.load("./data/data_noise_zy.npy")
zx_projections = numpy.load("./data/data_noise_zx.npy")
yx_projections = numpy.load("./data/data_noise_yx.npy")
zy_projections_sgnl = numpy.load("./data/data_signal_zy.npy")
zx_projections_sgnl = numpy.load("./data/data_signal_zx.npy")
yx_projections_sgnl = numpy.load("./data/data_signal_yx.npy")

zx_projections = zx_projections.reshape( (*zx_projections.shape, 1) )
zx_projections_sgnl = zx_projections_sgnl.reshape( (*zx_projections_sgnl.shape, 1) )

model_zx = Model.load("CLS_MODEL", "zx")
keras.utils.plot_model(model_zx.model, to_file="./models/CLS_MODEL.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)
data = QualityEstimator.reconstructedSignals(zx_projections_sgnl[15000:], model_zx.model(zx_projections[15000:]))
model_zx.saveSignalMetricData(data, "./models/sgn_CLS_MODEL.txt")
data = QualityEstimator.filteredNoise(zx_projections_sgnl[15000:], model_zx.model(zx_projections[15000:]))
model_zx.saveSignalMetricData(data, "./models/noise_CLS_MODEL.txt")

#Plotting.plotRandomData(model_zx, zx_projections_sgnl, zx_projections, 4)