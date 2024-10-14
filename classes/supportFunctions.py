from tensorflow import keras
import numpy
import tensorflow

SATURATION_THRESHOLD = 800

def normalise(event :numpy.ndarray):
	'''
	Linearly map @event to [0,1] interval.
	'''

	M = numpy.max(event)
	if M == 0:	return event
	return event / M


def removeChimneys(event):
	'''
	Remove noisy waveforms in cases where the waveform crosses the saturation threshold.
	'''
	xs, ys, zs = numpy.where(event > SATURATION_THRESHOLD)
	pads = {}
	for i in range(xs.shape[0]):
		x, y, z = xs[i], ys[i], zs[i]
		if (x,y) not in pads:	pads[(x,y)] = z
		else:	pads[(x,y)] = min(z, pads[(x,y)])	# store the smallest z coordinate where the crossing occured
	
	for (x,y) in pads:
		z = pads[(x,y)]
		#event[x,y,(z+1):] = 0
		waveform = event[x,y,z:]
		waveform = waveform[waveform>0]
		length = waveform.shape[0]
		if length < 20:	continue	# The chimney is small, we let it be
		baseline = numpy.min(waveform)
		variance = numpy.var(waveform)
		#threshold = variance/length
		threshold = numpy.sqrt(variance)
		#threshold=0
		#threshold=numpy.infty
		event[x,y,z:] = numpy.where(event[x,y,z:]-baseline > threshold, event[x,y,z:], 0)
		


AVGPOOL = keras.layers.AveragePooling3D((1,1,8))
def customBCE(target, pred):
	'''
	Custom loss function, returns BCE(@target, @pred) + BCE( avgDownSample(1,1,8)(@target), avgDownSample(1,1,8)(@pred) ).
	'''

	pred_pool = AVGPOOL(pred)
	target_pool = AVGPOOL(target)
	#return tensorflow.math.reduce_mean( keras.losses.binary_crossentropy(target,pred) )
	return ( tensorflow.math.reduce_mean( keras.losses.binary_crossentropy(target_pool, pred_pool) )
			+ tensorflow.math.reduce_mean( keras.losses.binary_crossentropy(target, pred) ) )
