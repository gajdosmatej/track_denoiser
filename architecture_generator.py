import numpy
from tensorflow import keras
import tensorflow
import matplotlib.pyplot
import keras.utils
import os
import copy

def get_random_model():
	input_shape = (12,14,208,1)
	num_additional_conv_range = (0,3)
	num_sampling_range = (0, 3)
	kernel_size_ranges = ((2, 5), (2, 5), (3,10))
	filter_num_range = (1, 4)	#/5
	activation = "relu"
	padding = (1,1,1)

	num_samplings = numpy.random.randint(*num_sampling_range)
	min_num_conv = 2*num_samplings + 1
	num_additional_conv = min( numpy.random.randint(*num_additional_conv_range), min_num_conv)
	i = 0
	additional_conv_positions = []
	while i < num_additional_conv:	#positions of additional convolutional layers
		pos = numpy.random.randint(min_num_conv)
		if pos not in additional_conv_positions:
			additional_conv_positions.append(pos)
			i += 1

	layers_str = ["keras.layers.Input(shape=" + str(input_shape) + ")"]
	layers = [keras.layers.Input(shape=input_shape)]
	current_conv_num = 0
	last_was_conv = False
	additional_done = False
	already_upsampled = 0
	for i in range(2*num_samplings + min_num_conv + num_additional_conv):
		if last_was_conv:	#time for downsample/upsample
			last_was_conv = False
			additional_done = False
			if already_upsampled < num_samplings:	#downsample
				already_upsampled += 1
				layers_str.append("keras.layers.AveragePooling3D(pool_size=(1,1,2)),")
				layers.append(keras.layers.AveragePooling3D(pool_size=(1,1,2)))
				input_shape = (input_shape[0], input_shape[1], input_shape[2] // 2)
			else:	#upsample
				layers_str.append("keras.layers.UpSampling3D(size=(1,1,2)),")
				layers.append(keras.layers.UpSampling3D(size=(1,1,2)))
				input_shape = (input_shape[0], input_shape[1], input_shape[2] * 2)
		
		elif current_conv_num in additional_conv_positions and not additional_done:	#time for additional convolution
			kernel_size = tuple(numpy.random.randint(*kernel_size_ranges[i]) for i in range(3))
			num_filters = numpy.random.randint(*filter_num_range) * 5
			layers_str.append("keras.layers.Conv3D(padding=\"same\", strides=(1,1,1), kernel_size=" + str(kernel_size) 
							+ ", filters=" + str(num_filters) + ", activation=\"relu\"),")
			layers.append(keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=kernel_size, filters=num_filters, activation="relu"))
			additional_done = True
			
		elif not last_was_conv:	#time for mandatory convolution
			last_was_conv = True
			kernel_size = tuple(numpy.random.randint(*kernel_size_ranges[i]) for i in range(3))
			num_filters = numpy.random.randint(*filter_num_range) * 5
			if i+1 < 2*num_samplings + min_num_conv + num_additional_conv:
				layers_str.append("keras.layers.Conv3D(padding=\"same\", strides=(1,1,1), kernel_size=" + str(kernel_size) 
								+ ", filters=" + str(num_filters) + ", activation=\"relu\"),")
				layers.append(keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=kernel_size, filters=num_filters, 
								activation="relu"))
			else:
				layers_str.append("keras.layers.Conv3D(padding=\"same\", strides=(1,1,1), kernel_size=" + str(kernel_size) 
								+ ", filters=1, activation=\"relu\"),")
				layers.append(keras.layers.Conv3D(padding="same", strides=(1,1,1), kernel_size=kernel_size, filters=1, 
								activation="sigmoid"))
			current_conv_num += 1
	
	return (layers_str, keras.Sequential(layers))
