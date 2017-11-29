#! /usr/bin/python3

import tensorflow as tf

class cnn:
	def __init__(
			self,
			weight_stddev	= 0.1,
			bias_constant	= 0.1,
			padding			= "SAME",
			):
			self.weight_stddev	= weight_stddev
			self.bias_constant	= bias_constant
			self.padding		= padding

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev = self.weigh_stddev)
		return tf.Variable(initial)


	def bias_variable(shape):
		initial = tf.constant(self.bias_constant, shape = shape)
	return tf.Variable(initial)


	def conv1d(x, W, kernel_stride):
	# API: must strides[0]=strides[4]=1
		return tf.nn.conv1d(x, W, stride=kernel_stride, padding=self.padding)


	def conv2d(x, W, kernel_stride):
	# API: must strides[0]=strides[4]=1
		return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding=self.padding)


	def conv3d(x, W, kernel_stride):
	# API: must strides[0]=strides[4]=1
		return tf.nn.conv3d(x, W, strides=[1, kernel_stride, kernel_stride, kernel_stride, 1], padding=self.padding)


	def apply_conv1d(x, filter_width, in_channels, out_channels, kernel_stride):
		weight = self.weight_variable([filter_width, in_channels, out_channels])
		bias = self.bias_variable([out_channels]) # each feature map shares the same weight and bias
		conv_1d = tf.add(conv1d(x, weight, kernel_stride), bias)
		# conv_1d_bn = batch_norm(conv_1d, out_channels, phase_train)
		return tf.nn.elu(conv_1d)


	def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
	# API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
		return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, 1], padding=self.padding)

	
	def apply_conv3d(x, filter_depth, filter_height, filter_width, in_channels, out_channels, kernel_stride):
		weight = weight_variable([filter_depth, filter_height, filter_width, in_channels, out_channels])
		bias = bias_variable([out_channels]) # each feature map shares the same weight and bias
		conv_3d = tf.add(conv3d(x, weight, kernel_stride), bias)
		return tf.nn.elu(conv_3d)


	def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
	# API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
		return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, 1], padding=self.padding)


	def apply_max_pooling3d(x, pooling_depth, pooling_height, pooling_width, pooling_stride):
	# API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
		return tf.nn.max_pool3d(x, ksize=[1, pooling_depth, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, pooling_stride, 1], padding=self.padding)

	
	def apply_fully_connect(x, x_size, fc_size):
		fc_weight = weight_variable([x_size, fc_size])
		fc_bias = bias_variable([fc_size])
		fc = tf.add(tf.matmul(x, fc_weight), fc_bias)
		return tf.nn.elu(fc)

	
	def apply_readout(x, x_size, readout_size):
		readout_weight = weight_variable([x_size, readout_size])
		readout_bias = bias_variable([readout_size])
		return tf.add(tf.matmul(x, readout_weight), readout_bias)
