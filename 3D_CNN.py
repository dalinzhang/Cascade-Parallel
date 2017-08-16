#! /usr/bin/python3

########################################################
# Implementing a 3D CNN for EEG classification
# mainly refering to:
# 1: URL: http://aqibsaeed.github.io/2016-11-04-human-activity-recognition-cnn/ 
# 2: URL: https://github.com/jibikbam/CNN-3D-images-Tensorflow/blob/master/simpleCNN_MRI.py 
# 3: URL: http://shuaizhang.tech/2016/12/08/Tensorflow%E6%95%99%E7%A8%8B2-Deep-MNIST-Using-CNN/ 
########################################################

import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time

np.random.seed(33)

conv_1_shape = '3*3*3*1*32'
pool_1_shape = 'None'

conv_2_shape = '3*3*3*1*64'
pool_2_shape = 'None'

conv_3_shape = '3*3*3*1*128'
pool_3_shape = 'None'

conv_4_shape = 'None'
pool_4_shape = 'None'

n_person = 108
window_size = 10
fc_size = 1024
dropout_prob = 0.5

calibration = 'N'
norm_type='2D'
regularization_method = 'dropout+l2'
enable_penalty = True 

output_dir 	= "conv_3l_win_10_108_l2_N_summary_075_train_posibility_roc"
output_file = "conv_3l_win_10_108_fc_1024_l2_N_summary_075_train_posibility_roc"

dataset_dir = "/home/dalinzhang/datasets/EEG_motor_imagery/3D_CNN_dataset/raw_data/"

with open(dataset_dir+"top108_shuffle_dataset_3D_win_10.pkl", "rb") as fp:
  	datasets = pickle.load(fp)
with open(dataset_dir+"top108_shuffle_labels_3D_win_10.pkl", "rb") as fp:
  	labels = pickle.load(fp)

datasets = datasets.reshape(len(datasets), window_size, 10, 11, 1)
one_hot_labels = np.array(list(pd.get_dummies(labels)))
print("\n",one_hot_labels)
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)


split = np.random.rand(len(datasets)) < 0.75

train_x = datasets[split] 
train_y = labels[split]

train_sample = len(train_x)
print("train sample:", train_sample)

test_x = datasets[~split] 
test_y = labels[~split]

test_sample = len(test_x)
print("test sample:", test_sample)

print("**********("+time.asctime(time.localtime(time.time()))+") Load and Split dataset End **********\n")

print("**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions Begin: **********\n")

# input parameter
input_channel_num = 1

input_depth = window_size
input_height = 10
input_width = 11

num_labels = 5

# training parameter

lambda_loss_amount = 0.0005
training_epochs = 500

batch_size = 100
batch_num_per_epoch = train_x.shape[0]//batch_size

accuracy_batch_size = 300
train_accuracy_batch_num = train_x.shape[0]//accuracy_batch_size
test_accuracy_batch_num = test_x.shape[0]//accuracy_batch_size
# kernel parameter
kernel_depth_1st = 3
kernel_height_1st= 3
kernel_width_1st = 3

kernel_depth_2nd = 3
kernel_height_2nd= 3
kernel_width_2nd = 3

kernel_depth_3rd = 3
kernel_height_3rd= 3
kernel_width_3rd = 3

kernel_depth_4th = 3
kernel_height_4th= 3
kernel_width_4th = 3

kernel_stride 	 = 1

conv_channel_num = 32

# pooling parameter
pooling_depth 	= 2
pooling_height 	= 2
pooling_width 	= 2

pooling_stride = 2
# full connected parameter
fc_size = 1024

# algorithn parameter
learning_rate = 1e-4


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv3d(x, W, kernel_stride):
# API: must strides[0]=strides[4]=1
	return tf.nn.conv3d(x, W, strides=[1, kernel_stride, kernel_stride, kernel_stride, 1], padding='SAME')

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def apply_conv3d(x, filter_depth, filter_height, filter_width, in_channels, out_channels, kernel_stride, phase_train):
	weight = weight_variable([filter_depth, filter_height, filter_width, in_channels, out_channels])
	bias = bias_variable([out_channels]) # each feature map shares the same weight and bias
	conv_3d = tf.add(conv3d(x, weight, kernel_stride), bias)
	# conv_3d_bn = tf.contrib.layers.batch_norm(conv_3d, decay=0.5, center=True, scale=True, is_training=phase_train)
	# the last dimension of conv3d output indicates different output channel
	# the firs dimension of conv3d output indicates different object in one batch 
	return tf.nn.elu(conv_3d)

def apply_max_pooling(x, pooling_depth, pooling_height, pooling_width, pooling_stride):
# API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
	return tf.nn.max_pool3d(x, ksize=[1, pooling_depth, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, pooling_stride, 1], padding='VALID')

def apply_fully_connect(x, x_size, fc_size, phase_train):
	fc_weight = weight_variable([x_size, fc_size])
	fc_bias = bias_variable([fc_size])
	fc = tf.add(tf.matmul(x, fc_weight), fc_bias)
	# fc_bn = tf.contrib.layers.batch_norm(fc, decay=0.5, center=True, scale=True, is_training=phase_train, scope='fc_batch_norm')
	return tf.nn.elu(fc)

def apply_readout(x, x_size, readout_size):
	readout_weight = weight_variable([x_size, readout_size])
	readout_bias = bias_variable([readout_size])
	return tf.add(tf.matmul(x, readout_weight), readout_bias)

print("**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions End **********\n")

print("**********("+time.asctime(time.localtime(time.time()))+") Define NN structure Begin: **********\n")


# input placeholder
X = tf.placeholder(tf.float32, shape=[None, input_depth, input_height, input_width, input_channel_num], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None, num_labels], name = 'Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

# first CNN layer
conv_1 = apply_conv3d(X, kernel_depth_1st, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride, phase_train)
# conv1_drop = tf.nn.dropout(conv_1, keep_prob)	
# pool_1 = apply_max_pooling(conv_1, pooling_depth, pooling_height, pooling_width, pooling_stride)

# second CNN layer
conv_2 = apply_conv3d(conv_1, kernel_depth_2nd, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num*2, kernel_stride, phase_train)
# conv2_drop = tf.nn.dropout(conv_2, keep_prob)	
# pool_2 = apply_max_pooling(conv_2, pooling_depth, pooling_height, pooling_width, pooling_stride)

# third CNN layer
conv_3 = apply_conv3d(conv_2, kernel_depth_3rd, kernel_height_3rd, kernel_width_3rd, conv_channel_num*2, conv_channel_num*4, kernel_stride, phase_train)
# conv3_drop = tf.nn.dropout(conv_3, keep_prob)	
# pool_3 = apply_max_pooling(conv_3, pooling_depth, pooling_height, pooling_width, pooling_stride)

# fourth CNN layer
# conv_4 = apply_conv3d(conv_3, kernel_depth_4th, kernel_height_4th, kernel_width_4th, conv_channel_num*4, conv_channel_num*8, kernel_stride, phase_train)

# fully connected layer
shape = conv_3.get_shape().as_list()

pool_2_flat = tf.reshape(conv_3, [-1, shape[1]*shape[2]*shape[3]*shape[4]])
fc = apply_fully_connect(pool_2_flat, shape[1]*shape[2]*shape[3]*shape[4], fc_size, phase_train)

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
fc_drop = tf.nn.dropout(fc, keep_prob)	

# readout layer
y_ = apply_readout(fc_drop, fc_size, num_labels)
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name = "y_pred")
y_posi = tf.nn.softmax(y_, name = "y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
	tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
	# cross entropy cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y)) + l2
else:
	# cross entropy cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.AdagradDAOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.FtrlOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

print("**********("+time.asctime(time.localtime(time.time()))+") Define NN structure End **********\n")

print("**********("+time.asctime(time.localtime(time.time()))+") Train and Test NN Begin: **********\n")
# run
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
	session.run(tf.global_variables_initializer())
	train_accuracy_save = np.zeros(shape=[0], dtype=float)
	test_accuracy_save 	= np.zeros(shape=[0], dtype=float)
	test_loss_save 		= np.zeros(shape=[0], dtype=float)
	train_loss_save 	= np.zeros(shape=[0], dtype=float)
	for epoch in range(training_epochs):
		cost_history = np.zeros(shape=[0], dtype=float)
		for b in range(batch_num_per_epoch):
			offset = (b * batch_size) % (train_y.shape[0] - batch_size) 
			batch_x = train_x[offset:(offset + batch_size), :, :, :, :]
			batch_y = train_y[offset:(offset + batch_size), :]
			_, c = session.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-dropout_prob, phase_train: True})
			cost_history = np.append(cost_history, c)
		# test training accuracy after each epoch
		if(epoch%1 == 0):
			train_accuracy 	= np.zeros(shape=[0], dtype=float)
			test_accuracy	= np.zeros(shape=[0], dtype=float)
			test_loss 		= np.zeros(shape=[0], dtype=float)
			train_loss 		= np.zeros(shape=[0], dtype=float)
			for i in range(train_accuracy_batch_num):
				offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size) 
				train_batch_x = train_x[offset:(offset + accuracy_batch_size), :, :, :, :]
				train_batch_y = train_y[offset:(offset + accuracy_batch_size), :]
				
				train_a, train_c = session.run([accuracy, cost], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0, phase_train: False})
				
				train_loss = np.append(train_loss, train_c)
				train_accuracy = np.append(train_accuracy, train_a)
			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Training Cost: ", np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
			train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
			train_loss_save = np.append(train_loss_save, np.mean(train_loss))
			for j in range(test_accuracy_batch_num):
				offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
				test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
				test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
				
				test_a, test_c = session.run([accuracy, cost], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, phase_train: False})
				
				test_accuracy = np.append(test_accuracy, test_a)
				test_loss = np.append(test_loss, test_c)

			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Test Cost: ", np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy),"\n")
			test_accuracy_save 	= np.append(test_accuracy_save, np.mean(test_accuracy))
			test_loss_save 		= np.append(test_loss_save, np.mean(test_loss))
			# if (np.mean(test_accuracy) > 0.885):
			# 	break
	test_accuracy 	= np.zeros(shape=[0], dtype=float)
	test_loss 		= np.zeros(shape=[0], dtype=float)
	test_pred		= np.zeros(shape=[0], dtype=float)
	test_true		= np.zeros(shape=[0, 5], dtype=float)
	test_posi		= np.zeros(shape=[0, 5], dtype=float)
	for k in range(test_accuracy_batch_num):
		offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
		test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
		test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
		
		test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, phase_train: False})
		test_t = test_batch_y

		test_accuracy 	= np.append(test_accuracy, test_a)
		test_loss 		= np.append(test_loss, test_c)
		test_pred 		= np.append(test_pred, test_p)
		test_true 		= np.vstack([test_true, test_t])
		test_posi		= np.vstack([test_posi, test_r])
	# test_true = tf.argmax(test_true, 1)
	test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype = np.int8)
	test_true_list	= tf.argmax(test_true, 1).eval()
	print(test_pred.shape)
	print(test_pred)
	print(test_true.shape)
	print(test_true)
	
	# recall
	test_recall = recall_score(test_true, test_pred_1_hot, average=None)
	# precision
	test_precision = precision_score(test_true, test_pred_1_hot, average=None)
	# f1 score
	test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
	# auc
	test_auc = roc_auc_score(test_true, test_pred_1_hot, average=None)
	# confusion matrix
	confusion_matrix = confusion_matrix(test_true_list, test_pred)

	print("********************recall:", test_recall)
	print("*****************precision:", test_precision)
	print("******************test_auc:", test_auc)
	print("******************f1_score:", test_f1)
	print("**********confusion_matrix:\n", confusion_matrix)

	print("("+time.asctime(time.localtime(time.time()))+") Final Test Cost: ", np.mean(test_loss), "Final Test Accuracy: ", np.mean(test_accuracy))
	# save result
	os.system("mkdir ./result/"+output_dir+" -p")
	result 	= pd.DataFrame({'epoch':range(1,epoch+2), "train_accuracy":train_accuracy_save, "test_accuracy":test_accuracy_save,"train_loss":train_loss_save,"test_loss":test_loss_save})
	ins 	= pd.DataFrame({'conv_1':conv_1_shape, 'pool_1':pool_1_shape, 'conv_2':conv_2_shape, 'pool_2':pool_2_shape, 'conv_3':conv_3_shape, 'pool_3':pool_3_shape, 'conv_4':conv_4_shape, 'pool_3':pool_3_shape, 'fc':fc_size,'accuracy':np.mean(test_accuracy), 'keep_prob': 1-dropout_prob, 'n_person':n_person, "calibration":calibration, 'sliding_window':window_size, "epoch":epoch+1, "norm":norm_type, "learning_rate":learning_rate, "regularization":regularization_method}, index=[0])
	summary = pd.DataFrame({'class':one_hot_labels, 'recall':test_recall, 'precision':test_precision, 'f1_score':test_f1, 'roc_auc':test_auc})

	writer = pd.ExcelWriter("./result/"+output_dir+"/"+output_file+".xlsx")
	ins.to_excel(writer, 'condition', index=False)
	result.to_excel(writer, 'result', index=False)
	summary.to_excel(writer, 'summary', index=False)
	# fpr, tpr, auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	i = 0
	for key in one_hot_labels:
		fpr[key], tpr[key], _ = roc_curve(test_true[:, i], test_posi[:, i])
		roc_auc[key] = auc(fpr[key], tpr[key])
		roc = pd.DataFrame({"fpr":fpr[key], "tpr":tpr[key], "roc_auc":roc_auc[key]})
		roc.to_excel(writer, key, index=False)
		i += 1
	writer.save()
	with open("./result/"+output_dir+"/confusion_matrix.pkl", "wb") as fp:
  		pickle.dump(confusion_matrix, fp)
	# save model
	saver = tf.train.Saver()
	saver.save(session, "./result/"+output_dir+"/model_"+output_file)
print("**********("+time.asctime(time.localtime(time.time()))+") Train and Test NN End **********\n")































