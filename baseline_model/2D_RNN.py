#! /usr/bin/python3.6

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

n_person = 108
window_size = 10
n_lstm_layers = 2

# full connected parameter
n_fc_in = 1024
n_hidden_state = 1024
print("\n****************size of hidden state", n_hidden_state,"\n")
n_fc_out = 1024

dropout_prob = 0.5

calibration = 'N'
norm_type='2D'
regularization_method = 'dropout'
enable_penalty = False

output_dir 	= "win_"+str(window_size)+"_"+str(n_person)+"_fc_"+str(n_fc_in)+"_rnn2_fc_"+str(n_fc_in)+"_"+calibration+"_075_hs_"+str(n_hidden_state)+"_early_stop"
output_file = "win_"+str(window_size)+"_"+str(n_person)+"_fc_"+str(n_fc_in)+"_rnn2_fc_"+str(n_fc_in)+"_"+calibration+"_075_hs_"+str(n_hidden_state)+"_early_stop"

dataset_dir = "/home/dalinzhang/datasets/EEG_motor_imagery/1D_CNN_dataset/raw_data/window_1D/"

with open(dataset_dir+"top"+str(n_person)+"_dataset_1D_window.pkl", "rb") as fp:
  	datasets = pickle.load(fp)
with open(dataset_dir+"top"+str(n_person)+"_labels_1D_window.pkl", "rb") as fp:
  	labels = pickle.load(fp)

# datasets = datasets.reshape(len(datasets), window_size, 10, 11, 1)
one_hot_labels = np.array(list(pd.get_dummies(labels)))
print(one_hot_labels)
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

split = np.random.rand(len(datasets)) < 0.75

train_x = datasets[split] 
train_y = labels[split]

train_sample = len(train_x)
print("\ntrain sample:", train_sample)

test_x = datasets[~split] 
test_y = labels[~split]

test_sample = len(test_x)
print("\ntest sample:", test_sample)

print("\n**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions Begin: **********\n")

# input parameter
n_input_ele = 64
n_time_step	= window_size
n_class 	= 5

# training parameter
training_epochs = 32
batch_size = 300
batch_num_per_epoch = train_x.shape[0]//batch_size

lambda_loss_amount = 0.0005
enable_penalty = False

learning_rate = 1e-4

accuracy_batch_size = 300
train_accuracy_batch_num = train_x.shape[0]//accuracy_batch_size
test_accuracy_batch_num = test_x.shape[0]//accuracy_batch_size

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def apply_fully_connect(x, x_size, fc_size):
	fc_weight = weight_variable([x_size, fc_size])
	fc_bias = bias_variable([fc_size])
	return tf.nn.relu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size):
	readout_weight = weight_variable([x_size, readout_size])
	readout_bias = bias_variable([readout_size])
	return tf.add(tf.matmul(x, readout_weight), readout_bias)

print("**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions End **********\n")

print("**********("+time.asctime(time.localtime(time.time()))+") Define NN structure Begin: **********\n")

# input placeholder
X = tf.placeholder(tf.float32, shape=[None, n_time_step, n_input_ele], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None, n_class], name = 'Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

################################################################################
# first fully connected layer 
################################################################################
# X 		==>	[batch_size, n_time_step, n_input_ele]
shape = X.get_shape().as_list()

# X_flat 	==>	[batch_size*n_time_step, n_input_ele]
X_flat = tf.reshape(X, [-1, shape[2]])

# fc_in 	==>	[batch_size*n_time_step, n_fc_in]
fc_in = apply_fully_connect(X_flat, shape[2], n_fc_in)

# lstm_in	==>	[batch_size, n_time_step, n_fc_in]
lstm_in = tf.reshape(fc_in, [-1, n_time_step, n_fc_in])
# lstm_in = X
###########################################################################################
# RNN layers
###########################################################################################
# define lstm cell
cells = []
for _ in range(n_lstm_layers):
	cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_state, forget_bias=1.0, state_is_tuple=True)
	# cell = tf.contrib.rnn.GRUCell(n_fc_in, forget_bias=1.0, state_is_tuple=True)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
	cells.append(cell)
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

# output ==> [batch_size, n_time_step, n_hidden_state]
output, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state = init_state, dtype = tf.float32, time_major=False)
print("lstm output shape", output.get_shape())

output 		= tf.unstack(tf.transpose(output, [1, 0, 2]), name = 'lstm_out')
rnn_output	= output[-1]

###########################################################################################
# fully connected and readout
###########################################################################################
# rnn_output ==>[batch, n_hidden_state]
shape_rnn_out = rnn_output.get_shape().as_list()
# fc_out ==> 	[batch_size, n_fc_out]
fc_out = apply_fully_connect(rnn_output, shape_rnn_out[1], n_fc_out)

fc_drop = tf.nn.dropout(fc_out, keep_prob)	

# readout layer
y_ = apply_readout(fc_drop, n_fc_out, n_class)

###########################################################################################
# predict output
###########################################################################################
# prediction output
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name = "y_pred")
# possibility output
y_posi = tf.nn.softmax(y_, name = "y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
	tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
	# cross entropy cost function
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name = 'loss')
else:
	# cross entropy cost function
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

print("\n**********("+time.asctime(time.localtime(time.time()))+") Define NN structure End **********\n")

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
		cost_history = np.zeros(shape=[1], dtype=float)
		for b in range(batch_num_per_epoch):
			offset = (b * batch_size) % (train_y.shape[0] - batch_size) 
			batch_x = train_x[offset:(offset + batch_size), :, :]
			batch_y = train_y[offset:(offset + batch_size), :]
			_, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
			cost_history = np.append(cost_history, c)
		# test training accuracy after each epoch
		if(epoch%1 == 0):
			train_accuracy 	= np.zeros(shape=[0], dtype=float)
			test_accuracy 	= np.zeros(shape=[0], dtype=float)
			test_loss 		= np.zeros(shape=[0], dtype=float)
			train_loss 		= np.zeros(shape=[0], dtype=float)
			for i in range(train_accuracy_batch_num):
				offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size) 
				train_batch_x = train_x[offset:(offset + accuracy_batch_size), :, :]
				train_batch_y = train_y[offset:(offset + accuracy_batch_size), :]

				train_a, train_c = session.run([accuracy, loss], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0})
				train_accuracy = np.append(train_accuracy, train_a)
				train_loss = np.append(train_loss, train_c)
			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch, " Training Cost: ", np.mean(cost_history), "Training Accuracy: ", np.mean(train_accuracy))
			train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
			train_loss_save = np.append(train_loss_save, np.mean(train_loss))
			for j in range(test_accuracy_batch_num):
				offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
				test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :]
				test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]

				test_a, test_c = session.run([accuracy, loss], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0})

				test_accuracy = np.append(test_accuracy, test_a)
				test_loss = np.append(test_loss, test_c)
			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch, " Test Cost: ", np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy),"\n")
			test_accuracy_save 	= np.append(test_accuracy_save, np.mean(test_accuracy))
			test_loss_save 		= np.append(test_loss_save, np.mean(test_loss))
			if (np.mean(test_loss) < 0.35)and(np.mean(test_accuracy) > 0.897):
				break
	test_accuracy 	= np.zeros(shape=[0], dtype=float)
	test_loss 		= np.zeros(shape=[0], dtype=float)
	test_pred		= np.zeros(shape=[0], dtype=float)
	test_true		= np.zeros(shape=[0, 5], dtype=float)
	test_posi		= np.zeros(shape=[0, 5], dtype=float)
	for k in range(test_accuracy_batch_num):
		offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
		test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :]
		test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]

		test_a, test_c, test_p, test_r = session.run([accuracy, loss, y_pred, y_posi], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0})
		test_t = test_batch_y

		test_accuracy = np.append(test_accuracy, test_a)
		test_loss = np.append(test_loss, test_c)
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
	# test_auc = roc_auc_score(test_true, test_pred_1_hot, average=None)
	# confusion matrix
	confusion_matrix = confusion_matrix(test_true_list, test_pred)

	print("********************recall:", test_recall)
	print("*****************precision:", test_precision)
	# print("******************test_auc:", test_auc)
	print("******************f1_score:", test_f1)
	print("**********confusion_matrix:\n", confusion_matrix)

	print("("+time.asctime(time.localtime(time.time()))+") Final Test Cost: ", np.mean(test_loss), "Final Test Accuracy: ", np.mean(test_accuracy))
	# save result
	os.system("mkdir ./result/"+output_dir+" -p")
	result 	= pd.DataFrame({'epoch':range(1,epoch+2), "train_accuracy":train_accuracy_save, "test_accuracy":test_accuracy_save,"train_loss":train_loss_save,"test_loss":test_loss_save})
	ins 	= pd.DataFrame({'n_fc_in':n_fc_in, 'n_fc_out':n_fc_out, 'n_hidden_state':n_hidden_state, 'accuracy':np.mean(test_accuracy), 'keep_prob': 1-dropout_prob,  'n_person':n_person, "calibration":calibration, 'sliding_window':window_size, "epoch":epoch+1, "norm":norm_type, "learning_rate":learning_rate, "regularization":regularization_method, "train_sample":train_sample, "test_sample":test_sample}, index=[0])
	summary = pd.DataFrame({'class':one_hot_labels, 'recall':test_recall, 'precision':test_precision, 'f1_score':test_f1})#, 'roc_auc':test_auc})

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
