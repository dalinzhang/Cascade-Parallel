#! /usr/bin/python3.6
########################################################
# implement RNN for EEG decode
########################################################
from cnn_class import cnn
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

###########################################################################
# set dataset parameters
###########################################################################
# size of one labeled sample 
# 10x64

# prediction class
num_labels = 5

# id o begin subject
begin_subject = 1

# id of end subject
end_subject = 108

# sliding window size
window_size = 10

# size of the vector of one input time step
n_input_ele = 64

# train test split
train_test_split = 0.75

# dataset directory
dataset_dir = "/home/dalinzhang/datasets/EEG_motor_imagery/1D_CNN_dataset/raw_data/window_1D/"

# load dataset and label
with open(dataset_dir+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_1D_win_"+str(window_size)+".pkl", "rb") as fp:
  	datasets = pickle.load(fp)
with open(dataset_dir+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_1D_win_"+str(window_size)+".pkl", "rb") as fp:
  	labels = pickle.load(fp)

# set label to one hot
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

# train test split
split = np.random.rand(len(datasets)) < train_test_split 

train_x = datasets[split]
train_y = labels[split]

test_x = datasets[~split] 
test_y = labels[~split]

# print label
one_hot_labels = np.array(list(pd.get_dummies(labels)))
print(one_hot_labels)

###########################################################################
# set model parameters
###########################################################################
# set the number of lsmt layer
n_lstm_layers = 2

# set the input fully connected layer size
n_fc_in = 1024

# set the input fully connected layer size
n_fc_out = 1024

# set the hidden state of lstm unit
n_hidden_state = 1024

# set the time step of RNN/set the number of lstm unit in one RNN layer
n_time_step	= window_size

###########################################################################
# set training parameters
###########################################################################
# set learning rate
learning_rate = 1e-4

# set maximum traing epochs
training_epochs = 200

# set batch size
batch_size = 100

# set dropout probability
dropout_prob = 0.5

# set whether use L2 regularization
enable_penalty = False

# set L2 penalty
lambda_loss_amount = 0.0005

# set train batch number per epoch
batch_num_per_epoch = train_x.shape[0]//batch_size

# set test batch number per epoch
accuracy_batch_size = 300
train_accuracy_batch_num = train_x.shape[0]//accuracy_batch_size
test_accuracy_batch_num = test_x.shape[0]//accuracy_batch_size

###########################################################################
# for output record
###########################################################################

# regularization method
if enable_penalty:
	regularization_method = 'dropout+l2'
else:
	regularization_method = 'dropout'

# result output
result_dir = "/home/dadafly/experiment_result/eeg_physiobank_rnn_result"
output_dir 	= "win_"+str(window_size)+"_"+str(begin_subjec)+"_"+str(end_subject)+"_fc_"+str(n_fc_in)+"_RNN"+str(n_lstm_layers)+"_fc_"+str(n_fc_out)+"_"+regularization_method+"_"+str(format(train_test_split*100, '03d')+"_hs_"+str(n_hidden_state))
output_file = "win_"+str(window_size)+"_"+str(begin_subjec)+"_"+str(end_subject)+"_fc_"+str(n_fc_in)+"_RNN"+str(n_lstm_layers)+"_fc_"+str(n_fc_out)+"_"+regularization_method+"_"+str(format(train_test_split*100, '03d')+"_hs_"+str(n_hidden_state))

os.system("mkdir "+result_dir+"/"+output_dir+" -p")
###########################################################################
# build network
###########################################################################

# instance cnn class
rnn = cnn()

# input placeholder
X = tf.placeholder(tf.float32, shape=[None, n_time_step, n_input_ele], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None, num_labels], name = 'Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

####################### input fully connected layer ##########################
# X 		==>	[batch_size, n_time_step, n_input_ele]
shape = X.get_shape().as_list()

# X_flat 	==>	[batch_size*n_time_step, n_input_ele]
X_flat = tf.reshape(X, [-1, shape[2]])

# fc_in 	==>	[batch_size*n_time_step, n_fc_in]
fc_in = rnn.apply_fully_connect(X_flat, shape[2], n_fc_in)

# lstm_in	==>	[batch_size, n_time_step, n_fc_in]
lstm_in = tf.reshape(fc_in, [-1, n_time_step, n_fc_in])

################################# RNN layer ###################################
# define RNN network
cells = []
for _ in range(n_lstm_layers):
	cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_state, forget_bias=1.0, state_is_tuple=True)
	# cell = tf.contrib.rnn.GRUCell(n_fc_in, forget_bias=1.0, state_is_tuple=True)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
	cells.append(cell)
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)
init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

# feed RNN network
# output ==> [batch_size, n_time_step, n_hidden_state]
output, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state = init_state, dtype = tf.float32, time_major=False)

# extract the output of the last time step
output 		= tf.unstack(tf.transpose(output, [1, 0, 2]), name = 'lstm_out')
rnn_output	= output[-1]

####################### output fully connected layer ##########################
# rnn_output ==>[batch, n_hidden_state]
shape_rnn_out = rnn_output.get_shape().as_list()

# fc_out ==> 	[batch_size, n_fc_out]
fc_out = rnn.apply_fully_connect(rnn_output, shape_rnn_out[1], n_fc_out)

# Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
fc_drop = tf.nn.dropout(fc_out, keep_prob)	

# readout layer
y_ = rnn.apply_readout(fc_drop, n_fc_out, num_labels)

# possibility prediction 
y_posi = tf.nn.softmax(y_, name = "y_posi")

# class prediction 
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name = "y_pred")

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

# set training SGD optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))

# calculate prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

###########################################################################
# train test and save result
###########################################################################

# run with gpu memory growth
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
		# training process
		for b in range(batch_num_per_epoch):
			offset = (b * batch_size) % (train_y.shape[0] - batch_size) 
			batch_x = train_x[offset:(offset + batch_size), :, :]
			batch_y = train_y[offset:(offset + batch_size), :]
			_, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
			cost_history = np.append(cost_history, c)
		# calculate train and test accuracy after each training epoch
		if(epoch%1 == 0):
			train_accuracy 	= np.zeros(shape=[0], dtype=float)
			test_accuracy 	= np.zeros(shape=[0], dtype=float)
			test_loss 		= np.zeros(shape=[0], dtype=float)
			train_loss 		= np.zeros(shape=[0], dtype=float)
			# calculate train accuracy after each training epoch
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
			# calculate test accuracy after each training epoch
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
###########################################################################
# save result and model after training 
###########################################################################
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
	
	# recall
	test_recall = recall_score(test_true, test_pred_1_hot, average=None)
	# precision
	test_precision = precision_score(test_true, test_pred_1_hot, average=None)
	# f1 score
	test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
	confusion_matrix = confusion_matrix(test_true_list, test_pred)

	with open("./result/"+output_dir+"/confusion_matrix.pkl", "wb") as fp:
  		pickle.dump(confusion_matrix, fp)

	print("("+time.asctime(time.localtime(time.time()))+") Final Test Cost: ", np.mean(test_loss), "Final Test Accuracy: ", np.mean(test_accuracy))
	# save result
	result 	= pd.DataFrame({'epoch':range(1,epoch+2), "train_accuracy":train_accuracy_save, "test_accuracy":test_accuracy_save,"train_loss":train_loss_save,"test_loss":test_loss_save})
	ins 	= pd.DataFrame({'n_fc_in':n_fc_in, 'n_fc_out':n_fc_out, 'n_hidden_state':n_hidden_state, 'accuracy':np.mean(test_accuracy), 'keep_prob': 1-dropout_prob, 'sliding_window':window_size, "epoch":epoch+1, "learning_rate":learning_rate, "regularization":regularization_method}, index=[0])
	summary = pd.DataFrame({'class':one_hot_labels, 'recall':test_recall, 'precision':test_precision, 'f1_score':test_f1})

	writer = pd.ExcelWriter("./result/"+output_dir+"/"+output_file+".xlsx")
	# save model implementation paralmeters
	ins.to_excel(writer, 'condition', index=False)
	# save train/test accuracy and loss for each epoch
	result.to_excel(writer, 'result', index=False)
	# save recall/precision/f1 for each class
	summary.to_excel(writer, 'summary', index=False)
	# save fpr, tpr, auc for each class
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

	# save model
	saver = tf.train.Saver()
	saver.save(session, "./result/"+output_dir+"/model_"+output_file)
print("**********("+time.asctime(time.localtime(time.time()))+") Train and Test NN End **********\n")
