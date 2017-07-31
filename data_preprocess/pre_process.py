#! /usr/bin/python3

########################################################
# EEG data preprocess for 3D CNN
########################################################
import os
import pyedflib
import numpy as np
import pandas as pd
from scipy import stats
import pickle

np.random.seed(0)
window_size = 10
dataset_dir = "/home/dadafly/datasets/EEG_motor_imagery/"

def read_data(file_name):
	f = pyedflib.EdfReader(file_name)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
	    sigbufs[i, :] = f.readSignal(i)
	sigbuf_transpose = np.transpose(sigbufs)
	signal = np.asarray(sigbuf_transpose)
	signal_labels = np.asarray(signal_labels)
	f._close()
	del f
	return signal, signal_labels

def data_1Dto2D(data, Y=10, X=11):
	data_2D = np.zeros([Y, X])
	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0, data[21], data[22], data[23], 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[1] = (	  	 0, 	   0,  	   	 0, data[24], data[25], data[26], data[27], data[28], 	   	 0,   	   0, 	 	 0) 
	data_2D[2] = (	  	 0, data[29], data[30], data[31], data[32], data[33], data[34], data[35], data[36], data[37], 	 	 0) 
	data_2D[3] = (	  	 0, data[38],  data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6], data[39], 		 0) 
	data_2D[4] = (data[42], data[40],  data[7],  data[8],  data[9], data[10], data[11], data[12], data[13], data[41], data[43]) 
	data_2D[5] = (	  	 0, data[44], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[45], 		 0) 
	data_2D[6] = (	  	 0, data[46], data[47], data[48], data[49], data[50], data[51], data[52], data[53], data[54], 		 0) 
	data_2D[7] = (	  	 0, 	   0, 	 	 0, data[55], data[56], data[57], data[58], data[59], 	   	 0, 	   0, 		 0) 
	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0, data[60], data[61], data[62], 	   0, 	   	 0, 	   0, 		 0) 
	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0, data[63], 		 0, 	   0, 	   	 0, 	   0, 		 0) 
	return data_2D

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data.nonzero()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized

def dataset_1Dto2D(dataset_1D):
	dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		dataset_2D[i] = data_1Dto2D(dataset_1D[i])
	return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
	norm_dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_2D[i] = feature_normalize(data_1Dto2D(dataset_1D[i]))
	return norm_dataset_2D

def windows(data, size):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += (size/2)

def segment_signal(data, label, window_size):
	for (start, end) in windows(data, window_size):
		if(len(data[start:end]) == window_size):
			if(start == 0):
				segments = data[start:end]
				labels = stats.mode(label["labels"][start:end])[0][0]
			else:
				segments = np.vstack([segments, data[start:end]])
				labels = np.append(labels, stats.mode(label["labels"][start:end])[0][0])
	return segments, labels

def segment_signal_without_transition(data, label, window_size):
	for (start, end) in windows(data, window_size):
		if((len(data[start:end]) == window_size) and (len(set(label[start:end]))==1)):
			if(start == 0):
				segments = data[start:end]
				# labels = stats.mode(label[start:end])[0][0]
				labels = np.array(list(set(label[start:end])))
			else:
				segments = np.vstack([segments, data[start:end]])
				labels = np.append(labels, np.array(list(set(label[start:end]))))
				# labels = np.append(labels, stats.mode(label[start:end])[0][0])
	return segments, labels

def normalize_segment_signal(data, label, window_size):
	for (start, end) in windows(data, window_size):
		if(len(data[start:end]) == window_size):
			if(start == 0):
				segments = data[start:end]
				normalized_segments = feature_normalize(segments)
				labels = np.array(list(set(label[start:end])))
				# labels = stats.mode(label["labels"][start:end])[0][0]
			else:
				norm_segments = feature_normalize(data[start:end])
				normalized_segments = np.vstack([normalized_segments, norm_segments])
				labels = np.append(labels, np.array(list(set(label[start:end]))))
				# labels = np.append(labels, stats.mode(label["labels"][start:end])[0][0])
	return normalized_segments, labels

def normalize_segment_signal_without_transition(data, label, window_size):
	for (start, end) in windows(data, window_size):
		if((len(data[start:end]) == window_size) and (len(set(label[start:end]))==1)):
			if(start == 0):
				segments = data[start:end]
				normalized_segments = feature_normalize(segments)
				labels = np.array(list(set(label[start:end])))
				# labels = stats.mode(label["labels"][start:end])[0][0]
			else:
				norm_segments = feature_normalize(data[start:end])
				normalized_segments = np.vstack([normalized_segments, norm_segments])
				labels = np.append(labels, np.array(list(set(label[start:end]))))
				# labels = np.append(labels, stats.mode(label["labels"][start:end])[0][0])
	return normalized_segments, labels
########################################################################################
# 1. for each person, segment each of his task to multiple windows
# 2. stack all window segments of all 14 tasks of one person together without indexing each task
# 3. all segments for all persons are concatenate to a list with person ID as index
# 4. final dataset is a list: first index: person; second index: window segment; third to fifth: 3-D of segments
# destination segments[person, sample_num, 3D_depth, 3D_height, 3D_weight]
########################################################################################

def apply_segment(dataset_dir, window_size):
	datadir_list=os.listdir(dataset_dir)
	segments = []
	labels = []
	index = []
	for j in range(len(datadir_list)):
		if(os.path.isdir(dataset_dir+datadir_list[j].strip())):
			index = np.append(index, "***person: "+datadir_list[j].strip())
			# dataset dir name for each person
			datadir_name = dataset_dir+datadir_list[j].strip()
			# dataset file list for each person
			datafile_list = os.popen("ls "+datadir_name+"/*.edf").readlines()
			# os.chdir(datadir_name)
			# initialize a segments array for each person
			j_segments = np.empty([0, window_size, 10, 11])
			j_labels = np.empty([0])
			# handle each task for one person
			for k in range(len(datafile_list)):
				datafile_name = datafile_list[k].strip()
				labelfile_name = datafile_list[k].rstrip(".edf\n")+".label.csv"
				file_name = os.path.split(datafile_list[k])[1].strip(".edf\n")
				if(("R04" in file_name) or ("R08" in file_name) or ("R12" in file_name)):
					print(file_name+" begin:")
					index = np.append(index, file_name)
					# read sensor data and convert to 2D
					dataset_1D, sensor_label = read_data(datafile_name)
					dataset_2D = dataset_1Dto2D(dataset_1D)
					# read label data
					labeldata = pd.read_csv(labelfile_name)
					# segment data and label
					k_segments, k_labels = normalize_segment_signal(dataset_2D, labeldata, window_size)
					k_segments = k_segments.reshape(int(k_segments.shape[0]/window_size), window_size, 10, 11)
					# labels = np.asarray(labels)
					# add each segments to the whole segments of one person
					with open(datadir_name+"/"+file_name+"_2D.pkl", "wb") as fp:
						pickle.dump(dataset_2D, fp) 
					with open(datadir_name+"/"+file_name+"_"+str(window_size)+"_segments.pkl", "wb") as fp:
						pickle.dump(k_segments, fp) 
					with open(datadir_name+"/"+file_name+"_"+str(window_size)+"_segments_labels.pkl", "wb") as fp:
						pickle.dump(k_labels, fp) 
					j_segments = np.vstack([j_segments, k_segments])
					j_labels = np.append(j_labels, k_labels)
				elif(("R06" in file_name) or ("R10" in file_name) or ("R14" in file_name)):
					print(file_name+" begin:")
					index = np.append(index, file_name)
					# read sensor data and convert to 2D
					dataset_1D, sensor_label = read_data(datafile_name)
					dataset_2D = dataset_1Dto2D(dataset_1D)
					# read label data
					labeldata = pd.read_csv(labelfile_name)
					# segment data and label
					k_segments, k_labels = normalize_segment_signal(dataset_2D, labeldata, window_size)
					k_segments = k_segments.reshape(int(k_segments.shape[0]/window_size), window_size, 10, 11)
					# labels = np.asarray(labels)
					# add each segments to the whole segments of one person
					with open(datadir_name+"/"+file_name+"_2D.pkl", "wb") as fp:
						pickle.dump(dataset_2D, fp) 
					with open(datadir_name+"/"+file_name+"_"+str(window_size)+"_segments.pkl", "wb") as fp:
						pickle.dump(k_segments, fp) 
					with open(datadir_name+"/"+file_name+"_"+str(window_size)+"_segments_labels.pkl", "wb") as fp:
						pickle.dump(k_labels, fp) 
					j_segments = np.vstack([j_segments, k_segments])
					j_labels = np.append(j_labels, k_labels)
				else:
					pass
			# os.system("touch "+datadir_name+"/"+datadir_list[j].strip()+"_"+str(window_size)+"_segments.pkl")
			# os.system("touch "+datadir_name+"/"+datadir_list[j].strip()+"_"+str(window_size)+"_segments_labels.pkl")
			with open(datadir_name+"/"+datadir_list[j]+"_"+str(window_size)+"_segments.pkl", "wb") as fp:
				pickle.dump(j_segments, fp) 
			with open(datadir_name+"/"+datadir_list[j]+"_"+str(window_size)+"_segments_labels.pkl", "wb") as fp:
				pickle.dump(j_labels, fp) 
			segments = segments + [j_segments]
			labels = labels + [j_labels]
		else:
			pass
	# save segment file
	with open(dataset_dir+str(window_size)+"_segments.pkl", "wb") as fp:
		pickle.dump(segments, fp) 
	with open(dataset_dir+str(window_size)+"_segments_labels.pkl", "wb") as fp:
		pickle.dump(labels, fp) 

	return segments, labels, index


# segments, labels, index = apply_segment(dataset_dir, window_size)

def apply_normalize_segment_signal(dataset_dir):
	norm_segments = []
	norm_labels = []
	datadir_list=os.listdir(dataset_dir)
	for j in range(len(datadir_list)):
		if(os.path.isdir(dataset_dir+datadir_list[j].strip())):
			print(datadir_list[j]+"begin")
			datadir_name = dataset_dir+datadir_list[j].strip()
			with open(datadir_name+"/"+datadir_list[j]+"_"+str(window_size)+"_segments.pkl", "rb") as fp:
				j_segments = pickle.load(fp)
			with open(datadir_name+"/"+datadir_list[j]+"_"+str(window_size)+"_segments_labels.pkl", "rb") as fp:
				j_norm_labels = pickle.load(fp)
			for i in range(len(j_segments)):
				j_segments[i] = feature_normalize(j_segments[i])
			with open(datadir_name+"/"+datadir_list[j]+"_"+str(window_size)+"_norm_segments.pkl", "wb") as fp:
				pickle.dump(j_segments, fp) 
			norm_segments = norm_segments + [j_segments]
			norm_labels = norm_labels + [j_norm_labels]
	with open(dataset_dir+str(window_size)+"_norm_segments.pkl", "wb") as fp:
		pickle.dump(norm_segments, fp) 
	with open(dataset_dir+str(window_size)+"_norm_labels.pkl", "wb") as fp:
		pickle.dump(norm_labels, fp) 
	return None

def apply_segment_without_transition(dataset_dir, window_size, start=1, end=108):
	segments = np.empty([0, window_size, 10, 11])
	labels = np.empty([0])
	for j in range(start, end):
		if(j<10):
			datadir_name = "/home/dadafly/datasets/EEG_motor_imagery/S00"+str(j)
		elif(j==89):
			j = 109	
			datadir_name = "/home/dadafly/datasets/EEG_motor_imagery/S"+str(j)
		elif(j<100):	
			datadir_name = "/home/dadafly/datasets/EEG_motor_imagery/S0"+str(j)
		else:	
			datadir_name = "/home/dadafly/datasets/EEG_motor_imagery/S"+str(j)
		datafile_list = os.popen("ls "+datadir_name+"/*.edf").readlines()
		j_segments = np.empty([0, window_size, 10, 11])
		j_labels = np.empty([0])
		for k in range(len(datafile_list)):
			labelfile_name = datafile_list[k].rstrip(".edf\n")+".label.csv"
			datafile_name = datafile_list[k].rstrip(".edf\n")+".csv"
			file_name = os.path.split(datafile_list[k])[1].strip(".edf\n")
			if(("R02" in file_name)):
				print(file_name+" begin:")
				dataset_1D = pd.read_csv(datafile_name)
				labeldata = pd.read_csv(labelfile_name)
				all_data = pd.concat([dataset_1D, labeldata], axis=1)
				all_data = all_data.loc[all_data['labels']!= 'rest']
				all_data = all_data.loc[(all_data['Fc5.']!= 0)&(all_data['Fc3.']!= 0)&(all_data['Fc1.']!= 0)&(all_data['Fcz.']!= 0)& \
										(all_data['Fc2.']!= 0)&(all_data['Fc4.']!= 0)&(all_data['Fc6.']!= 0)&(all_data['C5..']!= 0)& \
										(all_data['C3..']!= 0)&(all_data['C1..']!= 0)&(all_data['Cz..']!= 0)&(all_data['C2..']!= 0)& \
										(all_data['C4..']!= 0)&(all_data['C6..']!= 0)&(all_data['Cp5.']!= 0)&(all_data['Cp3.']!= 0)& \
										(all_data['Cp1.']!= 0)&(all_data['Cpz.']!= 0)&(all_data['Cp2.']!= 0)&(all_data['Cp4.']!= 0)& \
										(all_data['Cp6.']!= 0)&(all_data['Fp1.']!= 0)&(all_data['Fpz.']!= 0)&(all_data['Fp2.']!= 0)& \
										(all_data['Af7.']!= 0)&(all_data['Af3.']!= 0)&(all_data['Afz.']!= 0)&(all_data['Af4.']!= 0)& \
										(all_data['Af8.']!= 0)&(all_data['F7..']!= 0)&(all_data['F5..']!= 0)&(all_data['F3..']!= 0)& \
										(all_data['F1..']!= 0)&(all_data['Fz..']!= 0)&(all_data['F2..']!= 0)&(all_data['F4..']!= 0)& \
										(all_data['F6..']!= 0)&(all_data['F8..']!= 0)&(all_data['Ft7.']!= 0)&(all_data['Ft8.']!= 0)& \
										(all_data['T7..']!= 0)&(all_data['T8..']!= 0)&(all_data['T9..']!= 0)&(all_data['T10.']!= 0)& \
										(all_data['Tp7.']!= 0)&(all_data['Tp8.']!= 0)&(all_data['P7..']!= 0)&(all_data['P5..']!= 0)& \
										(all_data['P3..']!= 0)&(all_data['P1..']!= 0)&(all_data['Pz..']!= 0)&(all_data['P2..']!= 0)& \
										(all_data['P4..']!= 0)&(all_data['P6..']!= 0)&(all_data['P8..']!= 0)&(all_data['Po7.']!= 0)& \
										(all_data['Po3.']!= 0)&(all_data['Poz.']!= 0)&(all_data['Po4.']!= 0)&(all_data['Po8.']!= 0)& \
										(all_data['O1..']!= 0)&(all_data['Oz..']!= 0)&(all_data['O2..']!= 0)&(all_data['Iz..']!= 0)]
				k_labels = all_data['labels']
	
				all_data.drop('labels', axis=1, inplace=True)
				k_features = all_data.as_matrix()
				k_features = norm_dataset_1Dto2D(k_features)
				k_segments, k_labels = segment_signal_without_transition(k_features, k_labels, window_size)
				k_segments = k_segments.reshape(int(k_segments.shape[0]/window_size), window_size, 10, 11)
	
				j_segments = np.vstack([j_segments, k_segments])
				j_labels = np.append(j_labels, k_labels)
			if(("R04" in file_name)):
				print(file_name+" begin:")
				dataset_1D = pd.read_csv(datafile_name)
				labeldata = pd.read_csv(labelfile_name)
				all_data = pd.concat([dataset_1D, labeldata], axis=1)
				all_data = all_data.loc[all_data['labels']!= 'rest']
				all_data = all_data.loc[(all_data['Fc5.']!= 0)&(all_data['Fc3.']!= 0)&(all_data['Fc1.']!= 0)&(all_data['Fcz.']!= 0)& \
										(all_data['Fc2.']!= 0)&(all_data['Fc4.']!= 0)&(all_data['Fc6.']!= 0)&(all_data['C5..']!= 0)& \
										(all_data['C3..']!= 0)&(all_data['C1..']!= 0)&(all_data['Cz..']!= 0)&(all_data['C2..']!= 0)& \
										(all_data['C4..']!= 0)&(all_data['C6..']!= 0)&(all_data['Cp5.']!= 0)&(all_data['Cp3.']!= 0)& \
										(all_data['Cp1.']!= 0)&(all_data['Cpz.']!= 0)&(all_data['Cp2.']!= 0)&(all_data['Cp4.']!= 0)& \
										(all_data['Cp6.']!= 0)&(all_data['Fp1.']!= 0)&(all_data['Fpz.']!= 0)&(all_data['Fp2.']!= 0)& \
										(all_data['Af7.']!= 0)&(all_data['Af3.']!= 0)&(all_data['Afz.']!= 0)&(all_data['Af4.']!= 0)& \
										(all_data['Af8.']!= 0)&(all_data['F7..']!= 0)&(all_data['F5..']!= 0)&(all_data['F3..']!= 0)& \
										(all_data['F1..']!= 0)&(all_data['Fz..']!= 0)&(all_data['F2..']!= 0)&(all_data['F4..']!= 0)& \
										(all_data['F6..']!= 0)&(all_data['F8..']!= 0)&(all_data['Ft7.']!= 0)&(all_data['Ft8.']!= 0)& \
										(all_data['T7..']!= 0)&(all_data['T8..']!= 0)&(all_data['T9..']!= 0)&(all_data['T10.']!= 0)& \
										(all_data['Tp7.']!= 0)&(all_data['Tp8.']!= 0)&(all_data['P7..']!= 0)&(all_data['P5..']!= 0)& \
										(all_data['P3..']!= 0)&(all_data['P1..']!= 0)&(all_data['Pz..']!= 0)&(all_data['P2..']!= 0)& \
										(all_data['P4..']!= 0)&(all_data['P6..']!= 0)&(all_data['P8..']!= 0)&(all_data['Po7.']!= 0)& \
										(all_data['Po3.']!= 0)&(all_data['Poz.']!= 0)&(all_data['Po4.']!= 0)&(all_data['Po8.']!= 0)& \
										(all_data['O1..']!= 0)&(all_data['Oz..']!= 0)&(all_data['O2..']!= 0)&(all_data['Iz..']!= 0)]
				k_labels = all_data['labels']
	
				all_data.drop('labels', axis=1, inplace=True)
				k_features = all_data.as_matrix()
				k_features = norm_dataset_1Dto2D(k_features)
				k_segments, k_labels = segment_signal_without_transition(k_features, k_labels, window_size)
				k_segments = k_segments.reshape(int(k_segments.shape[0]/window_size), window_size, 10, 11)
	
				j_segments = np.vstack([j_segments, k_segments])
				j_labels = np.append(j_labels, k_labels)
			if(("R06" in file_name)):
				print(file_name+" begin:")
				dataset_1D = pd.read_csv(datafile_name)
				labeldata = pd.read_csv(labelfile_name)
				all_data = pd.concat([dataset_1D, labeldata], axis=1)
				all_data = all_data.loc[all_data['labels']!= 'rest']
				all_data = all_data.loc[(all_data['Fc5.']!= 0)&(all_data['Fc3.']!= 0)&(all_data['Fc1.']!= 0)&(all_data['Fcz.']!= 0)& \
										(all_data['Fc2.']!= 0)&(all_data['Fc4.']!= 0)&(all_data['Fc6.']!= 0)&(all_data['C5..']!= 0)& \
										(all_data['C3..']!= 0)&(all_data['C1..']!= 0)&(all_data['Cz..']!= 0)&(all_data['C2..']!= 0)& \
										(all_data['C4..']!= 0)&(all_data['C6..']!= 0)&(all_data['Cp5.']!= 0)&(all_data['Cp3.']!= 0)& \
										(all_data['Cp1.']!= 0)&(all_data['Cpz.']!= 0)&(all_data['Cp2.']!= 0)&(all_data['Cp4.']!= 0)& \
										(all_data['Cp6.']!= 0)&(all_data['Fp1.']!= 0)&(all_data['Fpz.']!= 0)&(all_data['Fp2.']!= 0)& \
										(all_data['Af7.']!= 0)&(all_data['Af3.']!= 0)&(all_data['Afz.']!= 0)&(all_data['Af4.']!= 0)& \
										(all_data['Af8.']!= 0)&(all_data['F7..']!= 0)&(all_data['F5..']!= 0)&(all_data['F3..']!= 0)& \
										(all_data['F1..']!= 0)&(all_data['Fz..']!= 0)&(all_data['F2..']!= 0)&(all_data['F4..']!= 0)& \
										(all_data['F6..']!= 0)&(all_data['F8..']!= 0)&(all_data['Ft7.']!= 0)&(all_data['Ft8.']!= 0)& \
										(all_data['T7..']!= 0)&(all_data['T8..']!= 0)&(all_data['T9..']!= 0)&(all_data['T10.']!= 0)& \
										(all_data['Tp7.']!= 0)&(all_data['Tp8.']!= 0)&(all_data['P7..']!= 0)&(all_data['P5..']!= 0)& \
										(all_data['P3..']!= 0)&(all_data['P1..']!= 0)&(all_data['Pz..']!= 0)&(all_data['P2..']!= 0)& \
										(all_data['P4..']!= 0)&(all_data['P6..']!= 0)&(all_data['P8..']!= 0)&(all_data['Po7.']!= 0)& \
										(all_data['Po3.']!= 0)&(all_data['Poz.']!= 0)&(all_data['Po4.']!= 0)&(all_data['Po8.']!= 0)& \
										(all_data['O1..']!= 0)&(all_data['Oz..']!= 0)&(all_data['O2..']!= 0)&(all_data['Iz..']!= 0)]
				k_labels = all_data['labels']
	
				all_data.drop('labels', axis=1, inplace=True)
				k_features = all_data.as_matrix()
				k_features = norm_dataset_1Dto2D(k_features)
				k_segments, k_labels = segment_signal_without_transition(k_features, k_labels, window_size)
				k_segments = k_segments.reshape(int(k_segments.shape[0]/window_size), window_size, 10, 11)
	
				j_segments = np.vstack([j_segments, k_segments])
				j_labels = np.append(j_labels, k_labels)
			else:
				pass
		segments = np.vstack([segments, j_segments])
		labels = np.append(labels, j_labels)
	index = np.array(range(0, len(labels)))
	np.random.shuffle(index)
	shuffled_segments = segments[index]
	shuffled_labels = labels[index]
	return shuffled_segments, shuffled_labels

# for end in [20, 40, 60, 80, 100, 108]:
# 	shuffled_segments, shuffled_labels = apply_segment_without_transition(dataset_dir, window_size, 1, end+1)
# 	with open(dataset_dir+"3D_CNN/calibration_data/cal_top"+str(end)+"_shuffle_dataset_3D.pkl", "wb") as fp:
# 		pickle.dump(shuffled_segments, fp, protocol=4) 
# 	with open(dataset_dir+"3D_CNN/calibration_data/cal_top"+str(end)+"_shuffle_labels_3D.pkl", "wb") as fp:
# 		pickle.dump(shuffled_labels, fp)

def apply_raw_segment_without_transition(dataset_dir, window_size, start=1, end=109):
	segments = np.empty([0, window_size, 10, 11])
	labels = np.empty([0])
	for j in range(start, end):
		if(j<10):
			datadir_name = "/home/dadafly/datasets/EEG_motor_imagery/S00"+str(j)
		elif(j==89):
			j = 109	
			datadir_name = "/home/dadafly/datasets/EEG_motor_imagery/S"+str(j)
		elif(j<100):	
			datadir_name = "/home/dadafly/datasets/EEG_motor_imagery/S0"+str(j)
		else:	
			datadir_name = "/home/dadafly/datasets/EEG_motor_imagery/S"+str(j)
		datafile_list = os.popen("ls "+datadir_name+"/*.edf").readlines()
		j_segments = np.empty([0, window_size, 10, 11])
		j_labels = np.empty([0])
		for k in range(len(datafile_list)):
			labelfile_name = datafile_list[k].rstrip(".edf\n")+".label.csv"
			datafile_name = datafile_list[k].rstrip(".edf\n")+".csv"
			file_name = os.path.split(datafile_list[k])[1].strip(".edf\n")
			if(("R02" in file_name)):
				print(file_name+" begin:")
				dataset_1D = pd.read_csv(datafile_name)
				labeldata = pd.read_csv(labelfile_name)
				all_data = pd.concat([dataset_1D, labeldata], axis=1)
				all_data = all_data.loc[all_data['labels']!= 'rest']
				k_labels = all_data['labels']
	
				all_data.drop('labels', axis=1, inplace=True)
				k_features = all_data.as_matrix()
				k_features = norm_dataset_1Dto2D(k_features)
				k_segments, k_labels = segment_signal_without_transition(k_features, k_labels, window_size)
				# k_features = dataset_1Dto2D(k_features)
				# k_segments, k_labels = normalize_segment_signal_without_transition(k_features, k_labels, window_size)
				k_segments = k_segments.reshape(int(k_segments.shape[0]/window_size), window_size, 10, 11)
	
				j_segments = np.vstack([j_segments, k_segments])
				j_labels = np.append(j_labels, k_labels)
			if(("R04" in file_name)):
				print(file_name+" begin:")
				dataset_1D = pd.read_csv(datafile_name)
				labeldata = pd.read_csv(labelfile_name)
				all_data = pd.concat([dataset_1D, labeldata], axis=1)
				all_data = all_data.loc[all_data['labels']!= 'rest']
				k_labels = all_data['labels']
	
				all_data.drop('labels', axis=1, inplace=True)
				k_features = all_data.as_matrix()
				k_features = norm_dataset_1Dto2D(k_features)
				k_segments, k_labels = segment_signal_without_transition(k_features, k_labels, window_size)
				# k_features = dataset_1Dto2D(k_features)
				# k_segments, k_labels = normalize_segment_signal_without_transition(k_features, k_labels, window_size)
				k_segments = k_segments.reshape(int(k_segments.shape[0]/window_size), window_size, 10, 11)
	
				j_segments = np.vstack([j_segments, k_segments])
				j_labels = np.append(j_labels, k_labels)
			if(("R06" in file_name)):
				print(file_name+" begin:")
				dataset_1D = pd.read_csv(datafile_name)
				labeldata = pd.read_csv(labelfile_name)
				all_data = pd.concat([dataset_1D, labeldata], axis=1)
				all_data = all_data.loc[all_data['labels']!= 'rest']
				k_labels = all_data['labels']
	
				all_data.drop('labels', axis=1, inplace=True)
				k_features = all_data.as_matrix()
				k_features = norm_dataset_1Dto2D(k_features)
				k_segments, k_labels = segment_signal_without_transition(k_features, k_labels, window_size)
				# k_segments, k_labels = normalize_segment_signal_without_transition(k_features, k_labels, window_size)
				# k_features = dataset_1Dto2D(k_features)
				k_segments = k_segments.reshape(int(k_segments.shape[0]/window_size), window_size, 10, 11)
	
				j_segments = np.vstack([j_segments, k_segments])
				j_labels = np.append(j_labels, k_labels)
			else:
				pass
		segments = np.vstack([segments, j_segments])
		labels = np.append(labels, j_labels)
	index = np.array(range(0, len(labels)))
	np.random.shuffle(index)
	shuffled_segments = segments[index]
	shuffled_labels = labels[index]
	return shuffled_segments, shuffled_labels

# for end in [20, 40, 60, 80, 100, 108]:
for end in [108]:
	# shuffled_segments, shuffled_labels = apply_raw_segment_without_transition(dataset_dir, window_size, 1, end+1)
	shuffled_segments, shuffled_labels = apply_raw_segment_without_transition(dataset_dir, window_size, 91, end+1)
	# with open(dataset_dir+"3D_CNN/raw_data/top"+str(end)+"_shuffle_dataset_3D_win_10.pkl", "wb") as fp:
	with open(dataset_dir+"3D_CNN/raw_data/last108_shuffle_dataset_3D_win_10.pkl", "wb") as fp:
		pickle.dump(shuffled_segments, fp, protocol=4) 
	# with open(dataset_dir+"3D_CNN/raw_data/top"+str(end)+"_shuffle_labels_3D_win_10.pkl", "wb") as fp:
	with open(dataset_dir+"3D_CNN/raw_data/last108_shuffle_labels_3D_win_10.pkl", "wb") as fp:
		pickle.dump(shuffled_labels, fp)

###############################################################################################
# train test split 
###############################################################################################

# person independent train test split
# person_split = np.random.rand(len(segments)) < 0.75
# 
# train_person_segments = list(compress(segments, person_split))
# train_person_labels = list(compress(labels, person_split))
# 
# test_person_segments = list(compress(segments, ~person_split))
# test_person_labels = list(compress(labels, ~person_split))
# 
# # person dependent train test split
# for i in range(len(train_person_segments)):
# 	if(i==0):
# 		segments_mix = train_person_segments[i]
# 		labels_mix = train_person_labels[i]
# 	else:
# 		segments_mix = np.vstack([segments_mix, train_person_segments[i]])
# 		labels_mix = np.append(labels_mix, train_person_labels[i])
# 
# labels_mix = np.asarray(pd.get_dummies(labels_mix), dtype = np.int8)
# person_dependent_split = np.random.rand(len(segments_mix)) < 0.75
# 
# train_dependent_segments = segments_mix[person_dependent_split] 
# train_dependent_labels = labels_mix[person_dependent_split]
# 
# test_dependent_segments = segments_mix[~person_dependent_split] 
# test_dependent_labels = labels_mix[~person_dependent_split]




#######################################################################################

# labelfile_name = "/home/dadafly/datasets/EEG_motor_imagery/S002/S002R09.label.csv"
# datafile_name = "/home/dadafly/datasets/EEG_motor_imagery/S002/S002R09.edf"
# 
# dataset_1D, sensor_label = read_data(datafile_name)
# dataset_2D = dataset_1Dto2D(dataset_1D)
# 
# labeldata = pd.read_csv(labelfile_name)
# 
# segments, labels = segment_signal(dataset_2D, labeldata, window_size)
# segments = segments.reshape(int(segments.shape[0]/window_size), window_size, 10, 11)
# # labels = np.asarray(labels)



###############################################################################
# sensor map matrix
###############################################################################

# matrix = np.zeros([10,11], dtype = object)
# 
# matrix[0] = (	  "0", 		  "0", 	 	 "0", 		"0", label[21], label[22], label[23], 		"0", 	   "0", 	  "0", 		 "0") 
# matrix[1] = (	  "0", 		  "0", 	 	 "0", label[24], label[25], label[26], label[27], label[28], 	   "0", 	  "0", 		 "0") 
# matrix[2] = (	  "0",  label[29], label[30], label[31], label[32], label[33], label[34], label[35], label[36], label[37], 		 "0") 
# matrix[3] = (	  "0",  label[38], 	label[0],  label[1],  label[2],  label[3],  label[4],  label[5],  label[6], label[39], 		 "0") 
# matrix[4] = (label[42], label[40], 	label[7],  label[8],  label[9], label[10], label[11], label[12], label[13], label[41], label[43]) 
# matrix[5] = (	  "0",  label[44], label[14], label[15], label[16], label[17], label[18], label[19], label[20], label[45], 		 "0") 
# matrix[6] = (	  "0",  label[46], label[47], label[48], label[49], label[50], label[51], label[52], label[53], label[54], 		 "0") 
# matrix[7] = (	  "0", 		  "0", 	 	 "0", label[55], label[56], label[57], label[58], label[59], 	   "0", 	  "0", 		 "0") 
# matrix[8] = (	  "0", 		  "0", 	 	 "0", 		"0", label[60], label[61], label[62], 		"0", 	   "0", 	  "0", 		 "0") 
# matrix[9] = (	  "0", 		  "0", 	 	 "0", 		"0", 	   "0", label[63], 		 "0", 		"0", 	   "0", 	  "0", 		 "0") 

#############################################################
# test output transpose 
#############################################################

# data, label = read_data(file_name)
# print(data.shape)
# 
# for i in range(len(label)):
# 	print(i, label[i])

#############################################################
# test all sensor data are in the same sequence
#############################################################

# file_name_1 = "/home/dadafly/datasets/EEG_motor_imagery/S001/S001R01.edf"
# file_name_2 = "/home/dadafly/datasets/EEG_motor_imagery/S002/S002R09.edf"
# file_name_3 = "/home/dadafly/datasets/EEG_motor_imagery/S002/S002R08.edf"
# file_name_4 = "/home/dadafly/datasets/EEG_motor_imagery/S100/S100R14.edf"
# 
# data, label_1 = read_data(file_name_1)
# data, label_2 = read_data(file_name_2)
# data, label_3 = read_data(file_name_3)
# data, label_4 = read_data(file_name_4)
# 
# if(label_1 == label_2):
# 	print("yes 1")
# if(label_3 == label_2):
# 	print("yes 2")
# if(label_3 == label_4):
# 	print("yes 3")
# if(label_1 == label_4):
# 	print("yes 4")
