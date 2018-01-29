import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
# cascade confusion matrix plot
# with open("/home/dadafly/program/Emotive/result/cascade/conv_3l_win_10_9_fc_1024_rnn2_fc_1024_N_075_train_feature_map_32_hs_64/confusion_matrix.pkl", "rb") as fp:
#   	cascade_cm = pickle.load(fp)
# cascade_cm=cascade_cm.astype(np.float64)
# cascade_cm[0]=cascade_cm[0]/sum(cascade_cm[0])
# cascade_cm[1]=cascade_cm[1]/sum(cascade_cm[1])
# cascade_cm[2]=cascade_cm[2]/sum(cascade_cm[2])
# cascade_cm[3]=cascade_cm[3]/sum(cascade_cm[3])
# cascade_cm[4]=cascade_cm[4]/sum(cascade_cm[4])
# 
# df_cm = pd.DataFrame(cascade_cm, index = [i for i in ["forward","backward","left","right","null"]], columns = [i for i in ["forward","backward","left","right","null"]])
# f=plt.figure(figsize = (100,70))
# ax=sn.heatmap(df_cm, annot=True, annot_kws={"size": 280, 'weight': 'bold'}, cbar=False, cmap="RdPu", vmin=0, vmax=1)
# # sn.set(font_scale=6)
# for label in ax.get_xticklabels():
# 	label.set_size(150)
# 	label.set_weight("bold")
# 	label.set_color("black")
# 	label.set_rotation(30)
#  
# for label in ax.get_yticklabels():
# 	label.set_size(150)
# 	label.set_weight("bold")
# 	label.set_color("black")
# 	label.set_rotation(0)
# #sn.heatmap(df_cm, annot=True, annot_kws={"size": 10, 'weight': 'bold'}, cmap=sn.light_palette((210, 90, 60), input="husl"), vmin=0, vmax=1)
# #sn.heatmap(df_cm, annot=True, annot_kws={"size": 10, 'weight': 'bold'}, cmap="YlGnBu", vmin=0, vmax=1)
# #sn.heatmap(df_cm, annot=True, annot_kws={"size": 52, 'weight': 'bold'}, cmap="Blues", vmin=0, vmax=1)
# #sn.heatmap(df_cm, annot=True, annot_kws={"size": 52, 'weight': 'bold'}, cmap="YlGnBu", vmin=0, vmax=1)
# #sn.heatmap(df_cm, annot=True)
# #sn.heatmap(df_cm, annot=True, cmap="Purples_r")
# #sn.heatmap(df_cm, annot=True, cmap=sn.cubehelix_palette(light=1, as_cmap=True))
# 
# plt.show()
# f.savefig('/home/dadafly/Doc/draft/tkde/casecade_cm.png')
# 
# # parallel confusion matrix plot
# 
# with open("/home/dadafly/program/Emotive/result/parallel/parallel_win_10_9_conv_3l_fc_1024__fc_1024_rnn2_fc_1024_hs_16_concat/confusion_matrix.pkl", "rb") as fp:
#   	parallel_cm = pickle.load(fp)
# parallel_cm=parallel_cm.astype(np.float64)
# parallel_cm[0]=parallel_cm[0]/sum(parallel_cm[0])
# parallel_cm[1]=parallel_cm[1]/sum(parallel_cm[1])
# parallel_cm[2]=parallel_cm[2]/sum(parallel_cm[2])
# parallel_cm[3]=parallel_cm[3]/sum(parallel_cm[3])
# parallel_cm[4]=parallel_cm[4]/sum(parallel_cm[4])
# 
# df_cm = pd.DataFrame(parallel_cm, index = [i for i in ["forward","backward","left","right","null"]], columns = [i for i in ["forward","backward","left","right","null"]])
# f=plt.figure(figsize = (100,70))
# ax=sn.heatmap(df_cm, annot=True, annot_kws={"size": 280, 'weight': 'bold'}, cbar=False, cmap="RdPu", vmin=0, vmax=1)
# #sn.set(font_scale=10)
# cax = plt.gcf().axes[-1]
# cax.tick_params(labelsize=150,width=100)
# for label_x in ax.get_xticklabels():
# 	label_x.set_weight("bold")
# 	label_x.set_color("black")
# 	label_x.set_rotation(30)
# 	label_x.set_size(150)
# 
# for label_y in ax.get_yticklabels():
# 	label_y.set_size(150)
# 	label_y.set_weight("bold")
# 	label_y.set_color("black")
# 	label_y.set_rotation(0)
# 
# 
# #sn.heatmap(df_cm, annot=True, annot_kws={"size": 10, 'weight': 'bold'}, cmap=sn.light_palette((210, 90, 60), input="husl"), vmin=0, vmax=1)
# #sn.heatmap(df_cm, annot=True, annot_kws={"size": 10, 'weight': 'bold'}, cmap="YlGnBu", vmin=0, vmax=1)
# #sn.heatmap(df_cm, annot=True, annot_kws={"size": 52, 'weight': 'bold'}, cmap="Blues", vmin=0, vmax=1)
# #sn.heatmap(df_cm, annot=True, annot_kws={"size": 52, 'weight': 'bold'}, cmap="YlGnBu", vmin=0, vmax=1)
# #sn.heatmap(df_cm, annot=True)
# #sn.heatmap(df_cm, annot=True, cmap="Purples_r")
# #sn.heatmap(df_cm, annot=True, cmap=sn.cubehelix_palette(light=1, as_cmap=True))
# 
# plt.show()
# f.savefig('/home/dadafly/Doc/draft/tkde/parallel_cm.png')

# parallel 3D confusion matrix plot

with open("/home/dadafly/program/Emotive/result/2D_CNN/conv_3l_9_fc_1024_N_075_train_feature_map_32/confusion_matrix.pkl", "rb") as fp:
  	parallel_cm = pickle.load(fp)
parallel_cm=parallel_cm.astype(np.float64)
parallel_cm[0]=parallel_cm[0]/sum(parallel_cm[0])
parallel_cm[1]=parallel_cm[1]/sum(parallel_cm[1])
parallel_cm[2]=parallel_cm[2]/sum(parallel_cm[2])
parallel_cm[3]=parallel_cm[3]/sum(parallel_cm[3])
parallel_cm[4]=parallel_cm[4]/sum(parallel_cm[4])

df_cm = pd.DataFrame(parallel_cm, index = [i for i in ["forward","backward","left","right","null"]], columns = [i for i in ["forward","backward","left","right","null"]])
f=plt.figure(figsize = (100,76))
ax=sn.heatmap(df_cm, annot=True, annot_kws={"size": 250, 'weight': 'bold'}, cbar=True, cmap="RdPu", vmin=0, vmax=1)
#sn.set(font_scale=10)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=150,width=100)
for label_x in ax.get_xticklabels():
	label_x.set_weight("bold")
	label_x.set_color("black")
	label_x.set_rotation(30)
	label_x.set_size(150)

for label_y in ax.get_yticklabels():
	label_y.set_size(150)
	label_y.set_weight("bold")
	label_y.set_color("black")
	label_y.set_rotation(0)


#sn.heatmap(df_cm, annot=True, annot_kws={"size": 10, 'weight': 'bold'}, cmap=sn.light_palette((210, 90, 60), input="husl"), vmin=0, vmax=1)
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 10, 'weight': 'bold'}, cmap="YlGnBu", vmin=0, vmax=1)
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 52, 'weight': 'bold'}, cmap="Blues", vmin=0, vmax=1)
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 52, 'weight': 'bold'}, cmap="YlGnBu", vmin=0, vmax=1)
#sn.heatmap(df_cm, annot=True)
#sn.heatmap(df_cm, annot=True, cmap="Purples_r")
#sn.heatmap(df_cm, annot=True, cmap=sn.cubehelix_palette(light=1, as_cmap=True))

plt.show()
f.savefig('/home/dadafly/Doc/draft/tkde/parallel_3D_cm.png')
