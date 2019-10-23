#! usr/bin/python
## CNN model for cancer-specific CDR3 sequence prediction

import sys,os
import numpy as np
import keras
from keras import optimizers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from data_io_blosum import read_file,blosum_matrix,enc_list_bl,GetFeatureLabels
import sklearn
from sklearn import metrics

# Nm is the max sequence length in our dataset
# Nf is number of features for each amino acid
Nm=17
Nf=20
# Let's get features for each amino acid
blosum_file='BLOSUM62.txt'
blosum=blosum_matrix(blosum_file)    
# Obtain training and testing data
# For training let's take first 2000 sequences from cancer data and
# first 2000 sequences from healthy donors
# For test dataset we take 200 cancer sequences and 200 sequences from healthy
# donors 

ftrain_cancer='Data/Cancer.txt'
ftrain_healthy='Data/Healthy.txt'
Train_Cancer=read_file(ftrain_cancer)
Train_Healthy=read_file(ftrain_healthy)
TestFeature, TestLabels=GetFeatureLabels(Train_Cancer[2000:2200,],Train_Healthy[2000:2200,],Nm,blosum)
TrainFeature, TrainLabels=GetFeatureLabels(Train_Cancer[0:2000,],Train_Healthy[0:2000,],Nm,blosum)
# Let's reshape data
TrainFeature=TrainFeature.reshape(TrainFeature.shape[0], Nf, Nm, 1)
TestFeature=TestFeature.reshape(TestFeature.shape[0], Nf, Nm, 1)
    
# Define CNN model
input_shape = (Nf, Nm, 1)
inp=Input(shape=input_shape)
conv_1 = Conv2D(8, kernel_size=(Nf, 2),  activation="relu", padding="valid")(inp)
pool1 = MaxPooling2D(pool_size=(1, 2),strides=(1,1)) (conv_1)
conv_2 = Conv2D(16, kernel_size=(1,2), activation="relu", padding="valid") (pool1)
pool2 = MaxPooling2D(pool_size=(1, 2),strides=(1,1)) (conv_2)
flat = Flatten()(pool2)
out = Dense(10, activation='relu')(flat)
drop = Dropout(0.5)(out)
out_fin = Dense(1, activation='sigmoid') (drop)
model = Model(inputs=[inp], outputs=[out_fin])
model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])
                           
# Let's train a model and save a model with the highest validation accuracy         
mc=ModelCheckpoint('best_model.h5',monitor='val_acc',mode='max',verbose=1,save_best_only=True)             
model.fit([TrainFeature],  TrainLabels, epochs=25, batch_size=100, verbose=1, 
          callbacks=[mc], validation_data=([TestFeature], TestLabels))  
          
#
saved_model=load_model('best_model.h5')
predictions = saved_model.predict([TestFeature])
# estimate ROC_AUC score for testing data
roc_auc_score = metrics.roc_auc_score(TestLabels, predictions)
print(roc_auc_score)
# val_acc: 0.76
# roc_auc_score = 0.82
# pip list
# keras version: 2.2.4
# scikit-learn: 0.21.2