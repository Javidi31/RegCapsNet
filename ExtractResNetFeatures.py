# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:01:10 2019

@author: Malihe Javidi, m.javidi@qiet.ac.ir
"""

############################Visualization Exatract ResNet Features for X_train and X_test data

from keras.models import load_model
from keras import models
from keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np
import pickle  
from keras import backend as K
    


#Just run bellow code because wwhen save model we use recall and other similar measures  
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#End Just

#First I load the saved model 
model = load_model('model-79-0.6706_Mix_Cedar_MCYT_Tr12.h5',custom_objects={"f1_m": f1_m, "precision_m":precision_m,"recall_m":recall_m})#######

#then use it to visualize
layer_no = 33  #######     Extracts the outputs of the top 33 layers
number   = 16  #######     Extract Features of the outputs of the 16th layer
layer_outputs = [layer.output for layer in model.layers[1:layer_no]] # Extracts the outputs of the top 33 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activation_model.summary()


#Extract Train ResNet Features
maxSampleNo = X_train.shape[0]
for sampleNo in range(0,maxSampleNo):  
    sample = X_train[sampleNo,:,:,0]
    img_tensor = image.img_to_array(sample)
    img_tensor = np.expand_dims(img_tensor, axis=0)    
    activations = activation_model.predict(img_tensor)                                     
    layer_activation = activations[number]
    out=layer_activation[0, :, :, :]
    ###save Train Features
    f = open(str(sampleNo)+'Feat_Tr12_Mix_Cedar_MCYTGPDS_lay16.pckl', 'wb')#######    
    pickle.dump( out, f)
    f.close()


#Extract Test ResNet Features
maxSampleNo = X_test.shape[0]
for sampleNo in range(0,maxSampleNo):  
    sample = X_test[sampleNo,:,:,0]
    img_tensor = image.img_to_array(sample)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    activations = activation_model.predict(img_tensor)   
    layer_activation = activations[number]
    out=layer_activation[0, :, :, :]
    ###save Test Features
    f = open(str(sampleNo)+'Feat_Te5_Mix_Cedar_MCYTGPDS_lay16.pckl', 'wb')#######
    pickle.dump( out, f)
    f.close()