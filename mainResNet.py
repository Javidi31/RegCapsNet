"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py

Modified by:   
@author: Malihe Javidi, m.javidi@qiet.ac.ir
"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np
import ResnetOrg
import os
import cv2
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import utilities.loadDataForResNet as DS

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


lr_reducer    = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger    = CSVLogger('resnet18_Cedar_TrNo6.csv') #######

batch_size    = 32   #######
nb_epoch      = 50   #######
data_augmentation  = False
img_rows, img_cols = 200, 200  #######
img_channels     = 1           #######
num_classes      = 55          ####### 
VALIDATION_SPLIT = 0.2         #######


"""# Load Dataset"""
X_train, y_train, X_test, y_test = DS.loadDataset()


model = ResnetOrg.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',f1_m,precision_m, recall_m])


MODEL_DIR = ".\\ResNetModels"  #######
filepath = "saved-model-{epoch:02d}-{val_acc:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR,"model-{epoch:02d}-{val_acc:.4f}_Cedar_Tr6.h5"), #######
                             monitor = "val_acc", save_best_only = True)


print('Not using data augmentation.')
history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=VERBOSE,validation_split=VALIDATION_SPLIT,
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, csv_logger,checkpoint])
          
score = model.evaluate(X_test, Y_test, verbose=0)