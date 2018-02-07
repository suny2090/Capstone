from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
import glob

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import copy

from caiman.utils.utils import download_demo
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import estimate_components_quality
from skimage.util.montage import montage2d
from caiman.components_evaluation import evaluate_components

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten

import json as simplejson
from keras.models import model_from_json
from sklearn.utils import class_weight as cw
from caiman.utils.image_preprocessing_keras import ImageDataGenerator

# the data, shuffled and split between train and test sets
with np.load('/mnt/home/jshangying/CaImAn/final_label/ground_truth_components_curated_minions_cleaned_4.npz') as ld:
    print(ld.keys())
    all_masks_gt = ld['all_masks_gt_cur']
    labels_gt = ld['labels_gt_cur']

    
batch_size = 256
num_classes = 4
epochs = 1000
test_fraction = 0.2
augmentation = True
# input image dimensions
img_rows, img_cols = 50, 50

    
x_train, x_test, y_train, y_test = train_test_split(all_masks_gt, labels_gt, test_size = test_fraction ) 

class_weight = cw.compute_class_weight('balanced', np.unique(y_train), y_train)
#print(y_train[0],y_train.shape)  

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
#%%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=5e-5, decay=1e-6)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])
model.summary()
#%%
if augmentation:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
    #            featurewise_center=True,
    #            featurewise_std_normalization=True,
                shear_range=0.3,
                rotation_range=360,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range = [0.8,1.2],
                horizontal_flip=True,
                vertical_flip=True,
                random_mult_range = [.25,2]
                )

#history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        verbose=1,
                        class_weight = class_weight,
                        validation_data=(x_test, y_test))
else:
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
#%%
import pickle
#result = [step_list,acc,val_acc,loss,val_loss]
with open('vgg_like_16.p','wb') as fp:
    pickle.dump(history.history,fp)

#%%
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% Save model and weights
import datetime
save_dir = '/mnt/home/syang/CaImAn/use_cases/data_minions/'
model_name = str(datetime.datetime.now()).replace(' ','-').replace(':','-')
model_json = model.to_json()
with open(json_path, "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

print('Saved trained model at %s ' % json_path)
    

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name+'.h5')
model.save(model_path)
print('Saved trained model at %s ' % model_path)

