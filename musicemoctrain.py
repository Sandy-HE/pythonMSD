# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:49:34 2019

@author: 13035338
"""

import numpy as np

#X_data = np.load("A_pitchdata_300seg_withtags.npz")
X_data = np.load("A_timbredata_300seg_withtags.npz")
#X_data = np.load("A_loudnessdata_300seg_withtags.npz")
X_data = X_data['arr_0']
X_data = X_data.transpose([2,0,1])

Y_data = np.load("A_pitchdata_300seg_withtags_Y_new.npz")
Y_data = Y_data['arr_0']

#train:test = 9:1
X_train_num = 1854

seg_num=300
feature_dim=12

X_train= X_data[0:X_train_num,]
#X_train=X_train.reshape(-1,1,seg_num,feature_dim)
X_test = X_data[X_train_num:,]
#X_test=X_test.reshape(-1,1,seg_num,feature_dim)
Y_train = Y_data[0:X_train_num,]
Y_test = Y_data[X_train_num:,]



from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import Adam

def MusicTaggerCNN():
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    
    batch_size = 128
    #nb_classes = 10
    nb_epoch = 50
    feq_num = 12
    
    model= Sequential()
    
    # Conv layer 1 output shape (32, 300, 12)
    model.add(Convolution2D(
        batch_input_shape=(None, 1, seg_num,feq_num),
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',     # Padding method
        data_format='channels_first',
    ))
    model.add(Activation('relu'))
    
    # Pooling layer 1 (max pooling) output shape (32, 150, 6)
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',    # Padding method
        data_format='channels_first',
    ))
    
    # Conv layer 2 output shape (64, 150, 6)
    model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    
    # Pooling layer 2 (max pooling) output shape (64, 75, 3)
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
    
    # Fully connected layer 1 input shape (64 * 75 * 3) = (14400), output shape (1024)
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    
    # Fully connected layer 2 to shape (40) for 40 classes
    model.add(Dense(40))
    model.add(Activation('softmax'))
    
    # Another way to define your optimizer
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size,)

    return model


def MusicTaggerGRU():
    
    TIME_STEPS = 300     # same as the height of the image
    INPUT_SIZE = 12     # same as the width of the image
    BATCH_SIZE = 50
    #BATCH_INDEX = 0
    OUTPUT_SIZE = 40
    CELL_SIZE = 50
    LR = 0.001
    # build RNN model
    model = Sequential()

    # RNN cell
    model.add(LSTM(
        # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
        # Otherwise, model.evaluate() will get error.
        batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        output_dim=CELL_SIZE,
        #unroll=True,
    ))

    #model.add(Dropout(0.5))
    # output layer
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))
    
    # optimizer
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=50, batch_size=50,)
    
    return model

#model = MusicTaggerCNN()
model = MusicTaggerGRU()

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])