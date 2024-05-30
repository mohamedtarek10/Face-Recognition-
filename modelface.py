import numpy as np
import cv2
import os
from keras.layers import Dense, Flatten,Sequential,Convolution2D,MaxPool2D
from tensorflow.keras.optimizers import Adam

def model(input_shape, num_classes):

    model= Sequential()

    model.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=input_shape, activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(num_classes+1, activation='softmax'))
   

    

    # Compile model
    opt_adam = Adam(learning_rate=1e-3)
    model.compile(optimizer=opt_adam, loss='categorical_crossentropy', metrics=['acc'])

    return model
    
