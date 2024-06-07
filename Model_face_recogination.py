import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import cv2
import os
import h5py
from imutils import face_utils
from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization,GlobalAveragePooling2D,Convolution2D,MaxPool2D
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split
from keras import regularizers
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input
from tensorflow.keras.models import Model





def model(input_shape, num_classes):
    # Load the VGG16 model, pre-trained on ImageNet, but without the top classification layer
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    
    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Create the custom top layers for our task
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
    predictions = Dense(num_classes+1, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    opt_adam = Adam(learning_rate=1e-3)
    model.compile(optimizer=opt_adam, loss='categorical_crossentropy', metrics=['acc'])

    return model
    
