# Face-Recognition
Face Recognition using CNN

## Overview

This project uses a Convolutional Neural Network (CNN) model to implement a face recognition system.

## Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- sklearn
- keras
- Image
- numpy
- matplotlib

## Data Preprocessing

- The images are loaded, resized, and normalized to prepare them for training.

## Model Architecture

The CNN model consists of several convolutional layers. The final layer uses softmax activation for face recognition.

## Data Augmentation

Data augmentation techniques enhance the model's generalization by applying random transformations to the training images.

## Training

The model is trained using the training dataset with the Adam optimizer and sparse_categorical_crossentropy loss. Callbacks such as ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping are used to monitor and improve training performance.

## Real-time

## Haar Cascade Classifier

The Haar Cascade classifier is employed for real-time face detection. The classifier is pre-trained and included in the OpenCV library.

The system is tested on a video file or cam (replace `video_path` with the path to your video or 0 for Camera) by capturing frames and applying face mask detection in real time.


