# Face-Recognition System for Attendance 
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

## Model Architecture

The CNN model used is VGG16 Architecture. The final layer uses softmax activation for face recognition u.

## Data Augmentation

Data augmentation techniques enhance the model's generalization by applying random transformations to the training images.

## Training

The model is trained using the training dataset with the Adam optimizer and sparse_categorical_crossentropy loss. Callbacks such as ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping are used to monitor and improve training performance.

## Real-time

## Haar Cascade Classifier

The Haar Cascade classifier is employed for real-time face recognition. The classifier is pre-trained and included in the OpenCV library.

The system is tested on a video file or cam (replace `video_path` with the path to your video or 0 for Camera) by capturing frames and applying face recognition in real-time.


