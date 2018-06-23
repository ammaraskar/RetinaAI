# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import glob
import json
import os
from PIL import Image
import numpy as np
import time

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_rotation(max_angle=10.)
img_aug.add_random_blur(sigma_max=2.)

# Define our network architecture:

# Input is a 72x28 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 28, 72, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, 72, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 128, 3, activation='relu')

# Step 4: Convolution yet again
# network = conv_2d(network, 128, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.85)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='checkpoints/eye_position.tfl.ckpt')

# Save model when training is complete to a file
model.load("./eye_position_classifier.tfl")
print("Network loaded from eye_position_classifier.tfl!")

import cv2
import face_recognition
from collect_data import resolve_corners

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 1920)
video_capture.set(4, 1080)
print("VideoCapture initialized")

while True:
    time.sleep(0.5)

    _, image = video_capture.read()

    face_landmarks_list = face_recognition.face_landmarks(image)

    if len(face_landmarks_list) == 0:
        continue
    landmarks = face_landmarks_list[0]
    if ('left_eye' not in landmarks) or ('right_eye' not in landmarks):
        continue

    left_bounds = resolve_corners(landmarks['left_eye'])
    right_bounds = resolve_corners(landmarks['right_eye'])

    #cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(image)

    left_eye = pil_im.crop(left_bounds)
    right_eye = pil_im.crop(right_bounds)

    combined = Image.new('RGB', (72, 28))
    combined.paste(left_eye, (0, 0))
    combined.paste(right_eye, (36, 0))
    combined = np.array(combined, dtype=np.float64)

    X = np.array([combined])

    print(model.predict(X))