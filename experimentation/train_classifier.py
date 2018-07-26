# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import glob
import json
import os
from PIL import Image
import numpy as np

X = []
Y = []

def generate_label(meta):
    if int(meta['x']) < 750:
        if int(meta['y']) > 500:
            return [1, 0, 0, 0]
        else:
            return [0, 1, 0, 0]
    else:
        if int(meta['y']) > 500:
            return [0, 0, 1, 0]
        else:
            return [0, 0, 0, 1]

# Load in the data
images = glob.glob('data/images/*.json')
for image in images:
    image_id = os.path.basename(image).split('_')[0]
    with open('data/images/{}_meta.json'.format(image_id), 'r') as f:
        image_meta = json.load(f)

    left_eye = Image.open('data/images/{}_left.png'.format(image_id)).resize((36, 14))
    right_eye = Image.open('data/images/{}_right.png'.format(image_id)).resize((36, 14))

    combined = Image.new('RGB', (72, 28))
    combined.paste(left_eye, (0, 0))
    combined.paste(right_eye, (36, 0))
    combined = np.array(combined, dtype=np.float64)

    X.append(combined)
    Y.append(generate_label(image_meta))

Y = np.array(Y)

# take the first 10 examples as validation
X_test = X[:10]
Y_test = Y[:10]

X = X[10:]
Y = Y[10:]
# Shuffle the data
X, Y = shuffle(X, Y)

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_rotation(max_angle=15.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:

# Input is a 72x28 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 28, 72, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, 72, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

# Step 3: Convolution again
network = conv_2d(network, 128, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 128, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.90)

# Step 8: Fully-connected neural network with four outputs to make the final prediction
network = fully_connected(network, 4, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='checkpoints/eye_position.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=350, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=150,
          snapshot_epoch=True,
          run_id='eye-classifier')

# Save model when training is complete to a file
model.save("eye_position_classifier.tfl")
print("Network trained and saved as eye_position_classifier.tfl!")