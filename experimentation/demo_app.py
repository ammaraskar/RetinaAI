import sys
import random
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon, QPainter, QImage, QColor
import cv2
import face_recognition
from collect_data import resolve_corners, increase_bounds
from PIL import Image
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


COLORS = [
    (0,174,239),
    (239,62,54),
    (23,190,187),
    (255,219,156),
    (121,112,122)
]

def get_model():
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
    network = dropout(network, 0.95)

    # Step 8: Fully-connected neural network with four outputs to make the final prediction
    network = fully_connected(network, 4, activation='softmax')

    # Tell tflearn how we want to train the network
    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)

    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='checkpoints/eye_position.tfl.ckpt')

    # Save model when training is complete to a file
    model.load("./eye_position_classifier.tfl")
    print("Network loaded from eye_position_classifier.tfl!")
    return model
 
class App(QWidget):
    image_data = QtCore.pyqtSignal(np.ndarray)
 
    def __init__(self):
        super().__init__()

        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(3, 1920)
        self.video_capture.set(4, 1080)
        print("VideoCapture initialized")

        self.timer = QtCore.QBasicTimer()
        self.image = QImage()
        self.image_data.connect(self.on_image)
        self.results = []

        self.model = get_model()

        self.initUI()
 
    def initUI(self):
        button = QPushButton('Quit', self)
        button.move(20,20) 
        button.clicked.connect(QtCore.QCoreApplication.quit)

        self.timer.start(0, self)
        self.showFullScreen()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setPen(Qt.black)

        # Draw the picture
        qp.drawImage(0, 0, self.image)

        # Draw quadrants
        # Vertical line in the middle
        qp.drawLine(self.width() / 2, 0, self.width() / 2, self.height())
        # Horizontal line in the middle
        qp.drawLine(0, self.height() / 2, self.width(), self.height() / 2)

        # Draw circles to represent where people are looking
        for i, res in enumerate(self.results):
            color = reversed(COLORS[i % len(COLORS)])
            color = QColor(*color)
            qp.setPen(color)
            qp.setBrush(color)
            radius = self.height() / 18

            offset = i * 20

            if res == 0:
                # top left
                qp.drawEllipse(self.width() / 4, self.height() / 4 + offset, radius, radius)
            elif res == 1:
                # bottom left
                qp.drawEllipse(self.width() / 4, self.height() * 3/4 + offset, radius, radius)
            elif res == 2:
                # top right
                qp.drawEllipse(self.width() * 3 / 4, self.height() / 4 + offset, radius, radius)
            else:
                # bottom right
                qp.drawEllipse(self.width() * 3 / 4, self.height() * 3 / 4 + offset, radius, radius)

    def on_image(self, image):
        self.image = self.get_qimage(image)
        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width

        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image


    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return

        _, image = self.video_capture.read()
        small_frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        face_landmarks_list = face_recognition.face_landmarks(small_frame)
        if len(face_landmarks_list) == 0:
            return

        results = []
        for i, landmarks in enumerate(face_landmarks_list):
            if ('left_eye' not in landmarks) or ('right_eye' not in landmarks):
                continue
            left_bounds = resolve_corners(landmarks['left_eye'])
            right_bounds = resolve_corners(landmarks['right_eye'])

            left_bounds = increase_bounds(left_bounds)
            right_bounds = increase_bounds(right_bounds)

            left = tuple([int(x) for x in left_bounds])
            right = tuple([int(x) for x in right_bounds])
            color = COLORS[i % len(COLORS)]

            cv2.rectangle(small_frame, (left[0], left[1]), (left[2], left[3]), color, 1)
            cv2.rectangle(small_frame, (right[0], right[1]), (right[2], right[3]), color, 1)

            left_bounds = tuple([point * 2 for point in left_bounds])
            right_bounds = tuple([point * 2 for point in right_bounds])

            pil_im = Image.fromarray(image)

            left_eye = pil_im.crop(left_bounds)
            right_eye = pil_im.crop(right_bounds)

            combined = Image.new('RGB', (72, 28))
            combined.paste(left_eye, (0, 0))
            combined.paste(right_eye, (36, 0))
            combined = np.array(combined, dtype=np.float64)

            X = np.array([combined])

            prediction = self.model.predict(X)
            result = np.argmax(prediction)
            results.append(result)
            print(result)

        image = cv2.resize(small_frame, (0,0), fx=1.0, fy=1.0)
        self.results = results
        self.image_data.emit(image)

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())