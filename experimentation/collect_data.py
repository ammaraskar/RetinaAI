import cv2
import win32api
import win32gui
import time
import face_recognition
import pathlib
import json
import glob
from PIL import Image
import os
from threading import Thread


def process_image(image, image_id, x, y):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    converted_image = image[:, :, ::-1]

    face_locations = face_recognition.face_locations(converted_image)
    face_landmarks_list = face_recognition.face_landmarks(image, face_locations=face_locations)

    if len(face_locations) == 0 or len(face_landmarks_list) == 0:
        return
    landmarks = face_landmarks_list[0]
    if ('left_eye' not in landmarks) or ('right_eye' not in landmarks):
        return

    print("Landmarks: ")
    print(landmarks)

    data = {
        'position': face_locations[0],
        'left': landmarks['left_eye'],
        'right': landmarks['right_eye'],
        'x': x,
        'y': y
    }

    left_bounds = resolve_corners(landmarks['left_eye'])
    right_bounds = resolve_corners(landmarks['right_eye'])

    #cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(converted_image)

    print('Image', pil_im.size)
    print('Left Eye', left_bounds)
    print('Right Eye', right_bounds)

    left_eye = pil_im.crop(left_bounds)
    right_eye = pil_im.crop(right_bounds)

    left_eye.save('data/images/{}_left.png'.format(image_id))
    right_eye.save('data/images/{}_right.png'.format(image_id))
    with open('data/images/{}_meta.json'.format(image_id), 'w') as f:
        json.dump(data, f)

def resolve_corners(cordinate_list):
    xs = [cord[0] for cord in cordinate_list]
    x_low, x_high = min(xs), max(xs)

    ys = [cord[1] for cord in cordinate_list]
    y_low, y_high = min(ys), max(ys)

    return (x_low, y_low, x_high, y_high)

def capture_position_and_image(video_capture, image_id):
    _, _, (posX, posY) = win32gui.GetCursorInfo()
    _, frame = video_capture.read()

    print("Mouse pos: {}, {}".format(posX, posY))

    thread = Thread(target=process_image, args=(frame, image_id, posX, posY))
    thread.start()

def mouse_click_loop(video_capture, last_image):
    state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
    state_right = win32api.GetKeyState(0x02)  # Right button down = 0 or 1. Button up = -127 or -128

    while True:
        a = win32api.GetKeyState(0x01)
        b = win32api.GetKeyState(0x02)

        if a != state_left:  # Button state changed
            state_left = a
            if a < 0:
                print('Left Button Pressed')
                last_image += 1
                capture_position_and_image(video_capture, last_image)

        if b != state_right:  # Button state changed
            state_right = b
            if b < 0:
                print('Right Button Pressed')
                last_image += 1
                capture_position_and_image(video_capture, last_image)
        time.sleep(0.001)


def main():
    pathlib.Path('data/images').mkdir(parents=True, exist_ok=True) 

    images = glob.glob('data/images/*.png')
    if len(images) == 0:
        last_image = 0
    else:
        images = [os.path.basename(image) for image in images]
        images = [int(image.split('_')[0]) for image in images]
        last_image = max(images)

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 1920)
    video_capture.set(4, 1080)
    print("VideoCapture initialized")

    mouse_click_loop(video_capture, last_image)

if __name__ == "__main__":
    main()