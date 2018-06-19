import cv2
import win32api
import win32gui
import time
import face_recognition


def process_image(image):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    converted_image = image[:, :, ::-1]

    face_locations = face_recognition.face_locations(converted_image)
    face_landmarks_list = face_recognition.face_landmarks(image)

    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

    if len(face_locations) == 0 or len(face_landmarks_list) == 0:
        return
    landmarks = face_landmarks_list[0]
    if ('left_eye' not in landmarks) or ('right_eye' not in landmarks):
        return

    print("Landmarks: ")
    print(landmarks)

    # Display the resulting image
    cv2.imshow('Video', image)
    cv2.waitKey(250)

def capture_position_and_image(video_capture):
    _, _, (posX, posY) = win32gui.GetCursorInfo()
    _, frame = video_capture.read()

    print("Mouse pos: {}, {}".format(posX, posY))
    process_image(frame)    

def mouse_click_loop(video_capture):
    state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
    state_right = win32api.GetKeyState(0x02)  # Right button down = 0 or 1. Button up = -127 or -128

    while True:
        a = win32api.GetKeyState(0x01)
        b = win32api.GetKeyState(0x02)

        if a != state_left:  # Button state changed
            state_left = a
            if a < 0:
                print('Left Button Pressed')
                capture_position_and_image(video_capture)

        if b != state_right:  # Button state changed
            state_right = b
            if b < 0:
                print('Right Button Pressed')
                capture_position_and_image(video_capture)
        time.sleep(0.001)


def main():
    video_capture = cv2.VideoCapture(0)
    print("VideoCapture initialized")

    mouse_click_loop(video_capture)

if __name__ == "__main__":
    main()