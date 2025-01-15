import os
import mediapipe as mp
import cv2
import numpy as np
import time


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


flag , list_pos, label_pos= 0, [], []
video = cv2.VideoCapture(r'dataset\video\right\right_1.avi')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)


while video.isOpened():
    ret, frame = video.read()
    if ret:
        results = hands_model.process(frame)
        w, h, _ =   frame.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                list_pos.append(np.array([[lm.x*w, lm.y*h] for lm in hand_landmarks.landmark], dtype=np.int32))
                flag += 1
        else:
            list_pos.append(np.zeros((21, 2), dtype=np.int32))

        cv2.imshow('frame', frame)
        cv2.waitKey(33)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
path = r'dataset\left\0.avi'
label_pos.append(os.path.dirname(path).split('\\')[-1])
list_pos = np.array(list_pos).reshape(total_frames,-1)
print(f'====Total Frames: {total_frames}\n====Detected Frames: {flag}')
print(f'====List of Positions Shape: {list_pos.shape}\n====List of Label: {label_pos}')


video.release()
hands_model.close()
cv2.destroyAllWindows()
