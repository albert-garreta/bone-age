# Use AGE as a feature!

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_hand_landmarks(img):
    result = hands.process(img)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            print(hand_landmarks)

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, channels = img.shape
                x_img = int(landmark.x * width)
                y_img = int(landmark.y * height)

                cv2.putText(
                    img,
                    str(id),
                    (x_img, y_img),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 255),
                    2,
                )
    return img
    # plt.imshow(img)
    # plt.show()
