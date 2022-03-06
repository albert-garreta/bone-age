import cv2
import matplotlib.pyplot as plt
import os
import random
import matplotlib.pyplot as plt
import config
import cv2
import math
import mediapipe as mp
import numpy as np

if __name__ == "__main__":
    dir = "data/data_tagged/1377.png"
    dir = "data/data_tagged"
    dir = "data/boneage-training-dataset"
    random_img_dir = random.choice(os.listdir(dir))
    img = cv2.imread(os.path.join(dir,random_img_dir))
    #img = cv2.imread(dir,1)
    #plt.imshow(img)
    #plt.show()
    
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # plt.imshow(img, )
    # plt.show()
    
    results = hands.process(img)
    print(results)
    print(results.multi_hand_landmarks[0])