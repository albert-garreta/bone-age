import cv2
from importlib_metadata import files
import matplotlib.pyplot as plt
import os
import random
import matplotlib.pyplot as plt
import config
import cv2
import math
import mediapipe as mp
import numpy as np
import json
import scipy.spatial.distance as distance
import random
from lib.hand import Hand
from tqdm import tqdm
if __name__ == "__main__":

    dir = "data/jsons"
    dir2 = "data/boneage-training-dataset"
    files_list = os.listdir(dir)
    good_hand = []
    random.shuffle(files_list)
    for file_name in tqdm(files_list):
        file_name2 = file_name.split(".")[0] + ".png"
        hand = Hand(cv2.imread(os.path.join(dir2, file_name2)), 0,0, file_name2, None)
        hand.get_hand_landmarks()
        try:
            if hand.raw_landmarks:
                good_hand.append(hand)
                print(file_name)
            #if file_name.split('.')[0] + '.json' in set(os.listdir("data/jsons")):
            #    print(file_name)
        except Exception as e:
            #print(e)
            pass
        
    
    print(len(good_hand))        
        
    
    #     keys.append(key)
    #     if key.split(".")[0] != file_name.split('.')[0]: 
    #         weird_cases.append(key)
    # 
    # print(len(files_list))
    # print(len(weird_cases))
    # print(len(set(keys)))