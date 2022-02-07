import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from feature_extractor import draw_hand_landmarks
import time

class Hand(object):
    def __init__(self, _image, _age, _gender, _id):
        self.img = _image
        self.age = _age
        self.gender = _gender
        self.id = _id

    def normalize_contrast(self):
        pass
    
    def draw_landmarks(self):
        self.img = draw_hand_landmarks(self.img)

    def show(self):
        plt.imshow(self.img)
        plt.show()

        
        

def get_id_from_file_name(_file_name):
    # _file_name = [img_id].png
    return _file_name.split(".")[0]


class DataProcessor(object):
    def __init__(self):
        self.hands = []
        self.hand_images_directory = "./data/boneage-training-dataset"
        self.metadata_dataframe_path = "./data/boneage-training-dataset.csv"
        self.batch_size = 300

    def load_batch_of_hands(self):
        metadata_df = pd.read_csv(self.metadata_dataframe_path)

        for hand_file in os.listdir(self.hand_images_directory)[: self.batch_size]:
            id = get_id_from_file_name(hand_file)
            img_file = os.path.join(self.hand_images_directory, hand_file)
            img = cv2.imread(img_file, 0)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            hand_metadata = metadata_df.loc[metadata_df["id"] == int(id)]
            age = int(hand_metadata["boneage"])
            gender = hand_metadata["male"].bool()
            gender = int(gender)
            hand = Hand(img, age, gender, id)
            self.hands.append(hand)

    def normalize_crop_scale_center_rotation():
        pass
