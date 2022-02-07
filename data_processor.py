import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import time
from hand import Hand


def get_id_from_file_name(_file_name):
    # _file_name = [img_id].png
    return _file_name.split(".")[0]


class DataProcessor(object):
    def __init__(self):
        self.hands = []
        self.hand_images_directory = "./data/boneage-training-dataset"
        self.metadata_dataframe_path = "./data/boneage-training-dataset.csv"
        self.list_hand_files = None
        self.batch_size = 1
        self.prepare_list_hand_files()

    def prepare_list_hand_files(self):
        self.list_hand_files = os.listdir(self.hand_images_directory)
        self.list_hand_files.sort()
        self.list_hand_files = self.list_hand_files[: self.batch_size]

    def load_batch_of_hands(self):
        metadata_df = pd.read_csv(self.metadata_dataframe_path)

        for hand_file in self.list_hand_files:
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
