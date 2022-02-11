import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import time
from tqdm import tqdm
from hand import Hand

# from feature_extractor import Featurizer
import config

# featurizer = Featurizer()


def get_id_from_file_name(_file_name):
    # _file_name = [img_id].png
    return _file_name.split(".")[0]


class HandFactory(object):
    def __init__(self):
        self.hands = []
        self.hand_images_directory = config.hand_img_folder
        self.metadata_dataframe_path = config.hand_metadata_folder
        self.list_hand_files = None
        self.batch_size = config.batch_size
        self.features_df = pd.DataFrame()
        self.prepare_list_hand_files()

    def prepare_list_hand_files(self):
        self.list_hand_files = os.listdir(self.hand_images_directory)
        self.list_hand_files.sort()
        self.list_hand_files = self.list_hand_files[: self.batch_size]

    def featurize_batch_of_hands(self):
        metadata_df = pd.read_csv(self.metadata_dataframe_path)

        print("Extracting features from hands...")
        for hand_file in tqdm(self.list_hand_files):
            id = get_id_from_file_name(hand_file)
            img_file = os.path.join(self.hand_images_directory, hand_file)
            img = cv2.imread(img_file, 0)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            hand_metadata = metadata_df.loc[metadata_df["id"] == int(id)]
            age = int(hand_metadata["boneage"])
            gender = hand_metadata["male"].bool()
            gender = int(gender)
            hand = Hand(img, age, gender, id)
            hand.featurize()
            self.hands.append(hand)
            self.add_hand_features_to_df(hand)
        print("Extraction complete")

        self.features_df.to_csv(config.features_df_path)

    def add_hand_features_to_df(self, _hand):
        hand_features = pd.DataFrame(
            {feature: [getattr(_hand, feature)] for feature in config.FEATURES_FOR_DATA_ANALYSIS}
        )
        self.features_df = pd.concat(
            [self.features_df, hand_features], ignore_index=True, axis=0
        )
