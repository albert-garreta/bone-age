import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import time
from tqdm import tqdm
from classes.hand import Hand
import config as config



def get_id_from_file_name(_file_name):
    # _file_name = [img_id].png
    return _file_name.split(".")[0]

class HandFactory(object):
    def __init__(self):
        self.hands = []
        self.hand_images_directory = config.hand_img_folder
        self.segmented_hand_images_directory = config.segmented_hand_img_folder
        self.metadata_dataframe_path = config.hand_metadata_folder
        self.list_hand_files = None
        self.batch_size = config.batch_size
        
        # Final dataframe where to perform the regression
        # The goal of this class is to compute this
        self.features_df = pd.DataFrame()
        
        self.prepare_list_hand_files()
        self.prepare_segmentations()

    def prepare_list_hand_files(self):
        # We use the segmented hands directory because it containts only a subset
        # of all hands
        self.list_hand_files = os.listdir(self.segmented_hand_images_directory)
        self.list_hand_files = [file for file in self.list_hand_files if file!='.DS_Store']
        self.list_hand_files.sort()
        self.list_hand_files = self.list_hand_files[: self.batch_size]

    def get_features_dataframe(self) -> pd.DataFrame:
        """
            Creates self.features_df: a dataframe containing the features in FEATURES_FOR_DATA_ANALYSIS for each of the hands files
            in self.list_hand_files
            
            This df is saved in `config.features_df_path`
        """
        metadata_df = pd.read_csv(self.metadata_dataframe_path)

        print("Extracting features from hands...")
        for hand_file in tqdm(self.list_hand_files):
            id = get_id_from_file_name(hand_file)
            img_file = os.path.join(self.hand_images_directory, hand_file)
            img = cv2.imread(img_file, 1)
            segmented_img_file = os.path.join(self.segmented_hand_images_directory, hand_file)
            segmented_img = cv2.imread(segmented_img_file, 1)
            hand_metadata = metadata_df.loc[metadata_df["id"] == int(id)]
            age = int(hand_metadata["boneage"])
            gender = hand_metadata["male"].bool()
            gender = int(gender)
            hand = Hand(img, age, gender, id, segmented_img)
            success = hand.featurize()
            if success:
                self.add_hand(hand)

        print("Extraction complete")
        self.add_all_hand_features_to_df()
        self.features_df.to_csv(config.features_df_path)

    def add_hand(self, _hand):
        self.hands.append(_hand)

    def add_all_hand_features_to_df(self):
        for _hand in self.hands:
            self.add_hand_to_df(_hand)

    def add_hand_to_df(self, _hand):
        hand_features = pd.DataFrame(
            {
                feature: [getattr(_hand, feature)]
                for feature in config.FEATURES_FOR_DATA_ANALYSIS
            }
        )
        self.features_df = pd.concat(
            [self.features_df, hand_features], ignore_index=True, axis=0
        )
