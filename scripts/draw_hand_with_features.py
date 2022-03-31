import matplotlib.pyplot as plt
import numpy as np
import cv2
import config
from ..lib.hand import Hand
import pandas as pd
from ..lib.segmentations import segmentations

metadata_df = pd.read_csv(config.metadata_dataframe_path)


def draw_hand_with_features(hand_id, img=None):
    
    if img is None:
        img = cv2.imread(f"../data/{config.hand_img_folder}/{hand_id}.png")
        
    hand_metadata = metadata_df.loc[metadata_df["id"] == int(id)]
    age = int(hand_metadata["boneage"])
    gender = hand_metadata["male"].bool()
    gender = int(gender)
    segments = segmentations.get_segments(str(id), img.shape)
    hand = Hand(img, age, gender, hand_id, segments)
    hand.get_hand_landmarks()
    hand.draw_landmarks()
    segmentations.draw_all_contours(hand.img, segments, write_contour_number=False)
    