import cv2
from cv2 import DFT_ROWS
import numpy as np
import matplotlib.pyplot as plt
import config
import os
from utils import *
from lib.hand import Hand
from tqdm import tqdm
import segmentations
from data_analysis import cut_out_outlier_samples
import pandas as pd

valid_ids = get_valid_hand_ids()
img_files = get_working_hand_img_files()

df = pd.read_csv("./data/features_df.csv")

df = cut_out_outlier_samples(df, "max_purple_diameter", 0, None)
df = cut_out_outlier_samples(df, "gap_ratio_5", None, 0.05)
df = cut_out_outlier_samples(df, "gap_ratio_13", None, 0.13)
df = cut_out_outlier_samples(df, "gap_ratio_9", None, 0.1)
df = cut_out_outlier_samples(df, "carp_bones_max_diameter_ratio", 0, None)

#print(valid_ids)
valid_ids = [x for x in valid_ids if int(x) in set(df["id"])]
img_files = [x for x in img_files if int(x.split("/")[-1].split(".")[0]) in set(df["id"])]
for id, img in tqdm(zip(valid_ids, img_files)):
    img = cv2.imread(img)
    try:
        segments = segmentations.get_segmentations(id, img.shape)
        hand = Hand(img, 0, 0, id, segments)
        hand.get_hand_landmarks()
        hand.featurize()
        cv2.imwrite(f"./data/drawings_gap_features/{id}.png", hand.img)
    except Exception as e:
        print(e)
