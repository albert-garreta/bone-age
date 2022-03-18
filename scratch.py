import cv2
import os
dir = "data/data_tagged"
import seaborn as sbn
import pandas as pd
from tqdm import tqdm
from lib.hand import Hand
import segmentations
from lib.hand_factory import get_id_from_file_name
import json

if __name__ == "__main__":
    df = pd.read_csv("./data/features_df.csv")
    print(df.head())
    df2 = df.loc[df["boneage"]<=150]
    print(df2.head())