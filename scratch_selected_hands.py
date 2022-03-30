import os
from lib.utils import get_working_hand_img_files, get_valid_hand_ids
from lib.hand import Hand
from lib.segmentations import get_segmentations
import lib.segmentations as segmentations
import shutil
import cv2
import pandas as pd

hand_img_files = get_working_hand_img_files()
valid_hand_ids = get_valid_hand_ids()
# for dir in hand_img_files:
# try:
#    hand_id = dir.split("/")[-1].split(".")[0]
#    img = cv2.imread(dir)
#    write_path = os.path.join("data/selected_hands", f"{hand_id}" + ".png")
#    print(write_path)
#    cv2.imwrite(write_path, img)
# except Exception as e:
#    print(e)

# -----------------------
# Added by me (Albert)
age_df = pd.read_csv("data/boneage-training-dataset.csv")
total_ids = len(age_df.index)
valid_ids = [str(hand_id) in valid_hand_ids for hand_id in age_df["id"]]
age_df = age_df.loc[valid_ids]
age_df.index = range(len(age_df))
train_df = age_df
train_df.to_csv("selected_boneage_dataset.csv")
# -----------------------
