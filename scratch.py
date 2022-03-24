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
    metadata_df = pd.read_csv("./data/boneage-training-dataset.csv")
    list_hand_files = os.listdir("./data/boneage-training-dataset")
    json_list = os.listdir("./data/jsons")
    for hand_file in tqdm(list_hand_files):
        id = get_id_from_file_name(hand_file)
        if f"{id}.json" in json_list:
            hand_metadata = metadata_df.loc[metadata_df["id"] == int(id)]
            age = int(hand_metadata["boneage"])
            if age <= 20 * 12:
                print(age / 12)
                img_file = os.path.join("./data/boneage-training-dataset", hand_file)
                img = cv2.imread(img_file)
                hand = Hand(img, 0, 0, id, None)
                hand.get_hand_landmarks()
                hand.draw_landmarks()
                cv2.imwrite(f"./data/babies_landmarks/{id}.png", hand.img)
