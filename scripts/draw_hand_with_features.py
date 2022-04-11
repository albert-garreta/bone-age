import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

sys.path.append("../bone-age")
from lib.hand import Hand
import pandas as pd
import config as config

from lib.segmentations import get_segmentations, draw_all_contours

metadata_df = pd.read_csv(config.hand_metadata_folder)


def draw_hand_with_features(hand_id):
    """
    Takes a hand id (e.g. 1717) and draws:
        - Google landmarks
        - Bone contours from the json files
        - A line indicating the max diameter of the "epifisis" bones,
        the "carp" bones, and the bones formed inbetween the phalanxes etc.
    """
    img = cv2.imread(f"{config.hand_img_folder}/{hand_id}.png")
    hand_metadata = metadata_df.loc[metadata_df["id"] == int(id)]
    age = int(hand_metadata["boneage"])
    gender = hand_metadata["male"].bool()
    gender = int(gender)
    segments = get_segmentations(str(id), None)
    hand = Hand(img, age, gender, hand_id, segments)
    hand.get_hand_landmarks()
    hand.draw_landmarks()
    draw_all_contours(hand.img, segments, write_contour_number=False)
    cv2.imwrite(f"{hand_id}_feats.png", hand.img)


if __name__ == "__main__":

    ids = [
        9348,
        2481,
        1766,
        10582,
        4255,
        8996,
        4824,
        3499,
        1391,
        3546,
        1855,
        2401,
        1758,
        10653,
        11363,
        10532,
        1562,
        3356,
        6270,
        12081,
    ]

    for id in ids:
        draw_hand_with_features(id)
