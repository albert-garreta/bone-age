import os
from lib.utils import get_working_hand_img_files
from lib.hand import Hand
from lib.segmentations import get_segmentations
import lib.segmentations as segmentations
import shutil
import cv2

hand_img_files = get_working_hand_img_files()
for dir in hand_img_files:
    try:
        hand_id = dir.split('/')[-1].split(".")[0]
        print(hand_id)
        img = cv2.imread(dir)
        segments = get_segmentations(hand_id, img.shape)
        hand = Hand(img, 0, 0, hand_id.split(".")[0], segments)
        
        
        hand.get_hand_landmarks()
        segmentations.draw_all_contours(hand.img, segments, write_contour_number=True)
        
        write_path = os.path.join("data/hands_with_landmarks", f"{hand_id}" + ".png")
        print(write_path)
        cv2.imwrite(write_path, hand.img)
    except Exception as e:
        print(e)

def get_worse_performing_samples(df, losses):
    assert df.shape[0] == len(losses), f"{df.shape[0]}, {len(losses)}"
    df["losses"] = losses
    df = df.sort_values(by="losses", axis=0)
    #print( df["losses"].iloc[-20:])
    return df["id"].iloc[-20:]